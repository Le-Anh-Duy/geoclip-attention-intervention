"""Inspectable background jobs for long-running file transfers."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kaggle_jupyter_mcp.file_transfer import (
    FileTransferService,
    normalize_server_file_path,
    resolve_local_path,
)

TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


@dataclass(slots=True)
class TransferJob:
    transfer_id: str
    direction: str
    source: str
    destination: str
    total_bytes: int = 0
    transferred_bytes: int = 0
    completed_parts: int = 0
    total_parts: int = 0
    resumed_bytes: int = 0
    state: str = "queued"
    stage: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    sha256: str | None = None
    remote_transfer_id: str | None = None
    partial_path: str | None = None
    error: str | None = None
    result: str | None = None
    resume_enabled: bool = True
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    def update(self, values: dict[str, Any]) -> None:
        for key in (
            "stage",
            "transferred_bytes",
            "total_bytes",
            "completed_parts",
            "total_parts",
            "resumed_bytes",
            "sha256",
            "remote_transfer_id",
            "partial_path",
        ):
            if key in values:
                setattr(self, key, values[key])
        self.updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        now = self.finished_at or time.time()
        started = self.started_at or self.created_at
        elapsed = max(0.0, now - started)
        active_bytes = max(0, self.transferred_bytes - self.resumed_bytes)
        speed = active_bytes / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total_bytes - self.transferred_bytes)
        eta = remaining / speed if speed > 0 and self.state == "running" else None
        percent = (
            round(self.transferred_bytes * 100 / self.total_bytes, 2)
            if self.total_bytes
            else None
        )
        return {
            "transfer_id": self.transfer_id,
            "direction": self.direction,
            "state": self.state,
            "stage": self.stage,
            "source": self.source,
            "destination": self.destination,
            "transferred_bytes": self.transferred_bytes,
            "total_bytes": self.total_bytes,
            "percent": percent,
            "completed_parts": self.completed_parts,
            "total_parts": self.total_parts,
            "resumed_bytes": self.resumed_bytes,
            "speed_mib_s": round(speed / (1024 * 1024), 3),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": round(eta, 2) if eta is not None else None,
            "sha256": self.sha256,
            "remote_transfer_id": self.remote_transfer_id,
            "partial_path": self.partial_path,
            "resume_enabled": self.resume_enabled,
            "created_at": _utc_iso(self.created_at),
            "started_at": _utc_iso(self.started_at),
            "updated_at": _utc_iso(self.updated_at),
            "finished_at": _utc_iso(self.finished_at),
            "error": self.error,
            "result": self.result,
        }


class FileTransferJobManager:
    """Run transfers in the background while status tools remain responsive."""

    def __init__(self, service: FileTransferService, max_history: int = 100) -> None:
        self.service = service
        self.max_history = max_history
        self.jobs: dict[str, TransferJob] = {}

    def _add_job(self, job: TransferJob) -> TransferJob:
        terminal = [
            item
            for item in self.jobs.values()
            if item.state in TERMINAL_STATES
        ]
        for old in sorted(terminal, key=lambda item: item.updated_at)[
            : max(0, len(self.jobs) - self.max_history + 1)
        ]:
            self.jobs.pop(old.transfer_id, None)
        self.jobs[job.transfer_id] = job
        return job

    @staticmethod
    def _job_id() -> str:
        return uuid.uuid4().hex[:16]

    async def start_upload(
        self,
        *,
        local_path: str,
        server_path: str,
        notebook_path: str,
        notebook_name: str | None,
        overwrite: bool,
        chunk_size_mib: int,
        timeout: int,
        resume: bool,
    ) -> dict[str, Any]:
        source = resolve_local_path(local_path, must_exist=True)
        if not source.is_file():
            raise ValueError("local_path must be a regular file")
        destination = normalize_server_file_path(server_path)
        job = self._add_job(
            TransferJob(
                transfer_id=self._job_id(),
                direction="upload",
                source=str(source),
                destination=destination,
                total_bytes=source.stat().st_size,
                resume_enabled=resume,
            )
        )
        job.task = asyncio.create_task(
            self._run_upload(
                job,
                notebook_path=notebook_path,
                notebook_name=notebook_name,
                overwrite=overwrite,
                chunk_size_mib=chunk_size_mib,
                timeout=timeout,
                resume=resume,
            ),
            name=f"kaggle-upload-{job.transfer_id}",
        )
        await asyncio.sleep(0)
        return job.snapshot()

    async def _run_upload(self, job: TransferJob, **options: Any) -> None:
        async def progress(values: dict[str, Any]) -> None:
            job.update(values)

        await self._run(
            job,
            self.service.upload(
                local_path=job.source,
                server_path=job.destination,
                progress=progress,
                cancel_event=job.cancel_event,
                **options,
            ),
        )

    async def start_download(
        self,
        *,
        server_path: str,
        local_path: str,
        notebook_path: str,
        notebook_name: str | None,
        overwrite: bool,
        chunk_size_mib: int,
        timeout: int,
        resume: bool,
    ) -> dict[str, Any]:
        source = normalize_server_file_path(server_path)
        destination = resolve_local_path(local_path, must_exist=False)
        job = self._add_job(
            TransferJob(
                transfer_id=self._job_id(),
                direction="download",
                source=source,
                destination=str(destination),
                resume_enabled=resume,
            )
        )
        job.task = asyncio.create_task(
            self._run_download(
                job,
                notebook_path=notebook_path,
                notebook_name=notebook_name,
                overwrite=overwrite,
                chunk_size_mib=chunk_size_mib,
                timeout=timeout,
                resume=resume,
            ),
            name=f"kaggle-download-{job.transfer_id}",
        )
        await asyncio.sleep(0)
        return job.snapshot()

    async def _run_download(self, job: TransferJob, **options: Any) -> None:
        async def progress(values: dict[str, Any]) -> None:
            job.update(values)

        await self._run(
            job,
            self.service.download(
                server_path=job.source,
                local_path=job.destination,
                progress=progress,
                cancel_event=job.cancel_event,
                keep_partial_on_failure=True,
                **options,
            ),
        )

    async def _run(self, job: TransferJob, operation: Any) -> None:
        job.state = "running"
        job.stage = "starting"
        job.started_at = time.time()
        job.updated_at = job.started_at
        try:
            result = await operation
            job.state = "completed"
            job.stage = "completed"
            job.result = next((item for item in result if isinstance(item, str)), None)
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.stage = "cancelled"
        except Exception as error:  # noqa: BLE001 - surfaced through status tool
            job.state = "failed"
            job.stage = "failed"
            job.error = f"{type(error).__name__}: {error}"
        finally:
            job.finished_at = time.time()
            job.updated_at = job.finished_at

    def status(self, transfer_id: str) -> dict[str, Any]:
        job = self.jobs.get(transfer_id)
        if job is None:
            raise ValueError(f"Unknown transfer_id: {transfer_id}")
        return job.snapshot()

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
        return [job.snapshot() for job in jobs[:limit]]

    async def cancel(self, transfer_id: str) -> dict[str, Any]:
        job = self.jobs.get(transfer_id)
        if job is None:
            raise ValueError(f"Unknown transfer_id: {transfer_id}")
        if job.state in TERMINAL_STATES:
            return job.snapshot()
        job.cancel_event.set()
        if job.task is not None and not job.task.done():
            job.task.cancel()
        await asyncio.sleep(0)
        return job.snapshot()
