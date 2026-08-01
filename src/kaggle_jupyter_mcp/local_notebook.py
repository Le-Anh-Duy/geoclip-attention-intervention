"""Run a local notebook against a raw kernel on the remote Kaggle server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat
from mcp.types import ImageContent, TextContent

from kaggle_jupyter_mcp.file_transfer import resolve_local_path
from kaggle_jupyter_mcp.rest import JupyterRestClient

TERMINAL_RUN_STATES = {"completed", "failed", "cancelled"}


def _utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _server_fingerprint(client: JupyterRestClient) -> str:
    return hashlib.sha256(client.base_url.encode()).hexdigest()[:12]


def _load_notebook(path: Path) -> Any:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def _save_notebook(path: Path, notebook: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        nbformat.write(notebook, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_nbformat_outputs(outputs: list[Any]) -> list[Any]:
    converted: list[Any] = []
    for item in outputs:
        if isinstance(item, ImageContent):
            converted.append(
                nbformat.v4.new_output(
                    "display_data", data={item.mimeType: item.data}, metadata={}
                )
            )
            continue
        if isinstance(item, TextContent):
            text = item.text
        else:
            text = str(item)
        if text:
            converted.append(
                nbformat.v4.new_output(
                    "stream", name="stdout", text=text.rstrip("\n") + "\n"
                )
            )
    return converted


def _output_preview(outputs: list[Any], limit: int = 2_000) -> str:
    text = "\n".join(
        item.text if isinstance(item, TextContent) else str(item)
        for item in outputs
        if not isinstance(item, ImageContent)
    )
    return text[:limit]


@dataclass(slots=True)
class LocalNotebookSession:
    local_path: Path
    kernel_id: str
    server_fingerprint: str
    execution_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(slots=True)
class NotebookRunJob:
    run_id: str
    local_path: str
    kernel_id: str
    selected_cells: list[int]
    state: str = "queued"
    stage: str = "queued"
    current_cell: int | None = None
    completed_cells: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        total = len(self.selected_cells)
        return {
            "run_id": self.run_id,
            "state": self.state,
            "stage": self.stage,
            "local_path": self.local_path,
            "kernel_id": self.kernel_id,
            "current_cell": self.current_cell,
            "completed_cells": self.completed_cells,
            "total_cells": total,
            "percent": round(self.completed_cells * 100 / total, 2) if total else 100.0,
            "results": self.results,
            "created_at": _utc_iso(self.created_at),
            "started_at": _utc_iso(self.started_at),
            "updated_at": _utc_iso(self.updated_at),
            "finished_at": _utc_iso(self.finished_at),
            "error": self.error,
        }


class LocalNotebookManager:
    """Own local notebook-to-remote-kernel bindings and monitored runs."""

    def __init__(self, execute_code) -> None:
        self.execute_code = execute_code
        self.sessions: dict[Path, LocalNotebookSession] = {}
        self.runs: dict[str, NotebookRunJob] = {}

    @staticmethod
    def _path(local_path: str) -> Path:
        path = resolve_local_path(local_path, must_exist=True)
        if path.suffix.lower() != ".ipynb" or not path.is_file():
            raise ValueError("local_path must be an existing .ipynb file")
        return path

    async def connect(
        self,
        local_path: str,
        *,
        kernel_id: str | None,
        kernel_name: str,
        remote_working_dir: str,
        reuse_existing: bool,
    ) -> dict[str, Any]:
        path = self._path(local_path)
        notebook = await asyncio.to_thread(_load_notebook, path)
        client = JupyterRestClient.from_current_config()
        fingerprint = _server_fingerprint(client)
        existing = self.sessions.get(path)
        if kernel_id is None and reuse_existing and existing:
            if existing.server_fingerprint == fingerprint:
                try:
                    await asyncio.to_thread(client.get_kernel, existing.kernel_id)
                    return await self.status(str(path))
                except Exception:  # noqa: BLE001 - a stale binding is replaced below
                    pass
        if kernel_id:
            kernel = await asyncio.to_thread(client.get_kernel, kernel_id)
        else:
            kernel = await asyncio.to_thread(
                client.start_kernel, kernel_name, remote_working_dir
            )
            kernel_id = str(kernel["id"])
        counts = [
            int(cell.execution_count)
            for cell in notebook.cells
            if cell.cell_type == "code" and cell.execution_count is not None
        ]
        self.sessions[path] = LocalNotebookSession(
            local_path=path,
            kernel_id=kernel_id,
            server_fingerprint=fingerprint,
            execution_count=max(counts, default=0),
        )
        return await self.status(str(path))

    async def status(self, local_path: str) -> dict[str, Any]:
        path = self._path(local_path)
        session = self.sessions.get(path)
        if session is None:
            raise ValueError("Notebook is not connected; call connect_local_notebook first")
        notebook = await asyncio.to_thread(_load_notebook, path)
        client = JupyterRestClient.from_current_config()
        kernel = await asyncio.to_thread(client.get_kernel, session.kernel_id)
        return {
            "local_path": str(path),
            "kernel_id": session.kernel_id,
            "kernel_state": kernel.get("execution_state"),
            "cell_count": len(notebook.cells),
            "code_cell_count": sum(cell.cell_type == "code" for cell in notebook.cells),
            "last_execution_count": session.execution_count,
        }

    async def execute_cell(
        self, local_path: str, cell_index: int, *, timeout: int
    ) -> list[Any]:
        path = self._path(local_path)
        session = self.sessions.get(path)
        if session is None:
            raise ValueError("Notebook is not connected; call connect_local_notebook first")
        async with session.lock:
            notebook = await asyncio.to_thread(_load_notebook, path)
            if cell_index < 0 or cell_index >= len(notebook.cells):
                raise IndexError(
                    f"cell_index {cell_index} is outside 0..{len(notebook.cells) - 1}"
                )
            cell = notebook.cells[cell_index]
            if cell.cell_type != "code":
                raise ValueError(f"Cell {cell_index} is {cell.cell_type}, not code")
            original_mtime = path.stat().st_mtime_ns
            outputs = await self.execute_code(
                code=cell.source, timeout=timeout, kernel_id=session.kernel_id
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            saved = path.stat().st_mtime_ns == original_mtime
            if saved:
                session.execution_count += 1
                cell.execution_count = session.execution_count
                cell.outputs = _as_nbformat_outputs(outputs)
                await asyncio.to_thread(_save_notebook, path, notebook)
            header = {
                "local_path": str(path),
                "kernel_id": session.kernel_id,
                "cell_index": cell_index,
                "execution_count": session.execution_count if saved else None,
                "outputs_saved": saved,
                "save_warning": None
                if saved
                else "Local notebook changed during execution; outputs were not overwritten",
            }
            return [
                json.dumps(header, ensure_ascii=False, sort_keys=True),
                *(outputs or ["(cell completed with no output)"]),
            ]

    async def start_run(
        self,
        local_path: str,
        *,
        start_cell: int,
        stop_cell: int | None,
        timeout_per_cell: int,
        stop_on_error: bool,
    ) -> dict[str, Any]:
        path = self._path(local_path)
        session = self.sessions.get(path)
        if session is None:
            raise ValueError("Notebook is not connected; call connect_local_notebook first")
        notebook = await asyncio.to_thread(_load_notebook, path)
        end = len(notebook.cells) if stop_cell is None else min(stop_cell, len(notebook.cells))
        selected = [
            index
            for index in range(max(0, start_cell), end)
            if notebook.cells[index].cell_type == "code"
        ]
        job = NotebookRunJob(
            run_id=uuid.uuid4().hex[:16],
            local_path=str(path),
            kernel_id=session.kernel_id,
            selected_cells=selected,
        )
        self.runs[job.run_id] = job
        job.task = asyncio.create_task(
            self._run(job, timeout_per_cell=timeout_per_cell, stop_on_error=stop_on_error),
            name=f"kaggle-local-notebook-{job.run_id}",
        )
        await asyncio.sleep(0)
        return job.snapshot()

    async def _run(
        self, job: NotebookRunJob, *, timeout_per_cell: int, stop_on_error: bool
    ) -> None:
        job.state = "running"
        job.stage = "executing"
        job.started_at = time.time()
        try:
            for cell_index in job.selected_cells:
                job.current_cell = cell_index
                job.updated_at = time.time()
                try:
                    result = await self.execute_cell(
                        job.local_path, cell_index, timeout=timeout_per_cell
                    )
                    outputs = result[1:]
                    job.results.append(
                        {
                            "cell_index": cell_index,
                            "status": "completed",
                            "output_preview": _output_preview(outputs),
                        }
                    )
                except Exception as error:  # noqa: BLE001 - recorded for polling
                    job.results.append(
                        {
                            "cell_index": cell_index,
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if stop_on_error:
                        raise
                finally:
                    job.completed_cells += 1
            job.state = "completed"
            job.stage = "completed"
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.stage = "cancelled"
        except Exception as error:  # noqa: BLE001 - surfaced through status
            job.state = "failed"
            job.stage = "failed"
            job.error = f"{type(error).__name__}: {error}"
        finally:
            job.current_cell = None
            job.finished_at = time.time()
            job.updated_at = job.finished_at

    def run_status(self, run_id: str) -> dict[str, Any]:
        job = self.runs.get(run_id)
        if job is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        return job.snapshot()

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        job = self.runs.get(run_id)
        if job is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        if job.state in TERMINAL_RUN_STATES:
            return job.snapshot()
        client = JupyterRestClient.from_current_config()
        await asyncio.to_thread(client.kernel_action, job.kernel_id, "interrupt")
        if job.task and not job.task.done():
            job.task.cancel()
            await job.task
        return job.snapshot()

    async def close(self, local_path: str, *, shutdown_kernel: bool) -> dict[str, Any]:
        path = self._path(local_path)
        session = self.sessions.pop(path, None)
        if session is None:
            raise ValueError("Notebook is not connected")
        if shutdown_kernel:
            client = JupyterRestClient.from_current_config()
            await asyncio.to_thread(client.shutdown_kernel, session.kernel_id)
        return {
            "local_path": str(path),
            "kernel_id": session.kernel_id,
            "kernel_shutdown": shutdown_kernel,
            "state": "closed",
        }
