from __future__ import annotations

import asyncio

import pytest

from kaggle_jupyter_mcp.transfer_jobs import FileTransferJobManager


class FakeTransferService:
    def __init__(self) -> None:
        self.progress_reported = asyncio.Event()
        self.release = asyncio.Event()

    async def upload(self, **kwargs):
        progress = kwargs["progress"]
        total = 10
        await progress(
            {
                "stage": "uploading",
                "transferred_bytes": 5,
                "total_bytes": total,
                "completed_parts": 1,
                "total_parts": 2,
                "resumed_bytes": 0,
                "remote_transfer_id": "content-hash-id",
                "partial_path": "/kaggle/temp/partial.bin",
            }
        )
        self.progress_reported.set()
        await self.release.wait()
        await progress(
            {
                "stage": "completed",
                "transferred_bytes": total,
                "total_bytes": total,
                "completed_parts": 2,
                "total_parts": 2,
                "sha256": "a" * 64,
            }
        )
        return ["upload complete"]

    async def download(self, **kwargs):
        return ["download complete"]


@pytest.mark.asyncio
async def test_background_upload_is_pollable_and_completes(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    monkeypatch.setenv("KAGGLE_JUPYTER_LOCAL_ROOTS", str(tmp_path))
    service = FakeTransferService()
    manager = FileTransferJobManager(service)

    started = await manager.start_upload(
        local_path=str(source),
        server_path="artifact.bin",
        notebook_path="transfer.ipynb",
        notebook_name=None,
        overwrite=False,
        chunk_size_mib=1,
        timeout=30,
        resume=True,
    )
    transfer_id = started["transfer_id"]
    await asyncio.wait_for(service.progress_reported.wait(), timeout=1)

    running = manager.status(transfer_id)
    assert running["state"] == "running"
    assert running["stage"] == "uploading"
    assert running["percent"] == 50.0
    assert running["completed_parts"] == 1
    assert running["partial_path"] == "/kaggle/temp/partial.bin"

    service.release.set()
    await asyncio.wait_for(manager.jobs[transfer_id].task, timeout=1)
    completed = manager.status(transfer_id)
    assert completed["state"] == "completed"
    assert completed["percent"] == 100.0
    assert completed["sha256"] == "a" * 64
    assert completed["result"] == "upload complete"


@pytest.mark.asyncio
async def test_background_upload_can_be_cancelled_and_remains_resumable(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    monkeypatch.setenv("KAGGLE_JUPYTER_LOCAL_ROOTS", str(tmp_path))
    service = FakeTransferService()
    manager = FileTransferJobManager(service)
    started = await manager.start_upload(
        local_path=str(source),
        server_path="artifact.bin",
        notebook_path="transfer.ipynb",
        notebook_name=None,
        overwrite=False,
        chunk_size_mib=1,
        timeout=30,
        resume=True,
    )
    transfer_id = started["transfer_id"]
    await asyncio.wait_for(service.progress_reported.wait(), timeout=1)

    cancelled = await manager.cancel(transfer_id)

    assert cancelled["state"] == "cancelled"
    assert cancelled["resume_enabled"] is True
    assert cancelled["transferred_bytes"] == 5
    assert manager.list()[0]["transfer_id"] == transfer_id
