"""End-to-end monitored upload/cancel/resume/download smoke test."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _text(result) -> str:
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


def _structured(result):
    value = getattr(result, "structuredContent", None)
    if value is not None:
        return value
    return json.loads(_text(result))


async def _call(session, name: str, arguments: dict):
    result = await session.call_tool(name, arguments)
    if result.isError:
        raise RuntimeError(f"{name}: {_text(result)}")
    return _structured(result)


async def _wait_for_transfer(
    session, transfer_id: str, *, timeout: float = 180.0
) -> dict:
    deadline = time.monotonic() + timeout
    last_marker = None
    while time.monotonic() < deadline:
        status = await _call(
            session, "get_file_transfer_status", {"transfer_id": transfer_id}
        )
        marker = (status["state"], status["stage"], status["completed_parts"])
        if marker != last_marker:
            print(json.dumps(status, sort_keys=True))
            last_marker = marker
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.25)
    raise TimeoutError(f"Transfer did not finish: {transfer_id}")


async def main() -> None:
    if not os.getenv("JUPYTER_URL"):
        raise SystemExit("JUPYTER_URL is required")

    env = os.environ.copy()
    env.setdefault("KAGGLE_JUPYTER_AUTO_CHECKPOINT", "true")
    suffix = uuid4().hex[:10]
    upload_name = f"mcp_transfer_cancel_{suffix}"
    upload_notebook = f"{upload_name}.ipynb"
    resume_name = f"mcp_transfer_resume_{suffix}"
    resume_notebook = f"{resume_name}.ipynb"
    download_name = f"mcp_transfer_download_{suffix}"
    download_notebook = f"{download_name}.ipynb"
    remote_path = f"/kaggle/temp/mcp_transfer_smoke_{suffix}.bin"

    configured_roots = [
        value
        for value in env.get("KAGGLE_JUPYTER_LOCAL_ROOTS", "").split(os.pathsep)
        if value
    ]
    local_root = Path(configured_roots[0] if configured_roots else Path.cwd()).resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaggle_jupyter_mcp"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    errlog_path = os.getenv("MCP_SMOKE_ERRLOG", os.devnull)
    with tempfile.TemporaryDirectory(
        prefix="kaggle-mcp-transfer-", dir=local_root
    ) as temp_dir:
        source = Path(temp_dir) / "source.bin"
        destination = Path(temp_dir) / "downloaded.bin"
        source.write_bytes(bytes(range(256)) * (48 * 1024))  # 12 MiB
        expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        with open(errlog_path, "w", encoding="utf-8") as errlog:  # noqa: ASYNC230
            async with stdio_client(params, errlog=errlog) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = {item.name for item in (await session.list_tools()).tools}
                    required = {
                        "start_upload_local_file_to_jupyter",
                        "start_download_jupyter_file_to_local",
                        "get_file_transfer_status",
                        "list_file_transfers",
                        "cancel_file_transfer",
                    }
                    if missing := sorted(required - tools):
                        raise RuntimeError(f"Missing monitored transfer tools: {missing}")

                    try:
                        started = await _call(
                            session,
                            "start_upload_local_file_to_jupyter",
                            {
                                "local_path": str(source),
                                "server_path": remote_path,
                                "chunk_size_mib": 1,
                                "resume": True,
                                "notebook_path": upload_notebook,
                                "notebook_name": upload_name,
                            },
                        )
                        transfer_id = started["transfer_id"]
                        while True:
                            status = await _call(
                                session,
                                "get_file_transfer_status",
                                {"transfer_id": transfer_id},
                            )
                            if status["transferred_bytes"] > 0:
                                break
                            if status["state"] in {"failed", "cancelled"}:
                                raise RuntimeError(status)
                            await asyncio.sleep(0.1)
                        cancelled = await _call(
                            session,
                            "cancel_file_transfer",
                            {"transfer_id": transfer_id},
                        )
                        if cancelled["state"] != "cancelled":
                            raise AssertionError(cancelled)
                        print("CANCEL_OK", json.dumps(cancelled, sort_keys=True))

                        resumed = await _call(
                            session,
                            "start_upload_local_file_to_jupyter",
                            {
                                "local_path": str(source),
                                "server_path": remote_path,
                                "chunk_size_mib": 1,
                                "resume": True,
                                "notebook_path": resume_notebook,
                                "notebook_name": resume_name,
                            },
                        )
                        upload = await _wait_for_transfer(
                            session, resumed["transfer_id"]
                        )
                        if upload["state"] != "completed":
                            raise RuntimeError(upload)
                        if upload["resumed_bytes"] <= 0:
                            raise AssertionError("Upload did not resume its partial bytes")

                        download_started = await _call(
                            session,
                            "start_download_jupyter_file_to_local",
                            {
                                "server_path": remote_path,
                                "local_path": str(destination),
                                "chunk_size_mib": 1,
                                "resume": True,
                                "notebook_path": download_notebook,
                                "notebook_name": download_name,
                            },
                        )
                        download = await _wait_for_transfer(
                            session, download_started["transfer_id"]
                        )
                        if download["state"] != "completed":
                            raise RuntimeError(download)
                        actual_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
                        if actual_hash != expected_hash:
                            raise AssertionError((expected_hash, actual_hash))
                        print("TRANSFER_SHA256_OK", actual_hash)
                    finally:
                        for name in (upload_name, resume_name, download_name):
                            try:
                                await session.call_tool(
                                    "unuse_notebook", {"notebook_name": name}
                                )
                            except Exception:  # noqa: BLE001 - best-effort cleanup
                                pass
                        for path in (
                            remote_path,
                            upload_notebook,
                            resume_notebook,
                            download_notebook,
                        ):
                            try:
                                await session.call_tool(
                                    "delete_server_path", {"path": path}
                                )
                            except Exception:  # noqa: BLE001 - best-effort cleanup
                                pass

    print("MONITORED_TRANSFER_SMOKE_OK")


if __name__ == "__main__":
    asyncio.run(main())
