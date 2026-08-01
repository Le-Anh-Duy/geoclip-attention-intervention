"""Local <-> Kaggle file transfers with chunking and end-to-end hashes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from base64 import b64decode, b64encode
from pathlib import Path, PurePosixPath
from collections.abc import Awaitable, Callable
from typing import Any

from kaggle_jupyter_mcp.visible_execution import VisibleNotebookRunner

DEFAULT_CHUNK_MIB = 4
MAX_CHUNK_MIB = 8
META_PREFIX = "KAGGLE_MCP_TRANSFER_META "
CHUNK_PREFIX = "KAGGLE_MCP_TRANSFER_CHUNK "
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


async def _report_progress(callback: ProgressCallback | None, **values: Any) -> None:
    if callback is None:
        return
    result = callback(values)
    if result is not None:
        await result


def _allowed_local_roots() -> tuple[Path, ...]:
    configured = os.getenv("KAGGLE_JUPYTER_LOCAL_ROOTS", "")
    values = [item for item in configured.split(os.pathsep) if item.strip()]
    roots = values or [os.getcwd()]
    return tuple(Path(item).expanduser().resolve() for item in roots)


def resolve_local_path(path: str, *, must_exist: bool) -> Path:
    candidate = Path(path).expanduser().resolve(strict=must_exist)
    if not any(candidate == root or root in candidate.parents for root in _allowed_local_roots()):
        allowed = ", ".join(str(root) for root in _allowed_local_roots())
        raise ValueError(f"Local path is outside KAGGLE_JUPYTER_LOCAL_ROOTS: {allowed}")
    return candidate


def normalize_server_file_path(path: str) -> str:
    clean = path.replace("\\", "/").strip()
    if not clean:
        raise ValueError("server_path cannot be empty")
    if clean.startswith("/"):
        pure = PurePosixPath(clean)
        allowed = (PurePosixPath("/kaggle/working"), PurePosixPath("/kaggle/temp"))
        if not any(pure == root or root in pure.parents for root in allowed):
            raise ValueError("Absolute server_path must be under /kaggle/working or /kaggle/temp")
        return str(pure)
    pure = PurePosixPath(clean)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("Relative server_path must stay under /kaggle/working")
    return pure.as_posix()


def _server_path_code(path: str) -> str:
    return json.dumps(normalize_server_file_path(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_outputs(outputs: list[Any]) -> str:
    return "\n".join(item for item in outputs if isinstance(item, str))


def _parse_meta(outputs: list[Any]) -> dict[str, Any]:
    for line in reversed(_text_outputs(outputs).splitlines()):
        if line.startswith(META_PREFIX):
            return json.loads(line[len(META_PREFIX) :])
    raise RuntimeError("Transfer notebook cell did not return machine-readable metadata")


def _parse_chunk(outputs: list[Any]) -> bytes:
    for line in reversed(_text_outputs(outputs).splitlines()):
        if line.startswith(CHUNK_PREFIX):
            return b64decode(line[len(CHUNK_PREFIX) :], validate=True)
    raise RuntimeError("Kernel did not return a machine-readable transfer chunk")


def _remote_resolver_code(path_literal: str) -> str:
    return f"""
working = Path('/kaggle/working').resolve()
temp_root = Path('/kaggle/temp').resolve()
raw_path = {path_literal}
target = Path(raw_path)
target = target.resolve() if target.is_absolute() else (working / target).resolve()
if not any(target == root or root in target.parents for root in (working, temp_root)):
    raise ValueError('Path must remain under /kaggle/working or /kaggle/temp')
""".strip()


class FileTransferService:
    def __init__(self, runner: VisibleNotebookRunner) -> None:
        self.runner = runner

    async def upload(
        self,
        *,
        local_path: str,
        server_path: str,
        notebook_path: str,
        notebook_name: str | None,
        overwrite: bool,
        chunk_size_mib: int,
        timeout: int,
        progress: ProgressCallback | None = None,
        cancel_event: asyncio.Event | None = None,
        resume: bool = False,
    ) -> list[Any]:
        source = resolve_local_path(local_path, must_exist=True)
        if not source.is_file():
            raise ValueError("local_path must be a regular file")
        destination = normalize_server_file_path(server_path)
        chunk_size = chunk_size_mib * 1024 * 1024
        size = source.stat().st_size
        await _report_progress(
            progress, stage="hashing", transferred_bytes=0, total_bytes=size
        )
        sha256 = await asyncio.to_thread(_sha256_file, source)
        transfer_id = hashlib.sha256(f"{destination}:{sha256}".encode()).hexdigest()[:16]
        part_count = max(1, (size + chunk_size - 1) // chunk_size)
        remote_temp = f"/kaggle/temp/kaggle_mcp_upload_{transfer_id}.partial"
        await _report_progress(
            progress,
            stage="connecting",
            transferred_bytes=0,
            total_bytes=size,
            total_parts=part_count,
            remote_transfer_id=transfer_id,
            partial_path=remote_temp,
        )

        async with self.runner.lock:
            _, _, kernel_id = await self.runner.ensure_connected(
                notebook_path, notebook_name
            )
            start_offset = 0
            if resume:
                outputs = await self.runner.run_hidden(
                    self._upload_partial_size_code(remote_temp),
                    timeout=timeout,
                    kernel_id=kernel_id,
                )
                partial_meta = _parse_meta(outputs)
                start_offset = int(partial_meta["size"])
                if start_offset > size or (
                    start_offset != size and start_offset % chunk_size != 0
                ):
                    raise RuntimeError(
                        "Remote partial size is incompatible with this local file/chunk size"
                    )
            await _report_progress(
                progress,
                stage="uploading",
                transferred_bytes=start_offset,
                total_bytes=size,
                completed_parts=min(start_offset // chunk_size, part_count),
                total_parts=part_count,
                resumed_bytes=start_offset,
            )
            with source.open("rb") as stream:
                stream.seek(start_offset)
                offset = start_offset
                while offset < size:
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError
                    data = stream.read(chunk_size)
                    encoded = b64encode(data).decode("ascii")
                    chunk_code = self._upload_chunk_code(
                        remote_temp=remote_temp,
                        offset=offset,
                        encoded=encoded,
                        reset=offset == 0 and not resume,
                    )
                    await self.runner.run_hidden(
                        chunk_code, timeout=timeout, kernel_id=kernel_id
                    )
                    offset += len(data)
                    await _report_progress(
                        progress,
                        stage="uploading",
                        transferred_bytes=offset,
                        total_bytes=size,
                        completed_parts=min(
                            (offset + chunk_size - 1) // chunk_size, part_count
                        ),
                        total_parts=part_count,
                    )

            await _report_progress(
                progress, stage="verifying", transferred_bytes=size, total_bytes=size
            )
            code = self._upload_finalize_code(
                destination=destination,
                transfer_id=transfer_id,
                expected_size=size,
                expected_sha256=sha256,
                overwrite=overwrite,
            )
            finalize_outputs = await self.runner.run_hidden(
                code, timeout=timeout, kernel_id=kernel_id
            )
        meta = _parse_meta(finalize_outputs)
        await _report_progress(
            progress,
            stage="completed",
            transferred_bytes=size,
            total_bytes=size,
            sha256=meta["sha256"],
        )
        return [
            f"Uploaded {source} -> {meta['server_path']}\n"
            f"Bytes: {meta['size']}\nSHA-256: {meta['sha256']}\nParts: {part_count}",
            *finalize_outputs,
        ]

    @staticmethod
    def _upload_partial_size_code(remote_temp: str) -> str:
        return f"""from pathlib import Path
import json
path = Path({json.dumps(remote_temp)})
meta = {{'direction':'upload','partial_path':str(path),'size':path.stat().st_size if path.exists() else 0}}
print({META_PREFIX!r} + json.dumps(meta, sort_keys=True))
"""

    @staticmethod
    def _upload_chunk_code(
        *, remote_temp: str, offset: int, encoded: str, reset: bool
    ) -> str:
        return f"""from pathlib import Path
import base64
path = Path({json.dumps(remote_temp)})
path.parent.mkdir(parents=True, exist_ok=True)
if {reset!r}:
    path.unlink(missing_ok=True)
actual_size = path.stat().st_size if path.exists() else 0
if actual_size != {offset}:
    raise RuntimeError(f'Remote partial offset mismatch: expected {offset}, got {{actual_size}}')
with path.open('ab') as stream:
    stream.write(base64.b64decode({json.dumps(encoded)}))
print(path.stat().st_size)
"""

    @staticmethod
    def _upload_finalize_code(
        *,
        destination: str,
        transfer_id: str,
        expected_size: int,
        expected_sha256: str,
        overwrite: bool,
    ) -> str:
        resolver = _remote_resolver_code(_server_path_code(destination))
        return f"""from pathlib import Path
import hashlib, json, os, shutil
{resolver}
if target.exists() and not {overwrite!r}:
    raise FileExistsError(f'{{target}} already exists; set overwrite=true to replace it')
temp_root.mkdir(parents=True, exist_ok=True)
assembled = temp_root / 'kaggle_mcp_upload_{transfer_id}.partial'
digest = hashlib.sha256()
with assembled.open('rb') as source:
    while chunk := source.read(1024 * 1024): digest.update(chunk)
actual_sha256 = digest.hexdigest()
if assembled.stat().st_size != {expected_size} or actual_sha256 != {expected_sha256!r}:
    assembled.unlink(missing_ok=True)
    raise RuntimeError('Uploaded file failed size/SHA-256 verification')
target.parent.mkdir(parents=True, exist_ok=True)
if target.parent == temp_root:
    os.replace(assembled, target)
else:
    partial = target.with_name(target.name + '.kaggle-mcp-partial')
    if partial.exists(): partial.unlink()
    shutil.copyfile(assembled, partial)
    os.replace(partial, target)
    assembled.unlink()
meta = {{'direction':'upload','server_path':str(target),'size':target.stat().st_size,'sha256':actual_sha256}}
print({META_PREFIX!r} + json.dumps(meta, sort_keys=True))
"""

    async def download(
        self,
        *,
        server_path: str,
        local_path: str,
        notebook_path: str,
        notebook_name: str | None,
        overwrite: bool,
        chunk_size_mib: int,
        timeout: int,
        progress: ProgressCallback | None = None,
        cancel_event: asyncio.Event | None = None,
        resume: bool = False,
        keep_partial_on_failure: bool = False,
    ) -> list[Any]:
        source = normalize_server_file_path(server_path)
        destination = resolve_local_path(local_path, must_exist=False)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} already exists; set overwrite=true")
        destination.parent.mkdir(parents=True, exist_ok=True)
        chunk_size = chunk_size_mib * 1024 * 1024
        transfer_id = hashlib.sha256(
            f"{source}:{destination}".encode()
        ).hexdigest()[:16]
        partial = destination.with_name(destination.name + f".kaggle-mcp-{transfer_id}.partial")
        complete = False

        async with self.runner.lock:
            await _report_progress(
                progress, stage="connecting", transferred_bytes=0, total_bytes=0
            )
            _, _, kernel_id = await self.runner.ensure_connected(
                notebook_path, notebook_name
            )
            inspect_code = self._download_inspect_code(source)
            # Inspect through the kernel without persisting a cell.  Persisting the
            # inspection in the file being downloaded (when it is a notebook)
            # changes its size and hash between inspection and chunk reads.
            inspect_outputs = await self.runner.run_hidden(
                inspect_code, timeout=timeout, kernel_id=kernel_id
            )
            meta = _parse_meta(inspect_outputs)
            part_count = max(1, (meta["size"] + chunk_size - 1) // chunk_size)
            digest = hashlib.sha256()
            start_offset = 0
            if resume and partial.exists():
                start_offset = partial.stat().st_size
                if start_offset > meta["size"] or (
                    start_offset != meta["size"] and start_offset % chunk_size != 0
                ):
                    raise RuntimeError(
                        "Local partial size is incompatible with the remote file/chunk size"
                    )
                with partial.open("rb") as existing:
                    for data in iter(lambda: existing.read(1024 * 1024), b""):
                        digest.update(data)
            await _report_progress(
                progress,
                stage="downloading",
                transferred_bytes=start_offset,
                total_bytes=meta["size"],
                completed_parts=min(start_offset // chunk_size, part_count),
                total_parts=part_count,
                resumed_bytes=start_offset,
                sha256=meta["sha256"],
            )
            try:
                mode = "ab" if start_offset else "wb"
                with partial.open(mode) as output:
                    for offset in range(start_offset, meta["size"], chunk_size):
                        if cancel_event is not None and cancel_event.is_set():
                            raise asyncio.CancelledError
                        stage_code = self._download_stage_code(
                            source=source,
                            offset=offset,
                            length=chunk_size,
                        )
                        outputs = await self.runner.run_hidden(
                            stage_code, timeout=timeout, kernel_id=kernel_id
                        )
                        data = _parse_chunk(outputs)
                        digest.update(data)
                        output.write(data)
                        transferred = offset + len(data)
                        await _report_progress(
                            progress,
                            stage="downloading",
                            transferred_bytes=transferred,
                            total_bytes=meta["size"],
                            completed_parts=min(
                                (transferred + chunk_size - 1) // chunk_size,
                                part_count,
                            ),
                            total_parts=part_count,
                        )
                await _report_progress(
                    progress,
                    stage="verifying",
                    transferred_bytes=meta["size"],
                    total_bytes=meta["size"],
                )
                actual_sha256 = digest.hexdigest()
                if partial.stat().st_size != meta["size"] or actual_sha256 != meta["sha256"]:
                    raise RuntimeError("Downloaded file failed size/SHA-256 verification")
                os.replace(partial, destination)
                complete = True
            finally:
                if not complete and not keep_partial_on_failure and partial.exists():
                    partial.unlink()

            confirmation = await self.runner.run_hidden(
                "print(" + json.dumps(
                    f"MCP download verified: {Path(destination).name} | "
                    f"{meta['size']} bytes | SHA-256 {meta['sha256']}"
                ) + ")",
                timeout=timeout,
                kernel_id=kernel_id,
            )
        await _report_progress(
            progress,
            stage="completed",
            transferred_bytes=meta["size"],
            total_bytes=meta["size"],
            sha256=meta["sha256"],
        )
        return [
            f"Downloaded {meta['server_path']} -> {destination}\n"
            f"Bytes: {meta['size']}\nSHA-256: {meta['sha256']}\nParts: {part_count}",
            *confirmation,
        ]

    @staticmethod
    def _download_inspect_code(source: str) -> str:
        resolver = _remote_resolver_code(_server_path_code(source))
        return f"""from pathlib import Path
import hashlib, json
{resolver}
if not target.is_file(): raise FileNotFoundError(str(target))
digest = hashlib.sha256()
with target.open('rb') as stream:
    while chunk := stream.read(1024 * 1024): digest.update(chunk)
meta = {{'direction':'download','server_path':str(target),'size':target.stat().st_size,'sha256':digest.hexdigest()}}
print({META_PREFIX!r} + json.dumps(meta, sort_keys=True))
"""

    @staticmethod
    def _download_stage_code(
        *, source: str, offset: int, length: int
    ) -> str:
        resolver = _remote_resolver_code(_server_path_code(source))
        return f"""from pathlib import Path
import base64
{resolver}
with target.open('rb') as source_file:
    source_file.seek({offset})
    chunk = source_file.read({length})
print({CHUNK_PREFIX!r} + base64.b64encode(chunk).decode('ascii'))
"""


def validate_chunk_size(chunk_size_mib: int) -> int:
    if not 1 <= chunk_size_mib <= MAX_CHUNK_MIB:
        raise ValueError(f"chunk_size_mib must be between 1 and {MAX_CHUNK_MIB}")
    return chunk_size_mib
