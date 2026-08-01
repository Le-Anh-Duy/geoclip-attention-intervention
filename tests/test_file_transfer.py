from __future__ import annotations

import asyncio
import hashlib
import json
from base64 import b64encode
from types import SimpleNamespace

import pytest

from kaggle_jupyter_mcp.file_transfer import (
    CHUNK_PREFIX,
    FileTransferService,
    _parse_chunk,
    normalize_server_file_path,
    resolve_local_path,
    validate_chunk_size,
)
from kaggle_jupyter_mcp.workflow_tools import build_shell_code


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("artifacts/result.bin", "artifacts/result.bin"),
        ("/kaggle/working/result.bin", "/kaggle/working/result.bin"),
        ("/kaggle/temp/result.bin", "/kaggle/temp/result.bin"),
    ],
)
def test_normalize_server_file_path_allows_working_and_temp(value, expected):
    assert normalize_server_file_path(value) == expected


@pytest.mark.parametrize("value", ["", "../secret", "/etc/passwd"])
def test_normalize_server_file_path_rejects_escape(value):
    with pytest.raises(ValueError):
        normalize_server_file_path(value)


def test_resolve_local_path_is_limited_to_configured_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "source.bin"
    source.write_bytes(b"data")
    monkeypatch.setenv("KAGGLE_JUPYTER_LOCAL_ROOTS", str(allowed))

    assert resolve_local_path(str(source), must_exist=True) == source.resolve()
    with pytest.raises(ValueError):
        resolve_local_path(str(tmp_path / "outside.bin"), must_exist=False)


def test_download_chunk_parser_round_trips_binary():
    payload = b"\x00binary\xff"
    outputs = [CHUNK_PREFIX + b64encode(payload).decode("ascii")]
    assert _parse_chunk(outputs) == payload


def test_download_inspection_does_not_mutate_source_notebook(tmp_path, monkeypatch):
    payload = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5}'
    digest = hashlib.sha256(payload).hexdigest()

    class FakeRunner:
        def __init__(self):
            self.lock = asyncio.Lock()
            self.hidden_calls = []
            self.visible_calls = []

        async def ensure_connected(self, notebook_path, notebook_name):
            return notebook_path, notebook_name or "transfer", "kernel-id"

        async def run_hidden(self, code, *, timeout, kernel_id):
            self.hidden_calls.append(code)
            if "'direction':'download'" in code:
                meta = {
                    "direction": "download",
                    "server_path": "/kaggle/working/source.ipynb",
                    "size": len(payload),
                    "sha256": digest,
                }
                return ["KAGGLE_MCP_TRANSFER_META " + json.dumps(meta)]
            return [CHUNK_PREFIX + b64encode(payload).decode("ascii")]

        async def run_locked(self, code, **kwargs):
            self.visible_calls.append(code)
            return SimpleNamespace(outputs=["verified"], as_mcp_content=lambda: [])

    monkeypatch.setenv("KAGGLE_JUPYTER_LOCAL_ROOTS", str(tmp_path))
    destination = tmp_path / "source.ipynb"
    runner = FakeRunner()

    asyncio.run(
        FileTransferService(runner).download(
            server_path="source.ipynb",
            local_path=str(destination),
            notebook_path="source.ipynb",
            notebook_name="source",
            overwrite=False,
            chunk_size_mib=1,
            timeout=30,
        )
    )

    assert destination.read_bytes() == payload
    assert len(runner.hidden_calls) == 3
    assert len(runner.visible_calls) == 0
    assert all("'direction':'download'" not in code for code in runner.visible_calls)


def test_upload_finalizer_uses_kaggle_temp_and_hash_verification():
    code = FileTransferService._upload_finalize_code(
        destination="results/model.bin",
        transfer_id="abc123",
        expected_size=42,
        expected_sha256="f" * 64,
        overwrite=False,
    )

    assert "/kaggle/temp" in code
    assert "kaggle_mcp_upload_abc123.partial" in code
    assert "failed size/SHA-256 verification" in code


@pytest.mark.parametrize("value", [0, 9])
def test_validate_chunk_size_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        validate_chunk_size(value)


def test_shell_bridge_runs_bash_and_reports_exit_code():
    code = build_shell_code("printf '%s\\n' hello | wc -l", "/kaggle/temp")

    compile(code, "<shell-bridge>", "exec")
    assert "['bash', '-lc'" in code
    assert "printf '%s\\\\n' hello | wc -l" in code
    assert "/kaggle/temp" in code
    assert "temp_root.mkdir(parents=True, exist_ok=True)" in code
    assert "MCP shell exit code" in code


def test_upload_chunk_requires_exact_remote_offset():
    code = FileTransferService._upload_chunk_code(
        remote_temp="/kaggle/temp/partial.bin",
        offset=1024,
        encoded="ZGF0YQ==",
        reset=False,
    )

    compile(code, "<upload-chunk>", "exec")
    assert "actual_size != 1024" in code
    assert "Remote partial offset mismatch" in code
