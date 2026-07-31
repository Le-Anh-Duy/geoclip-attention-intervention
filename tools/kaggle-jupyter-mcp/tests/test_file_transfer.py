from __future__ import annotations

from base64 import b64encode

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


@pytest.mark.parametrize("value", [0, 33])
def test_validate_chunk_size_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        validate_chunk_size(value)


def test_shell_bridge_runs_bash_and_reports_exit_code():
    code = build_shell_code("printf '%s\\n' hello | wc -l", "/kaggle/temp")

    compile(code, "<shell-bridge>", "exec")
    assert "['bash', '-lc'" in code
    assert "printf '%s\\\\n' hello | wc -l" in code
    assert "/kaggle/temp" in code
    assert "MCP shell exit code" in code
