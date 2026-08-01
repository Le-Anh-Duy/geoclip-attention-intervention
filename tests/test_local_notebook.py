from __future__ import annotations

import asyncio
import json

import nbformat
import pytest

from kaggle_jupyter_mcp.local_notebook import LocalNotebookManager
from kaggle_jupyter_mcp.rest import JupyterRestClient


class FakeJupyterClient:
    base_url = "https://example.test/jupyter"

    def __init__(self) -> None:
        self.started = 0
        self.interrupted: list[str] = []
        self.shutdown: list[str] = []

    def start_kernel(self, name: str, path: str):
        self.started += 1
        return {"id": "kernel-1", "name": name, "execution_state": "idle"}

    def get_kernel(self, kernel_id: str):
        return {"id": kernel_id, "execution_state": "idle"}

    def kernel_action(self, kernel_id: str, action: str):
        self.interrupted.append(f"{kernel_id}:{action}")

    def shutdown_kernel(self, kernel_id: str):
        self.shutdown.append(kernel_id)


def make_notebook(path):
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Demo"),
            nbformat.v4.new_code_cell("print('hello')"),
            nbformat.v4.new_code_cell("print('second')"),
        ]
    )
    nbformat.write(notebook, path)


@pytest.mark.asyncio
async def test_local_cell_runs_on_raw_kernel_and_saves_output(tmp_path, monkeypatch):
    path = tmp_path / "demo.ipynb"
    make_notebook(path)
    monkeypatch.setenv("KAGGLE_JUPYTER_LOCAL_ROOTS", str(tmp_path))
    client = FakeJupyterClient()
    monkeypatch.setattr(
        JupyterRestClient,
        "from_current_config",
        classmethod(lambda cls: client),
    )

    calls = []

    async def execute_code(**kwargs):
        calls.append(kwargs)
        return ["hello\n"]

    manager = LocalNotebookManager(execute_code)
    connected = await manager.connect(
        str(path),
        kernel_id=None,
        kernel_name="python3",
        remote_working_dir="/kaggle/working",
        reuse_existing=True,
    )
    result = await manager.execute_cell(str(path), 1, timeout=30)

    saved = nbformat.read(path, as_version=4)
    assert connected["kernel_id"] == "kernel-1"
    assert calls == [
        {"code": "print('hello')", "timeout": 30, "kernel_id": "kernel-1"}
    ]
    assert json.loads(result[0])["outputs_saved"] is True
    assert saved.cells[1].execution_count == 1
    assert saved.cells[1].outputs[0].text == "hello\n"


@pytest.mark.asyncio
async def test_background_local_notebook_run_is_pollable(tmp_path, monkeypatch):
    path = tmp_path / "demo.ipynb"
    make_notebook(path)
    monkeypatch.setenv("KAGGLE_JUPYTER_LOCAL_ROOTS", str(tmp_path))
    client = FakeJupyterClient()
    monkeypatch.setattr(
        JupyterRestClient,
        "from_current_config",
        classmethod(lambda cls: client),
    )

    async def execute_code(**kwargs):
        await asyncio.sleep(0)
        return [kwargs["code"]]

    manager = LocalNotebookManager(execute_code)
    await manager.connect(
        str(path),
        kernel_id=None,
        kernel_name="python3",
        remote_working_dir="/kaggle/working",
        reuse_existing=True,
    )
    started = await manager.start_run(
        str(path),
        start_cell=0,
        stop_cell=None,
        timeout_per_cell=30,
        stop_on_error=True,
    )
    await manager.runs[started["run_id"]].task
    completed = manager.run_status(started["run_id"])

    assert completed["state"] == "completed"
    assert completed["completed_cells"] == 2
    assert completed["percent"] == 100.0
    assert [item["cell_index"] for item in completed["results"]] == [1, 2]
