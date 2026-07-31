"""Visible code execution backed by a real, persisted notebook."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from kaggle_jupyter_mcp.rest import JupyterAPIError, JupyterRestClient


def normalize_notebook_target(path: str, name: str | None = None) -> tuple[str, str]:
    clean = path.replace("\\", "/").strip("/")
    if not clean or not clean.endswith(".ipynb"):
        raise ValueError("notebook_path must name a real .ipynb file")
    if any(part in {"", ".", ".."} for part in PurePosixPath(clean).parts):
        raise ValueError("notebook_path must stay under the Jupyter server root")
    notebook_name = (name or PurePosixPath(clean).stem).strip()
    if not notebook_name:
        raise ValueError("notebook_name cannot be empty")
    return clean, notebook_name


@dataclass(slots=True)
class VisibleRunResult:
    notebook_path: str
    notebook_name: str
    kernel_id: str
    cell_index: int
    outputs: list[Any]

    def as_mcp_content(self) -> list[Any]:
        header = (
            f"Notebook: {self.notebook_path}\n"
            f"Notebook name: {self.notebook_name}\n"
            f"Kernel: {self.kernel_id}\n"
            f"Executed cell: {self.cell_index}"
        )
        return [header, *(self.outputs or ["(cell completed with no output)"])]


class VisibleNotebookRunner:
    """Serialize notebook selection and preserve every execution as a code cell."""

    def __init__(self, original_execute_code) -> None:
        self.original_execute_code = original_execute_code
        self.lock = asyncio.Lock()

    async def ensure_connected(
        self, notebook_path: str, notebook_name: str | None = None
    ) -> tuple[str, str, str]:
        from jupyter_mcp_server import server

        path, name = normalize_notebook_target(notebook_path, notebook_name)
        client = JupyterRestClient.from_current_config()
        try:
            model = await asyncio.to_thread(client.get_content, path, content=False)
            if model.get("type") != "notebook":
                raise ValueError(f"'{path}' exists but is not a notebook")
            mode = "connect"
        except JupyterAPIError as error:
            if error.status != 404:
                raise
            mode = "create"

        connected = await server.use_notebook(
            notebook_name=name, notebook_path=path, mode=mode, kernel_id=None
        )
        if isinstance(connected, str) and connected.startswith("Error executing tool"):
            raise RuntimeError(connected)
        kernel_id = server.notebook_manager.get_kernel_id(name)
        if not kernel_id:
            raise RuntimeError(f"Notebook '{name}' has no connected kernel")
        return path, name, kernel_id

    async def run_locked(
        self,
        code: str,
        *,
        timeout: int,
        notebook_path: str,
        notebook_name: str | None = None,
    ) -> VisibleRunResult:
        from jupyter_mcp_server import server

        path, name, kernel_id = await self.ensure_connected(
            notebook_path, notebook_name
        )
        outputs = await server.insert_execute_code_cell(
            cell_index=-1, cell_source=code, timeout=timeout
        )
        if not isinstance(outputs, list):
            outputs = [outputs]
        client = JupyterRestClient.from_current_config()
        model = await asyncio.to_thread(
            client.get_content, path, content=True, model_type="notebook"
        )
        cells = model.get("content", {}).get("cells", [])
        if not cells:
            raise RuntimeError("Execution completed but the notebook has no cells")
        return VisibleRunResult(path, name, kernel_id, len(cells) - 1, outputs)

    async def run(
        self,
        code: str,
        *,
        timeout: int,
        notebook_path: str,
        notebook_name: str | None = None,
    ) -> VisibleRunResult:
        async with self.lock:
            return await self.run_locked(
                code,
                timeout=timeout,
                notebook_path=notebook_path,
                notebook_name=notebook_name,
            )

    async def run_hidden(
        self, code: str, *, timeout: int, kernel_id: str
    ) -> list[Any]:
        outputs = await self.original_execute_code(
            code=code, timeout=timeout, kernel_id=kernel_id
        )
        return outputs if isinstance(outputs, list) else [outputs]
