"""REST-backed notebook model used when Jupyter RTC is unavailable."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import MutableSequence
from copy import deepcopy
from typing import Any

import nbformat
import requests
from typing_extensions import Self

from kaggle_jupyter_mcp.rest import JupyterAPIError, JupyterRestClient

_COLLABORATION_SUPPORT: dict[str, bool] = {}


def _source_text(source: str | list[str] | None) -> str:
    if source is None:
        return ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class NonNotebookDocumentError(RuntimeError):
    pass


class RestNotebookModel(MutableSequence[dict[str, Any]]):
    """A minimal NbModelClient-compatible object persisted via Contents API."""

    def __init__(
        self,
        *,
        client: JupyterRestClient,
        path: str,
        notebook: dict[str, Any],
        last_modified: str | None,
    ) -> None:
        self.client = client
        self.path = path
        self._notebook = deepcopy(notebook)
        self._notebook.setdefault("cells", [])
        self._notebook.setdefault("metadata", {})
        self._notebook.setdefault("nbformat", 4)
        self._notebook.setdefault("nbformat_minor", 5)
        self._last_modified = last_modified
        self._dirty = False
        self._lock = threading.RLock()

    @classmethod
    async def open(cls, notebook_info: dict[str, Any]) -> RestNotebookModel:
        from jupyter_mcp_server.config import get_config
        from jupyter_mcp_server.server_context import ServerContext

        config = get_config()
        path = notebook_info.get("path") or config.document_id
        if not path:
            raise ValueError(
                "The active kernel is not associated with a notebook path. "
                "Call use_notebook with a real .ipynb path before using cell tools."
            )
        base_url = (
            notebook_info.get("server_url")
            or config.document_url
            or config.code_sandbox_url
        )
        token = (
            notebook_info.get("token")
            or config.document_token
            or config.code_sandbox_token
        )
        headers = dict(ServerContext.get_instance().document_auth_headers or {})
        client = JupyterRestClient(
            base_url=base_url, token=token, extra_headers=headers
        )
        try:
            model = await asyncio.to_thread(
                client.get_content, path, content=True, model_type="notebook"
            )
        except JupyterAPIError as error:
            if error.status == 400:
                try:
                    file_model = await asyncio.to_thread(
                        client.get_content,
                        path,
                        content=True,
                        model_type="file",
                        format="text",
                    )
                except JupyterAPIError:
                    raise error
                if file_model.get("type") == "file":
                    raise NonNotebookDocumentError(
                        f"'{path}' has an .ipynb name but the Kaggle server exposes it as plain "
                        "text, not nbformat JSON. Use read_server_file/edit_server_text instead; "
                        "cell indices are not available for this virtual source document."
                    ) from error
            raise
        if model.get("type") != "notebook" or not isinstance(
            model.get("content"), dict
        ):
            raise NonNotebookDocumentError(
                f"'{path}' is not a Jupyter notebook document"
            )
        return cls(
            client=client,
            path=path,
            notebook=model["content"],
            last_modified=model.get("last_modified"),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None and self._dirty:
            await asyncio.to_thread(self.save)

    def __len__(self) -> int:
        with self._lock:
            return len(self._notebook["cells"])

    def __getitem__(self, index):
        with self._lock:
            return deepcopy(self._notebook["cells"][index])

    def __setitem__(self, index, value: dict[str, Any]) -> None:
        with self._lock:
            self._notebook["cells"][index] = deepcopy(dict(value))
            self._dirty = True

    def __delitem__(self, index):
        with self._lock:
            removed = self._notebook["cells"].pop(index)
            self._dirty = True
            return deepcopy(removed)

    def insert(self, index: int, value: dict[str, Any]) -> None:
        with self._lock:
            self._notebook["cells"].insert(index, deepcopy(dict(value)))
            self._dirty = True

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._notebook["metadata"])

    @property
    def nbformat(self) -> int:
        return int(self._notebook["nbformat"])

    @property
    def nbformat_minor(self) -> int:
        return int(self._notebook["nbformat_minor"])

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._notebook)

    def insert_cell(
        self, index: int, source: str, cell_type: str, **kwargs: Any
    ) -> None:
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source=source, **kwargs)
        elif cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source, **kwargs)
        elif cell_type == "raw":
            cell = nbformat.v4.new_raw_cell(source=source, **kwargs)
        else:
            raise ValueError("cell_type must be code, markdown, or raw")
        # Cell IDs were added in nbformat 4.5. Kaggle's notebook template still
        # uses an older minor version, where an `id` is rejected as an extra key.
        if self.nbformat_minor >= 5:
            from uuid import uuid4

            cell.setdefault("id", uuid4().hex[:8])
        else:
            cell.pop("id", None)
        actual = len(self) if index == -1 else index
        if actual < 0 or actual > len(self):
            raise IndexError(f"Cell index {index} out of range for {len(self)} cells")
        self.insert(actual, dict(cell))

    def set_cell_source(self, index: int, source: str) -> None:
        cell = self[index]
        cell["source"] = source
        self[index] = cell

    def get_cell_source(self, index: int) -> str:
        """Return source text using the interface expected by upstream cell tools."""

        return _source_text(self[index].get("source"))

    def delete_cell(self, index: int) -> dict[str, Any]:
        return self.__delitem__(index)

    def delete_many_cells(self, index_list: list[int]) -> list[dict[str, Any]]:
        if len(set(index_list)) != len(index_list):
            raise ValueError("Cell indices must be unique")
        with self._lock:
            total = len(self._notebook["cells"])
            for index in index_list:
                if index < 0 or index >= total:
                    raise IndexError(
                        f"Cell index {index} out of range for {total} cells"
                    )
            removed_by_index: dict[int, dict[str, Any]] = {}
            for index in sorted(index_list, reverse=True):
                removed_by_index[index] = self._notebook["cells"].pop(index)
            self._dirty = True
            return [deepcopy(removed_by_index[index]) for index in index_list]

    def _append_kernel_message(self, message: dict[str, Any]) -> None:
        msg_type = message.get("msg_type") or message.get("header", {}).get("msg_type")
        content = deepcopy(message.get("content", {}))
        with self._lock:
            cell = self._notebook["cells"][self._executing_index]
            outputs = cell.setdefault("outputs", [])
            if msg_type == "clear_output":
                outputs.clear()
            elif msg_type == "stream":
                outputs.append(
                    {
                        "output_type": "stream",
                        "name": content.get("name", "stdout"),
                        "text": content.get("text", ""),
                    }
                )
            elif msg_type in {"display_data", "update_display_data"}:
                outputs.append(
                    {
                        "output_type": "display_data",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                    }
                )
            elif msg_type == "execute_result":
                outputs.append(
                    {
                        "output_type": "execute_result",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                        "execution_count": content.get("execution_count"),
                    }
                )
            elif msg_type == "error":
                outputs.append(
                    {
                        "output_type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                )
            else:
                return
            self._dirty = True

    def execute_cell(
        self,
        index: int,
        kernel_client: Any,
        silent: bool = False,
        store_history: bool = True,
        stop_on_error: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        cell = self[index]
        if cell.get("cell_type") != "code":
            raise ValueError(f"Cell {index} is not a code cell")
        with self._lock:
            self._executing_index = index
            live_cell = self._notebook["cells"][index]
            live_cell["outputs"] = []
            live_cell["execution_count"] = None
            self._dirty = True
        reply = kernel_client.execute_interactive(
            _source_text(cell.get("source")),
            output_hook=self._append_kernel_message,
            allow_stdin=False,
            silent=silent,
            store_history=False if silent else store_history,
            stop_on_error=stop_on_error,
            timeout=timeout,
        )
        reply_content = reply.get("content", {})
        with self._lock:
            live_cell = self._notebook["cells"][index]
            live_cell["execution_count"] = reply_content.get("execution_count")
            outputs = deepcopy(live_cell.get("outputs", []))
        return {
            "execution_count": reply_content.get("execution_count"),
            "status": reply_content.get("status", "unknown"),
            "outputs": outputs,
        }

    def save(self) -> None:
        if not self._dirty:
            return
        current = self.client.get_content(self.path, content=False)
        current_modified = current.get("last_modified")
        if (
            self._last_modified
            and current_modified
            and current_modified != self._last_modified
            and not _truthy_env("KAGGLE_JUPYTER_FORCE_OVERWRITE", False)
        ):
            raise RuntimeError(
                f"'{self.path}' changed on the server after it was read; refusing to overwrite it. "
                "Retry the operation against a fresh notebook state."
            )
        if _truthy_env("KAGGLE_JUPYTER_AUTO_CHECKPOINT", True):
            try:
                self.client.create_checkpoint(self.path)
            except JupyterAPIError as error:
                if error.status not in {400, 404, 405, 501}:
                    raise
        with self._lock:
            payload = deepcopy(self._notebook)
        # Validate before crossing the network so malformed cell output cannot corrupt a file.
        nbformat.validate(nbformat.from_dict(payload))
        saved = self.client.save_content(
            self.path, content=payload, model_type="notebook", format="json"
        )
        self._last_modified = saved.get("last_modified")
        self._dirty = False


def _collaboration_unavailable(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, requests.HTTPError):
            response = current.response
            if response is not None and response.status_code in {404, 405, 501}:
                return True
        text = str(current).lower()
        if "api/collaboration/session" in text and any(
            marker in text for marker in ("404", "405", "not found", "not implemented")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def install_notebook_fallback() -> None:
    """Patch upstream NotebookConnection once, preserving RTC when available."""

    from jupyter_mcp_server.notebook_manager import NotebookConnection

    if getattr(NotebookConnection, "_kaggle_rest_fallback_installed", False):
        return

    original_aenter = NotebookConnection.__aenter__

    async def patched_aenter(self):
        base_url = str(self.notebook_info.get("server_url") or "").rstrip("/")
        if _COLLABORATION_SUPPORT.get(base_url) is not False:
            try:
                notebook = await original_aenter(self)
                _COLLABORATION_SUPPORT[base_url] = True
                return notebook
            except Exception as error:
                if not _collaboration_unavailable(error):
                    raise
                _COLLABORATION_SUPPORT[base_url] = False

        model = await RestNotebookModel.open(self.notebook_info)
        self._notebook = model
        return await model.__aenter__()

    NotebookConnection.__aenter__ = patched_aenter
    NotebookConnection._kaggle_rest_fallback_installed = True
