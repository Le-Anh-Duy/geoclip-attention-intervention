"""Small, explicit client for the Jupyter Server REST API.

The upstream standalone MCP client uses the real-time collaboration WebSocket
for notebook documents. Kaggle exposes the regular Jupyter REST and kernel APIs
but not the collaboration endpoint. This client is the compatibility boundary
used by both the notebook fallback and the extra MCP tools.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

DEFAULT_TIMEOUT = 30


class JupyterAPIError(RuntimeError):
    """An HTTP error that does not disclose the signed Kaggle proxy URL."""

    def __init__(self, method: str, endpoint: str, status: int, message: str = ""):
        detail = f": {message}" if message else ""
        super().__init__(
            f"Jupyter API {method} {endpoint} returned HTTP {status}{detail}"
        )
        self.method = method
        self.endpoint = endpoint
        self.status = status
        self.message = message


def api_path(path: str) -> str:
    """Quote a Jupyter API path while preserving directory separators."""

    clean = path.replace("\\", "/").strip("/")
    if any(part == ".." for part in clean.split("/")):
        raise ValueError("Jupyter paths must not contain '..'")
    return quote(clean, safe="/")


@dataclass(slots=True)
class JupyterRestClient:
    base_url: str
    token: str | None = None
    extra_headers: dict[str, str] | None = None
    timeout: int = DEFAULT_TIMEOUT
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("A Jupyter document URL is required")
        if self.session is None:
            self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        if self.extra_headers:
            self.session.headers.update(self.extra_headers)

    @classmethod
    def from_current_config(cls) -> JupyterRestClient:
        """Create a client after dynamic MCP connection settings are resolved."""

        from jupyter_mcp_server.config import get_config
        from jupyter_mcp_server.server_context import ServerContext

        config = get_config()
        base_url = config.document_url or config.code_sandbox_url
        token = config.document_token or config.code_sandbox_token
        context = ServerContext.get_instance()
        headers = dict(context.document_auth_headers or {})
        return cls(base_url=base_url, token=token, extra_headers=headers)

    def _url(self, endpoint: str) -> str:
        # urljoin would discard the signed `/.../proxy` prefix used by Kaggle.
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:500].strip()
        if isinstance(payload, dict):
            return str(payload.get("message") or payload.get("reason") or "")[:500]
        return str(payload)[:500]

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        expected: Iterable[int] = (200,),
    ) -> requests.Response:
        assert self.session is not None
        response = self.session.request(
            method,
            self._url(endpoint),
            params=params,
            json=json,
            timeout=self.timeout,
        )
        if response.status_code not in set(expected):
            raise JupyterAPIError(
                method.upper(),
                endpoint,
                response.status_code,
                self._error_message(response),
            )
        return response

    def request_json(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        response = self.request(method, endpoint, **kwargs)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # Contents -------------------------------------------------------------

    def get_content(
        self,
        path: str,
        *,
        content: bool = True,
        model_type: str | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"content": 1 if content else 0}
        if model_type:
            params["type"] = model_type
        if format:
            params["format"] = format
        return self.request_json("GET", f"api/contents/{api_path(path)}", params=params)

    def save_content(
        self,
        path: str,
        *,
        content: Any,
        model_type: str,
        format: str,
    ) -> dict[str, Any]:
        body = {"type": model_type, "format": format, "content": content}
        return self.request_json(
            "PUT", f"api/contents/{api_path(path)}", json=body, expected=(200, 201)
        )

    def create_untitled(
        self, parent: str = "", *, model_type: str = "notebook", ext: str = ""
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"type": model_type}
        if ext:
            body["ext"] = ext
        return self.request_json(
            "POST", f"api/contents/{api_path(parent)}", json=body, expected=(201,)
        )

    def rename(self, path: str, new_path: str) -> dict[str, Any]:
        return self.request_json(
            "PATCH",
            f"api/contents/{api_path(path)}",
            json={"path": new_path},
            expected=(200,),
        )

    def copy(self, source_path: str, destination_dir: str = "") -> dict[str, Any]:
        return self.request_json(
            "POST",
            f"api/contents/{api_path(destination_dir)}",
            json={"copy_from": source_path},
            expected=(201,),
        )

    def delete(self, path: str) -> None:
        self.request("DELETE", f"api/contents/{api_path(path)}", expected=(204,))

    def list_checkpoints(self, path: str) -> list[dict[str, Any]]:
        return self.request_json("GET", f"api/contents/{api_path(path)}/checkpoints")

    def create_checkpoint(self, path: str) -> dict[str, Any]:
        return self.request_json(
            "POST", f"api/contents/{api_path(path)}/checkpoints", expected=(201,)
        )

    def restore_checkpoint(self, path: str, checkpoint_id: str) -> None:
        self.request(
            "POST",
            f"api/contents/{api_path(path)}/checkpoints/{quote(checkpoint_id, safe='')}",
            expected=(204,),
        )

    def delete_checkpoint(self, path: str, checkpoint_id: str) -> None:
        self.request(
            "DELETE",
            f"api/contents/{api_path(path)}/checkpoints/{quote(checkpoint_id, safe='')}",
            expected=(204,),
        )

    def trust_notebook(self, path: str) -> None:
        self.request("POST", f"api/contents/{api_path(path)}/trust", expected=(201,))

    # Kernels and sessions -------------------------------------------------

    def list_kernels(self) -> list[dict[str, Any]]:
        return self.request_json("GET", "api/kernels")

    def get_kernel(self, kernel_id: str) -> dict[str, Any]:
        return self.request_json("GET", f"api/kernels/{quote(kernel_id, safe='')}")

    def start_kernel(self, name: str = "python3", path: str = "") -> dict[str, Any]:
        return self.request_json(
            "POST", "api/kernels", json={"name": name, "path": path}, expected=(201,)
        )

    def kernel_action(self, kernel_id: str, action: str) -> dict[str, Any] | None:
        if action not in {"interrupt", "restart"}:
            raise ValueError("Kernel action must be 'interrupt' or 'restart'")
        expected = (200, 204)
        return self.request_json(
            "POST",
            f"api/kernels/{quote(kernel_id, safe='')}/{action}",
            expected=expected,
        )

    def shutdown_kernel(self, kernel_id: str) -> None:
        self.request(
            "DELETE", f"api/kernels/{quote(kernel_id, safe='')}", expected=(204,)
        )

    def list_kernel_specs(self) -> dict[str, Any]:
        return self.request_json("GET", "api/kernelspecs")

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.request_json("GET", "api/sessions")

    def create_session(
        self,
        *,
        path: str,
        name: str,
        kernel_name: str = "python3",
        session_type: str = "notebook",
    ) -> dict[str, Any]:
        body = {
            "path": path,
            "name": name,
            "type": session_type,
            "kernel": {"name": kernel_name},
        }
        return self.request_json("POST", "api/sessions", json=body, expected=(201,))

    def update_session(
        self, session_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        return self.request_json(
            "PATCH",
            f"api/sessions/{quote(session_id, safe='')}",
            json=changes,
            expected=(200,),
        )

    def shutdown_session(self, session_id: str) -> None:
        self.request(
            "DELETE", f"api/sessions/{quote(session_id, safe='')}", expected=(204,)
        )

    # Terminals, status, conversion ---------------------------------------

    def list_terminals(self) -> list[dict[str, Any]]:
        return self.request_json("GET", "api/terminals")

    def create_terminal(self, cwd: str | None = None) -> dict[str, Any]:
        body = {"cwd": cwd} if cwd else {}
        return self.request_json(
            "POST", "api/terminals", json=body, expected=(200, 201)
        )

    def shutdown_terminal(self, terminal_name: str) -> None:
        self.request(
            "DELETE", f"api/terminals/{quote(terminal_name, safe='')}", expected=(204,)
        )

    def server_status(self) -> dict[str, Any]:
        return self.request_json("GET", "api/status")

    def api_spec(self) -> dict[str, Any]:
        return self.request_json("GET", "api/spec.yaml")

    def nbconvert_formats(self) -> dict[str, Any]:
        return self.request_json("GET", "api/nbconvert")

    def export_notebook(self, path: str, export_format: str) -> tuple[bytes, str]:
        response = self.request(
            "GET",
            f"nbconvert/{quote(export_format, safe='')}/{api_path(path)}",
            expected=(200,),
        )
        return response.content, response.headers.get(
            "Content-Type", "application/octet-stream"
        )

    # JupyterLab settings and workspaces ----------------------------------

    def list_lab_settings(self, *, ids_only: bool = True) -> Any:
        return self.request_json(
            "GET",
            "lab/api/settings",
            params={"ids_only": "true" if ids_only else "false"},
        )

    def get_lab_setting(self, schema_name: str) -> dict[str, Any]:
        return self.request_json(
            "GET", f"lab/api/settings/{quote(schema_name, safe='@:/')}"
        )

    def update_lab_setting(self, schema_name: str, raw_json: str) -> None:
        self.request(
            "PUT",
            f"lab/api/settings/{quote(schema_name, safe='@:/')}",
            json={"raw": raw_json},
            expected=(204,),
        )

    def list_lab_workspaces(self) -> dict[str, Any]:
        return self.request_json("GET", "lab/api/workspaces")

    def get_lab_workspace(self, workspace_name: str) -> dict[str, Any]:
        return self.request_json(
            "GET", f"lab/api/workspaces/{quote(workspace_name.strip('/'), safe='')}"
        )

    def save_lab_workspace(
        self, workspace_name: str, workspace: dict[str, Any]
    ) -> None:
        self.request(
            "PUT",
            f"lab/api/workspaces/{quote(workspace_name.strip('/'), safe='')}",
            json=workspace,
            expected=(204,),
        )

    def delete_lab_workspace(self, workspace_name: str) -> None:
        self.request(
            "DELETE",
            f"lab/api/workspaces/{quote(workspace_name.strip('/'), safe='')}",
            expected=(204,),
        )
