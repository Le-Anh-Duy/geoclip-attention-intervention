"""Additional MCP tools backed by Jupyter Server's standard REST API."""

from __future__ import annotations

import asyncio
import json
from base64 import b64encode
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

from kaggle_jupyter_mcp.rest import JupyterAPIError, JupyterRestClient


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


async def _call(method, /, *args, **kwargs):
    return await asyncio.to_thread(method, *args, **kwargs)


def _client() -> JupyterRestClient:
    return JupyterRestClient.from_current_config()


def _content_summary(model: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "path",
        "type",
        "format",
        "mimetype",
        "writable",
        "created",
        "last_modified",
        "size",
    )
    return {key: model.get(key) for key in keys if key in model}


async def _delete_via_kernel(path: str) -> str:
    """Delete an exact server-root-relative path when Kaggle's trash is broken.

    Kaggle configures Jupyter's ContentsManager to use send2trash, while the
    container trash directory is not writable. The kernel cwd is the Contents
    root (`/kaggle/working`). The generated code resolves and checks the target
    before deleting, preventing traversal outside that root.
    """

    from jupyter_mcp_server import server

    path_literal = json.dumps(path)
    code = f"""
def __kaggle_mcp_delete_exact_path():
    from pathlib import Path
    import shutil
    root = Path.cwd().resolve()
    target = (root / Path({path_literal})).resolve()
    if target == root or root not in target.parents:
        raise ValueError('Refusing to delete outside the Jupyter server root')
    if not target.exists():
        raise FileNotFoundError(str(target))
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    print('deleted', target.relative_to(root).as_posix())
__kaggle_mcp_delete_exact_path()
del __kaggle_mcp_delete_exact_path
""".strip()
    outputs = await server.execute_code(code=code, timeout=30, kernel_id=None)
    return "\n".join(str(item) for item in outputs)


def register_extra_tools(mcp) -> None:
    if getattr(mcp, "_kaggle_extra_tools_installed", False):
        return

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Probe Jupyter Server Capabilities", readOnlyHint=True
        )
    )
    async def probe_server_capabilities() -> str:
        """Probe read-only Jupyter/Kaggle endpoints and report what this server exposes."""

        client = _client()
        endpoints = {
            "server": "api",
            "status": "api/status",
            "contents": "api/contents",
            "kernels": "api/kernels",
            "kernel_specs": "api/kernelspecs",
            "sessions": "api/sessions",
            "terminals": "api/terminals",
            "nbconvert": "api/nbconvert",
            "lab_settings": "lab/api/settings",
            "lab_workspaces": "lab/api/workspaces",
            "collaboration": "api/collaboration/session",
        }
        result: dict[str, Any] = {}
        for name, endpoint in endpoints.items():
            try:
                response = await _call(client.request, "GET", endpoint, expected=(200,))
                result[name] = {
                    "available": True,
                    "status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                }
            except JupyterAPIError as error:
                result[name] = {"available": False, "status": error.status}
        result["notebook_cell_mode"] = (
            "rtc" if result["collaboration"]["available"] else "rest-contents-fallback"
        )
        return _json(result)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Jupyter Server Status", readOnlyHint=True
        )
    )
    async def get_server_status() -> str:
        """Return Jupyter server activity and connection status."""

        return _json(await _call(_client().server_status))

    @mcp.tool(
        annotations=ToolAnnotations(title="Read Jupyter Server File", readOnlyHint=True)
    )
    async def read_server_file(
        path: Annotated[
            str, Field(description="Path relative to the Jupyter server root")
        ],
        format: Annotated[
            Literal["text", "base64"], Field(description="Requested file encoding")
        ] = "text",
        max_chars: Annotated[
            int, Field(description="Maximum characters returned", ge=1, le=2_000_000)
        ] = 200_000,
    ) -> str:
        """Read any server file, including Kaggle's virtual Python source document."""

        model = await _call(
            _client().get_content,
            path,
            content=True,
            model_type="file",
            format=format,
        )
        content = str(model.get("content", ""))
        truncated = len(content) > max_chars
        payload = _content_summary(model)
        payload["content"] = content[:max_chars]
        payload["truncated"] = truncated
        return _json(payload)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Write Jupyter Server File", destructiveHint=True
        )
    )
    async def write_server_file(
        path: Annotated[
            str, Field(description="Exact destination path under the server root")
        ],
        content: Annotated[str, Field(description="Complete replacement content")],
        format: Annotated[
            Literal["text", "base64"], Field(description="Encoding of content")
        ] = "text",
        create_checkpoint: Annotated[
            bool, Field(description="Checkpoint an existing file before overwriting it")
        ] = True,
    ) -> str:
        """Create or replace a text/binary file through the Contents API."""

        client = _client()
        if create_checkpoint:
            try:
                await _call(client.create_checkpoint, path)
            except JupyterAPIError as error:
                if error.status not in {400, 404, 405, 501}:
                    raise
        model = await _call(
            client.save_content,
            path,
            content=content,
            model_type="file",
            format=format,
        )
        return _json(_content_summary(model))

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Edit Jupyter Server Text", destructiveHint=True
        )
    )
    async def edit_server_text(
        path: Annotated[str, Field(description="Text file path under the server root")],
        old_string: Annotated[str, Field(description="Exact text to find")],
        new_string: Annotated[str, Field(description="Replacement text")],
        replace_all: Annotated[
            bool, Field(description="Replace every occurrence")
        ] = False,
        create_checkpoint: Annotated[
            bool, Field(description="Checkpoint before saving")
        ] = True,
    ) -> str:
        """Perform an exact, conflict-resistant replacement in a server text file."""

        client = _client()
        model = await _call(
            client.get_content, path, content=True, model_type="file", format="text"
        )
        source = str(model.get("content", ""))
        occurrences = source.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string was not found")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                f"old_string occurs {occurrences} times; set replace_all=true or provide more context"
            )
        updated = source.replace(old_string, new_string, -1 if replace_all else 1)
        # Re-read metadata immediately before saving to catch obvious concurrent changes.
        current = await _call(client.get_content, path, content=False)
        if current.get("last_modified") != model.get("last_modified"):
            raise RuntimeError(
                "The file changed after it was read; retry with fresh content"
            )
        if create_checkpoint:
            try:
                await _call(client.create_checkpoint, path)
            except JupyterAPIError as error:
                if error.status not in {400, 404, 405, 501}:
                    raise
        saved = await _call(
            client.save_content,
            path,
            content=updated,
            model_type="file",
            format="text",
        )
        result = _content_summary(saved)
        result["replacements"] = occurrences if replace_all else 1
        return _json(result)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Create Jupyter Server Item", destructiveHint=True
        )
    )
    async def create_server_item(
        parent: Annotated[
            str, Field(description="Parent directory, empty for server root")
        ] = "",
        item_type: Annotated[
            Literal["notebook", "file", "directory"], Field(description="Item type")
        ] = "notebook",
        extension: Annotated[
            str, Field(description="Optional extension for a new file, e.g. .py")
        ] = "",
    ) -> str:
        """Create an untitled notebook, file, or directory with a server-selected name."""

        model = await _call(
            _client().create_untitled, parent, model_type=item_type, ext=extension
        )
        return _json(_content_summary(model))

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Rename Jupyter Server Path", destructiveHint=True
        )
    )
    async def rename_server_path(
        path: Annotated[str, Field(description="Existing file or directory path")],
        new_path: Annotated[str, Field(description="New full path")],
    ) -> str:
        """Rename or move a file/directory through the Contents API."""

        return _json(_content_summary(await _call(_client().rename, path, new_path)))

    @mcp.tool(annotations=ToolAnnotations(title="Copy Jupyter Server Path"))
    async def copy_server_path(
        source_path: Annotated[str, Field(description="Existing source path")],
        destination_dir: Annotated[
            str, Field(description="Destination directory, empty for server root")
        ] = "",
    ) -> str:
        """Copy a file or notebook through the Contents API."""

        return _json(
            _content_summary(await _call(_client().copy, source_path, destination_dir))
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Delete Jupyter Server Path", destructiveHint=True
        )
    )
    async def delete_server_path(
        path: Annotated[
            str, Field(description="Exact file or directory path to delete")
        ],
    ) -> str:
        """Delete an exact path from the writable Jupyter server filesystem."""

        try:
            await _call(_client().delete, path)
            return f"Deleted '{path}' through the Contents API"
        except JupyterAPIError as error:
            if error.status != 400 or "send2trash failed" not in error.message.lower():
                raise
        output = await _delete_via_kernel(path)
        return f"Deleted '{path}' through the validated kernel fallback.\n{output}"

    @mcp.tool(
        annotations=ToolAnnotations(title="List File Checkpoints", readOnlyHint=True)
    )
    async def list_server_checkpoints(
        path: Annotated[str, Field(description="File or notebook path")],
    ) -> str:
        """List checkpoints available for a file or notebook."""

        return _json(await _call(_client().list_checkpoints, path))

    @mcp.tool(annotations=ToolAnnotations(title="Create File Checkpoint"))
    async def create_server_checkpoint(
        path: Annotated[str, Field(description="File or notebook path")],
    ) -> str:
        """Create a recoverable checkpoint of the current server-side content."""

        return _json(await _call(_client().create_checkpoint, path))

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Restore File Checkpoint", destructiveHint=True
        )
    )
    async def restore_server_checkpoint(
        path: Annotated[str, Field(description="File or notebook path")],
        checkpoint_id: Annotated[str, Field(description="Checkpoint identifier")],
    ) -> str:
        """Restore a file/notebook to a checkpoint."""

        await _call(_client().restore_checkpoint, path, checkpoint_id)
        return f"Restored checkpoint '{checkpoint_id}' for '{path}'"

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Delete File Checkpoint", destructiveHint=True
        )
    )
    async def delete_server_checkpoint(
        path: Annotated[str, Field(description="File or notebook path")],
        checkpoint_id: Annotated[str, Field(description="Checkpoint identifier")],
    ) -> str:
        """Delete a checkpoint."""

        await _call(_client().delete_checkpoint, path, checkpoint_id)
        return f"Deleted checkpoint '{checkpoint_id}' for '{path}'"

    @mcp.tool(
        annotations=ToolAnnotations(title="Trust Server Notebook", destructiveHint=True)
    )
    async def trust_server_notebook(
        path: Annotated[str, Field(description="Notebook path")],
    ) -> str:
        """Sign a notebook as trusted using the Jupyter server's notebook signer."""

        await _call(_client().trust_notebook, path)
        return f"Trusted '{path}'"

    @mcp.tool(
        annotations=ToolAnnotations(
            title="List Kernel Specifications", readOnlyHint=True
        )
    )
    async def list_kernel_specs() -> str:
        """List available kernel specs and the server default."""

        return _json(await _call(_client().list_kernel_specs))

    @mcp.tool(annotations=ToolAnnotations(title="Get Kernel", readOnlyHint=True))
    async def get_kernel(
        kernel_id: Annotated[str, Field(description="Kernel UUID")],
    ) -> str:
        """Get one running kernel's state."""

        return _json(await _call(_client().get_kernel, kernel_id))

    @mcp.tool(annotations=ToolAnnotations(title="Start Kernel", destructiveHint=True))
    async def start_server_kernel(
        kernel_name: Annotated[str, Field(description="Kernel spec name")] = "python3",
        path: Annotated[
            str, Field(description="Working directory under server root")
        ] = "",
    ) -> str:
        """Start a raw kernel through the Jupyter REST API."""

        return _json(await _call(_client().start_kernel, kernel_name, path))

    @mcp.tool(
        annotations=ToolAnnotations(title="Interrupt Kernel", destructiveHint=True)
    )
    async def interrupt_server_kernel(
        kernel_id: Annotated[str, Field(description="Kernel UUID")],
    ) -> str:
        """Interrupt a running kernel without restarting it."""

        await _call(_client().kernel_action, kernel_id, "interrupt")
        return f"Interrupted kernel '{kernel_id}'"

    @mcp.tool(annotations=ToolAnnotations(title="Restart Kernel", destructiveHint=True))
    async def restart_server_kernel(
        kernel_id: Annotated[str, Field(description="Kernel UUID")],
    ) -> str:
        """Restart a raw kernel by UUID."""

        result = await _call(_client().kernel_action, kernel_id, "restart")
        return _json(result or {"kernel_id": kernel_id, "status": "restarted"})

    @mcp.tool(
        annotations=ToolAnnotations(title="Shutdown Kernel", destructiveHint=True)
    )
    async def shutdown_server_kernel(
        kernel_id: Annotated[str, Field(description="Exact kernel UUID")],
    ) -> str:
        """Permanently stop a raw kernel."""

        await _call(_client().shutdown_kernel, kernel_id)
        return f"Shut down kernel '{kernel_id}'"

    @mcp.tool(annotations=ToolAnnotations(title="List Sessions", readOnlyHint=True))
    async def list_server_sessions() -> str:
        """List notebook/console sessions and their associated kernels."""

        return _json(await _call(_client().list_sessions))

    @mcp.tool(annotations=ToolAnnotations(title="Create Session", destructiveHint=True))
    async def create_server_session(
        path: Annotated[str, Field(description="Document path")],
        name: Annotated[str, Field(description="Session name")],
        kernel_name: Annotated[str, Field(description="Kernel spec name")] = "python3",
        session_type: Annotated[
            str, Field(description="Usually notebook or console")
        ] = "notebook",
    ) -> str:
        """Create a Jupyter session and associated kernel."""

        result = await _call(
            _client().create_session,
            path=path,
            name=name,
            kernel_name=kernel_name,
            session_type=session_type,
        )
        return _json(result)

    @mcp.tool(annotations=ToolAnnotations(title="Update Session", destructiveHint=True))
    async def update_server_session(
        session_id: Annotated[str, Field(description="Session UUID")],
        path: Annotated[str | None, Field(description="New document path")] = None,
        name: Annotated[str | None, Field(description="New session name")] = None,
    ) -> str:
        """Rename or rebind a Jupyter session without exposing arbitrary PATCH fields."""

        changes = {
            key: value for key, value in {"path": path, "name": name}.items() if value
        }
        if not changes:
            raise ValueError("At least one of path or name is required")
        return _json(await _call(_client().update_session, session_id, changes))

    @mcp.tool(
        annotations=ToolAnnotations(title="Shutdown Session", destructiveHint=True)
    )
    async def shutdown_server_session(
        session_id: Annotated[str, Field(description="Exact session UUID")],
    ) -> str:
        """Delete a session and stop its associated kernel."""

        await _call(_client().shutdown_session, session_id)
        return f"Shut down session '{session_id}'"

    @mcp.tool(annotations=ToolAnnotations(title="List Terminals", readOnlyHint=True))
    async def list_server_terminals() -> str:
        """List terminals running on the Jupyter server."""

        return _json(await _call(_client().list_terminals))

    @mcp.tool(
        annotations=ToolAnnotations(title="Create Terminal", destructiveHint=True)
    )
    async def create_server_terminal(
        cwd: Annotated[
            str | None, Field(description="Optional server-side working directory")
        ] = None,
    ) -> str:
        """Create a Jupyter terminal. Use execute_code for non-interactive shell commands."""

        return _json(await _call(_client().create_terminal, cwd))

    @mcp.tool(
        annotations=ToolAnnotations(title="Shutdown Terminal", destructiveHint=True)
    )
    async def shutdown_server_terminal(
        terminal_name: Annotated[str, Field(description="Exact terminal name")],
    ) -> str:
        """Stop and delete a Jupyter terminal."""

        await _call(_client().shutdown_terminal, terminal_name)
        return f"Shut down terminal '{terminal_name}'"

    @mcp.tool(
        annotations=ToolAnnotations(title="List Nbconvert Formats", readOnlyHint=True)
    )
    async def list_nbconvert_formats() -> str:
        """List export formats supported by this Jupyter server."""

        return _json(await _call(_client().nbconvert_formats))

    @mcp.tool(
        annotations=ToolAnnotations(title="Export Notebook", destructiveHint=True)
    )
    async def export_server_notebook(
        path: Annotated[str, Field(description="Source .ipynb path")],
        export_format: Annotated[
            str, Field(description="Format returned by list_nbconvert_formats")
        ],
        destination_path: Annotated[
            str | None,
            Field(
                description="Destination server path; generated beside source when omitted"
            ),
        ] = None,
    ) -> str:
        """Convert a notebook and save the exported artifact on the Jupyter server."""

        client = _client()
        data, mime = await _call(client.export_notebook, path, export_format)
        if not destination_path:
            source = PurePosixPath(path)
            suffix = {
                "python": ".py",
                "script": ".txt",
                "notebook": ".ipynb",
                "html": ".html",
                "markdown": ".md",
                "pdf": ".pdf",
                "slides": ".slides.html",
            }.get(export_format, f".{export_format}")
            destination_path = str(source.with_suffix(suffix))
        is_text = mime.startswith("text/") or export_format in {
            "python",
            "script",
            "html",
            "markdown",
            "slides",
        }
        if is_text:
            content = data.decode("utf-8")
            encoding = "text"
        else:
            content = b64encode(data).decode("ascii")
            encoding = "base64"
        saved = await _call(
            client.save_content,
            destination_path,
            content=content,
            model_type="file",
            format=encoding,
        )
        result = _content_summary(saved)
        result.update(
            {"source": path, "export_format": export_format, "bytes": len(data)}
        )
        return _json(result)

    @mcp.tool(
        annotations=ToolAnnotations(title="List JupyterLab Settings", readOnlyHint=True)
    )
    async def list_lab_settings(
        ids_only: Annotated[
            bool,
            Field(description="Return only plugin/schema IDs instead of full schemas"),
        ] = True,
    ) -> str:
        """List JupyterLab settings schemas exposed by the Kaggle server."""

        return _json(await _call(_client().list_lab_settings, ids_only=ids_only))

    @mcp.tool(
        annotations=ToolAnnotations(title="Get JupyterLab Setting", readOnlyHint=True)
    )
    async def get_lab_setting(
        schema_name: Annotated[
            str, Field(description="Plugin/schema ID returned by list_lab_settings")
        ],
    ) -> str:
        """Get one JupyterLab settings schema and its current user values."""

        return _json(await _call(_client().get_lab_setting, schema_name))

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Update JupyterLab Setting", destructiveHint=True
        )
    )
    async def update_lab_setting(
        schema_name: Annotated[str, Field(description="Plugin/schema ID")],
        raw_json: Annotated[
            str, Field(description="Complete JSON text for the plugin's user settings")
        ],
    ) -> str:
        """Validate and replace one plugin's JupyterLab user settings."""

        json.loads(raw_json)
        await _call(_client().update_lab_setting, schema_name, raw_json)
        return f"Updated JupyterLab setting '{schema_name}'"

    @mcp.tool(
        annotations=ToolAnnotations(
            title="List JupyterLab Workspaces", readOnlyHint=True
        )
    )
    async def list_lab_workspaces() -> str:
        """List saved JupyterLab workspace layouts."""

        return _json(await _call(_client().list_lab_workspaces))

    @mcp.tool(
        annotations=ToolAnnotations(title="Get JupyterLab Workspace", readOnlyHint=True)
    )
    async def get_lab_workspace(
        workspace_name: Annotated[str, Field(description="Workspace name or ID")],
    ) -> str:
        """Get a saved JupyterLab workspace layout."""

        return _json(await _call(_client().get_lab_workspace, workspace_name))

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Save JupyterLab Workspace", destructiveHint=True
        )
    )
    async def save_lab_workspace(
        workspace_name: Annotated[str, Field(description="Workspace name")],
        workspace_json: Annotated[
            str,
            Field(description="Complete workspace JSON including metadata.id and data"),
        ],
    ) -> str:
        """Create or replace a JupyterLab workspace layout."""

        workspace = json.loads(workspace_json)
        if not isinstance(workspace, dict):
            raise TypeError("workspace_json must decode to an object")
        expected_id = "/" + workspace_name.strip("/")
        metadata = workspace.setdefault("metadata", {})
        actual_id = str(metadata.get("id", expected_id))
        if not actual_id.startswith("/"):
            actual_id = "/" + actual_id
        if actual_id != expected_id:
            raise ValueError(f"metadata.id must equal '{expected_id}'")
        metadata["id"] = expected_id
        workspace.setdefault("data", {})
        await _call(_client().save_lab_workspace, workspace_name, workspace)
        return f"Saved JupyterLab workspace '{workspace_name}'"

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Delete JupyterLab Workspace", destructiveHint=True
        )
    )
    async def delete_lab_workspace(
        workspace_name: Annotated[str, Field(description="Exact workspace name")],
    ) -> str:
        """Delete a saved JupyterLab workspace layout."""

        await _call(_client().delete_lab_workspace, workspace_name)
        return f"Deleted JupyterLab workspace '{workspace_name}'"

    mcp._kaggle_extra_tools_installed = True
