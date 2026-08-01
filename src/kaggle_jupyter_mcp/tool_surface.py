"""Expose a small, unambiguous Kaggle-focused MCP tool surface."""

from __future__ import annotations

from typing import Any


TOOL_DESCRIPTIONS = {
    "connect_to_jupyter": (
        "Switch the running MCP process to a new Jupyter/Kaggle server URL without "
        "restarting Codex. Use this only when the VS Code-compatible URL or token changes. "
        "After switching, reconnect local notebooks because old kernel IDs belong to the "
        "previous server. Treat signed URLs as secrets."
    ),
    "get_server_status": (
        "Check whether the currently configured Jupyter server is reachable and report "
        "aggregate kernel activity. Use after connect_to_jupyter; use get_kernel for one "
        "specific kernel."
    ),
    "list_kernels": (
        "List raw kernels on the current Jupyter server with UUID and execution state. "
        "Use to discover a kernel_id; this does not open or execute a notebook."
    ),
    "get_kernel": (
        "Read the current state of one raw kernel UUID. Use for monitoring; it never "
        "executes code or changes the kernel."
    ),
    "start_server_kernel": (
        "Create an unattached raw kernel on Kaggle and return its UUID. Prefer "
        "connect_local_notebook when the goal is to run a local .ipynb."
    ),
    "interrupt_server_kernel": (
        "Interrupt the code currently running on one kernel while preserving its Python "
        "state. Use for a stuck execution, not for switching Jupyter URLs."
    ),
    "restart_server_kernel": (
        "Restart one raw kernel and erase all variables/model state in it. Use only when a "
        "clean kernel is required."
    ),
    "shutdown_server_kernel": (
        "Permanently terminate one raw kernel UUID. Use close_local_notebook for a kernel "
        "owned by a local-notebook binding."
    ),
    "execute_python": (
        "Run ad-hoc Python or IPython on a Kaggle kernel and return stdout, errors, rich "
        "output, and images. Never pass Bash here: use execute_shell. For a cell from a "
        "local .ipynb, use execute_local_notebook_cell so outputs are saved locally."
    ),
    "execute_shell": (
        "Run one ordinary Bash command inside /kaggle/working or /kaggle/temp and return "
        "stdout, stderr, and exit code. Do not pass Python source; use execute_python."
    ),
    "connect_local_notebook": (
        "Bind an existing local .ipynb to a raw Kaggle kernel without uploading or creating "
        "a server-side notebook. Call once after opening a notebook or changing Jupyter URL."
    ),
    "execute_local_notebook_cell": (
        "Execute one zero-based code cell from a connected local .ipynb on its Kaggle "
        "kernel, return the outputs, and save outputs/execution_count back into the local file."
    ),
    "start_local_notebook_run": (
        "Start selected code cells from a connected local .ipynb as a background run. "
        "Returns run_id immediately; poll get_local_notebook_run_status instead of waiting."
    ),
    "get_local_notebook_run_status": (
        "Poll a background local-notebook run by run_id. Returns current cell, percentage, "
        "per-cell output previews, final state, and error."
    ),
    "cancel_local_notebook_run": (
        "Interrupt the bound Kaggle kernel and cancel one active local-notebook run. "
        "Already completed/failed/cancelled runs are left unchanged."
    ),
    "close_local_notebook": (
        "Close one local-notebook binding and optionally shut down its remote kernel. "
        "This never deletes the local .ipynb."
    ),
    "list_files": (
        "List remote Jupyter/Kaggle files only. Use paths relative to /kaggle/working; this "
        "cannot inspect the local workspace."
    ),
    "read_server_file": (
        "Read a remote regular file through Jupyter Contents API. Do not use for local files "
        "or for executing notebook cells."
    ),
    "write_server_file": (
        "Create or completely replace a remote regular text/binary file. Do not use to "
        "upload a local file or create a runnable .ipynb; use monitored transfer or local "
        "notebook tools instead."
    ),
    "edit_server_text": (
        "Perform conflict-checked literal replacement in a remote text file. Use only for "
        "small targeted edits; this does not edit local files or notebook cells."
    ),
    "rename_server_path": (
        "Rename or move one exact remote Jupyter path through Contents API. This does not "
        "operate on local workspace paths."
    ),
    "copy_server_path": (
        "Copy one existing remote Jupyter file/notebook into a remote destination directory. "
        "This is server-to-server copy, not local file transfer."
    ),
    "delete_server_path": (
        "Delete one exact remote path under the writable Jupyter root. Resolve/list the "
        "target first; this never deletes local files."
    ),
    "start_upload_local_file_to_jupyter": (
        "Start a resumable local-to-Kaggle file upload in the background and immediately "
        "return transfer_id. Always poll get_file_transfer_status; do not use server copy."
    ),
    "start_download_jupyter_file_to_local": (
        "Start a resumable Kaggle-to-local file download in the background and immediately "
        "return transfer_id. Always poll get_file_transfer_status."
    ),
    "get_file_transfer_status": (
        "Poll one transfer_id for stage, bytes, percent, speed, ETA, resumable partial path, "
        "result, or error. This never starts a transfer."
    ),
    "list_file_transfers": (
        "List recent background upload/download jobs known to this MCP process. Job history "
        "does not survive an MCP process restart."
    ),
    "cancel_file_transfer": (
        "Cancel one active transfer_id while retaining its compatible partial file for a "
        "later resume. This does not delete the source or final destination."
    ),
}

KEEP_TOOLS = frozenset(TOOL_DESCRIPTIONS)


def curate_tool_surface(mcp: Any) -> None:
    """Remove low-value/ambiguous tools and replace descriptions on retained tools."""

    if getattr(mcp, "_kaggle_tool_surface_curated", False):
        return
    registered = list(mcp._tool_manager.list_tools())
    for tool in registered:
        if tool.name not in KEEP_TOOLS:
            mcp.remove_tool(tool.name)
    remaining = {tool.name: tool for tool in mcp._tool_manager.list_tools()}
    missing = KEEP_TOOLS - remaining.keys()
    if missing:
        raise RuntimeError(f"Expected MCP tools were not registered: {sorted(missing)}")
    for name, description in TOOL_DESCRIPTIONS.items():
        remaining[name].description = description
    mcp._kaggle_tool_surface_curated = True
