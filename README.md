# Kaggle Jupyter MCP

An opinionated MCP layer for controlling a Kaggle Jupyter runtime from a local
workspace. It pins `jupyter-mcp-server==1.2.0`, supports Kaggle's signed proxy
URLs, returns execution output to the client, runs local notebooks on raw Kaggle
kernels, and provides monitored resumable file transfer.

The public tool registry is deliberately curated. Upstream server-notebook cell
tools, synchronous transfer tools, terminals, sessions, checkpoints, nbconvert,
and JupyterLab settings/workspaces are hidden because they are redundant,
unreliable on Kaggle, or too easy to select accidentally.

## Tool contract

### Connection and kernels

| Tool | Use it for |
|---|---|
| `connect_to_jupyter` | Switch to a new VS Code-compatible URL/token without restarting MCP or Codex. Reconnect local notebooks afterward. |
| `get_server_status` | Verify the current server is reachable and see aggregate activity. |
| `list_kernels` | Discover raw kernel UUIDs and states. |
| `get_kernel` | Monitor one kernel without changing it. |
| `start_server_kernel` | Create an unattached raw kernel for ad-hoc work. Prefer `connect_local_notebook` for notebooks. |
| `interrupt_server_kernel` | Stop currently running code but preserve kernel variables. |
| `restart_server_kernel` | Erase variables/model state and restart one kernel. |
| `shutdown_server_kernel` | Permanently terminate one raw kernel. |

### Execution

| Tool | Use it for |
|---|---|
| `execute_python` | Run ad-hoc Python/IPython and receive stdout, errors, rich output, and images. Never pass Bash. |
| `execute_shell` | Run ordinary Bash under `/kaggle/working` or `/kaggle/temp`; returns stdout, stderr, and exit code. Never pass Python source. |

### Local notebook on Kaggle kernel

The `.ipynb` stays local. No notebook is uploaded to Kaggle.

| Tool | Use it for |
|---|---|
| `connect_local_notebook` | Bind an existing local `.ipynb` to an existing or newly created raw Kaggle kernel. |
| `execute_local_notebook_cell` | Execute one zero-based local code cell remotely, return its output, and save output/execution count into the local file. |
| `start_local_notebook_run` | Start all selected code cells in the background and immediately return `run_id`. |
| `get_local_notebook_run_status` | Poll current cell, percentage, output previews, final state, and errors. |
| `cancel_local_notebook_run` | Interrupt the kernel and cancel an active notebook run. |
| `close_local_notebook` | Remove the binding and optionally shut down its kernel; never deletes the local notebook. |

If the local file changes while a cell is running, its outputs are returned but
not written over the newer file.

### Remote files

| Tool | Use it for |
|---|---|
| `list_files` | Explore remote Jupyter/Kaggle files only. |
| `read_server_file` | Read one remote regular file, not a local file or notebook cell. |
| `write_server_file` | Completely replace a remote regular text/binary file. Do not use it to create runnable notebooks. |
| `edit_server_text` | Conflict-checked literal replacement in a remote text file. |
| `rename_server_path` | Rename/move one exact remote path. |
| `copy_server_path` | Copy remote-to-remote; this is not local transfer. |
| `delete_server_path` | Delete one exact remote path after resolving it. |

### Monitored file transfer

| Tool | Use it for |
|---|---|
| `start_upload_local_file_to_jupyter` | Start resumable local-to-Kaggle upload and immediately return `transfer_id`. |
| `start_download_jupyter_file_to_local` | Start resumable Kaggle-to-local download and immediately return `transfer_id`. |
| `get_file_transfer_status` | Poll bytes, percentage, chunks, speed, ETA, partial path, result, or error. |
| `list_file_transfers` | Recover recent transfer IDs in the current MCP process. |
| `cancel_file_transfer` | Cancel while retaining a compatible partial file for resume. |

Local access is restricted to `KAGGLE_JUPYTER_LOCAL_ROOTS`. Remote transfer
paths are restricted to `/kaggle/working` and `/kaggle/temp`. Chunks are capped
at 8 MiB because base64 expansion can exceed Kaggle's WebSocket message limit.
Every final file is verified by byte size and SHA-256 before replacement.

## Why server-notebook tools are hidden

Kaggle exposes the active browser document at
`.virtual_documents/__notebook_source__.ipynb`, but it is concatenated Python
text without trustworthy cell boundaries. Some Kaggle runtimes also return an
empty collaborative notebook model even when a valid `.ipynb` exists through
the Contents API. Consequently `use_notebook`, `execute_cell`,
`execute_code_in_notebook`, and related upstream tools are not part of the
public API. Local notebook tools use a raw kernel and standard nbformat files,
which avoids that ambiguity.

## Configuration

```toml
[mcp_servers.kaggle_jupyter]
command = "uv"
args = [
  "run",
  "--project",
  "C:\\path\\to\\kaggle-jupyter-mcp",
  "kaggle-jupyter-mcp",
]
startup_timeout_sec = 60
tool_timeout_sec = 900
enabled = true

[mcp_servers.kaggle_jupyter.env]
JUPYTER_URL = "<signed Kaggle proxy URL>"
JUPYTER_TOKEN = "<optional separate token>"
ALLOW_IMG_OUTPUT = "true"
KAGGLE_JUPYTER_LOCAL_ROOTS = "C:\\path\\to\\workspace"
```

Changing the configured executable requires restarting Codex. Changing only the
Kaggle URL does not: call `connect_to_jupyter` at runtime. Signed URLs grant
access to the runtime and must not be logged or shared.

## Development

```powershell
uv sync --extra test
uv run pytest -q
ruff check .
```

`scripts/transfer_smoke_test.py` performs upload, progress polling, cancellation,
resume, download, and SHA-256 verification against a live Kaggle runtime.
