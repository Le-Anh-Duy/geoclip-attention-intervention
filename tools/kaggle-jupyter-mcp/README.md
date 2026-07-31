# Kaggle Jupyter MCP compatibility layer

This package extends `jupyter-mcp-server==1.2.0` for Kaggle's managed Jupyter
proxy. Kaggle exposes kernels, sessions, terminals, nbconvert, and the Contents
API, but it does not expose Jupyter's real-time collaboration endpoint. The
upstream standalone server normally requires that endpoint for cell operations.

## What the compatibility layer changes

- keeps upstream RTC/WebSocket behavior when `/api/collaboration/session` is
  available;
- falls back to `/api/contents` for real nbformat `.ipynb` documents;
- preserves cell execution through the existing Kaggle kernel WebSocket;
- creates a checkpoint before notebook/file replacement when supported;
- rejects stale writes when a notebook changed after it was read;
- adds REST tools for files, text editing, checkpoints, kernels, sessions,
  terminals, server status/capability probing, kernelspecs, and nbconvert;
- adds `execute_shell` for ordinary Bash commands with stdout, stderr, and exit
  code returned to MCP;
- adds opt-in visible execution through `execute_code_in_notebook`, which
  creates/connects `mcp_execution.ipynb`, saves the cell and its output, and
  returns that output to MCP;
- transfers files in both directions with chunking and SHA-256 verification.
  Upload chunks are written directly by the kernel to `/kaggle/temp` before the
  verified file is moved to `/kaggle/working` or its requested temp path.

## Execution tools

- `execute_code`: upstream direct Python/IPython kernel execution. It is not
  persisted as a notebook cell.
- `execute_shell`: ordinary Bash such as `pwd`, `ls -la`, or a pipeline. It is
  not persisted as a notebook cell.
- `execute_code_in_notebook`: visible, persisted execution. It accepts optional
  `notebook_path` and `notebook_name`; the default is `mcp_execution.ipynb`.
- `insert_execute_code_cell`: visible execution at an explicit cell index in an
  already active notebook.

## File transfer

- `upload_local_file_to_jupyter`: local file to a relative
  `/kaggle/working` path or an absolute `/kaggle/working/...` or
  `/kaggle/temp/...` path.
- `download_jupyter_file_to_local`: the reverse direction with the same remote
  path rules.

Both tools log their verification cells in `mcp_file_transfer.ipynb`. Local
access is restricted to `KAGGLE_JUPYTER_LOCAL_ROOTS`, separated with the host
OS path separator. If unset, only the MCP process working directory is allowed.
Existing destination files are preserved unless `overwrite=true` is explicit.

The package pins the upstream version so an update cannot silently break the
compatibility patch.

## Important Kaggle limitation

Kaggle exposes the active browser notebook as:

```text
.virtual_documents/__notebook_source__.ipynb
```

Despite its name, that resource is a plain concatenated Python source file, not
nbformat JSON, and contains no cell boundaries. It can be read or edited with
`read_server_file` and `edit_server_text`, but it cannot truthfully support
index-based notebook cell operations. Kaggle's browser may also regenerate the
virtual file. Use a real `.ipynb` stored under `/kaggle/working` when durable
cell-level edits are required.

## Codex configuration

Replace the command and arguments of the existing MCP entry while keeping its
secret URL/token values:

```toml
[mcp_servers.kaggle_jupyter]
command = "uv"
args = [
  "run",
  "--project",
  "E:\\GHuy\\Study\\Thesis\\geoclip-attention-intervention\\tools\\kaggle-jupyter-mcp",
  "kaggle-jupyter-mcp",
]
startup_timeout_sec = 60
tool_timeout_sec = 900
enabled = true

[mcp_servers.kaggle_jupyter.env]
JUPYTER_URL = "<existing signed Kaggle proxy URL>"
JUPYTER_TOKEN = "<existing token>"
ALLOW_IMG_OUTPUT = "true"
KAGGLE_JUPYTER_AUTO_CHECKPOINT = "true"
KAGGLE_JUPYTER_LOCAL_ROOTS = "E:\\GHuy\\Study\\Thesis\\geoclip-attention-intervention"
```

Restart Codex after changing the MCP command. The signed Kaggle proxy URL is
session-scoped and still needs to be refreshed when Kaggle creates a new
interactive runtime.

## Development

```powershell
uv sync --extra test
uv run pytest -q
```

The integration smoke test reads `JUPYTER_URL` and `JUPYTER_TOKEN` from the
environment and uses a temporary notebook:

```powershell
uv run python scripts/smoke_test.py
```
