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
  terminals, server status/capability probing, kernelspecs, and nbconvert.

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
