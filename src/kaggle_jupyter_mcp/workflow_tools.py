"""Register opinionated notebook execution and local file-transfer tools."""

from __future__ import annotations

import json
from typing import Annotated, Any

from mcp.types import ImageContent, ToolAnnotations
from pydantic import Field

from kaggle_jupyter_mcp.file_transfer import (
    DEFAULT_CHUNK_MIB,
    FileTransferService,
    normalize_server_file_path,
    validate_chunk_size,
)
from kaggle_jupyter_mcp.local_notebook import LocalNotebookManager
from kaggle_jupyter_mcp.transfer_jobs import FileTransferJobManager
from kaggle_jupyter_mcp.visible_execution import VisibleNotebookRunner


def build_shell_code(command: str, cwd: str) -> str:
    """Build the small Python bridge used to run an ordinary Bash command."""

    normalized_cwd = normalize_server_file_path(cwd)
    return f"""from pathlib import Path
import subprocess, sys
working = Path('/kaggle/working').resolve()
temp_root = Path('/kaggle/temp')
temp_root.mkdir(parents=True, exist_ok=True)
temp_root = temp_root.resolve()
raw_cwd = {json.dumps(normalized_cwd)}
shell_cwd = Path(raw_cwd)
shell_cwd = shell_cwd.resolve() if shell_cwd.is_absolute() else (working / shell_cwd).resolve()
if not any(shell_cwd == root or root in shell_cwd.parents for root in (working, temp_root)):
    raise ValueError('cwd must remain under /kaggle/working or /kaggle/temp')
if not shell_cwd.is_dir():
    raise NotADirectoryError(str(shell_cwd))
completed = subprocess.run(
    ['bash', '-lc', {json.dumps(command)}],
    cwd=shell_cwd,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if completed.stdout:
    print(completed.stdout, end='' if completed.stdout.endswith('\\n') else '\\n')
if completed.stderr:
    print(completed.stderr, end='' if completed.stderr.endswith('\\n') else '\\n', file=sys.stderr)
print(f'[MCP shell exit code: {{completed.returncode}}]')
"""


def register_workflow_tools(mcp) -> VisibleNotebookRunner:
    """Add explicit shell, visible-notebook, and bidirectional transfer tools."""

    if getattr(mcp, "_kaggle_workflow_tools_installed", False):
        return mcp._kaggle_visible_notebook_runner

    from jupyter_mcp_server import server

    runner = VisibleNotebookRunner(server.execute_code)
    local_notebooks = LocalNotebookManager(runner.original_execute_code)
    transfer = FileTransferService(runner)
    transfer_jobs = FileTransferJobManager(transfer)

    @mcp.tool(
        name="execute_python",
        title="Execute Python on a Kaggle Kernel",
        annotations=ToolAnnotations(
            title="Execute Python on a Kaggle Kernel", destructiveHint=True
        ),
        structured_output=False,
    )
    async def execute_python(
        code: Annotated[
            str,
            Field(
                description="Python/IPython source. Do not pass Bash; use execute_shell for Bash."
            ),
        ],
        kernel_id: Annotated[
            str | None,
            Field(
                description="Existing raw kernel UUID. Omit only for a disposable quick command."
            ),
        ] = None,
        timeout: Annotated[
            int, Field(description="Maximum execution seconds; 0 uses server default", ge=0)
        ] = 0,
    ) -> list[Any]:
        """Run Python immediately and return stdout, errors, rich output, and images."""

        outputs = await runner.original_execute_code(
            code=code, timeout=timeout, kernel_id=kernel_id
        )
        return outputs if isinstance(outputs, list) else [outputs]

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Connect Local Notebook to Kaggle Kernel", destructiveHint=True
        )
    )
    async def connect_local_notebook(
        local_path: Annotated[
            str,
            Field(
                description="Existing local .ipynb under KAGGLE_JUPYTER_LOCAL_ROOTS"
            ),
        ],
        kernel_id: Annotated[
            str | None,
            Field(description="Existing Kaggle kernel UUID; omit to create a raw kernel"),
        ] = None,
        kernel_name: Annotated[
            str, Field(description="Kernel spec used only when creating a kernel")
        ] = "python3",
        remote_working_dir: Annotated[
            str, Field(description="Remote kernel working directory")
        ] = "/kaggle/working",
        reuse_existing: Annotated[
            bool,
            Field(description="Reuse this local notebook's live binding on the current server"),
        ] = True,
    ) -> dict[str, Any]:
        """Bind a local notebook file to one raw Kaggle kernel; no server notebook is created."""

        return await local_notebooks.connect(
            local_path,
            kernel_id=kernel_id,
            kernel_name=kernel_name,
            remote_working_dir=remote_working_dir,
            reuse_existing=reuse_existing,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Execute Local Notebook Cell", destructiveHint=True
        ),
        structured_output=False,
    )
    async def execute_local_notebook_cell(
        local_path: Annotated[
            str, Field(description="Connected local .ipynb path")
        ],
        cell_index: Annotated[
            int, Field(description="Zero-based code-cell index in the local notebook", ge=0)
        ],
        timeout: Annotated[
            int, Field(description="Maximum execution seconds; 0 uses server default", ge=0)
        ] = 0,
    ) -> list[Any]:
        """Run one local notebook cell remotely, return output, and save it into the local .ipynb."""

        return await local_notebooks.execute_cell(
            local_path, cell_index, timeout=timeout
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Start Monitored Local Notebook Run", destructiveHint=True
        )
    )
    async def start_local_notebook_run(
        local_path: Annotated[
            str, Field(description="Connected local .ipynb path")
        ],
        start_cell: Annotated[
            int, Field(description="Inclusive first cell index", ge=0)
        ] = 0,
        stop_cell: Annotated[
            int | None, Field(description="Exclusive last cell index; omit for notebook end", ge=0)
        ] = None,
        timeout_per_cell: Annotated[
            int, Field(description="Maximum seconds for each code cell", ge=0)
        ] = 300,
        stop_on_error: Annotated[
            bool, Field(description="Stop at the first failed code cell")
        ] = True,
    ) -> dict[str, Any]:
        """Start all selected code cells in the background and return a run_id immediately."""

        return await local_notebooks.start_run(
            local_path,
            start_cell=start_cell,
            stop_cell=stop_cell,
            timeout_per_cell=timeout_per_cell,
            stop_on_error=stop_on_error,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Local Notebook Run Status", readOnlyHint=True
        )
    )
    async def get_local_notebook_run_status(
        run_id: Annotated[str, Field(description="ID returned by start_local_notebook_run")],
    ) -> dict[str, Any]:
        """Return current cell, progress, output previews, final state, and any error."""

        return local_notebooks.run_status(run_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Cancel Local Notebook Run", destructiveHint=True
        )
    )
    async def cancel_local_notebook_run(
        run_id: Annotated[str, Field(description="ID returned by start_local_notebook_run")],
    ) -> dict[str, Any]:
        """Interrupt the remote kernel and cancel one running local-notebook job."""

        return await local_notebooks.cancel_run(run_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Close Local Notebook Kernel", destructiveHint=True
        )
    )
    async def close_local_notebook(
        local_path: Annotated[str, Field(description="Connected local .ipynb path")],
        shutdown_kernel: Annotated[
            bool, Field(description="Also terminate the bound Kaggle kernel")
        ] = True,
    ) -> dict[str, Any]:
        """Remove a local-notebook binding and optionally shut down its Kaggle kernel."""

        return await local_notebooks.close(
            local_path, shutdown_kernel=shutdown_kernel
        )

    @mcp.tool(
        name="execute_code_in_notebook",
        title="Execute Code in a Visible Notebook",
        annotations=ToolAnnotations(title="Execute Code in a Visible Notebook", destructiveHint=True),
        structured_output=False,
    )
    async def execute_code_in_visible_notebook(
        code: Annotated[str, Field(description="Python or IPython code to execute")],
        timeout: Annotated[
            int, Field(description="Maximum execution seconds (0 uses server default)", ge=0)
        ] = 0,
        notebook_path: Annotated[
            str,
            Field(
                description="Real .ipynb path under /kaggle/working; created if missing"
            ),
        ] = "mcp_execution.ipynb",
        notebook_name: Annotated[
            str | None, Field(description="Optional stable notebook connection name")
        ] = None,
    ) -> list[str | ImageContent]:
        """Append code to a real notebook, execute its connected kernel, save outputs, and return them."""

        result = await runner.run(
            code,
            timeout=timeout,
            notebook_path=notebook_path,
            notebook_name=notebook_name,
        )
        return result.as_mcp_content()

    @mcp.tool(
        name="execute_shell",
        title="Execute Bash Command",
        annotations=ToolAnnotations(title="Execute Bash Command", destructiveHint=True),
        structured_output=False,
    )
    async def execute_shell(
        command: Annotated[
            str, Field(description="Ordinary Bash command, for example: ls -la")
        ],
        timeout: Annotated[
            int, Field(description="Maximum execution seconds (0 uses server default)", ge=0)
        ] = 0,
        cwd: Annotated[
            str,
            Field(
                description="Working directory under /kaggle/working or /kaggle/temp"
            ),
        ] = "/kaggle/working",
    ) -> list[str | ImageContent]:
        """Run an ordinary Bash command and return stdout, stderr, and its exit code."""

        outputs = await runner.original_execute_code(
            code=build_shell_code(command, cwd), timeout=timeout, kernel_id=None
        )
        return outputs if isinstance(outputs, list) else [outputs]

    @mcp.tool(
        annotations=ToolAnnotations(title="Upload Local File to Kaggle", destructiveHint=True),
        structured_output=False,
    )
    async def upload_local_file_to_jupyter(
        local_path: Annotated[
            str, Field(description="Local source file under KAGGLE_JUPYTER_LOCAL_ROOTS")
        ],
        server_path: Annotated[
            str,
            Field(
                description="Destination relative to /kaggle/working, or absolute under /kaggle/working or /kaggle/temp"
            ),
        ],
        overwrite: Annotated[
            bool, Field(description="Replace an existing destination file")
        ] = False,
        chunk_size_mib: Annotated[
            int, Field(description="Kernel transfer chunk size in MiB", ge=1, le=8)
        ] = DEFAULT_CHUNK_MIB,
        timeout: Annotated[int, Field(description="Per-kernel-operation timeout", ge=0)] = 120,
        notebook_path: Annotated[
            str, Field(description="Visible transfer-log notebook")
        ] = "mcp_file_transfer.ipynb",
        notebook_name: Annotated[
            str | None, Field(description="Optional stable notebook connection name")
        ] = None,
    ) -> list[str | ImageContent]:
        """Upload a local file through the notebook kernel, staging bytes in /kaggle/temp."""

        return await transfer.upload(
            local_path=local_path,
            server_path=server_path,
            notebook_path=notebook_path,
            notebook_name=notebook_name,
            overwrite=overwrite,
            chunk_size_mib=validate_chunk_size(chunk_size_mib),
            timeout=timeout,
        )

    @mcp.tool(
        annotations=ToolAnnotations(title="Download Kaggle File to Local", destructiveHint=True),
        structured_output=False,
    )
    async def download_jupyter_file_to_local(
        server_path: Annotated[
            str,
            Field(
                description="Source relative to /kaggle/working, or absolute under /kaggle/working or /kaggle/temp"
            ),
        ],
        local_path: Annotated[
            str, Field(description="Local destination under KAGGLE_JUPYTER_LOCAL_ROOTS")
        ],
        overwrite: Annotated[
            bool, Field(description="Replace an existing local destination")
        ] = False,
        chunk_size_mib: Annotated[
            int, Field(description="Kernel transfer chunk size in MiB", ge=1, le=8)
        ] = DEFAULT_CHUNK_MIB,
        timeout: Annotated[int, Field(description="Per-kernel-operation timeout", ge=0)] = 120,
        notebook_path: Annotated[
            str, Field(description="Visible transfer-log notebook")
        ] = "mcp_file_transfer.ipynb",
        notebook_name: Annotated[
            str | None, Field(description="Optional stable notebook connection name")
        ] = None,
    ) -> list[str | ImageContent]:
        """Download and verify a Kaggle file through its connected notebook kernel."""

        return await transfer.download(
            server_path=server_path,
            local_path=local_path,
            notebook_path=notebook_path,
            notebook_name=notebook_name,
            overwrite=overwrite,
            chunk_size_mib=validate_chunk_size(chunk_size_mib),
            timeout=timeout,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Start Monitored Upload to Kaggle", destructiveHint=True
        )
    )
    async def start_upload_local_file_to_jupyter(
        local_path: Annotated[
            str, Field(description="Local source file under KAGGLE_JUPYTER_LOCAL_ROOTS")
        ],
        server_path: Annotated[
            str,
            Field(
                description="Destination relative to /kaggle/working, or absolute under /kaggle/working or /kaggle/temp"
            ),
        ],
        overwrite: Annotated[
            bool, Field(description="Replace an existing destination file")
        ] = False,
        resume: Annotated[
            bool,
            Field(description="Resume a compatible partial upload after interruption"),
        ] = True,
        chunk_size_mib: Annotated[
            int, Field(description="Kernel transfer chunk size in MiB", ge=1, le=8)
        ] = DEFAULT_CHUNK_MIB,
        timeout: Annotated[int, Field(description="Per-kernel-operation timeout", ge=0)] = 120,
    ) -> dict:
        """Start an upload job and immediately return a transfer ID for polling."""

        return await transfer_jobs.start_upload(
            local_path=local_path,
            server_path=server_path,
            notebook_path="__mcp_transfer__.ipynb",
            notebook_name="__mcp_transfer__",
            overwrite=overwrite,
            chunk_size_mib=validate_chunk_size(chunk_size_mib),
            timeout=timeout,
            resume=resume,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Start Monitored Download from Kaggle", destructiveHint=True
        )
    )
    async def start_download_jupyter_file_to_local(
        server_path: Annotated[
            str,
            Field(
                description="Source relative to /kaggle/working, or absolute under /kaggle/working or /kaggle/temp"
            ),
        ],
        local_path: Annotated[
            str, Field(description="Local destination under KAGGLE_JUPYTER_LOCAL_ROOTS")
        ],
        overwrite: Annotated[
            bool, Field(description="Replace an existing local destination")
        ] = False,
        resume: Annotated[
            bool,
            Field(description="Resume a compatible local partial download"),
        ] = True,
        chunk_size_mib: Annotated[
            int, Field(description="Kernel transfer chunk size in MiB", ge=1, le=8)
        ] = DEFAULT_CHUNK_MIB,
        timeout: Annotated[int, Field(description="Per-kernel-operation timeout", ge=0)] = 120,
    ) -> dict:
        """Start a download job and immediately return a transfer ID for polling."""

        return await transfer_jobs.start_download(
            server_path=server_path,
            local_path=local_path,
            notebook_path="__mcp_transfer__.ipynb",
            notebook_name="__mcp_transfer__",
            overwrite=overwrite,
            chunk_size_mib=validate_chunk_size(chunk_size_mib),
            timeout=timeout,
            resume=resume,
        )

    @mcp.tool(
        annotations=ToolAnnotations(title="Get File Transfer Status", readOnlyHint=True)
    )
    async def get_file_transfer_status(
        transfer_id: Annotated[str, Field(description="ID returned by a start_* transfer tool")],
    ) -> dict:
        """Return live byte/chunk progress, speed, ETA, stage, result, or error."""

        return transfer_jobs.status(transfer_id)

    @mcp.tool(
        annotations=ToolAnnotations(title="List File Transfers", readOnlyHint=True)
    )
    async def list_file_transfers(
        limit: Annotated[int, Field(description="Maximum recent jobs", ge=1, le=100)] = 20,
    ) -> list[dict]:
        """List recent active and completed background file transfers."""

        return transfer_jobs.list(limit)

    @mcp.tool(
        annotations=ToolAnnotations(title="Cancel File Transfer", destructiveHint=True)
    )
    async def cancel_file_transfer(
        transfer_id: Annotated[str, Field(description="ID returned by a start_* transfer tool")],
    ) -> dict:
        """Cancel a running transfer while retaining its resumable partial file."""

        return await transfer_jobs.cancel(transfer_id)

    mcp._kaggle_visible_notebook_runner = runner
    mcp._kaggle_local_notebooks = local_notebooks
    mcp._kaggle_file_transfer_jobs = transfer_jobs
    mcp._kaggle_workflow_tools_installed = True
    return runner
