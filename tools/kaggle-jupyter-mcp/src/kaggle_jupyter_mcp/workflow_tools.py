"""Register opinionated notebook execution and local file-transfer tools."""

from __future__ import annotations

from typing import Annotated

from mcp.types import ImageContent, ToolAnnotations
from pydantic import Field

from kaggle_jupyter_mcp.file_transfer import (
    DEFAULT_CHUNK_MIB,
    FileTransferService,
    validate_chunk_size,
)
from kaggle_jupyter_mcp.visible_execution import VisibleNotebookRunner


def register_workflow_tools(mcp) -> VisibleNotebookRunner:
    """Replace ephemeral execute_code and add bidirectional transfer tools."""

    if getattr(mcp, "_kaggle_workflow_tools_installed", False):
        return mcp._kaggle_visible_notebook_runner

    from jupyter_mcp_server import server

    runner = VisibleNotebookRunner(server.execute_code)
    transfer = FileTransferService(runner)

    # The upstream implementation executes against a kernel but does not save a
    # cell. Preserve its callable for transfer internals, and expose a visible
    # implementation under the same public tool name.
    mcp.remove_tool("execute_code")

    @mcp.tool(
        name="execute_code",
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
            int, Field(description="Kernel transfer chunk size in MiB", ge=1, le=32)
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
            int, Field(description="Kernel transfer chunk size in MiB", ge=1, le=32)
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

    mcp._kaggle_visible_notebook_runner = runner
    mcp._kaggle_workflow_tools_installed = True
    return runner
