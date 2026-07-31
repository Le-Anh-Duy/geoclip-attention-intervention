"""End-to-end smoke test against a configured Jupyter/Kaggle server."""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
from pathlib import Path
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kaggle_jupyter_mcp.rest import JupyterRestClient


def text_content(result) -> str:
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


async def main() -> None:
    if not os.getenv("JUPYTER_URL"):
        raise SystemExit("JUPYTER_URL is required")

    suffix = uuid4().hex[:10]
    notebook_name = f"mcp_smoke_{suffix}"
    notebook_path = f"mcp_smoke_{suffix}.ipynb"
    env = os.environ.copy()
    env.setdefault("KAGGLE_JUPYTER_AUTO_CHECKPOINT", "true")
    # Track the baseline so the test can remove only kernels it created. This is
    # registered with atexit so cleanup also runs after an assertion or MCP crash.
    probe_client = JupyterRestClient(
        base_url=env["JUPYTER_URL"], token=env.get("JUPYTER_TOKEN") or None
    )
    running_kernels = await asyncio.to_thread(probe_client.list_kernels)
    baseline_kernel_ids = {item["id"] for item in running_kernels}

    def cleanup_new_kernels() -> None:
        for kernel in probe_client.list_kernels():
            if kernel["id"] not in baseline_kernel_ids:
                probe_client.shutdown_kernel(kernel["id"])

    atexit.register(cleanup_new_kernels)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaggle_jupyter_mcp"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    # Upstream debug logging includes the signed proxy URL. Keep subprocess logs
    # out of test output so credentials never appear in CI or chat transcripts.
    errlog_path = os.getenv("MCP_SMOKE_ERRLOG", os.devnull)
    with open(errlog_path, "w", encoding="utf-8") as errlog:  # noqa: ASYNC230
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                required = {
                    "use_notebook",
                    "insert_execute_code_cell",
                    "read_cell",
                    "probe_server_capabilities",
                    "delete_server_path",
                    "list_lab_settings",
                    "list_lab_workspaces",
                    "list_nbconvert_formats",
                }
                missing = sorted(required - tool_names)
                if missing:
                    raise RuntimeError(f"Missing tools: {missing}")

                capabilities = await session.call_tool("probe_server_capabilities", {})
                print("CAPABILITIES")
                print(text_content(capabilities))

                for tool_name, arguments in (
                    ("list_lab_settings", {"ids_only": True}),
                    ("list_lab_workspaces", {}),
                    ("list_nbconvert_formats", {}),
                ):
                    result = await session.call_tool(tool_name, arguments)
                    if result.isError:
                        raise RuntimeError(f"{tool_name}: {text_content(result)}")
                    print(f"{tool_name.upper()}_OK")

                created = False
                activated = False
                try:
                    result = await session.call_tool(
                        "use_notebook",
                        {
                            "notebook_name": notebook_name,
                            "notebook_path": notebook_path,
                            "mode": "create",
                        },
                    )
                    if result.isError:
                        raise RuntimeError(text_content(result))
                    created = True
                    activated = True
                    print("USE_NOTEBOOK")
                    print(text_content(result))

                    result = await session.call_tool(
                        "insert_execute_code_cell",
                        {
                            "cell_index": -1,
                            "cell_source": "values = [6, 7]\nprint('rest-fallback-result', values[0] * values[1])",
                            "timeout": 30,
                        },
                    )
                    if result.isError:
                        raise RuntimeError(text_content(result))
                    execution_text = text_content(result)
                    print("INSERT_EXECUTE")
                    print(execution_text)
                    if "42" not in execution_text:
                        raise AssertionError("Expected execution output 42")

                    result = await session.call_tool(
                        "read_cell",
                        {
                            "cell_index": 1,
                            "include_outputs": True,
                            "notebook_name": notebook_name,
                        },
                    )
                    if result.isError:
                        raise RuntimeError(text_content(result))
                    cell_text = text_content(result)
                    print("READ_CELL")
                    print(cell_text)
                    if "rest-fallback-result" not in cell_text or "42" not in cell_text:
                        raise AssertionError(
                            "Saved cell source/output was not returned"
                        )
                finally:
                    if activated:
                        result = await session.call_tool(
                            "unuse_notebook", {"notebook_name": notebook_name}
                        )
                        print("UNUSE_NOTEBOOK")
                        print(text_content(result))
                    if created:
                        result = await session.call_tool(
                            "delete_server_path", {"path": notebook_path}
                        )
                        print("DELETE_TEMP")
                        print(text_content(result))

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    asyncio.run(main())
