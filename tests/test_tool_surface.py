from __future__ import annotations

import pytest

from kaggle_jupyter_mcp.tool_surface import TOOL_DESCRIPTIONS


@pytest.mark.asyncio
async def test_bootstrap_exposes_only_curated_tools(monkeypatch):
    monkeypatch.setenv("JUPYTER_URL", "http://127.0.0.1:8888")
    from kaggle_jupyter_mcp.bootstrap import install

    install()
    from jupyter_mcp_server import server

    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert set(tools) == set(TOOL_DESCRIPTIONS)
    for name, description in TOOL_DESCRIPTIONS.items():
        assert tools[name].description == description

    assert "execute_code" not in tools
    assert "execute_code_in_notebook" not in tools
    assert "upload_local_file_to_jupyter" not in tools
    assert "use_notebook" not in tools
