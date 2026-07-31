"""Install the compatibility patch and tool extensions exactly once."""

from __future__ import annotations

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    from kaggle_jupyter_mcp.notebook_fallback import install_notebook_fallback

    install_notebook_fallback()

    # Importing upstream server builds its FastMCP registry. Extra tools are then
    # added to that same registry before the CLI starts accepting requests.
    from jupyter_mcp_server import server

    from kaggle_jupyter_mcp.extra_tools import register_extra_tools

    register_extra_tools(server.mcp)
    _installed = True
