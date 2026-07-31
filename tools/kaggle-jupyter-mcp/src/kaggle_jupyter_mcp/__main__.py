"""CLI entry point that installs Kaggle compatibility before starting MCP."""


def main() -> None:
    from kaggle_jupyter_mcp.bootstrap import install

    install()

    from jupyter_mcp_server.cli.cli import serve

    serve()


if __name__ == "__main__":
    main()
