from __future__ import annotations

import pytest

from kaggle_jupyter_mcp.visible_execution import normalize_notebook_target


def test_normalize_notebook_target_uses_real_ipynb_and_stable_name():
    assert normalize_notebook_target("runs/demo.ipynb") == (
        "runs/demo.ipynb",
        "demo",
    )
    assert normalize_notebook_target("runs\\demo.ipynb", "visible-run") == (
        "runs/demo.ipynb",
        "visible-run",
    )


@pytest.mark.parametrize("path", ["", "script.py", "../outside.ipynb"])
def test_normalize_notebook_target_rejects_non_notebook_or_traversal(path):
    with pytest.raises(ValueError):
        normalize_notebook_target(path)
