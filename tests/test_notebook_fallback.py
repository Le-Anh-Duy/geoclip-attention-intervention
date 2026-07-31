from __future__ import annotations

from copy import deepcopy

import nbformat
import pytest

from kaggle_jupyter_mcp.notebook_fallback import RestNotebookModel


class FakeContentsClient:
    def __init__(self, notebook):
        self.notebook = deepcopy(notebook)
        self.last_modified = "2026-01-01T00:00:00Z"
        self.checkpoints = 0
        self.saves = 0

    def get_content(self, path, *, content=True, **kwargs):
        model = {"path": path, "type": "notebook", "last_modified": self.last_modified}
        if content:
            model["content"] = deepcopy(self.notebook)
        return model

    def create_checkpoint(self, path):
        self.checkpoints += 1
        return {"id": "checkpoint", "last_modified": self.last_modified}

    def save_content(self, path, *, content, model_type, format):
        assert model_type == "notebook"
        assert format == "json"
        self.notebook = deepcopy(content)
        self.saves += 1
        self.last_modified = "2026-01-01T00:00:01Z"
        return {"path": path, "type": "notebook", "last_modified": self.last_modified}


class FakeKernel:
    def execute_interactive(self, code, output_hook, **kwargs):
        assert code == "print(42)"
        output_hook(
            {
                "msg_type": "stream",
                "content": {"name": "stdout", "text": "42\n"},
            }
        )
        output_hook(
            {
                "msg_type": "execute_result",
                "content": {
                    "execution_count": 7,
                    "data": {"text/plain": "42"},
                    "metadata": {},
                },
            }
        )
        return {"content": {"status": "ok", "execution_count": 7}}


def make_notebook():
    return nbformat.writes(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(42)")])
    )


@pytest.mark.asyncio
async def test_rest_model_mutations_are_saved_on_clean_exit():
    notebook = nbformat.reads(make_notebook(), as_version=4)
    client = FakeContentsClient(notebook)
    model = RestNotebookModel(
        client=client,
        path="test.ipynb",
        notebook=notebook,
        last_modified=client.last_modified,
    )

    async with model:
        model.insert_cell(-1, "# Notes", "markdown")
        model.set_cell_source(0, "print(6 * 7)")

    assert client.saves == 1
    assert client.checkpoints == 1
    assert len(client.notebook["cells"]) == 2
    assert client.notebook["cells"][0]["source"] == "print(6 * 7)"


@pytest.mark.asyncio
async def test_rest_model_discards_mutation_when_tool_raises():
    notebook = nbformat.reads(make_notebook(), as_version=4)
    client = FakeContentsClient(notebook)
    model = RestNotebookModel(
        client=client,
        path="test.ipynb",
        notebook=notebook,
        last_modified=client.last_modified,
    )

    with pytest.raises(RuntimeError):
        async with model:
            model.set_cell_source(0, "broken")
            raise RuntimeError("tool failed")

    assert client.saves == 0
    assert client.notebook["cells"][0]["source"] == "print(42)"


def test_rest_model_executes_and_captures_nbformat_outputs():
    notebook = nbformat.reads(make_notebook(), as_version=4)
    client = FakeContentsClient(notebook)
    model = RestNotebookModel(
        client=client,
        path="test.ipynb",
        notebook=notebook,
        last_modified=client.last_modified,
    )

    result = model.execute_cell(0, FakeKernel())

    assert result["status"] == "ok"
    assert result["execution_count"] == 7
    assert result["outputs"][0] == {
        "output_type": "stream",
        "name": "stdout",
        "text": "42\n",
    }
    assert model[0]["execution_count"] == 7


def test_delete_many_cells_validates_before_mutating():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("a"),
            nbformat.v4.new_code_cell("b"),
            nbformat.v4.new_code_cell("c"),
        ]
    )
    model = RestNotebookModel(
        client=FakeContentsClient(notebook),
        path="test.ipynb",
        notebook=notebook,
        last_modified="now",
    )

    with pytest.raises(IndexError):
        model.delete_many_cells([0, 99])
    assert len(model) == 3

    removed = model.delete_many_cells([2, 0])
    assert [cell["source"] for cell in removed] == ["c", "a"]
    assert model[0]["source"] == "b"


def test_insert_cell_respects_pre_cell_id_nbformat_minor():
    notebook = nbformat.v4.new_notebook()
    notebook["nbformat_minor"] = 4
    model = RestNotebookModel(
        client=FakeContentsClient(notebook),
        path="legacy.ipynb",
        notebook=notebook,
        last_modified="now",
    )

    model.insert_cell(-1, "print('legacy')", "code")

    assert "id" not in model[0]
    nbformat.validate(nbformat.from_dict(model.as_dict()))


def test_get_and_set_cell_source_match_upstream_tool_interface():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell()])
    notebook["cells"][0]["source"] = ["first line\n", "second line"]
    model = RestNotebookModel(
        client=FakeContentsClient(notebook),
        path="test.ipynb",
        notebook=notebook,
        last_modified="now",
    )

    assert model.get_cell_source(0) == "first line\nsecond line"

    model.set_cell_source(0, "replacement")

    assert model.get_cell_source(0) == "replacement"
