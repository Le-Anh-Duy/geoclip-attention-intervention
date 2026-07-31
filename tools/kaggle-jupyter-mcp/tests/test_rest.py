from __future__ import annotations

import pytest

from kaggle_jupyter_mcp.rest import JupyterAPIError, JupyterRestClient, api_path


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = b"{}"
        self.text = ""
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.headers = {}
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


def test_api_path_preserves_slashes_and_quotes_names():
    assert api_path("folder/a b.ipynb") == "folder/a%20b.ipynb"
    with pytest.raises(ValueError):
        api_path("../outside")


def test_signed_proxy_prefix_is_preserved():
    session = FakeSession(FakeResponse(payload={"ok": True}))
    client = JupyterRestClient(
        "https://example.test/k/123/signed/proxy", token="secret", session=session
    )

    assert client.request_json("GET", "api/status") == {"ok": True}
    _, url, _ = session.calls[0]
    assert url == "https://example.test/k/123/signed/proxy/api/status"
    assert session.headers["Authorization"] == "token secret"


def test_http_errors_do_not_disclose_base_url_or_token():
    session = FakeSession(FakeResponse(status_code=404, payload={"message": "missing"}))
    client = JupyterRestClient(
        "https://example.test/private-signed-url", token="super-secret", session=session
    )

    with pytest.raises(JupyterAPIError) as caught:
        client.request_json("GET", "api/thing")

    message = str(caught.value)
    assert "private-signed-url" not in message
    assert "super-secret" not in message
    assert "HTTP 404" in message
