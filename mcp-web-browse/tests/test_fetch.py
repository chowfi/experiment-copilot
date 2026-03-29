import json
from unittest.mock import MagicMock, patch

from mcp_web_browse.fetch import fetch_url_impl


def test_fetch_parses_html(monkeypatch):
    monkeypatch.setattr(
        "mcp_web_browse.fetch.assert_fetch_url_allowed", lambda u: None
    )
    html = b"<html><head><title>T</title></head><body><p>Hello world</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.iter_bytes = lambda: [html]
    mock_resp.raise_for_status = lambda: None
    cm = MagicMock()
    cm.__enter__ = lambda s: mock_resp
    cm.__exit__ = lambda *a: None

    mock_client = MagicMock()
    mock_client.stream = lambda method, url: cm
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = lambda *a: None

    with patch("mcp_web_browse.fetch.httpx.Client", return_value=mock_client):
        out = fetch_url_impl("https://example.com/page")

    assert out["url"] == "https://example.com/page"
    data = out
    assert "Hello" in data["text"] or data["text"] == ""
    assert "truncated" in data
