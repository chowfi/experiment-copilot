from unittest.mock import MagicMock, patch

from mcp_web_browse.search import web_search_impl


def test_web_search_maps_fields():
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"title": "T", "href": "https://x.com", "body": "snippet"},
        {"title": "U", "url": "https://y.com", "body": "b2"},
    ]
    mock_cls = MagicMock(return_value=mock_ddgs)
    with patch("mcp_web_browse.search.DDGS", mock_cls):
        out = web_search_impl("q", max_results=2, region="us-en")
    assert len(out) == 2
    assert out[0] == {"title": "T", "url": "https://x.com", "snippet": "snippet"}
    assert out[1] == {"title": "U", "url": "https://y.com", "snippet": "b2"}
    mock_ddgs.text.assert_called_once()
