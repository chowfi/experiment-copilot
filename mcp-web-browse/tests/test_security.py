import socket

import pytest

from mcp_web_browse.security import assert_fetch_url_allowed


def test_blocks_file_scheme():
    with pytest.raises(ValueError, match="http"):
        assert_fetch_url_allowed("file:///etc/passwd")


def test_blocks_private_ip(monkeypatch):
    def fake_getaddrinfo(host, port, *a, **k):
        if host == "evil.local":
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))]
        raise socket.gaierror("nx")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="private"):
        assert_fetch_url_allowed("http://evil.local/")


def test_allows_public_when_resolves_public(monkeypatch):
    def fake_getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert_fetch_url_allowed("http://example.com/")
