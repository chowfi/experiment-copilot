"""URL validation and SSRF-minded host checks."""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse


def _allow_private_hosts() -> bool:
    return os.environ.get("MCP_WEB_ALLOW_PRIVATE_HOSTS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        return True
    if ip.version == 4:
        if ip in ipaddress.ip_network("169.254.169.254/32"):
            return True
    return False


def assert_fetch_url_allowed(url: str) -> None:
    """
    Raise ValueError if url must not be fetched (scheme, host, resolved IP).

    Set MCP_WEB_ALLOW_PRIVATE_HOSTS=1 to permit loopback/private/link-local
    (use only in trusted environments).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise ValueError("URL must include a host.")

    host = parsed.hostname
    allow_private = _allow_private_hosts()

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host: {e!s}") from e

    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _ip_blocked(ip) and not allow_private:
            raise ValueError(
                "Host resolves to a private, loopback, or link-local address. "
                "Set MCP_WEB_ALLOW_PRIVATE_HOSTS=1 only if you trust this server."
            )
