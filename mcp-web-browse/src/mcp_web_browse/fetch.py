"""HTTP fetch with size limits and text extraction."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import trafilatura

from mcp_web_browse.security import assert_fetch_url_allowed

DEFAULT_UA = (
    "mcp-web-browse/0.1 (+https://github.com; local MCP research tool)"
)


def _limits() -> tuple[int, float, int]:
    max_bytes = int(os.environ.get("MCP_WEB_MAX_BYTES", str(2 * 1024 * 1024)))
    timeout = float(os.environ.get("MCP_WEB_TIMEOUT", "20"))
    max_redirects = int(os.environ.get("MCP_WEB_MAX_REDIRECTS", "5"))
    return max_bytes, timeout, max_redirects


def fetch_url_impl(url: str) -> dict[str, Any]:
    assert_fetch_url_allowed(url)
    max_bytes, timeout, max_redirects = _limits()

    headers = {"User-Agent": os.environ.get("MCP_WEB_USER_AGENT", DEFAULT_UA)}

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        max_redirects=max_redirects,
        headers=headers,
    ) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            raw = b"".join(chunks)[:max_bytes]

    truncated = total >= max_bytes

    text = ""
    title = ""
    try:
        html = raw.decode("utf-8", errors="replace")
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
        )
        text = (extracted or "").strip()
        meta = trafilatura.extract_metadata(html, url=url)
        if meta and meta.title:
            title = meta.title.strip()
    except Exception:
        text = raw.decode("utf-8", errors="replace")[:8000]

    if not title:
        title = url

    return {
        "url": url,
        "title": title,
        "text": text,
        "truncated": truncated,
        "bytes_read": len(raw),
    }


def fetch_url_json(url: str) -> str:
    return json.dumps(fetch_url_impl(url), ensure_ascii=False, indent=2)
