"""Web search via the ``ddgs`` metasearch library (no API key)."""

from __future__ import annotations

import json
import os
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException


def _search_timeout() -> int:
    return int(os.environ.get("MCP_WEB_SEARCH_TIMEOUT", "20"))


def _default_region() -> str:
    return os.environ.get("MCP_WEB_SEARCH_REGION", "us-en").strip() or "us-en"


def web_search_impl(
    query: str,
    max_results: int = 8,
    *,
    region: str | None = None,
) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    max_results = max(1, min(max_results, 20))
    reg = (region or _default_region()).strip() or "us-en"

    ddgs = DDGS(timeout=_search_timeout())
    try:
        raw = ddgs.text(
            q,
            region=reg,
            max_results=max_results,
            backend="auto",
        )
    except DDGSException:
        raise
    except Exception as ex:
        raise DDGSException(str(ex)) from ex

    hits: list[dict[str, Any]] = []
    for r in raw:
        hits.append(
            {
                "title": str(r.get("title") or ""),
                "url": str(r.get("href") or r.get("url") or ""),
                "snippet": str(r.get("body") or ""),
            }
        )
    return hits


def web_search_json(
    query: str,
    max_results: int = 8,
    *,
    region: str | None = None,
) -> str:
    return json.dumps(
        web_search_impl(query, max_results=max_results, region=region),
        ensure_ascii=False,
        indent=2,
    )
