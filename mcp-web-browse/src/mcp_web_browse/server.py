"""FastMCP server: web_search + fetch_url."""

from __future__ import annotations

import argparse
import os

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from mcp_web_browse.fetch import fetch_url_json
from mcp_web_browse.search import web_search_json

mcp = FastMCP(
    name="mcp-web-browse",
    instructions=(
        "Tools for open-web research. Prefer web_search when the user has not "
        "given a URL, then fetch_url on 1–3 relevant results to read full pages."
    ),
)


@mcp.tool()
def web_search(query: str, max_results: int = 8, region: str | None = None) -> str:
    """Search the public web (metasearch via ddgs). Returns JSON: title, url, snippet.

    Use for discovery when the user has not given a URL. Prefer specific queries
    (e.g. \"ice cream shop Prospect Park Brooklyn\") to avoid ambiguous terms like \"ICE\" alone.

    region: Locale hint, e.g. us-en, uk-en. Omit to use server default (see MCP_WEB_SEARCH_REGION).
    """
    reg = region.strip() if isinstance(region, str) and region.strip() else None
    return web_search_json(query, max_results=max_results, region=reg)


@mcp.tool()
def fetch_url(url: str) -> str:
    """Download an http(s) page and return JSON with title, plain text, and truncation flag.

    Only public internet hosts are allowed unless MCP_WEB_ALLOW_PRIVATE_HOSTS=1.
    """
    return fetch_url_json(url)


def _default_stateless_http() -> bool:
    """Stateful Streamable HTTP requires MCP-Session-Id on every POST; llama.cpp proxy often omits it."""
    v = os.environ.get("MCP_WEB_STATELESS_HTTP", "1").strip().lower()
    if v in ("0", "false", "no"):
        return False
    return True


def _cors_middleware() -> list[Middleware]:
    """Browser-based clients (e.g. llama.cpp Web UI via MCP proxy) send OPTIONS preflight."""
    raw = os.environ.get("MCP_WEB_CORS_ORIGINS", "*").strip()
    if raw == "*":
        origins: list[str] = ["*"]
    else:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        if not origins:
            origins = ["*"]
    return [
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP web browse server")
    parser.add_argument(
        "--transport",
        choices=("streamable-http", "stdio", "sse", "http"),
        default=os.environ.get("MCP_WEB_TRANSPORT", "streamable-http"),
        help="MCP transport (default: streamable-http for llama.cpp Web UI)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_WEB_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_WEB_PORT", "8765")),
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("MCP_WEB_PATH", "/mcp"),
        help="HTTP path for MCP endpoint",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        help=(
            "Use stateful Streamable HTTP (session id required on POST). "
            "Default is stateless so browser/proxy clients like llama.cpp Web UI work without MCP-Session-Id."
        ),
    )
    args = parser.parse_args()

    stateless_http = _default_stateless_http()
    if args.stateful:
        stateless_http = False
    # SSE transport cannot use stateless mode (FastMCP / MCP SDK constraint).
    if args.transport == "sse":
        stateless_http = False

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            middleware=_cors_middleware(),
            stateless_http=stateless_http,
        )


if __name__ == "__main__":
    main()
