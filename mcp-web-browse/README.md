# mcp-web-browse

Local **[FastMCP](https://gofastmcp.com/)** server with two tools for research:

- **`web_search`** — Metasearch via the [`ddgs`](https://pypi.org/project/ddgs/) library (no API key). Returns JSON: `title`, `url`, `snippet`.
- **`fetch_url`** — HTTP GET, HTML → readable text via [trafilatura](https://github.com/adbar/trafilatura). Returns JSON: `url`, `title`, `text`, `truncated`, `bytes_read`.

Install lives under **`~/Documents/agents/mcp-web-browse`** (not inside a specific git repo unless you put it there).

## Install

```bash
cd ~/Documents/agents/mcp-web-browse
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run (llama.cpp Web UI)

Use **streamable HTTP** on localhost so the browser can reach the MCP endpoint through llama-server’s experimental proxy:

```bash
source .venv/bin/activate
mcp-web-browse --transport streamable-http --host 127.0.0.1 --port 8765 --path /mcp
# Default is stateless Streamable HTTP. If your client manages MCP-Session-Id, use: --stateful
```

Then start **llama-server** with the Web UI MCP proxy enabled (see [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)):

```bash
# Example only — adjust model path and flags for your build.
./llama-server -m your.gguf --webui-mcp-proxy
```

In the Web UI, add an MCP server pointing at:

`http://127.0.0.1:8765/mcp`

(Exact path must match `--path`; default is `/mcp`.)

**Security:** `--webui-mcp-proxy` is unsafe on untrusted networks. Use on localhost / trusted LAN only.

## Run (Cursor / stdio)

```bash
source .venv/bin/activate
mcp-web-browse --transport stdio
```

Add to Cursor MCP config, for example:

```json
{
  "mcpServers": {
    "web-browse": {
      "command": "/home/YOU/Documents/agents/mcp-web-browse/.venv/bin/mcp-web-browse",
      "args": ["--transport", "stdio"]
    }
  }
}
```

## Environment variables

| Variable | Meaning |
|----------|---------|
| `MCP_WEB_TRANSPORT` | Default transport if not passing `--transport` |
| `MCP_WEB_HOST` / `MCP_WEB_PORT` / `MCP_WEB_PATH` | HTTP bind / path |
| `MCP_WEB_MAX_BYTES` | Max response body read (default 2 MiB) |
| `MCP_WEB_TIMEOUT` | HTTP timeout seconds (default 20) |
| `MCP_WEB_ALLOW_PRIVATE_HOSTS` | Set to `1` to allow loopback/private IPs in `fetch_url` (trusted env only) |
| `MCP_WEB_USER_AGENT` | Override User-Agent |
| `MCP_WEB_CORS_ORIGINS` | Comma-separated allowed `Origin` values for browser clients (default `*` = any). Needed so `OPTIONS /mcp` preflight succeeds from the llama Web UI. |
| `MCP_WEB_STATELESS_HTTP` | `1` (default) = stateless Streamable HTTP (no `MCP-Session-Id` required; works with llama.cpp MCP proxy). Set to `0` for stateful sessions. |
| `MCP_WEB_SEARCH_REGION` | Default locale for search, e.g. `us-en` (helps local/business queries). |
| `MCP_WEB_SEARCH_TIMEOUT` | Seconds for search HTTP work (default 20). |

## Tests

```bash
cd ~/Documents/agents/mcp-web-browse
source .venv/bin/activate
pytest
```

## Typical agent flow

Ask the model to **search first** when you have not supplied a URL, then **fetch** one or a few result URLs for detail.
