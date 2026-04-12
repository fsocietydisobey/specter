"""MCP server exposing Specter's browser debugging tools.

Maintains a persistent CDP connection to Firefox. Tools are designed
for an AI debugging workflow:
  1. Take a screenshot to see the visual state
  2. Check console logs for errors
  3. Check network for failed requests
  4. Evaluate JS to inspect runtime state
  5. Read DOM to check the rendered output
"""

from __future__ import annotations

import asyncio
import logging

from mcp.server.fastmcp import FastMCP

from specter.browser.connection import CDPConnection
from specter.browser.console import ConsoleCapture
from specter.browser.network import NetworkCapture
from specter.browser.react import ReactInspector
from specter.browser.runtime import Runtime
from specter.config import load_config

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "specter",
    instructions=(
        "Specter gives you eyes into the browser. It connects to a running "
        "Firefox instance and captures console logs, errors, network activity, "
        "screenshots, and React component internals in real time.\n\n"
        "Firefox must be running with: firefox --remote-debugging-port 9222\n\n"
        "Debugging workflow:\n"
        "1. take_screenshot — see what the user sees\n"
        "2. get_console_logs — check for errors and warnings\n"
        "3. get_network_errors — check for failed API calls\n"
        "4. get_component_tree or get_component_at — inspect React components, props, hooks\n"
        "5. get_redux_state — check Redux store state\n"
        "6. evaluate_js — inspect any runtime state\n"
        "7. Fix the code, then take_screenshot again to verify"
    ),
)

# Singleton state — persists across tool calls within one MCP session
_connection: CDPConnection | None = None
_console: ConsoleCapture | None = None
_network: NetworkCapture | None = None
_runtime: Runtime | None = None
_react: ReactInspector | None = None


async def _ensure_connected() -> tuple[CDPConnection, ConsoleCapture, NetworkCapture, Runtime, ReactInspector]:
    """Ensure we have a live CDP connection, reconnecting if needed."""
    global _connection, _console, _network, _runtime, _react

    config = load_config()

    if _connection is not None and _connection.is_connected:
        return _connection, _console, _network, _runtime, _react

    _connection = CDPConnection(config)
    _console = ConsoleCapture(config)
    _network = NetworkCapture(config)
    _runtime = Runtime(config)
    _react = ReactInspector()

    _console.register(_connection)
    _network.register(_connection)

    target = await _connection.connect()
    await _console.enable(_connection)
    await _network.enable(_connection)

    logger.info("Connected to: %s (%s)", target.title, target.url)

    return _connection, _console, _network, _runtime, _react


@mcp.tool()
async def take_screenshot(
    full_page: bool = False,
    selector: str | None = None,
) -> dict:
    """Capture a screenshot of the current browser page.

    The screenshot is saved as a PNG file. Use the returned file_path
    with the Read tool to view the image — Claude Code is multimodal
    and can analyze screenshots directly.

    Args:
        full_page: If true, capture the entire scrollable page (not just viewport).
        selector: Optional CSS selector to screenshot a specific element only.

    Returns:
        Dict with file_path to the saved PNG, timestamp, and dimensions.
    """
    conn, _, _, runtime, _ = await _ensure_connected()
    return await runtime.take_screenshot(conn, full_page=full_page, selector=selector)


@mcp.tool()
async def get_console_logs(
    level: str | None = None,
    since: float | None = None,
    limit: int = 50,
) -> list[dict]:
    """Retrieve console output from the browser.

    Captures everything written via console.log, console.warn,
    console.error, and console.info. Includes source locations
    and stack traces for errors.

    Args:
        level: Filter by level — "log", "warn", "error", "info", "debug".
        since: Only entries after this Unix timestamp.
        limit: Max entries to return (default 50, newest first).

    Returns:
        List of console entries with timestamp, level, text, source location.
    """
    _, console, _, _, _ = await _ensure_connected()
    return console.get_logs(level=level, since=since, limit=limit)


@mcp.tool()
async def get_errors(since: float | None = None, limit: int = 50) -> list[dict]:
    """Retrieve unhandled JavaScript exceptions from the browser.

    These are errors that weren't caught by try/catch or error boundaries.
    Each entry includes the error message, source file, line/column, and
    full stack trace.

    Args:
        since: Only entries after this Unix timestamp.
        limit: Max entries to return (default 50).

    Returns:
        List of exception entries with message, source, line, column, stack_trace.
    """
    _, console, _, _, _ = await _ensure_connected()
    return console.get_errors(since=since, limit=limit)


@mcp.tool()
async def get_network_errors(
    since: float | None = None,
    limit: int = 50,
    url_filter: str | None = None,
) -> list[dict]:
    """Retrieve failed HTTP requests (4xx, 5xx, and network errors).

    Useful for debugging API call failures, CORS issues, and network
    connectivity problems.

    Args:
        since: Only entries after this Unix timestamp.
        limit: Max entries to return.
        url_filter: Only URLs containing this substring (e.g., "/api/v1").

    Returns:
        List of failed network entries with method, URL, status, error text, duration.
    """
    _, _, network, _, _ = await _ensure_connected()
    return network.get_requests(errors_only=True, since=since, limit=limit, url_filter=url_filter)


@mcp.tool()
async def get_network_log(
    since: float | None = None,
    limit: int = 50,
    url_filter: str | None = None,
) -> list[dict]:
    """Retrieve all HTTP requests (not just errors).

    Useful for tracing API flow, checking request timing, and verifying
    that the right endpoints are being called.

    Args:
        since: Only entries after this Unix timestamp.
        limit: Max entries to return.
        url_filter: Only URLs containing this substring.

    Returns:
        List of all network entries with method, URL, status, duration.
    """
    _, _, network, _, _ = await _ensure_connected()
    return network.get_requests(errors_only=False, since=since, limit=limit, url_filter=url_filter)


@mcp.tool()
async def evaluate_js(expression: str) -> dict:
    """Evaluate a JavaScript expression in the browser page context.

    Runs the expression in the active page and returns the result.
    Useful for inspecting runtime state: checking variables, reading
    localStorage, querying the DOM, checking Redux state, etc.

    Examples:
      - "document.title"
      - "localStorage.getItem('token')"
      - "window.__NEXT_DATA__"
      - "document.querySelectorAll('.error-message').length"

    Args:
        expression: JavaScript expression to evaluate.

    Returns:
        Dict with type, value, and description of the result.
    """
    conn, _, _, runtime, _ = await _ensure_connected()
    return await runtime.evaluate_js(conn, expression)


@mcp.tool()
async def get_page_info() -> dict:
    """Get current page info: URL, title, document state.

    Quick way to verify which page the browser is on before running
    other debug commands.

    Returns:
        Dict with url, title, readyState.
    """
    conn, _, _, runtime, _ = await _ensure_connected()
    return await runtime.get_page_info(conn)


@mcp.tool()
async def get_dom_html(selector: str = "body", outer: bool = False) -> dict:
    """Get the rendered HTML of an element.

    Useful for checking what the browser actually rendered vs what
    the React component tree produced.

    Args:
        selector: CSS selector for the element (default: "body").
        outer: If true, return outerHTML; if false, innerHTML.

    Returns:
        Dict with the HTML content (truncated at 50KB if very large).
    """
    conn, _, _, runtime, _ = await _ensure_connected()
    return await runtime.get_dom_html(conn, selector=selector, outer=outer)


@mcp.tool()
async def list_tabs() -> list[dict]:
    """List all open browser tabs.

    Returns tab IDs, titles, and URLs. Use this to find the right tab
    before running other debug commands, especially when multiple tabs
    are open.

    Returns:
        List of tab dicts with id, title, url.
    """
    conn, _, _, _, _ = await _ensure_connected()
    targets = await conn.list_targets()
    return [t.to_dict() for t in targets]


@mcp.tool()
async def clear_logs() -> dict:
    """Clear all buffered console logs and network entries.

    Useful to reset the capture before reproducing a specific bug.

    Returns:
        Dict with count of entries cleared.
    """
    _, console, network, _, _ = await _ensure_connected()
    console_count = console.clear()
    network_count = network.clear()
    return {"console_cleared": console_count, "network_cleared": network_count}


# ─── React component inspection tools ─────────────────────────────────


@mcp.tool()
async def check_react() -> dict:
    """Check if React is running in development mode and what's available.

    Run this first before using React inspection tools. Reports:
    React version, renderer info, whether fiber roots exist, whether
    Redux DevTools and Next.js data are present.

    Returns:
        Dict with availability info for React, Redux, and Next.js.
    """
    conn, _, _, _, react = await _ensure_connected()
    return await react.check_react_available(conn)


@mcp.tool()
async def get_component_tree(max_depth: int = 15, max_children: int = 50) -> dict:
    """Walk the React component tree and return the full hierarchy.

    Returns every React component with its name, source file + line,
    current props, hooks (useState values, useEffect deps, useRef values),
    and children. This is the same information the React DevTools
    "Components" panel shows.

    Only works in React development mode.

    Args:
        max_depth: Maximum tree depth to walk (default 15).
        max_children: Maximum children per component (default 50).

    Returns:
        Nested component tree with names, source, props, hooks, children.
    """
    conn, _, _, _, react = await _ensure_connected()
    return await react.get_component_tree(conn, max_depth=max_depth, max_children=max_children)


@mcp.tool()
async def get_component_at(selector: str) -> dict:
    """Get the React component that owns a specific DOM element.

    Finds the DOM element by CSS selector, then walks up the React fiber
    tree to find the nearest component. Returns the component's name,
    source file, current props, and the chain of parent components above
    it in the tree.

    Useful for answering: "what component renders this element, and what
    props is it receiving?"

    Args:
        selector: CSS selector for the DOM element (e.g., ".quote-details",
                  "#main-content", "[data-testid='login-form']").

    Returns:
        Dict with component name, source location, props, and parent chain.
    """
    conn, _, _, _, react = await _ensure_connected()
    return await react.get_component_at(conn, selector)


@mcp.tool()
async def get_redux_state(path: str = "") -> dict:
    """Read the Redux store state.

    With no path: returns a summary of top-level state keys and their
    shapes (useful for orientation).
    With a path: returns the value at that path (e.g., "auth.session"
    returns the session object).

    The store is located via common globals: window.__REDUX_STORE__,
    window.__NEXT_REDUX_WRAPPER_STORE__, or window.store. If your app
    uses a different pattern, use evaluate_js to access it directly.

    Args:
        path: Dot-separated path into the state tree (e.g., "auth.session",
              "quotes.selectedBidId"). Empty string = summary view.

    Returns:
        Dict with the state value or a summary of top-level keys.
    """
    conn, _, _, _, react = await _ensure_connected()
    return await react.get_redux_state(conn, path=path)


@mcp.tool()
async def get_redux_actions() -> dict:
    """Get info about Redux action dispatch capabilities.

    Reports whether Redux DevTools is available and lists current
    state keys. Full action history replay requires the Redux DevTools
    browser extension.

    Returns:
        Dict with Redux DevTools status and current state shape.
    """
    conn, _, _, _, react = await _ensure_connected()
    return await react.get_redux_actions(conn)
