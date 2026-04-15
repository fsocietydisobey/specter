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
from specter.browser.interact import Interactor
from specter.browser.network import NetworkCapture
from specter.browser.react import ReactInspector
from specter.browser.runtime import Runtime
from specter.browser.structure import StructureAnalyzer
from specter.config import load_config

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "specter",
    instructions=(
        "Specter gives you eyes into the browser via CDP.\n\n"
        "CRITICAL DEBUGGING RULES — read these before every tool call:\n\n"
        "1. NEVER GUESS at data shapes, props, or field names. ALWAYS inspect "
        "the actual runtime data first using evaluate_js + get_console_logs. "
        "Example: evaluate_js(\"console.log(JSON.stringify(myVar))\") then "
        "get_console_logs() to read it. 10 seconds of inspection saves 10 "
        "minutes of wrong guesses.\n\n"
        "2. ALWAYS check get_console_logs(level='error') and get_errors() "
        "BEFORE trying to fix anything. The answer is usually already in the "
        "console as a stack trace pointing to the exact line.\n\n"
        "3. When debugging wrong data after an API call, check "
        "get_network_log(url_filter='/api/v1/...') to see what the API "
        "actually returned. Don't guess at response shapes.\n\n"
        "4. Use debug_snapshot() as the starting point — it returns screenshot "
        "+ page info + console errors + network errors + page structure in "
        "one call.\n\n"
        "5. To navigate, use click_element on links/buttons — the same way a "
        "user would. Use get_interactive_elements to find clickable links. "
        "Do NOT try to manipulate URLs or use history API — the app's router "
        "controls navigation.\n\n"
        "6. Don't fight with navigation to debug. If you need to see data, "
        "use evaluate_js to inspect it directly — don't navigate to a "
        "different page hoping to see it visually.\n\n"
        "Tab selection: call list_tabs() then connect_to_tab(id) on first "
        "use. Pick the app tab, NOT pipeline-tracer.html or devtools:// URLs."
    ),
)

# Singleton state — persists across tool calls within one MCP session
_connection: CDPConnection | None = None
_console: ConsoleCapture | None = None
_network: NetworkCapture | None = None
_runtime: Runtime | None = None
_react: ReactInspector | None = None
_interact: Interactor | None = None
_structure: StructureAnalyzer | None = None


async def _ensure_connected():
    """Ensure we have a live CDP connection, reconnecting if needed."""
    global _connection, _console, _network, _runtime, _react, _interact, _structure

    config = load_config()

    if _connection is not None and _connection.is_connected:
        return _connection, _console, _network, _runtime, _react, _interact, _structure

    _connection = CDPConnection(config)
    _console = ConsoleCapture(config)
    _network = NetworkCapture(config)
    _runtime = Runtime(config)
    _react = ReactInspector()
    _interact = Interactor()
    _structure = StructureAnalyzer()

    _console.register(_connection)
    _network.register(_connection)

    target = await _connection.connect()

    # Enable all CDP domains we need
    await _console.enable(_connection)     # Runtime domain
    await _network.enable(_connection)     # Network domain
    await _connection.send("Page.enable")  # Page domain (needed for screenshots)

    logger.info("Connected to: %s (%s)", target.title, target.url)

    return _connection, _console, _network, _runtime, _react, _interact, _structure


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
    conn, _, _, runtime, _, _, _ = await _ensure_connected()
    try:
        return await runtime.take_screenshot(conn, full_page=full_page, selector=selector)
    except (RuntimeError, TimeoutError, OSError) as e:
        # Connection may have gone stale — force reconnect and retry once
        logger.warning("Screenshot failed (%s), reconnecting...", e)
        await _force_reconnect()
        conn, _, _, runtime, _, _, _ = await _ensure_connected()
        return await runtime.take_screenshot(conn, full_page=full_page, selector=selector)


async def _force_reconnect() -> None:
    """Force-close the current connection so _ensure_connected creates a fresh one."""
    global _connection
    if _connection:
        try:
            await _connection.disconnect()
        except Exception:
            pass
        _connection = None


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
    _, console, _, _, _, _, _ = await _ensure_connected()
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
    _, console, _, _, _, _, _ = await _ensure_connected()
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
    _, _, network, _, _, _, _ = await _ensure_connected()
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
    _, _, network, _, _, _, _ = await _ensure_connected()
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
    conn, _, _, runtime, _, _, _ = await _ensure_connected()
    return await runtime.evaluate_js(conn, expression)


@mcp.tool()
async def get_page_info() -> dict:
    """Get current page info: URL, title, document state.

    Quick way to verify which page the browser is on before running
    other debug commands.

    Returns:
        Dict with url, title, readyState.
    """
    conn, _, _, runtime, _, _, _ = await _ensure_connected()
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
    conn, _, _, runtime, _, _, _ = await _ensure_connected()
    return await runtime.get_dom_html(conn, selector=selector, outer=outer)


@mcp.tool()
async def list_tabs() -> list[dict]:
    """List all open browser tabs.

    Returns tab IDs, titles, and URLs. Call this first, then use
    connect_to_tab(id) to switch to the correct app tab.

    Returns:
        List of tab dicts with id, title, url, and which one is
        currently connected (if any).
    """
    conn, _, _, _, _, _, _ = await _ensure_connected()
    targets = await conn.list_targets()
    current = conn.current_target
    result = []
    for t in targets:
        d = t.to_dict()
        d["connected"] = current is not None and t.id == current.id
        result.append(d)
    return result


@mcp.tool()
async def connect_to_tab(tab_id: str) -> dict:
    """Switch the Specter connection to a specific browser tab.

    Use list_tabs() first to see all tabs and their IDs, then call this
    with the ID of the tab you want to debug. This disconnects from the
    current tab and reconnects to the specified one. All event buffers
    (console, network) are cleared on reconnect.

    Args:
        tab_id: The tab ID from list_tabs() output.

    Returns:
        Dict with the connected tab's title and URL.
    """
    global _connection, _console, _network, _runtime, _react, _interact, _structure

    config = load_config()

    # Disconnect existing connection
    if _connection and _connection.is_connected:
        await _connection.disconnect()

    # Fresh connection to the specified tab
    _connection = CDPConnection(config)
    _console = ConsoleCapture(config)
    _network = NetworkCapture(config)
    _runtime = Runtime(config)
    _react = ReactInspector()
    _interact = Interactor()
    _structure = StructureAnalyzer()

    _console.register(_connection)
    _network.register(_connection)

    target = await _connection.connect(target_id=tab_id)
    await _console.enable(_connection)
    await _network.enable(_connection)

    logger.info("Switched to tab: %s (%s)", target.title, target.url)

    return {
        "connected": True,
        "tab_id": target.id,
        "title": target.title,
        "url": target.url,
    }


@mcp.tool()
async def reload_page(ignore_cache: bool = False) -> dict:
    """Reload the current page.

    Useful after making code changes to see the updated result, or to
    reset the page state before reproducing a bug.

    Args:
        ignore_cache: If true, does a hard reload (bypasses cache, like Ctrl+Shift+R).

    Returns:
        Dict confirming the reload.
    """
    conn, _, _, _, _, _, _ = await _ensure_connected()
    await conn.send("Page.reload", {"ignoreCache": ignore_cache})
    return {"reloaded": True, "ignore_cache": ignore_cache}


@mcp.tool()
async def clear_logs() -> dict:
    """Clear all buffered console logs and network entries.

    Useful to reset the capture before reproducing a specific bug.

    Returns:
        Dict with count of entries cleared.
    """
    _, console, network, _, _, _, _ = await _ensure_connected()
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
    conn, _, _, _, react, _, _ = await _ensure_connected()
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
    conn, _, _, _, react, _, _ = await _ensure_connected()
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
    conn, _, _, _, react, _, _ = await _ensure_connected()
    return await react.get_component_at(conn, selector)


@mcp.tool()
async def get_elements_grouped_by_component(selector: str) -> dict:
    """Find elements matching a selector and group them by owning React component.

    Use this when a CSS selector returns more elements than you expect —
    common when the same data is rendered in multiple views simultaneously
    (e.g., a "verified" view and "review" view both rendering the same
    rows). Instead of a flat list of 6 ambiguous elements, this returns
    two groups: {VerifiedSourcesView: 3 rows, ReviewSourcesView: 3 rows}.

    Each element in a group includes: tag, text (first 80 chars), visible
    state, bounding rect, data-testid, and id — so you can identify
    which one to interact with.

    Args:
        selector: CSS selector to match (e.g., "[class*='SourceRow']").

    Returns:
        Dict with total count and groups keyed by component name.
    """
    conn, _, _, _, react, _, _ = await _ensure_connected()
    return await react.get_elements_grouped_by_component(conn, selector)


@mcp.tool()
async def get_redux_state(path: str = "") -> dict:
    """PREFERRED: read Redux store state. Use THIS instead of evaluate_js.

    This tool already handles the hard parts:
      - Checks window.__REDUX_STORE__, __NEXT_REDUX_WRAPPER_STORE__, window.store
      - Walks ALL React fiber roots across ALL renderers (Next.js App Router
        creates multiple roots — the store is often in a non-first root)
      - Looks for Provider.store, Provider.value.store, and stateNode.store
      - Caches the found store on window.__SPECTER_STORE__ for fast subsequent reads
      - Safely serializes nested state with depth capping

    DO NOT manually walk __REACT_DEVTOOLS_GLOBAL_HOOK__ with evaluate_js —
    this tool does it for you in one call. Reaching for evaluate_js first
    is the #1 waste of calls in Specter.

    With no path: summary of top-level state keys and their shapes.
    With a path: the value at that path (e.g., "auth.session").

    Args:
        path: Dot-separated path into the state tree (e.g., "auth.session",
              "quotes.selectedBidId"). Empty string = summary view.

    Returns:
        Dict with the state value or a summary of top-level keys.
    """
    conn, _, _, _, react, _, _ = await _ensure_connected()
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
    conn, _, _, _, react, _, _ = await _ensure_connected()
    return await react.get_redux_actions(conn)


# ─── Browser interaction tools ─────────────────────────────────────────


@mcp.tool()
async def get_interactive_elements(role: str | None = None) -> list[dict]:
    """Get all interactive elements on the page (buttons, links, inputs, etc.).

    Returns every clickable, typeable, and selectable element with:
      - A stable CSS selector (prefers data-testid, then id, then aria-label)
      - The visible label/text
      - The element's role (button, link, textbox, checkbox, etc.)
      - Current state (value, checked, disabled)
      - Bounding box (x, y, width, height)

    Use this to understand the page's interactive surface, then use
    click_element, fill_input, or select_option to interact.

    Args:
        role: Optional filter by role ("button", "link", "textbox", etc.).

    Returns:
        List of interactive element descriptors.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.get_interactive_elements(conn, role_filter=role)


@mcp.tool()
async def click_element(selector: str) -> dict:
    """Click an element by CSS selector.

    Scrolls the element into view, then dispatches the full mouse event
    sequence (mousedown → mouseup → click) that React expects.

    Use get_interactive_elements first to find the right selector.

    Args:
        selector: CSS selector for the element to click.

    Returns:
        Confirmation of the click or error if element not found.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.click_element(conn, selector)


@mcp.tool()
async def fill_input(selector: str, value: str) -> dict:
    """Type a value into an input field or textarea.

    Handles React controlled inputs by using the native value setter
    and dispatching both input and change events. Clears existing
    content before typing.

    Args:
        selector: CSS selector for the input element.
        value: Text to type into the field.

    Returns:
        Confirmation with the resulting value.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.fill_input(conn, selector, value)


@mcp.tool()
async def select_option(selector: str, option_value: str) -> dict:
    """Select an option from a dropdown by value or visible text.

    Args:
        selector: CSS selector for the select element.
        option_value: The option's value attribute or visible text.

    Returns:
        Confirmation or list of available options if not found.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.select_option(conn, selector, option_value)


@mcp.tool()
async def wait_for_element(selector: str, timeout_ms: int = 10000) -> dict:
    """Wait for an element to appear and become visible.

    Polls every 200ms until the element exists and has layout. Useful
    after clicking something that triggers navigation, a modal, or
    lazy-loaded content.

    Args:
        selector: CSS selector to wait for.
        timeout_ms: Maximum wait time in milliseconds (default 10000).

    Returns:
        Dict with found status and elapsed time.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.wait_for_element(conn, selector, timeout_ms=timeout_ms)


# ─── New v0.4 tools ────────────────────────────────────────────────────


@mcp.tool()
async def hover_element(selector: str) -> dict:
    """Hover over an element to reveal hidden UI.

    Many UIs show action buttons, edit controls, dropdown triggers, and
    tooltips only on hover. This dispatches mouseenter + mouseover +
    mousemove events to trigger those states.

    After hovering, call get_interactive_elements() to see the newly-
    revealed elements, or take_screenshot() to see the visual change.

    Args:
        selector: CSS selector for the element to hover over (e.g., a
                  table row, a card, a menu trigger).

    Returns:
        Confirmation of the hover or error if element not found.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.hover_element(conn, selector)


@mcp.tool()
async def press_key(
    key: str,
    modifiers: list[str] | None = None,
    selector: str | None = None,
) -> dict:
    """Press a keyboard key.

    Common uses:
      - press_key("Enter") — submit a form
      - press_key("Escape") — close a modal/dialog
      - press_key("Tab") — move focus to next element
      - press_key("ArrowDown") — navigate a dropdown
      - press_key("a", modifiers=["ctrl"]) — select all
      - press_key("Backspace") — delete character

    Args:
        key: Key name — "Enter", "Escape", "Tab", "ArrowDown", "ArrowUp",
             "Backspace", "Delete", "Space", "Home", "End", or a single
             character like "a".
        modifiers: Optional list of modifier keys: "ctrl", "shift", "alt", "meta".
        selector: Optional CSS selector to focus before pressing the key.

    Returns:
        Confirmation of the key press.
    """
    conn, _, _, _, _, interact, _ = await _ensure_connected()
    return await interact.press_key(conn, key, modifiers=modifiers, selector=selector)


# NOTE: navigate_to and router_navigate are DISABLED.
# Next.js App Router strips query params and resets state on programmatic
# navigation. Use click_element + get_interactive_elements to navigate
# the app the same way a user would — by clicking links and buttons.
# The code is in runtime.py if these are ever re-enabled.


@mcp.tool()
async def wait_for_network_idle(idle_ms: int = 500, timeout_ms: int = 10000) -> dict:
    """Wait until all network requests have completed.

    Monitors in-flight HTTP requests and waits until none are pending
    for at least idle_ms milliseconds. Use this after navigation or
    clicking something that triggers API calls — ensures the page is
    fully loaded before taking a screenshot or inspecting state.

    Args:
        idle_ms: How long the network must be quiet to count as idle
                 (default 500ms).
        timeout_ms: Maximum wait time (default 10000ms).

    Returns:
        Dict with idle status, remaining in-flight count, and elapsed time.
    """
    _, _, network, _, _, _, _ = await _ensure_connected()
    return await network.wait_for_idle(idle_ms=idle_ms, timeout_ms=timeout_ms)


@mcp.tool()
async def get_page_structure() -> dict | list:
    """Get the semantic structure of the current page.

    Walks the DOM using ARIA landmarks, roles, and semantic HTML to build
    a structural map. Returns a tree showing:

      - Major sections: navigation, main content, sidebars, dialogs
      - Widget state: which tab is selected, what's expanded/collapsed
      - Section contents: headings, interactive element counts, labels
      - Data-testid anchors for stable references

    This is the "what am I looking at?" tool. Use it to understand the
    page layout before deciding what to interact with. Much more useful
    than parsing a screenshot for structural understanding.

    Returns:
        Nested tree of page sections with roles, labels, and states.
    """
    conn, _, _, _, _, _, structure = await _ensure_connected()
    return await structure.get_page_structure(conn)


@mcp.tool()
async def debug_snapshot() -> dict:
    """Capture a complete debugging snapshot in one call.

    Returns everything Claude needs to understand the current page state:
      - Screenshot (file path to PNG)
      - Page URL and title
      - Console errors (last 10)
      - Network errors (last 10)
      - Page structure (semantic layout map)

    This replaces the 5-call sequence of take_screenshot + get_page_info +
    get_console_logs + get_network_errors + get_page_structure with a
    single tool call. Use this as the starting point for any debugging
    session.

    Returns:
        Dict with screenshot path, page info, errors, network errors,
        and page structure.
    """
    conn, console, network, runtime, _, _, structure = await _ensure_connected()

    # Gather everything in parallel where possible
    screenshot = await runtime.take_screenshot(conn)
    page_info = await runtime.get_page_info(conn)
    console_errors = console.get_logs(level="error", limit=10)
    exceptions = console.get_errors(limit=10)
    network_errors = network.get_requests(errors_only=True, limit=10)
    page_struct = await structure.get_page_structure(conn)

    return {
        "screenshot": screenshot.get("file_path"),
        "page": page_info,
        "console_errors": console_errors,
        "exceptions": exceptions,
        "network_errors": network_errors,
        "page_structure": page_struct,
    }
