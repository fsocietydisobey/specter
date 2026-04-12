"""Browser interaction tools — click, type, select, wait.

Gives Claude higher-level interaction primitives instead of requiring
raw evaluate_js scripts. The key tool is get_interactive_elements(),
which returns every clickable/typeable element on the page with its
label, role, and a stable selector — so Claude can reason about what
to interact with based on the screenshot + element list.

Flow:
  1. Claude takes a screenshot (sees the page visually)
  2. Claude calls get_interactive_elements() (gets the interactive surface)
  3. Claude picks the right element by matching its label to the goal
  4. Claude calls click_element/fill_input/select_option

This is the pattern AI browser agents use. The intelligence (deciding
WHAT to click) stays in Claude. The mechanics (HOW to click) are here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from specter.browser.connection import CDPConnection

logger = logging.getLogger(__name__)

# JavaScript that extracts all interactive elements with stable selectors
EXTRACT_INTERACTIVES_SCRIPT = """
(() => {
    const results = [];
    const seen = new Set();

    // All potentially interactive elements
    const selectors = [
        'a[href]',
        'button',
        'input',
        'select',
        'textarea',
        '[role="button"]',
        '[role="link"]',
        '[role="tab"]',
        '[role="menuitem"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[role="switch"]',
        '[onclick]',
        '[tabindex]',
        'summary',
        'label[for]',
    ];

    const allElements = document.querySelectorAll(selectors.join(', '));

    for (const el of allElements) {
        // Skip hidden elements
        if (el.offsetParent === null && el.tagName !== 'INPUT' && el.type !== 'hidden') continue;
        if (el.disabled) continue;
        const style = getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;

        // Build a stable selector for this element
        const selector = buildSelector(el);
        if (seen.has(selector)) continue;
        seen.add(selector);

        // Get the visible label/text
        const label = getLabel(el);
        if (!label && el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA' && el.tagName !== 'SELECT') continue;

        const rect = el.getBoundingClientRect();

        results.push({
            selector,
            tag: el.tagName.toLowerCase(),
            type: el.type || null,
            role: el.getAttribute('role') || inferRole(el),
            label: label || '',
            placeholder: el.placeholder || null,
            value: el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ? (el.value || '').substring(0, 100) : null,
            href: el.href || null,
            disabled: el.disabled || false,
            checked: el.checked !== undefined ? el.checked : null,
            rect: {
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height),
            },
        });
    }

    function buildSelector(el) {
        // Prefer data-testid, then id, then build a positional selector
        if (el.dataset?.testid) return '[data-testid="' + el.dataset.testid + '"]';
        if (el.id) return '#' + CSS.escape(el.id);

        // Use aria-label if unique
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) {
            const matches = document.querySelectorAll('[aria-label="' + CSS.escape(ariaLabel) + '"]');
            if (matches.length === 1) return '[aria-label="' + ariaLabel + '"]';
        }

        // Build a path: tag.class:nth-of-type
        const parts = [];
        let current = el;
        while (current && current !== document.body && parts.length < 5) {
            let seg = current.tagName.toLowerCase();
            if (current.className && typeof current.className === 'string') {
                const classes = current.className.trim().split(/\\s+/).filter(c => !c.includes('_') || c.length < 30).slice(0, 2);
                if (classes.length > 0) seg += '.' + classes.join('.');
            }
            // Add nth-of-type if needed for uniqueness
            const parent = current.parentElement;
            if (parent) {
                const siblings = parent.querySelectorAll(':scope > ' + current.tagName.toLowerCase());
                if (siblings.length > 1) {
                    const index = Array.from(siblings).indexOf(current) + 1;
                    seg += ':nth-of-type(' + index + ')';
                }
            }
            parts.unshift(seg);
            current = current.parentElement;
        }
        return parts.join(' > ');
    }

    function getLabel(el) {
        // Direct text content (for buttons, links)
        const directText = el.textContent?.trim();
        if (directText && directText.length < 100 && directText.length > 0) {
            return directText.replace(/\\s+/g, ' ');
        }

        // aria-label
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return ariaLabel;

        // title
        if (el.title) return el.title;

        // For inputs: associated label
        if (el.id) {
            const label = document.querySelector('label[for="' + el.id + '"]');
            if (label) return label.textContent?.trim();
        }

        // Placeholder
        if (el.placeholder) return el.placeholder;

        // alt text (for images inside buttons)
        const img = el.querySelector('img');
        if (img?.alt) return img.alt;

        return '';
    }

    function inferRole(el) {
        const tag = el.tagName.toLowerCase();
        if (tag === 'a') return 'link';
        if (tag === 'button') return 'button';
        if (tag === 'input') {
            const type = el.type?.toLowerCase() || 'text';
            if (type === 'submit') return 'button';
            if (type === 'checkbox') return 'checkbox';
            if (type === 'radio') return 'radio';
            return 'textbox';
        }
        if (tag === 'select') return 'combobox';
        if (tag === 'textarea') return 'textbox';
        return el.getAttribute('role') || 'unknown';
    }

    return JSON.stringify(results);
})()
"""

# Click by selector
CLICK_SCRIPT = """
((selector) => {
    const el = document.querySelector(selector);
    if (!el) return JSON.stringify({ error: 'Element not found: ' + selector });

    // Scroll into view first
    el.scrollIntoView({ behavior: 'instant', block: 'center' });

    // Dispatch a real click event sequence
    el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
    el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }));
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

    const label = el.textContent?.trim().substring(0, 50) || el.getAttribute('aria-label') || selector;
    return JSON.stringify({ clicked: true, label, tag: el.tagName });
})('%SELECTOR%')
"""

# Fill input by selector
FILL_SCRIPT = """
((selector, value) => {
    const el = document.querySelector(selector);
    if (!el) return JSON.stringify({ error: 'Element not found: ' + selector });

    // Focus the element
    el.focus();

    // Clear existing value
    el.value = '';
    el.dispatchEvent(new Event('input', { bubbles: true }));

    // Set new value character by character (React needs this for controlled inputs)
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
    )?.set || Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
    )?.set;

    if (nativeInputValueSetter) {
        nativeInputValueSetter.call(el, value);
    } else {
        el.value = value;
    }

    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));

    return JSON.stringify({ filled: true, selector, value: el.value.substring(0, 50) });
})('%SELECTOR%', '%VALUE%')
"""

# Select option by selector and value
SELECT_SCRIPT = """
((selector, optionValue) => {
    const el = document.querySelector(selector);
    if (!el) return JSON.stringify({ error: 'Element not found: ' + selector });

    // Find the option
    let option = null;
    for (const opt of el.options) {
        if (opt.value === optionValue || opt.textContent.trim() === optionValue) {
            option = opt;
            break;
        }
    }
    if (!option) {
        const available = Array.from(el.options).map(o => o.textContent.trim());
        return JSON.stringify({ error: 'Option not found: ' + optionValue, available });
    }

    el.value = option.value;
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return JSON.stringify({ selected: true, value: option.value, text: option.textContent.trim() });
})('%SELECTOR%', '%VALUE%')
"""

# Wait for element
WAIT_SCRIPT = """
(async (selector, timeoutMs) => {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        const el = document.querySelector(selector);
        if (el && el.offsetParent !== null) {
            return JSON.stringify({ found: true, elapsed: Date.now() - start });
        }
        await new Promise(r => setTimeout(r, 200));
    }
    return JSON.stringify({ found: false, timeout: true, elapsed: timeoutMs });
})('%SELECTOR%', %TIMEOUT%)
"""


class Interactor:
    """High-level browser interaction tools."""

    async def get_interactive_elements(
        self,
        connection: CDPConnection,
        role_filter: str | None = None,
    ) -> list[dict]:
        """Extract all interactive elements on the page.

        Returns buttons, links, inputs, selects, etc. with:
          - A stable CSS selector (prefers data-testid, id, aria-label)
          - The visible label/text
          - The element's role (button, link, textbox, etc.)
          - Bounding box (x, y, width, height)
          - Current state (value, checked, disabled)

        Args:
            connection: Active CDP connection.
            role_filter: Optional filter by role ("button", "link", "textbox").

        Returns:
            List of interactive element descriptors.
        """
        result = await connection.send(
            "Runtime.evaluate",
            {"expression": EXTRACT_INTERACTIVES_SCRIPT, "returnByValue": True},
        )

        value = result.get("result", {}).get("value", "[]")
        try:
            elements = json.loads(value)
        except json.JSONDecodeError:
            return [{"error": "Failed to parse interactive elements"}]

        if role_filter:
            elements = [e for e in elements if e.get("role") == role_filter]

        return elements

    async def click_element(
        self,
        connection: CDPConnection,
        selector: str,
    ) -> dict:
        """Click an element by CSS selector.

        Scrolls the element into view, then dispatches mousedown + mouseup + click
        events (the full sequence React expects).

        Args:
            connection: Active CDP connection.
            selector: CSS selector for the element to click.

        Returns:
            Dict confirming the click or error if element not found.
        """
        script = CLICK_SCRIPT.replace("%SELECTOR%", selector.replace("'", "\\'"))

        result = await connection.send(
            "Runtime.evaluate",
            {"expression": script, "returnByValue": True},
        )

        return _parse_result(result)

    async def fill_input(
        self,
        connection: CDPConnection,
        selector: str,
        value: str,
    ) -> dict:
        """Type a value into an input or textarea.

        Handles React controlled inputs by using the native value setter
        and dispatching both input and change events.

        Args:
            connection: Active CDP connection.
            selector: CSS selector for the input element.
            value: Text to type into the input.

        Returns:
            Dict confirming the fill or error if element not found.
        """
        script = FILL_SCRIPT.replace("%SELECTOR%", selector.replace("'", "\\'")).replace(
            "%VALUE%", value.replace("'", "\\'")
        )

        result = await connection.send(
            "Runtime.evaluate",
            {"expression": script, "returnByValue": True},
        )

        return _parse_result(result)

    async def select_option(
        self,
        connection: CDPConnection,
        selector: str,
        option_value: str,
    ) -> dict:
        """Select an option from a dropdown by value or visible text.

        Args:
            connection: Active CDP connection.
            selector: CSS selector for the select element.
            option_value: The option's value attribute or visible text.

        Returns:
            Dict confirming the selection or listing available options.
        """
        script = SELECT_SCRIPT.replace("%SELECTOR%", selector.replace("'", "\\'")).replace(
            "%VALUE%", option_value.replace("'", "\\'")
        )

        result = await connection.send(
            "Runtime.evaluate",
            {"expression": script, "returnByValue": True},
        )

        return _parse_result(result)

    async def wait_for_element(
        self,
        connection: CDPConnection,
        selector: str,
        timeout_ms: int = 10000,
    ) -> dict:
        """Wait for an element to appear and be visible.

        Polls every 200ms until the element exists and has layout (offsetParent
        is not null). Useful after clicking a button that triggers a navigation
        or renders a new section.

        Args:
            connection: Active CDP connection.
            selector: CSS selector to wait for.
            timeout_ms: Maximum wait time in milliseconds (default 10s).

        Returns:
            Dict with found status and elapsed time.
        """
        script = WAIT_SCRIPT.replace("%SELECTOR%", selector.replace("'", "\\'")).replace(
            "%TIMEOUT%", str(timeout_ms)
        )

        result = await connection.send(
            "Runtime.evaluate",
            {"expression": script, "returnByValue": True, "awaitPromise": True},
        )

        return _parse_result(result)


def _parse_result(result: dict) -> dict:
    """Parse a CDP Runtime.evaluate result that returns a JSON string."""
    value = result.get("result", {}).get("value")
    if value is None:
        exc = result.get("exceptionDetails", {})
        return {"error": exc.get("text", "Evaluation failed")}

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}

    return {"value": value}
