"""Network activity monitoring via CDP.

Subscribes to Network events, tracks HTTP requests and responses,
and surfaces failed requests (4xx/5xx) for debugging.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from specter.browser.connection import CDPConnection
from specter.config import SpecterConfig


@dataclass
class NetworkEntry:
    """A tracked HTTP request/response pair."""

    request_id: str
    timestamp: float
    method: str
    url: str
    status: int | None = None
    status_text: str | None = None
    response_headers: dict[str, str] | None = None
    error_text: str | None = None
    duration_ms: float | None = None
    _start_time: float = 0.0

    @property
    def is_error(self) -> bool:
        return (self.status is not None and self.status >= 400) or self.error_text is not None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "method": self.method,
            "url": self.url,
            "status": self.status,
            "status_text": self.status_text,
            "duration_ms": self.duration_ms,
        }
        if self.error_text:
            d["error"] = self.error_text
        return d


class NetworkCapture:
    """Captures and buffers browser network events."""

    def __init__(self, config: SpecterConfig) -> None:
        self._buffer: deque[NetworkEntry] = deque(maxlen=config.max_buffer_size)
        self._inflight: dict[str, NetworkEntry] = {}

    def register(self, connection: CDPConnection) -> None:
        """Register CDP event handlers for network capture."""
        connection.on("Network.requestWillBeSent", self._on_request)
        connection.on("Network.responseReceived", self._on_response)
        connection.on("Network.loadingFailed", self._on_failed)

    async def enable(self, connection: CDPConnection) -> None:
        """Enable the Network domain."""
        await connection.send("Network.enable")

    def get_requests(
        self,
        errors_only: bool = False,
        since: float | None = None,
        limit: int = 50,
        url_filter: str | None = None,
    ) -> list[dict]:
        """Retrieve buffered network entries.

        Args:
            errors_only: Only return 4xx/5xx and failed requests.
            since: Only entries after this Unix timestamp.
            limit: Max entries to return.
            url_filter: Only URLs containing this substring.

        Returns:
            List of network entry dicts, newest first.
        """
        entries = list(self._buffer)

        if errors_only:
            entries = [e for e in entries if e.is_error]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        if url_filter:
            entries = [e for e in entries if url_filter in e.url]

        return [e.to_dict() for e in entries[-limit:]]

    def clear(self) -> int:
        """Clear the buffer. Returns entries cleared."""
        count = len(self._buffer)
        self._buffer.clear()
        self._inflight.clear()
        return count

    def _on_request(self, params: dict) -> None:
        """Handle Network.requestWillBeSent."""
        request = params.get("request", {})
        request_id = params.get("requestId", "")

        entry = NetworkEntry(
            request_id=request_id,
            timestamp=time.time(),
            method=request.get("method", "GET"),
            url=request.get("url", ""),
            _start_time=time.time(),
        )
        self._inflight[request_id] = entry

    def _on_response(self, params: dict) -> None:
        """Handle Network.responseReceived."""
        request_id = params.get("requestId", "")
        response = params.get("response", {})

        entry = self._inflight.pop(request_id, None)
        if entry is None:
            return

        entry.status = response.get("status")
        entry.status_text = response.get("statusText")
        entry.duration_ms = round((time.time() - entry._start_time) * 1000, 1)

        self._buffer.append(entry)

    def _on_failed(self, params: dict) -> None:
        """Handle Network.loadingFailed."""
        request_id = params.get("requestId", "")

        entry = self._inflight.pop(request_id, None)
        if entry is None:
            return

        entry.error_text = params.get("errorText", "Unknown error")
        entry.duration_ms = round((time.time() - entry._start_time) * 1000, 1)

        self._buffer.append(entry)
