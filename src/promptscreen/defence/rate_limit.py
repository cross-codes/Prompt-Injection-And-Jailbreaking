"""Rate-limit guard — per-instance sliding-window request throttle.

Automated jailbreak probing sends thousands of prompt variations per minute.
This guard tracks the rate of calls to *this guard instance* and blocks once
the configured request budget is exhausted within a rolling time window.

The guard is **stateful** and **thread-safe**.  Each instance maintains its
own counter, so sharing one instance across all API requests (the typical
FastAPI singleton pattern) gives per-server rate limiting.  If you need
per-user or per-session limiting, create a separate guard instance per
session and dispose of it when the session ends.
"""

import logging
import threading
import time
from collections import deque

from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class RateLimitGuard(AbstractDefence):
    """Guard that enforces a sliding-window request rate limit.

    Parameters
    ----------
    max_requests:
        Maximum number of prompts allowed within ``window_seconds``.
        Default: 60 requests.
    window_seconds:
        Length of the rolling time window in seconds.
        Default: 60 seconds (i.e. 60 requests per minute).
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def remaining(self) -> int:
        """Return how many requests remain in the current window."""
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            return max(0, self.max_requests - len(self._timestamps))

    def reset(self) -> None:
        """Clear all recorded timestamps (useful in tests)."""
        with self._lock:
            self._timestamps.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _purge(self, now: float) -> None:
        """Remove timestamps that have fallen outside the current window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    # ------------------------------------------------------------------
    # AbstractDefence
    # ------------------------------------------------------------------

    @override
    def analyse(self, query: str) -> AnalysisResult:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            current_count = len(self._timestamps)

            if current_count >= self.max_requests:
                logger.warning(
                    "Rate limit exceeded: %d requests in %.0fs window (limit=%d)",
                    current_count,
                    self.window_seconds,
                    self.max_requests,
                )
                return AnalysisResult(
                    reason=(
                        f"Rate limit exceeded: {current_count} requests received within "
                        f"the last {self.window_seconds:.0f}s window "
                        f"(limit: {self.max_requests}). "
                        "Automated probing or abuse suspected."
                    ),
                    is_safe=False,
                    confidence=1.0,
                )

            # Record this request *after* the limit check so a request that
            # hits the limit is not counted (it was rejected).
            self._timestamps.append(now)

        return AnalysisResult(
            reason=(
                f"Request {current_count + 1}/{self.max_requests} "
                f"within the {self.window_seconds:.0f}s window."
            ),
            is_safe=True,
            confidence=1.0,
        )
