"""Tests for RateLimitGuard."""

import time

import pytest

from promptscreen.defence.rate_limit import RateLimitGuard


class TestRateLimitGuard:
    """Test suite for RateLimitGuard."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_valid_init(self) -> None:
        g = RateLimitGuard(max_requests=10, window_seconds=30)
        assert g.max_requests == 10
        assert g.window_seconds == 30

    def test_invalid_max_requests(self) -> None:
        with pytest.raises(ValueError, match="max_requests"):
            RateLimitGuard(max_requests=0)

    def test_invalid_window_seconds(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimitGuard(window_seconds=0)

    def test_negative_window_seconds(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimitGuard(window_seconds=-1)

    def test_default_values(self) -> None:
        g = RateLimitGuard()
        assert g.max_requests == 60
        assert g.window_seconds == 60.0

    # ------------------------------------------------------------------
    # Basic allow / block behaviour
    # ------------------------------------------------------------------

    def test_first_request_allowed(self) -> None:
        guard = RateLimitGuard(max_requests=5, window_seconds=60)
        result = guard.analyse("Hello")
        assert result.is_safe is True
        assert result.confidence == 1.0

    def test_requests_within_limit_allowed(self) -> None:
        guard = RateLimitGuard(max_requests=5, window_seconds=60)
        for _ in range(5):
            result = guard.analyse("Hello")
            assert result.is_safe is True

    def test_request_exceeding_limit_blocked(self) -> None:
        guard = RateLimitGuard(max_requests=3, window_seconds=60)
        for _ in range(3):
            guard.analyse("Hello")
        # 4th request exceeds limit
        result = guard.analyse("Hello")
        assert result.is_safe is False
        assert result.confidence == 1.0

    def test_blocked_reason_mentions_rate_limit(self) -> None:
        guard = RateLimitGuard(max_requests=1, window_seconds=60)
        guard.analyse("first")
        result = guard.analyse("second")
        assert result.is_safe is False
        assert "rate limit" in result.reason.lower() or "Rate limit" in result.reason

    def test_reason_includes_counts(self) -> None:
        guard = RateLimitGuard(max_requests=2, window_seconds=60)
        result = guard.analyse("hello")
        assert "1/2" in result.reason or "1" in result.reason

    # ------------------------------------------------------------------
    # Remaining / reset helpers
    # ------------------------------------------------------------------

    def test_remaining_decrements(self) -> None:
        guard = RateLimitGuard(max_requests=5, window_seconds=60)
        assert guard.remaining() == 5
        guard.analyse("a")
        assert guard.remaining() == 4
        guard.analyse("b")
        assert guard.remaining() == 3

    def test_remaining_does_not_go_below_zero(self) -> None:
        guard = RateLimitGuard(max_requests=2, window_seconds=60)
        for _ in range(10):
            guard.analyse("x")
        assert guard.remaining() == 0

    def test_reset_clears_counter(self) -> None:
        guard = RateLimitGuard(max_requests=2, window_seconds=60)
        guard.analyse("a")
        guard.analyse("b")
        assert guard.remaining() == 0
        guard.reset()
        assert guard.remaining() == 2

    def test_after_reset_requests_are_allowed_again(self) -> None:
        guard = RateLimitGuard(max_requests=1, window_seconds=60)
        guard.analyse("a")
        assert guard.analyse("b").is_safe is False
        guard.reset()
        assert guard.analyse("c").is_safe is True

    # ------------------------------------------------------------------
    # Sliding window expiry
    # ------------------------------------------------------------------

    def test_old_requests_expire_from_window(self) -> None:
        """Requests older than window_seconds should no longer count."""
        guard = RateLimitGuard(max_requests=2, window_seconds=0.1)
        guard.analyse("a")
        guard.analyse("b")
        # Both slots used
        assert guard.analyse("c").is_safe is False
        # Wait for window to expire
        time.sleep(0.15)
        # Window has reset — new requests should be allowed
        result = guard.analyse("d")
        assert result.is_safe is True

    def test_partial_expiry_allows_more_requests(self) -> None:
        """Partial window expiry should free up exactly the right number of slots."""
        guard = RateLimitGuard(max_requests=3, window_seconds=0.2)
        guard.analyse("a")
        time.sleep(0.25)  # "a" expires
        guard.analyse("b")
        guard.analyse("c")
        # "a" has expired; only "b" and "c" are in window → 1 slot left
        assert guard.remaining() == 1
        result = guard.analyse("d")
        assert result.is_safe is True
        # Now all 3 slots used
        assert guard.remaining() == 0
        assert guard.analyse("e").is_safe is False

    # ------------------------------------------------------------------
    # Thread safety (basic smoke test)
    # ------------------------------------------------------------------

    def test_thread_safety(self) -> None:
        """Multiple threads hammering the guard should not cause data races."""
        import threading

        guard = RateLimitGuard(max_requests=50, window_seconds=60)
        results: list[bool] = []
        lock = threading.Lock()

        def hammer() -> None:
            for _ in range(10):
                r = guard.analyse("x")
                with lock:
                    results.append(r.is_safe)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 100 total requests, limit is 50 → exactly 50 safe, 50 blocked
        assert results.count(True) == 50
        assert results.count(False) == 50
