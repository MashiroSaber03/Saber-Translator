"""Small in-process limiter for failed public authentication attempts."""

from __future__ import annotations

from collections import defaultdict, deque
import threading
import time

MAX_BUCKETS = 10_000


class AuthRateLimited(RuntimeError):
    pass


class FailedAuthLimiter:
    def __init__(self, *, window_seconds: int = 600) -> None:
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, values: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while values and values[0] <= cutoff:
            values.popleft()

    def _prune(self, now: float, *, reserve: int = 0) -> None:
        # Keep the common path O(1). A full expiry sweep is only useful when the
        # bounded map is actually close to capacity.
        if len(self._attempts) >= MAX_BUCKETS - reserve:
            for key in list(self._attempts):
                values = self._attempts[key]
                self._trim(values, now)
                if not values:
                    del self._attempts[key]
        while len(self._attempts) > MAX_BUCKETS - reserve:
            self._attempts.pop(next(iter(self._attempts)))

    def check(self, *, route: str, client_ip: str, username: str) -> None:
        now = time.monotonic()
        keys = (
            (f"{route}:{client_ip}:{username}", 8),
            (f"{route}:{client_ip}:*", 40),
        )
        with self._lock:
            self._prune(now, reserve=len(keys))
            for key, limit in keys:
                values = self._attempts[key]
                self._trim(values, now)
                if len(values) >= limit:
                    raise AuthRateLimited("尝试次数过多，请稍后再试")

    def record_failure(self, *, route: str, client_ip: str, username: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(now, reserve=2)
            for key in (
                f"{route}:{client_ip}:{username}",
                f"{route}:{client_ip}:*",
            ):
                values = self._attempts[key]
                self._trim(values, now)
                values.append(now)

    def clear_user(self, *, route: str, client_ip: str, username: str) -> None:
        with self._lock:
            self._attempts.pop(f"{route}:{client_ip}:{username}", None)
