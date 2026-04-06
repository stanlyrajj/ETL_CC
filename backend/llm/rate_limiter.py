"""
rate_limiter.py — Token bucket rate limiter for LLM API calls.

Configured via .env:
  LLM_CALLS_PER_MINUTE  — max calls per 60-second window (default: 10)

Usage:
  from llm.rate_limiter import llm_rate_limiter
  await llm_rate_limiter.acquire()   # call before every LLM API call

If the bucket is empty, acquire() sleeps until a token is available
rather than raising an error. This prevents cascading 429s from
providers during heavy operations like Technical analysis (5 rapid calls)
or batch paper processing.

Thread-safe: uses asyncio.Lock so concurrent coroutines queue correctly.
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """
    Token bucket rate limiter.

    Tokens refill at rate = calls_per_minute / 60 per second.
    Bucket capacity = calls_per_minute (allows a full minute's worth
    of calls in a burst before throttling kicks in).
    """

    def __init__(self, calls_per_minute: int):
        self._rate      = max(1, calls_per_minute) / 60.0   # tokens per second
        self._capacity  = float(max(1, calls_per_minute))
        self._tokens    = self._capacity
        self._last_refill = time.monotonic()
        self._lock      = asyncio.Lock()

    def _refill(self) -> None:
        now     = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """
        Wait until a token is available, then consume one.
        Logs a warning if it has to wait so operators can tune the limit.
        """
        async with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Calculate wait time and sleep
            deficit    = 1.0 - self._tokens
            wait_secs  = deficit / self._rate
            logger.warning(
                "LLM rate limit: waiting %.1fs before next API call "
                "(bucket empty, rate=%.2f calls/s). "
                "Increase LLM_CALLS_PER_MINUTE in .env to reduce waits.",
                wait_secs, self._rate,
            )
            await asyncio.sleep(wait_secs)
            self._refill()
            self._tokens -= 1.0


def _get_calls_per_minute() -> int:
    raw = os.getenv("LLM_CALLS_PER_MINUTE", "10")
    try:
        v = int(raw)
        return max(1, v)
    except ValueError:
        logger.warning("Invalid LLM_CALLS_PER_MINUTE=%r — using default 10", raw)
        return 10


# Shared singleton — import this everywhere LLM calls are made
llm_rate_limiter = TokenBucketLimiter(calls_per_minute=_get_calls_per_minute())
