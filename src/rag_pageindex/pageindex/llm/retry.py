import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    delay_s: float,
) -> T:
    """Call a function with exponential backoff retry logic.

    Retries on any exception, with a fixed delay between attempts.
    `max_retries` is the total attempt count; values < 1 are clamped to 1
    (one attempt, no retries). Re-raises the final exception.

    Args:
        fn: Callable with no arguments to invoke.
        max_retries: Total number of attempts (not additional retries).
        delay_s: Delay in seconds between attempts.

    Returns:
        Result of fn() if successful.

    Raises:
        Final exception from fn() if all attempts fail.
    """
    attempts = max(1, max_retries)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM call failed (attempt {}/{}): {}",
                attempt + 1,
                attempts,
                exc,
            )
            if attempt < attempts - 1:
                time.sleep(delay_s)
    assert last_exc is not None
    raise last_exc


async def awith_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    delay_s: float,
) -> T:
    """Asynchronously call a function with retry logic.

    Async variant of with_retries(). Retries on any exception with a
    fixed delay between attempts. Re-raises the final exception.

    Args:
        fn: Async callable with no arguments to invoke.
        max_retries: Total number of attempts (not additional retries).
        delay_s: Delay in seconds between attempts.

    Returns:
        Result of fn() if successful.

    Raises:
        Final exception from fn() if all attempts fail.
    """
    attempts = max(1, max_retries)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Async LLM call failed (attempt {}/{}): {}",
                attempt + 1,
                attempts,
                exc,
            )
            if attempt < attempts - 1:
                await asyncio.sleep(delay_s)
    assert last_exc is not None
    raise last_exc
