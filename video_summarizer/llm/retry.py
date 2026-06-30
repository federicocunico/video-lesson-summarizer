from __future__ import annotations

import random
import time
from collections.abc import Callable

import httpx
from rich.console import Console

console = Console()


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(
    fn: Callable[[], str],
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    on_rotate: Callable[[], None] | None = None,
) -> str:
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as exc:
            attempt += 1
            if attempt > max_retries:
                if on_rotate:
                    console.print("[yellow]Rate limit exceeded, rotating model...[/yellow]")
                    on_rotate()
                    attempt = 0
                    continue
                raise
            delay = exc.retry_after if exc.retry_after else min(
                max_delay, base_delay * (2 ** (attempt - 1))
            )
            jitter = random.uniform(0, delay * 0.1)
            wait = delay + jitter
            console.print(f"[dim]429 rate limit, waiting {wait:.1f}s (attempt {attempt})[/dim]")
            time.sleep(wait)


def parse_retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
