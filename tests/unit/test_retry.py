from __future__ import annotations

import pytest

from video_summarizer.llm.retry import RateLimitError, with_retry


def test_with_retry_success():
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    assert with_retry(fn, max_retries=3) == "ok"
    assert calls["n"] == 1


def test_with_retry_eventual_success():
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("rate limit", retry_after=0.01)
        return "ok"

    assert with_retry(fn, max_retries=5, base_delay=0.01, max_delay=0.05) == "ok"
    assert calls["n"] == 3


def test_with_retry_rotate():
    calls = {"n": 0}
    rotated = {"v": False}

    def fn() -> str:
        calls["n"] += 1
        if rotated["v"]:
            return "after rotate"
        raise RateLimitError("rate limit", retry_after=0.01)

    def on_rotate() -> None:
        rotated["v"] = True

    result = with_retry(
        fn,
        max_retries=1,
        base_delay=0.01,
        on_rotate=on_rotate,
    )
    assert result == "after rotate"
    assert rotated["v"] is True
