from __future__ import annotations

import httpx

from video_summarizer.config import Settings
from video_summarizer.llm.base import ChatMessage
from video_summarizer.llm.retry import RateLimitError, parse_retry_after, with_retry


class OpenAICompatClient:
    """Client for OpenAI-compatible APIs (llama.cpp, vLLM)."""

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
    ) -> None:
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model

    def _chat_once(self, messages: list[ChatMessage], max_tokens: int) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        url = f"{self.base_url}/v1/chat/completions"

        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code == 429:
            raise RateLimitError(
                "Rate limit",
                retry_after=parse_retry_after(response.headers),
            )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> str:
        tokens = max_tokens or self.settings.llm_max_tokens

        def call() -> str:
            return self._chat_once(messages, tokens)

        return with_retry(
            call,
            max_retries=self.settings.llm_retry_max,
            base_delay=self.settings.llm_retry_base_delay,
            max_delay=self.settings.llm_retry_max_delay,
        )
