from __future__ import annotations

import httpx

from video_summarizer.config import Settings
from video_summarizer.llm.base import ChatMessage
from video_summarizer.llm.retry import RateLimitError, parse_retry_after, with_retry


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    @property
    def model_name(self) -> str:
        return self._model

    def _chat_once(self, messages: list[ChatMessage], max_tokens: int) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": self.settings.ollama_num_ctx,
                "temperature": self.settings.ollama_temperature,
            },
        }
        url = f"{self.base_url}/api/chat"

        with httpx.Client(timeout=600.0) as client:
            response = client.post(url, json=payload)

        if response.status_code == 429:
            raise RateLimitError(
                "Ollama rate limit",
                retry_after=parse_retry_after(response.headers),
            )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

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
