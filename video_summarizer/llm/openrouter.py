from __future__ import annotations

from dataclasses import dataclass

import httpx

from video_summarizer.config import Settings
from video_summarizer.llm.base import ChatMessage
from video_summarizer.llm.retry import RateLimitError, parse_retry_after, with_retry
from video_summarizer.summarize.chunking import parse_param_count

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class OpenRouterModel:
    id: str
    name: str
    context_length: int
    created: int
    param_count: int


def _is_free_model(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    prompt = str(pricing.get("prompt", "1"))
    completion = str(pricing.get("completion", "1"))
    if prompt == "0" and completion == "0":
        return True
    return model.get("id", "").endswith(":free")


def _supports_text(model: dict) -> bool:
    arch = model.get("architecture") or {}
    output_modalities = arch.get("output_modalities") or ["text"]
    return "text" in output_modalities


def rank_free_models(models: list[dict]) -> list[OpenRouterModel]:
    free = [m for m in models if _is_free_model(m) and _supports_text(m)]
    ranked: list[OpenRouterModel] = []
    for m in free:
        model_id = m.get("id", "")
        ranked.append(
            OpenRouterModel(
                id=model_id,
                name=m.get("name", model_id),
                context_length=int(m.get("context_length") or 0),
                created=int(m.get("created") or 0),
                param_count=parse_param_count(model_id + " " + m.get("name", "")),
            )
        )

    ranked.sort(
        key=lambda x: (x.created, x.param_count, x.context_length),
        reverse=True,
    )
    return ranked


def fetch_free_models(settings: Settings) -> list[OpenRouterModel]:
    headers = {}
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"

    with httpx.Client(timeout=60.0) as client:
        response = client.get(OPENROUTER_MODELS_URL, headers=headers)
        response.raise_for_status()
        data = response.json()

    return rank_free_models(data.get("data", []))


class OpenRouterClient:
    def __init__(self, settings: Settings, model_id: str | None = None) -> None:
        self.settings = settings
        self._models = fetch_free_models(settings)
        if not self._models:
            raise RuntimeError("No free OpenRouter models available")
        self._model_index = 0
        if model_id:
            for i, m in enumerate(self._models):
                if m.id == model_id:
                    self._model_index = i
                    break
        self._rotate_count = 0

    @property
    def model_name(self) -> str:
        return self._models[self._model_index].id

    @property
    def available_models(self) -> list[OpenRouterModel]:
        return list(self._models)

    def _rotate_model(self) -> None:
        self._rotate_count += 1
        if self._rotate_count >= len(self._models):
            raise RuntimeError("All free OpenRouter models exhausted due to rate limits")
        self._model_index = (self._model_index + 1) % len(self._models)

    def _chat_once(self, messages: list[ChatMessage], max_tokens: int) -> str:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/video-summarizer",
            "X-Title": "video-summarizer",
        }
        if self.settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openrouter_api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)

        if response.status_code == 429:
            raise RateLimitError(
                "OpenRouter rate limit",
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
            on_rotate=self._rotate_model,
        )
