from __future__ import annotations

from video_summarizer.config import Settings
from video_summarizer.llm.base import LLMClient
from video_summarizer.llm.ollama import OllamaClient
from video_summarizer.llm.openai_compat import OpenAICompatClient
from video_summarizer.llm.openrouter import OpenRouterClient


def create_llm_client(settings: Settings) -> LLMClient:
    backend = settings.llm_backend
    if backend == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for openrouter backend")
        return OpenRouterClient(settings)
    if backend == "ollama":
        return OllamaClient(settings)
    if backend == "llamacpp":
        return OpenAICompatClient(
            settings,
            base_url=settings.llamacpp_base_url,
            model=settings.llamacpp_model,
        )
    if backend == "vllm":
        return OpenAICompatClient(
            settings,
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
