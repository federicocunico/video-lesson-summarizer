from __future__ import annotations

from typing import Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def chat(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> str:
        ...
