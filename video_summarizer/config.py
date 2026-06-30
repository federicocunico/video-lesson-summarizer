from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = ""
    llm_backend: Literal["openrouter", "ollama", "llamacpp", "vllm"] = "openrouter"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:32b"
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.3

    llamacpp_base_url: str = "http://localhost:8080"
    llamacpp_model: str = "default"

    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "default"

    whisper_model: str = "large-v3"
    whisper_device: Literal["cuda", "cpu", "auto"] = "cuda"
    whisper_compute_type: str = "float16"

    summary_language: str = "it"
    output_dir: Path = Path("output")

    chunk_size: int = 3500
    chunk_overlap: int = 200
    sample_char_limit: int = 2000
    segment_duration_sec: int = 300
    segment_overlap_sec: int = 30
    max_chunk_chars: int = 8000

    reduce_batch_size: int = 6
    reduce_max_input_chars: int = 12000

    llm_max_tokens: int = 2048
    map_max_tokens: int = 4096
    llm_retry_max: int = 5
    llm_retry_base_delay: float = 2.0
    llm_retry_max_delay: float = 60.0


def get_settings() -> Settings:
    return Settings()


def job_dir(video_path: Path, output_dir: Path | None = None) -> Path:
    base = output_dir or get_settings().output_dir
    return base / video_path.stem
