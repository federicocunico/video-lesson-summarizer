FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WHISPER_DEVICE=cuda \
    WHISPER_MODEL=large-v3 \
    WHISPER_COMPUTE_TYPE=float16 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml README.md ./
COPY video_summarizer ./video_summarizer

RUN uv pip install --system -e ".[dev]"

COPY docker/healthcheck.py /app/docker/healthcheck.py

HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=3 \
    CMD python3 /app/docker/healthcheck.py

ENTRYPOINT ["video-summarizer"]
CMD ["--help"]
