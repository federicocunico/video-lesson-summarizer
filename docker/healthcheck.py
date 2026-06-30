"""Docker healthcheck: verify CUDA and faster-whisper can load."""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        import ctranslate2

        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count < 1:
            print("No CUDA devices found", file=sys.stderr)
            return 1
        print(f"CUDA devices: {cuda_count}")
    except Exception as exc:
        print(f"CUDA check failed: {exc}", file=sys.stderr)
        return 1

    model_size = os.environ.get("WHISPER_HEALTHCHECK_MODEL", "tiny")
    device = os.environ.get("WHISPER_DEVICE", "cuda")

    try:
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"
        WhisperModel(model_size, device=device, compute_type=compute)
        print(f"Whisper model '{model_size}' loaded on {device}")
    except Exception as exc:
        print(f"Whisper load failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
