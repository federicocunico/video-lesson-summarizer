from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegNotFoundError(
            "ffmpeg not found in PATH. Install ffmpeg or use the Docker image."
        )
    return ffmpeg


def extract_audio(
    video_path: Path,
    output_path: Path,
    *,
    max_duration: int | None = None,
) -> Path:
    """Extract mono 16kHz PCM WAV from video via ffmpeg."""
    ffmpeg = ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
    ]
    if max_duration is not None:
        cmd.extend(["-t", str(max_duration)])
    cmd.extend(
        [
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path
