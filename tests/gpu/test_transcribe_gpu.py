from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from video_summarizer.audio import extract_audio
from video_summarizer.config import Settings
from video_summarizer.transcribe import transcribe_audio


def _has_cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


@pytest.mark.gpu
def test_transcribe_gpu_smoke(tmp_path: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    if not _has_cuda():
        pytest.skip("CUDA not available")

    data_dir = Path(__file__).resolve().parents[2] / "data"
    videos = list(data_dir.glob("*.mp4"))
    if not videos:
        pytest.skip("No test videos in data/")

    audio_path = tmp_path / "clip.wav"
    extract_audio(videos[0], audio_path, max_duration=5)

    settings = Settings(
        whisper_model="tiny",
        whisper_device="cuda",
        whisper_compute_type="float16",
    )
    transcript_path = tmp_path / "transcript.json"
    transcript = transcribe_audio(audio_path, transcript_path, settings)

    assert transcript_path.exists()
    assert isinstance(transcript.text, str)
