from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from video_summarizer.audio import FFmpegNotFoundError, extract_audio, ensure_ffmpeg


@pytest.mark.ffmpeg
def test_ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    assert ensure_ffmpeg()


def test_ensure_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr("video_summarizer.audio.shutil.which", lambda _: None)
    with pytest.raises(FFmpegNotFoundError):
        ensure_ffmpeg()


@pytest.mark.ffmpeg
def test_extract_audio_from_video(tmp_path: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    data_dir = Path(__file__).resolve().parents[2] / "data"
    videos = list(data_dir.glob("*.mp4"))
    if not videos:
        pytest.skip("No test videos in data/")

    video = videos[0]
    out = tmp_path / "audio.wav"
    extract_audio(video, out, max_duration=5)

    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
