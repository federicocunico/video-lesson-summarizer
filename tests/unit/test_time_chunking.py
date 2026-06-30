from __future__ import annotations

from video_summarizer.summarize.chunking import (
    _sample_fractions,
    split_transcript_by_time,
)
from video_summarizer.transcribe import TranscriptSegment


def test_split_transcript_by_time():
    segments = [
        TranscriptSegment(i * 30.0, i * 30.0 + 25.0, f"segmento {i} del tutorial")
        for i in range(40)  # ~20 min
    ]
    chunks = split_transcript_by_time(
        segments,
        window_sec=300.0,
        overlap_sec=30.0,
        max_chunk_chars=5000,
    )
    assert len(chunks) >= 3
    assert chunks[0].start_time == 0.0
    assert all(c.text for c in chunks)
    assert chunks[0].index == 0


def test_split_oversized_time_window():
    long_text = "parola " * 3000
    segments = [
        TranscriptSegment(0.0, 300.0, long_text),
    ]
    chunks = split_transcript_by_time(
        segments,
        window_sec=300.0,
        max_chunk_chars=1000,
        char_overlap=50,
    )
    assert len(chunks) > 1


def test_sample_fractions_short_video():
    points = _sample_fractions(10 * 60)
    labels = [p[0] for p in points]
    assert "inizio" in labels
    assert "fine" in labels
    assert len(points) == 5


def test_sample_fractions_long_video():
    points = _sample_fractions(60 * 60)
    labels = [p[0] for p in points]
    assert "10%" in labels
    assert "90%" in labels
    assert len(points) >= 11
