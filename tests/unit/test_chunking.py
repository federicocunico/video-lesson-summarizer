from __future__ import annotations

from video_summarizer.summarize.chunking import (
    parse_param_count,
    sample_transcript_points,
    split_transcript,
    split_transcript_with_timestamps,
)
from video_summarizer.transcribe import Transcript, TranscriptSegment


def test_split_transcript_overlap():
    text = "word " * 500
    chunks = split_transcript(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)
    assert chunks[0].index == 0


def test_split_transcript_empty():
    assert split_transcript("") == []
    assert split_transcript("   ") == []


def test_split_with_timestamps():
    segments = [
        TranscriptSegment(0.0, 5.0, "Hello world"),
        TranscriptSegment(5.0, 10.0, "this is a test"),
        TranscriptSegment(10.0, 20.0, "of timestamp chunking"),
    ]
    chunks = split_transcript_with_timestamps(segments, chunk_size=50, overlap=5)
    assert len(chunks) >= 1
    assert chunks[0].start_time == 0.0


def test_sample_transcript_points_distribution():
    segments = [
        TranscriptSegment(i * 10.0, i * 10.0 + 5.0, f"segment {i} content here")
        for i in range(20)
    ]
    transcript = Transcript(
        language="it",
        duration=200.0,
        text=" ".join(s.text for s in segments),
        segments=segments,
    )
    samples = sample_transcript_points(transcript, char_limit=100)
    labels = {s.label for s in samples}
    assert "inizio" in labels
    assert "fine" in labels
    assert len(samples) >= 3


def test_parse_param_count():
    assert parse_param_count("meta-llama/llama-3.3-70b-instruct:free") == 70
    assert parse_param_count("qwen/qwen3-8b") == 8
    assert parse_param_count("some-model") == 0
