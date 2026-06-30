from __future__ import annotations

import re
from dataclasses import dataclass

from video_summarizer.transcribe import Transcript, TranscriptSegment


@dataclass
class TextChunk:
    index: int
    text: str
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class TextSample:
    label: str
    text: str
    start_time: float | None = None
    end_time: float | None = None


def split_transcript(
    text: str,
    *,
    chunk_size: int = 3500,
    overlap: int = 200,
) -> list[TextChunk]:
    if not text.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(TextChunk(index=index, text=chunk_text))
            index += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _enrich_chunks_with_timestamps(
    full_text: str,
    segments: list[TranscriptSegment],
    chunks: list[TextChunk],
) -> list[TextChunk]:
    if not segments:
        return chunks

    char_offsets: list[tuple[int, int, TranscriptSegment]] = []
    pos = 0
    for seg in segments:
        if not seg.text:
            continue
        start_pos = pos
        pos += len(seg.text) + 1
        char_offsets.append((start_pos, pos, seg))

    enriched: list[TextChunk] = []
    cursor = 0
    for chunk in chunks:
        chunk_start = full_text.find(chunk.text, cursor)
        if chunk_start < 0:
            chunk_start = cursor
        chunk_end = chunk_start + len(chunk.text)
        cursor = chunk_start + 1

        start_time: float | None = None
        end_time: float | None = None
        for c_start, c_end, seg in char_offsets:
            if c_end <= chunk_start:
                continue
            if c_start >= chunk_end:
                break
            if start_time is None:
                start_time = seg.start
            end_time = seg.end

        enriched.append(
            TextChunk(
                index=chunk.index,
                text=chunk.text,
                start_time=start_time,
                end_time=end_time,
            )
        )

    return enriched


def split_transcript_with_timestamps(
    segments: list[TranscriptSegment],
    *,
    chunk_size: int = 3500,
    overlap: int = 200,
) -> list[TextChunk]:
    full_text = " ".join(s.text for s in segments if s.text)
    chunks = split_transcript(full_text, chunk_size=chunk_size, overlap=overlap)
    return _enrich_chunks_with_timestamps(full_text, segments, chunks)


def _split_oversized_chunk(
    chunk: TextChunk,
    *,
    max_chars: int,
    overlap: int,
) -> list[TextChunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]

    sub_chunks = split_transcript(
        chunk.text,
        chunk_size=max_chars,
        overlap=overlap,
    )
    return [
        TextChunk(
            index=chunk.index,
            text=sub.text,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
        )
        for sub in sub_chunks
    ]


def split_transcript_by_time(
    segments: list[TranscriptSegment],
    *,
    window_sec: float = 300.0,
    overlap_sec: float = 30.0,
    max_chunk_chars: int = 8000,
    char_overlap: int = 200,
) -> list[TextChunk]:
    """Group Whisper segments into time windows (~5 min) with overlap."""
    active_segments = [s for s in segments if s.text.strip()]
    if not active_segments:
        return []

    duration = active_segments[-1].end
    if duration <= 0:
        return split_transcript_with_timestamps(active_segments)

    chunks: list[TextChunk] = []
    index = 0
    window_start = 0.0

    while window_start < duration:
        window_end = min(window_start + window_sec, duration)
        window_segments = [
            s
            for s in active_segments
            if s.end > window_start and s.start < window_end
        ]
        if window_segments:
            text = " ".join(s.text for s in window_segments).strip()
            if text:
                chunk = TextChunk(
                    index=index,
                    text=text,
                    start_time=window_segments[0].start,
                    end_time=window_segments[-1].end,
                )
                for part in _split_oversized_chunk(
                    chunk,
                    max_chars=max_chunk_chars,
                    overlap=char_overlap,
                ):
                    part.index = index
                    chunks.append(part)
                    index += 1

        if window_end >= duration:
            break
        window_start = max(0.0, window_end - overlap_sec)

    # Re-index sequentially
    for i, chunk in enumerate(chunks):
        chunk.index = i

    return chunks


def _segments_in_window(
    segments: list[TranscriptSegment],
    start_time: float,
    end_time: float,
    char_limit: int,
) -> str:
    parts: list[str] = []
    total = 0
    for seg in segments:
        if seg.end < start_time:
            continue
        if seg.start > end_time:
            break
        if not seg.text:
            continue
        parts.append(seg.text)
        total += len(seg.text) + 1
        if total >= char_limit:
            break
    return " ".join(parts).strip()


def _sample_fractions(duration: float) -> list[tuple[str, float]]:
    """Adaptive sampling: more points for longer videos."""
    base = [
        ("inizio", 0.0),
        ("25%", 0.25),
        ("50%", 0.50),
        ("75%", 0.75),
        ("fine", 1.0),
    ]
    if duration <= 30 * 60:
        return [(label, duration * frac) for label, frac in base]

    # >30 min: sample every 10%
    points: list[tuple[str, float]] = [("inizio", 0.0)]
    for pct in range(10, 100, 10):
        points.append((f"{pct}%", duration * pct / 100))
    points.append(("fine", duration))
    return points


def sample_transcript_points(
    transcript: Transcript,
    *,
    char_limit: int = 2000,
) -> list[TextSample]:
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        if transcript.text.strip():
            return [TextSample(label="full", text=transcript.text[:char_limit])]
        return []

    duration = transcript.duration or segments[-1].end
    if duration <= 0:
        duration = segments[-1].end

    labels_and_times = _sample_fractions(duration)
    samples: list[TextSample] = []
    seen_texts: set[str] = set()
    window = min(120.0, max(60.0, duration * 0.02))

    for label, target_time in labels_and_times:
        if label == "fine":
            target_time = segments[-1].start
        text = _segments_in_window(
            segments,
            max(0.0, target_time - window / 2),
            target_time + window / 2,
            char_limit,
        )
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        samples.append(
            TextSample(
                label=label,
                text=text[:char_limit],
                start_time=max(0.0, target_time - window / 2),
                end_time=target_time + window / 2,
            )
        )

    return samples


def parse_param_count(model_id: str) -> int:
    """Extract parameter count from model slug (e.g. 70b -> 70)."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", model_id.lower())
    if match:
        return int(float(match.group(1)))
    return 0
