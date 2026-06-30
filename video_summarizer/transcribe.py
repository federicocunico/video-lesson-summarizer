from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import os

from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from video_summarizer.config import Settings

# Avoid Rich braille spinner encoding errors on Windows cp1252 consoles.
console = Console(legacy_windows=os.name == "nt")


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str | None
    duration: float | None
    text: str
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "text": self.text,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        return cls(
            language=data.get("language"),
            duration=data.get("duration"),
            text=data["text"],
            segments=[
                TranscriptSegment(**s) for s in data.get("segments", [])
            ],
        )


def _resolve_device(settings: Settings) -> tuple[str, str]:
    device = settings.whisper_device
    compute_type = settings.whisper_compute_type

    if device == "auto":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
            else:
                device = "cpu"
                compute_type = "int8"
        except Exception:
            device = "cpu"
            compute_type = "int8"

    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    return device, compute_type


def load_whisper_model(settings: Settings) -> WhisperModel:
    device, compute_type = _resolve_device(settings)
    console.print(f"[dim]Loading Whisper {settings.whisper_model} ({device}, {compute_type})[/dim]")
    try:
        return WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        if device == "cuda":
            console.print(f"[yellow]CUDA load failed ({exc}), falling back to int8_float16[/yellow]")
            try:
                return WhisperModel(
                    settings.whisper_model,
                    device="cuda",
                    compute_type="int8_float16",
                )
            except Exception:
                console.print("[yellow]Falling back to CPU int8[/yellow]")
                return WhisperModel(
                    settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
        raise


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    settings: Settings,
) -> Transcript:
    model = load_whisper_model(settings)
    segments_iter, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=5,
    )

    segments: list[TranscriptSegment] = []
    progress_columns: list = [TextColumn("[progress.description]{task.description}"), TimeElapsedColumn()]
    if os.name != "nt":
        progress_columns.insert(0, SpinnerColumn())

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("Transcribing...", total=None)
        for seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                )
            )
            progress.update(task, description=f"Transcribing... {len(segments)} segments")

    full_text = " ".join(s.text for s in segments if s.text)
    transcript = Transcript(
        language=info.language,
        duration=info.duration,
        text=full_text,
        segments=segments,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return transcript


def load_transcript(path: Path) -> Transcript:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript.from_dict(data)


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript_plain(transcript: Transcript) -> str:
    lines: list[str] = []
    if transcript.language:
        lines.append(f"Lingua: {transcript.language}")
    if transcript.duration is not None:
        lines.append(f"Durata: {format_timestamp(transcript.duration)}")
    if lines:
        lines.append("")

    lines.append("=== Testo completo ===")
    lines.append("")
    lines.append(transcript.text.strip() or "(vuoto)")
    lines.append("")
    lines.append("=== Segmenti con timestamp ===")
    lines.append("")

    for segment in transcript.segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        if text:
            lines.append(f"[{start} – {end}] {text}")

    return "\n".join(lines)


def save_transcript_markdown(transcript: Transcript, path: Path) -> None:
    parts = ["# Trascrizione completa", ""]
    if transcript.language:
        parts.append(f"**Lingua:** {transcript.language}")
    if transcript.duration is not None:
        parts.append(f"**Durata:** {format_timestamp(transcript.duration)}")
    parts.extend(["", "## Testo completo", "", transcript.text.strip() or "_(vuoto)_", ""])
    parts.extend(["## Segmenti", ""])

    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        parts.append(f"### [{start} – {end}]")
        parts.append("")
        parts.append(text)
        parts.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
