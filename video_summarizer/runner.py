from __future__ import annotations

from pathlib import Path

from rich.console import Console

from video_summarizer.audio import extract_audio
from video_summarizer.checkpoint import CheckpointManager
from video_summarizer.config import Settings, get_settings, job_dir
from video_summarizer.llm import create_llm_client
from video_summarizer.summarize.pipeline import run_summarize_pipeline
from video_summarizer.transcribe import load_transcript, transcribe_audio

console = Console()


def process_video(
    video_path: Path,
    *,
    settings: Settings | None = None,
    max_duration: int | None = None,
    force: bool = False,
    output_dir: Path | None = None,
) -> Path:
    settings = settings or get_settings()
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir = job_dir(video_path, output_dir or settings.output_dir)
    checkpoint = CheckpointManager(work_dir)

    if not checkpoint.state.video_path:
        checkpoint.update(video_path=str(video_path))

    if max_duration is not None:
        checkpoint.update(max_duration=max_duration)
    elif force:
        checkpoint.update(max_duration=None)

    clip_duration = (
        max_duration
        if max_duration is not None
        else (None if force else checkpoint.state.max_duration)
    )

    audio_path = work_dir / "audio.wav"
    if checkpoint.state.audio_done and audio_path.exists() and not force:
        console.print("[green]Audio already extracted, skipping[/green]")
    else:
        console.print(f"[bold]Extracting audio from {video_path.name}[/bold]")
        if clip_duration:
            console.print(f"[dim]Clip limit: {clip_duration}s[/dim]")
        extract_audio(
            video_path,
            audio_path,
            max_duration=clip_duration,
        )
        checkpoint.update(audio_done=True, audio_path=str(audio_path))

    transcript_path = work_dir / "transcript.json"
    if checkpoint.state.transcript_done and transcript_path.exists() and not force:
        console.print("[green]Transcript exists, skipping[/green]")
        transcript = load_transcript(transcript_path)
    else:
        console.print("[bold]Transcribing audio with Whisper[/bold]")
        transcript = transcribe_audio(audio_path, transcript_path, settings)
        checkpoint.update(
            transcript_done=True,
            transcript_path=str(transcript_path),
        )

    console.print(
        f"[dim]Transcript: {len(transcript.text)} chars, "
        f"{len(transcript.segments)} segments, lang={transcript.language}[/dim]"
    )

    if not transcript.text.strip():
        console.print("[yellow]Empty transcript, skipping summarization[/yellow]")
        return work_dir

    console.print("[bold]Generazione appunti di lezione[/bold]")
    llm = create_llm_client(settings)
    notes = run_summarize_pipeline(
        transcript,
        work_dir,
        llm,
        settings,
        checkpoint,
        force=force,
        video_name=video_path.name,
    )

    console.print(f"[green]Fatto! Appunti salvati in {work_dir / 'appunti.md'}[/green]")
    console.print(notes[:500] + ("..." if len(notes) > 500 else ""))
    return work_dir


def resume_job(
    work_dir: Path,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> Path:
    settings = settings or get_settings()
    work_dir = work_dir.resolve()
    checkpoint = CheckpointManager(work_dir)

    if not checkpoint.state.video_path:
        raise RuntimeError(f"No checkpoint video_path in {work_dir}")

    video_path = Path(checkpoint.state.video_path)
    return process_video(
        video_path,
        settings=settings,
        max_duration=checkpoint.state.max_duration,
        force=force,
        output_dir=work_dir.parent,
    )
