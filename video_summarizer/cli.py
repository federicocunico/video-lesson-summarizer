from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from video_summarizer.config import Settings, get_settings
from video_summarizer.llm.openrouter import fetch_free_models
from video_summarizer.runner import process_video, resume_job

app = typer.Typer(
    name="video-summarizer",
    help="Extract audio, transcribe with Whisper, generate lesson notes.",
)
models_app = typer.Typer(help="Manage LLM models")
app.add_typer(models_app, name="models")

console = Console()


@app.command()
def gui() -> None:
    """Launch the desktop GUI."""
    from video_summarizer.gui import run_gui

    run_gui()


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to input video (mp4)"),
    max_duration: Optional[int] = typer.Option(
        None,
        "--max-duration",
        "-t",
        help="Extract/transcribe only first N seconds",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-run all steps even if checkpoints exist",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: output/)",
    ),
) -> None:
    """Process a video: extract audio, transcribe, generate lesson notes (appunti)."""
    settings = get_settings()
    if output_dir:
        settings = Settings(**{**settings.model_dump(), "output_dir": output_dir})

    try:
        work_dir = process_video(
            video,
            settings=settings,
            max_duration=max_duration,
            force=force,
            output_dir=output_dir,
        )
        console.print(f"[bold green]Output:[/bold green] {work_dir}")
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def resume(
    work_dir: Path = typer.Argument(..., help="Job output directory to resume"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-run steps"),
) -> None:
    """Resume an interrupted job from checkpoint."""
    settings = get_settings()
    try:
        result = resume_job(work_dir, settings=settings, force=force)
        console.print(f"[bold green]Resumed:[/bold green] {result}")
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


@models_app.command("list")
def list_models(
    backend: str = typer.Option("openrouter", "--backend", "-b"),
) -> None:
    """List available LLM models for the selected backend."""
    settings = get_settings()

    if backend == "openrouter":
        if not settings.openrouter_api_key:
            console.print("[yellow]OPENROUTER_API_KEY not set; fetching public model list[/yellow]")
        models = fetch_free_models(settings)
        table = Table(title="OpenRouter Free Models (ranked)")
        table.add_column("Rank", style="dim")
        table.add_column("Model ID")
        table.add_column("Context")
        table.add_column("Params")
        table.add_column("Created")
        for i, m in enumerate(models, 1):
            table.add_row(
                str(i),
                m.id,
                str(m.context_length),
                f"{m.param_count}B" if m.param_count else "-",
                str(m.created),
            )
        console.print(table)
    else:
        console.print(f"Backend '{backend}' uses a single configured model:")
        if backend == "ollama":
            console.print(f"  {settings.ollama_model} @ {settings.ollama_base_url}")
        elif backend == "llamacpp":
            console.print(f"  {settings.llamacpp_model} @ {settings.llamacpp_base_url}")
        elif backend == "vllm":
            console.print(f"  {settings.vllm_model} @ {settings.vllm_base_url}")
        else:
            console.print(f"[red]Unknown backend: {backend}[/red]")
            raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
