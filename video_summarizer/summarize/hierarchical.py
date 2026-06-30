from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from video_summarizer.checkpoint import CheckpointManager
from video_summarizer.config import Settings
from video_summarizer.llm.base import LLMClient

console = Console()

REDUCE_BATCH_PROMPT = """Sei un assistente che unisce riassunti parziali di un video.
Ti vengono forniti diversi riassunti di sezioni consecutive o sovrapposte.
Crea un unico riassunto coerente in italiano che:
- Mantenga tutti i punti chiave
- Elimini ripetizioni
- Preservi strumenti, nomi e termini tecnici menzionati
Non inventare contenuti assenti nei riassunti forniti."""


def _batch_items(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _truncate_for_reduce(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 50] + "\n\n[... testo troncato per limite contesto ...]"


def _format_batch(batch: list[str], start_index: int) -> str:
    parts = []
    for i, summary in enumerate(batch):
        parts.append(f"### Sezione {start_index + i + 1}\n{summary}")
    return "\n\n---\n\n".join(parts)


def _reduce_path(work_dir: Path, level: int, batch_index: int) -> Path:
    return work_dir / "reduce" / f"L{level}" / f"batch_{batch_index:04d}.json"


def _load_reduce_cache(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["summary"]


def _save_reduce_cache(
    path: Path,
    *,
    level: int,
    batch_index: int,
    summary: str,
    model: str,
    input_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "level": level,
                "batch_index": batch_index,
                "summary": summary,
                "model": model,
                "input_count": input_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def hierarchical_reduce(
    summaries: list[str],
    work_dir: Path,
    llm: LLMClient,
    settings: Settings,
    checkpoint: CheckpointManager,
    *,
    force: bool = False,
) -> str:
    if not summaries:
        return "Nessun contenuto da riassumere."

    if len(summaries) == 1:
        return summaries[0]

    if force:
        checkpoint.clear_reduce_checkpoints()

    checkpoint.set_phase("reduce")
    current = list(summaries)
    level = 0

    while len(current) > 1:
        batches = _batch_items(current, settings.reduce_batch_size)
        next_level: list[str] = []

        for batch_index, batch in enumerate(batches):
            cache_path = _reduce_path(work_dir, level, batch_index)
            if (
                checkpoint.is_reduce_done(level, batch_index)
                and cache_path.exists()
                and not force
            ):
                next_level.append(_load_reduce_cache(cache_path))
                continue

            batch_text = _format_batch(
                batch,
                batch_index * settings.reduce_batch_size,
            )
            user_content = _truncate_for_reduce(
                batch_text,
                settings.reduce_max_input_chars,
            )

            messages = [
                {"role": "system", "content": REDUCE_BATCH_PROMPT},
                {"role": "user", "content": user_content},
            ]

            console.print(
                f"[dim]Reduce L{level} batch {batch_index + 1}/{len(batches)} "
                f"({len(batch)} input) via {llm.model_name}[/dim]"
            )
            merged = llm.chat(messages)

            _save_reduce_cache(
                cache_path,
                level=level,
                batch_index=batch_index,
                summary=merged,
                model=llm.model_name,
                input_count=len(batch),
            )
            checkpoint.mark_reduce_done(level, batch_index)
            next_level.append(merged)

        current = next_level
        level += 1

    result = current[0]
    hierarchical_path = work_dir / "hierarchical_summary.json"
    hierarchical_path.write_text(
        json.dumps(
            {
                "summary": result,
                "levels": level,
                "model": llm.model_name,
                "input_chunks": len(summaries),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    checkpoint.update(
        hierarchical_summary_path=str(hierarchical_path),
        reduce_level=level - 1,
    )
    return result
