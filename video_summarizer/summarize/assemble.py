from __future__ import annotations

import json
from pathlib import Path

from video_summarizer.summarize.notes_schema import (
    ChunkRecord,
    SectionNotes,
    collect_glossary,
    section_notes_to_markdown,
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _dedupe_strings(items: list[str], *, prev_items: list[str] | None = None) -> list[str]:
    seen = {_normalize(x) for x in (prev_items or [])}
    result: list[str] = []
    for item in items:
        key = _normalize(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def dedupe_adjacent_sections(current: ChunkRecord, previous: ChunkRecord) -> ChunkRecord:
    """Remove repeated bullets likely caused by temporal chunk overlap."""
    prev = previous.notes
    notes = current.notes
    return ChunkRecord(
        index=current.index,
        start_time=current.start_time,
        end_time=current.end_time,
        notes=SectionNotes(
            title=notes.title,
            explanations=_dedupe_strings(notes.explanations, prev_items=prev.explanations),
            concepts=_dedupe_strings(notes.concepts, prev_items=prev.concepts),
            comparisons=_dedupe_strings(notes.comparisons, prev_items=prev.comparisons),
            steps=_dedupe_strings(notes.steps, prev_items=prev.steps),
            parameters=_dedupe_strings(notes.parameters, prev_items=prev.parameters),
            tips=_dedupe_strings(notes.tips, prev_items=prev.tips),
            terms=notes.terms,
            raw_markdown=notes.raw_markdown,
        ),
        model=current.model,
    )


def load_chunk_records(chunks_dir: Path) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for path in sorted(chunks_dir.glob("chunk_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "notes" in data:
            notes = SectionNotes.model_validate(data["notes"])
        else:
            notes = SectionNotes(
                title=f"Sezione {data.get('index', 0) + 1}",
                concepts=[data.get("summary", "")],
                raw_markdown=data.get("summary", ""),
            )
        records.append(
            ChunkRecord(
                index=int(data["index"]),
                start_time=data.get("start_time"),
                end_time=data.get("end_time"),
                notes=notes,
                model=data.get("model", ""),
            )
        )
    records.sort(key=lambda r: r.index)
    return records


def assemble_sections(
    records: list[ChunkRecord],
    *,
    dedupe_overlap: bool = True,
) -> list[ChunkRecord]:
    if not records:
        return []

    assembled: list[ChunkRecord] = [records[0]]
    for record in records[1:]:
        if dedupe_overlap:
            assembled.append(dedupe_adjacent_sections(record, assembled[-1]))
        else:
            assembled.append(record)
    return assembled


def build_sections_draft(
    sections: list[ChunkRecord],
    work_dir: Path,
) -> str:
    parts: list[str] = []
    for i, record in enumerate(sections, 1):
        parts.append(section_notes_to_markdown(i, record))
    draft = "\n\n---\n\n".join(parts)
    draft_path = work_dir / "sections_draft.md"
    draft_path.write_text(draft, encoding="utf-8")
    return draft


def build_index_lines(sections: list[ChunkRecord]) -> list[str]:
    from video_summarizer.summarize.notes_schema import format_time_range

    lines: list[str] = []
    for i, record in enumerate(sections, 1):
        time_range = format_time_range(record.start_time, record.end_time)
        lines.append(f"{i}. {record.notes.title} ({time_range})")
    return lines
