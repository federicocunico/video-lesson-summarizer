from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from video_summarizer.checkpoint import CheckpointManager
from video_summarizer.config import Settings
from video_summarizer.llm.base import LLMClient
from video_summarizer.summarize.assemble import (
    assemble_sections,
    build_index_lines,
    build_sections_draft,
    load_chunk_records,
)
from video_summarizer.summarize.chunking import (
    TextSample,
    sample_transcript_points,
    split_transcript_by_time,
)
from video_summarizer.summarize.notes_schema import (
    ChunkRecord,
    LessonNotes,
    SectionNotes,
    SyllabusOutline,
    TermEntry,
    parse_enrich_response,
    parse_section_notes_response,
    parse_syllabus_response,
)
from video_summarizer.summarize.prompts import (
    ENRICH_PROMPT,
    MAP_NOTES_PROMPT,
    SYLLABUS_PROMPT,
)
from video_summarizer.transcribe import Transcript

console = Console()


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_samples(samples: list[TextSample]) -> str:
    parts = []
    for s in samples:
        time_info = ""
        if s.start_time is not None:
            time_info = f" (t={_format_time(s.start_time)})"
        parts.append(f"### {s.label}{time_info}\n{s.text}")
    return "\n\n".join(parts)


def _format_chunk_user_content(chunk_text: str, start: float | None, end: float | None) -> str:
    header = f"Intervallo video: {_format_time(start)} – {_format_time(end)}\n\n"
    return header + chunk_text


def _load_syllabus(path: Path) -> SyllabusOutline:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "syllabus" in data:
        return SyllabusOutline.model_validate(data["syllabus"])
    return SyllabusOutline(title="Lezione", raw_text=data.get("topic", ""))


def _chat_with_json_retry(
    llm: LLMClient,
    messages: list[dict[str, str]],
    parser,
    *,
    fallback,
):
    response = llm.chat(messages)
    try:
        return parser(response)
    except Exception:
        retry_messages = messages + [
            {
                "role": "user",
                "content": "La risposta non era JSON valido. Ripeti SOLO con JSON valido, senza markdown.",
            }
        ]
        response = llm.chat(retry_messages)
        return parser(response)


def run_syllabus_phase(
    transcript: Transcript,
    work_dir: Path,
    llm: LLMClient,
    settings: Settings,
    checkpoint: CheckpointManager,
    *,
    force: bool = False,
) -> SyllabusOutline:
    syllabus_path = work_dir / "syllabus.json"
    legacy_topic_path = work_dir / "topic.json"

    if checkpoint.state.topic_done and syllabus_path.exists() and not force:
        return _load_syllabus(syllabus_path)

    checkpoint.set_phase("topic")
    samples = sample_transcript_points(
        transcript,
        char_limit=settings.sample_char_limit,
    )
    if not samples:
        syllabus = SyllabusOutline(title="Lezione senza contenuto trascrivibile")
        syllabus_path.write_text(
            json.dumps({"syllabus": syllabus.model_dump()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        checkpoint.update(topic_done=True, topic_path=str(syllabus_path))
        return syllabus

    messages = [
        {"role": "system", "content": SYLLABUS_PROMPT},
        {"role": "user", "content": _format_samples(samples)},
    ]

    console.print(f"[dim]Programma lezione via {llm.model_name}[/dim]")
    syllabus = _chat_with_json_retry(
        llm,
        messages,
        parse_syllabus_response,
        fallback=SyllabusOutline(title="Lezione"),
    )

    payload = {
        "syllabus": syllabus.model_dump(),
        "model": llm.model_name,
        "samples": [
            {"label": s.label, "text": s.text, "start_time": s.start_time}
            for s in samples
        ],
    }
    syllabus_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    legacy_topic_path.write_text(
        json.dumps({"topic": syllabus.raw_text or syllabus.title}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    checkpoint.update(topic_done=True, topic_path=str(syllabus_path))
    return syllabus


def run_map_notes_phase(
    transcript: Transcript,
    work_dir: Path,
    llm: LLMClient,
    settings: Settings,
    checkpoint: CheckpointManager,
    *,
    force: bool = False,
) -> list[ChunkRecord]:
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    checkpoint.set_phase("map")
    chunks = split_transcript_by_time(
        transcript.segments,
        window_sec=float(settings.segment_duration_sec),
        overlap_sec=float(settings.segment_overlap_sec),
        max_chunk_chars=settings.max_chunk_chars,
        char_overlap=settings.chunk_overlap,
    )
    checkpoint.update(total_chunks=len(chunks))

    records: list[ChunkRecord] = []
    for chunk in chunks:
        chunk_path = chunks_dir / f"chunk_{chunk.index:04d}.json"
        if checkpoint.is_chunk_done(chunk.index) and chunk_path.exists() and not force:
            data = json.loads(chunk_path.read_text(encoding="utf-8"))
            if "notes" in data:
                records.append(
                    ChunkRecord(
                        index=data["index"],
                        start_time=data.get("start_time"),
                        end_time=data.get("end_time"),
                        notes=SectionNotes.model_validate(data["notes"]),
                        model=data.get("model", ""),
                    )
                )
                continue

        messages = [
            {"role": "system", "content": MAP_NOTES_PROMPT},
            {
                "role": "user",
                "content": _format_chunk_user_content(
                    chunk.text,
                    chunk.start_time,
                    chunk.end_time,
                ),
            },
        ]

        console.print(
            f"[dim]Appunti chunk {chunk.index + 1}/{len(chunks)} "
            f"({_format_time(chunk.start_time)}–{_format_time(chunk.end_time)}) "
            f"via {llm.model_name}[/dim]"
        )
        response = llm.chat(messages, max_tokens=settings.map_max_tokens)
        notes = parse_section_notes_response(
            response,
            fallback_title=f"Parte {chunk.index + 1}",
        )
        record = ChunkRecord(
            index=chunk.index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            notes=notes,
            model=llm.model_name,
        )

        chunk_path.write_text(
            json.dumps(
                {
                    "index": record.index,
                    "notes": record.notes.model_dump(),
                    "model": record.model,
                    "start_time": record.start_time,
                    "end_time": record.end_time,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        checkpoint.mark_chunk_done(chunk.index)
        records.append(record)

    return records


def run_assemble_phase(
    work_dir: Path,
    records: list[ChunkRecord],
    checkpoint: CheckpointManager,
    *,
    force: bool = False,
) -> tuple[list[ChunkRecord], str]:
    draft_path = work_dir / "sections_draft.md"
    if checkpoint.state.assemble_done and draft_path.exists() and not force:
        return load_chunk_records(work_dir / "chunks"), draft_path.read_text(encoding="utf-8")

    checkpoint.set_phase("assemble")
    if not records:
        records = load_chunk_records(work_dir / "chunks")
    assembled = assemble_sections(records)
    draft = build_sections_draft(assembled, work_dir)
    checkpoint.update(
        assemble_done=True,
        sections_draft_path=str(draft_path),
    )
    return assembled, draft


def run_enrich_phase(
    syllabus: SyllabusOutline,
    sections: list[ChunkRecord],
    sections_draft: str,
    work_dir: Path,
    llm: LLMClient,
    checkpoint: CheckpointManager,
    *,
    video_name: str = "",
    duration: float | None = None,
    force: bool = False,
) -> str:
    appunti_md_path = work_dir / "appunti.md"
    appunti_json_path = work_dir / "appunti.json"

    if checkpoint.state.summary_done and appunti_md_path.exists() and not force:
        return appunti_md_path.read_text(encoding="utf-8")

    checkpoint.set_phase("enrich")

    user_content = (
        f"Programma lezione:\n{json.dumps(syllabus.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Sezioni strutturate:\n{sections_draft}"
    )
    messages = [
        {"role": "system", "content": ENRICH_PROMPT},
        {"role": "user", "content": user_content},
    ]

    console.print(f"[dim]Glossario e domande di ripasso via {llm.model_name}[/dim]")
    enrich_raw = llm.chat(messages)
    enrich_data = parse_enrich_response(enrich_raw)

    glossary = [
        TermEntry(term=str(g.get("term", "")), definition=str(g.get("definition", "")))
        for g in enrich_data.get("glossary", [])
        if g.get("term")
    ]
    review_questions = [str(q) for q in enrich_data.get("review_questions", []) if str(q).strip()]

    if not glossary:
        from video_summarizer.summarize.notes_schema import collect_glossary

        glossary = collect_glossary(sections)

    duration_str = _format_time(duration) if duration else "?"
    index_lines = build_index_lines(sections)

    md_parts = [
        f"# Appunti — {syllabus.title}",
        "",
        f"> Durata: {duration_str} | Fonte: {video_name}",
        "",
        "## Obiettivi della lezione",
    ]
    if syllabus.objectives:
        md_parts.extend(f"- {obj}" for obj in syllabus.objectives)
    else:
        md_parts.append("- (non specificati)")

    if syllabus.audience:
        md_parts.extend(["", f"**Pubblico:** {syllabus.audience}"])
    if syllabus.prerequisites:
        md_parts.extend(["", "**Prerequisiti:**"])
        md_parts.extend(f"- {p}" for p in syllabus.prerequisites)

    md_parts.extend(["", "## Indice"])
    md_parts.extend(index_lines)
    md_parts.extend(["", "---", "", sections_draft, "", "---", ""])

    if glossary:
        md_parts.extend(["## Glossario", "", "| Termine | Definizione |", "| --- | --- |"])
        for term in glossary:
            definition = term.definition.replace("|", "\\|")
            md_parts.append(f"| {term.term} | {definition} |")
        md_parts.append("")

    if review_questions:
        md_parts.extend(["## Domande di ripasso"])
        for i, question in enumerate(review_questions, 1):
            md_parts.append(f"{i}. {question}")
        md_parts.append("")

    markdown = "\n".join(md_parts).strip() + "\n"
    appunti_md_path.write_text(markdown, encoding="utf-8")

    lesson = LessonNotes(
        title=syllabus.title,
        duration=duration_str,
        source=video_name,
        objectives=syllabus.objectives,
        audience=syllabus.audience,
        prerequisites=syllabus.prerequisites,
        sections=sections,
        glossary=glossary,
        review_questions=review_questions,
        markdown=markdown,
        model=llm.model_name,
    )
    appunti_json_path.write_text(
        lesson.model_dump_json(indent=2),
        encoding="utf-8",
    )

    checkpoint.update(
        summary_done=True,
        summary_path=str(appunti_md_path),
        notes_path=str(appunti_md_path),
        phase="done",
        pipeline_version=3,
    )
    return markdown


def run_summarize_pipeline(
    transcript: Transcript,
    work_dir: Path,
    llm: LLMClient,
    settings: Settings,
    checkpoint: CheckpointManager,
    *,
    force: bool = False,
    video_name: str = "",
) -> str:
    if force:
        checkpoint.clear_notes_checkpoints()

    syllabus = run_syllabus_phase(
        transcript, work_dir, llm, settings, checkpoint, force=force
    )
    records = run_map_notes_phase(
        transcript, work_dir, llm, settings, checkpoint, force=force
    )
    sections, draft = run_assemble_phase(work_dir, records, checkpoint, force=force)
    return run_enrich_phase(
        syllabus,
        sections,
        draft,
        work_dir,
        llm,
        checkpoint,
        video_name=video_name or work_dir.name,
        duration=transcript.duration,
        force=force,
    )
