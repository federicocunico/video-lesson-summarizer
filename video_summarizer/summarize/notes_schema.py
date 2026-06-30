from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class TermEntry(BaseModel):
    term: str
    definition: str


class SectionNotes(BaseModel):
    title: str = "Sezione senza titolo"
    explanations: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    terms: list[TermEntry] = Field(default_factory=list)
    raw_markdown: str = ""


class SyllabusOutline(BaseModel):
    title: str = "Lezione"
    objectives: list[str] = Field(default_factory=list)
    audience: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    section_outline: list[str] = Field(default_factory=list)
    raw_text: str = ""


class ChunkRecord(BaseModel):
    index: int
    start_time: float | None = None
    end_time: float | None = None
    notes: SectionNotes
    model: str = ""


class LessonNotes(BaseModel):
    title: str
    duration: str = ""
    source: str = ""
    objectives: list[str] = Field(default_factory=list)
    audience: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    sections: list[ChunkRecord] = Field(default_factory=list)
    glossary: list[TermEntry] = Field(default_factory=list)
    review_questions: list[str] = Field(default_factory=list)
    markdown: str = ""
    model: str = ""


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def parse_section_notes_response(text: str, *, fallback_title: str = "Sezione") -> SectionNotes:
    try:
        data = _extract_json_block(text)
        terms_raw = data.get("terms", [])
        terms = []
        for item in terms_raw:
            if isinstance(item, dict):
                terms.append(
                    TermEntry(
                        term=str(item.get("term", "")),
                        definition=str(item.get("definition", "")),
                    )
                )
            elif isinstance(item, str):
                terms.append(TermEntry(term=item, definition=""))
        return SectionNotes(
            title=str(data.get("title", fallback_title)),
            explanations=[str(x) for x in data.get("explanations", [])],
            concepts=[str(x) for x in data.get("concepts", [])],
            comparisons=[str(x) for x in data.get("comparisons", [])],
            steps=[str(x) for x in data.get("steps", [])],
            parameters=[str(x) for x in data.get("parameters", [])],
            tips=[str(x) for x in data.get("tips", [])],
            terms=terms,
            raw_markdown=str(data.get("raw_markdown", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return SectionNotes(
            title=fallback_title,
            concepts=[text[:500]] if text else [],
            raw_markdown=text,
        )


def parse_syllabus_response(text: str) -> SyllabusOutline:
    try:
        data = _extract_json_block(text)
        return SyllabusOutline(
            title=str(data.get("title", "Lezione")),
            objectives=[str(x) for x in data.get("objectives", [])],
            audience=str(data.get("audience", "")),
            prerequisites=[str(x) for x in data.get("prerequisites", [])],
            section_outline=[str(x) for x in data.get("section_outline", [])],
            raw_text=text,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return SyllabusOutline(title="Lezione", raw_text=text)


def parse_enrich_response(text: str) -> dict[str, Any]:
    try:
        return _extract_json_block(text)
    except (json.JSONDecodeError, ValueError):
        return {"glossary": [], "review_questions": []}


def format_time_range(start: float | None, end: float | None) -> str:
    def fmt(seconds: float | None) -> str:
        if seconds is None:
            return "??:??"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    return f"{fmt(start)}–{fmt(end)}"


def section_notes_to_markdown(
    section_num: int,
    record: ChunkRecord,
) -> str:
    notes = record.notes
    time_range = format_time_range(record.start_time, record.end_time)
    lines = [f"## Sezione {section_num} — {notes.title} ({time_range})", ""]

    if notes.explanations:
        lines.append("### Contenuto e spiegazioni")
        for exp in notes.explanations:
            lines.append(exp)
            lines.append("")
    elif notes.raw_markdown:
        lines.append("### Contenuto e spiegazioni")
        lines.append(notes.raw_markdown)
        lines.append("")

    if notes.concepts:
        lines.append("### Concetti chiave")
        lines.extend(f"- {c}" for c in notes.concepts)
        lines.append("")

    if notes.comparisons:
        lines.append("### Confronti e alternative")
        lines.extend(f"- {c}" for c in notes.comparisons)
        lines.append("")

    if notes.steps:
        lines.append("### Procedura")
        for i, step in enumerate(notes.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if notes.parameters:
        lines.append("### Parametri e impostazioni")
        lines.extend(f"- {p}" for p in notes.parameters)
        lines.append("")

    if notes.tips:
        lines.append("### Note del docente")
        lines.extend(f"- {t}" for t in notes.tips)
        lines.append("")

    if notes.terms:
        lines.append("### Termini tecnici")
        for t in notes.terms:
            if t.definition:
                lines.append(f"- **{t.term}**: {t.definition}")
            else:
                lines.append(f"- **{t.term}**")
        lines.append("")

    return "\n".join(lines).rstrip()


def collect_glossary(sections: list[ChunkRecord]) -> list[TermEntry]:
    seen: set[str] = set()
    glossary: list[TermEntry] = []
    for record in sections:
        for term in record.notes.terms:
            key = term.term.lower().strip()
            if key and key not in seen:
                seen.add(key)
                glossary.append(term)
    return glossary
