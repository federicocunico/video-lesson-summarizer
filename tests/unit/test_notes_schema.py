from __future__ import annotations

import json

from video_summarizer.summarize.notes_schema import (
    ChunkRecord,
    SectionNotes,
    TermEntry,
    parse_section_notes_response,
    parse_syllabus_response,
    section_notes_to_markdown,
)
from video_summarizer.summarize.assemble import assemble_sections, dedupe_adjacent_sections


def test_parse_section_notes_json():
    raw = json.dumps(
        {
            "title": "Organizzazione cartelle",
            "concepts": ["Le cartelle devono essere in minuscolo"],
            "steps": ["Creare cartella lights"],
            "parameters": ["formato .fit"],
            "tips": ["L'istogramma sarà scuro"],
            "terms": [{"term": "stacking", "definition": "impilamento foto"}],
            "raw_markdown": "test",
        }
    )
    notes = parse_section_notes_response(raw)
    assert notes.title == "Organizzazione cartelle"
    assert len(notes.concepts) == 1
    assert notes.terms[0].term == "stacking"


def test_parse_syllabus_json():
    raw = json.dumps(
        {
            "title": "Elaborazione astrofoto",
            "objectives": ["Capire lo stacking"],
            "audience": "Principianti",
            "prerequisites": ["Siril installato"],
            "section_outline": ["Intro", "Stacking"],
        }
    )
    syllabus = parse_syllabus_response(raw)
    assert syllabus.title == "Elaborazione astrofoto"
    assert len(syllabus.objectives) == 1


def test_section_notes_to_markdown():
    record = ChunkRecord(
        index=0,
        start_time=0.0,
        end_time=300.0,
        notes=SectionNotes(
            title="Introduzione",
            concepts=["Concetto A"],
            steps=["Passo 1"],
        ),
    )
    md = section_notes_to_markdown(1, record)
    assert "## Sezione 1 — Introduzione" in md
    assert "### Concetti chiave" in md
    assert "### Procedura" in md


def test_section_notes_with_explanations():
    record = ChunkRecord(
        index=0,
        notes=SectionNotes(
            title="LAB",
            explanations=["Il metodo LAB separa luminosità e colore."],
            comparisons=["LAB vs RGB: correzione colore indipendente dalla luminosità"],
            terms=[TermEntry(term="LAB", definition="L=Luminosità, A e B=assi cromatici")],
        ),
    )
    md = section_notes_to_markdown(1, record)
    assert "### Contenuto e spiegazioni" in md
    assert "### Confronti e alternative" in md
    assert "LAB" in md


def test_dedupe_adjacent_sections():
    prev = ChunkRecord(
        index=0,
        notes=SectionNotes(title="A", concepts=["Stesso concetto", "Unico"]),
    )
    curr = ChunkRecord(
        index=1,
        notes=SectionNotes(title="B", concepts=["Stesso concetto", "Nuovo"]),
    )
    deduped = dedupe_adjacent_sections(curr, prev)
    assert deduped.notes.concepts == ["Nuovo"]


def test_assemble_sections():
    records = [
        ChunkRecord(index=0, notes=SectionNotes(title="A", concepts=["x"])),
        ChunkRecord(index=1, notes=SectionNotes(title="B", concepts=["x", "y"])),
    ]
    assembled = assemble_sections(records)
    assert len(assembled) == 2
    assert assembled[1].notes.concepts == ["y"]
