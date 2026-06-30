from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from video_summarizer.checkpoint import CheckpointManager
from video_summarizer.config import Settings
from video_summarizer.runner import process_video
from video_summarizer.summarize.pipeline import run_summarize_pipeline
from video_summarizer.transcribe import Transcript, TranscriptSegment

SAMPLE_SECTION_JSON = json.dumps(
    {
        "title": "Metodo LAB per il bilanciamento del bianco",
        "explanations": [
            "In questa parte il docente introduce il metodo LAB come alternativa al classico bilanciamento in RGB.",
            "Spiega che LAB separa la luminosità (canale L) dai canali cromatici A e B, permettendo di correggere il colore senza alterare l'esposizione."
        ],
        "concepts": [
            "Il canale L controlla la luminosità dell'immagine, mentre A e B gestiscono la componente cromatica"
        ],
        "comparisons": [
            "LAB rispetto a RGB: in RGB si agisce sui tre canali insieme, in LAB si può correggere il bianco senza spostare la luminosità"
        ],
        "steps": ["Aprire il pannello LAB in Photoshop", "Regolare i cursori A e B"],
        "parameters": ["Valore A consigliato intorno a 0 per neutro"],
        "tips": ["Attenzione a non esagerare con A e B per evitare dominanti"],
        "terms": [{"term": "LAB", "definition": "Spazio colore dove L è luminosità, A e B sono assi cromatici opposti (verde-rosso, blu-giallo)"}],
        "raw_markdown": "Appunti descrittivi",
    }
)

SAMPLE_SYLLABUS_JSON = json.dumps(
    {
        "title": "Tutorial astronomico",
        "objectives": ["Imparare lo stacking", "Usare Siril"],
        "audience": "Principianti",
        "prerequisites": ["Foto RAW"],
        "section_outline": ["Intro", "Stacking", "Export"],
    }
)

SAMPLE_ENRICH_JSON = json.dumps(
    {
        "glossary": [{"term": "stacking", "definition": "impilamento foto"}],
        "review_questions": ["Cos'è lo stacking?", "A cosa serve Siril?"],
    }
)


class MockLLM:
    model_name = "mock/model"
    calls = 0

    def chat(self, messages, *, max_tokens=None):
        self.calls += 1
        system = messages[0]["content"]
        if "PROGRAMMA della lezione" in system:
            return SAMPLE_SYLLABUS_JSON
        if "APPUNTI DI LEZIONE" in system:
            return SAMPLE_SECTION_JSON
        if "glossary" in system:
            return SAMPLE_ENRICH_JSON
        return SAMPLE_SYLLABUS_JSON


@pytest.fixture
def sample_transcript() -> Transcript:
    segments = [
        TranscriptSegment(
            i * 5.0,
            i * 5.0 + 4.0,
            f"Questo è il segmento numero {i} con contenuto descrittivo del tutorial.",
        )
        for i in range(30)
    ]
    return Transcript(
        language="it",
        duration=150.0,
        text=" ".join(s.text for s in segments),
        segments=segments,
    )


def test_lesson_notes_pipeline_mock(tmp_path, sample_transcript: Transcript):
    llm = MockLLM()
    settings = Settings(segment_duration_sec=60)
    checkpoint = CheckpointManager(tmp_path)

    notes = run_summarize_pipeline(
        sample_transcript,
        tmp_path,
        llm,
        settings,
        checkpoint,
        video_name="test.mp4",
    )

    assert "# Appunti — Tutorial astronomico" in notes
    assert "## Obiettivi della lezione" in notes
    assert "## Sezione 1 —" in notes
    assert "### Contenuto e spiegazioni" in notes or "### Concetti chiave" in notes
    assert "## Glossario" in notes
    assert "## Domande di ripasso" in notes
    assert (tmp_path / "appunti.md").exists()
    assert (tmp_path / "appunti.json").exists()
    assert (tmp_path / "syllabus.json").exists()
    assert (tmp_path / "sections_draft.md").exists()
    assert not (tmp_path / "summary.md").exists()
    assert checkpoint.state.summary_done
    assert checkpoint.state.phase == "done"
    assert checkpoint.state.pipeline_version == 3


@pytest.mark.ffmpeg
def test_process_video_integration(tmp_path: Path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    data_dir = Path(__file__).resolve().parents[2] / "data"
    videos = list(data_dir.glob("*.mp4"))
    if not videos:
        pytest.skip("No test videos in data/")

    settings = Settings(
        output_dir=tmp_path,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_compute_type="int8",
        segment_duration_sec=60,
    )

    mock_llm = MockLLM()
    monkeypatch.setattr(
        "video_summarizer.runner.create_llm_client",
        lambda s: mock_llm,
    )

    work_dir = process_video(
        videos[0],
        settings=settings,
        max_duration=15,
        force=True,
        output_dir=tmp_path,
    )

    assert (work_dir / "audio.wav").exists()
    assert (work_dir / "transcript.json").exists()
    assert (work_dir / "appunti.md").exists()

    transcript = json.loads((work_dir / "transcript.json").read_text(encoding="utf-8"))
    assert "text" in transcript
