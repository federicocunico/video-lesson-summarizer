from __future__ import annotations

from pathlib import Path

from video_summarizer.checkpoint import CheckpointManager
from video_summarizer.config import Settings
from video_summarizer.summarize.hierarchical import hierarchical_reduce


class MockLLM:
    model_name = "mock/model"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, max_tokens=None):
        self.calls += 1
        user = messages[-1]["content"]
        if "Sezione" in user:
            count = user.count("### Sezione")
            return f"Riassunto unito di {count} sezioni in italiano."
        return "Riassunto."


def test_hierarchical_reduce_single(tmp_path: Path):
    llm = MockLLM()
    cp = CheckpointManager(tmp_path)
    settings = Settings(reduce_batch_size=6)

    result = hierarchical_reduce(["unico"], tmp_path, llm, settings, cp)
    assert result == "unico"
    assert llm.calls == 0


def test_hierarchical_reduce_two_levels(tmp_path: Path):
    llm = MockLLM()
    cp = CheckpointManager(tmp_path)
    settings = Settings(reduce_batch_size=3, reduce_max_input_chars=50000)

    # 7 summaries -> L0: 3 batches (3+3+1) -> L1: 2 batches -> L2: 1
    inputs = [f"Riassunto sezione {i}" for i in range(7)]
    result = hierarchical_reduce(inputs, tmp_path, llm, settings, cp)

    assert "Riassunto unito" in result
    assert llm.calls >= 2
    assert (tmp_path / "reduce" / "L0").exists()
    assert cp.is_reduce_done(0, 0)
    assert (tmp_path / "hierarchical_summary.json").exists()


def test_hierarchical_reduce_resume(tmp_path: Path):
    llm = MockLLM()
    cp = CheckpointManager(tmp_path)
    settings = Settings(reduce_batch_size=2)

    inputs = ["a", "b", "c", "d"]
    hierarchical_reduce(inputs, tmp_path, llm, settings, cp)
    first_calls = llm.calls

    llm2 = MockLLM()
    result = hierarchical_reduce(inputs, tmp_path, llm2, settings, cp)
    assert llm2.calls == 0
    assert result
