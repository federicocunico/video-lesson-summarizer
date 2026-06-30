from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Phase = Literal[
    "pending",
    "topic",
    "map",
    "assemble",
    "enrich",
    "reduce",
    "final",
    "done",
]


class CheckpointState(BaseModel):
    video_path: str = ""
    audio_done: bool = False
    audio_path: str = ""
    transcript_done: bool = False
    transcript_path: str = ""
    topic_done: bool = False
    topic_path: str = ""
    chunks_done: list[int] = Field(default_factory=list)
    assemble_done: bool = False
    summary_done: bool = False
    summary_path: str = ""
    notes_path: str = ""
    max_duration: int | None = None
    pipeline_version: int = 3
    phase: Phase = "pending"
    total_chunks: int = 0
    reduce_level: int = 0
    reduce_nodes_done: dict[str, list[int]] = Field(default_factory=dict)
    hierarchical_summary_path: str = ""
    sections_draft_path: str = ""


class CheckpointManager:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = work_dir / "checkpoint.json"
        self.state = self._load()

    def _load(self) -> CheckpointState:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return CheckpointState.model_validate(data)
        return CheckpointState()

    def save(self) -> None:
        self.path.write_text(
            self.state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def update(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self.state, key, value)
        self.save()

    def set_phase(self, phase: Phase) -> None:
        self.update(phase=phase)

    def current_phase(self) -> Phase:
        return self.state.phase

    def mark_chunk_done(self, index: int) -> None:
        if index not in self.state.chunks_done:
            self.state.chunks_done.append(index)
            self.state.chunks_done.sort()
            self.save()

    def is_chunk_done(self, index: int) -> bool:
        return index in self.state.chunks_done

    def _level_key(self, level: int) -> str:
        return f"L{level}"

    def mark_reduce_done(self, level: int, index: int) -> None:
        key = self._level_key(level)
        nodes = self.state.reduce_nodes_done.setdefault(key, [])
        if index not in nodes:
            nodes.append(index)
            nodes.sort()
            self.state.reduce_level = max(self.state.reduce_level, level)
            self.save()

    def is_reduce_done(self, level: int, index: int) -> bool:
        return index in self.state.reduce_nodes_done.get(self._level_key(level), [])

    def clear_notes_checkpoints(self) -> None:
        self.update(
            topic_done=False,
            chunks_done=[],
            assemble_done=False,
            summary_done=False,
            summary_path="",
            notes_path="",
            sections_draft_path="",
            reduce_nodes_done={},
            reduce_level=0,
            hierarchical_summary_path="",
            phase="pending",
        )

    def clear_reduce_checkpoints(self) -> None:
        self.clear_notes_checkpoints()
