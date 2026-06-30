from __future__ import annotations

from pathlib import Path

from video_summarizer.checkpoint import CheckpointManager


def test_checkpoint_save_and_load(tmp_path: Path):
    cp = CheckpointManager(tmp_path)
    cp.update(video_path="/data/video.mp4", audio_done=True)
    cp2 = CheckpointManager(tmp_path)
    assert cp2.state.video_path == "/data/video.mp4"
    assert cp2.state.audio_done is True


def test_checkpoint_chunks(tmp_path: Path):
    cp = CheckpointManager(tmp_path)
    assert not cp.is_chunk_done(0)
    cp.mark_chunk_done(0)
    cp.mark_chunk_done(2)
    assert cp.is_chunk_done(0)
    assert cp.is_chunk_done(2)
    assert not cp.is_chunk_done(1)

    cp2 = CheckpointManager(tmp_path)
    assert cp2.state.chunks_done == [0, 2]
