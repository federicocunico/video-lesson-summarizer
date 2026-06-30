from __future__ import annotations

from pathlib import Path

from video_summarizer.checkpoint import CheckpointManager


def test_checkpoint_v2_fields(tmp_path: Path):
    cp = CheckpointManager(tmp_path)
    cp.update(
        pipeline_version=2,
        phase="map",
        total_chunks=10,
        reduce_level=1,
    )
    cp.mark_reduce_done(0, 0)
    cp.mark_reduce_done(0, 1)
    cp.mark_reduce_done(1, 0)

    cp2 = CheckpointManager(tmp_path)
    assert cp2.state.pipeline_version == 2
    assert cp2.state.phase == "map"
    assert cp2.state.total_chunks == 10
    assert cp2.is_reduce_done(0, 0)
    assert cp2.is_reduce_done(1, 0)
    assert cp2.state.reduce_nodes_done["L0"] == [0, 1]


def test_clear_reduce_checkpoints(tmp_path: Path):
    cp = CheckpointManager(tmp_path)
    cp.mark_reduce_done(0, 0)
    cp.update(hierarchical_summary_path="/tmp/x.json")
    cp.clear_reduce_checkpoints()

    assert cp.state.reduce_nodes_done == {}
    assert cp.state.reduce_level == 0
    assert cp.state.hierarchical_summary_path == ""


def test_set_phase(tmp_path: Path):
    cp = CheckpointManager(tmp_path)
    cp.set_phase("reduce")
    assert cp.current_phase() == "reduce"
