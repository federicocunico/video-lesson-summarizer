from __future__ import annotations

from video_summarizer.llm.openrouter import _is_free_model, _supports_text, rank_free_models


def test_is_free_model():
    assert _is_free_model({"id": "x", "pricing": {"prompt": "0", "completion": "0"}})
    assert _is_free_model({"id": "model:free", "pricing": {"prompt": "1", "completion": "1"}})
    assert not _is_free_model({"id": "paid", "pricing": {"prompt": "0.001", "completion": "0.002"}})


def test_supports_text():
    assert _supports_text({"architecture": {"output_modalities": ["text"]}})
    assert not _supports_text({"architecture": {"output_modalities": ["image"]}})


def test_rank_free_models_priority():
    models = [
        {
            "id": "old/small-3b:free",
            "name": "Small 3B",
            "context_length": 8000,
            "created": 1000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        },
        {
            "id": "new/large-70b-instruct:free",
            "name": "Large 70B",
            "context_length": 128000,
            "created": 2000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        },
        {
            "id": "paid/model",
            "name": "Paid",
            "context_length": 128000,
            "created": 3000,
            "pricing": {"prompt": "0.001", "completion": "0.002"},
            "architecture": {"output_modalities": ["text"]},
        },
    ]
    ranked = rank_free_models(models)
    assert len(ranked) == 2
    assert ranked[0].id == "new/large-70b-instruct:free"
    assert ranked[0].param_count == 70
