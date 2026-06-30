from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: tests requiring NVIDIA GPU")
    config.addinivalue_line("markers", "ffmpeg: tests requiring ffmpeg in PATH")
