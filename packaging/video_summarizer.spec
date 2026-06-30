# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows GUI release."""

from pathlib import Path

block_cipher = None
root = Path(SPECPATH).parent

a = Analysis(
    [str(root / "video_summarizer" / "gui.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "faster_whisper",
        "ctranslate2",
        "av",
        "httpx",
        "pydantic_settings",
        "rich",
        "typer",
        "video_summarizer",
        "video_summarizer.audio",
        "video_summarizer.checkpoint",
        "video_summarizer.config",
        "video_summarizer.gui",
        "video_summarizer.llm",
        "video_summarizer.llm.base",
        "video_summarizer.llm.ollama",
        "video_summarizer.llm.openai_compat",
        "video_summarizer.llm.openrouter",
        "video_summarizer.llm.retry",
        "video_summarizer.runner",
        "video_summarizer.summarize.assemble",
        "video_summarizer.summarize.chunking",
        "video_summarizer.summarize.hierarchical",
        "video_summarizer.summarize.notes_schema",
        "video_summarizer.summarize.pipeline",
        "video_summarizer.summarize.prompts",
        "video_summarizer.transcribe",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="video-lesson-summarizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="video-lesson-summarizer",
)
