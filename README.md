# Video Lesson Summarizer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Trasforma video lezioni in **trascrizione completa** e **appunti di lezione** strutturati in italiano.

Estrae l'audio con ffmpeg, trascrivi con [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CUDA consigliato) e genera appunti descrittivi tramite LLM (OpenRouter, Ollama, llama.cpp o vLLM).

## Funzionalità

- Trascrizione speech-to-text con segmenti timestampati
- Appunti di lezione in italiano: concetti, procedure, confronti, glossario
- Pipeline con checkpoint e resume
- **Interfaccia desktop** con visualizzazione di trascrizione e `appunti.md`
- CLI per automazione e Docker con supporto CUDA

## Requisiti

- Python 3.12+ e [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) nel `PATH`
- NVIDIA GPU + CUDA (consigliato per Whisper)
- Chiave OpenRouter **oppure** Ollama / llama.cpp / vLLM locale

## Installazione

```bash
git clone https://github.com/federicocunico/video-lesson-summarizer.git
cd video-lesson-summarizer
uv sync --extra dev
cp .env.example .env
# Modifica .env con OPENROUTER_API_KEY o impostazioni Ollama
```

### Release Windows (senza Python)

Scarica l'ultimo `.zip` dalla pagina [Releases](https://github.com/federicocunico/video-lesson-summarizer/releases). Estrai l'archivio ed esegui `video-lesson-summarizer.exe`.

> **Nota:** ffmpeg deve essere installato e disponibile nel `PATH`. Al primo avvio Whisper scaricherà il modello configurato (es. `large-v3`).

## Interfaccia desktop

```bash
uv run video-summarizer gui
# oppure
uv run video-summarizer-gui
```

L'app mostra due pannelli principali:

| Tab | Contenuto |
|-----|-----------|
| **Trascrizione** | Testo completo + segmenti con timestamp (`transcript.md`) |
| **Appunti** | Appunti di lezione generati (`appunti.md`) |

Dopo l'elaborazione i file vengono salvati in `output/<nome_video>/` e visualizzati nell'interfaccia.

## CLI

```bash
# Pipeline completa
uv run video-summarizer process data/lezione.mp4

# Test rapido (primi 60 secondi)
uv run video-summarizer process data/lezione.mp4 --max-duration 60

# Rigenera solo gli appunti (mantiene la trascrizione)
uv run video-summarizer process data/lezione.mp4 --force

# Riprendi job interrotto
uv run video-summarizer resume output/lezione/

# Elenco modelli OpenRouter gratuiti
uv run video-summarizer models list --backend openrouter
```

## Output

Per ogni video, in `output/<video_stem>/`:

| File | Descrizione |
|------|-------------|
| `transcript.json` | Trascrizione strutturata (JSON) |
| `transcript.md` | Trascrizione completa leggibile |
| `appunti.md` | Appunti di lezione in Markdown |
| `appunti.json` | Appunti strutturati (JSON) |
| `syllabus.json` | Programma della lezione |
| `sections_draft.md` | Bozza sezioni intermedie |
| `chunks/` | Appunti per segmento temporale |
| `checkpoint.json` | Stato per resume |

### Esempio sezione in `appunti.md`

```markdown
## Sezione 3 — Bilanciamento del bianco con metodo LAB (10:00–15:00)

### Contenuto e spiegazioni
Il docente introduce il metodo LAB come alternativa al bilanciamento classico in RGB.

### Termini tecnici
- **LAB**: spazio colore dove L = luminosità, A = asse verde-rosso, B = asse blu-giallo
```

## Pipeline (v3)

1. **Syllabus** — titolo, obiettivi, prerequisiti, indice macro-sezioni
2. **Map** — appunti per segmento (~5 min): concetti, procedura, parametri
3. **Assemble** — unione sezioni con deduplica overlap
4. **Enrich** — glossario unificato + domande di ripasso

## Backend LLM

Imposta `LLM_BACKEND` in `.env`:

| Backend | Variabili |
|---------|-----------|
| `openrouter` | `OPENROUTER_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| `llamacpp` | `LLAMACPP_BASE_URL`, `LLAMACPP_MODEL` |
| `vllm` | `VLLM_BASE_URL`, `VLLM_MODEL` |

### Ollama locale (24 GB VRAM)

```powershell
.\scripts\setup_ollama.ps1
# oppure: ollama pull qwen2.5:32b
```

## Docker (CUDA)

```bash
docker compose -f docker/docker-compose.cuda.yml build
docker compose -f docker/docker-compose.cuda.yml run --rm video-summarizer process /data/video.mp4 --max-duration 60
```

## Build eseguibile (Windows)

```powershell
uv sync --extra build
uv run pyinstaller build/video_summarizer.spec --noconfirm
# Output: dist/video-lesson-summarizer/
```

## Release automatiche

Push di un tag `v*` (es. `v0.1.0`) attiva [`.github/workflows/release.yml`](.github/workflows/release.yml):

1. Build dell'eseguibile Windows con PyInstaller
2. Creazione archivio `.zip`
3. Pubblicazione su GitHub Releases

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Test

```bash
uv run pytest tests/unit -v
uv run pytest tests/integration -v -m ffmpeg
uv run pytest tests/gpu -v -m gpu
```

## Licenza

Questo progetto è rilasciato sotto licenza [MIT](LICENSE).
