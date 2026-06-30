# Video Summarizer

Extract audio from video (ffmpeg), transcribe with faster-whisper (CUDA), and generate structured lesson notes in Italian.

## Requirements

- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- ffmpeg
- NVIDIA GPU + CUDA (recommended for Whisper)
- OpenRouter API key (for cloud LLM) or local Ollama/llama.cpp/vLLM

## Setup

```bash
uv sync --extra dev
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY or Ollama settings
```

## Usage

```bash
# Full pipeline
uv run video-summarizer process data/tutorial_day_pano_edit_tenerife.mp4

# Quick test (first 60 seconds)
uv run video-summarizer process data/video.mp4 --max-duration 60

# Regenerate lesson notes only (keeps existing transcript)
uv run video-summarizer process data/video.mp4 -f

# Resume interrupted job
uv run video-summarizer resume output/tutorial_day_pano_edit_tenerife/

# List free OpenRouter models
uv run video-summarizer models list --backend openrouter
```

## Docker (CUDA)

```bash
docker compose -f docker/docker-compose.cuda.yml build
docker compose -f docker/docker-compose.cuda.yml run --rm video-summarizer process /data/video.mp4 --max-duration 60
```

## Lesson notes pipeline (v3)

Output: **appunti di lezione descrittivi** in italiano — spiegazioni complete, confronti tra metodi,
definizioni di acronimi (es. LAB, RGB), procedure e note del docente. Tutto rigorosamente dal trascritto.

1. **Syllabus** — programma: titolo, obiettivi, prerequisiti, indice macro-sezioni
2. **Map** — appunti strutturati per segmento temporale (~5 min): concetti, procedura, parametri, note docente
3. **Assemble** — unione sezioni preservando struttura, dedupe overlap
4. **Enrich** — glossario unificato + domande di ripasso

Output in `output/<video_stem>/`: `syllabus.json`, `chunks/`, `sections_draft.md`, `appunti.md`, `appunti.json`, `checkpoint.json`.

### Example section in `appunti.md`

```markdown
## Sezione 3 — Bilanciamento del bianco con metodo LAB (10:00–15:00)

### Contenuto e spiegazioni
Il docente introduce il metodo LAB come alternativa al bilanciamento classico in RGB.
LAB separa la luminosità (canale L) dai canali cromatici A e B, così si può correggere
il colore senza alterare l'esposizione complessiva dell'immagine.

### Confronti e alternative
- LAB vs RGB: in RGB i tre canali sono accoppiati; in LAB si agisce sul colore in modo indipendente dalla luminosità

### Termini tecnici
- **LAB**: spazio colore dove L = luminosità, A = asse verde-rosso, B = asse blu-giallo
```

## LLM Backends

Set `LLM_BACKEND` in `.env`:

| Backend | Variables |
|---------|-----------|
| `openrouter` | `OPENROUTER_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (default: `qwen2.5:32b`) |
| `llamacpp` | `LLAMACPP_BASE_URL`, `LLAMACPP_MODEL` |
| `vllm` | `VLLM_BASE_URL`, `VLLM_MODEL` |

## Ollama setup (locale, 24GB VRAM)

```powershell
.\scripts\setup_ollama.ps1
# oppure: ollama pull qwen2.5:32b
```

## Tests

```bash
uv run pytest tests/unit -v
uv run pytest tests/integration -v -m ffmpeg
uv run pytest tests/gpu -v -m gpu
```
