# Meeting Memoir

An AI-powered meeting summariser with speaker diarization, structured output extraction, multi-language support, and retrieval-augmented Q&A over past transcripts.

## Features

| Feature | Implementation |
|---|---|
| Audio transcription | faster-whisper |
| Language detection | faster-whisper probability scores |
| Speaker diarization | pyannote/speaker-diarization-3.1 |
| Structured extraction | Two-stage Ollama prompt chain + Pydantic |
| Streaming summary | Token-by-token via Ollama streaming API |
| Session history | SQLite + aiosqlite |
| RAG Q&A | ChromaDB + Ollama embeddings |
| UI | Gradio |
| Container | Docker + docker-compose |

## Quick start

### Option A — Docker (recommended)

```bash
# 1. Clone and enter the repo
git clone https://github.com/itsmar1/AI-Meeting-Summarizer-RAG.git
cd meeting-summariser

# 2. Set your Hugging Face token
echo "HF_TOKEN=hf_your_token_here" > .env

# 3. Start everything (Ollama + app)
docker compose up --build
```

Open http://localhost:7860 in your browser.

On first run `ollama-setup` will pull `llama3.2` and `nomic-embed-text` —
this takes a few minutes depending on your connection.

### Option B — Local (WSL / Linux / macOS)

**Prerequisites**

```bash
# ffmpeg (required for audio conversion)
sudo apt install ffmpeg          # WSL / Ubuntu / Debian
brew install ffmpeg              # macOS

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Hugging Face token** (required for pyannote diarization)

1. Create a free token at https://huggingface.co/settings/tokens
2. Accept the model licence at https://huggingface.co/pyannote/speaker-diarization-3.1

**Install and run**

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and fill in HF_TOKEN

# Run
python main.py
```

Open http://localhost:7860.

## Configuration

All settings are controlled via environment variables (or your `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Model for summarisation and RAG |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `WHISPER_MODEL_SIZE` | `small` | Whisper model: tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` | `cpu` | Inference device: cpu/cuda/auto |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantisation: int8/float16/float32 |
| `HF_TOKEN` | — | Hugging Face token (required) |
| `DB_PATH` | `data/meetings.db` | SQLite database path |
| `CHROMA_PATH` | `data/chroma` | ChromaDB persistence directory |
| `TRANSCRIPT_DIR` | `data/transcripts` | Transcript file output directory |
| `DEBUG` | `false` | Enable verbose logging |

## Project structure

```
meeting-summariser/
├── core/               # Config, logging, exceptions
├── pipeline/           # Audio → Transcribe → Diarise → Summarise
├── rag/                # Embed → Index → Retrieve → Answer
├── storage/            # SQLite schema and session repository
├── ui/                 # Gradio tabs and components
├── main.py             # Entry point
├── Dockerfile          # containerized deployment
├── docker-compose.yml  # app + Ollama service
├── .env.example        # HF token, Ollama URL, DB path
├── requirements.txt       
└── README.md
```

## Usage

### Summarise tab
1. Upload an audio file (any format ffmpeg supports: mp3, mp4, m4a, wav, …)
2. Optionally add context (e.g. "Q3 budget review")
3. Click **Process meeting**
4. Results appear across five sub-tabs: Summary, Decisions, Action items, Open questions, Full transcript

### History tab
- Browse all past sessions
- Paste a session ID to load full details
- Download the diarized transcript as a `.txt` file
- Delete sessions (also removes the RAG index)

### Chat (RAG) tab
- Select a specific session or search across all meetings
- Ask natural language questions about the transcript
- Source excerpts used to generate the answer are shown below

## Contributing

Contributions are welcome! Feel free to:

*   Open issues for bugs or suggestions
*   Submit pull requests with improvements

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [Ollama](https://ollama.com) — local LLM inference
- [ChromaDB](https://www.trychroma.com) — local vector database
- [Gradio](https://www.gradio.app) — UI framework

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.