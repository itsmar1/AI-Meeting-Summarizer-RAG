from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field



class Settings(BaseSettings):
    """
    Central configuration loaded from environment variables / .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Ollama
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the running Ollama server."
    )

    ollama_llm_model: str = Field(
        default="llama3.2",
        description="Ollama model used for summarisation and RAG answers"
    )

    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model used to generate transcript embeddings."
    )


    # Whisper (faster-whisper)
    whisper_model_size: str = Field(
        default="small",
        description="faster-whisper model size: tiny | base | small | medium | large-v3",
    )
    whisper_device: str = Field(
        default="cpu",
        description="Inference device for faster-whisper: cpu | cuda | auto",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="Quantisation type: int8 (CPU-friendly) | float16 | float32",
    )


    # Audio processing
    audio_sample_rate: int = Field(
        default=16000,
        description="Target sample rate (Hz) for Whisper input.",
    )

    audio_channels: int = Field(
        default=1,
        description="Target channel count (1 = mono) for Whisper input.",
    )


    # Pyannote
    hf_token: str = Field(
        default="",
        description="Hugging Face token required to download pyannote models.",
    )

    # Diarization
    enable_diarization: bool = Field(
        default=True,
        description=(
            "Set to false to skip pyannote and use Whisper segments directly. "
            "Recommended on slow hardware where pyannote is impractical."
        ),
    )



    # Storage
    db_path: Path = Field(
        default=Path("data/meetings.db"),
        description="Path to the SQLite database file."
    )

    chroma_path: Path = Field(
        default=Path("data/chroma"),
        description="Directory where ChromaDB persists its vector index.",
    )

    transcript_dir: Path = Field(
        default=Path("data/transcripts"),
        description="Directory where raw transcript .txt files are saved.",
    )


    # RAG
    rag_chunk_size: int = Field(
        default=500,
        description="Character length of each transcript chunk for embedding.",
    )

    rag_chunk_overlap: int = Field(
        default=50,
        description="Character overlap between consecutive chunks.",
    )

    rag_top_k: int = Field(
        default=4,
        description="Number of most-similar chunks to retrieve per query.",
    )


    # App
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=7860)
    debug: bool = Field(default=False)

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist yet."""
        for path in (self.db_path.parent, self.chroma_path, self.transcript_dir):
            path.mkdir(parents=True, exist_ok=True)



# Single shared instance
settings = Settings()


