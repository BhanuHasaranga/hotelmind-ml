from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# genai/config/genai_settings.py -> parents[2] == hotelmind-ml/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider selection ---
    LLM_PROVIDER: str = "ollama"  # ollama | openai | gemini
    LLM_MODEL: str = "llama3.1"
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1024

    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- Embeddings / RAG ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_DIR: str = "genai/rag/vector_store"
    DOCUMENT_DIR: str = "genai/rag/documents"

    # --- Data source ---
    DATA_SOURCE: str = "local"  # postgres | local

    # --- Feature flags ---
    ENABLE_RAG: bool = True
    ENABLE_MEMORY: bool = True

    # --- Reviews module ---
    REVIEWS_RAW_FILENAME: str = "guest_reviews_synthetic.csv"
    REVIEWS_OUTPUT_DIR: str = "data/processed/reviews"
    SYNTHETIC_REVIEW_COUNT: int = 60000
    SYNTHETIC_REVIEW_SEED: int = 42

    # --- Cache ---
    CACHE_DB_PATH: str = "genai/cache/genai_cache.sqlite3"
    QUERY_CACHE_TTL_SECONDS: int = 3600

    # --- Chunking / retrieval ---
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    RETRIEVER_TOP_K: int = 5
    MEMORY_MAX_TURNS: int = 8
    MEMORY_TTL_SECONDS: int = 1800

    @property
    def vector_db_path(self) -> Path:
        path = PROJECT_ROOT / self.VECTOR_DB_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def document_path(self) -> Path:
        path = PROJECT_ROOT / self.DOCUMENT_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Backwards/plan-name aliases
    vector_db_dir_path = vector_db_path
    document_dir_path = document_path

    @property
    def reviews_output_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.REVIEWS_OUTPUT_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_db_path(self) -> Path:
        path = PROJECT_ROOT / self.CACHE_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


genai_settings = GenAISettings()
