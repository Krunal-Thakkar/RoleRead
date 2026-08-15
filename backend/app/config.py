from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    chroma_persist_dir: str = "./chroma_data"

    chunk_size: int = 500
    chunk_overlap: int = 50

    top_k: int = 5

    max_history_turns: int = 6

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    extraction_model: str = "gpt-4o-mini"

    min_retrieval_score: float = 0.15
    # Calibrated against real text-embedding-3-small output on short skill-name phrases:
    # genuinely related pairs (e.g. "workflow automation" ~ "AI Workflow Automation") measured
    # ~0.5-0.73 cosine similarity, not the 0.75+ naively assumed for full-sentence embeddings.
    # See gap_analysis._FALSE_FRIENDS for the accompanying guard against lexically-similar-but-
    # unrelated pairs (e.g. "Java" ~ "JavaScript" measured ~0.66, which this lower threshold
    # would otherwise wrongly catch).
    semantic_match_threshold: float = 0.60

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_question_chars: int = 2000
    max_jobs_per_session: int = 5
    max_context_chars_per_doc: int = 8000  # cap per-document full text sent to the LLM


settings = Settings()
