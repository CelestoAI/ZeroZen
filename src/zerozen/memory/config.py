from typing import Optional, Literal
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    store_type: Literal["sqlite", "chroma"] = Field(
        default="sqlite", description="Type of storage backend to use"
    )

    database_url: str = Field(
        default="sqlite:///./llm_memory.db",
        description="Database connection URL (SQLite, Chroma, etc.)",
    )

    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="Model to use for generating embeddings",
    )

    max_context_memories: int = Field(
        default=10, description="Maximum number of memories to include in context"
    )

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for memory retrieval",
    )

    memory_retention_days: Optional[int] = Field(
        default=None,
        description="Number of days to retain memories (None for infinite)",
    )

    embedding_dimension: int = Field(
        default=1536, description="Dimension of embedding vectors"
    )

    auto_summarize_sessions: bool = Field(
        default=True,
        description="Automatically summarize old sessions for long-term memory",
    )

    class Config:
        env_prefix = "MEMORY_"
