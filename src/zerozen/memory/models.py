from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, JSON, Column
from sqlalchemy import Text, DateTime
import uuid


class MemoryEntry(SQLModel, table=True):
    __tablename__ = "memories"

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()), primary_key=True
    )

    content: str = Field(
        sa_column=Column(Text), description="The actual memory content"
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Vector embedding of the content",
    )

    user_id: str = Field(
        index=True, description="ID of the user this memory belongs to"
    )

    session_id: str = Field(
        index=True, description="Session where this memory was created"
    )

    memory_type: str = Field(
        default="general",
        description="Type of memory (preference, fact, context, etc.)",
    )

    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance score for memory ranking"
    )

    extra_data: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Additional metadata for the memory",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime),
        description="When the memory was created",
    )

    last_accessed: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime),
        description="When the memory was last retrieved",
    )

    access_count: int = Field(
        default=0, description="Number of times this memory has been accessed"
    )


class SessionModel(SQLModel, table=True):
    __tablename__ = "sessions"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)

    created_at: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime)
    )

    updated_at: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime)
    )

    session_data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON), description="Session metadata"
    )

    summary: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="Session summary for long-term memory",
    )

    is_active: bool = Field(
        default=True, description="Whether the session is currently active"
    )


class InteractionModel(SQLModel, table=True):
    __tablename__ = "interactions"

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()), primary_key=True
    )

    session_id: str = Field(index=True, foreign_key="sessions.id")

    user_query: str = Field(sa_column=Column(Text))
    llm_response: str = Field(sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime)
    )

    interaction_data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )
