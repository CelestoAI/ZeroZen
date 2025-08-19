from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from openai import OpenAI

from .config import MemoryConfig
from .models import MemoryEntry, SessionModel
from .store import MemoryStore, SQLiteMemoryStore
from .chroma_store import ChromaMemoryStore


class MemoryManager:
    def __init__(self, config: MemoryConfig, store: Optional[MemoryStore] = None):
        self.config = config

        # Initialize store based on config or provided store
        if store:
            self._store = store
        elif config.store_type == "chroma":
            self._store = ChromaMemoryStore(config)
        else:
            self._store = SQLiteMemoryStore(config)
        self.openai_client = None

        # Initialize OpenAI client if embedding model is specified
        if config.embedding_model.startswith(("text-embedding", "ada")):
            try:
                self.openai_client = OpenAI()
            except Exception:
                # OpenAI client initialization failed - embeddings will be disabled
                pass

    def start_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        timestamp = int(datetime.now().timestamp())
        session_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{timestamp}"

        self._store.create_session(session_id, user_id, metadata)
        return session_id

    def store(
        self,
        content: str,
        session_id: str,
        user_id: Optional[str] = None,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> str:
        # Generate embedding if OpenAI client is available
        embedding = None
        if self.openai_client:
            try:
                embedding = self._generate_embedding(content)
            except Exception:
                # Embedding generation failed - continue without embedding
                pass

        # Extract user_id from session_id if not provided
        if not user_id:
            # Simple extraction - works if user_id doesn't contain underscores
            # For production, should lookup from sessions table
            user_id = session_id.rsplit("_", 2)[
                0
            ]  # Get everything before last 2 underscores

        memory = MemoryEntry(
            content=content,
            embedding=embedding,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            importance_score=importance,
            extra_data=metadata,
        )

        return self._store.store_memory(memory)

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        limit: int = None,
        time_range: Optional[str] = None,
        scope: str = "session",
    ) -> List[MemoryEntry]:
        # Use config default if limit not specified
        if limit is None:
            limit = self.config.max_context_memories

        # Generate query embedding
        query_embedding = None
        if self.openai_client:
            try:
                query_embedding = self._generate_embedding(query)
            except Exception:
                # Embedding generation failed - continue without semantic search
                pass

        # Determine scope parameters
        if scope == "session" and session_id:
            # Single session scope
            memories = self._store.retrieve_memories(
                query_embedding=query_embedding,
                session_id=session_id,
                memory_type=memory_type,
                similarity_threshold=self.config.similarity_threshold,
                limit=limit,
                time_range=time_range,
            )
        elif scope == "user" and user_id:
            # All sessions for user
            memories = self._store.retrieve_memories(
                query_embedding=query_embedding,
                user_id=user_id,
                memory_type=memory_type,
                similarity_threshold=self.config.similarity_threshold,
                limit=limit,
                time_range=time_range,
            )
        elif scope == "sessions" and session_ids:
            # Multiple specific sessions
            memories = self._store.retrieve_memories(
                query_embedding=query_embedding,
                session_ids=session_ids,
                memory_type=memory_type,
                similarity_threshold=self.config.similarity_threshold,
                limit=limit,
                time_range=time_range,
            )
        elif scope == "global":
            # No user/session restrictions
            memories = self._store.retrieve_memories(
                query_embedding=query_embedding,
                memory_type=memory_type,
                similarity_threshold=self.config.similarity_threshold,
                limit=limit,
                time_range=time_range,
            )
        else:
            # Fallback to basic retrieval
            memories = self._store.retrieve_memories(
                query_embedding=query_embedding,
                user_id=user_id,
                session_id=session_id,
                session_ids=session_ids,
                memory_type=memory_type,
                similarity_threshold=self.config.similarity_threshold,
                limit=limit,
                time_range=time_range,
            )

        # Update access counts
        for memory in memories:
            self._store.update_memory_access(memory.id)

        return memories

    def store_interaction(
        self,
        query: str,
        response: str,
        session_id: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        return self._store.store_interaction(session_id, query, response, metadata)

    def build_context_prompt(
        self,
        current_query: str,
        relevant_memories: List[MemoryEntry],
        include_recent: bool = True,
        max_context_length: int = 2000,
    ) -> str:
        if not relevant_memories:
            return ""

        context_parts = ["# Relevant Context from Memory:"]

        current_length = len(context_parts[0])

        # Sort memories by importance and recency
        sorted_memories = sorted(
            relevant_memories,
            key=lambda m: (m.importance_score, m.created_at),
            reverse=True,
        )

        for memory in sorted_memories:
            memory_text = f"\n- {memory.content}"
            if memory.memory_type != "general":
                memory_text += f" (Type: {memory.memory_type})"

            if current_length + len(memory_text) > max_context_length:
                break

            context_parts.append(memory_text)
            current_length += len(memory_text)

        context_parts.append("\n\n# Current Query:")
        return "".join(context_parts)

    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[SessionModel]:
        return self._store.get_user_sessions(user_id, limit)

    def get_user_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        memories = self._store.retrieve_memories(
            user_id=user_id, time_range=f"last_{days}_days", limit=100
        )

        if not memories:
            return {"summary": "No recent memories found", "memory_count": 0}

        # Group by memory type
        memory_types = {}
        for memory in memories:
            memory_type = memory.memory_type
            if memory_type not in memory_types:
                memory_types[memory_type] = []
            memory_types[memory_type].append(memory.content)

        return {
            "summary": f"User has {len(memories)} memories from the last {days} days",
            "memory_count": len(memories),
            "memory_types": memory_types,
            "most_recent": memories[0].content if memories else None,
            "period": f"last_{days}_days",
        }

    def _generate_embedding(self, text: str) -> List[float]:
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        response = self.openai_client.embeddings.create(
            model=self.config.embedding_model, input=text
        )
        return response.data[0].embedding
