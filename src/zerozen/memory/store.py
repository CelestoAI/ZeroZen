from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlmodel import SQLModel, create_engine, Session, select, and_
from sqlalchemy import desc
import numpy as np

from .models import MemoryEntry, SessionModel, InteractionModel
from .config import MemoryConfig


class MemoryStore(ABC):
    @abstractmethod
    def create_tables(self) -> None:
        pass

    @abstractmethod
    def store_memory(self, memory: MemoryEntry) -> str:
        pass

    @abstractmethod
    def retrieve_memories(
        self,
        query_embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        similarity_threshold: float = 0.7,
        limit: int = 10,
        time_range: Optional[str] = None,
    ) -> List[MemoryEntry]:
        pass

    @abstractmethod
    def create_session(
        self, session_id: str, user_id: str, metadata: Optional[Dict] = None
    ) -> SessionModel:
        pass

    @abstractmethod
    def store_interaction(
        self,
        session_id: str,
        user_query: str,
        llm_response: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        pass

    @abstractmethod
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[SessionModel]:
        pass

    @abstractmethod
    def update_memory_access(self, memory_id: str) -> None:
        pass


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.engine = create_engine(config.database_url, echo=False)
        self.create_tables()

    def create_tables(self) -> None:
        SQLModel.metadata.create_all(self.engine)

    def store_memory(self, memory: MemoryEntry) -> str:
        with Session(self.engine) as session:
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return memory.id

    def retrieve_memories(
        self,
        query_embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        similarity_threshold: float = 0.7,
        limit: int = 10,
        time_range: Optional[str] = None,
    ) -> List[MemoryEntry]:
        with Session(self.engine) as session:
            query = select(MemoryEntry)

            # Filter conditions
            conditions = []

            if user_id:
                conditions.append(MemoryEntry.user_id == user_id)

            if session_id:
                conditions.append(MemoryEntry.session_id == session_id)

            if session_ids:
                conditions.append(MemoryEntry.session_id.in_(session_ids))

            if memory_type:
                conditions.append(MemoryEntry.memory_type == memory_type)

            if time_range:
                cutoff_date = self._parse_time_range(time_range)
                if cutoff_date:
                    conditions.append(MemoryEntry.created_at >= cutoff_date)

            if conditions:
                query = query.where(and_(*conditions))

            # Order by importance and recency
            query = query.order_by(
                desc(MemoryEntry.importance_score), desc(MemoryEntry.created_at)
            ).limit(limit)

            memories = session.exec(query).all()

            # If we have query embedding, calculate similarity
            if query_embedding and memories:
                memories_with_similarity = []
                for memory in memories:
                    if memory.embedding:
                        similarity = self._cosine_similarity(
                            query_embedding, memory.embedding
                        )
                        if similarity >= similarity_threshold:
                            memories_with_similarity.append((memory, similarity))

                # Sort by similarity score
                memories_with_similarity.sort(key=lambda x: x[1], reverse=True)
                return [memory for memory, _ in memories_with_similarity]

            return list(memories)

    def create_session(
        self, session_id: str, user_id: str, metadata: Optional[Dict] = None
    ) -> SessionModel:
        session_model = SessionModel(id=session_id, user_id=user_id, metadata=metadata)

        with Session(self.engine) as session:
            session.add(session_model)
            session.commit()
            session.refresh(session_model)
            return session_model

    def store_interaction(
        self,
        session_id: str,
        user_query: str,
        llm_response: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        interaction = InteractionModel(
            session_id=session_id,
            user_query=user_query,
            llm_response=llm_response,
            metadata=metadata,
        )

        with Session(self.engine) as session:
            session.add(interaction)
            session.commit()
            session.refresh(interaction)
            return interaction.id

    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[SessionModel]:
        with Session(self.engine) as session:
            query = (
                select(SessionModel)
                .where(SessionModel.user_id == user_id)
                .order_by(desc(SessionModel.updated_at))
                .limit(limit)
            )

            return list(session.exec(query).all())

    def update_memory_access(self, memory_id: str) -> None:
        with Session(self.engine) as session:
            memory = session.get(MemoryEntry, memory_id)
            if memory:
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                session.add(memory)
                session.commit()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _parse_time_range(self, time_range: str) -> Optional[datetime]:
        now = datetime.now()

        if time_range == "last_hour":
            return now - timedelta(hours=1)
        elif time_range == "last_day":
            return now - timedelta(days=1)
        elif time_range == "last_7_days":
            return now - timedelta(days=7)
        elif time_range == "last_30_days":
            return now - timedelta(days=30)
        elif time_range.startswith("last_") and time_range.endswith("_days"):
            try:
                days = int(time_range.split("_")[1])
                return now - timedelta(days=days)
            except (ValueError, IndexError):
                return None

        return None
