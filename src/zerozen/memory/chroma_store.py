from typing import List, Optional, Dict
from datetime import datetime, timedelta
import uuid

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from .models import MemoryEntry, SessionModel, InteractionModel
from .store import MemoryStore
from .config import MemoryConfig


class ChromaMemoryStore(MemoryStore):
    def __init__(self, config: MemoryConfig):
        if chromadb is None:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        self.config = config

        # Initialize Chroma client
        if config.database_url.startswith("chroma://"):
            # Remote Chroma server
            host_port = config.database_url.replace("chroma://", "")
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                port = int(port)
            else:
                host, port = host_port, 8000

            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local persistent Chroma
            persist_directory = config.database_url.replace("sqlite://", "").replace(
                ".db", "_chroma"
            )
            self.client = chromadb.PersistentClient(path=persist_directory)

        # Create collections
        self.memories_collection = self._get_or_create_collection("memories")
        self.sessions_collection = self._get_or_create_collection("sessions")
        self.interactions_collection = self._get_or_create_collection("interactions")

        # In-memory session tracking (since Chroma is primarily for vectors)
        self._sessions = {}
        self._interactions = {}

    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except (
            ValueError,
            Exception,
        ):  # Catch broader exceptions for collection not found
            return self.client.create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )

    def create_tables(self) -> None:
        # No-op for Chroma - collections are created lazily
        pass

    def store_memory(self, memory: MemoryEntry) -> str:
        # Prepare metadata for Chroma
        metadata = {
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "memory_type": memory.memory_type,
            "importance_score": memory.importance_score,
            "created_at": memory.created_at.isoformat(),
            "access_count": memory.access_count,
        }

        # Add extra_data if present
        if memory.extra_data:
            # Flatten extra_data into metadata (Chroma doesn't support nested objects)
            for key, value in memory.extra_data.items():
                metadata[f"extra_{key}"] = (
                    str(value)
                    if not isinstance(value, (str, int, float, bool))
                    else value
                )

        # Store in Chroma
        self.memories_collection.add(
            documents=[memory.content],
            embeddings=[memory.embedding] if memory.embedding else None,
            metadatas=[metadata],
            ids=[memory.id],
        )

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
        # Build Chroma where clause - Chroma requires operators for multiple conditions
        conditions = []

        if user_id:
            conditions.append({"user_id": user_id})

        if session_id:
            conditions.append({"session_id": session_id})

        if session_ids:
            # Chroma doesn't support IN operator directly, so we'll handle this with multiple queries
            pass

        if memory_type:
            conditions.append({"memory_type": memory_type})

        # Build proper where clause
        if len(conditions) == 0:
            where_conditions = None
        elif len(conditions) == 1:
            where_conditions = conditions[0]
        else:
            where_conditions = {"$and": conditions}

        # Handle time range filtering
        if time_range:
            cutoff_date = self._parse_time_range(time_range)
            if cutoff_date:
                # Chroma doesn't have native date comparison, we'll filter post-query
                pass

        # Perform the search
        if session_ids and not session_id:
            # Handle multiple session IDs with separate queries
            all_results = []
            for sid in session_ids:
                # Build proper where clause for this session
                session_conditions = []
                if user_id:
                    session_conditions.append({"user_id": user_id})
                if memory_type:
                    session_conditions.append({"memory_type": memory_type})
                session_conditions.append({"session_id": sid})

                if len(session_conditions) == 1:
                    where_with_session = session_conditions[0]
                else:
                    where_with_session = {"$and": session_conditions}
                try:
                    if query_embedding:
                        results = self.memories_collection.query(
                            query_embeddings=[query_embedding],
                            where=where_with_session,
                            n_results=limit,
                        )
                    else:
                        # Fallback to getting all documents with filters
                        results = self.memories_collection.get(
                            where=where_with_session, limit=limit
                        )
                        # Convert get results to query format
                        if results["documents"]:
                            results = {
                                "documents": [results["documents"]],
                                "metadatas": [results["metadatas"]],
                                "ids": [results["ids"]],
                                "distances": [[0.0] * len(results["documents"])],
                            }

                    all_results.extend(
                        self._process_chroma_results(
                            results, similarity_threshold, time_range
                        )
                    )
                except Exception:
                    continue

            # Sort by importance and limit
            all_results.sort(
                key=lambda m: (m.importance_score, m.created_at), reverse=True
            )
            return all_results[:limit]

        else:
            # Single query
            try:
                if query_embedding:
                    results = self.memories_collection.query(
                        query_embeddings=[query_embedding],
                        where=where_conditions if where_conditions else None,
                        n_results=limit,
                    )
                else:
                    # Get all documents with filters
                    results = self.memories_collection.get(
                        where=where_conditions if where_conditions else None,
                        limit=limit,
                    )
                    # Convert get results to query format
                    if results["documents"]:
                        results = {
                            "documents": [results["documents"]],
                            "metadatas": [results["metadatas"]],
                            "ids": [results["ids"]],
                            "distances": [[0.0] * len(results["documents"])],
                        }

                return self._process_chroma_results(
                    results, similarity_threshold, time_range
                )

            except Exception:
                return []

    def _process_chroma_results(
        self, results: Dict, similarity_threshold: float, time_range: Optional[str]
    ) -> List[MemoryEntry]:
        memories = []

        if not results["documents"] or not results["documents"][0]:
            return memories

        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []
        distances = (
            results["distances"][0] if results["distances"] else [0.0] * len(documents)
        )

        cutoff_date = self._parse_time_range(time_range) if time_range else None

        for i, (doc, metadata, doc_id, distance) in enumerate(
            zip(documents, metadatas, ids, distances)
        ):
            # Check similarity threshold (distance is 1 - cosine_similarity for cosine distance)
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue

            # Parse metadata back to MemoryEntry
            try:
                created_at = datetime.fromisoformat(metadata["created_at"])

                # Check time range
                if cutoff_date and created_at < cutoff_date:
                    continue

                # Extract extra_data
                extra_data = {}
                for key, value in metadata.items():
                    if key.startswith("extra_"):
                        extra_data[key[6:]] = value  # Remove "extra_" prefix

                memory = MemoryEntry(
                    id=doc_id,
                    content=doc,
                    user_id=metadata["user_id"],
                    session_id=metadata["session_id"],
                    memory_type=metadata["memory_type"],
                    importance_score=float(metadata["importance_score"]),
                    extra_data=extra_data if extra_data else None,
                    created_at=created_at,
                    access_count=int(metadata.get("access_count", 0)),
                )
                memories.append(memory)

            except (KeyError, ValueError, TypeError):
                # Skip malformed entries
                continue

        # Sort by importance score (descending) then by created_at (descending)
        memories.sort(key=lambda m: (m.importance_score, m.created_at), reverse=True)

        return memories

    def create_session(
        self, session_id: str, user_id: str, metadata: Optional[Dict] = None
    ) -> SessionModel:
        session = SessionModel(id=session_id, user_id=user_id, session_data=metadata)

        # Store in memory (Chroma isn't ideal for non-vector session data)
        self._sessions[session_id] = session

        return session

    def store_interaction(
        self,
        session_id: str,
        user_query: str,
        llm_response: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        interaction_id = str(uuid.uuid4())

        interaction = InteractionModel(
            id=interaction_id,
            session_id=session_id,
            user_query=user_query,
            llm_response=llm_response,
            interaction_data=metadata,
        )

        # Store in memory for now
        self._interactions[interaction_id] = interaction

        return interaction_id

    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[SessionModel]:
        # Filter in-memory sessions by user_id
        user_sessions = [
            session for session in self._sessions.values() if session.user_id == user_id
        ]

        # Sort by updated_at and limit
        user_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return user_sessions[:limit]

    def update_memory_access(self, memory_id: str) -> None:
        try:
            # Get the current memory metadata
            result = self.memories_collection.get(
                ids=[memory_id], include=["metadatas"]
            )

            if result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["access_count"] = int(metadata.get("access_count", 0)) + 1
                metadata["last_accessed"] = datetime.now().isoformat()

                # Update in Chroma (requires re-adding with same ID)
                documents = self.memories_collection.get(
                    ids=[memory_id], include=["documents", "embeddings"]
                )

                if documents["documents"]:
                    self.memories_collection.upsert(
                        documents=documents["documents"],
                        embeddings=documents["embeddings"]
                        if documents["embeddings"]
                        else None,
                        metadatas=[metadata],
                        ids=[memory_id],
                    )

        except Exception:
            # Fail silently for access count updates
            pass

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
