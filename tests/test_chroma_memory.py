import pytest
import tempfile
import os
import shutil

from zerozen.memory import MemoryManager, MemoryConfig


class TestChromaMemorySystem:
    def setup_method(self):
        # Create a temporary directory for Chroma data
        self.temp_dir = tempfile.mkdtemp()

        self.config = MemoryConfig(
            store_type="chroma",
            database_url=self.temp_dir,
            embedding_model="text-embedding-ada-002",
            max_context_memories=5,
            similarity_threshold=0.7,
        )

        # Initialize without OpenAI to avoid API calls in tests
        self.memory = MemoryManager(self.config)
        self.memory.openai_client = None  # Disable embeddings for testing

    def teardown_method(self):
        # Clean up temporary Chroma directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_chroma_session_creation(self):
        session_id = self.memory.start_session("testuser")

        assert session_id is not None
        assert "testuser" in session_id
        assert len(session_id.split("_")) == 3  # user_id + uuid + timestamp

    def test_chroma_memory_storage(self):
        session_id = self.memory.start_session("test_user")

        memory_id = self.memory.store(
            content="User prefers Python",
            session_id=session_id,
            memory_type="preference",
            importance=0.8,
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)

    def test_chroma_memory_retrieval_by_session(self):
        session_id = self.memory.start_session("test_user")

        # Store multiple memories
        self.memory.store(
            content="User prefers Python",
            session_id=session_id,
            memory_type="preference",
            importance=0.8,
        )

        self.memory.store(
            content="User is working on AI project",
            session_id=session_id,
            memory_type="context",
            importance=0.9,
        )

        # Retrieve memories
        memories = self.memory.retrieve(
            query="programming preferences", session_id=session_id, scope="session"
        )

        assert len(memories) == 2
        assert any("Python" in m.content for m in memories)
        assert any("AI project" in m.content for m in memories)

    def test_chroma_memory_retrieval_by_user(self):
        session1 = self.memory.start_session("test_user")
        session2 = self.memory.start_session("test_user")

        # Store memories in different sessions
        self.memory.store(
            content="Memory from session 1", session_id=session1, memory_type="context"
        )

        self.memory.store(
            content="Memory from session 2", session_id=session2, memory_type="context"
        )

        # Retrieve across all user sessions
        memories = self.memory.retrieve(
            query="memory", user_id="test_user", scope="user"
        )

        assert len(memories) == 2
        session_ids = [m.session_id for m in memories]
        assert session1 in session_ids
        assert session2 in session_ids

    def test_chroma_memory_filtering_by_type(self):
        session_id = self.memory.start_session("test_user")

        # Store different types of memories
        self.memory.store(
            content="User likes Python", session_id=session_id, memory_type="preference"
        )

        self.memory.store(
            content="Technical discussion about APIs",
            session_id=session_id,
            memory_type="technical",
        )

        # Filter by type
        preferences = self.memory.retrieve(
            query="preferences",
            session_id=session_id,
            memory_type="preference",
            scope="session",
        )

        assert len(preferences) == 1
        assert preferences[0].memory_type == "preference"
        assert "Python" in preferences[0].content

    def test_chroma_interaction_storage(self):
        session_id = self.memory.start_session("test_user")

        interaction_id = self.memory.store_interaction(
            query="What is Python?",
            response="Python is a programming language",
            session_id=session_id,
        )

        assert interaction_id is not None
        assert isinstance(interaction_id, str)

    def test_chroma_memory_importance_ordering(self):
        session_id = self.memory.start_session("test_user")

        # Store memories with different importance scores
        self.memory.store(
            content="Low importance memory", session_id=session_id, importance=0.3
        )

        self.memory.store(
            content="High importance memory", session_id=session_id, importance=0.9
        )

        # Retrieve and check ordering
        memories = self.memory.retrieve(
            query="memory", session_id=session_id, scope="session"
        )

        # Should be ordered by importance (high to low)
        assert len(memories) == 2
        assert memories[0].importance_score >= memories[1].importance_score
        assert "High importance" in memories[0].content


@pytest.mark.skipif(
    True,  # Skip by default since ChromaDB might not be available in CI
    reason="ChromaDB integration test - requires chromadb package",
)
class TestChromaBackendSelection:
    def test_config_selects_chroma_backend(self):
        config = MemoryConfig(store_type="chroma", database_url="./test_chroma")

        memory = MemoryManager(config)

        # Verify that ChromaMemoryStore is selected
        from zerozen.memory.chroma_store import ChromaMemoryStore

        assert isinstance(memory._store, ChromaMemoryStore)

    def test_config_selects_sqlite_backend(self):
        config = MemoryConfig(store_type="sqlite", database_url="sqlite:///test.db")

        memory = MemoryManager(config)

        # Verify that SQLiteMemoryStore is selected
        from zerozen.memory.store import SQLiteMemoryStore

        assert isinstance(memory._store, SQLiteMemoryStore)
