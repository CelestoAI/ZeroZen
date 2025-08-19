import tempfile
import os

from zerozen.memory import MemoryManager, MemoryConfig


class TestMemorySystem:
    def setup_method(self):
        # Create a temporary database for each test
        self.db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_file.close()

        self.config = MemoryConfig(
            database_url=f"sqlite:///{self.db_file.name}",
            embedding_model="text-embedding-ada-002",
            max_context_memories=5,
            similarity_threshold=0.7,
        )

        # Initialize without OpenAI to avoid API calls in tests
        self.memory = MemoryManager(self.config)
        self.memory.openai_client = None  # Disable embeddings for testing

    def teardown_method(self):
        # Clean up temporary database
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)

    def test_session_creation(self):
        session_id = self.memory.start_session("testuser")

        assert session_id is not None
        assert "testuser" in session_id
        assert len(session_id.split("_")) == 3  # user_id + uuid + timestamp

    def test_memory_storage(self):
        session_id = self.memory.start_session("test_user")

        memory_id = self.memory.store(
            content="User prefers Python",
            session_id=session_id,
            memory_type="preference",
            importance=0.8,
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)

    def test_memory_retrieval_by_session(self):
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

    def test_memory_retrieval_by_user(self):
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

    def test_memory_filtering_by_type(self):
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

    def test_interaction_storage(self):
        session_id = self.memory.start_session("test_user")

        interaction_id = self.memory.store_interaction(
            query="What is Python?",
            response="Python is a programming language",
            session_id=session_id,
        )

        assert interaction_id is not None
        assert isinstance(interaction_id, str)

    def test_context_prompt_building(self):
        session_id = self.memory.start_session("test_user")

        # Store memories
        self.memory.store(
            content="User prefers Python",
            session_id=session_id,
            memory_type="preference",
            importance=0.9,
        )

        self.memory.store(
            content="User is building an AI system",
            session_id=session_id,
            memory_type="context",
            importance=0.8,
        )

        # Retrieve memories
        memories = self.memory.retrieve(
            query="development", session_id=session_id, scope="session"
        )

        # Build context
        context = self.memory.build_context_prompt(
            current_query="What should I use for my project?",
            relevant_memories=memories,
        )

        assert "# Relevant Context from Memory:" in context
        assert "# Current Query:" in context
        assert "Python" in context or "AI system" in context

    def test_user_summary(self):
        session_id = self.memory.start_session("test_user")

        # Store some memories
        self.memory.store(
            content="User prefers Python",
            session_id=session_id,
            memory_type="preference",
        )

        self.memory.store(
            content="Technical discussion",
            session_id=session_id,
            memory_type="technical",
        )

        # Get summary
        summary = self.memory.get_user_summary("test_user", days=1)

        assert summary["memory_count"] == 2
        assert "memory_types" in summary
        assert "preference" in summary["memory_types"]
        assert "technical" in summary["memory_types"]

    def test_memory_importance_ordering(self):
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


class TestMemoryConfig:
    def test_default_config(self):
        config = MemoryConfig()

        assert config.database_url == "sqlite:///./llm_memory.db"
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.max_context_memories == 10
        assert config.similarity_threshold == 0.7
        assert config.embedding_dimension == 1536

    def test_custom_config(self):
        config = MemoryConfig(
            database_url="postgresql://localhost/test",
            max_context_memories=20,
            similarity_threshold=0.8,
        )

        assert config.database_url == "postgresql://localhost/test"
        assert config.max_context_memories == 20
        assert config.similarity_threshold == 0.8
