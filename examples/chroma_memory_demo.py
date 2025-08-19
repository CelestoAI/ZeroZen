"""
Chroma Vector Database Memory Demo for ZeroZen

This example demonstrates using ChromaDB as the backend for LLM memory,
which provides optimized vector similarity search and better performance
for semantic memory retrieval.
"""

import os
from dotenv import load_dotenv

from zerozen.memory import MemoryManager, MemoryConfig

# Load environment variables
load_dotenv()


def main():
    print("🧠 Initializing Chroma Memory System...")

    # Initialize memory system with Chroma backend
    config = MemoryConfig(
        store_type="chroma",
        database_url="./chroma_memory_demo",  # Local Chroma storage path
        embedding_model="text-embedding-ada-002",
        max_context_memories=10,
        similarity_threshold=0.7,
    )

    try:
        memory = MemoryManager(config)
        print("✅ Chroma memory system initialized!")
    except ImportError as e:
        print(f"❌ ChromaDB not available: {e}")
        print("Install ChromaDB with: pip install chromadb")
        return

    # Start a conversation session
    print("\n=== Starting Session ===")
    session_id = memory.start_session(
        user_id="chroma_user", metadata={"demo": "chroma"}
    )
    print(f"Session ID: {session_id}")

    # Store various types of memories
    print("\n=== Storing Memories in Chroma ===")

    memories_to_store = [
        {
            "content": "User is building an AI-powered personal assistant with memory capabilities",
            "type": "project_context",
            "importance": 0.95,
        },
        {
            "content": "User prefers TypeScript over JavaScript for large applications",
            "type": "preference",
            "importance": 0.8,
        },
        {
            "content": "Previous discussion about vector databases like Chroma and Qdrant",
            "type": "technical_discussion",
            "importance": 0.9,
        },
        {
            "content": "User is interested in implementing semantic search for better memory retrieval",
            "type": "interest",
            "importance": 0.85,
        },
        {
            "content": "User mentioned using SQLite initially but wanting to scale to vector DBs",
            "type": "architecture_decision",
            "importance": 0.7,
        },
        {
            "content": "Team is using OpenAI embeddings for semantic similarity",
            "type": "technical_detail",
            "importance": 0.6,
        },
    ]

    stored_ids = []
    for mem in memories_to_store:
        memory_id = memory.store(
            content=mem["content"],
            session_id=session_id,
            memory_type=mem["type"],
            importance=mem["importance"],
            metadata={"demo": True},
        )
        stored_ids.append(memory_id)
        print(f"✅ Stored: {mem['content'][:50]}...")

    print(f"\n📊 Stored {len(stored_ids)} memories in Chroma vector database")

    # Test semantic similarity search
    print("\n=== Testing Semantic Search ===")

    test_queries = [
        "What database technologies were discussed?",
        "What are the user's programming language preferences?",
        "Tell me about the AI project",
        "What architectural decisions were made?",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")

        # Retrieve similar memories
        relevant_memories = memory.retrieve(
            query=query, session_id=session_id, scope="session", limit=3
        )

        if relevant_memories:
            print("   📋 Most relevant memories:")
            for i, mem in enumerate(relevant_memories, 1):
                print(f"      {i}. [{mem.memory_type}] {mem.content}")
                print(f"         Importance: {mem.importance_score:.2f}")
        else:
            print("   ❌ No relevant memories found")

    # Test cross-session retrieval
    print("\n=== Testing Cross-Session Memory ===")
    session2_id = memory.start_session(user_id="chroma_user", metadata={"session": 2})

    # Store a memory in the new session
    memory.store(
        content="User wants to integrate Chroma with existing SQLite system",
        session_id=session2_id,
        memory_type="integration_plan",
        importance=0.9,
    )

    # Search across all user sessions
    cross_session_memories = memory.retrieve(
        query="vector database integration",
        user_id="chroma_user",
        scope="user",
        limit=4,
    )

    print(f"🔗 Found {len(cross_session_memories)} memories across sessions:")
    for i, mem in enumerate(cross_session_memories, 1):
        session_indicator = (
            "📝 Current" if mem.session_id == session2_id else "📋 Previous"
        )
        print(f"   {i}. {session_indicator} [{mem.memory_type}] {mem.content[:60]}...")

    # Build context for LLM
    print("\n=== Building LLM Context ===")
    context_query = "How should we implement vector search in our memory system?"
    context_memories = memory.retrieve(
        query=context_query, user_id="chroma_user", scope="user", limit=5
    )

    context_prompt = memory.build_context_prompt(
        current_query=context_query, relevant_memories=context_memories
    )

    print("🤖 Generated context for LLM:")
    print("-" * 60)
    print(context_prompt)
    print(context_query)
    print("-" * 60)

    # Performance comparison note
    print("\n=== Chroma vs SQLite Benefits ===")
    print("🚀 Chroma Advantages:")
    print("   • Optimized vector similarity search (HNSW indexing)")
    print("   • Better performance for large-scale memory retrieval")
    print("   • Native embedding support")
    print("   • Scalable to millions of memories")
    print("   • Advanced filtering capabilities")

    print("\n💾 SQLite Advantages:")
    print("   • Simple deployment (single file)")
    print("   • No external dependencies")
    print("   • ACID transactions")
    print("   • Better for small-scale applications")

    print(f"\n🎉 Chroma demo completed! Vector data stored in: {config.database_url}")


if __name__ == "__main__":
    # Check dependencies
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Embeddings will be disabled.")
        print("   Set your OpenAI API key to enable semantic search.")
        print("   The demo will still work but without vector similarity.\n")

    main()
