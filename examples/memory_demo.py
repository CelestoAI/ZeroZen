"""
Memory Layer Demo for ZeroZen LLM Integration

This example demonstrates how to use the memory layer to store and retrieve
context across LLM conversations.
"""

import os
from dotenv import load_dotenv

from zerozen.memory import MemoryManager, MemoryConfig

# Load environment variables
load_dotenv()


def main():
    # Initialize memory system
    config = MemoryConfig(
        database_url="sqlite:///./memory_demo.db",
        embedding_model="text-embedding-ada-002",
        max_context_memories=10,
        similarity_threshold=0.7,
    )

    memory = MemoryManager(config)
    print("🧠 Memory system initialized!")

    # Simulate a conversation session
    print("\n=== Starting Session ===")
    session_id = memory.start_session(user_id="demo_user", metadata={"demo": True})
    print(f"Session ID: {session_id}")

    # Store some memories during conversation
    print("\n=== Storing Memories ===")

    memory.store(
        content="User prefers Python over JavaScript for backend development",
        session_id=session_id,
        memory_type="preference",
        importance=0.8,
    )
    print("✅ Stored preference about Python")

    memory.store(
        content="User is working on an LLM memory system called ZeroZen",
        session_id=session_id,
        memory_type="context",
        importance=0.9,
    )
    print("✅ Stored project context")

    memory.store(
        content="User mentioned they use VS Code as their primary editor",
        session_id=session_id,
        memory_type="preference",
        importance=0.6,
    )
    print("✅ Stored editor preference")

    memory.store(
        content="Discussion about SQLModel and database design patterns",
        session_id=session_id,
        memory_type="technical",
        importance=0.7,
    )
    print("✅ Stored technical discussion")

    # Store an interaction
    memory.store_interaction(
        query="What's the best way to implement memory in LLMs?",
        response="You can use vector embeddings with similarity search...",
        session_id=session_id,
    )
    print("✅ Stored Q&A interaction")

    # Retrieve relevant memories
    print("\n=== Retrieving Memories ===")

    # Test 1: Session-scoped retrieval
    print("\n1. Session-scoped search:")
    query = "What programming language should I use for my API?"
    relevant_memories = memory.retrieve(
        query=query, session_id=session_id, scope="session", limit=3
    )

    for i, mem in enumerate(relevant_memories, 1):
        print(f"   {i}. [{mem.memory_type}] {mem.content}")

    # Test 2: User-scoped retrieval (across sessions)
    print("\n2. User-scoped search:")
    user_memories = memory.retrieve(
        query="development preferences", user_id="demo_user", scope="user", limit=5
    )

    for i, mem in enumerate(user_memories, 1):
        print(f"   {i}. [{mem.memory_type}] {mem.content}")

    # Test 3: Type-specific retrieval
    print("\n3. Preference-specific search:")
    preferences = memory.retrieve(
        query="user preferences",
        session_id=session_id,
        memory_type="preference",
        scope="session",
    )

    for i, mem in enumerate(preferences, 1):
        print(f"   {i}. {mem.content}")

    # Build context for LLM
    print("\n=== Building LLM Context ===")
    context_prompt = memory.build_context_prompt(
        current_query=query, relevant_memories=relevant_memories, include_recent=True
    )

    print("Generated context prompt:")
    print("-" * 50)
    print(context_prompt)
    print("-" * 50)

    # Get user summary
    print("\n=== User Summary ===")
    summary = memory.get_user_summary("demo_user", days=1)
    print(f"Summary: {summary['summary']}")
    print(f"Memory count: {summary['memory_count']}")
    if summary.get("memory_types"):
        print("Memory types:")
        for mem_type, contents in summary["memory_types"].items():
            print(f"  - {mem_type}: {len(contents)} memories")

    print("\n🎉 Demo completed! Check memory_demo.db for stored data.")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Embeddings will be disabled.")
        print("   Set your OpenAI API key to enable semantic search.")
        print("   The demo will still work with basic text matching.\n")

    main()
