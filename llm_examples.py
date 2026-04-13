"""
Examples demonstrating LLM Engine integration with HN Client.

Shows:
1. Fetch HN data with hn_client
2. Process with llm_engine for structured analysis  
3. Use RAG for follow-up questions
4. Monitor token usage for cost optimization
"""

import os
from typing import List, Dict, Any
from hn_client import fetch_and_clean_story
from llm_engine import (
    chunk_comments,
    generate_digest,
    process_hn_data_with_llm,
    CommentRAG,
    estimate_total_tokens,
)


def example_1_basic_digest():
    """Example 1: Fetch HN story and generate digest."""
    print("=" * 80)
    print("Example 1: Fetch HN Data and Generate Digest")
    print("=" * 80)

    # Step 1: Fetch story from Hacker News
    print("\n1. Fetching story from Hacker News...")
    try:
        results = fetch_and_clean_story("python", story_limit=1, max_comments=30)
        if not results:
            print("   No stories found")
            return

        comments = results[0]["comments"]
        story = results[0]["story"]
        print(f"   Story: {story.get('title', 'N/A')}")
        print(f"   Comments fetched: {len(comments)}")
    except Exception as e:
        print(f"   Error fetching: {e}")
        return

    # Step 2: Chunk comments by thread
    print("\n2. Chunking comments by thread...")
    chunks = chunk_comments(comments, max_tokens=2000)
    print(f"   Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"     Chunk {i}: {len(chunk.comment_ids)} comments, {chunk.token_count} tokens, branch: {chunk.branch_id}")

    # Step 3: Generate digest (requires OpenAI API key)
    print("\n3. Generating digest...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   SKIPPED: Set OPENAI_API_KEY environment variable to enable")
        print("   Example: export OPENAI_API_KEY=sk-...")
        return

    try:
        digest = generate_digest(chunks, api_key=api_key)

        if "error" in digest:
            print(f"   Error: {digest['error']}")
            return

        print(f"   Sentiment: {digest.get('sentiment', 'unknown')}")
        print(f"   Tokens used: {digest.get('total_tokens', 'N/A')}")

        print("\n   Main Arguments:")
        for arg in digest.get("technical_arguments", [])[:3]:
            print(f"     - {arg}")

        print("\n   Pros:")
        for pro in digest.get("pros", [])[:3]:
            print(f"     - {pro}")

        print("\n   Cons:")
        for con in digest.get("cons", [])[:3]:
            print(f"     - {con}")

        print("\n   Tools Mentioned:")
        for tool in digest.get("alternative_tools", [])[:3]:
            print(f"     - {tool}")

    except Exception as e:
        print(f"   Error generating digest: {e}")


def example_2_token_optimization():
    """Example 2: Analyze token usage and optimize chunks."""
    print("\n" + "=" * 80)
    print("Example 2: Token Usage Optimization")
    print("=" * 80)

    # Fetch data
    print("\n1. Fetching HN story...")
    try:
        results = fetch_and_clean_story("rust", story_limit=1, max_comments=50)
        if not results:
            print("   No stories found")
            return

        comments = results[0]["comments"]
        story = results[0]["story"]
        print(f"   Story: {story.get('title', 'N/A')}")
        print(f"   Total comments: {len(comments)}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Test different chunk sizes
    print("\n2. Testing different chunk sizes...")
    chunk_configs = [
        ("Small (1000)", 1000),
        ("Medium (2000)", 2000),
        ("Large (4000)", 4000),
    ]

    for name, max_tokens in chunk_configs:
        chunks = chunk_comments(comments, max_tokens=max_tokens)
        total_tokens = estimate_total_tokens(chunks)
        avg_tokens = total_tokens // max(len(chunks), 1)

        # Estimate API cost (rough: 0.0015 per 1K tokens)
        estimated_cost = (total_tokens / 1000) * 0.0015

        print(f"\n   {name}:")
        print(f"     Chunks: {len(chunks)}")
        print(f"     Total tokens: {total_tokens}")
        print(f"     Avg per chunk: {avg_tokens}")
        print(f"     Est. cost: ${estimated_cost:.4f}")

    # Recommend optimal config
    optimal_chunks = chunk_comments(comments, max_tokens=2000)
    optimal_tokens = estimate_total_tokens(optimal_chunks)
    print(f"\n   Recommended: 2000 tokens/chunk")
    print(f"   Total chunks: {len(optimal_chunks)}")
    print(f"   Total tokens: {optimal_tokens}")


def example_3_rag_follow_up():
    """Example 3: Use RAG for follow-up questions."""
    print("\n" + "=" * 80)
    print("Example 3: RAG System for Follow-Up Questions")
    print("=" * 80)

    # Fetch data
    print("\n1. Fetching HN story...")
    try:
        results = fetch_and_clean_story("machine learning", story_limit=1, max_comments=40)
        if not results:
            print("   No stories found")
            return

        comments = results[0]["comments"]
        story = results[0]["story"]
        print(f"   Story: {story.get('title', 'N/A')}")
        print(f"   Comments: {len(comments)}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Chunk and setup RAG
    print("\n2. Setting up RAG system...")
    chunks = chunk_comments(comments, max_tokens=2000)
    rag = CommentRAG(chunks, api_key=os.getenv("OPENAI_API_KEY"))
    print(f"   RAG ready with {len(chunks)} chunks")

    # Ask follow-up questions (without LLM)
    print("\n3. Simulating follow-up questions...")
    questions = [
        "What tools and frameworks were recommended?",
        "What are the main concerns discussed?",
        "Is there disagreement among commenters?",
    ]

    for q in questions:
        print(f"\n   Q: {q}")
        relevant_chunks = rag._simple_similarity_search(q, top_k=2)
        print(f"   Found in {len(relevant_chunks)} chunks")
        for chunk in relevant_chunks:
            print(f"     - {len(chunk.comment_ids)} comments, branch: {chunk.branch_id}")

    # Full answer (requires OpenAI)
    print("\n4. Getting answers from LLM...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   SKIPPED: Set OPENAI_API_KEY to enable")
        return

    try:
        # Ask a specific question
        test_question = "What programming languages were discussed?"
        result = rag.answer_question(test_question)

        if "error" in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"\n   Q: {test_question}")
            print(f"   A: {result['answer'][:200]}...")
            print(f"\n   Metadata:")
            print(f"     Source chunks: {result.get('relevant_chunks', 0)}")
            print(f"     Input tokens: {result.get('input_tokens', 0)}")
            print(f"     Output tokens: {result.get('output_tokens', 0)}")

    except Exception as e:
        print(f"   Error: {e}")


def example_4_end_to_end():
    """Example 4: End-to-end pipeline with everything integrated."""
    print("\n" + "=" * 80)
    print("Example 4: End-to-End Pipeline")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")

    # Fetch story
    print("\n1. Fetching HN story...")
    try:
        results = fetch_and_clean_story("golang", story_limit=1, max_comments=40)
        if not results:
            print("   No stories found")
            return

        comments = results[0]["comments"]
        story = results[0]["story"]
        print(f"   Story: {story.get('title', 'N/A')}")
        print(f"   Author: {story.get('author')}")
        print(f"   Score: {story.get('score')}")
        print(f"   Comments: {len(comments)}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Process with LLM engine
    print("\n2. Processing with LLM engine...")
    try:
        result = process_hn_data_with_llm(
            comments,
            api_key=api_key,
            max_chunk_tokens=2000,
            enable_rag=True,
        )

        outline = result["digest"]
        stats = result["stats"]

        print(f"   Chunks created: {stats['chunk_count']}")
        print(f"   Total tokens : {stats['total_tokens_for_digest']}")
        print(f"   Avg per chunk: {stats['avg_tokens_per_chunk']}")

        # Show digest summary
        if "error" not in outline:
            print(f"\n   Digest Summary:")
            print(f"   Sentiment: {outline.get('sentiment', 'unknown')}")
            print(f"   Main arguments: {len(outline.get('technical_arguments', []))}")
            print(f"   Pros mentioned: {len(outline.get('pros', []))}")
            print(f"   Cons mentioned: {len(outline.get('cons', []))}")
            print(f"   Tools mentioned: {len(outline.get('alternative_tools', []))}")

            # Show available RAG queries
            if result["rag"]:
                print(f"\n   RAG System Ready - Ask questions like:")
                print(f"   - 'What tools were recommended?'")
                print(f"   - 'What are the main pros and cons?'")
                print(f"   - 'Is there disagreement on this topic?'")

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


def example_5_cost_comparison():
    """Example 5: Compare cost of different chunking strategies."""
    print("\n" + "=" * 80)
    print("Example 5: Cost Comparison of Chunking Strategies")
    print("=" * 80)

    # Fetch large dataset
    print("\n1. Fetching large HN thread...")
    try:
        results = fetch_and_clean_story("web", story_limit=1, max_comments=100)
        if not results:
            print("   No stories found")
            return

        comments = results[0]["comments"]
        print(f"   Fetched {len(comments)} comments")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Compare strategies
    print("\n2. Comparing chunking strategies...")

    strategies = {
        "Minimal Chunking (4000 tokens)": 4000,
        "Balanced Chunking (2000 tokens)": 2000,
        "Fine-Grained (1000 tokens)": 1000,
    }

    pricing = {
        "input": 0.0015 / 1000,  # $0.0015 per 1K tokens
        "output": 0.002 / 1000,  # $0.002 per 1K tokens
    }

    for name, max_tokens in strategies.items():
        chunks = chunk_comments(comments, max_tokens=max_tokens)
        total_tokens = estimate_total_tokens(chunks)

        # Estimate input cost
        input_cost = (total_tokens * pricing["input"])

        # Estimate output (rough: 30% of input)
        output_tokens = int(total_tokens * 0.3)
        output_cost = (output_tokens * pricing["output"])
        total_cost = input_cost + output_cost

        print(f"\n   {name}:")
        print(f"     Chunks: {len(chunks)}")
        print(f"     Total tokens: {total_tokens}")
        print(f"     Est. input cost: ${input_cost:.4f}")
        print(f"     Est. output cost: ${output_cost:.4f}")
        print(f"     Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    print("\n")
    print("HN INTELLIGENCE TOOL - LLM ENGINE EXAMPLES")
    print("=" * 80)
    print("\nThese examples show how to:")
    print("  1. Fetch HN stories and comments")
    print("  2. Chunk comments intelligently by thread")
    print("  3. Generate structured digests using LLM")
    print("  4. Use RAG for follow-up questions")
    print("  5. Optimize for token usage and costs")
    print("\nTo enable LLM features:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  pip install langchain openai tiktoken")
    print()

    try:
        # Run examples
        example_1_basic_digest()
        example_2_token_optimization()
        example_3_rag_follow_up()
        # example_4_end_to_end()
        # example_5_cost_comparison()

        print("\n" + "=" * 80)
        print("Examples completed!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
