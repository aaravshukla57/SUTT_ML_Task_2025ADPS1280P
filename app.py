"""
Streamlit Application for Hacker News Intelligence Tool

Integrates hn_client.py and llm_engine.py for fetching HN discussions
and generating structured analysis with RAG-powered Q&A.

Local Execution Mode: Uses LM Studio with a local LLM (no cloud APIs)
Run: streamlit run app.py
"""

import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests

# Local imports
from hn_client import fetch_and_clean_story
from llm_engine import (
    chunk_comments,
    generate_digest,
    CommentRAG,
    estimate_total_tokens,
    process_hn_data_with_llm,
    optimize_search_query,
    get_active_model_name,
)


# ============================================================================
# Configuration & Constants
# ============================================================================

DEFAULT_QUERY = "SQLite in production"
DEFAULT_MAX_COMMENTS = 2000
DEFAULT_MAX_TOKENS = 8000

# Page config
st.set_page_config(
    page_title="HN Intelligence Tool",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "fetched_data" not in st.session_state:
        st.session_state.fetched_data = None
    
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    
    if "digest" not in st.session_state:
        st.session_state.digest = None
    
    if "rag" not in st.session_state:
        st.session_state.rag = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "query" not in st.session_state:
        st.session_state.query = DEFAULT_QUERY
    
    if "max_comments" not in st.session_state:
        st.session_state.max_comments = DEFAULT_MAX_COMMENTS
    
    if "max_chunk_tokens" not in st.session_state:
        st.session_state.max_chunk_tokens = DEFAULT_MAX_TOKENS
    
    if "processing_error" not in st.session_state:
        st.session_state.processing_error = None
    
    if "lm_studio_connected" not in st.session_state:
        st.session_state.lm_studio_connected = False
    
    if "search_used_fallback" not in st.session_state:
        st.session_state.search_used_fallback = False

    if "optimized_query" not in st.session_state:
        st.session_state.optimized_query = None

    if "optimized_query_source" not in st.session_state:
        st.session_state.optimized_query_source = None

    if "active_model" not in st.session_state:
        st.session_state.active_model = "Local LLM"


# ============================================================================
# LM Studio Health Check
# ============================================================================

def check_lm_studio_running() -> bool:
    """
    Check if LM Studio server is running on localhost:1234.
    
    Returns:
        bool: True if LM Studio is accessible, False otherwise
    """
    try:
        response = requests.get(
            "http://localhost:1234/v1/models",
            timeout=2
        )
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
    except Exception:
        return False


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_hn_data(query: str, max_comments: int) -> Optional[Dict[str, Any]]:
    """
    Fetch HN stories with error handling and loading spinner.
    
    Args:
        query: Search query
        max_comments: Max comments per story
    
    Returns:
        Story data or None if error
    """
    try:
        with st.spinner("📡 Fetching from Hacker News..."):
            results, used_fallback = fetch_and_clean_story(
                query=query,
                story_limit=1,
                max_comments=max_comments,
            )
        
        # Store fallback flag in session state
        st.session_state.search_used_fallback = used_fallback
        
        if not results:
            return None
        
        return results[0]
    
    except Exception as e:
        st.error(f"Error fetching HN data: {str(e)}")
        return None


# ============================================================================
# LLM Processing
# ============================================================================

def process_with_llm(comments: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Process comments with local LM Studio for digest generation and RAG setup.
    
    Args:
        comments: List of cleaned comments
    
    Returns:
        Result dict with digest, chunks, RAG system or None if error
    """
    try:
        with st.spinner(f"Analyzing with {st.session_state.active_model}..."):
            result = process_hn_data_with_llm(
                comments,
                max_chunk_tokens=st.session_state.max_chunk_tokens,
                tokens_per_chunk=st.session_state.max_chunk_tokens,
                enable_rag=True,
            )
        
        if "error" in result["digest"]:
            st.error(f"LLM Error: {result['digest']['error']}")
            return None
        
        return result
    
    except Exception as e:
        st.error(f"Error processing with LLM: {str(e)}")
        return None


# ============================================================================
# UI Components
# ============================================================================

def sidebar_configuration():
    """Configure sidebar settings and check LM Studio connection."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # LM Studio Status
        st.subheader("🤖 Local Execution Mode")
        lm_studio_status = check_lm_studio_running()
        active_model = get_active_model_name()
        st.session_state.active_model = active_model
        st.sidebar.caption(f"Active Model: {active_model}")
        
        if lm_studio_status:
            st.success("✅ LM Studio is running")
            st.caption(f"Connected to localhost:1234 ({active_model})")
        else:
            st.error("⚠️ LM Studio is not running")
            with st.expander("📖 Setup Instructions", expanded=True):
                st.markdown("""
                **LM Studio is required. Follow these steps:**
                
                1. **Download LM Studio**
                   - Visit: https://lmstudio.ai/
                   - Download and install for your OS
                
                2. **Load a Model**
                   - Open LM Studio
                   - Search for a model compatible with your hardware
                   - Click download (⬇️) to load the model
                
                3. **Start Local Server**
                   - Go to the "Local Server" tab
                   - Click "Start Server" 
                   - Verify port is **1234**
                   - Wait for "Server is running"
                
                4. **Run This App**
                   - Return here and refresh (F5)
                   - Start analyzing!
                """)
        
        st.divider()
        
        # --- HARDWARE SYNC & SAFEGUARDS ---
        st.sidebar.subheader("⚙️ Hardware Sync")
        st.sidebar.caption("Match this to your LM Studio 'n_ctx' setting to prevent crashes.")

        # User inputs their LM Studio context window (Defaulting to 4096, max 128k)
        model_context_window = st.sidebar.number_input(
            "LM Studio Context Limit",
            min_value=2048, max_value=128000, value=4096, step=1024
        )

        # Reserve 1500 tokens for the system prompt + LLM response generation
        # The absolute maximum text chunk we can send is the context window MINUS the buffer
        SAFE_BUFFER = 1500
        max_safe_chunk = max(500, model_context_window - SAFE_BUFFER)

        st.sidebar.subheader("Search Settings")
        query = st.text_input(
            "What topic do you want to analyze?",
            value=st.session_state.query,
            placeholder="e.g., Python, React, Kubernetes...",
        )

        if (
            st.session_state.optimized_query
            and st.session_state.optimized_query_source == query
        ):
            st.info(
                f"Optimized query: **{st.session_state.optimized_query}** "
                f"(Original: {query})"
            )
        
        # Make the Chunk Size slider strictly bound by the dynamic max_safe_chunk
        tokens_per_chunk = st.sidebar.slider(
            "Tokens per chunk",
            min_value=500,
            max_value=max_safe_chunk,
            value=min(max_safe_chunk, 2500),
            help="Dynamically capped to prevent context overflow.",
        )

        max_comments = st.sidebar.slider(
            "Max comments to fetch",
            min_value=10,
            max_value=3000,
            value=200,
        )

        max_chunk_tokens = tokens_per_chunk
        
        # Info section
        st.subheader("ℹ️ About")
        st.markdown(f"""
        This tool analyzes Hacker News discussions using:
        - **HN Client**: Fetch & clean comments
        - **LLM Engine**: {st.session_state.active_model} analysis
        - **RAG System**: Question answering

        **Zero Cost, Zero API Keys, Complete Privacy** ?
        """)
        return query, max_comments, max_chunk_tokens, lm_studio_status, active_model


def display_story_info(story: Dict[str, Any]):
    """Display story metadata."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Score", story.get("score", "N/A"))
    
    with col2:
        st.metric("Author", story.get("author", "N/A"))
    
    with col3:
        url = story.get("url", "")
        if url:
            st.markdown(f"[🔗 View Source]({url})")
        else:
            st.text("No URL")
    
    with col4:
        if story.get("timestamp"):
            date = datetime.fromtimestamp(story["timestamp"]).strftime("%Y-%m-%d")
            st.text(f"📅 {date}")


def display_digest(digest: Dict[str, Any]):
    """Display structured digest in markdown format."""
    st.subheader("📊 Technical Digest")
    
    # Sentiment badge
    sentiment = digest.get("sentiment", "unknown")
    sentiment_colors = {
        "skeptical": "🔴",
        "hyped": "🟢",
        "pragmatic": "🟡",
        "mixed": "🟠",
        "unknown": "⚪",
    }
    emoji = sentiment_colors.get(sentiment, "⚪")
    st.markdown(f"### {emoji} Overall Sentiment: **{sentiment.capitalize()}**")
    if digest.get("sentiment_summary"):
        st.markdown(f"*{digest['sentiment_summary']}*")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "Arguments",
        "Pros & Cons",
        "Tools",
    ])
    
    with tab1:
        st.markdown("### Main Technical Arguments")
        arguments = digest.get("arguments", [])
        if arguments:
            for arg in arguments:
                st.markdown(f"- {arg}")
        else:
            st.info("No arguments extracted")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Pros ✅")
            pros = digest.get("pros", [])
            if pros:
                for pro in pros:
                    st.markdown(f"✓ {pro}")
            else:
                st.info("No pros extracted")
        
        with col2:
            st.markdown("### Cons ❌")
            cons = digest.get("cons", [])
            if cons:
                for con in cons:
                    st.markdown(f"✗ {con}")
            else:
                st.info("No cons extracted")
    
    with tab3:
        st.markdown("### Alternative Tools & Libraries")
        tools = digest.get("tools", [])
        if tools:
            for tool in tools:
                st.markdown(f"- {tool}")
        else:
            st.info("No tools mentioned")
    


def display_chat_interface():
    """Display chat interface for follow-up questions."""
    st.subheader("💬 Ask Follow-Up Questions")
    
    if not st.session_state.rag:
        st.warning("RAG system not initialized. Fetch a story first.")
        return

    user_question = None
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Chat History")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
        
        st.markdown("---")
    
    # Chat input
    user_question = st.chat_input("Ask a question about the discussion...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
        })
        
        # Get RAG answer
        with st.spinner("Searching comments..."):
            try:
                result = st.session_state.rag.answer_question(user_question)
                
                if "error" in result:
                    answer = f"Error: {result['error']}"
                else:
                    # Format answer with metadata
                    answer = result["answer"]
                    answer += f"\n\n---\n*Based on {result['relevant_chunks']} comment chunks*"
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                })
                
                # Display the new message
                with st.chat_message("assistant"):
                    st.markdown(answer)
                
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
        
        # Rerun to update chat display
        st.rerun()


def display_stats(fetched_data: Dict, result: Dict):
    """Display processing statistics."""
    st.subheader("📈 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        comments_count = len(fetched_data.get("comments", []))
        st.metric("Comments Fetched", comments_count)
    
    with col2:
        chunks_count = result["stats"].get("chunk_count", 0)
        st.metric("Chunks Created", chunks_count)
    
    with col3:
        st.metric("Execution", st.session_state.active_model)
    
    # Detailed breakdown
    with st.expander("💾 Detailed Breakdown"):
        st.json({
            "Comments": comments_count,
            "Chunks": chunks_count,
            "Execution Mode": result["stats"].get("execution_mode", "Local"),
            "Questions Asked": len([m for m in st.session_state.chat_history if m["role"] == "user"]),
        })


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application logic."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📰 Hacker News Intelligence Tool")
    st.markdown(
        f"Analyze tech discussions with {st.session_state.active_model} inference (zero cost, complete privacy)"
    )
    
    # Sidebar configuration and LM Studio check
    query, max_comments, max_chunk_tokens, lm_studio_running, active_model = sidebar_configuration()
    
    # Update session state
    st.session_state.query = query
    st.session_state.max_comments = max_comments
    st.session_state.max_chunk_tokens = max_chunk_tokens
    st.session_state.lm_studio_connected = lm_studio_running
    st.session_state.active_model = active_model
    
    # Halt if LM Studio is not running
    if not lm_studio_running:
        st.stop()
    
    # Main workflow
    st.markdown("---")
    
    # Step A: Fetch stories
    st.markdown("### Step A: Fetch Stories & Comments")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Query:** {query}")
    with col2:
        fetch_button = st.button("🔄 Fetch & Analyze", use_container_width=True)
    
    # Fetch data on button click
    if fetch_button or st.session_state.fetched_data is None:
        optimized_query = optimize_search_query(query)
        st.caption(f"Optimized: `{optimized_query}`")
        st.session_state.optimized_query = optimized_query
        st.session_state.optimized_query_source = query
        st.session_state.fetched_data = fetch_hn_data(optimized_query, max_comments)
        st.session_state.digest = None  # Reset digest when fetching new data
        st.session_state.rag = None
        st.session_state.chat_history = []  # Clear chat history
    if not st.session_state.fetched_data:
        st.error("No stories found. Try a different topic.")
        st.stop()
    elif st.session_state.search_used_fallback:
        st.warning("Strict search yielded 0 results. Using fallback.")

    if st.session_state.fetched_data:
        # Display story info
        story = st.session_state.fetched_data.get("story", {})
        comments = st.session_state.fetched_data.get("comments", [])

        raw_comments_text = "\n\n".join(
            comment.get("text", "") for comment in comments if comment.get("text")
        )
        estimated_total_tokens = len(raw_comments_text) // 4
        if estimated_total_tokens > st.session_state.max_chunk_tokens:
            st.warning(
                f"?? **Hardware Safeguard Triggered:** The full thread is ~{estimated_total_tokens} "
                f"tokens. Analyzed only the top {st.session_state.max_chunk_tokens} tokens to prevent "
                "local server crash."
            )
        if len(comments) == 0:
            st.error("Found a matching story, but it has 0 comments. Nothing to analyze.")
            st.stop()


        
        st.success(f"✅ Fetched story with {len(comments)} comments")
        
        with st.expander("📖 Story Details"):
            st.markdown(f"**Title:** {story.get('title', 'N/A')}")
            display_story_info(story)
        
        # Step B: Generate digest
        st.markdown("### Step B: Generate Technical Digest")
        
        if st.session_state.digest is None:
            result = process_with_llm(comments)
            
            if result:
                st.session_state.chunks = result["chunks"]
                st.session_state.digest = result["digest"]
                st.session_state.rag = result["rag"]
        
        if st.session_state.digest:
            st.success("✅ Digest generated successfully (local inference)")
            
            # Display digest
            display_digest(st.session_state.digest)
            
            # Display stats
            display_stats(st.session_state.fetched_data, {
                "digest": st.session_state.digest,
                "stats": {
                    "chunk_count": len(st.session_state.chunks),
                    "execution_mode": f"Local (LM Studio - {st.session_state.active_model})",
                },
            })
            
            # Step C: Chat interface
            st.markdown("### Step C: Ask Follow-Up Questions")
            display_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        **Execution**: Fully local with LM Studio + {st.session_state.active_model} ✅
        
        **Features**: Zero API costs, complete data privacy, unlimited rate limits, bound only by your hardware
        
        **Performance**: Speed depends on your CPU/GPU and model size
        """
    )


if __name__ == "__main__":
    main()
