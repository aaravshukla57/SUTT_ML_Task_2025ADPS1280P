"""
LLM Engine Module for Hacker News Intelligence

Provides:
- Context-aware comment chunking by thread/branch
- Structured digest generation using LangChain
- Basic RAG (Retrieval Augmented Generation) for follow-up questions
- Token counting and optimization for cost savings

Dependencies: langchain, openai, tiktoken
"""

import json
import re
import requests
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# Configuration & Constants
# ============================================================================

DEFAULT_MAX_TOKENS = 2000
DEFAULT_MODEL = "gpt-3.5-turbo"


# ============================================================================
# Pydantic Schema for Structured Output
# ============================================================================

if PYDANTIC_AVAILABLE:
    class DigestSchema(BaseModel):
        """Structured digest output schema for HN analysis."""
        technical_arguments: List[str] = Field(
            default_factory=list,
            description="Key claims and technical positions discussed (max 5)"
        )
        pros: List[str] = Field(
            default_factory=list,
            description="Specific advantages and benefits mentioned (max 5)"
        )
        cons: List[str] = Field(
            default_factory=list,
            description="Specific disadvantages and drawbacks mentioned (max 5)"
        )
        alternative_tools: List[str] = Field(
            default_factory=list,
            description="Tools and libraries explicitly mentioned (max 5)"
        )
        sentiment: str = Field(
            default="unknown",
            description="Overall sentiment: skeptical, hyped, pragmatic, or mixed"
        )
        sentiment_summary: str = Field(
            default="",
            description="1-2 sentence explanation of sentiment (max 200 chars)"
        )


# ============================================================================
# Aggressive System Prompt (Local LLM Optimized)
# ============================================================================

SYSTEM_PROMPT_RESEARCHER = """You are a strict data extraction system. Your ONLY job is to extract structured data from Hacker News discussions.

CRITICAL INSTRUCTIONS:
- Output ONLY valid JSON. NOT MARKDOWN. NOT CODE BLOCKS.
- Do NOT include: backticks, triple quotes, markdown formatting, or conversational text.
- Start your response IMMEDIATELY with { and end with }
- If you cannot output valid JSON, the system will fail.

Extract these EXACTLY from the comments:

1. "technical_arguments": Array of 1-2 sentence factual claims discussed (max 5 items)
2. "pros": Array of specific, concrete benefits mentioned (max 5 items)
3. "cons": Array of specific, concrete drawbacks mentioned (max 5 items)
4. "alternative_tools": Array of tools/libraries explicitly named (max 5 items)
5. "sentiment": One of: "skeptical", "hyped", "pragmatic", or "mixed"
6. "sentiment_summary": 1-2 sentences explaining the sentiment (max 200 chars)

REQUIRED JSON FORMAT (start immediately, no preamble):
{
    "technical_arguments": ["claim1", "claim2"],
    "pros": ["benefit1", "benefit2"],
    "cons": ["drawback1", "drawback2"],
    "alternative_tools": ["tool1", "tool2"],
    "sentiment": "pragmatic",
    "sentiment_summary": "Community values practical tradeoffs."
}

Replace the examples with actual extracted data.
Be PRECISE. Extract ONLY what was actually discussed in the comments.
START WITH { IMMEDIATELY."""


# ============================================================================
# Robust JSON Extraction with Debug Output
# ============================================================================

def extract_json_from_response(raw_text: str, debug: bool = True) -> Dict[str, Any]:
    """
    Bulletproof JSON extraction from messy LLM outputs.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Conversational preamble/postamble
    - Trailing commas and syntax errors
    - Debug output if extraction fails
    
    Args:
        raw_text (str): Raw text from LLM
        debug (bool): Print debug info if extraction fails
    
    Returns:
        Dict: Extracted JSON or empty dict if failed
    """
    if not raw_text:
        return {}
    
    # Strategy 1: Try direct JSON parse (happy path)
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks (```json ... ```)
    try:
        md_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', raw_text, re.DOTALL)
        if md_match:
            json_str = md_match.group(1)
            return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Strategy 3: Extract first { to last } (most robust)
    try:
        # Find first { and last }
        first_brace = raw_text.find('{')
        last_brace = raw_text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_text[first_brace:last_brace + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Try to clean and parse (remove trailing commas, etc.)
    try:
        # Remove markdown backticks
        cleaned = re.sub(r'```.*?```', '', raw_text, flags=re.DOTALL)
        # Extract JSON region
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1:
            json_str = cleaned[first_brace:last_brace + 1]
            # Remove trailing commas before } or ]
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # All extraction failed - debug output
    if debug:
        print("\n" + "="*80)
        print("JSON EXTRACTION FAILED!")
        print("="*80)
        print("RAW MODEL OUTPUT (first 2000 chars):")
        print(raw_text[:2000])
        print("="*80)
        print("ATTEMPTING STRATEGIES:")
        print(f"- Direct JSON parse: FAILED")
        print(f"- Markdown extraction: FAILED")
        print(f"- First {{ to last }}: FAILED")
        print(f"- Cleaned extraction: FAILED")
        print("="*80)
    
    return {}


# ============================================================================
# LM Studio Local LLM Setup
# ============================================================================

def get_local_llm(temperature: float = 0.1):
    """
    Initialize local Llama 3 via LM Studio on localhost:1234.
    
    This function connects to a locally-running LM Studio server that must be:
    1. Running LM Studio application
    2. Model loaded: Meta Llama 3 8B Instruct (or compatible)
    3. Local Inference Server started on Port 1234
    
    Args:
        temperature (float): Sampling temperature (0.0-1.0). Recommend 0.1 for factual extraction.
    
    Returns:
        ChatOpenAI: LangChain chat model pointing to local LM Studio
    
    Raises:
        Exception: If LM Studio server is not running on localhost:1234
        
    Example:
        >>> llm = get_local_llm(temperature=0.1)
        >>> # LM Studio must be running and model loaded on port 1234
    """
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",  # LM Studio requires dummy API key
        model="meta-llama-3-8b-instruct",  # Model name (must be loaded in LM Studio)
        temperature=temperature,
    )


def get_active_model_name(base_url: str = "http://localhost:1234/v1") -> str:
    """Dynamically fetches the loaded model name from LM Studio."""
    try:
        response = requests.get(f"{base_url}/models", timeout=2)
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                return models[0].get("id", "Local LLM")
    except Exception:
        pass
    return "Local LLM (Disconnected/Unknown)"


def _fallback_optimize_search_query(user_query: str) -> str:
    """
    Simple keyword extraction fallback when LangChain/LM Studio is unavailable.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "what", "how", "why", "when", "where", "which",
        "do", "does", "did", "can", "could", "should", "would", "to", "for", "of",
        "and", "or", "in", "on", "with", "about", "from", "by", "this", "that",
    }
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", user_query.lower())
    keywords = []
    for word in words:
        if word in stop_words:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= 3:
            break
    return " ".join(keywords) if keywords else user_query.strip()


@st.cache_data(ttl=3600)
def optimize_search_query(user_query: str) -> str:
    """
    Optimize a conversational query into 2-3 technical keywords for search.

    Uses local LM Studio via LangChain when available, with a robust fallback.
    """
    if not user_query or not user_query.strip():
        return ""

    if not LANGCHAIN_AVAILABLE:
        return _fallback_optimize_search_query(user_query)

    try:
        chat = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="meta-llama-3-8b-instruct",
            temperature=0,
        )

        query_prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Hacker News search query generator. Convert the user's conversational question "
                "into 2-3 broad, high-level technical keywords that would yield the best results on an Algolia "
                "keyword search.\nRULES:\n\n    Translate conversational phrasing into standard industry terms "
                "(e.g., 'how to make an optimized prompt' -> 'prompt engineering').\n\n    KEEP core technologies "
                "exactly as typed. If the user asks about 'sqlite', you MUST include 'sqlite'.\n\n    STRICTLY "
                "FORBIDDEN: Do not add unrelated actions or technologies (e.g., do NOT add 'deploy' if they only "
                "asked about 'sqlite').\n    Example 1: 'how to make an optimized prompt for an llm' -> "
                "'llm prompt engineering'\n    Example 2: 'what is the best way to scale postgresql' -> "
                "'scale postgresql'\n    Example 3: 'sqlite in production' -> 'sqlite production'\n    Return ONLY "
                "the final keywords separated by spaces. No intro text, no quotes.",
            ),
            ("human", "{user_query}"),
        ])

        chain = query_prompt_template | chat | StrOutputParser()
        result = chain.invoke({"user_query": user_query}).strip()
    except Exception:
        return _fallback_optimize_search_query(user_query)

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", result)
    if not tokens:
        return _fallback_optimize_search_query(user_query)

    return " ".join(tokens[:3])


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CommentNode:
    """Represents a comment in a thread hierarchy."""
    id: int
    author: str
    text: str
    parent_id: Optional[int]
    depth: int
    children: List['CommentNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    comment_ids: List[int]
    depth_range: Tuple[int, int]
    token_count: int
    branch_id: str  # Which thread/branch this chunk belongs to


# ============================================================================
# Token Counting Utilities
# ============================================================================

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count the number of tokens in text using tiktoken.

    Falls back to character-based estimation if tiktoken unavailable.

    Args:
        text (str): Text to count tokens for.
        model (str): Model name for tokenizer (default: gpt-3.5-turbo).

    Returns:
        int: Estimated token count.

    Example:
        >>> count_tokens("Hello world", "gpt-3.5-turbo")
        4
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return max(1, len(text) // 4)

    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback if model not found
        return max(1, len(text) // 4)


def estimate_total_tokens(chunks: List[TextChunk]) -> int:
    """
    Estimate total tokens across multiple chunks.

    Args:
        chunks (List[TextChunk]): List of text chunks.

    Returns:
        int: Total estimated token count.
    """
    return sum(chunk.token_count for chunk in chunks)


# ============================================================================
# Thread-Based Comment Chunking
# ============================================================================

def build_comment_tree(comments: List[Dict[str, Any]]) -> Dict[int, CommentNode]:
    """
    Build a tree structure from flat comment list using parent_id.

    Args:
        comments (List[Dict]): List of cleaned comments with id, parent_id, etc.

    Returns:
        Dict[int, CommentNode]: Map of comment_id -> CommentNode.

    Example:
        >>> comments = [
        ...     {"id": 1, "author": "user1", "text": "Root", "parent_id": None, "depth": 0},
        ...     {"id": 2, "author": "user2", "text": "Reply", "parent_id": 1, "depth": 1},
        ... ]
        >>> tree = build_comment_tree(comments)
        >>> len(tree)
        2
    """
    nodes = {}

    # Create all nodes
    for comment in comments:
        node = CommentNode(
            id=comment["id"],
            author=comment["author"],
            text=comment["text"],
            parent_id=comment.get("parent_id"),
            depth=comment.get("depth", 0),
        )
        nodes[comment["id"]] = node

    # Link parents and children
    for comment in comments:
        node = nodes[comment["id"]]
        parent_id = comment.get("parent_id")

        if parent_id and parent_id in nodes:
            parent_node = nodes[parent_id]
            parent_node.children.append(node)

    return nodes


def _serialize_thread(node: CommentNode, prefix: str = "") -> str:
    """
    Serialize a comment thread (node + children) to formatted text.

    Preserves hierarchy with indentation for easy reading.

    Args:
        node (CommentNode): Root comment node.
        prefix (str): Indentation prefix for nested replies.

    Returns:
        str: Formatted thread text.
    """
    lines = []

    # Format current comment
    author_line = f"{prefix}@{node.author}:"
    text_lines = node.text.split("\n")
    lines.append(author_line)
    for i, line in enumerate(text_lines):
        if i == 0:
            lines.append(f"{prefix}  {line}")
        else:
            lines.append(f"{prefix}  {line}")

    # Recursively format children
    for child in node.children:
        child_text = _serialize_thread(child, prefix + "  ")
        lines.append(child_text)

    return "\n".join(lines)


def chunk_comments(
    comments: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
) -> List[TextChunk]:
    """
    Group comments into chunks by thread/branch to keep context together.

    Instead of splitting by character count, groups comments by their parent-child
    relationships. Each chunk represents a conversation thread, and children stay
    close to their parents for better context preservation.

    Args:
        comments (List[Dict]): Cleaned comments with id, author, text, parent_id, depth.
        max_tokens (int): Target max tokens per chunk (default: 2000).
        model (str): LLM model for token counting (default: gpt-3.5-turbo).

    Returns:
        List[TextChunk]: List of grouped comment chunks with metadata.

    Example:
        >>> comments = [
        ...     {"id": 1, "text": "Main topic", "author": "user1", "parent_id": None, "depth": 0},
        ...     {"id": 2, "text": "Reply 1", "author": "user2", "parent_id": 1, "depth": 1},
        ...     {"id": 3, "text": "Reply 2", "author": "user3", "parent_id": 1, "depth": 1},
        ... ]
        >>> chunks = chunk_comments(comments, max_tokens=500)
        >>> len(chunks) >= 1
        True
    """
    if not comments:
        return []

    tree = build_comment_tree(comments)

    # Identify root comments (top-level discussion threads)
    # Note: If no root comments exist, treat all as independent
    root_ids = [c["id"] for c in comments if c.get("parent_id") is None]
    
    if not root_ids and comments:
        # Fallback: if all comments have parents, group by first few comments
        # This handles edge case where comment tree is corrupted
        root_ids = [comments[0]["id"]]

    chunks = []
    branch_id_counter = 0

    for root_id in root_ids:
        root_node = tree[root_id]

        # Serialize each thread
        thread_text = _serialize_thread(root_node)
        thread_tokens = count_tokens(thread_text, model)

        # Collect all IDs in thread
        def collect_ids(node):
            ids = [node.id]
            for child in node.children:
                ids.extend(collect_ids(child))
            return ids

        thread_ids = collect_ids(root_node)

        # If thread is small enough, keep it whole
        if thread_tokens <= max_tokens:
            depth_values = [tree[cid].depth for cid in thread_ids]
            chunk = TextChunk(
                content=thread_text,
                comment_ids=thread_ids,
                depth_range=(min(depth_values), max(depth_values)),
                token_count=thread_tokens,
                branch_id=f"thread_{branch_id_counter}",
            )
            chunks.append(chunk)
            branch_id_counter += 1
        else:
            # Split large thread by depth levels
            by_depth = defaultdict(list)
            for node_id in thread_ids:
                node = tree[node_id]
                by_depth[node.depth].append(node)

            current_chunk_text = ""
            current_chunk_ids = []
            current_tokens = 0

            for depth in sorted(by_depth.keys()):
                for node in by_depth[depth]:
                    node_text = _serialize_thread(node, prefix="  " * depth)
                    node_tokens = count_tokens(node_text, model)

                    # If adding this node exceeds limit, save current chunk
                    if current_tokens + node_tokens > max_tokens and current_chunk_ids:
                        depth_values = [tree[cid].depth for cid in current_chunk_ids]
                        chunk = TextChunk(
                            content=current_chunk_text,
                            comment_ids=current_chunk_ids,
                            depth_range=(min(depth_values), max(depth_values)),
                            token_count=current_tokens,
                            branch_id=f"thread_{branch_id_counter}",
                        )
                        chunks.append(chunk)
                        branch_id_counter += 1

                        current_chunk_text = ""
                        current_chunk_ids = []
                        current_tokens = 0

                    current_chunk_text += node_text + "\n\n"
                    current_chunk_ids.append(node.id)
                    current_tokens += node_tokens

            # Add remaining chunk
            if current_chunk_ids:
                depth_values = [tree[cid].depth for cid in current_chunk_ids]
                chunk = TextChunk(
                    content=current_chunk_text,
                    comment_ids=current_chunk_ids,
                    depth_range=(min(depth_values), max(depth_values)),
                    token_count=current_tokens,
                    branch_id=f"thread_{branch_id_counter}",
                )
                chunks.append(chunk)
                branch_id_counter += 1

    return chunks


# ============================================================================
# Digest Generation with LangChain
# ============================================================================

def generate_digest(
    chunks: List[TextChunk],
    model: str = DEFAULT_MODEL,
    tokens_per_chunk: int = 3000,
) -> Dict[str, Any]:
    """
    Generate a structured technical digest from comment chunks using local LM Studio.

    Uses a highly structured system prompt to extract:
    - Main technical arguments
    - Pros and cons (prioritized by developer experience)
    - Alternative tools mentioned
    - Sentiment analysis
    
    Requires LM Studio to be running on localhost:1234 with Llama 3 8B loaded.

    Args:
        chunks (List[TextChunk]): List of text chunks to analyze.
        model (str): Model to use for token counting (default: gpt-3.5-turbo, for tiktoken compatibility).

    Returns:
        Dict[str, Any]: Structured digest with keys:
            - technical_arguments
            - pros
            - cons
            - alternative_tools
            - sentiment
            - sentiment_summary
            - chunk_count
            - total_tokens

    Example:
        >>> chunks = [TextChunk(content="...", comment_ids=[1,2], ...)]
        >>> digest = generate_digest(chunks)
        >>> digest["sentiment"]
        'pragmatic'
    """
    if not chunks:
        return {
            "arguments": [],
            "pros": [],
            "cons": [],
            "tools": [],
            "sentiment": "unknown",
        }

    if not LANGCHAIN_AVAILABLE:
        return {
            "error": "LangChain not installed. Run: pip install -r requirements.txt",
            "chunk_count": len(chunks),
            "total_tokens": estimate_total_tokens(chunks),
        }

    # Concatenate all chunks
    combined_text = "\n\n---THREAD BREAK---\n\n".join(chunk.content for chunk in chunks)

    # 1 token is approximately 4 characters.
    max_chars = int(tokens_per_chunk * 4)

    compiled_comments = ""
    for chunk in chunks:
        c_text = chunk.content

        if len(compiled_comments) + len(c_text) < max_chars:
            compiled_comments += f"\n---\n{c_text}"
        else:
            compiled_comments += "\n\n... [REMAINING COMMENTS TRUNCATED TO PRESERVE LLM ATTENTION]"
            break

    comments_text = compiled_comments
    total_tokens_used = count_tokens(comments_text + SYSTEM_PROMPT_RESEARCHER, model)

    # Create LLM instance - connects to local LM Studio server
    try:
        chat = get_local_llm(temperature=0.1)
    except Exception as e:
        return {
            "error": f"Failed to connect to LM Studio on localhost:1234: {str(e)}. Make sure LM Studio is running with Llama 3 loaded.",
            "chunk_count": len(chunks),
            "total_tokens": estimate_total_tokens(chunks),
        }

    # Create messages
    system_prompt = (
        SYSTEM_PROMPT_RESEARCHER
        + "\n\nYou MUST output ONLY a valid JSON object. Do not include any markdown "
        "formatting, backticks, or conversational text. Your output must exactly match "
        'this structure: {"sentiment": "...", "arguments": [], "pros": [], "cons": [], "tools": []}\n\n'
        "You MUST use the exact JSON keys shown in this example below. Do not deviate.\n"
        "Example Output:\n"
        "{\n"
        "\"sentiment\": \"Pragmatic\",\n"
        "\"arguments\": [\"The first main argument...\", \"The second main argument...\"],\n"
        "\"pros\": [\"Pro 1\", \"Pro 2\"],\n"
        "\"cons\": [\"Con 1\", \"Con 2\"],\n"
        "\"tools\": [\"Tool A\", \"Tool B\"]\n"
        "}\n\n"
        "You are an expert technical analyst. You MUST thoroughly analyze the provided Hacker News comments. "
        "Extract highly specific, concrete technical arguments, pros, and cons. DO NOT output generic phrases "
        "like 'No specific pros mentioned'. If the text contains any technical debate, infer the pros and cons "
        "and list them explicitly. Be exhaustive."
    )
    system_message = SystemMessage(content=system_prompt)
    user_message = HumanMessage(
        content=f"Analyze these Hacker News comments:\n\n{comments_text}"
    )

    # Call LLM
    try:
        response = chat.invoke([system_message, user_message])
        response_text = response.content
    except Exception as e:
        return {
            "error": f"LM Studio inference failed: {str(e)}. Ensure the Local Inference Server is running on port 1234.",
            "chunk_count": len(chunks),
            "total_tokens": total_tokens_used,
        }

    llm_output = response_text
    # 1. Strip Markdown blocks safely
    cleaned_output = re.sub(r"`{3}(?:json)?\s*(.*?)\s*`{3}", r"\1", llm_output, flags=re.DOTALL).strip()

    # 2. Extract first valid JSON object if filler text exists
    if not (cleaned_output.startswith('{') and cleaned_output.endswith('}')):
        match = re.search(r"(\{.*\})", cleaned_output, re.DOTALL)
        if match:
            cleaned_output = match.group(1)

    # 3. Parse JSON safely
    try:
        data = json.loads(cleaned_output)
    except json.JSONDecodeError:
        data = {}

    # 4. Handle Llama 3 key hallucinations
    return {
        "arguments": data.get("arguments", data.get("technical_arguments", data.get("main_arguments", []))),
        "tools": data.get("tools", data.get("alternative_tools", data.get("alternatives", []))),
        "pros": data.get("pros", []),
        "cons": data.get("cons", []),
        "sentiment": data.get("sentiment", "Unknown")
    }


# ============================================================================
# RAG (Retrieval Augmented Generation) - Phase 2
# ============================================================================

class CommentRAG:
    """
    Simple Retrieval Augmented Generation system for follow-up questions.

    Allows users to ask questions about fetched HN comments using semantic similarity.
    Uses local Llama 3 via LM Studio for inference.
    """

    def __init__(
        self,
        chunks: List[TextChunk],
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize RAG system with comment chunks.

        Requires LM Studio running on localhost:1234 with Llama 3 8B loaded.

        Args:
            chunks (List[TextChunk]): Pre-chunked comments.
            model (str): Model to use for token counting (default: gpt-3.5-turbo).
        """
        self.chunks = chunks
        self.model = model
        self.embeddings = None
        self.chat = None

        if LANGCHAIN_AVAILABLE:
            try:
                self.chat = get_local_llm(temperature=0.5)
            except Exception:
                pass

    def _simple_similarity_search(self, query: str, top_k: int = 3) -> List[TextChunk]:
        """
        Keyword-based similarity search with robust fallback.
        
        CRITICAL: Always returns at least top_k chunks, even if no keywords match.
        This prevents the "0 chunks" bug where chat returns "Based on 0 comment chunks".

        Args:
            query (str): User question/query.
            top_k (int): Number of top chunks to return. Defaults to 3.

        Returns:
            List[TextChunk]: Most relevant chunks (guaranteed at least top_k if available).
        """
        if not self.chunks:
            print("[DEBUG] No chunks available for RAG retrieval!")
            return []
        
        # Split query into keywords and filter stop words
        keywords = set(query.lower().split())
        stop_words = {"the", "a", "is", "are", "what", "how", "why", "when", "where", 
                      "which", "do", "does", "did", "can", "could", "should", "would"}
        keywords = keywords - stop_words
        
        print(f"[DEBUG] RAG Query: '{query}'")
        print(f"[DEBUG] Keywords extracted: {keywords}")
        print(f"[DEBUG] Available chunks: {len(self.chunks)}")
        
        # Score chunks by keyword matches
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.content.lower()
            score = sum(chunk_lower.count(kw) for kw in keywords) if keywords else 0
            scored_chunks.append((chunk, score, i))
            if score > 0:
                print(f"[DEBUG] Chunk {i}: score={score}")
        
        # CRITICAL FIX: Always sort and return top_k
        # Don't filter by score > 0; if keywords don't match, still return top_k chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k chunks regardless of score
        result_chunks = [chunk for chunk, score, idx in scored_chunks[:top_k]]
        
        if not result_chunks:
            print(f"[DEBUG] WARNING: No chunks returned (empty chunk list)")
        else:
            print(f"[DEBUG] Returning {len(result_chunks)} chunks for RAG")
        
        return result_chunks

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a follow-up question based on fetched HN comments.

        Uses retrieval augmented generation: finds relevant chunks, then asks LLM
        to answer based only on those chunks.

        Args:
            question (str): User's follow-up question.

        Returns:
            Dict[str, Any]: Answer with source chunks and reasoning.

        Example:
            >>> rag = CommentRAG(chunks)
            >>> result = rag.answer_question("What tools were mentioned?")
            >>> result["answer"]
            'The discussion mentioned...'
        """
        if not self.chat:
            # Try to reinitialize chat connection
            try:
                self.chat = get_local_llm(temperature=0.5)
            except Exception as e:
                return {
                    "error": f"LangChain not available. LM Studio may not be running on localhost:1234: {str(e)}",
                    "question": question,
                }

        # Retrieve relevant chunks
        relevant_chunks = self._simple_similarity_search(question, top_k=3)

        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the comments.",
                "question": question,
                "relevant_chunks": 0,
            }

        # Build context from relevant chunks
        context_text = "\n\n---\n\n".join(chunk.content for chunk in relevant_chunks)

        # Create system prompt for answering
        system_prompt = """You are an expert at synthesizing information from Hacker News discussions.

Your task is to answer the user's question ONLY using the provided Hacker News comments.

CRITICAL INSTRUCTIONS:
1. Use the retrieved Hacker News comments below to answer the question.
2. If the answer IS contained in these comments, synthesize the information clearly and cite specific comments.
3. If the answer is NOT contained in these comments, state clearly: "The fetched Hacker News comments do not contain information that directly answers this question."
4. Do NOT make up information or use external knowledge beyond what's in the comments.
5. Be concise and cite specific comments when relevant."""

        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(
            content=f"""Retrieved Hacker News comments from the discussion:

{context_text}

---

Question: {question}

Answer ONLY based on the retrieved comments above. If the information is not in these comments, state that clearly:"""
        )

        # Get answer with better error handling
        try:
            response = self.chat.invoke([system_message, user_message])
            answer = response.content
        except Exception as e:
            import traceback
            return {
                "error": f"Failed to generate answer: {str(e)}. Make sure LM Studio server is running on localhost:1234 with a model loaded.",
                "question": question,
                "traceback": traceback.format_exc()[:500],
            }

        # Count tokens
        total_input = context_text + question + system_prompt
        input_tokens = count_tokens(total_input, self.model)
        output_tokens = count_tokens(answer, self.model)

        return {
            "answer": answer,
            "question": question,
            "relevant_chunks": len(relevant_chunks),
            "relevant_branches": list(set(c.branch_id for c in relevant_chunks)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "source_comment_ids": [c_id for chunk in relevant_chunks for c_id in chunk.comment_ids],
        }


# ============================================================================
# Integration Function
# ============================================================================

def process_hn_data_with_llm(
    comments: List[Dict[str, Any]],
    max_chunk_tokens: int = 2000,
    model: str = DEFAULT_MODEL,
    tokens_per_chunk: int = 3000,
    enable_rag: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end processing: chunk comments, generate digest, and setup RAG.
    
    Uses local LM Studio (Llama 3 8B) for all LLM operations.
    Requires LM Studio to be running on localhost:1234.

    Args:
        comments (List[Dict]): Cleaned HN comments from hn_client.
        max_chunk_tokens (int): Max tokens per chunk (default: 2000).
        model (str): Model for token counting (default: gpt-3.5-turbo).
        enable_rag (bool): Enable RAG system for follow-up questions.

    Returns:
        Dict[str, Any]: Contains:
            - digest: Structured analysis
            - chunks: Comment chunks used
            - rag: RAG system instance (if enabled)
            - stats: Token and processing stats

    Example:
        >>> results = process_hn_data_with_llm(comments)
        >>> print(results["digest"]["sentiment"])
        'pragmatic'
        >>> qa = results["rag"].answer_question("What tools were recommended?")
    """
    # Step 1: Chunk comments
    chunks = chunk_comments(comments, max_tokens=max_chunk_tokens, model=model)

    # Step 2: Generate digest using local LM Studio
    digest = generate_digest(chunks, model=model, tokens_per_chunk=tokens_per_chunk)

    # Step 3: Setup RAG if enabled
    rag = None
    if enable_rag:
        rag = CommentRAG(chunks, model=model)

    # Compile stats
    stats = {
        "comment_count": len(comments),
        "chunk_count": len(chunks),
        "total_tokens_for_digest": digest.get("total_tokens", 0),
        "avg_tokens_per_chunk": sum(c.token_count for c in chunks) // max(len(chunks), 1),
        "estimated_api_calls": 1 + (1 if enable_rag else 0),
        "execution_mode": "local (LM Studio - Llama 3 8B)",
    }

    return {
        "digest": digest,
        "chunks": chunks,
        "rag": rag,
        "stats": stats,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("LLM Engine Module - Example Usage")
    print("=" * 70)

    # Sample comments
    sample_comments = [
        {
            "id": 1,
            "author": "alice",
            "text": "I think Rust is great for systems programming.",
            "parent_id": None,
            "depth": 0,
        },
        {
            "id": 2,
            "author": "bob",
            "text": "Agreed, but the learning curve is steep.",
            "parent_id": 1,
            "depth": 1,
        },
        {
            "id": 3,
            "author": "charlie",
            "text": "Have you tried Go? It's more pragmatic.",
            "parent_id": 1,
            "depth": 1,
        },
        {
            "id": 4,
            "author": "alice",
            "text": "Go is simpler but Rust has better guarantees.",
            "parent_id": 3,
            "depth": 2,
        },
    ]

    print("\n1. Building comment tree...")
    tree = build_comment_tree(sample_comments)
    print(f"   Created tree with {len(tree)} nodes")

    print("\n2. Chunking comments by thread...")
    chunks = chunk_comments(sample_comments, max_tokens=200)
    print(f"   Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {chunk.token_count} tokens, {len(chunk.comment_ids)} comments")

    print("\n3. Token counting...")
    total_tokens = estimate_total_tokens(chunks)
    print(f"   Total tokens: {total_tokens}")

    print("\n4. RAG System Setup...")
    rag = CommentRAG(chunks)
    print(f"   RAG initialized with {len(chunks)} chunks")

    print("\n(Skipping actual LLM calls - requires OpenAI API key)")
    print("\nTo use with OpenAI:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Call: digest = generate_digest(chunks, api_key='sk-...')")
    print("  3. Call: result = rag.answer_question('What tools were mentioned?')")

    print("\n" + "=" * 70)
