"""
Hacker News Client Module

This module provides utilities for fetching and processing Hacker News stories and comments
using the Algolia search API and Firebase real-time database API.

Main Components:
- get_top_stories: Fetch top story IDs by search query
- fetch_comment_tree: Recursively fetch story details and comments
- clean_comments: Clean and filter comments with HTML tag removal
"""

import concurrent.futures
import re
import time
from typing import List, Dict, Any, Optional
from functools import wraps, partial

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# Configuration & Constants
# ============================================================================

ALGOLIA_API_URL = "https://hn.algolia.com/api/v1"
FIREBASE_API_URL = "https://hacker-news.firebaseio.com/v0"

# Retry strategy configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5  # exponential backoff: 0.5s, 1s, 2s
TIMEOUT = 10  # seconds


# ============================================================================
# HTTP Session with Retry Logic
# ============================================================================

def create_session_with_retries() -> requests.Session:
    """
    Create a requests Session with exponential backoff retry strategy.

    Returns:
        requests.Session: Configured session with retry logic for GET/POST requests.

    Example:
        session = create_session_with_retries()
        response = session.get("https://example.com")
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# Create a module-level session
_session = create_session_with_retries()


# ============================================================================
# Error Handling Decorator
# ============================================================================

def handle_api_errors(func):
    """
    Decorator to handle common API errors with logging.

    Wraps API calls to catch and log errors gracefully.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            print(f"[ERROR] Request timeout in {func.__name__}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Connection error in {func.__name__}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error in {func.__name__}: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error in {func.__name__}: {str(e)}")
            return None

    return wrapper


# ============================================================================
# Search Logic
# ============================================================================

@handle_api_errors
def get_top_stories(query: str, limit: int = 3, min_points: int = 50, min_comments: int = 20) -> tuple:
    """
    Fetch top story IDs from Hacker News by search query using Algolia API.
    
    Implements TWO-PASS FALLBACK SEARCH:
    - Attempt 1 (High Signal): Query titles only with moderate points threshold (points > 10)
    - Attempt 2 (Broad Net): If Attempt 1 yields 0 results, search all fields with no point threshold
    
    This ensures specific queries like "how to sort an array" still get results even if no
    high-upvoted title exactly matches.

    Args:
        query (str): Search query string (e.g., 'python sort array', 'database scaling').
        limit (int): Maximum number of story IDs to return. Defaults to 3.
        min_points (int): Minimum upvote threshold for ATTEMPT 1 (points > X). Defaults to 50.
        min_comments (int): Minimum comment threshold for ATTEMPT 1 (num_comments >= X). Defaults to 20.

    Returns:
        tuple: (story_ids, used_fallback)
            - story_ids: List of story object IDs (as strings) from top search results.
            - used_fallback: Boolean flag indicating if Attempt 2 (broad search) was used.

    Example:
        >>> ids, used_fallback = get_top_stories("sort array", limit=3, min_points=10, min_comments=5)
        >>> ids
        ['28223156', '28221987', '28220450']
        >>> used_fallback
        True
    """
    url = f"{ALGOLIA_API_URL}/search"
    
    # ATTEMPT 1: High-signal search (title only, moderate thresholds)
    params_strict = {
        "query": query,
        "tags": "story",
        "hitsPerPage": limit * 3,  # Fetch 3x to account for filtering
        "restrictSearchableAttributes": "title",  # STRICT: Only search in title
        "numericFilters": "points>10,num_comments>5",  # Moderate threshold + active discussions
        "typoTolerance": "false",  # Exact match required
    }
    
    response = _session.get(url, params=params_strict, timeout=TIMEOUT)
    response.raise_for_status()
    
    data = response.json()
    hits = data.get("hits", [])
    active_hits = [hit for hit in hits if hit.get("num_comments", 0) > 5]
    story_ids = [str(hit["objectID"]) for hit in active_hits[:limit]]
    
    if story_ids:
        # ATTEMPT 1 succeeded
        print(f"[INFO] Attempt 1 (Strict): Found {len(story_ids)} stories for '{query}' (title search, points > 10)")
        return story_ids, False
    
    # ATTEMPT 2: Fallback to broad search (all fields, no point threshold, pure relevance)
    print(f"[INFO] Attempt 1 (Strict) returned 0 results. Falling back to Attempt 2 (Broad)...")
    
    params_broad = {
        "query": query,
        "tags": "story",
        "hitsPerPage": limit * 3,  # Fetch 3x for consistency
        # NO restrictSearchableAttributes: Search all fields (title + content)
        "numericFilters": "num_comments>5",  # Ensure active discussions
        "typoTolerance": "false",  # Still require exact match
    }
    
    response = _session.get(url, params=params_broad, timeout=TIMEOUT)
    response.raise_for_status()
    
    data = response.json()
    hits = data.get("hits", [])
    active_hits = [hit for hit in hits if hit.get("num_comments", 0) > 5]
    story_ids = [str(hit["objectID"]) for hit in active_hits[:limit]]
    
    print(f"[INFO] Attempt 2 (Broad): Found {len(story_ids)} stories for '{query}' (all fields, no point threshold)")
    return story_ids, True  # Return True to indicate fallback was used


# ============================================================================
# Recursive Comment Fetching
# ============================================================================

@handle_api_errors
def _fetch_item(item_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a single item (story or comment) from Firebase API.

    Args:
        item_id (int): The HN item ID.

    Returns:
        Optional[Dict[str, Any]]: Item data or None if fetch fails.
    """
    url = f"{FIREBASE_API_URL}/item/{item_id}.json"
    response = _session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=1800)
def fetch_comment_tree(
    story_id: int, max_comments: int = 50, _depth: int = 0, _parent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Recursively fetch a story and its entire comment tree from Firebase API.

    Traverses the nested 'kids' structure to build a flat list of comments with
    metadata (id, author, text, parent_id, depth).

    Args:
        story_id (int): The HN story ID to fetch.
        max_comments (int): Maximum number of comments to process (per depth level).
                            Defaults to 50. Helps prevent infinite recursion.
        _depth (int): Internal parameter tracking recursion depth.
        _parent_id (Optional[int]): Internal parameter for tracking parent comment.

    Returns:
        Dict[str, Any]: Story data containing 'story' and 'comments' keys:
            - 'story': Dictionary with story metadata.
            - 'comments': List of cleaned comment dictionaries with id, author, text,
                          parent_id, and depth.

    Example:
        >>> data = fetch_comment_tree(28223156)
        >>> len(data["comments"])
        42
    """
    story_item = _fetch_item(story_id)

    if not story_item:
        print(f"[ERROR] Failed to fetch story {story_id}")
        return {"story": {}, "comments": []}

    comments = []

    if "kids" in story_item and story_item["kids"]:
        kid_ids = story_item["kids"][:max_comments]
        fetch_subtree = partial(_collect_comment_subtree, depth=1, parent_id=story_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for subtree in executor.map(fetch_subtree, kid_ids):
                if subtree:
                    comments.extend(subtree)

    result = {
        "story": {
            "id": story_item.get("id"),
            "title": story_item.get("title"),
            "url": story_item.get("url"),
            "author": story_item.get("by"),
            "score": story_item.get("score"),
            "timestamp": story_item.get("time"),
        },
        "comments": comments,
    }

    print(f"[INFO] Fetched story {story_id} with {len(comments)} comments")
    return result


def _collect_comment_subtree(
    comment_id: int, depth: int = 0, parent_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Recursively fetch comments and return a flat list for this subtree.

    This helper function traverses the comment tree depth-first, building a flat list.

    Args:
        comment_id (int): ID of the comment to fetch.
        depth (int): Current depth in the comment tree.
        parent_id (Optional[int]): ID of the parent comment.
    """
    comment_item = _fetch_item(comment_id)

    if not comment_item:
        return []

    # Skip deleted or dead comments
    if comment_item.get("deleted") or comment_item.get("dead"):
        return []

    comment = {
        "id": comment_item.get("id"),
        "author": comment_item.get("by"),
        "text": comment_item.get("text", ""),
        "parent_id": parent_id,
        "depth": depth,
    }

    comments = [comment]

    # Recursively fetch child comments
    if "kids" in comment_item and comment_item["kids"]:
        for child_id in comment_item["kids"]:
            comments.extend(
                _collect_comment_subtree(child_id, depth=depth + 1, parent_id=comment_id)
            )
    return comments


def _recursive_fetch_comments(
    comment_id: int, comments_list: List[Dict[str, Any]], depth: int = 0, parent_id: Optional[int] = None
) -> None:
    """
    Recursively fetch comments and append to comments_list.

    This helper function traverses the comment tree depth-first, building a flat list.

    Args:
        comment_id (int): ID of the comment to fetch.
        comments_list (List[Dict]): Accumulated list of comments (mutated in-place).
        depth (int): Current depth in the comment tree.
        parent_id (Optional[int]): ID of the parent comment.
    """
    comment_item = _fetch_item(comment_id)

    if not comment_item:
        return

    # Skip deleted or dead comments
    if comment_item.get("deleted") or comment_item.get("dead"):
        return

    comment = {
        "id": comment_item.get("id"),
        "author": comment_item.get("by"),
        "text": comment_item.get("text", ""),
        "parent_id": parent_id,
        "depth": depth,
    }

    comments_list.append(comment)

    # Recursively fetch child comments
    if "kids" in comment_item and comment_item["kids"]:
        for child_id in comment_item["kids"]:
            _recursive_fetch_comments(child_id, comments_list, depth=depth + 1, parent_id=comment_id)


# ============================================================================
# HTML Tag Stripping
# ============================================================================

def _strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Hacker News uses <p> for paragraphs and <i> for italics. This function removes
    these and other common HTML tags.

    Args:
        text (str): HTML text.

    Returns:
        str: Text with HTML tags removed.

    Example:
        >>> _strip_html_tags("<p>Hello</p> <i>world</i>")
        'Hello world'
    """
    # Remove <p> tags and preserve content
    text = re.sub(r"<p>", "\n", text)
    text = re.sub(r"</p>", "", text)

    # Remove <i>, <b>, <a>, and other tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================================
# Comment Cleaning Pipeline
# ============================================================================

def clean_comments(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and filter comments by removing low-value entries.

    Applies the following filters:
    - Removes deleted or dead comments.
    - Strips HTML tags (<p>, <i>, etc.) from comment text.
    - Discards comments shorter than 20 characters.

    Args:
        comments (List[Dict[str, Any]]): List of comment dictionaries with keys:
                                         id, author, text, parent_id, depth.

    Returns:
        List[Dict[str, Any]]: Cleaned comments that passed all filters.

    Example:
        >>> raw = [
        ...     {"id": 1, "text": "<p>Great discussion!</p>", "author": "user1", "parent_id": None, "depth": 0},
        ...     {"id": 2, "text": "ok", "author": "user2", "parent_id": 1, "depth": 1},
        ... ]
        >>> cleaned = clean_comments(raw)
        >>> len(cleaned)
        1
    """
    cleaned = []

    for comment in comments:
        # Skip if missing required fields
        if not comment.get("id") or not comment.get("text"):
            continue

        # Strip HTML tags
        text = _strip_html_tags(comment["text"])

        # Skip if text is too short (noise filtering)
        if len(text) < 20:
            continue

        # Build cleaned comment
        cleaned_comment = {
            "id": comment["id"],
            "author": comment.get("author"),
            "text": text,
            "parent_id": comment.get("parent_id"),
            "depth": comment.get("depth", 0),
        }

        cleaned.append(cleaned_comment)

    print(f"[INFO] Cleaned {len(comments)} comments -> {len(cleaned)} valid comments")
    return cleaned


# ============================================================================
# Integration Function
# ============================================================================

@st.cache_data(ttl=1800)
def fetch_and_clean_story(query: str, story_limit: int = 1, max_comments: int = 50) -> tuple:
    """
    End-to-end function: search for stories, fetch their comments, and clean.
    
    Returns metadata about whether fallback search was used.

    Args:
        query (str): Search query for Hacker News.
        story_limit (int): Number of top stories to fetch. Defaults to 1.
        max_comments (int): Maximum comments per story. Defaults to 50.

    Returns:
        tuple: (results, used_fallback)
            - results: List[Dict[str, Any]] of story objects with cleaned comments.
            - used_fallback: Boolean indicating if Attempt 2 (broad search) was used.

    Example:
        >>> results, used_fallback = fetch_and_clean_story("python", story_limit=2, max_comments=30)
        >>> results[0]["comments"][0]["text"]
        'Great article on decorators!'
        >>> used_fallback
        False
    """
    story_ids, used_fallback = get_top_stories(query, limit=story_limit)

    results = []
    for story_id in story_ids:
        story_data = fetch_comment_tree(int(story_id), max_comments=max_comments)
        story_data["comments"] = clean_comments(story_data["comments"])
        results.append(story_data)

    return results, used_fallback


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=== Hacker News Intelligence Tool ===\n")

    results = fetch_and_clean_story("python", story_limit=1, max_comments=30)

    for idx, result in enumerate(results, 1):
        story = result["story"]
        comments = result["comments"]

        print(f"\nStory {idx}: {story.get('title', 'No Title')}")
        print(f"  Score: {story.get('score')}")
        print(f"  Comments: {len(comments)}")

        if comments:
            print(f"\n  Top Comment:")
            top = comments[0]
            print(f"    Author: {top['author']}")
            print(f"    Text: {top['text'][:100]}...")
