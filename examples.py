"""
Example usage script for the HN Client module.

This script demonstrates practical workflows for fetching and analyzing
Hacker News stories and comments.
"""

from hn_client import fetch_and_clean_story, get_top_stories, fetch_comment_tree, clean_comments
import json


def example_1_simple_fetch():
    """Example 1: Simple one-liner fetch and clean."""
    print("=" * 70)
    print("Example 1: Simple Fetch and Clean")
    print("=" * 70)
    
    results = fetch_and_clean_story("python", story_limit=1, max_comments=30)
    
    if results:
        story = results[0]["story"]
        comments = results[0]["comments"]
        
        print(f"\nStory: {story.get('title', 'N/A')}")
        print(f"Author: {story.get('author', 'N/A')}")
        print(f"Score: {story.get('score', 'N/A')}")
        print(f"URL: {story.get('url', 'N/A')}")
        print(f"\nTotal comments fetched: {len(results[0]['comments'])}")
        print(f"Valid comments after cleaning: {len(comments)}")
        
        if comments:
            print(f"\nSample cleaned comments:")
            for i, comment in enumerate(comments[:3], 1):
                print(f"\n  {i}. {comment['author']} (depth: {comment['depth']})")
                text_preview = comment['text'][:80] + "..." if len(comment['text']) > 80 else comment['text']
                print(f"     {text_preview}")


def example_2_step_by_step():
    """Example 2: Step-by-step processing for fine-grained control."""
    print("\n" + "=" * 70)
    print("Example 2: Step-by-Step Processing")
    print("=" * 70)
    
    # Step 1: Search
    query = "rust"
    print(f"\nSearching for: '{query}'")
    story_ids = get_top_stories(query, limit=2)
    
    if not story_ids:
        print("No stories found!")
        return
    
    # Step 2: Process each story
    for idx, story_id in enumerate(story_ids, 1):
        print(f"\n--- Story {idx} ---")
        
        # Fetch the complete comment tree
        story_data = fetch_comment_tree(int(story_id), max_comments=50)
        
        if not story_data["story"]:
            print("Failed to fetch story")
            continue
        
        story = story_data["story"]
        print(f"Title: {story.get('title', 'N/A')}")
        print(f"Author: {story.get('author', 'N/A')}")
        print(f"Score: {story.get('score', 'N/A')}")
        
        # Step 3: Clean comments
        raw_count = len(story_data["comments"])
        cleaned = clean_comments(story_data["comments"])
        cleaned_count = len(cleaned)
        
        print(f"Raw comments: {raw_count}")
        print(f"Cleaned comments: {cleaned_count}")
        print(f"Removed: {raw_count - cleaned_count} ({100*(raw_count-cleaned_count)/max(raw_count,1):.1f}%)")
        
        # Step 4: Analyze cleaned comments
        if cleaned:
            depths = [c["depth"] for c in cleaned]
            avg_depth = sum(depths) / len(depths)
            print(f"Average nesting depth: {avg_depth:.2f}")
            print(f"Max depth: {max(depths)}")
            
            # Group by depth
            by_depth = {}
            for c in cleaned:
                d = c["depth"]
                by_depth[d] = by_depth.get(d, 0) + 1
            
            print(f"Comments by depth: {sorted(by_depth.items())}")


def example_3_analysis_pipeline():
    """Example 3: Analysis pipeline with statistics."""
    print("\n" + "=" * 70)
    print("Example 3: Analysis Pipeline with Statistics")
    print("=" * 70)
    
    results = fetch_and_clean_story("machine learning", story_limit=1, max_comments=50)
    
    if not results:
        print("No results found!")
        return
    
    story_data = results[0]
    story = story_data["story"]
    comments = story_data["comments"]
    
    if not comments:
        print("No comments to analyze!")
        return
    
    # Build statistics
    stats = {
        "story_title": story.get("title"),
        "story_author": story.get("author"),
        "story_score": story.get("score"),
        "total_comments": len(comments),
        "unique_authors": len(set(c["author"] for c in comments if c.get("author"))),
        "avg_comment_length": sum(len(c["text"]) for c in comments) / len(comments),
        "depth_distribution": {},
        "most_active_authors": {},
    }
    
    # Analyze depths
    for comment in comments:
        depth = comment["depth"]
        stats["depth_distribution"][str(depth)] = stats["depth_distribution"].get(str(depth), 0) + 1
        
        author = comment.get("author", "unknown")
        stats["most_active_authors"][author] = stats["most_active_authors"].get(author, 0) + 1
    
    # Get top authors
    top_authors = sorted(
        stats["most_active_authors"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    stats["top_authors"] = {a: c for a, c in top_authors}
    
    # Print analysis
    print(f"\nStory: {stats['story_title']}")
    print(f"Author: {stats['story_author']}")
    print(f"Score: {stats['story_score']}")
    print(f"\nComment Statistics:")
    print(f"  Total comments: {stats['total_comments']}")
    print(f"  Unique authors: {stats['unique_authors']}")
    print(f"  Avg comment length: {stats['avg_comment_length']:.0f} characters")
    print(f"  Depth distribution: {stats['depth_distribution']}")
    print(f"\nTop 5 Most Active Authors:")
    for author, count in stats["top_authors"].items():
        print(f"  - {author}: {count} comments")


def example_4_export_data():
    """Example 4: Export collected data to JSON."""
    print("\n" + "=" * 70)
    print("Example 4: Export Data to JSON")
    print("=" * 70)
    
    results = fetch_and_clean_story("python async", story_limit=1, max_comments=30)
    
    if not results:
        print("No results to export!")
        return
    
    # Prepare export format
    export_data = {
        "metadata": {
            "query": "python async",
            "stories_fetched": len(results),
        },
        "stories": []
    }
    
    for story_data in results:
        export_data["stories"].append({
            "story": story_data["story"],
            "comments": story_data["comments"],
        })
    
    # Save to file
    filename = "hn_export.json"
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nExported data to: {filename}")
    print(f"Stories: {len(export_data['stories'])}")
    print(f"Total comments: {sum(len(s['comments']) for s in export_data['stories'])}")
    print(f"File size: {len(json.dumps(export_data))} bytes")


if __name__ == "__main__":
    # Run examples
    try:
        example_1_simple_fetch()
        # example_2_step_by_step()
        # example_3_analysis_pipeline()
        # example_4_export_data()
        
        print("\n" + "=" * 70)
        print("Examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
