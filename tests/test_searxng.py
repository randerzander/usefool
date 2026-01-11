#!/usr/bin/env python3
"""
Test script for pysearx web_search function
Tests searching for current president of the US using pysearx
"""

import sys
import json

def web_search(query: str, max_results: int = 10):
    """
    Search using pysearx and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing index, title, href (url), and body (description)
    """
    print(f"Searching with pysearx for: '{query}'")
    print(f"Max results: {max_results}")
    print("-" * 60)
    
    try:
        from pysearx import search
        
        # Use pysearx with parallel mode for faster results
        print(f"Using pysearx search with parallel=True")
        raw_results = search(query, max_results=max_results, parallel=True)
        
        print(f"pysearx returned {len(raw_results)} results")
        print("-" * 60)
        
        # Convert pysearx format to expected format
        results = []
        for i, result in enumerate(raw_results, 1):
            results.append({
                'index': i,
                'title': result.get('title', 'N/A'),
                'href': result.get('url', 'N/A'),
                'body': result.get('description', ''),
                'engine': result.get('engine', 'N/A')
            })
        
        print(f"\n✅ Search completed successfully!")
        print(f"Retrieved {len(results)} results")
        print("=" * 60)
        
        if not results:
            print("⚠️  No results returned")
            return results
        
        # Print each result
        for result in results:
            print(f"\n--- Result {result['index']} ---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Engine: {result['engine']}")
            print(f"Description: {result['body'][:200]}...")
            print()
        
        # Print raw JSON for debugging
        print("=" * 60)
        print("Raw JSON results:")
        print(json.dumps(results, indent=2))
        
        return results
        
    except Exception as e:
        print(f"\n❌ Search failed!")
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{"error": f"Search failed: {str(e)}"}]

def main():
    """Run the test"""
    
    # Test query
    query = "current president of the United States"
    
    print("=" * 60)
    print("pysearx Web Search Test")
    print("=" * 60)
    print()
    
    results = web_search(query, max_results=10)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if results and "error" not in results[0]:
        print(f"✅ Success! Got {len(results)} results")
        print(f"\nFirst result title: {results[0].get('title', 'N/A')}")
    elif results and "error" in results[0]:
        print(f"❌ Error: {results[0]['error']}")
    else:
        print("⚠️  No results")

if __name__ == "__main__":
    main()
