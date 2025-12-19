#!/usr/bin/env python3
"""
Test script for DuckDuckGo search function
Tests searching for current president of the US
"""

import sys
import json
from duckduckgo_search import DDGS

def duckduckgo_search(query: str, max_results: int = 5):
    """
    Search DuckDuckGo and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing title, href, and body
    """
    print(f"Searching DuckDuckGo for: '{query}'")
    print(f"Max results: {max_results}")
    print("-" * 60)
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            print(f"\n✅ Search completed successfully!")
            print(f"Retrieved {len(results)} results")
            print("=" * 60)
            
            if not results:
                print("⚠️  No results returned")
                return results
            
            # Print each result
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"URL: {result.get('href', 'N/A')}")
                print(f"Body: {result.get('body', 'N/A')[:200]}...")
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
    print("DuckDuckGo Search Test")
    print("=" * 60)
    print()
    
    results = duckduckgo_search(query, max_results=5)
    
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
