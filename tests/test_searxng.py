#!/usr/bin/env python3
"""
Test script for SearXNG web_search function
Tests searching for current president of the US using local SearXNG instance
"""

import sys
import json
import requests
from bs4 import BeautifulSoup

def web_search(query: str, max_results: int = 10):
    """
    Search using local SearXNG instance and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing index, title, url, and description
    """
    print(f"Searching SearXNG for: '{query}'")
    print(f"Max results: {max_results}")
    print("-" * 60)
    
    try:
        # Make request to SearXNG
        url = f"http://localhost:8081/search?q={query}"
        print(f"Fetching: {url}")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        print(f"Content length: {len(response.text)} characters")
        print("-" * 60)
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all result articles
        articles = soup.find_all('article', class_='result')
        print(f"Found {len(articles)} article elements")
        print("-" * 60)
        
        results = []
        for i, article in enumerate(articles[:max_results], 1):
            # Extract title
            h3 = article.find('h3')
            title_link = h3.find('a') if h3 else None
            title = title_link.get_text(strip=True) if title_link else 'N/A'
            
            # Extract URL
            url_link = article.find('a', class_='url_header')
            url = url_link.get('href') if url_link else 'N/A'
            
            # Extract description
            content_p = article.find('p', class_='content')
            description = content_p.get_text(strip=True) if content_p else 'N/A'
            
            results.append({
                'index': i,
                'title': title,
                'href': url,
                'body': description
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
            print(f"Description: {result['body'][:200]}...")
            print()
        
        # Print raw JSON for debugging
        print("=" * 60)
        print("Raw JSON results:")
        print(json.dumps(results, indent=2))
        
        return results
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection Error!")
        print(f"Cannot connect to SearXNG at http://localhost:8081")
        print(f"Error: {str(e)}")
        print("\nMake sure SearXNG is running:")
        print("  ./start_searxng.sh")
        return [{"error": f"Connection failed: {str(e)}"}]
        
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
    print("SearXNG Web Search Test")
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
