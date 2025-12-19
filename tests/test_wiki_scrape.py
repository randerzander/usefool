#!/usr/bin/env python3
"""
Test script for Wikipedia scraping functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.wiki_scrape import scrape_wikipedia, search_wikipedia


def test_wikipedia_article():
    """Test scraping a Wikipedia article"""
    print("=" * 80)
    print("Testing Wikipedia Article Scraping")
    print("=" * 80)
    
    # Test with Python programming language article
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    print(f"\nTest 1: Fetching article from URL")
    print(f"URL: {url}")
    print("-" * 80)
    
    result = scrape_wikipedia(url)
    print(result[:1000])  # Print first 1000 chars
    print("..." if len(result) > 1000 else "")
    print(f"\nTotal length: {len(result)} chars")
    print("-" * 80)
    
    # Test with another article
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"\nTest 2: AI Article")
    print(f"URL: {url}")
    print("-" * 80)
    
    result = scrape_wikipedia(url)
    print(result[:500])
    print("..." if len(result) > 500 else "")
    print(f"\nTotal length: {len(result)} chars")
    print("-" * 80)
    
    # Test with non-existent article
    url = "https://en.wikipedia.org/wiki/ThisArticleDoesNotExist123456"
    print(f"\nTest 3: Non-existent article")
    print(f"URL: {url}")
    print("-" * 80)
    
    result = scrape_wikipedia(url)
    print(result)
    print("-" * 80)


def test_wikipedia_search():
    """Test Wikipedia search"""
    print("\n" + "=" * 80)
    print("Testing Wikipedia Search")
    print("=" * 80)
    
    query = "Machine Learning"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    result = search_wikipedia(query)
    print(result)
    print("-" * 80)


def test_from_read_url():
    """Test using Wikipedia through the read_url tool"""
    print("\n" + "=" * 80)
    print("Testing Wikipedia via read_url")
    print("=" * 80)
    
    from tools.read_url import read_url
    
    url = "https://en.wikipedia.org/wiki/Discord"
    print(f"\nURL: {url}")
    print("-" * 80)
    
    result = read_url(url)
    print(result[:800])
    print("..." if len(result) > 800 else "")
    print(f"\nTotal length: {len(result)} chars")
    print("-" * 80)


if __name__ == "__main__":
    test_wikipedia_article()
    test_wikipedia_search()
    test_from_read_url()
    print("\n✅ All tests completed!")
