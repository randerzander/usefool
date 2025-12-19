#!/usr/bin/env python3
"""
Test script for read_url with YouTube support.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.read_url import read_url


def test_youtube_url():
    """Test reading a YouTube video URL"""
    print("=" * 60)
    print("Testing YouTube URL reading")
    print("=" * 60)
    
    # Test with a regular YouTube video
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"\nTest 1: Regular YouTube video")
    print(f"URL: {youtube_url}")
    print("-" * 60)
    
    result = read_url(youtube_url)
    print(result[:500])  # Print first 500 chars
    print("..." if len(result) > 500 else "")
    print("-" * 60)
    
    # Test with a YouTube Short
    short_url = "https://www.youtube.com/shorts/ABC123"
    print(f"\nTest 2: YouTube Short")
    print(f"URL: {short_url}")
    print("-" * 60)
    
    result = read_url(short_url)
    print(result[:500])
    print("..." if len(result) > 500 else "")
    print("-" * 60)


def test_regular_url():
    """Test reading a regular web page"""
    print("\n" + "=" * 60)
    print("Testing regular URL reading")
    print("=" * 60)
    
    url = "https://example.com"
    print(f"URL: {url}")
    print("-" * 60)
    
    result = read_url(url)
    print(result[:300])
    print("..." if len(result) > 300 else "")
    print("-" * 60)


if __name__ == "__main__":
    test_youtube_url()
    test_regular_url()
    print("\n✅ Tests completed!")
