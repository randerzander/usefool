#!/usr/bin/env python3
"""
Test script for YouTube Short that failed to download.
From query log: data/query_logs/randerzander_20251220_150702.json
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.read_url import read_url

def test_youtube_short():
    """Test the YouTube short that failed: 0lE-D_2RF2I"""
    url = "https://youtube.com/shorts/0lE-D_2RF2I?si=HcxAxgecxujk_RK9"
    
    print("=" * 60)
    print("Testing YouTube Short: 0lE-D_2RF2I")
    print("=" * 60)
    print(f"\nURL: {url}\n")
    
    try:
        result = read_url(url)
        
        print("Result:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        
        # Check if it's an error
        if "Error" in result or "Failed" in result:
            print("\n⚠️  YouTube short failed to download")
            return False
        else:
            print("\n✓ YouTube short successfully processed")
            return True
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_youtube_short()
    sys.exit(0 if success else 1)
