#!/usr/bin/env python3
"""
Live test for image captioning with actual API calls.
Tests the image captioning functionality with the configured VLM model.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import caption_image_with_vlm


def test_image_caption_from_url():
    """Test captioning an image from a URL with actual API call"""
    
    # Use a public domain test image
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/800px-Cat03.jpg"
    
    print("\n" + "="*60)
    print("Image Captioning Live Test")
    print("="*60)
    print(f"\nImage URL: {test_image_url}")
    print("\nSending request to VLM...")
    print("-" * 60)
    
    try:
        # Test basic captioning
        caption = caption_image_with_vlm(
            image_url=test_image_url,
            prompt="Describe this image in detail."
        )
        
        print(f"\n✅ Caption received:")
        print(f"{caption}")
        print(f"\nCaption length: {len(caption)} characters")
        
        # Test with user context
        print("\n" + "="*60)
        print("Testing with user query context...")
        print("-" * 60)
        
        caption_with_query = caption_image_with_vlm(
            image_url=test_image_url,
            prompt="The user asked: 'What animal is this?'",
            user_query="What animal is this?"
        )
        
        print(f"\n✅ Caption with context:")
        print(f"{caption_with_query}")
        print(f"\nCaption length: {len(caption_with_query)} characters")
        
        print("\n" + "="*60)
        print("✅ Live image captioning test completed successfully!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during image captioning:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60 + "\n")
        return False


if __name__ == "__main__":
    success = test_image_caption_from_url()
    sys.exit(0 if success else 1)
