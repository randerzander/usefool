#!/usr/bin/env python3
"""
Test script for YouTube download and transcription tools.
Downloads YouTube videos (both regular and Shorts), gets transcripts,
comments, and thumbnails.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.youtube import (
    is_youtube_short,
    download_youtube_video,
    download_youtube_comments,
    get_youtube_transcript,
    transcribe_audio_with_whisper
)


def main():
    """Run the test"""
    
    # Example YouTube URL - replace with your own
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Allow URL to be passed as command line argument
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    # Whisper model to use (tiny, base, small, medium, large) - only for Shorts
    whisper_model = "base"
    if len(sys.argv) > 2:
        whisper_model = sys.argv[2]
    
    # Determine if it's a Short
    is_short = is_youtube_short(test_url)
    
    print("=" * 60)
    print(f"YouTube {'Short' if is_short else 'Video'} Downloader + Transcript")
    print("=" * 60)
    print(f"URL: {test_url}")
    if is_short:
        print(f"Whisper model: {whisper_model} (for audio transcription)")
    else:
        print("Will use YouTube API for transcript")
    print()
    
    # Step 1: Download the video
    downloaded_file, video_id, thumbnail_path = download_youtube_video(test_url)
    
    if not downloaded_file:
        print("\n❌ Test failed - download failed")
        return 1
    
    # Step 2: Download top comments
    if video_id:
        comments = download_youtube_comments(video_id, max_comments=10)
    
    # Step 3: Get transcript
    transcript = None
    if is_short:
        # For Shorts, transcribe the audio with whisper
        print("\n📹 This is a Short - transcribing audio with whisper.cpp...")
        transcript = transcribe_audio_with_whisper(downloaded_file, whisper_model)
    else:
        # For regular videos, get transcript from YouTube API
        print("\n📺 This is a regular video - fetching transcript from YouTube API...")
        if video_id:
            transcript = get_youtube_transcript(video_id)
            
            # If no transcript available, fall back to whisper
            if not transcript:
                print("\n⚠️  No YouTube transcript found - falling back to audio transcription...")
                transcript = transcribe_audio_with_whisper(downloaded_file, whisper_model)
    
    if transcript:
        print("\n✅ Test completed successfully!")
        return 0
    else:
        print("\n⚠️  Download succeeded but transcript/transcription failed")
        return 1


if __name__ == "__main__":
    exit(main())
