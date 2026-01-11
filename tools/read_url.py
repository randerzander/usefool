#!/usr/bin/env python3
"""
URL reading utilities with special handling for different content types.

Automatically detects URL type and extracts content appropriately:
- YouTube URLs: Fetches video transcripts
- Wikipedia URLs: Uses Wikipedia API for clean article text
- Regular web pages: Converts HTML to markdown

Provides a unified interface for reading any URL content.
"""

import logging
import requests
import html2text
import threading
import sys
import time
from pyreadability import Readability
from tools import retriever
from pathlib import Path
from .tool_utils import create_tool_spec


logger = logging.getLogger(__name__)


def _scrape_with_playwright(url: str) -> str:
    """
    Fallback scraper using Playwright for sites that block requests.
    Uses a real browser to render JavaScript and bypass bot detection.
    
    Runs in a separate thread to avoid conflicts with async event loops.
    
    Args:
        url: URL to scrape
        
    Returns:
        Markdown content of the page
    """
    import concurrent.futures
    
    def _do_playwright_scrape(url: str) -> str:
        """Inner function that does the actual Playwright scraping."""
        try:
            from playwright.sync_api import sync_playwright
            
            logger.info(f"Using Playwright to scrape {url}")
            
            with sync_playwright() as p:
                # Launch browser in headless mode
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                
                # Navigate to URL with timeout
                page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait for body to be visible
                page.wait_for_selector('body', timeout=10000)
                
                # Get the full HTML content
                html_content = page.content()
                
                # Extract title
                page_title = page.title()
                
                browser.close()
            
            # Use pyreadability to parse the HTML
            reader = Readability(html_content)
            result = reader.parse()
            
            if not result or not result.get('content'):
                # If readability fails, try to get article/main content directly
                logger.warning(f"Readability failed, attempting direct content extraction")
                
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    page = context.new_page()
                    page.goto(url, wait_until='networkidle', timeout=30000)
                    page.wait_for_selector('body', timeout=10000)
                    
                    # Try to find main content using common selectors
                    selectors = ['article', 'main', '[role="main"]', '.article-body', '.post-content', '#content']
                    content_html = None
                    
                    for selector in selectors:
                        try:
                            element = page.query_selector(selector)
                            if element:
                                content_html = element.inner_html()
                                logger.debug(f"Found content using selector: {selector}")
                                break
                        except:
                            continue
                    
                    if not content_html:
                        # Fallback to body
                        content_html = page.query_selector('body').inner_html()
                    
                    browser.close()
                
                # Convert to markdown
                h = html2text.HTML2Text()
                h.body_width = 0
                h.ignore_links = False
                markdown_content = h.handle(content_html)
                
                if page_title:
                    markdown_content = f"# {page_title}\n\n{markdown_content}"
                
            else:
                # Convert to markdown
                h = html2text.HTML2Text()
                h.body_width = 0
                h.ignore_links = False
                markdown_content = h.handle(result['content'])
                
                # Add title if available
                if result.get('title'):
                    markdown_content = f"# {result['title']}\n\n{markdown_content}"
                elif page_title:
                    markdown_content = f"# {page_title}\n\n{markdown_content}"
            
            logger.info(f"Successfully scraped {url} using Playwright ({len(markdown_content)} chars)")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {str(e)}")
            raise
    
    try:
        # Run Playwright in a thread pool to avoid async conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_playwright_scrape, url)
            return future.result(timeout=60)  # 60 second timeout
            
    except ImportError:
        logger.error("Playwright not installed. Install with: pip install playwright && playwright install chromium")
        return "Error: Playwright not available. Cannot bypass site restrictions."
    except Exception as e:
        logger.error(f"Playwright scraping failed for {url}: {str(e)}")
        return f"Error: Playwright scraping failed: {str(e)}"


# Tool specification for agent registration
TOOL_SPEC = create_tool_spec(
    name="read_url",
    description="Read and extract content from any URL. Handles YouTube videos (returns transcript), Wikipedia articles (returns article content), PDF files (downloads to scratch/ and extracts text), and regular web pages (returns markdown). Use this for ALL URLs including YouTube videos, Wikipedia articles, PDFs, documentation, blog posts, etc. If the document is expected to be large, providing a 'query' will return only the most relevant sections.",
    parameters={
        "url": "The URL to read (supports YouTube, Wikipedia, PDFs, and any web page)",
        "query": "Optional: A specific search query to find the most relevant parts of the document if it is large (like a PDF or long article)."
    },
    required=["url"]
)


def read_url(url: str, query: str = None) -> str:
    """
    Read content from any URL and return it in a readable format.
    
    Automatically handles different URL types:
    - YouTube videos: Returns transcript
    - Wikipedia articles: Returns article content via API
    - PDF files: Downloads to scratch/ and extracts text
    - Regular web pages: Returns HTML converted to markdown
    
    Also writes all fetched content to the vector database.
    If content is large (>1024 tokens), automatically searches VDB for relevant chunks
    using the user's original query.
    
    Args:
        url: URL to read
        query: Optional search query for large documents (if not provided, uses user's original query)
        
    Returns:
        Content from the URL in markdown format or relevant chunks if document is large
    """
    # Get user's original query from agent framework if not explicitly provided
    if not query:
        try:
            from agent import get_current_user_query
            query = get_current_user_query()
            if query:
                logger.debug(f"Using user's original query for VDB search: {query}")
        except Exception as e:
            logger.debug(f"Could not get user query: {e}")
    
    content = None
    
    # Check if it's a YouTube URL
    if 'youtube.com' in url or 'youtu.be' in url:
        content = _scrape_youtube_url(url)
    
    # Check if it's a Wikipedia URL
    elif 'wikipedia.org' in url:
        content = _scrape_wikipedia_url(url)
    
    # Regular web page scraping
    else:
        try:
            # Build headers with User-Agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            # Add GitHub token if scraping GitHub
            if 'github.com' in url:
                github_token_file = Path(__file__).parent.parent / '.github_token'
                if github_token_file.exists():
                    with open(github_token_file, 'r') as f:
                        github_token = f.read().strip()
                        if github_token:
                            headers['Authorization'] = f'token {github_token}'
                            logger.debug("Using GitHub token for authentication")
            
            # First do a HEAD request to check content type
            head_resp = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
            content_type = head_resp.headers.get('Content-Type', '').lower()
            
            if any(img_type in content_type for img_type in ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/bmp']):
                return f"This URL points to an image ({content_type}). Please use the 'caption_image' tool to analyze it."
            
            # Check if it's a PDF
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                content = _scrape_pdf_url(url)
            else:
                try:
                    response = requests.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    logger.debug(f"URL fetch completed - Status: {response.status_code}, Content length: {len(response.text)} chars")
                    
                    # Use pyreadability to parse HTML and extract main content
                    reader = Readability(response.text)
                    result = reader.parse()
                    
                    if not result:
                        logger.warning(f"Readability failed to parse content from {url}")
                        return f"Error: Failed to parse content from {url}. The page might be empty or not contain extractable text."
                    
                    # Convert HTML content to markdown using html2text
                    h = html2text.HTML2Text()
                    h.body_width = 0  # Don't wrap lines
                    h.ignore_links = False
                    markdown_content = h.handle(result['content'])
                    
                    # Add title if available
                    if result.get('title'):
                        markdown_content = f"# {result['title']}\n\n{markdown_content}"
                    
                    content = markdown_content
                    
                except requests.exceptions.HTTPError as e:
                    # Check if it's a blocking error (403, 429, etc.)
                    if e.response.status_code in [403, 429, 503]:
                        logger.warning(f"Got {e.response.status_code} error, trying Playwright fallback")
                        content = _scrape_with_playwright(url)
                    else:
                        raise
                        
        except Exception as e:
            # If it's not a handled error, check if we should try Playwright
            if 'Forbidden' in str(e) or '403' in str(e):
                logger.warning(f"URL scraping failed with blocking error, trying Playwright: {str(e)}")
                try:
                    content = _scrape_with_playwright(url)
                except Exception as playwright_error:
                    logger.error(f"Playwright fallback also failed: {str(playwright_error)}")
                    return f"Error scraping URL: {str(e)}. Playwright fallback also failed: {str(playwright_error)}"
            else:
                logger.error(f"URL scraping failed for {url}: {str(e)}")
                return f"Error scraping URL: {str(e)}"
    
    # Write to vector database if content was successfully fetched
    if content and not content.startswith("Error"):
        try:
            write_status = retriever.write(source=url, text=content)
            logger.debug(f"Wrote content from {url} to vector database: {write_status}")
            
            # Check if content is large (> 1024 tokens) AND a query was provided
            # retriever.write return status message like "Stored X chunks..."
            # If X > 1, it's > 1024 tokens.
            import re
            match = re.search(r"Stored (\d+) chunks", write_status)
            if match and int(match.group(1)) > 1 and query:
                num_chunks = int(match.group(1))
                
                logger.info(f"Large document detected ({num_chunks} chunks). Performing VDB search for query: {query}")
                search_results = retriever.search(query, limit=2)
                
                # Extract filename/title for the header
                title = "Document"
                if "**Source URL:**" in content:
                    title_match = re.search(r"# PDF Document: (.*)", content)
                    if title_match:
                        title = title_match.group(1)
                elif content.startswith("# "):
                    title = content.split("\n")[0][2:]

                return f"# {title} (Relevant Sections)\n\n**Source URL:** {url}\n\n[Note: This is a large document with {num_chunks} chunks. Showing top 2 most relevant sections for your query: {repr(query)}]\n\n{search_results}"
                
        except Exception as e:
            logger.warning(f"Failed to process vector database for {url}: {str(e)}")
    
    return content


def _scrape_pdf_url(url: str) -> str:
    """
    Handle PDF URLs by downloading the file and extracting text.
    Saves the PDF to scratch/ and extracts all text using pypdfium2.
    
    Args:
        url: PDF URL
        
    Returns:
        Extracted text content from the PDF
    """
    try:
        import pypdfium2 as pdfium
        from datetime import datetime
        
        # Create scratch directory if it doesn't exist
        scratch_dir = Path("scratch")
        scratch_dir.mkdir(exist_ok=True)
        
        # Generate filename from URL or timestamp
        url_filename = url.split('/')[-1].split('?')[0]
        if url_filename.lower().endswith('.pdf'):
            filename = url_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"PDF_{timestamp}.pdf"
        
        filepath = scratch_dir / filename
        
        # Download PDF
        logger.info(f"Downloading PDF from {url}...")
        
        # Build headers with User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Add GitHub token if downloading from GitHub
        if 'github.com' in url:
            github_token_file = Path(__file__).parent.parent / '.github_token'
            if github_token_file.exists():
                with open(github_token_file, 'r') as f:
                    github_token = f.read().strip()
                    if github_token:
                        headers['Authorization'] = f'token {github_token}'
                        logger.debug("Using GitHub token for PDF download")
        
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        
        # Save to file
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Saved PDF to {filepath} ({len(response.content)} bytes)")
        
        # Extract text using pypdfium2
        pdf = pdfium.PdfDocument(str(filepath))
        num_pages = len(pdf)
        all_text = []
        
        for page_num in range(num_pages):
            page = pdf[page_num]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            all_text.append(f"--- Page {page_num + 1} ---\n{text}")
        
        pdf.close()
        
        extracted_text = "\n\n".join(all_text)
        logger.info(f"Extracted {len(extracted_text)} characters from {num_pages} pages")
        
        # Limit PDF content to avoid context overflow
        # Use detected context length to calculate safe limit (reserve ~50% for context)
        import tools
        context_tokens = tools.DETECTED_CONTEXT_LENGTH
        max_pdf_tokens = int(context_tokens * 0.5)  # Use 50% of context for PDF
        MAX_PDF_CHARS = int(max_pdf_tokens * 3.9)  # Convert tokens to chars
        
        if len(extracted_text) > MAX_PDF_CHARS:
            logger.warning(f"PDF content truncated from {len(extracted_text)} to {MAX_PDF_CHARS} characters (context limit: {context_tokens:,} tokens)")
            extracted_text = extracted_text[:MAX_PDF_CHARS] + f"\n\n[... Content truncated to fit context window. Full PDF saved to {filepath}. Total: {len(extracted_text):,} chars, {num_pages} pages ...]"
        
        # Format response
        result = f"""# PDF Document: {filename}

**Source URL:** {url}
**Saved to:** {filepath}
**Pages:** {num_pages}
**Size:** {len(response.content)} bytes

## Extracted Text

{extracted_text}"""
        
        return result
        
    except ImportError as e:
        logger.error(f"pypdfium2 not available: {e}")
        return f"Error: PDF extraction functionality not available. Missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Error extracting PDF from {url}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error extracting PDF: {str(e)}"


def _scrape_youtube_url(url: str) -> str:
    """
    Handle YouTube URLs by extracting video ID and fetching transcript.
    For Shorts, automatically downloads and transcribes with whisper.
    If transcript is unavailable or very short (<1500 chars), enriches with thumbnail caption and top comments.
    
    Args:
        url: YouTube URL
        
    Returns:
        Transcript or video information in markdown format
    """
    try:
        from tools.youtube import (
            is_youtube_short, 
            get_youtube_transcript,
            download_youtube_video,
            transcribe_audio_with_whisper,
            download_youtube_comments
        )
        import re
        import tempfile
        
        # logger.info(f"Detected YouTube URL, fetching transcript...")
        
        # Extract video ID from various YouTube URL formats
        video_id = None
        
        # Pattern for /watch?v=VIDEO_ID
        match = re.search(r'[?&]v=([^&]+)', url)
        if match:
            video_id = match.group(1)
        
        # Pattern for /shorts/VIDEO_ID
        match = re.search(r'/shorts/([^/?]+)', url)
        if match:
            video_id = match.group(1)
        
        # Pattern for youtu.be/VIDEO_ID
        match = re.search(r'youtu\.be/([^/?]+)', url)
        if match:
            video_id = match.group(1)
        
        if not video_id:
            logger.warning(f"Could not extract video ID from YouTube URL: {url}")
            return f"Error: Could not extract video ID from YouTube URL"
        
        # Start manual spinner
        stop_spinner = threading.Event()
        start_time = time.time()
        
        def spin():
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while not stop_spinner.is_set():
                frame = frames[i % len(frames)]
                elapsed = time.time() - start_time
                
                msg = f"Fetching YouTube transcript | {video_id} | {elapsed:.1f}s"
                
                # Use stdout with ANSI clear + return to force single line update
                sys.stdout.write(f"\x1b[2K\r{frame} {msg}")
                sys.stdout.flush()
                
                time.sleep(0.1)
                i += 1
                
        spinner_thread = None
        if sys.stdout.isatty():
            spinner_thread = threading.Thread(target=spin)
            spinner_thread.daemon = True
            spinner_thread.start()
        
        transcript = None
        thumbnail_path = None
        
        try:
            # Check if it's a Short
            if is_youtube_short(url):
                # logger.info(f"Processing YouTube Short: {video_id}")
                
                # Use scratch directory for download
                scratch_dir = Path("scratch")
                scratch_dir.mkdir(exist_ok=True)
                
                # Download the Short (also gets thumbnail)
                video_path, _, thumbnail_path = download_youtube_video(url, output_dir=str(scratch_dir))
                
                if not video_path:
                    logger.error(f"Failed to download YouTube Short: {video_id}")
                    return f"""# YouTube Short: {video_id}

**Error:** Failed to download the video.

URL: {url}
Video ID: {video_id}"""
                
                # Transcribe with whisper
                transcript = transcribe_audio_with_whisper(video_path, model="base", output_dir=str(scratch_dir))
                
                if transcript:
                    pass # logger.info(f"Transcribed YouTube Short: {len(transcript)} chars")
                
                # Always enrich Shorts with comments and thumbnail caption
                # logger.info("Enriching YouTube Short with thumbnail caption and comments...")
                
                # Get top 10 comments - save to scratch
                comments = download_youtube_comments(video_id, output_dir=str(scratch_dir), max_comments=10)
                
                # Caption the thumbnail if available
                thumbnail_caption = None
                if thumbnail_path:
                    try:
                        import os
                        from utils import two_round_image_caption, MODEL_CONFIG
                        
                        # Get API key from environment
                        api_key_env = MODEL_CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
                        api_key = os.environ.get(api_key_env, "")
                        
                        thumbnail_caption = two_round_image_caption(thumbnail_path, api_key)
                        # logger.info(f"Generated thumbnail caption: {len(thumbnail_caption)} chars")
                    except Exception as e:
                        logger.warning(f"Failed to caption thumbnail: {e}")
                
                # Build the enriched Short response
                result = f"""# YouTube Short: {video_id}

URL: {url}
Video ID: {video_id}

"""
                
                # Add transcript section
                if transcript:
                    result += f"""## Transcript (Auto-transcribed with Whisper)

{transcript}

"""
                else:
                    result += "## Transcript\n\n*No transcript available*\n\n"
                
                # Add thumbnail caption section
                if thumbnail_caption:
                    result += f"""## Thumbnail Caption

{thumbnail_caption}

"""
                
                # Add comments section
                if comments:
                    result += "## Top 10 Comments\n\n"
                    for i, comment in enumerate(comments, 1):
                        author = comment.get('author', 'Unknown')
                        text = comment.get('text', '')
                        likes = comment.get('like_count', 0)
                        result += f"{i}. **{author}** ({likes} likes)\n   {text}\n\n"
                    # logger.info(f"Added {len(comments)} top comments")
                
                return result
            else:
                # Try to get transcript for regular videos
                # logger.info(f"Fetching YouTube transcript for: {video_id}")
                
                # Use a temporary directory that we'll ignore (don't save the file)
                with tempfile.TemporaryDirectory() as tmpdir:
                    transcript = get_youtube_transcript(video_id, output_dir=tmpdir)
                
                if transcript:
                    pass # logger.info(f"Got YouTube transcript: {len(transcript)} chars")
                else:
                    logger.warning(f"No transcript available for YouTube video: {video_id} - checking video length...")
                    
                    # Check video duration before attempting to download and transcribe
                    try:
                        import yt_dlp
                        
                        # Get video info using yt-dlp Python API
                        ydl_opts = {
                            'quiet': True,
                            'no_warnings': True,
                            'extract_flat': True,  # Don't download, just get metadata
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            video_info = ydl.extract_info(url, download=False)
                        
                        duration_seconds = video_info.get('duration', 0)
                        duration_minutes = duration_seconds / 60
                        
                        # logger.info(f"Video duration: {duration_minutes:.1f} minutes")
                        
                        # If video is 10 minutes or longer, don't transcribe
                        if duration_seconds >= 600:  # 10 minutes
                            logger.info(f"Video too long for transcription: {duration_minutes:.1f} minutes")
                            return f"""# YouTube Video: {video_id}

**No transcript available** for this video.

URL: {url}
Video ID: {video_id}
Duration: {duration_minutes:.1f} minutes

**Note:** This video is too long ({duration_minutes:.1f} minutes) to automatically transcribe without existing captions. Transcribing long videos requires significant processing time and resources.

If you need the transcript, you can:
1. Request the video creator to add captions
2. Use dedicated transcription services
3. Download and transcribe manually with the YouTube tools"""
                        
                        # logger.info(f"Video length OK ({duration_minutes:.1f} minutes), proceeding with transcription...")
                        
                    except Exception as e:
                        logger.warning(f"Could not check video duration: {e}")
                        # Continue anyway if we can't check duration
                    
                    # Fall back to downloading and transcribing with whisper
                    scratch_dir = Path("scratch")
                    scratch_dir.mkdir(exist_ok=True)
                    
                    # logger.info(f"Downloading YouTube video for transcription...")
                    video_path, _, thumbnail_path = download_youtube_video(url, output_dir=str(scratch_dir))
                    
                    if not video_path:
                        logger.error(f"Failed to download YouTube video: {video_id}")
                        return f"""# YouTube Video: {video_id}

**No transcript available** for this video and download failed.

URL: {url}
Video ID: {video_id}"""
                    
                    # logger.info(f"Transcribing YouTube video audio with whisper...")
                    transcript = transcribe_audio_with_whisper(video_path, model="base", output_dir=str(scratch_dir))
                    
                    if not transcript:
                        logger.error(f"Failed to transcribe YouTube video: {video_id}")
                        return f"""# YouTube Video: {video_id}

**Error:** No captions available and audio transcription failed.

URL: {url}
Video ID: {video_id}"""
                    
                    # logger.info(f"Successfully transcribed YouTube video - {len(transcript)} chars")
            
            # If we have no transcript or a very short transcript (<1500 chars),
            # enrich with thumbnail caption and top comments
            if not transcript or len(transcript) < 1500:
                # logger.info(f"Transcript is {'missing' if not transcript else 'short'} ({len(transcript) if transcript else 0} chars), enriching with comments and thumbnail...")
                
                # Get top 10 comments - save to scratch
                scratch_dir = Path("scratch")
                scratch_dir.mkdir(exist_ok=True)
                comments = download_youtube_comments(video_id, output_dir=str(scratch_dir), max_comments=10)
                
                # Build the enriched response
                result = f"""# YouTube {'Short' if is_youtube_short(url) else 'Video'}: {video_id}

URL: {url}
Video ID: {video_id}
"""
                
                if transcript:
                    result += f"""
## Transcript (Auto-transcribed with Whisper)

{transcript}
"""
                
                # Add comments if available
                if comments:
                    result += "\n## Top 10 Comments\n\n"
                    for i, comment in enumerate(comments, 1):
                        author = comment.get('author', 'Unknown')
                        text = comment.get('text', '')
                        likes = comment.get('like_count', 0)
                        result += f"{i}. **{author}** ({likes} likes)\n   {text}\n\n"
                    
                    # logger.info(f"Added {len(comments)} top comments")
                
                return result
            
            # Regular response with full transcript
            video_type = "Short" if is_youtube_short(url) else "Video"
            return f"""# YouTube {video_type} Transcript
                
**Video ID:** {video_id}
**URL:** {url}

## Transcript{' (Auto-transcribed with Whisper)' if is_youtube_short(url) or not get_youtube_transcript(video_id, output_dir=tempfile.gettempdir()) else ''}

{transcript}"""
        finally:
            if spinner_thread:
                stop_spinner.set()
                spinner_thread.join()
                # Clear the line
                sys.stdout.write("\x1b[2K\r")
                sys.stdout.flush()
            
            # Log final success/result
            duration = time.time() - start_time
            char_count = len(transcript) if transcript else 0
            logger.info(f"Got YouTube transcript for {video_id}: {char_count} chars (Time: {duration:.2f}s)")
            
    except ImportError as e:
        logger.error(f"YouTube tools not available: {e}")
        return f"Error: YouTube transcript functionality not available. Missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error fetching YouTube transcript: {str(e)}"


def _scrape_wikipedia_url(url: str) -> str:
    """
    Handle Wikipedia URLs by using the Wikipedia API.
    
    Args:
        url: Wikipedia URL
        
    Returns:
        Article content in markdown format
    """
    try:
        from tools.wiki_scrape import scrape_wikipedia
        
        logger.info(f"Detected Wikipedia URL, using Wikipedia API...")
        return scrape_wikipedia(url)
        
    except ImportError as e:
        logger.error(f"Wikipedia tools not available: {e}")
        return f"Error: Wikipedia functionality not available. Missing dependency: {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching Wikipedia article: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error fetching Wikipedia article: {str(e)}"
