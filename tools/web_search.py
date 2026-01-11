#!/usr/bin/env python3
"""
Web search tool using pysearx.
"""

import logging
import time
import yaml
import threading
import sys
from pathlib import Path
from typing import List, Dict
from .tool_utils import create_tool_spec


logger = logging.getLogger(__name__)

# Load config for default settings
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
try:
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

DEFAULT_MAX_RESULTS = CONFIG.get("max_search_results", 10)


# Tool specification for agent registration
TOOL_SPEC = create_tool_spec(
    name="web_search",
    description="Search the web for information. Returns a list of numbered search results with title, URL, and description.",
    parameters={
        "query": "The search query string",
        "max_results": {
            "type": "integer",
            "description": f"Maximum number of results to return (default: {DEFAULT_MAX_RESULTS})",
            "default": DEFAULT_MAX_RESULTS
        }
    },
    required=["query"]
)


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, str]]:
    """
    Search using pysearx and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing index, title, href (url), and body (description)
    """
    start_time = time.time()
    
    # Start manual spinner
    stop_spinner = threading.Event()
    
    def spin():
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not stop_spinner.is_set():
            frame = frames[i % len(frames)]
            elapsed = time.time() - start_time
            
            # Truncate query for display
            display_query = query[:50] + "..." if len(query) > 50 else query
            msg = f"Web Search | {display_query} | {elapsed:.1f}s"
            
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
    
    try:
        from pysearx import search
        
        # Use pysearx in sequential mode to avoid threading conflicts with Discord
        # parallel=False is safer in async environments like Discord bot
        raw_results = search(query, max_results=max_results, parallel=False)
        
        logger.debug(f"pysearx response - Results: {len(raw_results)}")
        
        # Convert pysearx format to expected format
        # pysearx returns: title, url, description, engine
        # we need: index, title, href, body
        results = []
        for i, result in enumerate(raw_results, 1):
            results.append({
                'index': i,
                'title': result.get('title', 'N/A'),
                'href': result.get('url', 'N/A'),
                'body': result.get('description', '')
            })
        
        # Check for no results
        if len(results) == 0:
            logger.warning(f"⚠️ WARNING: Web search returned 0 results for query '{query}'. This might indicate rate limiting or service issues.")
            return [{
                'error': 'No results found. The search service may be rate limited or temporarily unavailable. Try waiting a moment or rephrasing your query.'
            }]
        
        return results
        
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{'error': f'Error during search: {str(e)}'}]
    finally:
        if spinner_thread:
            stop_spinner.set()
            spinner_thread.join()
            # Clear the line
            sys.stdout.write("\x1b[2K\r")
            sys.stdout.flush()
        
        # Log final result
        duration = time.time() - start_time
        logger.info(f"Web search completed | Query: '{query[:50]}{'...' if len(query) > 50 else ''}' | Results: {len(results) if 'results' in locals() else 0} | Time: {duration:.2f}s")
