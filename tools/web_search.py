#!/usr/bin/env python3
"""
Web search tool using SearXNG.
"""

import logging
import time
import requests
import yaml
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
    Search using local SearXNG instance and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing index, title, url, and description
    """
    start_time = time.time()
    try:
        from bs4 import BeautifulSoup
        
        # Make request to SearXNG - disable DuckDuckGo which often returns CAPTCHAs
        url = f"http://localhost:8081/search?q={query}&disabled_engines=duckduckgo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        logger.debug(f"SearXNG response - Status: {response.status_code}, Content length: {len(response.text)}")
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all result articles
        articles = soup.find_all('article', class_='result')
        
        results = []
        for i, article in enumerate(articles[:max_results], 1):
            # Extract title
            h3 = article.find('h3')
            title_link = h3.find('a') if h3 else None
            title = title_link.get_text(strip=True) if title_link else 'N/A'
            
            # Extract URL
            url_link = article.find('a', class_='url_header')
            url = url_link.get('href') if url_link else 'N/A'
            
            # Extract description/body
            body_div = article.find('p', class_='content')
            body = body_div.get_text(strip=True) if body_div else ''
            
            results.append({
                'index': i,
                'title': title,
                'href': url,
                'body': body
            })
        
        # Check for rate limiting or no results
        if len(results) == 0:
            logger.warning(f"⚠️ WARNING: Web search returned 0 results for query '{query}'. This might indicate rate limiting or service issues.")
            return [{
                'error': 'No results found. The search service may be rate limited or temporarily unavailable. Try waiting a moment or rephrasing your query.'
            }]
        
        return results
        
    except requests.exceptions.Timeout:
        logger.error("SearXNG request timed out")
        return [{'error': 'Search request timed out. The search service may be unavailable.'}]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to SearXNG: {str(e)}")
        return [{'error': f'Error connecting to search service: {str(e)}'}]
    except Exception as e:
        logger.error(f"Unexpected error in web_search: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{'error': f'Unexpected error during search: {str(e)}'}]
