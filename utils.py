#!/usr/bin/env python3
"""
General utility functions for the usefool agent.
Includes image processing, configuration loading, and shared constants.
"""

import os
import json
import base64
import logging
import time
import yaml
import requests
import threading
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4

DEFAULT_MODEL_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "image_caption_model": "nvidia/nemotron-nano-12b-v2-vl:free"
}

def load_model_config():
    """Load model configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Config loading failed: {e}, using defaults")
        return DEFAULT_MODEL_CONFIG.copy()

MODEL_CONFIG = load_model_config()

def is_image_url(url: str) -> bool:
    """Check if a URL likely points to an image based on common extensions."""
    # Remove query parameters for extension check
    path = url.split('?')[0].lower()
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    return any(path.endswith(ext) for ext in image_extensions) or url.startswith("data:image/")

def download_image(url: str) -> str:
    """Download an image from a URL and save it to a temporary file."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        import tempfile
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(response.content)
            return f.name
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise

def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def caption_image_with_vlm(image_url: str, api_key: str, prompt: str = None, model: str = None, base_url: str = None):
    """
    Caption an image using a Vision Language Model.
    """
    if model is None:
        model = MODEL_CONFIG.get("image_caption_model", "nvidia/nemotron-nano-12b-v2-vl:free")
    
    if base_url is None:
        base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    if prompt is None:
        prompt = "Describe this image in detail."

    image_path = None
    try:
        # If it's a URL, download it
        if image_url.startswith(("http://", "https://")):
            if not is_image_url(image_url):
                logger.warning(f"URL does not appear to be an image: {image_url}")
                raise ValueError(f"URL does not appear to be an image: {image_url}")
            image_path = download_image(image_url)
        else:
            image_path = image_url

        base64_image = image_to_base64(image_path)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Manual spinner implementation to match agent.py
        stop_spinner = threading.Event()
        start_time = time.time()
        
        def spin():
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while not stop_spinner.is_set():
                frame = frames[i % len(frames)]
                elapsed = time.time() - start_time
                
                # Format message - shorten model for cleaner display
                short_model = model.split('/')[-1] if '/' in model else model
                msg = f"VLM Call | {short_model} | {elapsed:.1f}s"
                
                # Use stdout with ANSI clear + return to force single line update
                sys.stdout.write(f"\x1b[2K\r{frame} {msg}")
                sys.stdout.flush()
                
                time.sleep(0.1)
                i += 1
                
        spinner_thread = None
        # Only run spinner if likely interactive, otherwise fallback to log
        if sys.stdout.isatty():
            spinner_thread = threading.Thread(target=spin)
            spinner_thread.daemon = True
            spinner_thread.start()
        else:
            logger.info(f"VLM LLM Call: model={model}")
        
        try:
            response = requests.post(base_url, headers=headers, json=data, timeout=60)
        finally:
            if spinner_thread:
                stop_spinner.set()
                spinner_thread.join()
                # Clear the line
                sys.stdout.write("\x1b[2K\r")
                sys.stdout.flush()
        
        response.raise_for_status()
        
        # Log completion with time
        total_time = time.time() - start_time
        logger.info(f"VLM call completed | Status: {response.status_code} | Time: {total_time:.2f}s")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    finally:
        # Clean up temp file if we downloaded it
        if image_path and image_url.startswith(("http://", "https://")) and os.path.exists(image_path):
            os.remove(image_path)

def two_round_image_caption(image_url: str, api_key: str, user_query: str = None) -> str:
    """
    Two-round image captioning for better detail.
    """
    # Round 1: General description
    round1_caption = caption_image_with_vlm(image_url, api_key, "Describe this image in detail.")
    
    # Round 2: Focused analysis
    prompt = f"Based on this initial description: '{round1_caption}'"
    if user_query:
        prompt += f"\n\nAnd the user's question: '{user_query}'"
    prompt += "\n\nProvide a more detailed and focused analysis of the image."
    
    round2_caption = caption_image_with_vlm(image_url, api_key, prompt)
    
    return f"Initial Description: {round1_caption}\n\nDetailed Analysis: {round2_caption}"