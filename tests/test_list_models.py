#!/usr/bin/env python3
"""
Test script to list available models from the OpenAI-compatible API.
"""

import yaml
import requests
from pathlib import Path


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_models(base_url: str, api_key: str = None):
    """
    List available models from OpenAI-compatible API.
    
    Args:
        base_url: Base URL for the API (e.g., "http://localhost:8080/v1/chat/completions")
        api_key: Optional API key
        
    Returns:
        List of model dictionaries
    """
    # Convert chat/completions URL to models endpoint
    # http://localhost:8080/v1/chat/completions -> http://localhost:8080/v1/models
    if "/chat/completions" in base_url:
        models_url = base_url.replace("/chat/completions", "/models")
    elif base_url.endswith("/v1"):
        models_url = f"{base_url}/models"
    else:
        models_url = f"{base_url}/v1/models"
    
    print(f"Requesting models from: {models_url}")
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(models_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # OpenAI API format: {"object": "list", "data": [...]}
        if "data" in data:
            return data["data"]
        # Some APIs return models directly as a list
        elif isinstance(data, list):
            return data
        else:
            return [data]
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        return None


def main():
    print("=" * 60)
    print("OpenAI-Compatible API Model List Test")
    print("=" * 60)
    print()
    
    # Load config
    config = load_config()
    base_url = config.get("base_url", "http://localhost:8080/v1/chat/completions")
    
    # Try to get API key from config
    api_key_env = config.get("api_key_env")
    api_key = None
    if api_key_env:
        import os
        api_key = os.environ.get(api_key_env)
    
    print(f"Base URL: {base_url}")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    print()
    
    # List models
    models = list_models(base_url, api_key)
    
    if models:
        print(f"Found {len(models)} model(s):")
        print()
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {'-' * 55}")
            
            # Handle different response formats
            if isinstance(model, dict):
                # Get model ID/name
                model_id = model.get("id") or model.get("name") or model.get("model")
                if model_id:
                    print(f"   ID: {model_id}")
                
                # Show other useful fields
                for key, value in model.items():
                    if key not in ["id", "name", "model"]:
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 80:
                            value = value[:77] + "..."
                        print(f"   {key}: {value}")
            else:
                print(f"   {model}")
            print()
    else:
        print("No models found or error occurred.")
        print()
        print("Alternative endpoints to try:")
        print("  - http://localhost:8080/models")
        print("  - http://localhost:8080/v1/models")
        print("  - http://localhost:11434/api/tags (for Ollama)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
