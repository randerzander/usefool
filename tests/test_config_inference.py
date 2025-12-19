#!/usr/bin/env python3
"""
Test script for LLM inference using config.yaml settings
Tests connection and chat completion with configured model and API
"""

import os
import yaml
from openai import OpenAI

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_inference():
    """Test basic inference using config.yaml settings"""
    
    # Load config
    config = load_config()
    
    base_url = config.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
    
    # OpenAI client expects base URL without /chat/completions endpoint
    # Remove it if present since the client adds it automatically
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    
    model = config.get("default_model", "nemotron")
    api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
    temperature = config.get("temperature", 1)
    top_p = config.get("top_p", 1)
    max_tokens = config.get("max_tokens", 16384)
    enable_thinking = config.get("enable_thinking", False)
    
    print(f"Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  API Key Env: {api_key_env}")
    print(f"  Temperature: {temperature}")
    print(f"  Top P: {top_p}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Enable Thinking: {enable_thinking}")
    print()
    
    # Get API key from environment
    api_key = os.getenv(api_key_env) if api_key_env else None
    
    if not api_key and api_key_env:
        print(f"❌ ERROR: {api_key_env} environment variable not set")
        print(f"\nPlease set it with:")
        print(f"  export {api_key_env}='your-api-key-here'")
        return
    
    # Initialize OpenAI client
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        print("Testing LLM inference...\n")
        
        # Test message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello! Tell me a short joke about programming."
            }
        ]
        
        # Make API call
        call_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        # Add thinking mode for Nemotron models if enabled
        if enable_thinking:
            call_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        
        response = client.chat.completions.create(**call_params)
        
        print("✅ Success!")
        print(f"\nModel: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        print(f"\n📝 Response:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}")
        print(f"Details: {str(e)}")
        
        if "Connection" in str(e):
            print(f"\nMake sure the server at {base_url} is running and accessible.")

if __name__ == "__main__":
    test_inference()
