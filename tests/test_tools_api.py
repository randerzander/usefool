#!/usr/bin/env python3
"""
Test script to verify if the configured LLM supports OpenAI tools API
Uses settings from config.yaml
"""

import os
import yaml
import json
from openai import OpenAI

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_tools_api():
    """Test if the LLM supports tools API"""
    
    # Load config
    config = load_config()
    
    base_url = config.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
    model = config.get("default_model", "gpt-3.5-turbo")
    api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
    
    # OpenAI client expects base URL without /chat/completions endpoint
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    
    print(f"Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  API Key Env: {api_key_env}")
    print()
    
    # Get API key from environment
    api_key = os.getenv(api_key_env) if api_key_env else None
    
    if not api_key and api_key_env:
        print(f"⚠️  WARNING: {api_key_env} environment variable not set")
        print(f"Proceeding without API key (may work for local servers)")
        api_key = "not-needed"
    
    # Define a simple weather tool for testing
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?"
        }
    ]
    
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        print("=" * 60)
        print("TEST 1: Basic completion without tools")
        print("=" * 60)
        
        # First test without tools
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello! Just say 'Hi' back."}],
                max_tokens=50
            )
            print("✅ Basic completion works!")
            print(f"Response: {response.choices[0].message.content}")
            print()
        except Exception as e:
            print(f"❌ Basic completion failed: {e}")
            print()
            return
        
        print("=" * 60)
        print("TEST 2: Completion WITH tools API")
        print("=" * 60)
        print(f"Sending message: {messages[0]['content']}")
        print(f"Tool: {weather_tool['function']['name']}")
        print()
        
        # Test with tools
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[weather_tool],
            tool_choice="auto"
        )
        
        print("✅ Tools API call succeeded!")
        print()
        
        # Check response
        message = response.choices[0].message
        
        print(f"Response finish reason: {response.choices[0].finish_reason}")
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"✅ LLM requested tool calls! (Tools API is supported)")
            print(f"Number of tool calls: {len(message.tool_calls)}")
            print()
            
            for i, tool_call in enumerate(message.tool_calls):
                print(f"Tool Call {i+1}:")
                print(f"  ID: {tool_call.id}")
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
                print()
        else:
            print("⚠️  No tool calls in response")
            if message.content:
                print(f"LLM responded with text instead:")
                print(f"  {message.content}")
                print()
                print("❌ Tools API may not be fully supported by this server/model")
            print()
        
        # Print full response for debugging
        print("=" * 60)
        print("Full API Response (for debugging):")
        print("=" * 60)
        print(json.dumps(response.model_dump(), indent=2))
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}")
        print(f"Details: {str(e)}")
        print()
        print("This server/model likely does not support the OpenAI tools API.")
        print("Set 'use_tools_api: false' in config.yaml to use basic completion mode.")

if __name__ == "__main__":
    test_tools_api()
