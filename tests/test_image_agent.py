#!/usr/bin/env python3
"""
Standalone test case for agent with image input.
Tests vision model capability with real images.
"""

import os
import sys
from agent import Agent

def test_agent_with_image():
    """Test agent with a real image URL."""
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY', 'test')
    
    # Create agent with localhost server
    print("=" * 80)
    print("Initializing agent...")
    agent = Agent(
        api_key=api_key,
        base_url='http://localhost:8080/v1/chat/completions',
        enable_logging=True
    )
    
    print(f"\nModel: {agent.model}")
    print(f"Supports vision: {agent.supports_vision}")
    print(f"API URL: {agent.api_url}")
    print("=" * 80)
    
    if not agent.supports_vision:
        print("\n❌ ERROR: Model doesn't support vision!")
        print("This test requires a vision-capable model.")
        return False
    
    # Test with a simple image URL
    # Using a publicly accessible test image
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/300px-Cat03.jpg"
    
    print(f"\n📸 Testing with image: {test_image_url}")
    print("\nQuestion: What do you see in this image?")
    print("\n" + "=" * 80)
    
    try:
        # Run agent with image
        result = agent.run(
            question="What do you see in this image? Describe it in detail.",
            image_urls=[test_image_url],
            max_iterations=10,
            verbose=True
        )
        
        print("\n" + "=" * 80)
        print("RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Check if we got a valid response
        if not result or not result.strip():
            print("\n❌ ERROR: Got empty response!")
            return False
        
        if result.strip().lower() in ["none", "i couldn't generate a response.", "maximum iterations reached without a response."]:
            print("\n❌ ERROR: Got error/fallback response!")
            print(f"Response was: {result}")
            return False
        
        print("\n✅ SUCCESS: Got valid response!")
        
        # Print tracking data
        tracking = agent.get_tracking_data()
        print("\n" + "=" * 80)
        print("TRACKING DATA:")
        print("=" * 80)
        print(f"Total LLM calls: {sum(1 for e in tracking['call_sequence'] if e['type'] == 'llm_call')}")
        print(f"Total tool calls: {sum(1 for e in tracking['call_sequence'] if e['type'] == 'tool_call')}")
        
        print("\nCall sequence:")
        for i, call in enumerate(tracking['call_sequence'], 1):
            if call['type'] == 'llm_call':
                content = call['output'].get('content', '')
                tool_calls = call['output'].get('tool_calls', [])
                print(f"  {i}. LLM Call:")
                print(f"     Content: {content[:100] if content else 'None'}...")
                print(f"     Tool calls: {len(tool_calls) if tool_calls else 0}")
            else:
                print(f"  {i}. Tool Call: {call['tool_name']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Exception occurred!")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_without_image():
    """Test agent without images to verify basic functionality."""
    
    api_key = os.getenv('OPENROUTER_API_KEY', 'test')
    
    print("\n" + "=" * 80)
    print("Testing agent WITHOUT images (baseline test)...")
    print("=" * 80)
    
    agent = Agent(
        api_key=api_key,
        base_url='http://localhost:8080/v1/chat/completions',
        enable_logging=False
    )
    
    result = agent.run(
        question="What is 2+2? Just give me the number.",
        max_iterations=10,
        verbose=False
    )
    
    print(f"\nQuestion: What is 2+2?")
    print(f"Result: {result}")
    
    if "4" in result:
        print("✅ Baseline test passed")
        return True
    else:
        print("❌ Baseline test failed")
        return False


if __name__ == "__main__":
    print("VISION MODEL IMAGE TEST")
    print("=" * 80)
    
    # Run baseline test first
    baseline_ok = test_agent_without_image()
    
    if not baseline_ok:
        print("\n❌ Baseline test failed! Check your model setup.")
        sys.exit(1)
    
    # Run image test
    print("\n\n")
    image_ok = test_agent_with_image()
    
    if image_ok:
        print("\n\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n\n❌ IMAGE TEST FAILED!")
        sys.exit(1)
