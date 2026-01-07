#!/usr/bin/env python3
"""
Test script for the ReAct agent functionality.
This script tests the agent's parsing and reasoning logic without requiring API access.
"""

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import Agent

# Mock LLM response for testing
class MockAgent(Agent):
    """Mock agent for testing without API access."""
    
    def __init__(self):
        # Initialize without API key
        self.api_key = "mock_key"
        self.model = "mock_model"
        self.api_url = "mock_url"
        self.tool_functions = {
            "duckduckgo_search": lambda query=None, **kwargs: str([{"title": "Test Result", "href": "http://test.com", "body": f"Test content for {query}"}]),
            "read_url": lambda url=None, **kwargs: f"# Test Article for {url}\n\nThis is test content from the URL."
        }
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search the web using DuckDuckGo. Input should be a search query string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_url",
                    "description": "Scrape and parse HTML content from a URL into markdown format. Input should be a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"}
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
        # Initialize tracking for logging
        self.call_sequence = []
        self.token_stats = {}
        self.test_responses = []
        self.response_index = 0
    
    def _parse_response(self, text: str) -> dict:
        """
        Parse a ReAct format response (Thought/Action/Action Input/Final Answer).
        """
        import re
        
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
            
        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()
            
        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(.*?)(?=\n|$)", text)
        if input_match:
            result["action_input"] = input_match.group(1).strip()
            
        # Extract Final Answer
        answer_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        if answer_match:
            result["final_answer"] = answer_match.group(1).strip()
            
        return result

    def _execute_action(self, action: str, action_input: str) -> str:
        """
        Execute an action.
        """
        if action not in self.tool_functions:
            return f"Error: Unknown action '{action}'"
            
        try:
            return str(self.tool_functions[action](action_input))
        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def set_test_responses(self, responses):
        """Set predefined responses for testing."""
        self.test_responses = responses
        self.response_index = 0

    def _call_llm(self, messages, use_tools=True, stream=False):
        """Return mock response instead of calling actual LLM."""
        if self.model not in self.token_stats:
            self.token_stats[self.model] = {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0}
        self.token_stats[self.model]["total_calls"] += 1
        
        if self.response_index < len(self.test_responses):
            response_text = self.test_responses[self.response_index]
            self.response_index += 1
            
            # Simple simulation of Tool API response format
            parsed = self._parse_response(response_text)
            
            if stream:
                def gen():
                    if parsed['action']:
                        # Simulate tool call chunk
                        chunk = {
                            "choices": [{
                                "delta": {
                                    "tool_calls": [{
                                        "id": f"call_{self.response_index}",
                                        "function": {
                                            "name": parsed['action'],
                                            "arguments": json.dumps({"query" if "search" in parsed['action'] else "url": parsed['action_input']})
                                        }
                                    }]
                                }
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}".encode('utf-8')
                    else:
                        # Simulate content chunk
                        chunk = {
                            "choices": [{
                                "delta": {
                                    "content": parsed['final_answer']
                                }
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}".encode('utf-8')
                    yield b"data: [DONE]"
                return gen()
            else:
                # Non-stream return
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": parsed['final_answer'],
                            "tool_calls": [{
                                "id": f"call_{self.response_index}",
                                "function": {
                                    "name": parsed['action'],
                                    "arguments": json.dumps({"query" if "search" in parsed['action'] else "url": parsed['action_input']})
                                }
                            }] if parsed['action'] else None
                        }
                    }]
                }
        
        # Default end
        if stream:
            def gen():
                chunk = {"choices": [{"delta": {"content": "Final Answer: Test completed"}}]}
                yield f"data: {json.dumps(chunk)}".encode('utf-8')
                yield b"data: [DONE]"
            return gen()
        return {"choices": [{"message": {"role": "assistant", "content": "Final Answer: Test completed"}}]}


def test_parse_response():
    """Test the response parsing logic."""
    agent = MockAgent()
    
    print("Testing response parsing...")
    print("="*60)
    
    # Test 1: Parse thought, action, and action input
    response1 = """Thought: I need to search for information about Python.
Action: duckduckgo_search
Action Input: Python programming language"""
    
    parsed = agent._parse_response(response1)
    print("\nTest 1 - Parse complete ReAct response:")
    print(f"Response: {response1}")
    print(f"Parsed: {parsed}")
    assert parsed['thought'] == "I need to search for information about Python."
    assert parsed['action'] == "duckduckgo_search"
    assert parsed['action_input'] == "Python programming language"
    print("✓ Test 1 passed")
    
    # Test 2: Parse final answer
    response2 = """Final Answer: Python is a high-level programming language known for its simplicity and readability."""
    
    parsed = agent._parse_response(response2)
    print("\nTest 2 - Parse final answer:")
    print(f"Response: {response2}")
    print(f"Parsed: {parsed}")
    assert parsed['final_answer'] is not None
    assert "Python is a high-level" in parsed['final_answer']
    print("✓ Test 2 passed")
    
    print("\n✓ All parsing tests passed!")


def test_tool_execution():
    """Test the tool execution logic."""
    agent = MockAgent()
    
    print("\nTesting tool execution...")
    print("="*60)
    
    # Test search tool
    result = agent._execute_action("duckduckgo_search", "test query")
    print("\nTest 3 - Execute search tool:")
    print(f"Result: {result[:100]}...")
    assert "Test Result" in result
    print("✓ Test 3 passed")
    
    # Test scrape tool
    result = agent._execute_action("read_url", "http://example.com")
    print("\nTest 4 - Execute scrape tool:")
    print(f"Result: {result[:100]}...")
    assert "Test Article" in result
    print("✓ Test 4 passed")
    
    # Test unknown tool
    result = agent._execute_action("unknown_tool", "input")
    print("\nTest 5 - Unknown tool error handling:")
    print(f"Result: {result}")
    assert "Error" in result
    print("✓ Test 5 passed")
    
    print("\n✓ All tool execution tests passed!")


def test_agent_loop():
    """Test the full agent loop."""
    agent = MockAgent()
    
    print("\nTesting agent loop...")
    print("="*60)
    
    # Set up mock responses
    agent.set_test_responses([
        """Thought: I need to search for information.
Action: duckduckgo_search
Action Input: test query""",
        """Thought: Based on the search results, I can now provide an answer.
Final Answer: Here is the answer based on the search results."""
    ])
    
    result = agent.run("What is a test?", max_iterations=3, verbose=False)
    print("\nTest 6 - Full agent loop:")
    print(f"Result: {result}")
    assert "answer based on the search results" in result
    print("✓ Test 6 passed")
    
    print("\n✓ All agent loop tests passed!")


if __name__ == "__main__":
    print("ReAct Agent Tests")
    print("="*60)
    
    test_parse_response()
    test_tool_execution()
    test_agent_loop()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Note: To use the actual agent with real API calls:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Run: python utils.py (or use via discord_bot.py)")
    print("="*60)
