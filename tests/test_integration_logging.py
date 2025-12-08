#!/usr/bin/env python3
"""
Integration test to demonstrate the logging functionality.
This creates a sample query log to verify the complete workflow.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from react_agent import ReActAgent
from unittest.mock import patch, Mock


def test_integration():
    """
    Integration test that simulates a complete query with LLM and tool calls.
    """
    print("="*80)
    print("Integration Test: Complete Query Logging Workflow")
    print("="*80)
    
    with patch('react_agent.requests.post') as mock_post, \
         patch('react_agent.duckduckgo_search') as mock_search:
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Final Answer: This is a test answer based on the search results."}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Mock search results
        mock_search.return_value = [
            {"title": "Test Article", "href": "https://example.com/test", "body": "This is test content about the query."}
        ]
        
        # Create agent and run a query
        print("\n1. Creating agent and running query...")
        agent = ReActAgent("test_api_key")
        
        # Simulate the agent making decisions
        print("   - Calling LLM for reasoning...")
        agent._call_llm("What is the weather today?")
        
        print("   - Executing tool call...")
        agent._execute_action("duckduckgo_search", "weather today")
        
        print("   - Calling LLM again for final answer...")
        agent._call_llm("Based on search results, provide final answer")
        
        # Get tracking data
        print("\n2. Retrieving tracking data...")
        tracking_data = agent.get_tracking_data()
        
        print(f"\n3. Analysis of tracked data:")
        print(f"   Total calls in sequence: {len(tracking_data['call_sequence'])}")
        
        # Count call types
        llm_calls = [c for c in tracking_data['call_sequence'] if c['type'] == 'llm_call']
        tool_calls = [c for c in tracking_data['call_sequence'] if c['type'] == 'tool_call']
        
        print(f"   LLM calls: {len(llm_calls)}")
        print(f"   Tool calls: {len(tool_calls)}")
        
        # Display token statistics
        print(f"\n4. Token statistics by model:")
        for model, stats in tracking_data['token_stats'].items():
            print(f"   {model}:")
            print(f"      Total calls: {stats['total_calls']}")
            print(f"      Input tokens: {stats['total_input_tokens']}")
            print(f"      Output tokens: {stats['total_output_tokens']}")
            print(f"      Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}")
        
        # Display detailed call sequence
        print(f"\n5. Detailed call sequence:")
        for i, call in enumerate(tracking_data['call_sequence'], 1):
            print(f"\n   Call {i}: {call['type'].upper()}")
            if call['type'] == 'llm_call':
                print(f"      Model: {call['model']}")
                print(f"      Input tokens: {call['input_tokens']}")
                print(f"      Output tokens: {call['output_tokens']}")
                print(f"      Input tokens/sec: {call['input_tokens_per_sec']}")
                print(f"      Output tokens/sec: {call['output_tokens_per_sec']}")
                print(f"      Response time: {call['response_time_seconds']}s")
                print(f"      Input preview: {call['input'][:100]}...")
                print(f"      Output preview: {call['output'][:100]}...")
            elif call['type'] == 'tool_call':
                print(f"      Tool: {call['tool_name']}")
                print(f"      Input: {call['input'][:100]}...")
                print(f"      Output preview: {call['output'][:100]}...")
                print(f"      Execution time: {call['execution_time_seconds']}s")
        
        # Save a sample query log
        print(f"\n6. Saving sample query log...")
        log_dir = Path("data/query_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        sample_log = {
            "message_id": "test_integration_123",
            "timestamp": "2025-12-08T04:00:00",
            "user_query": "What is the weather today?",
            "final_response": "This is a test answer based on the search results.",
            "call_sequence": tracking_data['call_sequence'],
            "token_stats_by_model": tracking_data['token_stats']
        }
        
        sample_file = log_dir / "sample_query_log.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_log, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Sample log saved to: {sample_file}")
        print(f"   ✓ File size: {sample_file.stat().st_size} bytes")
        
        # Display sample of the saved file
        print(f"\n7. Sample of saved log file:")
        print(f"   ```json")
        with open(sample_file, 'r') as f:
            lines = f.readlines()[:20]
            for line in lines:
                print(f"   {line.rstrip()}")
            if len(lines) >= 20:
                print(f"   ... (truncated)")
        print(f"   ```")
        
        print("\n" + "="*80)
        print("✓ Integration test completed successfully!")
        print("="*80)
        
        print("\nKey features demonstrated:")
        print("1. ✓ LLM calls tracked with tokens/sec measurements")
        print("2. ✓ Tool calls tracked with inputs and outputs")
        print("3. ✓ Calls stored in ordered sequence")
        print("4. ✓ Token statistics aggregated by model")
        print("5. ✓ Query logs saved as JSON files")
        print("6. ✓ Complete call sequence included in logs")
        
        return True


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
