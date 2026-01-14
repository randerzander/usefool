#!/usr/bin/env python3
"""
Standalone script to test the agent with a single query.

Usage:
    python test_agent.py "What is the current date?"
    python test_agent.py "Who is the president?" --verbose
    python test_agent.py "Search for Python tutorials" --exclude web_search
"""

import argparse
import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from agent import Agent
from utils import is_localhost

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Setup query logs directory
DATA_DIR = Path(__file__).parent / "data"
QUERY_LOGS_DIR = DATA_DIR / "query_logs"
QUERY_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Test the agent with a single query')
    parser.add_argument('query', help='The question or task to give the agent')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')
    parser.add_argument('--exclude', nargs='+', help='Tools to exclude (e.g., web_search read_url)')
    parser.add_argument('--max-iterations', type=int, default=30, help='Max iterations (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize agent
    api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    base_url = CONFIG.get("base_url", "http://localhost:8080/v1")
    
    # Pass model=None for localhost to trigger auto-detection
    model = None if is_localhost(base_url) else CONFIG.get("default_model", "gpt-3.5-turbo")
    
    agent = Agent(api_key=api_key, model=model, base_url=base_url)
    
    print(f"🤖 Agent initialized")
    print(f"   Model: {agent.model}")
    print(f"   API: {base_url}")
    print(f"   Query: {args.query}")
    if args.exclude:
        print(f"   Excluding tools: {', '.join(args.exclude)}")
    print(f"\n{'='*60}")
    
    # Run the agent
    try:
        response = agent.run(
            args.query,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            exclude_tools=args.exclude
        )
        
        print(f"\n{'='*60}")
        print(f"✅ RESPONSE:")
        print(response)
        print(f"{'='*60}")
        
        # Show tracking stats
        tracking = agent.get_tracking_data()
        call_sequence = tracking.get("call_sequence", [])
        token_stats = tracking.get("token_stats", {})
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total calls: {len(call_sequence)}")
        
        # Count tool calls
        tool_calls = [c for c in call_sequence if c.get("type") == "tool_call"]
        if tool_calls:
            tool_counts = {}
            for call in tool_calls:
                tool = call.get("tool", "unknown")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            print(f"   Tool calls: {sum(tool_counts.values())}")
            for tool, count in sorted(tool_counts.items()):
                print(f"      {tool}: {count}")
        
        # Token stats
        if token_stats:
            total_input = sum(stats.get("input_tokens", 0) for stats in token_stats.values())
            total_output = sum(stats.get("output_tokens", 0) for stats in token_stats.values())
            print(f"   Tokens: {total_input} in + {total_output} out = {total_input + total_output} total")
        
        print(f"{'='*60}")
        
        # Save execution trace to log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = QUERY_LOGS_DIR / f"test_{timestamp}.json"
        
        log_data = {
            "username": "test_agent",
            "timestamp": datetime.now().isoformat(),
            "user_query": args.query,
            "final_response": response,
            "excluded_tools": args.exclude or [],
            "max_iterations": args.max_iterations,
            "call_sequence": tracking.get("call_sequence", []),
            "token_stats": tracking.get("token_stats", {})
        }
        
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n💾 Execution trace saved to: {log_filename.name}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*60}")
        sys.exit(1)

if __name__ == "__main__":
    main()
