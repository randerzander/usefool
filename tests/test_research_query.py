#!/usr/bin/env python3
"""
Standalone test script for research queries.
Simulates a Discord bot research query: "@Usefool research nv-ingest vs docling"
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def test_research_query():
    """Test a research query similar to Discord bot usage."""
    
    from agent import Agent
    from utils import MODEL_CONFIG
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set!")
        return False
    
    # Get bot name from config
    bot_name = MODEL_CONFIG.get("bot_name", "UseFool")
    
    # Simulate Discord query
    user_query = "research nv-ingest vs docling"
    user_display_name = "TestUser"
    
    print("=" * 80)
    print(f"SIMULATED DISCORD RESEARCH QUERY TEST")
    print("=" * 80)
    print(f"User: {user_display_name}")
    print(f"Query: @{bot_name} {user_query}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Create agent
    logger.info("Initializing agent...")
    base_url = MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
    agent = Agent(api_key, base_url=base_url, enable_logging=True)
    
    logger.info(f"Model: {agent.model}")
    logger.info(f"API URL: {agent.api_url}")
    
    # Check if deep_research tool is available
    if "deep_research" in agent.tool_functions:
        logger.info("✓ deep_research tool is available")
    else:
        logger.warning("✗ deep_research tool is NOT available - will use regular agent")
    
    # Build context similar to Discord bot
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    question_with_context = f"[You are a Discord bot named {bot_name}]\n\nUser question: {user_query}"
    
    print("\n" + "=" * 80)
    print("STARTING QUERY PROCESSING")
    print("=" * 80)
    print()
    
    # Track start time
    query_start_time = datetime.now()
    
    try:
        # Run the agent
        answer = agent.run(
            question=question_with_context,
            max_iterations=30,
            verbose=True
        )
        
        # Track end time
        query_end_time = datetime.now()
        query_duration = (query_end_time - query_start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("QUERY COMPLETED")
        print("=" * 80)
        print(f"Duration: {query_duration:.2f} seconds")
        print()
        
        # Get tracking data
        tracking = agent.get_tracking_data()
        
        # Count tool usage
        tool_counts = {}
        for entry in tracking["call_sequence"]:
            if entry["type"] == "tool_call":
                tool_name = entry["tool_name"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        # Display statistics
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        
        total_llm_calls = sum(1 for e in tracking["call_sequence"] if e["type"] == "llm_call")
        total_tool_calls = sum(1 for e in tracking["call_sequence"] if e["type"] == "tool_call")
        
        print(f"Total LLM calls: {total_llm_calls}")
        print(f"Total tool calls: {total_tool_calls}")
        print()
        
        if tool_counts:
            print("Tool usage:")
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {tool}: {count}x")
            print()
        
        # Display token stats
        if tracking["token_stats"]:
            print("Token statistics by model:")
            for model, stats in tracking["token_stats"].items():
                print(f"  Model: {model}")
                print(f"    Calls: {stats['total_calls']}")
                print(f"    Input tokens: {stats['total_input_tokens']:,}")
                print(f"    Output tokens: {stats['total_output_tokens']:,}")
                print(f"    Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']:,}")
            print()
        
        # Display answer
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        print()
        
        # Check if we got a valid answer
        if not answer or not answer.strip():
            logger.error("❌ Got empty answer!")
            return False
        
        if answer.strip().lower() in ["none", "i couldn't generate a response.", "maximum iterations reached without a response."]:
            logger.error(f"❌ Got error/fallback response: {answer}")
            return False
        
        logger.info("✅ Test completed successfully!")
        
        # Save query log
        log_dir = Path("data/query_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        log_file = log_dir / f"test_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Always save query log for debugging
        try:
            log_dir = Path("data/query_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            log_file = log_dir / f"test_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            if 'tracking' in locals() and 'answer' in locals():
                with open(log_file, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "user_query": user_query,
                        "final_response": answer if 'answer' in locals() else "ERROR: No answer",
                        "duration_seconds": query_duration if 'query_duration' in locals() else 0,
                        "call_sequence": tracking["call_sequence"] if 'tracking' in locals() else [],
                        "token_stats": tracking["token_stats"] if 'tracking' in locals() else {},
                        "tool_counts": tool_counts if 'tool_counts' in locals() else {}
                    }, f, indent=2)
                
                logger.info(f"Query log saved to: {log_file}")
        except Exception as log_error:
            logger.warning(f"Failed to save query log: {log_error}")


if __name__ == "__main__":
    print("\n")
    success = test_research_query()
    print("\n")
    
    if success:
        print("✅ TEST PASSED")
        sys.exit(0)
    else:
        print("❌ TEST FAILED")
        sys.exit(1)
