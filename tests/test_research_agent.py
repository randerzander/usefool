#!/usr/bin/env python3
"""
Test script for real ResearchAgent functionality.
Requires OPENROUTER_API_KEY environment variable.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.research_agent import ResearchAgent

def test_real_research_workflow():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    print("🚀 Initializing Real ResearchAgent...")
    # Use default model from config.yaml
    agent = ResearchAgent(api_key)
    
    # Complex query to trigger decomposition
    query = "Who is the current CEO of Nvidia, what is their background, and what were Nvidia's major accomplishments in 2024?"
    
    print(f"\n🧐 Starting Research for: \"{query}\"")
    print("="*60)
    
    # Run the research workflow
    # Using verbose=True so you can see the internal agent iterations
    result = agent.run(query, verbose=True)
    
    print("\n" + "="*60)
    print("📝 FINAL SYNTHESIZED ANSWER:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    # Verify file logs
    scratch_dir = Path("scratch")
    subq_file = scratch_dir / "subquestions.jsonl"
    
    print("\n📂 Checking Logs in scratch/...")
    if subq_file.exists():
        print(f"✅ Found {subq_file}")
        with open(subq_file, "r") as f:
            lines = f.readlines()
            print(f"   -> Log contains {len(lines)} entries (includes historical runs).")
    else:
        print(f"❌ {subq_file} NOT FOUND")

    # Check for individual activity logs
    activity_logs = list(scratch_dir.glob("subq_*.jsonl"))
    if activity_logs:
        print(f"✅ Found {len(activity_logs)} sub-question activity logs.")
        for log in sorted(activity_logs):
            print(f"   -> {log.name}")
    else:
        print("❓ No subq_*.jsonl logs found. (This is normal if the agent didn't use web_search/read_url for this specific run)")

if __name__ == "__main__":
    try:
        test_real_research_workflow()
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()