#!/usr/bin/env python3
"""
Standalone test for write_code and run_code tools.
Checks if the coding model endpoint is working and if Docker execution is functional.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path so we can import tools
sys.path.append(str(Path(__file__).parent.parent))

from tools.code import write_code, run_code

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_code_flow():
    print("\n--- Testing Code Flow ---")
    
    # 1. Test write_code
    task = "Write a simple python script that prints 'Hello from the sandbox!' and calculates 2+2."
    print(f"\n1. Calling write_code with task: {task}")
    
    filename = write_code(task=task, filename="test_standalone.py")
    
    if filename.startswith("Error"):
        print(f"❌ write_code FAILED: {filename}")
        return
    
    print(f"✅ write_code SUCCESS: Saved to {filename}")
    
    # Verify file existence and content
    scratch_path = Path("scratch") / filename
    if scratch_path.exists():
        with open(scratch_path, 'r') as f:
            content = f.read()
        print(f"--- File Content ---\n{content}\n--------------------")
    else:
        print(f"❌ Error: {scratch_path} does not exist!")
        return

    # 2. Test run_code
    print(f"\n2. Calling run_code for: {filename}")
    result = run_code(filename=filename)
    
    if result["success"]:
        print(f"✅ run_code SUCCESS!")
        print(f"--- STDOUT ---\n{result['stdout']}\n--------------")
    else:
        print(f"❌ run_code FAILED!")
        print(f"--- STDERR ---\n{result['stderr']}\n--------------")
        if result.get("stdout"):
            print(f"--- STDOUT ---\n{result['stdout']}\n--------------")

if __name__ == "__main__":
    # Ensure scratch directory exists
    Path("scratch").mkdir(exist_ok=True)
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️ Warning: OPENROUTER_API_KEY not found in environment.")
    
    try:
        test_code_flow()
    except Exception as e:
        print(f"❌ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
