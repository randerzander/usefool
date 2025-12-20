#!/usr/bin/env python3
"""
Test script for the code generation tool.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.code import write_code, explain_code, fix_code


def test_write_code():
    """Test generating code from a task description."""
    print("\n" + "="*60)
    print("Test 1: Generate Code")
    print("="*60 + "\n")
    
    task = "Create a function that calculates the fibonacci sequence up to n terms"
    print(f"Task: {task}\n")
    
    code = write_code(task)
    print("Generated Code:")
    print("-" * 60)
    print(code)
    print("\n")


def test_explain_code():
    """Test explaining existing code."""
    print("\n" + "="*60)
    print("Test 2: Explain Code")
    print("="*60 + "\n")
    
    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    
    print("Code to explain:")
    print("-" * 60)
    print(code)
    print("\n")
    
    explanation = explain_code(code)
    print("Explanation:")
    print("-" * 60)
    print(explanation)
    print("\n")


def test_fix_code():
    """Test fixing buggy code."""
    print("\n" + "="*60)
    print("Test 3: Fix Code")
    print("="*60 + "\n")
    
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""
    
    error = "ZeroDivisionError when the list is empty"
    
    print("Buggy Code:")
    print("-" * 60)
    print(buggy_code)
    print(f"\nError: {error}\n")
    
    fixed = fix_code(buggy_code, error)
    print("Fixed Code:")
    print("-" * 60)
    print(fixed)
    print("\n")


def test_run_code():
    """Test running code in Docker container."""
    print("\n" + "="*60)
    print("Test 4: Run Code in Docker")
    print("="*60 + "\n")
    
    test_code = """
print("Hello from Docker container!")
print(f"2 + 2 = {2 + 2}")

# Test imports
import sys
print(f"Python version: {sys.version}")
"""
    
    print("Code to run:")
    print("-" * 60)
    print(test_code)
    print("\n")
    
    from tools.code import run_code
    result = run_code(test_code)
    
    print("Execution Result:")
    print("-" * 60)
    print(f"Success: {result['success']}")
    print(f"Exit Code: {result['exit_code']}")
    print(f"Filename: {result['filename']}")
    
    if result['stdout']:
        print("\nStdout:")
        print(result['stdout'])
    
    if result['stderr']:
        print("\nStderr:")
        print(result['stderr'])
    
    print("\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Code Generation Tool Tests")
    print("="*60)
    
    # Run all tests
    test_write_code()
    test_explain_code()
    test_fix_code()
    test_run_code()
    
    print("="*60)
    print("All tests completed!")
    print("="*60 + "\n")
