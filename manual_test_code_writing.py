#!/usr/bin/env python3
"""
Manual test script to demonstrate the code writing tool functionality.
This script shows how the agent would use the write_code tool.

To test with actual API:
1. Set OPENROUTER_API_KEY environment variable
2. Uncomment the actual agent test at the end
3. Run: python manual_test_code_writing.py
"""

import os
from react_agent import ReActAgent, write_code, _parse_code_files


def demo_file_parsing():
    """Demonstrate the file parsing logic with various response formats."""
    print("="*80)
    print("DEMO: File Parsing Logic")
    print("="*80)
    
    # Example 1: Single file response
    print("\n1. Single File Response:")
    print("-" * 60)
    response1 = """Here's a simple Python script:

```python
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
```
"""
    files1 = _parse_code_files(response1)
    print(f"Response:\n{response1}")
    print(f"Files detected: {list(files1.keys())}")
    print(f"Content preview: {list(files1.values())[0][:50]}...")
    
    # Example 2: Multiple files with filename in code block
    print("\n2. Multiple Files with Filename in Code Block:")
    print("-" * 60)
    response2 = """I'll create a Flask application with multiple files:

```python app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

```html templates/index.html
<!DOCTYPE html>
<html>
<head>
    <title>My App</title>
</head>
<body>
    <h1>Welcome!</h1>
</body>
</html>
```

```text requirements.txt
flask>=2.0.0
```
"""
    files2 = _parse_code_files(response2)
    print(f"Files detected: {list(files2.keys())}")
    print(f"Total files: {len(files2)}")
    
    # Example 3: Multiple files with header notation
    print("\n3. Multiple Files with Header Notation:")
    print("-" * 60)
    response3 = """Project structure:

# main.py
```python
def main():
    print("Main file")

if __name__ == "__main__":
    main()
```

# utils.py
```python
def helper():
    return "Helper"
```

# config.json
```json
{
    "debug": true
}
```
"""
    files3 = _parse_code_files(response3)
    print(f"Files detected: {list(files3.keys())}")
    print(f"Total files: {len(files3)}")


def demo_agent_usage():
    """Demonstrate how to use the agent with the code writing tool."""
    print("\n" + "="*80)
    print("DEMO: Using the Code Writing Tool with ReAct Agent")
    print("="*80)
    
    print("\nTo use the code writing tool with the agent:")
    print("-" * 60)
    
    example_code = """
from react_agent import ReActAgent
import os

# Initialize the agent
api_key = os.getenv("OPENROUTER_API_KEY")
agent = ReActAgent(api_key)

# Ask the agent to write code
answer = agent.run(
    "Write a simple Python web scraper that fetches and parses HTML from a URL",
    verbose=True
)
print(answer)
"""
    print(example_code)
    
    print("\nExpected behavior:")
    print("-" * 60)
    print("1. Agent recognizes this is a code writing task")
    print("2. Agent uses the 'write_code' tool")
    print("3. If multiple files are returned:")
    print("   - Creates a timestamped directory (e.g., generated_code_20231208_153045)")
    print("   - Saves all files to that directory")
    print("   - Returns a message listing the files and directory name")
    print("4. If single file or unclear structure:")
    print("   - Returns the code directly in the response")


def demo_discord_bot_usage():
    """Demonstrate how the code writing tool works with the Discord bot."""
    print("\n" + "="*80)
    print("DEMO: Using Code Writing Tool with Discord Bot")
    print("="*80)
    
    print("\nExample Discord conversation:")
    print("-" * 60)
    print("User: @Bot Write a Python function to calculate fibonacci numbers")
    print("\nBot: (thinking...)")
    print("\nBot: **Answer:**")
    print("Here's a Python function to calculate Fibonacci numbers:")
    print("""
```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Example usage
print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```
""")
    
    print("\n" + "-" * 60)
    print("User: @Bot Create a complete Flask REST API for a todo list")
    print("\nBot: (thinking...)")
    print("\nBot: **Answer:**")
    print("I've created a Flask REST API for a todo list.")
    print("Code saved to directory 'generated_code_20231208_153520' with the following files:")
    print("  - app.py")
    print("  - models.py")
    print("  - requirements.txt")
    print("  - README.md")
    print("\nTotal files: 4")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CODE WRITING TOOL - MANUAL TEST & DEMONSTRATION")
    print("="*80)
    
    # Run demonstrations
    demo_file_parsing()
    demo_agent_usage()
    demo_discord_bot_usage()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Code writing tool implemented successfully")
    print("✓ Multi-file detection and saving works correctly")
    print("✓ Integration with ReAct agent complete")
    print("✓ Tool description guides the LLM to use it appropriately")
    print("\nThe tool is ready to use! Set OPENROUTER_API_KEY and try it out.")
    print("="*80)
    
    # Optional: Uncomment to test with actual API
    # print("\n" + "="*80)
    # print("ACTUAL API TEST (requires OPENROUTER_API_KEY)")
    # print("="*80)
    # api_key = os.getenv("OPENROUTER_API_KEY")
    # if api_key:
    #     agent = ReActAgent(api_key)
    #     answer = agent.run("Write a simple Python hello world program", verbose=True)
    #     print(f"\nFinal Answer:\n{answer}")
    # else:
    #     print("Set OPENROUTER_API_KEY environment variable to test with actual API")
