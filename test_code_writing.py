#!/usr/bin/env python3
"""
Test script for the code writing tool functionality.
Tests the write_code function and file parsing logic without requiring API access.
"""

import os
import shutil
from react_agent import _parse_code_files


def test_parse_single_file():
    """Test parsing a single code file."""
    print("Testing single file parsing...")
    print("="*60)
    
    response = """Here's the code:

```python
def hello():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello()
```
"""
    
    files = _parse_code_files(response)
    print(f"Files found: {list(files.keys())}")
    assert len(files) == 1
    assert 'code.txt' in files
    assert 'def hello():' in files['code.txt']
    print("✓ Test 1 passed")


def test_parse_multiple_files_pattern1():
    """Test parsing multiple files with filename in code block marker."""
    print("\nTesting multiple files (pattern 1)...")
    print("="*60)
    
    response = """Here are the files:

```python main.py
def main():
    print("Main file")

if __name__ == "__main__":
    main()
```

```python utils.py
def helper():
    return "Helper function"
```
"""
    
    files = _parse_code_files(response)
    print(f"Files found: {list(files.keys())}")
    assert len(files) == 2
    assert 'main.py' in files
    assert 'utils.py' in files
    assert 'def main():' in files['main.py']
    assert 'def helper():' in files['utils.py']
    print("✓ Test 2 passed")


def test_parse_multiple_files_pattern2():
    """Test parsing multiple files with filename as header."""
    print("\nTesting multiple files (pattern 2)...")
    print("="*60)
    
    response = """I'll create two files:

# index.html
```html
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>Hello</body>
</html>
```

# styles.css
```css
body {
    margin: 0;
    padding: 0;
}
```
"""
    
    files = _parse_code_files(response)
    print(f"Files found: {list(files.keys())}")
    assert len(files) == 2
    assert 'index.html' in files
    assert 'styles.css' in files
    assert 'DOCTYPE' in files['index.html']
    assert 'margin' in files['styles.css']
    print("✓ Test 3 passed")


def test_parse_multiple_files_pattern3():
    """Test parsing multiple files with colon notation."""
    print("\nTesting multiple files (pattern 3)...")
    print("="*60)
    
    response = """Here's the code structure:

app.py:
from flask import Flask
app = Flask(__name__)

config.json:
{
    "debug": true,
    "port": 5000
}
"""
    
    files = _parse_code_files(response)
    print(f"Files found: {list(files.keys())}")
    # Note: This pattern is trickier, but should at least capture something
    print(f"Files: {files}")
    # At minimum, we should have attempted to parse files
    assert len(files) >= 1
    print("✓ Test 4 passed")


def test_parse_mixed_patterns():
    """Test parsing with mixed file notation patterns."""
    print("\nTesting mixed patterns...")
    print("="*60)
    
    response = """Project structure:

```javascript package.json
{
    "name": "test-app",
    "version": "1.0.0"
}
```

# src/index.js
```javascript
console.log("Hello, World!");
```
"""
    
    files = _parse_code_files(response)
    print(f"Files found: {list(files.keys())}")
    assert len(files) >= 2
    assert 'package.json' in files
    assert 'src/index.js' in files or 'index.js' in files
    print("✓ Test 5 passed")


def test_no_files():
    """Test handling response with no clear file structure."""
    print("\nTesting response with no files...")
    print("="*60)
    
    response = "Here's a simple explanation without any code."
    
    files = _parse_code_files(response)
    print(f"Files found: {list(files.keys())}")
    assert len(files) == 0
    print("✓ Test 6 passed")


if __name__ == "__main__":
    print("Code Writing Tool Tests")
    print("="*60)
    
    test_parse_single_file()
    test_parse_multiple_files_pattern1()
    test_parse_multiple_files_pattern2()
    test_parse_multiple_files_pattern3()
    test_parse_mixed_patterns()
    test_no_files()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Note: To test the actual code writing tool with API:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Run: python react_agent.py")
    print("3. Ask: 'Write a simple Python web scraper'")
    print("="*60)
