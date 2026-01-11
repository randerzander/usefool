#!/bin/bash
# Quick runner for research query test

cd "$(dirname "$0")"

echo "=================================="
echo "Research Query Test Runner"
echo "=================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY not set!"
    echo "Run: export OPENROUTER_API_KEY=your_key_here"
    exit 1
fi

echo "✓ Virtual environment found"
echo "✓ API key is set"
echo ""

# Activate and run
source venv/bin/activate
python test_research_query.py

exit $?
