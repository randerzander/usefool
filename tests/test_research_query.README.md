# Research Query Test Script

Standalone test script that simulates a Discord research query.

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the test
python test_research_query.py
```

## What it does

1. Simulates a Discord user asking: `@UseFool research nv-ingest vs docling`
2. Initializes the agent with your model
3. Runs the full query processing pipeline
4. Displays:
   - Real-time progress with spinners
   - Tool calls and results
   - Final statistics (tokens, duration, tool usage)
   - The complete answer
5. Saves a detailed query log to `data/query_logs/`

## Expected behavior

The agent should:
- Use the `deep_research` tool if available
- OR use `web_search` and `read_url` tools to gather information
- Compare nv-ingest vs docling
- Provide a comprehensive answer

## Output

You'll see:
- Spinner progress: `⠋ LLM Call Attempt 1 | Model: ... | 1234 tokens in | 2.5s`
- Completion logs: `LLM call completed | Status: 200 | Tokens: 1234 in, 56 out | Time: 2.50s | Throughput: 493.6 in/s, 22.4 out/s`
- Tool calls: `Tool Call: web_search | Arguments: {"query": "nv-ingest"}`
- Final answer with statistics

## Troubleshooting

If you get "I couldn't generate a response":
- Check that your model is running (llama.cpp server on localhost:8080)
- Check OPENROUTER_API_KEY is set
- Increase max_iterations in the script if needed
