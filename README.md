# scraper

A simple ReAct (Reasoning + Acting) agent that can search the web and scrape URLs to answer questions.

## Features

- **ReAct Agent**: Uses the ReAct pattern for reasoning and tool selection
- **DuckDuckGo Search**: Search the web for information
- **URL Scraping**: Parse HTML content into readable markdown using [pyreadability](https://github.com/randerzander/pyreadability)
- **OpenRouter Integration**: Uses OpenRouter API with tngtech/deepseek-r1t2-chimera:free model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/randerzander/scraper.git
cd scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Or create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

Run the agent:
```bash
python react_agent.py
```

Or use it in your own code:
```python
from react_agent import ReActAgent
import os

# Initialize the agent
api_key = os.getenv("OPENROUTER_API_KEY")
agent = ReActAgent(api_key)

# Ask a question
answer = agent.run("What is the latest news about artificial intelligence?", verbose=True)
print(answer)
```

## How It Works

The ReAct agent follows a thought-action-observation loop:

1. **Thought**: The agent reasons about what information it needs
2. **Action**: The agent selects a tool to use (search or scrape)
3. **Action Input**: The agent provides input for the selected tool
4. **Observation**: The agent receives results from the tool
5. Repeat until the agent has enough information to provide a **Final Answer**

## Available Tools

- `duckduckgo_search`: Search the web using DuckDuckGo
  - Input: A search query string
  - Output: List of search results with titles, URLs, and snippets

- `scrape_url`: Scrape and parse HTML content from a URL
  - Input: A URL to scrape
  - Output: Markdown-formatted content extracted from the page

## Example

```
Question: What is the latest news about artificial intelligence?

Iteration 1:
Thought: I need to search for the latest news about artificial intelligence.
Action: duckduckgo_search
Action Input: latest artificial intelligence news
Observation: [Search results with recent AI news articles]

Iteration 2:
Thought: Based on the search results, I can see recent developments in AI...
Final Answer: [Comprehensive answer based on search results]
```

## License

MIT