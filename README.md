# scraper

A simple ReAct (Reasoning + Acting) agent that can search the web, scrape URLs, and write code to answer questions.

## Features

- **ReAct Agent**: Uses the ReAct pattern for reasoning and tool selection
- **DuckDuckGo Search**: Search the web for information
- **URL Scraping**: Parse HTML content into readable markdown using [pyreadability](https://github.com/randerzander/pyreadability)
- **Code Writing**: Generate code using AI with automatic multi-file project support
- **OpenRouter Integration**: Uses OpenRouter API with tngtech/deepseek-r1t2-chimera:free model for reasoning and kwaipilot/kat-coder-pro:free for code generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/randerzander/scraper.git
cd scraper
```

2. Install dependencies:

   **Option A: Using uv (recommended - faster installation)**
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Create a virtual environment named "scraper"
   uv venv scraper
   
   # Activate the virtual environment
   source scraper/bin/activate  # On macOS/Linux
   # Or on Windows: scraper\Scripts\activate
   
   # Install dependencies with uv
   uv pip install -r requirements.txt
   ```

   **Option B: Using pip (traditional method)**
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

### Basic Usage

Run the example script:
```bash
python example.py
```

Or run the default demo:
```bash
python react_agent.py
```

### Using in Your Own Code

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

### Using a Custom Code Generation Model

```python
from react_agent import ReActAgent
import os

# Initialize the agent with a custom code generation model
api_key = os.getenv("OPENROUTER_API_KEY")
agent = ReActAgent(
    api_key,
    code_model="anthropic/claude-3.5-sonnet"  # Use a different model for code generation
)

# Ask the agent to write code
answer = agent.run("Write a Python function to calculate fibonacci numbers", verbose=True)
print(answer)
```

### Discord Bot

Run the agent as a Discord bot that responds to mentions:

```bash
python discord_bot.py
```

**Requirements:**
1. Create a `token.txt` file in the current directory with your Discord bot token
2. Set the `OPENROUTER_API_KEY` environment variable

**How to get a Discord bot token:**
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application (or select an existing one)
3. Go to the "Bot" section and click "Reset Token"
4. Save the token to `token.txt` in the project directory
5. Enable "Message Content Intent" in the Bot settings
6. Use the OAuth2 URL Generator to create an invite link with:
   - Scopes: `bot`
   - Bot Permissions: `Read Messages/View Channels`, `Send Messages`, `Read Message History`

**Usage:**
- Mention the bot in a Discord message followed by your question
- Example: `@YourBot What is the latest news about artificial intelligence?`
- The bot will use the ReAct agent to search and scrape information to answer your question

### Testing

Run the test suite to verify the implementation:
```bash
python test_react_agent.py
```

This will test the agent's parsing logic, tool execution, and reasoning loop without requiring API access.

### Testing uv Virtual Environment Setup

To verify that the uv virtual environment setup works correctly:
```bash
python test_uv_venv.py
```

This test script will:
1. Verify uv is installed
2. Create a virtual environment named "scraper"
3. Install dependencies using uv pip
4. Run the test suite in the virtual environment
5. Clean up the test environment

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

- `write_code`: Write code using AI code generation
  - Input: A detailed description of the code to write
  - Output: Generated code. If multiple files are detected, they are saved to a timestamped directory (e.g., `generated_code_20231208_153045`)
  - Model: Uses `kwaipilot/kat-coder-pro:free` by default (can be customized via `code_model` parameter when initializing the agent)

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

### Code Writing Example

```
Question: Write a simple Python Flask web application with routes for home and about pages.

Iteration 1:
Thought: I need to write code for a Flask web application with multiple routes.
Action: write_code
Action Input: Create a Python Flask web application with a home route (/) that returns "Welcome to my app!" and an about route (/about) that returns "About this app". Include necessary imports and if __name__ == '__main__' block to run the app.
Observation: Code saved to directory 'generated_code_20231208_153045' with the following files:
  - app.py
  - requirements.txt

Total files: 2

Iteration 2:
Thought: The code has been generated and saved. I can provide the final answer.
Final Answer: I've created a Flask web application with home and about routes. The code has been saved to the 'generated_code_20231208_153045' directory with two files:
- app.py: Contains the Flask application with routes
- requirements.txt: Lists the Flask dependency
You can run the app by installing dependencies (pip install -r requirements.txt) and running python app.py.
```

## Repository Structure

- `react_agent.py` - Main implementation of the ReAct agent
- `example.py` - Example usage with multiple scenarios
- `discord_bot.py` - Discord bot wrapper for the ReAct agent
- `test_react_agent.py` - Test suite for the agent
- `test_code_writing.py` - Test suite for the code writing tool
- `test_uv_venv.py` - Test script for uv virtual environment setup
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment configuration
- `README.md` - This file

## Requirements

- Python 3.8+
- OpenRouter API key (free tier available)
- Internet connection for web search and URL scraping

## Implementation Details

The ReAct agent uses a simple but effective approach:

1. **Prompt Engineering**: Uses clear instructions to guide the LLM through the ReAct pattern
2. **Regex Parsing**: Extracts structured information from LLM responses
3. **Tool Integration**: Seamlessly integrates DuckDuckGo search and URL scraping
4. **Iteration Control**: Prevents infinite loops with configurable max iterations
5. **Error Handling**: Gracefully handles network errors and parsing failures

The agent does not use any heavyweight frameworks like LangChain or LlamaIndex, making it:
- Easy to understand and modify
- Minimal dependencies
- Transparent reasoning process
- Fully customizable

## License

MIT