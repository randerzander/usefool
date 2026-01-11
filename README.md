# usefool

A lean AI agent that can search the web and scrape URLs. Optional Discord bot wrapper included.

## Quick Start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Set your OpenRouter API key
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

3) Run
```bash
python agent.py    # demo
```

## Discord Bot

Requirements:
- `.bot_token` file with your Discord bot token
- `OPENROUTER_API_KEY` environment variable

Run:
```bash
python discord_bot.py
```

Configuration (`config.yaml`):
```yaml
auto_restart: true  # auto-restart on file changes
base_url: "https://openrouter.ai/api/v1/chat/completions"
default_model: "amazon/nova-2-lite-v1:free"
image_caption_model: "nvidia/nemotron-nano-12b-v2-vl:free"
```

Features:
- Auto-restart on code changes (configurable)
- Reply chain context support
- Image captioning with vision models
- Reaction-based eval logging (🧪 to log question, ✅ to mark accepted answer)
- User info persistence (add_userinfo/read_userinfo tools)
- DM support (all DMs are treated as queries)

## Usage in Code

```python
from agent import Agent
import os

agent = Agent(os.getenv("OPENROUTER_API_KEY"))
answer = agent.run("Latest AI news?", verbose=True)
print(answer)
```

## Available Tools

- `web_search(query)` - Search the web via pysearx (uses Brave, Google, Bing, DuckDuckGo, Startpage)
- `read_url(url)` - Read and extract content from URLs (YouTube, Wikipedia, PDFs, web pages)
  - Supports GitHub authentication (create `.github_token` file to avoid rate limits)
- `code(task)` - Generate Python code and immediately execute it in an isolated Docker container
- `read_file(filepath)` - Read files from current directory (1MB limit, path traversal protected)
- `add_userinfo(username, info)` - Store user information for future reference
- `read_userinfo(username)` - Recall stored user information (case-insensitive)

## Optional Configuration

### GitHub Token (for scraping GitHub URLs)
Create a `.github_token` file in the project root with your [GitHub Personal Access Token](https://github.com/settings/tokens):
```bash
echo "ghp_yourTokenHere" > .github_token
```
This prevents rate limiting when scraping GitHub repositories.

### Proxy Whitelist (for improved web search)
Set the `PROXY_FILE` environment variable to point to a proxy whitelist file for pysearx:
```bash
export PROXY_FILE=whitelist_2026-01-11_15.txt
```
Or add it to your `.env` file. This improves search reliability by rotating through validated proxies.

## Requirements

- Python 3.8+
- OpenRouter API key (free tier available)
- Internet connection for search/scrape

## Tests

```bash
python tests/test_user_info.py
```

## License

MIT
