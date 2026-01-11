"""
Tools package for the scraper project.
Centralized tool registry for the Agent.
"""

import yaml
from pathlib import Path

from tools.web_search import web_search, TOOL_SPEC as WEB_SEARCH_SPEC
from tools.read_url import read_url, TOOL_SPEC as READ_URL_SPEC
from tools.code import code, write_code, run_code, CODE_SPEC, WRITE_CODE_SPEC, RUN_CODE_SPEC
from tools.research_tool import deep_research, RESEARCH_TOOL_SPEC as DEEP_RESEARCH_SPEC

# Load config to check which tools are enabled
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
try:
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

TOOL_CONFIG = CONFIG.get("tools", {})

# Global variable to store detected context length
# Set by Agent during initialization
DETECTED_CONTEXT_LENGTH = 32768  # Default value

# Registry of all available tools
ALL_TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_url": read_url,
    "code": code,
    "write_code": write_code,
    "run_code": run_code,
    "deep_research": deep_research
}

ALL_TOOL_SPECS = {
    "web_search": WEB_SEARCH_SPEC,
    "read_url": READ_URL_SPEC,
    "code": CODE_SPEC,
    "write_code": WRITE_CODE_SPEC,
    "run_code": RUN_CODE_SPEC,
    "deep_research": DEEP_RESEARCH_SPEC
}

def get_tool_functions():
    """Get enabled tool functions based on config."""
    enabled = {}
    for name, func in ALL_TOOL_FUNCTIONS.items():
        # Default to enabled if not specified in config
        if TOOL_CONFIG.get(name, True):
            enabled[name] = func
    return enabled

def get_tool_specs():
    """Get enabled tool specs based on config."""
    enabled = []
    for name, spec in ALL_TOOL_SPECS.items():
        # Default to enabled if not specified in config
        if TOOL_CONFIG.get(name, True):
            enabled.append(spec)
    return enabled

def set_context_length(context_length: int):
    """Set the detected context length for use by tools."""
    global DETECTED_CONTEXT_LENGTH
    DETECTED_CONTEXT_LENGTH = context_length