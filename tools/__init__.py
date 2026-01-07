"""
Tools package for the scraper project.
Centralized tool registry for the Agent.
"""

from tools.web_search import web_search, TOOL_SPEC as WEB_SEARCH_SPEC
from tools.read_url import read_url, TOOL_SPEC as READ_URL_SPEC
from tools.code import write_code, run_code, WRITE_CODE_SPEC, RUN_CODE_SPEC
from tools.research_tool import deep_research, RESEARCH_TOOL_SPEC as DEEP_RESEARCH_SPEC

# Registry of tool functions
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_url": read_url,
    "write_code": write_code,
    "run_code": run_code,
    "deep_research": deep_research
}

# Registry of tool specifications (OpenAI format)
TOOL_SPECS = [
    WEB_SEARCH_SPEC,
    READ_URL_SPEC,
    WRITE_CODE_SPEC,
    RUN_CODE_SPEC,
    DEEP_RESEARCH_SPEC
]

def get_tool_functions():
    return TOOL_FUNCTIONS.copy()

def get_tool_specs():
    return TOOL_SPECS.copy()