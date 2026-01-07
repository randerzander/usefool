import logging
from pathlib import Path
from .tool_utils import create_tool_spec
from .report_utils import generate_html_report, publish_to_litterbox

logger = logging.getLogger(__name__)

# Tool specification
RESEARCH_TOOL_SPEC = create_tool_spec(
    name="deep_research",
    description="Perform deep research on a complex topic. Returns a TL;DR and the report link.",
    parameters={"query": "The complex research topic or question"},
    required=["query"]
)

def deep_research(query: str) -> str:
    import os
    from .research_agent import ResearchAgent
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    research_agent = ResearchAgent(api_key)
    research_agent.run(query, verbose=False)
    
    # Generate HTML report using utility
    report_file = generate_html_report(query)
    
    if not report_file:
        return "Error: Failed to generate research report artifacts."
    
    # Publish to Litterbox
    litter_url = publish_to_litterbox(report_file)
    
    # Get final synthesized answer for TL;DR
    final_file = Path("scratch/final_answer.txt")
    with open(final_file, "r") as f:
        final_md = f.read()
        
    summary_res = research_agent._call_llm([{"role": "user", "content": f"Summarize in 3 sentences: {final_md}"}], use_tools=False)
    tldr = summary_res["choices"][0]["message"]["content"].strip()
    
    return f"**Summary:** {tldr}\n\n[Full Interactive Report]({litter_url})"