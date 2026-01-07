import json
import logging
import markdown
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def publish_to_litterbox(file_path: Path) -> Optional[str]:
    """Publish a file to Litterbox and return the URL."""
    url = "https://litterbox.catbox.moe/resources/internals/api.php"
    data = {'reqtype': 'fileupload', 'time': '72h'}
    try:
        with open(file_path, "rb") as f:
            files = {'fileToUpload': (file_path.name, f, 'text/html')}
            response = requests.post(url, data=data, files=files, timeout=30)
            return response.text.strip()
    except Exception as e:
        logger.error(f"Failed to publish to Litterbox: {e}")
        return None

def generate_html_report(query: str, scratch_dir: Path = Path("scratch")) -> Optional[Path]:
    """Generate an interactive HTML report from research artifacts."""
    subq_file = scratch_dir / "subquestions.jsonl"
    subans_file = scratch_dir / "subanswers.jsonl"
    final_file = scratch_dir / "final_answer.txt"
    report_file = scratch_dir / "report.html"

    if not subq_file.exists() or not final_file.exists():
        logger.error("Missing artifacts for report generation")
        return None

    # Read sub-questions
    subquestions = []
    with open(subq_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): subquestions.append(json.loads(line))
    subquestions.sort(key=lambda x: x.get("order", 999))

    # Read sub-answers
    subanswers = {}
    if subans_file.exists():
        with open(subans_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        order = data.get("order")
                        if order is not None:
                            subanswers[order] = {
                                "answer": markdown.markdown(data.get("answer", ""), extensions=['tables', 'fenced_code']),
                                "urls": data.get("urls", [])
                            }
                    except: pass

    # Read final synthesized answer
    with open(final_file, "r", encoding="utf-8") as f: 
        final_md = f.read()
    final_html = markdown.markdown(final_md, extensions=['tables', 'fenced_code'])

    # Build plan items HTML
    plan_items_html = ""
    for item in subquestions:
        order = item.get('order')
        ans = subanswers.get(order)
        css = "plan-item has-answer" if ans else "plan-item"
        item_html = f'<div class="{css}" onclick="toggleAnswer({order})">'
        item_html += f'<span class="plan-rank">{order}</span> <span class="plan-q">{item.get("question")}</span> <span class="toggle-hint">[Expand]</span>'
        item_html += f'<span class="plan-reason">{item.get("reasoning", "")}</span>'
        if ans:
            item_html += f'<div id="answer-{order}" class="subq-answer-container"><div>{ans["answer"]}</div>'
            if ans["urls"]:
                item_html += '<div class="sources"><strong>Sources:</strong><ul>'
                for u in ans["urls"]: item_html += f'<li><a href="{u}" target="_blank">{u}</a></li>'
                item_html += '</ul></div>'
            item_html += '</div>'
        item_html += '</div>'
        plan_items_html += item_html

    # Full HTML Template
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_template = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Research Report</title><style>
body { font-family: sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; background: #f9f9f9; }
.container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.plan { background: #f0f7ff; padding: 20px; border-radius: 6px; border-left: 4px solid #3498db; margin-bottom: 40px; }
.plan-item { margin-bottom: 15px; padding: 10px; border-radius: 4px; cursor: pointer; border: 1px solid transparent; }
.plan-item:hover { background: #e1f0fa; border-color: #bcdff1; }
.subq-answer-container { display: none; margin-top: 15px; padding: 15px; background: #fff; border: 1px solid #eee; }
.subq-answer-container.expanded { display: block; }
.sources { font-size: 0.8em; margin-top: 10px; border-top: 1px solid #eee; }
.plan-rank { display: inline-block; width: 24px; height: 24px; background: #3498db; color: white; text-align: center; border-radius: 50%; font-weight: bold; margin-right: 10px; }
.toggle-hint { float: right; color: #3498db; font-size: 0.8em; }
</style></head>
<body><div class="container">
<h1>Research Report</h1>
<p class="meta"><strong>Query:</strong> {{QUERY}}<br><strong>Generated:</strong> {{TIMESTAMP}}</p>
<div class="plan"><h2>Research Plan</h2>{{PLAN_ITEMS}}</div>
<div class="report-content"><h2>Final Synthesis</h2>{{FINAL_HTML}}</div>
</div>
<script>
function toggleAnswer(order) {
  const el = document.getElementById('answer-' + order);
  if (el) el.classList.toggle('expanded');
}
</script></body></html>"""

    final_html_page = html_template.replace("{{QUERY}}", query)
    final_html_page = final_html_page.replace("{{TIMESTAMP}}", timestamp)
    final_html_page = final_html_page.replace("{{PLAN_ITEMS}}", plan_items_html)
    final_html_page = final_html_page.replace("{{FINAL_HTML}}", final_html)
    
    with open(report_file, "w", encoding="utf-8") as f: 
        f.write(final_html_page)
    
    return report_file