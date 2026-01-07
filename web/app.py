import sys
import asyncio
import uuid
import json
import logging
import html
import markdown
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
import os
from agent import Agent
from tools.research_agent import ResearchAgent
from logging_config import setup_logging

# Initialize logging
setup_logging()

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app = FastAPI()
app.add_middleware(NoCacheMiddleware)

templates = Jinja2Templates(directory=str(project_root / "web" / "templates"))

raw_key = os.getenv("OPENROUTER_API_KEY")
key_snippet = f"{raw_key[:4]}...{raw_key[-4:]}" if raw_key else "MISSING"

# Initialize both agents
agent_standard = Agent(raw_key)
agent_research = ResearchAgent(raw_key)

# Log configuration at startup
logger = logging.getLogger("web.app")
logger.info(f"Web App started. Standard model: {agent_standard.model}, Key: {key_snippet}")

# Simple in-memory storage for queries
query_storage = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(textbox: str = Form(...), agent_type: str = Form("standard") ):
    session_id = str(uuid.uuid4())
    query_storage[session_id] = {
        "text": textbox,
        "type": agent_type
    }
    # Return trigger for SSE connection
    return HTMLResponse(content=f'<div hx-ext="sse" sse-connect="/stream/{session_id}" sse-swap="message,final_report"></div>')

@app.get("/stream/{session_id}")
async def stream(session_id: str):
    main_loop = asyncio.get_running_loop()

    async def event_generator():
        data = query_storage.pop(session_id, None)
        if not data:
            yield "event: close\ndata: \n\n"
            return

        query_text = data["text"]
        agent_type = data["type"]
        target_agent = agent_research if agent_type == "research" else agent_standard
        agent_name = "Research Agent" if agent_type == "research" else "BetaBro"

        try:
            # Initial status
            status_html = f"<div id='result' hx-swap-oob='true'><div class='thinking'><div class='spinner'></div> {agent_name} is starting...</div></div>"
            yield f"data: {status_html}\n\n"
            
            update_queue = asyncio.Queue()
            def iteration_callback(update):
                main_loop.call_soon_threadsafe(update_queue.put_nowait, update)

            loop = asyncio.get_running_loop()
            agent_task = loop.run_in_executor(None, target_agent.run, query_text, 30, True, iteration_callback)
            
            while not agent_task.done() or not update_queue.empty():
                try:
                    update = await asyncio.wait_for(update_queue.get(), timeout=0.1)
                    if isinstance(update, dict):
                        if update.get("type") == "plan":
                            # Send the research plan as OOB swap with answer placeholders
                            html_items = ""
                            for q in update['questions']:
                                o = q["order"]
                                q_text = q["question"]
                                html_items += f'<li class="plan-item" data-order="{o}" onclick="toggleSubAnswer({o})">'
                                html_items += f'<span class="rank">#{o}</span> {q_text}'
                                html_items += f'<div id="subq-answer-{o}" class="subq-answer rendered-markdown" data-state="collapsed" style="display: none;"></div>'
                                html_items += '</li>'
                            
                            plan_html = f'<ul id="plan-list" hx-swap-oob="true">{html_items}</ul>'
                            yield f"data: {plan_html.replace(chr(10), ' ')}\n\n"
                        elif update.get("type") == "sub_answer":
                            # Send sub-answer as OOB swap, rendered server-side
                            q_order = update.get("order")
                            ans_md = update.get("answer")
                            urls = update.get("urls", [])
                            
                            ans_html_content = markdown.markdown(ans_md, extensions=['tables', 'fenced_code'])
                            
                            # Add URLs list at the BOTTOM
                            url_html = '<div class="sources" style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px;">'
                            if urls:
                                url_html += f'<strong>Cited Sources ({len(urls)}):</strong><ul>'
                                for u in urls:
                                    url_html += f'<li><a href="{u}" target="_blank">{u}</a></li>'
                                url_html += '</ul>'
                            else:
                                url_html += '<em>No external sources cited.</em>'
                            url_html += '</div>'
                            
                            # Reorder: Answer first, then Sources
                            final_card_content = ans_html_content + url_html
                                
                            ans_html = f'<div id="subq-answer-{q_order}" hx-swap-oob="true" class="subq-answer rendered-markdown" data-state="expanded" style="display: block;">{final_card_content}</div>'
                            yield f"data: {ans_html.replace(chr(10), ' ')}\n\n"
                    else:
                        # Send status update as OOB swap
                        if isinstance(update, str):
                            msg = update
                        else:
                            # It's an integer iteration count
                            msg = f"Iteration {int(update)+1}..."
                        
                        # Handle sub-question highlighting via OOB logic
                        highlight_script = ""
                        if isinstance(msg, str) and "[Sub-question" in msg:
                            try:
                                parts = msg.split("[Sub-question ")
                                if len(parts) > 1:
                                    m_val = parts[1].split("/")[0]
                                    if m_val.isdigit():
                                        # Use explicit string comparison in the script
                                        highlight_script = f"<script>document.querySelectorAll('.plan-item').forEach(i => i.classList.toggle('active', String(i.dataset.order) === '{str(m_val)}'));</script>"
                            except Exception as e:
                                logger.error(f"Highlight script error: {e}")

                        update_html = f"<div id='result' hx-swap-oob='true'><div class='thinking'><div class='spinner'></div> {str(msg)}</div>{highlight_script}</div>"
                        sse_data = str(update_html).replace("\n", " ")
                        yield f"data: {sse_data}\n\n"
                except asyncio.TimeoutError:
                    continue
                except Exception as loop_e:
                    logger.error(f"Error in update loop: {loop_e}")
                    logger.error(traceback.format_exc())
                    continue

            response = await agent_task
            # Render final report server-side
            final_html_content = markdown.markdown(str(response), extensions=['tables', 'fenced_code'])
            final_html = f'<div id="result" hx-swap-oob="true" class="rendered-markdown">{final_html_content}</div>'
            
            # SSE multi-line fix: every newline must be followed by "data: "
            sse_data = final_html.replace("\n", "\ndata: ")
            yield f"data: {sse_data}\n\n"
            
            # Send an explicit cleanup script OOB
            cleanup_script = '<div id="cleanup" hx-swap-oob="true"><script>document.querySelectorAll(".subq-answer").forEach(el => { el.style.display = "none"; el.setAttribute("data-state", "collapsed"); }); document.querySelectorAll(".plan-item").forEach(el => el.classList.remove("active"));</script></div>'
            yield f"data: {cleanup_script}\n\n"
            
            # Close event
            yield "event: final_report\ndata: done\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_html = f"<div id='result' hx-swap-oob='true' style='color: red;'>Error: {str(e)}</div>"
            yield f"data: {error_html}\n\n"
        finally:
            yield "event: close\ndata: \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[str(project_root)])