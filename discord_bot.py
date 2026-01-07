#!/usr/bin/env python3
"""
Discord bot wrapper for the agent.
"""

import os
import sys
import asyncio
import discord
import requests
import json
import logging
import time
import yaml
import subprocess
import threading
import fcntl
import re
import pty
import termios
import struct
from datetime import datetime
from pathlib import Path
from agent import Agent
from logging_config import setup_logging
from utils import CHARS_PER_TOKEN
from discord_utils import translate_user_mentions, convert_usernames_to_mentions, format_metadata

# Configure logging
console = setup_logging()
logger = logging.getLogger(__name__)

logging.getLogger('discord.client').setLevel(logging.WARNING)
logging.getLogger('discord.gateway').setLevel(logging.WARNING)

def load_config():
    """Load bot configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Config loading failed: {e}, using defaults")
        return {
            "auto_restart": True,
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "image_caption_model": "nvidia/nemotron-nano-12b-v2-vl:free",
            "bot_name": "Usefool"
        }

CONFIG = load_config()
BOT_NAME = CONFIG.get("bot_name", "Usefool")

class DiscordBot:
    """Discord bot that wraps the agent."""
    
    DATA_DIR = Path("data")
    QA_FILE = DATA_DIR / "qa.jsonl"
    QUERY_LOGS_DIR = DATA_DIR / "query_logs"
    USER_INFO_DIR = Path("user_info")
    
    def __init__(self, token: str, api_key: str):
        self.token = token
        self.api_key = api_key
        base_url = CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        self.agent = Agent(api_key, base_url=base_url)
        
        self.DATA_DIR.mkdir(exist_ok=True)
        self.QUERY_LOGS_DIR.mkdir(exist_ok=True)
        
        self.current_query_log = []
        self.current_query_token_stats = {}
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        self.client = discord.Client(intents=intents)
        
        self._register_events()

    def _register_events(self):
        @self.client.event
        async def on_ready():
            logger.info(f"Bot logged in as {self.client.user} and ready!")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user: return
            
            is_query = isinstance(message.channel, discord.DMChannel) or self.client.user.mentioned_in(message)
            if not is_query: return

            question = self._remove_bot_mention(message.content)
            if message.guild:
                question = await translate_user_mentions(question, message.guild)
            
            image_urls = self._extract_image_urls(message)
            if not question and image_urls: question = "What do you see in this image?"
            
            await self._add_reaction(message, "⏳")
            logger.info(f"[USER QUERY] {message.author.display_name}: {question}")
            
            self._reset_query_tracking()
            query_start_time = time.time()
            
            reply_context, reply_image_urls = await self._get_reply_chain(message)
            if reply_image_urls: image_urls.extend(reply_image_urls)
            
            if reply_context:
                inferred = await self._infer_true_query(question, reply_context)
                if inferred != question:
                    question = inferred
                    logger.info(f"[INFERRED QUERY] {question}")

            try:
                self._register_channel_history_tool(message.channel, message.id)
                if image_urls: self._register_image_caption_tool(question)
                
                # Automatically caption images to provide context immediately
                auto_captions = []
                if image_urls:
                    logger.info(f"[AUTO-CAPTIONING] Processing {len(image_urls)} image(s)...")
                    from utils import two_round_image_caption
                    for i, url in enumerate(image_urls, 1):
                        try:
                            # Use thread to avoid blocking event loop
                            caption = await asyncio.to_thread(two_round_image_caption, url, self.api_key, question)
                            auto_captions.append(f"Auto-caption for Image {i} ({url}):\n{caption}")
                        except Exception as ce:
                            logger.warning(f"Auto-captioning failed for image {i}: {ce}")
                
                image_context = self._build_image_context(image_urls)
                if auto_captions:
                    image_context += "\n" + "\n\n".join(auto_captions) + "\n"
                
                question_with_context = f"[You are a Discord bot named {BOT_NAME}]\n\n{image_context}\n{reply_context}User question: {question}"
                
                async def update_hourglass(iteration_num):
                    if iteration_num > 0: await self._remove_reaction(message, "⏳" if iteration_num % 2 == 1 else "⌛")
                    await self._add_reaction(message, "⌛" if iteration_num % 2 == 1 else "⏳")
                
                answer = await asyncio.to_thread(
                    self.agent.run, question_with_context, max_iterations=30, verbose=False, 
                    iteration_callback=lambda n: asyncio.run_coroutine_threadsafe(update_hourglass(n), self.client.loop)
                )
                
                self._unregister_channel_history_tool()
                self._unregister_image_caption_tool()
                
                if len(answer) >= 1500:
                    condensed = await self._condense_response(answer)
                    tldr, _ = await self._generate_tldr(answer)
                    answer = f"{condensed}\n\n{tldr}"
                elif len(answer) >= 750:
                    tldr, _ = await self._generate_tldr(answer)
                    answer = f"{answer}\n\n{tldr}"
                
                await self._remove_reaction(message, "⏳")
                await self._remove_reaction(message, "⌛")
                
                stats = self.agent.get_tracking_data()
                merged_stats = dict(self.current_query_token_stats)
                self._merge_token_stats(stats["token_stats"], merged_stats)
                
                tool_counts = {}
                for e in stats["call_sequence"]:
                    if e["type"] == "tool_call": tool_counts[e["tool_name"]] = tool_counts.get(e["tool_name"], 0) + 1
                
                metadata = format_metadata(merged_stats, time.time() - query_start_time, tool_counts)
                complete_answer = await convert_usernames_to_mentions(answer + metadata, message.guild)
                
                files = self._get_files_to_attach(complete_answer, query_start_time)
                complete_answer = re.sub(r'attach\s+(?:scratch/)?[^\s,]+\.\w+', '', complete_answer, flags=re.IGNORECASE).strip()
                
                for chunk in self._split_long_message(complete_answer, 1900):
                    await message.channel.send(chunk, files=files if chunk == complete_answer[:1900] else None)
                
                self._save_query_log(str(message.id), question, complete_answer, message.author.display_name)
            
            except Exception as e:
                logger.error(f"Error processing question: {e}", exc_info=True)
                await message.channel.send(f"❌ Error: {str(e)}")
                await self._remove_reaction(message, "⏳")

        @self.client.event
        async def on_reaction_add(reaction, user):
            if user == self.client.user: return
            if str(reaction.emoji) == "🧪" and reaction.message.author != self.client.user:
                await self._log_eval_question(reaction.message, user)
            elif str(reaction.emoji) == "✅" and reaction.message.author == self.client.user:
                await self._log_accepted_answer(reaction.message, user)

    def _extract_image_urls(self, message):
        return [att.url for att in message.attachments if att.content_type and att.content_type.startswith('image/')]

    def _remove_bot_mention(self, text):
        return text.replace(f"<@{self.client.user.id}>", "").replace(f"<@!{self.client.user.id}>", "").strip()

    async def _add_reaction(self, message, emoji):
        try: await message.add_reaction(emoji)
        except: pass

    async def _remove_reaction(self, message, emoji):
        try: await message.remove_reaction(emoji, self.client.user)
        except: pass

    def _build_image_context(self, image_urls):
        if not image_urls: return ""
        details = "\n".join([f"Image {i} URL: {u}" for i, u in enumerate(image_urls, 1)])
        return f"\n\n[Images attached: {len(image_urls)} image(s)]\n{details}\n\nYou can use the 'caption_image' tool to analyze these images.\n"

    async def _get_reply_chain(self, message):
        chain, urls, curr, depth = [], [], message, 0
        while curr.reference and depth < 10:
            ref = getattr(curr, 'referenced_message', None)
            if not ref:
                try: ref = await curr.channel.fetch_message(curr.reference.message_id)
                except: break
            if not ref: break
            urls.extend(self._extract_image_urls(ref))
            content = self._remove_bot_mention(ref.content)
            if content: chain.append(f"{ref.author.display_name}: {content}")
            curr, depth = ref, depth + 1
        chain.reverse()
        urls.reverse()
        res = "Previous conversation context:\n" + "\n".join(chain) + "\n\n" if chain else ""
        if message.guild: res = await translate_user_mentions(res, message.guild)
        return res, urls

    def _get_files_to_attach(self, text, start_time):
        files = []
        scratch = Path(__file__).parent / 'scratch'
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
            for p in scratch.glob(ext):
                if not p.name.startswith(('YOUTUBE_', 'CODE_', 'DOWNLOAD_')) and p.stat().st_mtime > start_time:
                    files.append(discord.File(str(p)))
        for m in re.findall(r'attach\s+(?:scratch/)?([^\s,]+\.\w+)', text, re.IGNORECASE):
            p = scratch / m
            if p.exists() and not any(f.filename == m for f in files): files.append(discord.File(str(p)))
        return files

    def _split_long_message(self, text, limit=1900):
        chunks = []
        while len(text) > limit:
            pos = text.rfind(' ', 0, limit)
            if pos == -1: pos = limit
            chunks.append(text[:pos])
            text = text[pos:].strip()
        chunks.append(text)
        return chunks

    async def _infer_true_query(self, query, context):
        prompt = f"Synthesize a standalone query from context and message. Resolve pronouns. Return ONLY the synthesized query.\n\n{context}\nMessage: {query}\n\nStandalone Query:"
        try:
            res = await asyncio.to_thread(self._call_llm, prompt, 15)
            return res.strip().strip('"\'')
        except: return query

    async def _generate_tldr(self, text):
        prompt = f"Provide 1-2 sentence TL;DR. Preserve all URLs. Respond as {BOT_NAME}.\n\n{text}"
        try:
            start = time.time()
            res = await asyncio.to_thread(self._call_llm, prompt, 30)
            return res.strip(), time.time() - start
        except: return "**TL;DR:** Summary unavailable", 0.0

    async def _condense_response(self, text):
        prompt = f"Rewrite concisely (50-60% length). Preserve all URLs. Respond as {BOT_NAME}.\n\n{text}"
        try: return (await asyncio.to_thread(self._call_llm, prompt, 60)).strip()
        except: return text

    def _call_llm(self, prompt, timeout=10, model=None):
        m = model or self.agent.model
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        # Ensure prompt is a string for this simple call
        if not isinstance(prompt, str):
            prompt = str(prompt)
            
        data = {"model": m, "messages": [{"role": "user", "content": prompt}]}
        
        # Consistent model parameters
        if "temperature" in CONFIG: data["temperature"] = CONFIG["temperature"]
        if "max_tokens" in CONFIG: data["max_tokens"] = CONFIG["max_tokens"]
        
        # Handle thinking mode
        if CONFIG.get("enable_thinking", False):
            data["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        
        start = time.time()
        res = requests.post(CONFIG.get("base_url"), headers=headers, json=data, timeout=timeout)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"].strip()
        
        input_toks = int(len(prompt)/CHARS_PER_TOKEN)
        output_toks = int(len(content)/CHARS_PER_TOKEN)
        runtime = round(time.time() - start, 2)

        self.current_query_log.append({
            "type": "llm_call", "model": m, "timestamp": time.time(),
            "input": prompt, "output": content,
            "input_tokens": input_toks,
            "output_tokens": output_toks,
            "response_time_seconds": runtime
        })
        
        stats = self.current_query_token_stats.setdefault(m, {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0})
        stats["total_input_tokens"] += input_toks
        stats["total_output_tokens"] += output_toks
        stats["total_calls"] += 1
        return content

    def _merge_token_stats(self, src, dst):
        for m, s in src.items():
            d = dst.setdefault(m, {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0})
            for k in d: d[k] += s.get(k, 0)

    def _register_channel_history_tool(self, channel, msg_id):
        async def read_history(count=10):
            msgs = []
            async for m in channel.history(limit=count+10):
                if m.id != msg_id and m.author != self.client.user: msgs.append(m)
                if len(msgs) >= count: break
            msgs.reverse()
            fmt = [f"[{m.created_at.strftime('%Y-%m-%d %H:%M:%S')}] {m.author.display_name}: {self._remove_bot_mention(m.content)}" for m in msgs]
            return "Recent channel history:\n" + "\n".join(fmt) if fmt else "No history."

        self.agent.tool_functions["read_channel_history"] = lambda count=10: asyncio.run_coroutine_threadsafe(read_history(count), self.client.loop).result(timeout=10)
        from tools.tool_utils import create_tool_spec
        self.agent.tools.append(create_tool_spec("read_channel_history", "Read Discord messages.", {"count": "integer"}))

    def _unregister_channel_history_tool(self):
        self.agent.tool_functions.pop("read_channel_history", None)
        self.agent.tools = [t for t in self.agent.tools if t["function"]["name"] != "read_channel_history"]

    def _register_image_caption_tool(self, query):
        def caption(image_url=None):
            if not image_url:
                return "Error: No image_url provided to caption_image tool. Please provide a valid URL."
            try:
                from utils import caption_image_with_vlm
                p = f"Query: '{query}'. Analyze image." if query else "Describe image."
                return caption_image_with_vlm(image_url, self.api_key, p, model=CONFIG.get("image_caption_model"))
            except Exception as e: return f"Error: {e}"
        self.agent.tool_functions["caption_image"] = caption
        from tools.tool_utils import create_tool_spec
        self.agent.tools.append(create_tool_spec("caption_image", "Analyze image.", {"image_url": "string"}, required=["image_url"]))

    def _unregister_image_caption_tool(self):
        self.agent.tool_functions.pop("caption_image", None)
        self.agent.tools = [t for t in self.agent.tools if t["function"]["name"] != "caption_image"]

    def _reset_query_tracking(self):
        self.current_query_log, self.current_query_token_stats = [], {}
        self.agent.reset_tracking()

    async def _log_eval_question(self, message, user):
        q = self._remove_bot_mention(message.content)
        if not q: return
        qid = 1
        if self.QA_FILE.exists():
            with open(self.QA_FILE, 'r') as f:
                lines = [l for l in f if l.strip()]
                if lines: qid = json.loads(lines[-1]).get('qid', 0) + 1
        with open(self.QA_FILE, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({"question": q, "answer": "", "qid": qid}) + '\n')
            fcntl.flock(f, fcntl.LOCK_UN)
        await self._add_reaction(message, "📝")

    async def _log_accepted_answer(self, message, user):
        if not message.reference: return
        try:
            orig = await message.channel.fetch_message(message.reference.message_id)
            q = self._remove_bot_mention(orig.content)
            if not self.QA_FILE.exists(): return
            entries, updated = [], False
            with open(self.QA_FILE, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                for l in f:
                    if not l.strip(): continue
                    e = json.loads(l)
                    if e.get('question') == q and not e.get('answer'):
                        e['answer'], updated = message.content, True
                    entries.append(e)
                if updated:
                    f.seek(0); f.truncate()
                    for e in entries: f.write(json.dumps(e) + '\n')
                fcntl.flock(f, fcntl.LOCK_UN)
            if updated: await self._add_reaction(message, "💚")
        except: pass

    def _save_query_log(self, mid, q, resp, user):
        data = self.agent.get_tracking_data()
        logs = sorted(self.current_query_log + data["call_sequence"], key=lambda x: x.get("timestamp", 0))
        m_stats = dict(self.current_query_token_stats)
        self._merge_token_stats(data["token_stats"], m_stats)
        
        user_safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in user)[:50] or "user"
        path = self.QUERY_LOGS_DIR / f"{user_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, 'w') as f: json.dump({"message_id": mid, "username": user, "timestamp": datetime.now().isoformat(), "user_query": q, "final_response": resp[:1000], "call_sequence": logs, "token_stats_by_model": m_stats}, f, indent=2)
        for m, s in m_stats.items():
            logger.info(f"Model: {m} | Calls: {s['total_calls']} | Tokens: {s['total_input_tokens']}+{s['total_output_tokens']} | {path}")

    def run(self): self.client.run(self.token)

class BotRunner:
    def __init__(self): self.process, self.should_run = None, True
    def _read(self):
        try:
            while self.should_run:
                # Read from the master fd of the PTY
                try:
                    data = os.read(self.master_fd, 1024)
                    if data:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    else:
                        break
                except OSError:
                    break
        except Exception:
            pass
    def start_bot(self):
        if self.process: self.stop_bot()
        logger.info(f"Starting Discord bot...")
        env = os.environ.copy(); env["DISCORD_BOT_SUBPROCESS"] = "1"
        
        # Create a pseudo-terminal
        self.master_fd, self.slave_fd = pty.openpty()
        
        # Set slave PTY window size to match parent to prevent wrapping
        try:
            # Get parent window size from stdout (if TTY)
            winsize = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b'\0'*8)
            # Set slave window size
            fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            # Ignore if not running in a real TTY
            pass
        
        # Run subprocess attached to the PTY
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).absolute())],
            stdout=self.slave_fd,
            stderr=self.slave_fd, # Merge stderr into stdout via PTY
            stdin=self.slave_fd,
            text=False,
            bufsize=0,
            env=env,
            close_fds=True
        )
        
        # Close slave fd in parent so we get EOF when child closes it
        os.close(self.slave_fd)
        
        threading.Thread(target=self._read, daemon=True).start()
    def stop_bot(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except: self.process.kill(); self.process.wait()
        self.process = None
    def restart_bot(self):
        self.stop_bot(); time.sleep(1)
        if self.should_run: self.start_bot()

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class BotRestartHandler(FileSystemEventHandler):
    def __init__(self, callback): self.callback, self.last, self.debounce = callback, 0, 2
    def on_modified(self, event):
        if event.is_directory or any(x in str(Path(event.src_path).resolve()) for x in ['/scratch/', '/data/', '/web/']): return
        if event.src_path.endswith(('.py', '.yaml', '.yml')):
            if time.time() - self.last >= self.debounce:
                logger.info(f"Restarting due to {event.src_path}...")
                self.last = time.time(); self.callback()

def run_with_auto_restart():
    logger.info("Discord Bot with Auto-Restart Enabled")
    runner = BotRunner()
    obs = Observer()
    obs.schedule(BotRestartHandler(runner.restart_bot), str(Path.cwd()), recursive=True)
    obs.start()
    try:
        runner.start_bot()
        while runner.should_run:
            time.sleep(1)
            if runner.process and runner.process.poll() is not None and runner.process.poll() != 0:
                logger.warning("Bot crashed. Restarting..."); time.sleep(2); runner.restart_bot()
    except KeyboardInterrupt:
        runner.should_run = False; runner.stop_bot(); obs.stop()
    obs.join()

def main():
    token_file = ".bot_token"
    if not os.path.exists(token_file): return logger.error("ERROR: .bot_token missing")
    with open(token_file, 'r') as f: token = f.read().strip()
    api_key = os.getenv(CONFIG.get("api_key_env", "OPENROUTER_API_KEY"))
    if not api_key: return logger.error("ERROR: API key missing")
    if CONFIG.get("auto_restart", True) and not os.environ.get("DISCORD_BOT_SUBPROCESS"): run_with_auto_restart()
    else: DiscordBot(token, api_key).run()

if __name__ == "__main__": main()
