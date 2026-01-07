#!/usr/bin/env python3
"""
Core Agent implementation using OpenAI tools API.
Handles tool orchestration, LLM communication, and iteration logic.
"""

import json
import logging
import time
import requests
import traceback
import threading
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

from utils import MODEL_CONFIG, CHARS_PER_TOKEN
from logging_config import console
import tools

logger = logging.getLogger(__name__)

class Agent:
    """
    An agent that can use tools to answer questions via OpenAI tools API.
    """
    
    # Constants
    MAX_CONTENT_LENGTH = 4000
    
    def __init__(self, api_key: str, model: str = None, base_url: str = None, enable_logging: bool = True):
        """
        Initialize the agent. 
        """
        self.api_key = api_key
        self.api_url = base_url if base_url is not None else MODEL_CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        
        if model is None:
            if self._is_localhost(self.api_url):
                detected_model = self._get_model_from_api()
                if detected_model:
                    self.model = detected_model
                    if enable_logging:
                        logger.info(f"Auto-detected model from localhost: {self.model}")
                else:
                    self.model = MODEL_CONFIG.get("default_model")
            else:
                self.model = MODEL_CONFIG.get("default_model")
                if not self.model:
                    raise ValueError("No default_model specified in config.yaml")
        else:
            self.model = model
            
        self.enable_logging = enable_logging
        
        if not enable_logging:
            logging.getLogger(__name__).setLevel(logging.WARNING)
        else:
            logging.getLogger(__name__).setLevel(logging.INFO)
        
        self.tool_functions = tools.get_tool_functions()
        self.tools = tools.get_tool_specs()
        
        # Callback for tool call interception (used by ResearchAgent)
        self.on_tool_call: Optional[Callable[[str, Dict[str, Any], Any, float], None]] = None
        
        self._initialize_tracking() 
    
    def _is_localhost(self, url: str) -> bool:
        return "localhost" in url or "127.0.0.1" in url
    
    def _get_model_from_api(self) -> Optional[str]:
        try:
            if "/chat/completions" in self.api_url:
                models_url = self.api_url.replace("/chat/completions", "/models")
            elif self.api_url.endswith("/v1"):
                models_url = f"{self.api_url}/models"
            else:
                models_url = f"{self.api_url}/v1/models"
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(models_url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", []) if "data" in data else [data] if isinstance(data, dict) else data
            
            if models and len(models) > 0:
                model = models[0]
                if isinstance(model, dict):
                    model_id = model.get("id") or model.get("name") or model.get("model")
                    if model_id and model_id.endswith(".gguf"):
                        model_id = model_id[:-5]
                    return model_id
            return None
        except Exception as e:
            logger.warning(f"Failed to auto-detect model from API: {e}")
            return None
    
    def _initialize_tracking(self):
        self.call_sequence = []
        self.token_stats = {}
    
    def reset_tracking(self):
        self._initialize_tracking()
    
    def get_tracking_data(self) -> Dict[str, Any]:
        return {
            "call_sequence": self.call_sequence,
            "token_stats": self.token_stats
        }
    
    def _call_llm(self, messages: List[Dict[str, str]], use_tools: bool = True, tools_override: List[Dict[str, Any]] = None):
        # Filter messages to ensure only text and valid image content are passed
        safe_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Ensure each part is either a valid text or image structure
                safe_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and part.get("text"):
                            safe_parts.append(part)
                        elif part.get("type") == "image_url" and part.get("image_url", {}).get("url"):
                            safe_parts.append(part)
                safe_messages.append({"role": msg["role"], "content": safe_parts})
            elif isinstance(content, str):
                safe_messages.append(msg)
            else:
                # Fallback: convert unknown types to string if possible, or skip
                safe_messages.append({"role": msg["role"], "content": str(content)})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": safe_messages,
            "stream": False
        }
        
        if use_tools:
            data["tools"] = tools_override if tools_override is not None else self.tools
            data["tool_choice"] = "auto"
        
        if "temperature" in MODEL_CONFIG:
            data["temperature"] = MODEL_CONFIG["temperature"]
        if "top_p" in MODEL_CONFIG:
            data["top_p"] = MODEL_CONFIG["top_p"]
        if "max_tokens" in MODEL_CONFIG:
            data["max_tokens"] = MODEL_CONFIG["max_tokens"]
        
        if MODEL_CONFIG.get("enable_thinking", False):
            data["reasoning_effort"] = "high"
        
        input_text = str(messages)
        input_tokens = int(len(input_text) / CHARS_PER_TOKEN)
        start_time = time.time()
        
        # Track input tokens and call count
        if self.model not in self.token_stats:
            self.token_stats[self.model] = {
                "total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0
            }
        self.token_stats[self.model]["total_input_tokens"] += input_tokens
        self.token_stats[self.model]["total_calls"] += 1

        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries + 1):
            try:
                # Manual spinner implementation to ensure single-line updates
                # rich.console.status can sometimes spam newlines in certain environments
                stop_spinner = threading.Event()
                
                def spin():
                    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    i = 0
                    start = time.time()
                    while not stop_spinner.is_set():
                        frame = frames[i % len(frames)]
                        elapsed = time.time() - start
                        
                        # Format message
                        msg = f"LLM Call Attempt {attempt + 1} | Model: {self.model} | {elapsed:.1f}s"
                        
                        # Use stdout to match the logs
                        # Use ANSI clear line (\x1b[2K) + return (\r) to force single line update
                        sys.stdout.write(f"\x1b[2K\r{frame} {msg}")
                        sys.stdout.flush()
                        
                        time.sleep(0.1)
                        i += 1
                        
                spinner_thread = None
                if self.enable_logging:
                    spinner_thread = threading.Thread(target=spin)
                    spinner_thread.daemon = True
                    spinner_thread.start()
                
                try:
                    response = requests.post(self.api_url, headers=headers, json=data, timeout=None)
                finally:
                    if spinner_thread:
                        stop_spinner.set()
                        spinner_thread.join()
                        # Clear the line fully
                        sys.stdout.write("\x1b[2K\r")
                        sys.stdout.flush()

                if response.status_code == 429 and attempt < max_retries:
                    logger.warning(f"Rate limited (429). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                response_time = time.time() - start_time
                message = result["choices"][0]["message"]
                content = message.get("content", "") or ""
                
                # Calculate output tokens
                output_text = content
                if "tool_calls" in message and message["tool_calls"]:
                    output_text += str(message["tool_calls"])
                output_tokens = int(len(output_text) / CHARS_PER_TOKEN)
                
                logger.info(f"LLM call completed | Status: {response.status_code} | Tokens: {input_tokens} in, {output_tokens} out | Time: {response_time:.2f}s")
                
                call_entry = {
                    "type": "llm_call",
                    "model": self.model,
                    "timestamp": time.time(),
                    "input": messages,
                    "output": message,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "response_time_seconds": round(response_time, 2),
                    "input_tokens_per_sec": round(input_tokens / response_time, 2) if response_time > 0 else 0,
                    "output_tokens_per_sec": round(output_tokens / response_time, 2) if response_time > 0 else 0
                }
                self.call_sequence.append(call_entry)
                self.token_stats[self.model]["total_output_tokens"] += output_tokens
                
                return result
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"LLM call failed - Model: {self.model}, Error: {str(e)}")
                    raise
                logger.warning(f"LLM call error: {str(e)}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2

    def run(self, question: str, max_iterations: int = 30, verbose: bool = True, iteration_callback=None, stream: bool = False, status_prefix: str = "", exclude_tools: List[str] = None):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = MODEL_CONFIG.get("system_prompt", "You are a helpful assistant.")
        messages = [
            {"role": "system", "content": f"{system_prompt} Current date and time: {current_time}"},
            {"role": "user", "content": question}
        ]

        active_tools = self.tools
        if exclude_tools:
            active_tools = [t for t in self.tools if t["function"]["name"] not in exclude_tools]

        def process():
            full_response = ""
            for iteration in range(max_iterations):
                if iteration_callback:
                    iteration_callback(iteration)

                # Store iteration info to print with tool calls
                current_iter_str = f"{status_prefix} Iteration {iteration + 1}/{max_iterations}" if status_prefix else f"Iteration {iteration + 1}/{max_iterations}"

                result = self._call_llm(messages, use_tools=True, tools_override=active_tools)
                message = result["choices"][0]["message"]
                content = message.get("content", "") or ""
                tool_calls = message.get("tool_calls", [])
                
                if content:
                    full_response += content

                if content and not tool_calls and verbose:
                    logger.info(f"{current_iter_str} | Final Response")

                if tool_calls:
                    messages.append(message)
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        try:
                            function_args = json.loads(tool_call["function"]["arguments"])
                            
                            if verbose:
                                logger.info(f"{current_iter_str} | Tool Call: {function_name} | Arguments: {json.dumps(function_args)}")
                            
                            tool_start_time = time.time()
                            result_content = self.tool_functions[function_name](**function_args)
                            tool_runtime = time.time() - tool_start_time
                            
                            # Ensure result_content is a string
                            if not isinstance(result_content, str):
                                result_content_str = json.dumps(result_content)
                            else:
                                result_content_str = result_content

                            # Trigger interceptor callback if registered
                            if self.on_tool_call:
                                self.on_tool_call(function_name, function_args, result_content_str, tool_runtime)

                            if verbose:
                                logger.info(f"{current_iter_str} | Tool Result: {result_content_str[:100].replace(chr(10), ' ')}... ({round(tool_runtime, 2)}s)")
                                
                            self.call_sequence.append({
                                "type": "tool_call",
                                "tool_name": function_name,
                                "arguments": function_args,
                                "output": result_content_str[:1000] + "..." if len(result_content_str) > 1000 else result_content_str,
                                "timestamp": time.time(),
                                "runtime_seconds": round(tool_runtime, 2)
                            })
                            
                            result_content = result_content_str
                        except Exception as e:
                            logger.error(f"Error in tool {function_name}: {str(e)}")
                            result_content = f"Error executing tool {function_name}: {str(e)}\n{traceback.format_exc()}"
                            
                        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result_content})
                else:
                    return full_response

            return full_response or "Maximum iterations reached."

        final_result = process()
        
        if stream:
            def gen():
                yield final_result
            return gen()
        return final_result