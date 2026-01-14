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

from utils import MODEL_CONFIG, CHARS_PER_TOKEN, is_localhost, get_model_from_api
from logging_config import console
import tools

logger = logging.getLogger(__name__)

# Thread-local storage for current user query (accessible by tools)
_current_query = threading.local()

def get_current_user_query() -> Optional[str]:
    """Get the current user query from thread-local storage."""
    return getattr(_current_query, 'value', None)

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
            if is_localhost(self.api_url):
                detected_model = get_model_from_api(self.api_url, self.api_key)
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
        
        # Detect vision capabilities and context length for localhost models
        self.supports_vision = False
        self.context_length = 32768  # Default
        if is_localhost(self.api_url):
            self.supports_vision = self._detect_vision_support()
            self.context_length = self._detect_context_length()
            if enable_logging:
                if self.supports_vision:
                    logger.info(f"Model supports vision/image input")
                logger.info(f"Model context length: {self.context_length:,} tokens")
            
            # Make context length available to tools
            tools.set_context_length(self.context_length)
        
        # Callback for tool call interception (used by ResearchAgent)
        self.on_tool_call: Optional[Callable[[str, Dict[str, Any], Any, float], None]] = None
        
        self._initialize_tracking() 
    
    def _detect_vision_support(self) -> bool:
        """
        Detect if the model supports vision/image input by checking the /models endpoint.
        llama.cpp servers expose model metadata that indicates vision capabilities.
        """
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
            
            # Check for llama.cpp style models endpoint (has both "models" and "data")
            models_list = data.get("models", [])
            if models_list and len(models_list) > 0:
                model_info = models_list[0]
                # Check capabilities field (llama.cpp specific)
                capabilities = model_info.get("capabilities", [])
                if "multimodal" in capabilities:
                    return True
            
            # Check if there's model metadata indicating vision support
            models = data.get("data", []) if "data" in data else [data] if isinstance(data, dict) else data
            
            if models and len(models) > 0:
                model_info = models[0]
                if isinstance(model_info, dict):
                    # Check for vision-related metadata
                    # llama.cpp may expose this in different ways
                    metadata = model_info.get("metadata", {})
                    
                    # Check common indicators
                    if metadata.get("vision", False):
                        return True
                    if metadata.get("multimodal", False):
                        return True
                    if "vision" in str(metadata).lower():
                        return True
                    
                    # Check model ID for vision-related keywords
                    model_id = model_info.get("id", "") or model_info.get("name", "") or ""
                    vision_keywords = ["vision", "vl", "visual", "minicpm", "qwen", "llava", "bakllava"]
                    if any(keyword in model_id.lower() for keyword in vision_keywords):
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Vision detection failed: {e}")
            return False
    
    def _detect_context_length(self) -> int:
        """
        Detect the model's context length by checking the /models endpoint.
        Returns context length in tokens, or default of 32768 if detection fails.
        """
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
            
            # Check data field for metadata
            models = data.get("data", [])
            if models and len(models) > 0:
                model_info = models[0]
                if isinstance(model_info, dict):
                    metadata = model_info.get("meta", {})
                    
                    # Check for n_ctx_train (training context length)
                    if "n_ctx_train" in metadata:
                        ctx_len = metadata["n_ctx_train"]
                        logger.debug(f"Detected context length from n_ctx_train: {ctx_len}")
                        return ctx_len
            
            # Default to 32k if not found
            logger.debug("Could not detect context length, using default 32768")
            return 32768
            
        except Exception as e:
            logger.debug(f"Context length detection failed: {e}, using default 32768")
            return 32768
    
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
                        
                        # Format message with input tokens
                        msg = f"LLM Call Attempt {attempt + 1} | Model: {self.model} | {input_tokens} tokens in | {elapsed:.1f}s"
                        
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
                
                # Calculate throughput
                input_tokens_per_sec = round(input_tokens / response_time, 1) if response_time > 0 else 0
                output_tokens_per_sec = round(output_tokens / response_time, 1) if response_time > 0 else 0
                
                logger.info(f"LLM call completed | Status: {response.status_code} | Tokens: {input_tokens} in, {output_tokens} out | Time: {response_time:.2f}s | Throughput: {input_tokens_per_sec} in/s, {output_tokens_per_sec} out/s")
                
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

    def run(self, question: str, max_iterations: int = 30, verbose: bool = True, iteration_callback=None, stream: bool = False, status_prefix: str = "", exclude_tools: List[str] = None, image_urls: List[str] = None):
        # Store the user's question in thread-local storage so tools can access it
        global _current_query
        _current_query.value = question
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = MODEL_CONFIG.get("system_prompt", "You are a helpful assistant.")
        
        # Modify system prompt if we're using vision
        if self.supports_vision and image_urls:
            system_prompt += " You can see and analyze images directly in the conversation."
        
        # Build user message content - include images if model supports vision
        if self.supports_vision and image_urls:
            # For vision models, format message with image URLs
            import base64
            from utils import download_image, image_to_base64
            
            logger.info(f"Processing {len(image_urls)} image(s) for vision model...")
            content_parts = [{"type": "text", "text": question}]
            for i, img_url in enumerate(image_urls, 1):
                try:
                    # Download and convert to base64
                    img_path = download_image(img_url)
                    if not img_path:
                        logger.warning(f"Skipping image {i}: download failed")
                        continue
                        
                    b64_img = image_to_base64(img_path)
                    logger.info(f"Image {i} encoded: {len(b64_img)} chars of base64")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        }
                    })
                    # Clean up temp file
                    import os
                    os.remove(img_path)
                except Exception as e:
                    logger.warning(f"Failed to process image {img_url}: {e}")
            
            logger.info(f"Content parts: {len(content_parts)} (1 text + {len(content_parts)-1} images)")
            user_content = content_parts
        else:
            user_content = question
        
        messages = [
            {"role": "system", "content": f"{system_prompt} Current date and time: {current_time}"},
            {"role": "user", "content": user_content}
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
                
                # Some models return reasoning_content separately
                reasoning_content = message.get("reasoning_content", "")
                content = message.get("content", "") or ""
                tool_calls = message.get("tool_calls", [])
                
                # Filter out literal "None" responses from the model
                # Check if content is ONLY "None" (case-insensitive, with optional whitespace)
                if content and content.strip().lower() in ["none", "null", "n/a"]:
                    logger.warning(f"Model returned '{content.strip()}' - treating as empty response")
                    content = ""
                
                # If we have reasoning but no content, that's also a problem
                if reasoning_content and not content and not tool_calls:
                    logger.warning(f"Model returned reasoning but no content/tools - prompting for answer")
                    # Add system message and continue loop
                    messages.append(message)
                    messages.append({
                        "role": "system",
                        "content": "You provided reasoning but no answer. Please provide your final answer to the user's question now."
                    })
                    continue
                
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
                            
                            # Special handling for deep_research: return result directly
                            # This tool provides a complete, user-ready answer that shouldn't be modified
                            if function_name == "deep_research":
                                if verbose:
                                    logger.info(f"{current_iter_str} | deep_research completed - returning result directly")
                                return result_content_str if result_content_str.strip() else "Research completed but no summary available."
                        except Exception as e:
                            logger.error(f"Error in tool {function_name}: {str(e)}")
                            result_content = f"Error executing tool {function_name}: {str(e)}\n{traceback.format_exc()}"
                            
                        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result_content})
                    
                    # After all tool results, remind the model to answer the user's question
                    # This helps prevent "None" or empty responses
                    messages.append({
                        "role": "system",
                        "content": "Based on the tool results above, provide your answer to the user's question now. DO NOT output 'None'. Give a substantive response."
                    })
                else:
                    # No tool calls - this is the final response
                    return full_response if full_response.strip() else "I couldn't generate a response."

            return full_response if full_response.strip() else "Maximum iterations reached without a response."

        final_result = process()
        
        if stream:
            def gen():
                yield final_result
            return gen()
        return final_result