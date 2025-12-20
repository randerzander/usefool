#!/usr/bin/env python3
"""
Code generation tool using LLM.
Uses the coding_model from config.yaml to generate Python code.
"""

import os
import logging
import yaml
import requests
import time
from pathlib import Path


logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


# Tool specifications for agent registration
WRITE_CODE_SPEC = {
    "type": "function",
    "function": {
        "name": "write_code",
        "description": "Generate Python code based on a task description. Returns clean, executable Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of what the code should do"
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context or existing code to modify"
                }
            },
            "required": ["task"]
        }
    }
}

RUN_CODE_SPEC = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute Python code in an isolated Docker container. Returns stdout, stderr, and exit code. The code is saved to scratch/ and executed safely.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename to save the code as (default: auto-generated)"
                }
            },
            "required": ["code"]
        }
    }
}


def write_code(task: str, context: str = None) -> str:
    """
    Generate Python code based on a task description.
    
    Args:
        task: Description of what the code should do
        context: Optional additional context or existing code to modify
        
    Returns:
        Generated Python code as a string
    """
    try:
        # Get API settings from config
        api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        base_url = CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        model = CONFIG.get("coding_model", "mistralai/devstral-2512:free")
        
        # Build the prompt
        prompt = f"""You are an expert Python programmer. Generate clean, efficient, well-commented Python code.

Task: {task}"""
        
        if context:
            prompt += f"\n\nContext/Existing Code:\n{context}"
        
        prompt += "\n\nProvide ONLY the Python code without any explanations or markdown formatting. Do not include ```python markers."
        
        # Make API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add optional parameters if configured
        if "temperature" in CONFIG:
            data["temperature"] = CONFIG["temperature"]
        if "max_tokens" in CONFIG:
            data["max_tokens"] = CONFIG["max_tokens"]
        
        logger.info(f"Generating code with model: {model}")
        
        response = requests.post(base_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        code = result["choices"][0]["message"]["content"].strip()
        
        # Clean up markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]  # Remove ```python
        if code.startswith("```"):
            code = code[3:]  # Remove ```
        if code.endswith("```"):
            code = code[:-3]  # Remove trailing ```
        
        code = code.strip()
        
        logger.info(f"Generated {len(code)} characters of code")
        
        return code
        
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}")
        return f"# Error generating code: {str(e)}"


def explain_code(code: str) -> str:
    """
    Get an explanation of what a piece of code does.
    
    Args:
        code: Python code to explain
        
    Returns:
        Explanation of the code
    """
    try:
        # Get API settings from config
        api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        base_url = CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        model = CONFIG.get("coding_model", "mistralai/devstral-2512:free")
        
        prompt = f"""Explain what this Python code does in clear, concise language:

```python
{code}
```

Provide a brief explanation of the code's purpose and how it works."""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        logger.info(f"Explaining code with model: {model}")
        
        response = requests.post(base_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        explanation = result["choices"][0]["message"]["content"].strip()
        
        return explanation
        
    except Exception as e:
        logger.error(f"Code explanation failed: {str(e)}")
        return f"Error explaining code: {str(e)}"


def fix_code(code: str, error: str = None) -> str:
    """
    Fix bugs or issues in Python code.
    
    Args:
        code: Python code with issues
        error: Optional error message or description of the problem
        
    Returns:
        Fixed Python code
    """
    try:
        # Get API settings from config
        api_key_env = CONFIG.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        base_url = CONFIG.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        model = CONFIG.get("coding_model", "mistralai/devstral-2512:free")
        
        prompt = f"""Fix the issues in this Python code:

```python
{code}
```"""
        
        if error:
            prompt += f"\n\nError/Issue: {error}"
        
        prompt += "\n\nProvide ONLY the corrected Python code without any explanations or markdown formatting. Do not include ```python markers."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        logger.info(f"Fixing code with model: {model}")
        
        response = requests.post(base_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        fixed_code = result["choices"][0]["message"]["content"].strip()
        
        # Clean up markdown code blocks if present
        if fixed_code.startswith("```python"):
            fixed_code = fixed_code[9:]
        if fixed_code.startswith("```"):
            fixed_code = fixed_code[3:]
        if fixed_code.endswith("```"):
            fixed_code = fixed_code[:-3]
        
        fixed_code = fixed_code.strip()
        
        logger.info(f"Generated {len(fixed_code)} characters of fixed code")
        
        return fixed_code
        
    except Exception as e:
        logger.error(f"Code fixing failed: {str(e)}")
        return f"# Error fixing code: {str(e)}"


def run_code(code: str, filename: str = None) -> dict:
    """
    Run Python code in a Docker container using Python from the host system.
    
    Args:
        code: Python code to execute
        filename: Optional filename to save the code as (default: temp_code_{timestamp}.py)
        
    Returns:
        Dict with keys: success (bool), stdout (str), stderr (str), exit_code (int), filename (str)
    """
    try:
        import docker
        
        # Get project root directory
        project_root = Path(__file__).parent.parent.absolute()
        scratch_dir = project_root / "scratch"
        venv_dir = project_root / ".venv"
        
        # Resolve the actual Python executable path (follow symlinks)
        python_exe = Path(venv_dir / "bin" / "python").resolve()
        python_parent = python_exe.parent.parent.parent  # Get the uv python installation dir
        
        # Create scratch directory if it doesn't exist
        scratch_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"temp_code_{int(time.time())}.py"
        
        # Ensure filename ends with .py
        if not filename.endswith(".py"):
            filename += ".py"
        
        # Write code to scratch directory
        code_file = scratch_dir / filename
        with open(code_file, 'w') as f:
            f.write(code)
        
        logger.info(f"Running code from {code_file} in Docker container")
        
        # Initialize Docker client
        client = docker.from_env()
        
        # Get current user's UID and GID to run container as non-root
        import pwd
        uid = os.getuid()
        gid = os.getgid()
        
        # Run container with volumes mounted
        # Mount the actual Python installation and venv site-packages
        container = client.containers.run(
            "ubuntu:22.04",
            command=f"{python_exe} /scratch/{filename}",
            volumes={
                str(python_parent): {'bind': str(python_parent), 'mode': 'ro'},
                str(venv_dir): {'bind': str(venv_dir), 'mode': 'ro'},
                str(scratch_dir): {'bind': '/scratch', 'mode': 'rw'}
            },
            environment={
                'PYTHONPATH': str(venv_dir / 'lib' / 'python3.12' / 'site-packages')
            },
            working_dir="/scratch",
            user=f"{uid}:{gid}",  # Run as current user, not root
            remove=True,
            detach=False,
            stdout=True,
            stderr=True
        )
        
        # Container output is bytes
        output = container.decode('utf-8') if isinstance(container, bytes) else str(container)
        
        logger.info(f"Code execution completed successfully")
        
        return {
            "success": True,
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
            "filename": filename
        }
        
    except docker.errors.ContainerError as e:
        # Container exited with non-zero code
        # Note: container is already removed when remove=True, so we can't fetch logs
        # The error message from ContainerError contains stderr
        error_msg = str(e)
        # Try to extract stderr from the error message
        if hasattr(e, 'stderr') and e.stderr:
            stderr_str = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else str(e.stderr)
        else:
            stderr_str = error_msg
        
        # Log with truncated stderr for readability
        stderr_preview = stderr_str[:200] + "..." if len(stderr_str) > 200 else stderr_str
        logger.error(f"Code execution failed with exit code {e.exit_status}: {stderr_preview}")
        
        return {
            "success": False,
            "stdout": "",
            "stderr": stderr_str,
            "exit_code": e.exit_status,
            "filename": filename
        }
    except docker.errors.ImageNotFound:
        logger.error("Docker image ubuntu:22.04 not found")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Error: Docker image ubuntu:22.04 not found. Run: docker pull ubuntu:22.04",
            "exit_code": -1,
            "filename": filename
        }
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {str(e)}")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Docker API error: {str(e)}",
            "exit_code": -1,
            "filename": filename
        }
    except ImportError:
        logger.error("docker library not installed")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Error: docker library not installed. Run: pip install docker",
            "exit_code": -1,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Code execution failed: {str(e)}")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error executing code: {str(e)}",
            "exit_code": -1,
            "filename": filename
        }


