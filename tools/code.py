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
        "description": "Generate Python code and save it to scratch/. Returns only the filename, not the code itself.",
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
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename to save as (default: CODE_{timestamp}.py)"
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
        "description": "Execute Python code in an isolated Docker container. If no filename given, executes the most recently created .py file in scratch/.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Optional filename in scratch/ to execute. If not provided, runs the most recently created .py file."
                }
            },
            "required": []
        }
    }
}


def write_code(task: str, context: str = None, filename: str = None) -> str:
    """
    Generate Python code and save it to scratch/.
    
    Args:
        task: Description of what the code should do
        context: Optional additional context or existing code to modify
        filename: Optional filename to save as (default: CODE_{timestamp}.py)
        
    Returns:
        Filename where code was saved (not the code itself)
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
        
        prompt += """

IMPORTANT REQUIREMENTS:
- When creating visualizations (matplotlib/seaborn/etc), ALWAYS use plt.savefig('filename.png') instead of plt.show()
- Save all outputs (images, data files, etc.) to the current directory
- Print a confirmation message when files are saved
- Provide ONLY the Python code without any explanations or markdown formatting
- Do not include ```python markers"""
        
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
        
        start_time = time.time()
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
        
        # Save code to scratch/
        project_root = Path(__file__).parent.parent.absolute()
        scratch_dir = project_root / "scratch"
        scratch_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"CODE_{int(time.time())}.py"
        else:
            # User provided a filename - just ensure it ends with .py
            if not filename.endswith(".py"):
                filename += ".py"
        
        # Write code to file
        code_file = scratch_dir / filename
        with open(code_file, 'w') as f:
            f.write(code)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated code with model: {model}, saved to {filename}, {len(code)} characters, response time: {elapsed_time:.2f}s")
        
        return filename
        
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}")
        return f"Error: {str(e)}"


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


def run_code(filename: str = None) -> dict:
    """
    Run Python code in a Docker container using Python from the host system.
    If no filename is provided, executes the most recently created .py file in scratch/.
    
    Args:
        filename: Optional filename in scratch/ to execute. If None, runs most recent .py file.
        
    Returns:
        Dict with keys: success (bool), stdout (str), stderr (str), exit_code (int), filename (str)
    """
    import docker
    
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.absolute()
        scratch_dir = project_root / "scratch"
        venv_dir = project_root / ".venv"
        
        # Create scratch directory if it doesn't exist
        scratch_dir.mkdir(exist_ok=True)
        
        # If no filename provided, find most recent .py file
        if filename is None:
            py_files = list(scratch_dir.glob("*.py"))
            if not py_files:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "No Python files found in scratch/",
                    "exit_code": -1,
                    "filename": None
                }
            # Sort by modification time, most recent first
            py_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            code_file = py_files[0]
            filename = code_file.name
        else:
            # Ensure filename ends with .py
            if not filename.endswith(".py"):
                filename += ".py"
            code_file = scratch_dir / filename
            
            if not code_file.exists():
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"File not found: {filename}",
                    "exit_code": -1,
                    "filename": filename
                }
        
        logger.info(f"Running code from {code_file} in Docker container")
        
        # Resolve the actual Python executable path (follow symlinks)
        python_exe = Path(venv_dir / "bin" / "python").resolve()
        python_parent = python_exe.parent.parent.parent  # Get the uv python installation dir
        
        # Initialize Docker client
        client = docker.from_env()
        
        # Get current user's UID and GID to run container as non-root
        import pwd
        uid = os.getuid()
        gid = os.getgid()
        
        # Run container with volumes mounted
        # Mount the actual Python installation and venv site-packages
        # Mount scratch as the working directory so code runs with cwd = scratch/
        # This allows code to access files directly via relative paths
        # Add Python bin directory to PATH so subprocess calls work
        container_scratch = str(scratch_dir)
        python_bin_dir = python_exe.parent
        container = client.containers.run(
            "ubuntu:22.04",
            command=f"{python_exe} {filename}",
            volumes={
                str(python_parent): {'bind': str(python_parent), 'mode': 'ro'},
                str(venv_dir): {'bind': str(venv_dir), 'mode': 'ro'},
                str(scratch_dir): {'bind': container_scratch, 'mode': 'rw'}
            },
            environment={
                'PYTHONPATH': str(venv_dir / 'lib' / 'python3.12' / 'site-packages'),
                'MPLCONFIGDIR': f'{container_scratch}/.matplotlib',  # Avoid permission errors
                'HOME': container_scratch,  # Set HOME to writable directory
                'PATH': f"{python_bin_dir}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            },
            working_dir=container_scratch,
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
        # The error object contains both stdout and stderr
        stdout_str = ""
        stderr_str = ""
        
        # Extract stderr (always present in ContainerError)
        if hasattr(e, 'stderr') and e.stderr:
            stderr_str = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else str(e.stderr)
        else:
            stderr_str = str(e)
        
        # Some errors might have stdout too
        if hasattr(e, 'stdout') and e.stdout:
            stdout_str = e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else str(e.stdout)
        
        # Combine for full error context
        full_error = f"STDERR:\n{stderr_str}"
        if stdout_str:
            full_error = f"STDOUT:\n{stdout_str}\n\n{full_error}"
        
        # Log full error without truncation
        logger.error(f"Code execution failed with exit code {e.exit_status}:\n{full_error}")
        
        return {
            "success": False,
            "stdout": stdout_str,
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


