#!/usr/bin/env python3
"""
File listing tool for scratch directory.
"""

import os
from pathlib import Path
from datetime import datetime


# Tool specification for agent registration
LIST_FILES_SPEC = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in the scratch/ directory. Returns file names, sizes, and modification times. Useful for seeing what code files or outputs are available.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', '*.png', 'CODE_*'). If not provided, lists all files."
                }
            },
            "required": []
        }
    }
}


def list_files(pattern: str = None) -> str:
    """
    List files in the scratch/ directory.
    
    Args:
        pattern: Optional glob pattern to filter files (e.g., '*.py', '*.png', 'CODE_*')
        
    Returns:
        Formatted string with file listing
    """
    try:
        # Get scratch directory
        project_root = Path(__file__).parent.parent.absolute()
        scratch_dir = project_root / "scratch"
        
        # Create scratch directory if it doesn't exist
        scratch_dir.mkdir(exist_ok=True)
        
        # Get files based on pattern
        if pattern:
            files = list(scratch_dir.glob(pattern))
        else:
            files = [f for f in scratch_dir.iterdir() if f.is_file()]
        
        if not files:
            if pattern:
                return f"No files matching pattern '{pattern}' found in scratch/"
            else:
                return "No files found in scratch/"
        
        # Sort by modification time, most recent first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Format output
        lines = []
        lines.append(f"Files in scratch/ ({len(files)} files):")
        lines.append("=" * 70)
        
        for f in files:
            stat = f.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            # Format size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            
            # Format timestamp
            time_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
            
            lines.append(f"{f.name:<40} {size_str:>10}  {time_str}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error listing files: {str(e)}"
