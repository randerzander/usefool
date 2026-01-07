from typing import Dict, List, Any, Union

def create_tool_spec(
    name: str,
    description: str,
    parameters: Dict[str, Union[str, Dict[str, Any]]],
    required: List[str] = None
) -> Dict[str, Any]:
    """
    Helper to create a tool specification dictionary.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Dictionary of parameter names to definitions.
                   Definition can be a dict (full spec) or string (description only, defaults to string type).
        required: List of required parameter names
        
    Returns:
        Tool specification dictionary
    """
    if required is None:
        required = []
        
    properties = {}
    for param_name, param_def in parameters.items():
        if isinstance(param_def, str):
            properties[param_name] = {
                "type": "string",
                "description": param_def
            }
        else:
            properties[param_name] = param_def
            
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
