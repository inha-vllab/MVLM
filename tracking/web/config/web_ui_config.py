"""
Web UI Parameter Configuration

Defines parameters that can be adjusted through the Web UI,
including their types, ranges, and default values.
"""

from typing import Dict, Any, List

# Parameter groups accessible through the Web UI
PARAMETER_GROUPS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Model": {
        "search_factor": {
            "type": "float",
            "min": 1.0,
            "max": 8.0,
            "step": 0.5,
            "default": 4.0,
            "description": "Search area size multiplier relative to target size",
        },
        "template_factor": {
            "type": "float",
            "min": 1.0,
            "max": 5.0,
            "step": 0.5,
            "default": 2.0,
            "description": "Template area size multiplier relative to target size",
        },
    },

    "Update": {
        "update_interval": {
            "type": "int",
            "min": 1,
            "max": 50,
            "default": 10,
            "description": "Frames between model updates",
        },
        "update_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.5,
            "description": "Confidence threshold for model update",
        },
    },

    "Tracker": {
        "max_score_decay": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.3,
            "description": "Allowed score decay before re-detection",
        },
        "detection_window": {
            "type": "int",
            "min": 0,
            "max": 100,
            "default": 10,
            "description": "Window size for re-detection",
        },
    },

    "Visualization": {
        "show_score": {
            "type": "bool",
            "default": True,
            "description": "Show confidence score on display",
        },
        "show_bbox": {
            "type": "bool",
            "default": True,
            "description": "Show bounding box on display",
        },
        "show_text": {
            "type": "bool",
            "default": True,
            "description": "Show target description on display",
        },
        "bbox_color": {
            "type": "str",
            "choices": ["red", "green", "blue", "yellow", "cyan", "magenta"],
            "default": "red",
            "description": "Bounding box color",
        },
        "bbox_thickness": {
            "type": "int",
            "min": 1,
            "max": 5,
            "default": 2,
            "description": "Bounding box line thickness",
        },
    },

    "Performance": {
        "use_half_precision": {
            "type": "bool",
            "default": False,
            "description": "Use FP16 mixed precision (faster, less accurate)",
        },
        "frame_skip": {
            "type": "int",
            "min": 0,
            "max": 10,
            "default": 0,
            "description": "Skip N frames between tracking (0=track all)",
        },
    },
}

# Parameter descriptions for tooltips
PARAMETER_DESCRIPTIONS: Dict[str, str] = {
    "search_factor": "Determines how large of an area to search around the previous target position. Higher values search a wider area but may be slower.",
    "template_factor": "Determines the size of the template crop area. Higher values include more context around the target.",
    "update_interval": "Number of frames to process before updating the target model. Higher values make tracking more stable to appearance changes.",
    "update_threshold": "Minimum confidence score required to update the target model. Lower values make updates more frequent.",
    "max_score_decay": "Maximum allowed drop in confidence score before attempting re-detection.",
    "detection_window": "Number of frames to use for re-detection when confidence drops.",
}


def get_parameter_groups() -> List[str]:
    """Get list of parameter group names."""
    return list(PARAMETER_GROUPS.keys())


def get_parameters_for_group(group: str) -> Dict[str, Dict[str, Any]]:
    """Get parameters for a specific group."""
    return PARAMETER_GROUPS.get(group, {})


def get_parameter_info(param_path: str) -> Dict[str, Any]:
    """
    Get information about a specific parameter.

    Args:
        param_path: Parameter path in format "group.parameter_name"

    Returns:
        Parameter info dict or None if not found
    """
    group, name = param_path.split(".", 1) if "." in param_path else (None, param_path)

    if group:
        if group in PARAMETER_GROUPS and name in PARAMETER_GROUPS[group]:
            return PARAMETER_GROUPS[group][name]
    else:
        for g in PARAMETER_GROUPS.values():
            if name in g:
                return g[name]

    return None


def get_default_parameters() -> Dict[str, Any]:
    """Get all default parameter values."""
    defaults = {}

    for group_name, group_params in PARAMETER_GROUPS.items():
        for param_name, param_info in group_params.items():
            defaults[param_name] = param_info.get("default")

    return defaults


def validate_parameter(param_name: str, value: Any) -> bool:
    """
    Validate a parameter value.

    Args:
        param_name: Name of the parameter
        value: Value to validate

    Returns:
        True if valid, False otherwise
    """
    info = get_parameter_info(param_name)
    if not info:
        return False

    param_type = info.get("type")

    if param_type == "float":
        min_val = info.get("min")
        max_val = info.get("max")
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False

    elif param_type == "int":
        min_val = info.get("min")
        max_val = info.get("max")
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False

    elif param_type == "str":
        choices = info.get("choices")
        if choices and value not in choices:
            return False

    return True
