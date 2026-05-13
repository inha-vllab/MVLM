"""
Model Management for MVLM Web UI

Provides functions for scanning and listing available trained models
from the output directory.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Project root directory (resolved from this file's location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _extract_epoch(name: str):
    """Try to extract epoch number from checkpoint filename."""
    if "ep" in name.lower():
        parts = name.lower().split("ep")
        if len(parts) > 1:
            try:
                return int(parts[1].split("_")[0].split(".")[0])
            except ValueError:
                pass
    return None


def list_available_models() -> List[Dict[str, Any]]:
    """
    Scan output/ and models/ directories for trained checkpoints.

    Returns:
        List of model dictionaries with keys: name, path, config, epoch
    """
    models = []

    # Scan output/ directory (experiment checkpoints)
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        for exp_dir in output_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            for ckpt_file in exp_dir.glob("**/*.pth.tar"):
                models.append({
                    "name": ckpt_file.stem,
                    "path": str(ckpt_file),
                    "config": exp_dir.name,
                    "epoch": _extract_epoch(ckpt_file.stem),
                    "size_mb": ckpt_file.stat().st_size / (1024 * 1024)
                })

    # Scan models/ directory (standalone checkpoints)
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        for ckpt_file in models_dir.glob("**/*.pth.tar"):
            models.append({
                "name": ckpt_file.stem,
                "path": str(ckpt_file),
                "config": None,
                "epoch": _extract_epoch(ckpt_file.stem),
                "size_mb": ckpt_file.stat().st_size / (1024 * 1024)
            })

    # Sort by epoch (descending) if available, else by name
    models.sort(key=lambda m: (m.get("epoch") or 0), reverse=True)

    return models


def get_default_model() -> Dict[str, Any]:
    """
    Get the default model for demo.

    Returns:
        Model dictionary or None if not found
    """
    default_path = str(PROJECT_ROOT / "models" / "260126_sutrack_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003_mean_ep0060.pth.tar")

    if os.path.exists(default_path):
        return {
            "name": "default_mvlm",
            "path": default_path,
            "config": "mvlm_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003",
            "epoch": 60
        }
    return None


def list_available_configs() -> List[str]:
    """
    List available config YAML files from experiments directory.

    Returns:
        List of config names (without .yaml extension)
    """
    configs = []

    experiments_dir = PROJECT_ROOT / "experiments" / "mvlm"
    if not experiments_dir.exists():
        return configs

    for yaml_file in experiments_dir.glob("*.yaml"):
        configs.append(yaml_file.stem)

    return sorted(configs)


def get_config_path(config_name: str) -> str:
    """
    Get full path to config YAML file.

    Args:
        config_name: Config name (with or without .yaml extension)

    Returns:
        Full path to config file
    """
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_path = PROJECT_ROOT / "experiments" / "mvlm" / config_name

    if config_path.exists():
        return str(config_path)

    return None
