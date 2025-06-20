"""File and configuration helper functions."""

import json
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metadata(metadata: Dict[str, Any], filepath: Path):
    """Save metadata to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def ensure_directory(path: Path):
    """Ensure directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)


def load_prompt_file(filepath: str) -> Dict[str, Any]:
    """Load and validate prompt file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate required fields
    if "prompts" not in data:
        raise ValueError("Prompt file must contain 'prompts' array")

    return data


def create_project_structure(project_name: str, base_dir: str = "output"):
    """Create standardized project directory structure."""
    project_path = Path(base_dir) / project_name

    directories = [
        project_path / "raw",
        project_path / "processed",
        project_path / "metadata",
        project_path / "reports"
    ]

    for directory in directories:
        ensure_directory(directory)

    return project_path