"""
Centralized configuration loading.

Loads base.yaml + system-specific YAML configs and merges with .env variables.
Shared by ALL 4 systems.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")


def load_config(system_name: str | None = None) -> dict:
    """
    Load configuration from YAML files and environment variables.

    Args:
        system_name: Optional system config to merge (e.g., 'rag_monolith').
                     If provided, system-specific values override base values.

    Returns:
        Merged configuration dictionary.
    """
    # Load base config
    base_path = CONFIGS_DIR / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Merge system-specific config
    if system_name:
        system_path = CONFIGS_DIR / f"{system_name}.yaml"
        if system_path.exists():
            with open(system_path, encoding="utf-8") as f:
                system_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, system_config)

    # Inject environment variables
    config["google_cloud_project"] = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    config["google_cloud_location"] = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    config["sec_edgar_email"] = os.getenv("SEC_EDGAR_EMAIL", "")
    config["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
