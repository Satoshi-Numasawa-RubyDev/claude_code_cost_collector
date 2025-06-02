"""Configuration file handling for Claude Code Cost Collector.

This module provides functionality to read and manage configuration settings
for the Claude Code Cost Collector tool, including exchange rate API keys and
other configuration options.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values.

    Returns:
        Dict containing default configuration values.
    """
    return {
        "exchange_rate_api_key": None,
        "default_log_directory": str(Path.home() / ".claude" / "projects"),
        "default_granularity": "daily",
        "default_output_format": "text",
        "timezone": "auto",  # "auto", "UTC", or timezone name like "Asia/Tokyo"
        "default_date_range_days": 30,  # Default number of days to analyze (0 = all data)
    }


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dict containing configuration values.

    Raises:
        ConfigError: If the configuration file cannot be read or parsed.
    """
    try:
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return config_data  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in configuration file {config_path}: {e}")
    except (OSError, IOError) as e:
        raise ConfigError(f"Cannot read configuration file {config_path}: {e}")


def merge_config(default_config: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge file configuration with default configuration.

    Args:
        default_config: Default configuration values.
        file_config: Configuration values loaded from file.

    Returns:
        Merged configuration dict.
    """
    merged = default_config.copy()
    merged.update(file_config)
    return merged


def get_config_from_env() -> Dict[str, Any]:
    """Get configuration values from environment variables.

    Returns:
        Dict containing configuration values from environment variables.
    """
    env_config = {}

    # Exchange rate API key from environment
    api_key = os.getenv("CLAUDE_CODE_COST_COLLECTOR_API_KEY")
    if api_key:
        env_config["exchange_rate_api_key"] = api_key

    # Log directory from environment
    log_dir = os.getenv("CLAUDE_CODE_COST_COLLECTOR_LOG_DIR")
    if log_dir:
        env_config["default_log_directory"] = log_dir

    return env_config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file, environment, and defaults.

    Configuration precedence (highest to lowest):
    1. Environment variables
    2. Configuration file
    3. Default values

    Args:
        config_path: Optional path to configuration file.
                    If None, default locations will be checked.

    Returns:
        Final merged configuration dict.

    Raises:
        ConfigError: If specified config file cannot be loaded.
    """
    # Start with default configuration
    config = get_default_config()

    # Try to load from configuration file
    if config_path:
        # User specified a config file path
        file_config = load_config_file(Path(config_path))
        config = merge_config(config, file_config)
    else:
        # Check default configuration file locations
        default_config_paths = [
            Path.home() / ".claude_code_cost_collector.json",
            Path.cwd() / "claude_code_cost_collector.json",
        ]

        for path in default_config_paths:
            if path.exists():
                try:
                    file_config = load_config_file(path)
                    config = merge_config(config, file_config)
                    break
                except ConfigError:
                    # Continue to next config file if current one fails
                    continue

    # Override with environment variables (highest priority)
    env_config = get_config_from_env()
    config = merge_config(config, env_config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values.

    Args:
        config: Configuration dict to validate.

    Raises:
        ConfigError: If configuration is invalid.
    """
    required_keys = [
        "default_log_directory",
        "default_granularity",
        "default_output_format",
    ]

    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required configuration key: {key}")

    # Validate granularity values
    valid_granularities = ["daily", "monthly", "project", "session", "all"]
    if config["default_granularity"] not in valid_granularities:
        raise ConfigError(
            f"Invalid granularity '{config['default_granularity']}'. Must be one of: {', '.join(valid_granularities)}"
        )

    # Validate output format values
    valid_formats = ["text", "json", "yaml", "csv"]
    if config["default_output_format"] not in valid_formats:
        raise ConfigError(
            f"Invalid output format '{config['default_output_format']}'. Must be one of: {', '.join(valid_formats)}"
        )

    # Validate log directory exists or can be created
    log_dir = Path(config["default_log_directory"])
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, IOError) as e:
            raise ConfigError(f"Cannot access or create log directory {log_dir}: {e}")


def create_sample_config_file(output_path: Path) -> None:
    """Create a sample configuration file.

    Args:
        output_path: Path where to create the sample config file.

    Raises:
        ConfigError: If the sample config file cannot be created.
    """
    sample_config = {
        "exchange_rate_api_key": "your_api_key_here",
        "default_log_directory": str(Path.home() / ".claude" / "projects"),
        "default_granularity": "daily",
        "default_output_format": "text",
        "timezone": "auto",
        "default_date_range_days": 30,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_config, f, indent=2)
    except (OSError, IOError) as e:
        raise ConfigError(f"Cannot create sample config file {output_path}: {e}")
