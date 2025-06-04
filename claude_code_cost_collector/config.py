"""Configuration file handling for Claude Code Cost Collector.

This module provides functionality to read and manage configuration settings
for the Claude Code Cost Collector tool, including exchange rate API keys and
other configuration options.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .constants import CONFIDENCE_LEVELS, GRANULARITIES, OUTPUT_FORMATS
from .exceptions import ConfigError


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
        # Cost calculation settings
        "cost_calculation": {
            "enable_estimated_costs": True,  # Enable cost estimation for new format entries
            "confidence_threshold": "medium",  # Minimum confidence level: "high", "medium", "low"
            "show_confidence_level": False,  # Show confidence level in output
            "fallback_pricing_enabled": True,  # Use fallback pricing for unknown models
            "cache_read_cost_factor": 0.0,  # Multiplier for cache read costs (0.0 = free)
        },
        # Model pricing settings
        "model_pricing": {
            "custom_models": {},  # Custom model pricing overrides
            "pricing_update_check": False,  # Check for pricing updates (future feature)
            "unknown_model_fallback": "claude-3-sonnet",  # Fallback model for pricing
        },
    }


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON or YAML file.

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
            file_content = f.read()

        # Determine file format by extension
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            try:
                config_data = yaml.safe_load(file_content)
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in configuration file {config_path}: {e}")
        else:
            # Default to JSON
            try:
                config_data = json.loads(file_content)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in configuration file {config_path}: {e}")

        if config_data is None:
            config_data = {}

        return config_data  # type: ignore[no-any-return]
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
        # Check default configuration file locations (prefer YAML, fallback to JSON)
        default_config_paths = [
            Path.home() / ".claude_code_cost_collector.yaml",
            Path.home() / ".claude_code_cost_collector.yml",
            Path.home() / ".claude_code_cost_collector.json",
            Path.cwd() / "claude_code_cost_collector.yaml",
            Path.cwd() / "claude_code_cost_collector.yml",
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
    if config["default_granularity"] not in GRANULARITIES:
        raise ConfigError(f"Invalid granularity '{config['default_granularity']}'. Must be one of: {', '.join(GRANULARITIES)}")

    # Validate output format values
    if config["default_output_format"] not in OUTPUT_FORMATS:
        raise ConfigError(f"Invalid output format '{config['default_output_format']}'. Must be one of: {', '.join(OUTPUT_FORMATS)}")

    # Validate log directory exists or can be created
    log_dir = Path(config["default_log_directory"])
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, IOError) as e:
            raise ConfigError(f"Cannot access or create log directory {log_dir}: {e}")

    # Validate cost_calculation section
    if "cost_calculation" in config:
        cost_calc_config = config["cost_calculation"]
        if not isinstance(cost_calc_config, dict):
            raise ConfigError("cost_calculation must be a dictionary")

        # Validate confidence_threshold
        if "confidence_threshold" in cost_calc_config:
            confidence = cost_calc_config["confidence_threshold"]
            if confidence not in CONFIDENCE_LEVELS:
                raise ConfigError(f"Invalid confidence_threshold '{confidence}'. Must be one of: {', '.join(CONFIDENCE_LEVELS)}")

        # Validate boolean fields
        boolean_fields = ["enable_estimated_costs", "show_confidence_level", "fallback_pricing_enabled"]
        for field in boolean_fields:
            if field in cost_calc_config and not isinstance(cost_calc_config[field], bool):
                raise ConfigError(f"cost_calculation.{field} must be a boolean value")

        # Validate cache_read_cost_factor
        if "cache_read_cost_factor" in cost_calc_config:
            factor = cost_calc_config["cache_read_cost_factor"]
            if not isinstance(factor, (int, float)) or factor < 0.0:
                raise ConfigError("cost_calculation.cache_read_cost_factor must be a non-negative number")

    # Validate model_pricing section
    if "model_pricing" in config:
        model_pricing_config = config["model_pricing"]
        if not isinstance(model_pricing_config, dict):
            raise ConfigError("model_pricing must be a dictionary")

        # Validate custom_models
        if "custom_models" in model_pricing_config:
            custom_models = model_pricing_config["custom_models"]
            if not isinstance(custom_models, dict):
                raise ConfigError("model_pricing.custom_models must be a dictionary")

            # Validate each custom model pricing structure
            for model_name, pricing in custom_models.items():
                if not isinstance(pricing, dict):
                    raise ConfigError(f"Custom model pricing for '{model_name}' must be a dictionary")

                required_pricing_fields = ["input_cost", "output_cost"]
                for field in required_pricing_fields:
                    if field not in pricing:
                        raise ConfigError(f"Missing required pricing field '{field}' for model '{model_name}'")

                    cost_value = pricing[field]
                    if not isinstance(cost_value, (int, float)) or cost_value < 0.0:
                        raise ConfigError(f"Invalid {field} for model '{model_name}': must be a non-negative number")

                # Validate optional cache_creation_cost
                if "cache_creation_cost" in pricing:
                    cache_cost = pricing["cache_creation_cost"]
                    if not isinstance(cache_cost, (int, float)) or cache_cost < 0.0:
                        raise ConfigError(f"Invalid cache_creation_cost for model '{model_name}': must be a non-negative number")

        # Validate boolean fields
        boolean_fields = ["pricing_update_check"]
        for field in boolean_fields:
            if field in model_pricing_config and not isinstance(model_pricing_config[field], bool):
                raise ConfigError(f"model_pricing.{field} must be a boolean value")

        # Validate unknown_model_fallback
        if "unknown_model_fallback" in model_pricing_config:
            fallback = model_pricing_config["unknown_model_fallback"]
            if not isinstance(fallback, str) or not fallback.strip():
                raise ConfigError("model_pricing.unknown_model_fallback must be a non-empty string")


def create_sample_config_file(output_path: Path, format_type: str = "yaml") -> None:
    """Create a sample configuration file.

    Args:
        output_path: Path where to create the sample config file.
        format_type: Format of the config file ('yaml' or 'json').

    Raises:
        ConfigError: If the sample config file cannot be created.
    """
    sample_config = {
        "exchange_rate_api_key": "your_api_key_here",
        "default_log_directory": str(Path.home() / ".claude" / "projects"),
        "default_granularity": "daily",
        "default_output_format": "text",
        "timezone": "auto",  # "auto", "UTC", or timezone name like "Asia/Tokyo"
        "default_date_range_days": 30,  # Default number of days to analyze (0 = all data)
        # Cost calculation settings for new format support
        "cost_calculation": {
            "enable_estimated_costs": True,  # Enable cost estimation for new format entries
            "confidence_threshold": "medium",  # Minimum confidence level: "high", "medium", "low"
            "show_confidence_level": False,  # Show confidence level in output
            "fallback_pricing_enabled": True,  # Use fallback pricing for unknown models
            "cache_read_cost_factor": 0.0,  # Multiplier for cache read costs (0.0 = free)
        },
        # Model pricing settings for custom pricing overrides
        "model_pricing": {
            "custom_models": {
                # Example custom model pricing (remove if not needed)
                # "custom-model-name": {
                #     "input_cost": 0.003,          # USD per 1K input tokens
                #     "output_cost": 0.015,         # USD per 1K output tokens
                #     "cache_creation_cost": 0.0075  # USD per 1K cache creation tokens (optional)
                # }
            },
            "pricing_update_check": False,  # Check for pricing updates (future feature)
            "unknown_model_fallback": "claude-3-sonnet",  # Fallback model for pricing
        },
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if format_type.lower() == "yaml":
                yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            else:
                json.dump(sample_config, f, indent=2)
    except (OSError, IOError) as e:
        raise ConfigError(f"Cannot create sample config file {output_path}: {e}")
