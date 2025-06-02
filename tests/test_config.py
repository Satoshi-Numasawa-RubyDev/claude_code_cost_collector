"""Tests for config module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from claude_code_cost_collector.config import (
    ConfigError,
    create_sample_config_file,
    get_config_from_env,
    get_default_config,
    load_config,
    load_config_file,
    merge_config,
    validate_config,
)


class TestConfig(unittest.TestCase):
    """Test cases for configuration functionality."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()

        self.assertIsInstance(config, dict)
        self.assertIn("exchange_rate_api_key", config)
        self.assertIn("default_log_directory", config)
        self.assertIn("default_granularity", config)
        self.assertIn("default_output_format", config)

        self.assertEqual(config["default_granularity"], "daily")
        self.assertEqual(config["default_output_format"], "text")

    def test_load_config_file_valid(self):
        """Test loading valid configuration file."""
        config_data = {
            "exchange_rate_api_key": "test_key",
            "default_granularity": "monthly",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loaded_config = load_config_file(config_path)
            self.assertEqual(loaded_config, config_data)
        finally:
            config_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent configuration file."""
        config_path = Path("/nonexistent/config.json")

        with self.assertRaises(ConfigError) as cm:
            load_config_file(config_path)

        self.assertIn("Configuration file not found", str(cm.exception))

    def test_load_config_file_invalid_json(self):
        """Test loading configuration file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json ")
            config_path = Path(f.name)

        try:
            with self.assertRaises(ConfigError) as cm:
                load_config_file(config_path)

            self.assertIn("Invalid JSON", str(cm.exception))
        finally:
            config_path.unlink()

    def test_merge_config(self):
        """Test merging configurations."""
        default_config = {
            "key1": "default_value1",
            "key2": "default_value2",
            "key3": "default_value3",
        }

        file_config = {
            "key2": "file_value2",
            "key4": "file_value4",
        }

        merged = merge_config(default_config, file_config)

        expected = {
            "key1": "default_value1",
            "key2": "file_value2",  # Overridden by file config
            "key3": "default_value3",
            "key4": "file_value4",  # Added from file config
        }

        self.assertEqual(merged, expected)

    @patch.dict(
        os.environ,
        {
            "CLAUDE_CODE_COST_COLLECTOR_API_KEY": "env_api_key",
            "CLAUDE_CODE_COST_COLLECTOR_LOG_DIR": "/env/log/dir",
        },
    )
    def test_get_config_from_env(self):
        """Test getting configuration from environment variables."""
        env_config = get_config_from_env()

        expected = {
            "exchange_rate_api_key": "env_api_key",
            "default_log_directory": "/env/log/dir",
        }

        self.assertEqual(env_config, expected)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_from_env_empty(self):
        """Test getting configuration from environment when no vars are set."""
        env_config = get_config_from_env()
        self.assertEqual(env_config, {})

    def test_load_config_with_specified_file(self):
        """Test loading configuration with specified file path."""
        config_data = {
            "exchange_rate_api_key": "test_key",
            "default_granularity": "monthly",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                config = load_config(config_path)

            # Should merge default config with file config
            self.assertEqual(config["exchange_rate_api_key"], "test_key")
            self.assertEqual(config["default_granularity"], "monthly")
            self.assertEqual(config["default_output_format"], "text")  # From default
        finally:
            Path(config_path).unlink()

    @patch.dict(
        os.environ,
        {
            "CLAUDE_CODE_COST_COLLECTOR_API_KEY": "env_key",
        },
    )
    def test_load_config_env_override(self):
        """Test that environment variables override config file."""
        config_data = {
            "exchange_rate_api_key": "file_key",
            "default_granularity": "monthly",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)

            # Environment should override file
            self.assertEqual(config["exchange_rate_api_key"], "env_key")
            self.assertEqual(config["default_granularity"], "monthly")
        finally:
            Path(config_path).unlink()

    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        config = get_default_config()

        # Should not raise exception
        validate_config(config)

    def test_validate_config_missing_key(self):
        """Test validating configuration with missing required key."""
        config = get_default_config()
        del config["default_log_directory"]

        with self.assertRaises(ConfigError) as cm:
            validate_config(config)

        self.assertIn("Missing required configuration key", str(cm.exception))

    def test_validate_config_invalid_granularity(self):
        """Test validating configuration with invalid granularity."""
        config = get_default_config()
        config["default_granularity"] = "invalid"

        with self.assertRaises(ConfigError) as cm:
            validate_config(config)

        self.assertIn("Invalid granularity", str(cm.exception))

    def test_validate_config_invalid_output_format(self):
        """Test validating configuration with invalid output format."""
        config = get_default_config()
        config["default_output_format"] = "invalid"

        with self.assertRaises(ConfigError) as cm:
            validate_config(config)

        self.assertIn("Invalid output format", str(cm.exception))

    def test_create_sample_config_file(self):
        """Test creating sample configuration file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            create_sample_config_file(output_path)

            # Verify file was created and contains valid JSON
            self.assertTrue(output_path.exists())

            with open(output_path, "r") as f:
                sample_config = json.load(f)

            self.assertIn("exchange_rate_api_key", sample_config)
            self.assertIn("default_log_directory", sample_config)
            self.assertIn("default_granularity", sample_config)
            self.assertIn("default_output_format", sample_config)
        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    unittest.main()
