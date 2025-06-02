"""
Error handling integration tests for Claude Code Cost Collector.

These tests verify that the application handles various error conditions
gracefully and provides appropriate error messages.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestErrorHandling:
    """Error handling integration tests."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory path."""
        return Path(__file__).parent / "test_data"

    def run_command(self, args, input_text=None):
        """Helper method to run the application and capture output."""
        cmd = [sys.executable, "-m", "claude_code_cost_collector"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, input=input_text)
        return result

    def test_nonexistent_directory_error(self):
        """Test error handling for nonexistent directory."""
        result = self.run_command(["--directory", "/absolutely/nonexistent/directory"])

        # Should return error code (argparse returns 2)
        assert result.returncode == 2

        # Should provide meaningful error message
        error_output = result.stderr + result.stdout
        assert any(keyword in error_output.lower() for keyword in ["error", "エラー", "not found", "exist"])

    def test_permission_denied_error(self):
        """Test error handling for permission denied."""
        # This test might not work on all systems, so we'll test with a restricted directory
        # that commonly exists and has restricted access
        restricted_paths = ["/root", "/sys", "/proc/1"]

        for path in restricted_paths:
            if Path(path).exists():
                result = self.run_command(["--directory", path])

                # Should handle permission error gracefully
                # Either return error code or handle gracefully with warning
                assert result.returncode in [0, 1]
                break

    def test_invalid_json_files(self, test_data_dir):
        """Test handling of invalid JSON files."""
        # Create temporary directory with invalid JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            invalid_file = Path(temp_dir) / "invalid.json"
            invalid_file.write_text("{ invalid json without closing brace")

            result = self.run_command(["--directory", temp_dir])

            # Should handle gracefully - may skip invalid files or show specific error
            assert result.returncode in [0, 1]

            if result.returncode == 1:
                # If it returns error, should have meaningful message
                error_output = result.stderr + result.stdout
                assert any(keyword in error_output.lower() for keyword in ["json", "parse", "invalid", "エラー"])

    def test_missing_required_fields(self):
        """Test handling of JSON files with missing required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create JSON with missing required fields
            incomplete_file = Path(temp_dir) / "incomplete.json"
            incomplete_file.write_text('{"uuid": "test", "timestamp": "2025-05-09T10:00:00Z"}')

            result = self.run_command(["--directory", temp_dir])

            # Should handle gracefully
            assert result.returncode in [0, 1]

    def test_invalid_date_format(self, test_data_dir):
        """Test error handling for invalid date formats."""
        invalid_dates = [
            "invalid-date",
            "2025-13-01",  # Invalid month
            "2025-02-30",  # Invalid day
            "25-05-09",  # Wrong format
            "2025/05/09",  # Wrong separator
        ]

        for invalid_date in invalid_dates:
            result = self.run_command(["--directory", str(test_data_dir), "--start-date", invalid_date])

            # Should return error for invalid date (argparse returns 2)
            assert result.returncode == 2

            # Should provide meaningful error message
            error_output = result.stderr + result.stdout
            assert any(keyword in error_output.lower() for keyword in ["date", "format", "invalid", "エラー"])

    def test_invalid_granularity_option(self, test_data_dir):
        """Test error handling for invalid granularity option."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "invalid_granularity"])

        # Should return error for invalid granularity
        assert result.returncode == 2  # argparse typically returns 2 for invalid arguments

    def test_invalid_output_format(self, test_data_dir):
        """Test error handling for invalid output format."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "invalid_format"])

        # Should return error for invalid output format
        assert result.returncode == 2  # argparse typically returns 2 for invalid arguments

    def test_empty_directory_handling(self, test_data_dir):
        """Test handling of directory with no JSON files."""
        empty_dir = test_data_dir / "empty_dir"

        result = self.run_command(["--directory", str(empty_dir)])

        # Should handle gracefully with informative message
        assert result.returncode == 0

        # Should inform user that no files were found
        output = result.stdout + result.stderr
        assert any(keyword in output for keyword in ["No log files found", "no log files", "not found"])

    def test_date_range_no_results(self, test_data_dir):
        """Test handling when date range excludes all data."""
        result = self.run_command(["--directory", str(test_data_dir), "--start-date", "2030-01-01", "--end-date", "2030-12-31"])

        # Should handle gracefully
        assert result.returncode == 0

        # Should inform user that no entries were found in range
        output = result.stdout + result.stderr
        assert any(
            keyword in output
            for keyword in ["No log entries found in the specified date range", "no entries", "no data", "not found"]
        )

    @patch("claude_code_cost_collector.exchange.get_exchange_rate")
    def test_exchange_rate_failure_handling(self, mock_get_rate, test_data_dir):
        """Test handling of exchange rate API failures."""
        from claude_code_cost_collector.exchange import ExchangeRateError

        mock_get_rate.side_effect = ExchangeRateError("API connection failed")

        result = self.run_command(["--directory", str(test_data_dir)])

        # Should handle gracefully and fallback to USD
        assert result.returncode == 0

        # Should still show USD results
        assert "$" in result.stdout

    def test_corrupted_log_file_handling(self):
        """Test handling of corrupted log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various corrupted files
            corrupted_files = [
                ("binary.json", b"\x00\x01\x02\x03"),  # Binary data
                ("empty.json", ""),  # Empty file
                ("partial.json", '{"uuid": "test", "timestamp"'),  # Truncated
            ]

            for filename, content in corrupted_files:
                file_path = Path(temp_dir) / filename
                if isinstance(content, bytes):
                    file_path.write_bytes(content)
                else:
                    file_path.write_text(content)

            result = self.run_command(["--directory", temp_dir])

            # Should handle gracefully
            assert result.returncode in [0, 1]

    def test_very_large_numbers_handling(self):
        """Test handling of very large token counts and costs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create log with very large numbers
            large_numbers_log = {
                "uuid": "test-large",
                "sessionId": "session-large",
                "timestamp": "2025-05-09T10:00:00Z",
                "type": "assistant",
                "cwd": "/test",
                "originalCwd": "/test",
                "userType": "external",
                "version": "0.2.104",
                "isSidechain": False,
                "costUSD": 999999.999999,
                "durationMs": 1000,
                "message": {
                    "id": "msg_large",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-sonnet",
                    "usage": {"input_tokens": 999999999, "output_tokens": 999999999},
                },
                "model": "claude-3-sonnet",
            }

            import json

            large_file = Path(temp_dir) / "large.json"
            large_file.write_text(json.dumps(large_numbers_log))

            result = self.run_command(["--directory", temp_dir])

            # Should handle large numbers without overflow
            assert result.returncode == 0

    def test_malformed_timestamp_handling(self):
        """Test handling of malformed timestamps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create log with malformed timestamp
            malformed_log = {
                "uuid": "test-malformed",
                "sessionId": "session-malformed",
                "timestamp": "not-a-valid-timestamp",
                "type": "assistant",
                "cwd": "/test",
                "originalCwd": "/test",
                "userType": "external",
                "version": "0.2.104",
                "isSidechain": False,
                "costUSD": 0.01,
                "durationMs": 1000,
                "message": {
                    "id": "msg_malformed",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-sonnet",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
                "model": "claude-3-sonnet",
            }

            import json

            malformed_file = Path(temp_dir) / "malformed.json"
            malformed_file.write_text(json.dumps(malformed_log))

            result = self.run_command(["--directory", temp_dir])

            # Should handle gracefully
            assert result.returncode in [0, 1]

    def test_keyboard_interrupt_handling(self):
        """Test handling of keyboard interrupt (Ctrl+C)."""
        # This is difficult to test in automated testing, but we can at least
        # verify that the main function handles KeyboardInterrupt
        # The actual test would be done manually or with more complex setup
        pass

    def test_insufficient_memory_simulation(self):
        """Test behavior with very large datasets (memory stress test)."""
        # Create many log files to test memory handling
        with tempfile.TemporaryDirectory() as temp_dir:
            base_log = {
                "uuid": "test-{i}",
                "sessionId": "session-{i}",
                "timestamp": "2025-05-09T10:00:00Z",
                "type": "assistant",
                "cwd": "/test",
                "originalCwd": "/test",
                "userType": "external",
                "version": "0.2.104",
                "isSidechain": False,
                "costUSD": 0.01,
                "durationMs": 1000,
                "message": {
                    "id": "msg_{i}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-sonnet",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
                "model": "claude-3-sonnet",
            }

            import json

            # Create 100 log files (reasonable number for testing)
            for i in range(100):
                log_data = json.loads(json.dumps(base_log).replace("{i}", str(i)))
                log_file = Path(temp_dir) / f"log_{i}.json"
                log_file.write_text(json.dumps(log_data))

            result = self.run_command(["--directory", temp_dir])

            # Should handle large number of files gracefully
            assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__])
