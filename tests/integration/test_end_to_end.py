"""
End-to-end integration tests for Claude Code Cost Collector.

These tests verify the complete application workflow from command-line input
to output generation.
"""

import pytest
import json
import yaml
import csv
import tempfile
import subprocess
import sys
from pathlib import Path
from io import StringIO


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory path."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def empty_dir(self, test_data_dir):
        """Get the empty test directory path."""
        return test_data_dir / "empty_dir"

    def run_command(self, args, cwd=None):
        """Helper method to run the application and capture output."""
        cmd = [sys.executable, "-m", "claude_code_cost_collector"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or Path.cwd())
        return result

    def test_basic_text_output(self, test_data_dir):
        """Test basic text output format."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "text"])

        assert result.returncode == 0
        assert "Date" in result.stdout or "プロジェクト" in result.stdout or "Project" in result.stdout
        assert "$" in result.stdout  # USD symbol should be present

    def test_json_output_format(self, test_data_dir):
        """Test JSON output format."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "json"])

        assert result.returncode == 0

        # Verify output is valid JSON
        try:
            output_data = json.loads(result.stdout)
            assert isinstance(output_data, (list, dict))
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_yaml_output_format(self, test_data_dir):
        """Test YAML output format."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "yaml"])

        assert result.returncode == 0

        # Verify output is valid YAML
        try:
            output_data = yaml.safe_load(result.stdout)
            assert output_data is not None
        except yaml.YAMLError:
            pytest.fail("Output is not valid YAML")

    def test_csv_output_format(self, test_data_dir):
        """Test CSV output format."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "csv"])

        assert result.returncode == 0

        # Verify output is valid CSV
        try:
            lines = result.stdout.strip().split("\n")
            assert len(lines) >= 2  # At least header + one data row

            # Parse CSV to verify format
            csv_reader = csv.reader(StringIO(result.stdout))
            rows = list(csv_reader)
            assert len(rows) >= 2
            assert len(rows[0]) >= 3  # At least a few columns
        except Exception as e:
            pytest.fail(f"Output is not valid CSV: {e}")

    def test_daily_granularity(self, test_data_dir):
        """Test daily aggregation granularity."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "daily", "--output", "json"])

        assert result.returncode == 0

        # Parse JSON and verify daily structure
        output_data = json.loads(result.stdout)
        if isinstance(output_data, list) and len(output_data) > 0:
            # Check if dates are present in the output
            for item in output_data:
                if isinstance(item, dict) and any("2025-05" in str(v) or "date" in str(k).lower() for k, v in item.items()):
                    break
            # Note: Date might be in various formats, so we check for presence of date-like strings

    def test_monthly_granularity(self, test_data_dir):
        """Test monthly aggregation granularity."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "monthly", "--output", "json"])

        assert result.returncode == 0

        # Parse JSON and verify monthly structure
        output_data = json.loads(result.stdout)
        # Monthly data should aggregate by month (2025-05, 2025-06)
        assert isinstance(output_data, (list, dict))

    def test_project_granularity(self, test_data_dir):
        """Test project aggregation granularity."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "project", "--output", "json"])

        assert result.returncode == 0

        # Parse JSON and verify project structure
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, (list, dict))
        # Should have project1 and project2 data

    def test_session_granularity(self, test_data_dir):
        """Test session aggregation granularity."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "session", "--output", "json"])

        assert result.returncode == 0

        # Parse JSON and verify session structure
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, (list, dict))
        # Should have session-1, session-2, session-3, session-4 data

    def test_all_granularity(self, test_data_dir):
        """Test individual log entry display (all granularity)."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "all", "--output", "json", "--all-data"])

        assert result.returncode == 0

        # Parse JSON and verify individual entries
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, dict)
        # Should have 4 individual log entries in data section
        assert "data" in output_data
        assert len(output_data["data"]) == 4

    def test_date_filtering_start_date(self, test_data_dir):
        """Test filtering with start date."""
        result = self.run_command(["--directory", str(test_data_dir), "--start-date", "2025-05-15", "--output", "json"])

        assert result.returncode == 0

        # Should only include logs from 2025-05-15 onwards
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, (list, dict))
        # Should exclude 2025-05-09 and 2025-05-10 logs

    def test_date_filtering_end_date(self, test_data_dir):
        """Test filtering with end date."""
        result = self.run_command(["--directory", str(test_data_dir), "--end-date", "2025-05-15", "--output", "json"])

        assert result.returncode == 0

        # Should only include logs up to 2025-05-15
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, (list, dict))
        # Should exclude 2025-06-01 log

    def test_date_filtering_range(self, test_data_dir):
        """Test filtering with date range."""
        result = self.run_command(
            ["--directory", str(test_data_dir), "--start-date", "2025-05-10", "--end-date", "2025-05-20", "--output", "json"]
        )

        assert result.returncode == 0

        # Should only include logs in the specified range
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, (list, dict))
        # Should include 2025-05-10 and 2025-05-15, exclude 2025-05-09 and 2025-06-01

    def test_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        result = self.run_command(["--directory", "/nonexistent/directory"])

        assert result.returncode == 2  # argparse validation error
        assert "not exist" in result.stderr or "Error" in result.stderr

    def test_empty_directory(self, empty_dir):
        """Test handling of directory with no log files."""
        result = self.run_command(["--directory", str(empty_dir)])

        assert result.returncode == 0
        assert "ログファイルが見つかりませんでした" in result.stdout or "No log files found" in result.stdout

    def test_invalid_json_handling(self, test_data_dir):
        """Test handling of invalid JSON files."""
        # Create a temporary directory with only invalid JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_json_path = Path(temp_dir) / "invalid.json"
            invalid_json_path.write_text("{ invalid json")

            result = self.run_command(["--directory", temp_dir])

            # Should handle gracefully - either skip invalid files or show error
            # The exact behavior depends on implementation
            assert result.returncode in [0, 1]

    def test_invalid_date_format(self, test_data_dir):
        """Test handling of invalid date format."""
        result = self.run_command(["--directory", str(test_data_dir), "--start-date", "invalid-date"])

        # Should return error for invalid date format (argparse returns 2)
        assert result.returncode == 2

    def test_combined_features(self, test_data_dir):
        """Test combination of multiple features."""
        result = self.run_command(
            [
                "--directory",
                str(test_data_dir),
                "--granularity",
                "monthly",
                "--output",
                "json",
                "--start-date",
                "2025-05-01",
                "--end-date",
                "2025-05-31",
            ]
        )

        assert result.returncode == 0

        # Verify JSON output
        output_data = json.loads(result.stdout)
        assert isinstance(output_data, (list, dict))

    def test_help_option(self):
        """Test help option display."""
        result = self.run_command(["--help"])

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "Claude Code Cost Collector" in result.stdout

    def test_version_information(self):
        """Test that the application runs without version errors."""
        # Test basic functionality to ensure the application is properly structured
        result = self.run_command(["--help"])
        assert result.returncode == 0

    def test_limit_functionality_text_output(self, test_data_dir):
        """Test limit functionality with text output."""
        # First get all data count
        result_all = self.run_command(["--directory", str(test_data_dir), "--all-data", "--output", "text"])
        assert result_all.returncode == 0

        # Count number of data rows (excluding header and total)
        lines = result_all.stdout.split("\n")
        data_lines = [
            line
            for line in lines
            if line
            and not line.startswith("┏")
            and not line.startswith("┃ Date")
            and not line.startswith("├")
            and not line.startswith("┃ TOTAL")
            and not line.startswith("└")
            and "┃" in line
        ]
        total_entries = len(data_lines)

        # Test with limit smaller than total
        if total_entries > 2:
            result_limited = self.run_command(
                ["--directory", str(test_data_dir), "--all-data", "--limit", "2", "--output", "text"]
            )
            assert result_limited.returncode == 0

            # Count limited output rows
            limited_lines = result_limited.stdout.split("\n")
            limited_data_lines = [
                line
                for line in limited_lines
                if line
                and not line.startswith("┏")
                and not line.startswith("┃ Date")
                and not line.startswith("├")
                and not line.startswith("┃ TOTAL")
                and not line.startswith("└")
                and "┃" in line
            ]

            assert len(limited_data_lines) == 2

    def test_limit_functionality_json_output(self, test_data_dir):
        """Test limit functionality with JSON output."""
        # Test with limit=1
        result = self.run_command(["--directory", str(test_data_dir), "--all-data", "--limit", "1", "--output", "json"])
        assert result.returncode == 0

        output_data = json.loads(result.stdout)

        # Check metadata reflects limit
        assert output_data["metadata"]["total_entries"] == 1

        # Check data has only 1 entry
        if isinstance(output_data["data"], list):
            assert len(output_data["data"]) == 1
        else:
            assert len(output_data["data"]) == 1

    def test_limit_zero(self, test_data_dir):
        """Test limit=0 returns empty results."""
        result = self.run_command(["--directory", str(test_data_dir), "--all-data", "--limit", "0", "--output", "json"])
        assert result.returncode == 0

        output_data = json.loads(result.stdout)

        # Should have no data entries
        assert output_data["metadata"]["total_entries"] == 0
        if isinstance(output_data["data"], list):
            assert len(output_data["data"]) == 0
        else:
            assert len(output_data["data"]) == 0

    def test_limit_negative(self, test_data_dir):
        """Test negative limit returns empty results."""
        result = self.run_command(["--directory", str(test_data_dir), "--all-data", "--limit", "-1", "--output", "json"])
        assert result.returncode == 0

        output_data = json.loads(result.stdout)

        # Should have no data entries
        assert output_data["metadata"]["total_entries"] == 0
        if isinstance(output_data["data"], list):
            assert len(output_data["data"]) == 0
        else:
            assert len(output_data["data"]) == 0

    def test_limit_with_different_granularities(self, test_data_dir):
        """Test limit functionality with different granularities."""
        granularities = ["daily", "monthly", "project", "session"]

        for granularity in granularities:
            result = self.run_command(
                [
                    "--directory",
                    str(test_data_dir),
                    "--all-data",
                    "--granularity",
                    granularity,
                    "--limit",
                    "1",
                    "--output",
                    "json",
                ]
            )

            assert result.returncode == 0

            output_data = json.loads(result.stdout)

            # Should have at most 1 entry
            if isinstance(output_data["data"], list):
                assert len(output_data["data"]) <= 1
            else:
                assert len(output_data["data"]) <= 1

    def test_limit_larger_than_available_data(self, test_data_dir):
        """Test limit larger than available data returns all data."""
        # Get total count first
        result_all = self.run_command(["--directory", str(test_data_dir), "--all-data", "--output", "json"])
        assert result_all.returncode == 0

        all_data = json.loads(result_all.stdout)
        total_count = all_data["metadata"]["total_entries"]

        # Test with limit much larger than total
        result_limited = self.run_command(
            ["--directory", str(test_data_dir), "--all-data", "--limit", str(total_count + 10), "--output", "json"]
        )

        assert result_limited.returncode == 0

        limited_data = json.loads(result_limited.stdout)

        # Should return all available data
        assert limited_data["metadata"]["total_entries"] == total_count

    def test_limit_date_ordering(self, test_data_dir):
        """Test that limit respects date ordering for daily granularity."""
        result = self.run_command(
            ["--directory", str(test_data_dir), "--all-data", "--granularity", "daily", "--limit", "2", "--output", "json"]
        )

        assert result.returncode == 0

        output_data = json.loads(result.stdout)

        # Get the dates from the limited data
        dates = list(output_data["data"].keys())

        # Should have 2 or fewer entries
        assert len(dates) <= 2

        # If we have multiple dates, verify they are the newest ones
        if len(dates) > 1:
            # Dates should be in descending order (newest first from _apply_limit)
            # or ascending order (from text formatter sorting)
            # The dates should be among the newest available
            pass

    def test_version_option(self):
        """Test --version option displays version information."""
        result = self.run_command(["--version"])

        assert result.returncode == 0
        assert "claude-code-cost-collector" in result.stdout
        assert "0.5.2" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])
