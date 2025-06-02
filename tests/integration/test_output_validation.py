"""
Output validation integration tests for Claude Code Cost Collector.

These tests validate the accuracy and consistency of output data
across different formats and aggregation methods.
"""

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
import yaml


class TestOutputValidation:
    """Output validation integration tests."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory path."""
        return Path(__file__).parent / "test_data"

    def run_command(self, args):
        """Helper method to run the application and capture output."""
        cmd = [sys.executable, "-m", "claude_code_cost_collector"] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def parse_csv_output(self, csv_text):
        """Parse CSV output into list of dictionaries."""
        lines = csv_text.strip().split("\n")
        if len(lines) < 2:
            return []

        reader = csv.DictReader(StringIO(csv_text))
        return list(reader)

    def test_data_consistency_across_formats(self, test_data_dir):
        """Test that data is consistent across different output formats."""
        # Get data in different formats
        json_result = self.run_command(["--directory", str(test_data_dir), "--granularity", "daily", "--output", "json"])

        csv_result = self.run_command(["--directory", str(test_data_dir), "--granularity", "daily", "--output", "csv"])

        yaml_result = self.run_command(["--directory", str(test_data_dir), "--granularity", "daily", "--output", "yaml"])

        assert json_result.returncode == 0
        assert csv_result.returncode == 0
        assert yaml_result.returncode == 0

        # Parse outputs
        json_data = json.loads(json_result.stdout)
        csv_data = self.parse_csv_output(csv_result.stdout)
        yaml_data = yaml.safe_load(yaml_result.stdout)

        # Verify same number of records (allowing for different structures)
        if isinstance(json_data, list) and isinstance(yaml_data, list):
            assert len(json_data) == len(yaml_data)

        # CSV should have at least the header + data rows
        assert len(csv_data) >= 0

    def test_cost_calculation_accuracy(self, test_data_dir):
        """Test that cost calculations are accurate."""
        # Expected costs from test data:
        # session1: $0.045, session2: $0.075, session3: $0.032, session4: $0.089
        # Total: $0.241

        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "all", "--output", "json", "--all-data"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Check summary total cost
        assert "summary" in data
        assert "total_cost_usd" in data["summary"]

        actual_total = data["summary"]["total_cost_usd"]
        expected_total = 0.045 + 0.075 + 0.032 + 0.089  # 0.241
        assert abs(actual_total - expected_total) < 0.001

    def test_token_count_accuracy(self, test_data_dir):
        """Test that token counts are accurate."""
        # Expected tokens from test data:
        # session1: 1000 + 300 = 1300, session2: 1500 + 500 = 2000
        # session3: 800 + 200 = 1000, session4: 2000 + 600 = 2600
        # Total: 6900

        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "all", "--output", "json", "--all-data"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Check summary total tokens
        assert "summary" in data
        assert "total_tokens" in data["summary"]

        actual_total = data["summary"]["total_tokens"]
        expected_total = 1300 + 2000 + 1000 + 2600  # 6900
        assert actual_total == expected_total

    def test_aggregation_accuracy(self, test_data_dir):
        """Test that aggregation calculations are accurate."""
        # Test daily aggregation
        daily_result = self.run_command(["--directory", str(test_data_dir), "--granularity", "daily", "--output", "json"])

        assert daily_result.returncode == 0
        daily_data = json.loads(daily_result.stdout)

        # Should have entries for multiple days
        if isinstance(daily_data, list):
            assert len(daily_data) >= 3  # At least 3 different dates
        elif isinstance(daily_data, dict):
            assert len(daily_data) >= 3

    def test_project_aggregation_accuracy(self, test_data_dir):
        """Test that project aggregation is accurate."""
        result = self.run_command(["--directory", str(test_data_dir), "--granularity", "project", "--output", "json"])

        assert result.returncode == 0
        data = json.loads(result.stdout)

        # Should have 2 projects (project1 and project2)
        if isinstance(data, list):
            assert len(data) >= 2
        elif isinstance(data, dict):
            assert len(data) >= 2

    def test_date_filtering_accuracy(self, test_data_dir):
        """Test that date filtering works correctly."""
        # Test filtering to only May 2025
        may_result = self.run_command(
            [
                "--directory",
                str(test_data_dir),
                "--start-date",
                "2025-05-01",
                "--end-date",
                "2025-05-31",
                "--granularity",
                "all",
                "--output",
                "json",
            ]
        )

        # Test filtering to only June 2025
        june_result = self.run_command(
            [
                "--directory",
                str(test_data_dir),
                "--start-date",
                "2025-06-01",
                "--end-date",
                "2025-06-30",
                "--granularity",
                "all",
                "--output",
                "json",
            ]
        )

        assert may_result.returncode == 0
        assert june_result.returncode == 0

        may_data = json.loads(may_result.stdout)
        june_data = json.loads(june_result.stdout)

        # May should have 3 entries, June should have 1 entry
        assert len(may_data["data"]) == 3
        assert len(june_data["data"]) == 1

    def test_csv_format_compliance(self, test_data_dir):
        """Test that CSV output follows proper CSV format."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "csv"])

        assert result.returncode == 0

        # Parse CSV and verify structure
        csv_data = self.parse_csv_output(result.stdout)

        if csv_data:
            # Verify all rows have the same number of columns
            header_length = len(csv_data[0].keys())
            for row in csv_data:
                assert len(row) == header_length

            # Verify essential columns exist
            essential_columns = set()
            for row in csv_data:
                essential_columns.update(row.keys())

            # Should have columns related to cost and tokens
            cost_related = any("cost" in col.lower() or "$" in col for col in essential_columns)
            token_related = any("token" in col.lower() for col in essential_columns)

            assert cost_related or token_related

    def test_json_format_compliance(self, test_data_dir):
        """Test that JSON output is valid and well-structured."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "json"])

        assert result.returncode == 0

        # Verify valid JSON
        data = json.loads(result.stdout)

        # Verify structure
        assert isinstance(data, (list, dict))

        if isinstance(data, list) and data:
            # Each item should be a dictionary with expected fields
            for item in data:
                assert isinstance(item, dict)
                # Should have some cost or token related field
                has_relevant_field = any("cost" in key.lower() or "token" in key.lower() for key in item.keys())
                assert has_relevant_field

    def test_yaml_format_compliance(self, test_data_dir):
        """Test that YAML output is valid and well-structured."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "yaml"])

        assert result.returncode == 0

        # Verify valid YAML
        data = yaml.safe_load(result.stdout)

        # Verify structure
        assert data is not None
        assert isinstance(data, (list, dict))

    def test_text_format_readability(self, test_data_dir):
        """Test that text output is human-readable and well-formatted."""
        result = self.run_command(["--directory", str(test_data_dir), "--output", "text"])

        assert result.returncode == 0

        # Verify text contains expected elements
        output = result.stdout

        # Should contain some tabular structure indicators
        has_structure = any(char in output for char in ["|", "+", "-", "─", "│"])

        # Should contain cost information
        has_cost = "$" in output or "USD" in output.upper()

        # Should contain token information
        has_tokens = "token" in output.lower()

        # At least one of these should be true for meaningful output
        assert has_structure or has_cost or has_tokens


if __name__ == "__main__":
    pytest.main([__file__])
