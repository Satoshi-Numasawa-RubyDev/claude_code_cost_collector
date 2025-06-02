"""
Tests for the formatter module.
"""

import json

import pytest
import yaml

from claude_code_cost_collector.formatter import FormatterError, format_data, format_summary_statistics, get_supported_formats
from claude_code_cost_collector.models import create_sample_processed_log_entry


class TestFormatterCore:
    """Test the core formatter functionality."""

    def test_get_supported_formats(self):
        """Test that supported formats are returned correctly."""
        formats = get_supported_formats()
        expected = ["text", "json", "yaml", "csv"]
        assert formats == expected

    def test_unsupported_format_raises_error(self):
        """Test that unsupported formats raise FormatterError."""
        sample_data = [create_sample_processed_log_entry()]

        with pytest.raises(FormatterError) as exc_info:
            format_data(sample_data, output_format="xml")

        assert "Unsupported output format: xml" in str(exc_info.value)


class TestJSONFormatter:
    """Test JSON formatting functionality."""

    def test_json_format_individual_entries(self):
        """Test JSON formatting of individual log entries."""
        sample_entries = [create_sample_processed_log_entry()]
        result = format_data(sample_entries, output_format="json", granularity="all")

        # Parse JSON to validate structure
        parsed = json.loads(result)

        # Check top-level structure
        assert "metadata" in parsed
        assert "summary" in parsed
        assert "data" in parsed

        # Check metadata structure
        metadata = parsed["metadata"]
        assert metadata["granularity"] == "all"
        assert metadata["data_type"] == "individual_entries"
        assert metadata["total_entries"] == 1
        assert "format_version" in metadata
        assert "generated_at" in metadata

        # Check summary structure
        summary = parsed["summary"]
        assert "total_tokens" in summary
        assert "total_cost_usd" in summary
        assert summary["total_tokens"] == 24514  # From sample data
        assert abs(summary["total_cost_usd"] - 0.09317325) < 0.0001

        # Check data structure
        assert len(parsed["data"]) == 1
        entry_data = parsed["data"][0]
        assert "timestamp" in entry_data
        assert "project_name" in entry_data
        assert "cost_usd" in entry_data
        assert entry_data["project_name"] == "code_review"

    def test_json_format_aggregated_data(self):
        """Test JSON formatting of aggregated data."""
        aggregated_data = {
            "2025-05-09": {"total_input_tokens": 100, "total_output_tokens": 50, "total_tokens": 150, "total_cost_usd": 0.075}
        }

        result = format_data(aggregated_data, output_format="json", granularity="daily")
        parsed = json.loads(result)

        # Check metadata
        assert parsed["metadata"]["granularity"] == "daily"
        assert parsed["metadata"]["data_type"] == "aggregated"
        assert parsed["metadata"]["total_entries"] == 1

        # Check summary
        summary = parsed["summary"]
        assert summary["total_tokens"] == 150
        assert summary["total_cost_usd"] == 0.075

        # Check data
        assert "2025-05-09" in parsed["data"]
        assert parsed["data"]["2025-05-09"]["total_tokens"] == 150

    def test_json_format_empty_data(self):
        """Test JSON formatting with empty data."""
        result = format_data([], output_format="json", granularity="all")
        parsed = json.loads(result)

        assert parsed["metadata"]["total_entries"] == 0
        assert parsed["summary"]["total_tokens"] == 0
        assert parsed["summary"]["total_cost_usd"] == 0.0
        assert parsed["data"] == []

    def test_json_format_multiple_entries(self):
        """Test JSON formatting with multiple entries."""
        entries = [create_sample_processed_log_entry() for _ in range(3)]

        result = format_data(entries, output_format="json", granularity="all")
        parsed = json.loads(result)

        assert parsed["metadata"]["total_entries"] == 3
        assert len(parsed["data"]) == 3
        assert parsed["summary"]["total_tokens"] == 3 * 24514
        assert abs(parsed["summary"]["total_cost_usd"] - 3 * 0.09317325) < 0.0001


class TestCSVFormatter:
    """Test CSV formatting functionality."""

    def test_csv_format_individual_entries(self):
        """Test CSV formatting of individual log entries."""
        sample_entries = [create_sample_processed_log_entry()]
        result = format_data(sample_entries, output_format="csv", granularity="all")

        lines = result.strip().split("\n")

        # Check header
        header = lines[0]
        expected_headers = [
            "timestamp",
            "date",
            "project",
            "session_id",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "model",
        ]
        for expected_header in expected_headers:
            assert expected_header in header

        # Check data row
        assert len(lines) >= 2  # Header + at least one data row


class TestTextFormatter:
    """Test text formatting functionality (basic tests for stubs)."""

    def test_text_format_individual_entries(self):
        """Test text formatting of individual log entries."""
        sample_entries = [create_sample_processed_log_entry()]
        result = format_data(sample_entries, output_format="text", granularity="all")

        assert "Individual Log Entries" in result
        assert "code_review" in result  # project name from sample
        assert "$0.0932" in result  # cost from sample

    def test_text_format_aggregated_data(self):
        """Test text formatting of aggregated data."""
        aggregated_data = {
            "2025-05-09": {"total_input_tokens": 100, "total_output_tokens": 50, "total_tokens": 150, "total_cost_usd": 0.075}
        }

        result = format_data(aggregated_data, output_format="text", granularity="daily")

        assert "Daily Cost Summary" in result
        assert "2025-05-09" in result
        assert "150" in result
        assert "$0.0750" in result


class TestSummaryStatistics:
    """Test summary statistics functionality."""

    def test_summary_statistics_individual_entries(self):
        """Test summary statistics for individual entries."""
        entries = [create_sample_processed_log_entry(), create_sample_processed_log_entry()]
        stats = format_summary_statistics(entries)

        assert stats["total_entries"] == 2
        assert stats["total_tokens"] == 2 * 24514  # 2 entries with 24514 tokens each
        assert stats["total_cost_usd"] == 2 * 0.09317325
        assert stats["average_tokens_per_entry"] == 24514

    def test_summary_statistics_aggregated_data(self):
        """Test summary statistics for aggregated data."""
        aggregated_data = {
            "2025-05-09": {"total_tokens": 150, "total_cost_usd": 0.075},
            "2025-05-10": {"total_tokens": 200, "total_cost_usd": 0.100},
        }

        stats = format_summary_statistics(aggregated_data)

        assert stats["total_aggregation_groups"] == 2
        assert stats["total_tokens"] == 350
        assert stats["total_cost_usd"] == 0.175


class TestYAMLFormatter:
    """Test YAML formatting functionality."""

    def test_yaml_format_individual_entries(self):
        """Test YAML formatting of individual log entries."""
        sample_entries = [create_sample_processed_log_entry()]
        result = format_data(sample_entries, output_format="yaml", granularity="all")

        # Parse YAML to validate structure
        parsed = yaml.safe_load(result)

        assert "metadata" in parsed
        assert "data" in parsed
        assert parsed["metadata"]["granularity"] == "all"
        assert len(parsed["data"]) == 1

        # Check that the entry data is properly serialized
        entry_data = parsed["data"][0]
        assert "timestamp" in entry_data
        assert "project_name" in entry_data
        assert "cost_usd" in entry_data

    def test_yaml_format_aggregated_data(self):
        """Test YAML formatting of aggregated data."""
        aggregated_data = {"2025-05-09": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost_usd": 0.075}}

        result = format_data(aggregated_data, output_format="yaml", granularity="daily")
        parsed = yaml.safe_load(result)

        assert parsed["metadata"]["granularity"] == "daily"
        assert "2025-05-09" in parsed["data"]
        assert parsed["data"]["2025-05-09"]["total_tokens"] == 150


class TestLimitFunctionality:
    """Test the limit functionality for data restriction."""

    def test_limit_with_individual_entries(self):
        """Test limit functionality with individual log entries."""
        # Create multiple sample entries
        entries = []
        for i in range(5):
            entry = create_sample_processed_log_entry()
            entry.cost_usd = 0.1 + (i * 0.02)  # Different costs: 0.1, 0.12, 0.14, 0.16, 0.18
            entry.timestamp = entry.timestamp.replace(day=10 + i)  # Different dates
            entries.append(entry)

        # Test limit=2
        result = format_data(entries, output_format="json", granularity="all", limit=2)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 2
        assert parsed["metadata"]["total_entries"] == 2

    def test_limit_with_aggregated_data_daily(self):
        """Test limit functionality with daily aggregated data."""
        aggregated_data = {
            "2025-05-09": {"total_cost_usd": 0.075, "total_tokens": 100},
            "2025-05-10": {"total_cost_usd": 0.085, "total_tokens": 120},
            "2025-05-11": {"total_cost_usd": 0.095, "total_tokens": 140},
            "2025-06-01": {"total_cost_usd": 0.105, "total_tokens": 160},
        }

        # Test limit=2 with daily granularity (should get newest 2 dates)
        result = format_data(aggregated_data, output_format="json", granularity="daily", limit=2)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 2
        # Should have the newest dates: 2025-06-01 and 2025-05-11
        assert "2025-06-01" in parsed["data"]
        assert "2025-05-11" in parsed["data"]
        assert "2025-05-09" not in parsed["data"]
        assert "2025-05-10" not in parsed["data"]

    def test_limit_with_aggregated_data_monthly(self):
        """Test limit functionality with monthly aggregated data."""
        aggregated_data = {
            "2025-05": {"total_cost_usd": 0.075, "total_tokens": 100},
            "2025-06": {"total_cost_usd": 0.085, "total_tokens": 120},
            "2025-07": {"total_cost_usd": 0.095, "total_tokens": 140},
        }

        # Test limit=1 with monthly granularity (should get newest month)
        result = format_data(aggregated_data, output_format="json", granularity="monthly", limit=1)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 1
        assert "2025-07" in parsed["data"]
        assert "2025-05" not in parsed["data"]
        assert "2025-06" not in parsed["data"]

    def test_limit_with_project_granularity(self):
        """Test limit functionality with project granularity (cost-based sorting)."""
        aggregated_data = {
            "project_a": {"total_cost_usd": 0.050, "total_tokens": 100},
            "project_b": {"total_cost_usd": 0.150, "total_tokens": 300},
            "project_c": {"total_cost_usd": 0.100, "total_tokens": 200},
        }

        # Test limit=2 with project granularity (should get highest cost projects)
        result = format_data(aggregated_data, output_format="json", granularity="project", limit=2)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 2
        # Should have the highest cost projects: project_b and project_c
        assert "project_b" in parsed["data"]
        assert "project_c" in parsed["data"]
        assert "project_a" not in parsed["data"]

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns empty data."""
        entries = [create_sample_processed_log_entry()]

        result = format_data(entries, output_format="json", granularity="all", limit=0)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 0
        assert parsed["metadata"]["total_entries"] == 0

    def test_limit_negative_returns_empty(self):
        """Test that negative limit returns empty data."""
        entries = [create_sample_processed_log_entry()]

        result = format_data(entries, output_format="json", granularity="all", limit=-1)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 0
        assert parsed["metadata"]["total_entries"] == 0

    def test_limit_larger_than_data_size(self):
        """Test that limit larger than data size returns all data."""
        entries = [create_sample_processed_log_entry(), create_sample_processed_log_entry()]

        result = format_data(entries, output_format="json", granularity="all", limit=10)
        parsed = json.loads(result)

        assert len(parsed["data"]) == 2
        assert parsed["metadata"]["total_entries"] == 2

    def test_limit_with_text_format(self):
        """Test limit functionality with text output format."""
        aggregated_data = {
            "2025-05-09": {
                "total_cost_usd": 0.075,
                "total_tokens": 100,
                "total_input_tokens": 60,
                "total_output_tokens": 40,
                "total_cache_creation_tokens": 0,
                "total_cache_read_tokens": 0,
            },
            "2025-05-10": {
                "total_cost_usd": 0.085,
                "total_tokens": 120,
                "total_input_tokens": 70,
                "total_output_tokens": 50,
                "total_cache_creation_tokens": 0,
                "total_cache_read_tokens": 0,
            },
            "2025-06-01": {
                "total_cost_usd": 0.095,
                "total_tokens": 140,
                "total_input_tokens": 80,
                "total_output_tokens": 60,
                "total_cache_creation_tokens": 0,
                "total_cache_read_tokens": 0,
            },
        }

        result = format_data(aggregated_data, output_format="text", granularity="daily", limit=2)

        # Should contain newest 2 dates in the output
        assert "2025-06-01" in result
        assert "2025-05-10" in result
        assert "2025-05-09" not in result


if __name__ == "__main__":
    pytest.main([__file__])
