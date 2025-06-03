"""
Tests for the main module.
"""

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_code_cost_collector.main import (
    filter_entries_by_date_range,
    main,
    setup_logging,
)
from claude_code_cost_collector.models import ProcessedLogEntry


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging()

            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs["level"] == logging.WARNING
            assert kwargs["format"] == "%(message)s"

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging(verbose=True)

            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs["level"] == logging.DEBUG


class TestFilterEntriesByDateRange:
    """Test filter_entries_by_date_range function."""

    def create_sample_entry(self, date_str: str) -> ProcessedLogEntry:
        """Create a sample log entry for testing."""
        return ProcessedLogEntry(
            timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
            date_str=date_str,
            month_str=date_str[:7],
            project_name="test_project",
            session_id="test_session",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            model="claude-3",
        )

    def test_filter_no_date_range(self):
        """Test filtering without date range returns all entries."""
        entries = [
            self.create_sample_entry("2025-05-01"),
            self.create_sample_entry("2025-05-15"),
            self.create_sample_entry("2025-05-30"),
        ]

        result = filter_entries_by_date_range(entries)
        assert len(result) == 3
        assert result == entries

    def test_filter_with_start_date(self):
        """Test filtering with start date only."""
        entries = [
            self.create_sample_entry("2025-05-01"),
            self.create_sample_entry("2025-05-15"),
            self.create_sample_entry("2025-05-30"),
        ]

        result = filter_entries_by_date_range(entries, start_date="2025-05-15")
        assert len(result) == 2
        assert result[0].date_str == "2025-05-15"
        assert result[1].date_str == "2025-05-30"

    def test_filter_with_end_date(self):
        """Test filtering with end date only."""
        entries = [
            self.create_sample_entry("2025-05-01"),
            self.create_sample_entry("2025-05-15"),
            self.create_sample_entry("2025-05-30"),
        ]

        result = filter_entries_by_date_range(entries, end_date="2025-05-15")
        assert len(result) == 2
        assert result[0].date_str == "2025-05-01"
        assert result[1].date_str == "2025-05-15"

    def test_filter_with_both_dates(self):
        """Test filtering with both start and end dates."""
        entries = [
            self.create_sample_entry("2025-05-01"),
            self.create_sample_entry("2025-05-15"),
            self.create_sample_entry("2025-05-30"),
        ]

        result = filter_entries_by_date_range(entries, start_date="2025-05-10", end_date="2025-05-20")
        assert len(result) == 1
        assert result[0].date_str == "2025-05-15"

    def test_filter_empty_result(self):
        """Test filtering that results in empty list."""
        entries = [
            self.create_sample_entry("2025-05-01"),
            self.create_sample_entry("2025-05-15"),
        ]

        result = filter_entries_by_date_range(entries, start_date="2025-06-01", end_date="2025-06-30")
        assert len(result) == 0

    def test_filter_empty_entries(self):
        """Test filtering empty entries list."""
        result = filter_entries_by_date_range([], start_date="2025-05-01")
        assert len(result) == 0


class TestMainFunction:
    """Test the main function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments."""
        args = Mock()
        args.directory = "/test/directory"
        args.granularity = "daily"
        args.output = "text"
        args.start_date = None
        args.end_date = None
        args.config = None
        args.exchange_rate_api_key = None
        return args

    @pytest.fixture
    def sample_log_entry(self):
        """Create a sample log entry."""
        return ProcessedLogEntry(
            timestamp=datetime(2025, 5, 9, 12, 0, 0),
            date_str="2025-05-09",
            month_str="2025-05",
            project_name="test_project",
            session_id="test_session",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            model="claude-3",
        )

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("claude_code_cost_collector.main.aggregate_data")
    @patch("claude_code_cost_collector.main.format_data")
    @patch("builtins.print")
    def test_main_success_basic_flow(
        self,
        mock_print,
        mock_format,
        mock_aggregate,
        mock_parse_files,
        mock_collect,
        mock_validate,
        mock_load_config,
        mock_parse_args,
        mock_args,
        sample_log_entry,
    ):
        """Test successful main function execution."""
        # Setup mocks
        mock_parse_args.return_value = mock_args
        mock_load_config.return_value = {"test": "config"}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = [sample_log_entry]
        mock_aggregate.return_value = {"2025-05-09": {"total_cost_usd": 0.01}}
        mock_format.return_value = "Test output"

        result = main()

        assert result == 0
        mock_print.assert_called_with("Test output")

    @patch("claude_code_cost_collector.main.parse_arguments")
    def test_main_config_error(self, mock_parse_args, mock_args):
        """Test main function with config error."""
        from claude_code_cost_collector.config import ConfigError

        mock_parse_args.return_value = mock_args

        with patch("claude_code_cost_collector.main.load_config") as mock_load:
            mock_load.side_effect = ConfigError("Config error")

            result = main()

            assert result == 1

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("builtins.print")
    def test_main_no_log_files(self, mock_print, mock_collect, mock_validate, mock_load, mock_parse_args, mock_args):
        """Test main function when no log files are found."""
        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = []

        result = main()

        assert result == 0
        mock_print.assert_called_with("No log files found.")

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    def test_main_file_not_found_error(self, mock_collect, mock_validate, mock_load, mock_parse_args, mock_args):
        """Test main function with file not found error."""
        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.side_effect = FileNotFoundError("Directory not found")

        result = main()

        assert result == 1

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("builtins.print")
    def test_main_no_parseable_entries(
        self, mock_print, mock_parse_files, mock_collect, mock_validate, mock_load, mock_parse_args, mock_args
    ):
        """Test main function when no parseable entries are found."""
        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = []

        result = main()

        assert result == 0
        mock_print.assert_called_with("No parseable log entries found.")

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    def test_main_log_parse_error(self, mock_parse_files, mock_collect, mock_validate, mock_load, mock_parse_args, mock_args):
        """Test main function with log parse error."""
        from claude_code_cost_collector.parser import LogParseError

        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.side_effect = LogParseError("Parse error")

        result = main()

        assert result == 1

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("claude_code_cost_collector.main.filter_entries_by_date_range")
    @patch("builtins.print")
    def test_main_date_filtering_no_results(
        self,
        mock_print,
        mock_filter,
        mock_parse_files,
        mock_collect,
        mock_validate,
        mock_load,
        mock_parse_args,
        mock_args,
        sample_log_entry,
    ):
        """Test main function when date filtering returns no results."""
        mock_args.start_date = "2025-06-01"
        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = [sample_log_entry]
        mock_filter.return_value = []

        result = main()

        assert result == 0
        mock_print.assert_called_with("No log entries found in the specified date range.")

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("claude_code_cost_collector.main.aggregate_data")
    def test_main_aggregation_error(
        self,
        mock_aggregate,
        mock_parse_files,
        mock_collect,
        mock_validate,
        mock_load,
        mock_parse_args,
        mock_args,
        sample_log_entry,
    ):
        """Test main function with aggregation error."""
        from claude_code_cost_collector.aggregator import AggregationError

        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = [sample_log_entry]
        mock_aggregate.side_effect = AggregationError("Aggregation failed")

        result = main()

        assert result == 1

    @patch("claude_code_cost_collector.main.parse_arguments")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("claude_code_cost_collector.main.aggregate_data")
    @patch("claude_code_cost_collector.main.format_data")
    def test_main_formatter_error(
        self,
        mock_format,
        mock_aggregate,
        mock_parse_files,
        mock_collect,
        mock_validate,
        mock_load,
        mock_parse_args,
        mock_args,
        sample_log_entry,
    ):
        """Test main function with formatter error."""
        from claude_code_cost_collector.formatter import FormatterError

        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = [sample_log_entry]
        mock_aggregate.return_value = {"2025-05-09": {"total_cost_usd": 0.01}}
        mock_format.side_effect = FormatterError("Format failed")

        result = main()

        assert result == 1

    @patch("claude_code_cost_collector.main.parse_arguments")
    def test_main_keyboard_interrupt(self, mock_parse_args):
        """Test main function with keyboard interrupt."""
        mock_parse_args.side_effect = KeyboardInterrupt()

        result = main()

        assert result == 1

    @patch("claude_code_cost_collector.main.parse_arguments")
    def test_main_unexpected_exception(self, mock_parse_args):
        """Test main function with unexpected exception."""
        mock_parse_args.side_effect = RuntimeError("Unexpected error")

        result = main()

        assert result == 1

    @patch("claude_code_cost_collector.main.format_data")
    @patch("claude_code_cost_collector.main.aggregate_data")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.parse_arguments")
    def test_main_with_new_sort_interface(
        self,
        mock_parse_args,
        mock_load,
        mock_validate,
        mock_collect,
        mock_parse_files,
        mock_aggregate,
        mock_format,
        sample_log_entry,
        mock_args,
    ):
        """Test main function processes new sort arguments correctly."""
        from pathlib import Path

        mock_args.sort = "desc"  # New interface
        mock_args.sort_field = "date"  # New interface
        mock_args.all_data = True
        mock_args.timezone = None
        mock_args.currency = None
        mock_args.exchange_rate_api_key = None
        mock_args.limit = None
        mock_args.debug = False

        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = [sample_log_entry]
        mock_aggregate.return_value = {"2025-05-09": {"total_cost_usd": 0.01}}
        mock_format.return_value = "formatted output"

        result = main()

        assert result == 0

        # Verify that new sort arguments are passed correctly to aggregate_data
        mock_aggregate.assert_called_once()
        args, kwargs = mock_aggregate.call_args
        assert kwargs["sort_by"] == "date"
        assert kwargs["sort_desc"] is True  # desc = True

        # Verify that new sort arguments are passed correctly to format_data
        mock_format.assert_called_once()
        args, kwargs = mock_format.call_args
        assert kwargs["sort_by"] == "date"
        assert kwargs["sort_desc"] is True

    @patch("claude_code_cost_collector.main.format_data")
    @patch("claude_code_cost_collector.main.aggregate_data")
    @patch("claude_code_cost_collector.main.parse_multiple_log_files")
    @patch("claude_code_cost_collector.main.collect_log_files")
    @patch("claude_code_cost_collector.main.validate_config")
    @patch("claude_code_cost_collector.main.load_config")
    @patch("claude_code_cost_collector.main.parse_arguments")
    def test_main_with_sort_field_none(
        self,
        mock_parse_args,
        mock_load,
        mock_validate,
        mock_collect,
        mock_parse_files,
        mock_aggregate,
        mock_format,
        sample_log_entry,
        mock_args,
    ):
        """Test main function when sort_field is None."""
        from pathlib import Path

        mock_args.sort = "asc"  # Specified
        mock_args.sort_field = None  # Not specified
        mock_args.all_data = True
        mock_args.timezone = None
        mock_args.currency = None
        mock_args.exchange_rate_api_key = None
        mock_args.limit = None
        mock_args.debug = False

        mock_parse_args.return_value = mock_args
        mock_load.return_value = {}
        mock_collect.return_value = [Path("/test/log1.json")]
        mock_parse_files.return_value = [sample_log_entry]
        mock_aggregate.return_value = {"2025-05-09": {"total_cost_usd": 0.01}}
        mock_format.return_value = "formatted output"

        result = main()

        assert result == 0

        # Verify that None sort_field is passed through when CLI processing is bypassed
        mock_aggregate.assert_called_once()
        args, kwargs = mock_aggregate.call_args
        assert kwargs["sort_by"] is None  # Mock bypasses CLI processing
        assert kwargs["sort_desc"] is False  # asc = False


if __name__ == "__main__":
    pytest.main([__file__])
