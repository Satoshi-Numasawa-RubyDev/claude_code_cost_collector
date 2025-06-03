"""
Tests for the CLI module.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_code_cost_collector.cli import (
    create_parser,
    parse_arguments,
    validate_date,
    validate_directory,
)


class TestValidateDate:
    """Test date validation function."""

    def test_valid_date(self):
        """Test validation of valid date strings."""
        valid_dates = [
            "2025-01-01",
            "2025-12-31",
            "2024-02-29",  # Leap year
            "2025-05-15",
        ]

        for date_str in valid_dates:
            result = validate_date(date_str)
            assert result == date_str

    def test_invalid_date_format(self):
        """Test validation of invalid date formats."""
        invalid_formats = [
            "25-01-01",  # Wrong year format
            "2025-1-01",  # Missing leading zero in month
            "2025-01-1",  # Missing leading zero in day
            "2025/01/01",  # Wrong separator
            "2025-01",  # Missing day
            "01-01-2025",  # Wrong order
            "not-a-date",  # Non-date string
            "",  # Empty string
        ]

        for date_str in invalid_formats:
            with pytest.raises(argparse.ArgumentTypeError):
                validate_date(date_str)

    def test_invalid_date_values(self):
        """Test validation of invalid date values."""
        invalid_dates = [
            "2025-13-01",  # Invalid month
            "2025-02-30",  # Invalid day for February
            "2025-04-31",  # Invalid day for April
            "2025-00-01",  # Zero month
            "2025-01-00",  # Zero day
            "2025-01-32",  # Day too high
        ]

        for date_str in invalid_dates:
            with pytest.raises(argparse.ArgumentTypeError):
                validate_date(date_str)

    def test_leap_year_handling(self):
        """Test leap year date handling."""
        # Valid leap year date
        assert validate_date("2024-02-29") == "2024-02-29"

        # Invalid non-leap year date
        with pytest.raises(argparse.ArgumentTypeError):
            validate_date("2025-02-29")


class TestValidateDirectory:
    """Test directory validation function."""

    def test_valid_directory(self):
        """Test validation of valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_directory(temp_dir)
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_dir()

    def test_nonexistent_directory(self):
        """Test validation of non-existent directory."""
        with pytest.raises(argparse.ArgumentTypeError, match="Directory does not exist"):
            validate_directory("/nonexistent/directory")

    def test_file_instead_of_directory(self):
        """Test validation when path points to file instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(argparse.ArgumentTypeError, match="Path is not a directory"):
                validate_directory(temp_file.name)

    def test_home_directory_expansion(self):
        """Test that home directory is properly expanded."""
        with patch("pathlib.Path.expanduser") as mock_expanduser:
            with patch("pathlib.Path.resolve") as mock_resolve:
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.is_dir", return_value=True):
                        mock_path = Path("/expanded/path")
                        mock_expanduser.return_value.resolve.return_value = mock_path
                        mock_resolve.return_value = mock_path

                        validate_directory("~/test")
                        mock_expanduser.assert_called_once()


class TestCreateParser:
    """Test argument parser creation."""

    def test_parser_creation(self):
        """Test that parser is created correctly."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "claude-code-cost-collector"
        assert "Claude API usage costs" in parser.description

    def test_parser_arguments(self):
        """Test that parser has all expected arguments."""
        parser = create_parser()

        # Get all action destinations
        actions = {action.dest: action for action in parser._actions}

        # Check required arguments exist
        expected_args = ["directory", "granularity", "output", "start_date", "end_date", "config", "exchange_rate_api_key"]

        for arg_name in expected_args:
            assert arg_name in actions

    def test_granularity_choices(self):
        """Test granularity argument choices."""
        parser = create_parser()

        # Find granularity action
        granularity_action = None
        for action in parser._actions:
            if action.dest == "granularity":
                granularity_action = action
                break

        assert granularity_action is not None
        expected_choices = ["daily", "monthly", "project", "session", "all"]
        assert granularity_action.choices == expected_choices
        assert granularity_action.default == "daily"

    def test_output_choices(self):
        """Test output format argument choices."""
        parser = create_parser()

        # Find output action
        output_action = None
        for action in parser._actions:
            if action.dest == "output":
                output_action = action
                break

        assert output_action is not None
        expected_choices = ["text", "json", "yaml", "csv"]
        assert output_action.choices == expected_choices
        assert output_action.default == "text"


class TestParseArguments:
    """Test argument parsing function."""

    def test_parse_default_arguments(self):
        """Test parsing with default arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parse_arguments(["-d", temp_dir])

            assert args.directory == Path(temp_dir).resolve()
            assert args.granularity == "daily"
            assert args.output == "text"
            assert args.start_date is None
            assert args.end_date is None
            assert args.config is None
            assert args.exchange_rate_api_key is None

    def test_parse_all_arguments(self):
        """Test parsing with all arguments provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix=".json") as config_file:
                args_list = [
                    "-d",
                    temp_dir,
                    "-g",
                    "monthly",
                    "-o",
                    "json",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-12-31",
                    "--config",
                    config_file.name,
                    "--exchange-rate-api-key",
                    "test_key",
                ]

                args = parse_arguments(args_list)

                assert args.directory == Path(temp_dir).resolve()
                assert args.granularity == "monthly"
                assert args.output == "json"
                assert args.start_date == "2025-01-01"
                assert args.end_date == "2025-12-31"
                assert args.config == Path(config_file.name).resolve()
                assert args.exchange_rate_api_key == "test_key"

    def test_parse_short_arguments(self):
        """Test parsing with short argument forms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parse_arguments(["-d", temp_dir, "-g", "project", "-o", "csv"])

            assert args.directory == Path(temp_dir).resolve()
            assert args.granularity == "project"
            assert args.output == "csv"

    def test_invalid_granularity(self):
        """Test parsing with invalid granularity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SystemExit):
                parse_arguments(["-d", temp_dir, "-g", "invalid"])

    def test_invalid_output_format(self):
        """Test parsing with invalid output format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SystemExit):
                parse_arguments(["-d", temp_dir, "-o", "invalid"])

    def test_invalid_date_format(self):
        """Test parsing with invalid date format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SystemExit):
                parse_arguments(["-d", temp_dir, "--start-date", "invalid-date"])

    def test_start_date_after_end_date(self):
        """Test validation when start date is after end date."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SystemExit):
                parse_arguments(["-d", temp_dir, "--start-date", "2025-12-31", "--end-date", "2025-01-01"])

    def test_equal_start_and_end_dates(self):
        """Test that equal start and end dates are valid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parse_arguments(["-d", temp_dir, "--start-date", "2025-05-15", "--end-date", "2025-05-15"])

            assert args.start_date == "2025-05-15"
            assert args.end_date == "2025-05-15"

    def test_nonexistent_config_file(self):
        """Test parsing with non-existent config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(SystemExit):
                parse_arguments(["-d", temp_dir, "--config", "/nonexistent/config.json"])

    def test_config_file_expansion(self):
        """Test that config file path is properly expanded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix=".json") as config_file:
                args = parse_arguments(["-d", temp_dir, "--config", config_file.name])

                assert args.config == Path(config_file.name).resolve()

    def test_nonexistent_directory(self):
        """Test parsing with non-existent directory."""
        with pytest.raises(SystemExit):
            parse_arguments(["-d", "/nonexistent/directory"])

    def test_help_argument(self):
        """Test that help argument works."""
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(["--help"])

        # Help should exit with code 0
        assert exc_info.value.code == 0

    def test_version_argument(self):
        """Test that version argument works."""
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(["--version"])

        # Version should exit with code 0
        assert exc_info.value.code == 0

    def test_parsing_with_no_args(self):
        """Test parsing with no arguments (should use defaults)."""
        # Mock the default directory to exist
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            args = parse_arguments([])

            assert args.granularity == "daily"
            assert args.output == "text"

    def test_parsing_none_args(self):
        """Test parsing when args parameter is None."""
        with patch("sys.argv", ["claude-code-cost-collector"]):
            with patch("claude_code_cost_collector.cli.Path") as mock_path:
                mock_path.return_value.expanduser.return_value.exists.return_value = True
                mock_path.return_value.expanduser.return_value.is_dir.return_value = True
                mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

                args = parse_arguments(None)

                assert args.granularity == "daily"
                assert args.output == "text"

    def test_sort_argument_choices(self):
        """Test that sort argument accepts valid choices."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            # Test all valid sort order choices
            valid_sorts = ["asc", "desc"]
            for sort_choice in valid_sorts:
                args = parse_arguments(["--sort", sort_choice])
                assert args.sort == sort_choice

    def test_sort_argument_invalid_choice(self):
        """Test that sort argument rejects invalid choices."""
        with pytest.raises(SystemExit):
            parse_arguments(["--sort", "invalid"])

    def test_sort_field_argument_choices(self):
        """Test that sort-field argument accepts valid choices."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            # Test all valid sort field choices
            valid_fields = ["input", "output", "total", "cost", "date"]
            for field_choice in valid_fields:
                args = parse_arguments(["--sort-field", field_choice])
                assert args.sort_field == field_choice

    def test_sort_field_argument_invalid_choice(self):
        """Test that sort-field argument rejects invalid choices."""
        with pytest.raises(SystemExit):
            parse_arguments(["--sort-field", "invalid"])

    def test_sort_order_argument(self):
        """Test that sort argument works for order specification."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            # Test without sort argument (should use default)
            args = parse_arguments([])
            assert args.sort == "desc"

            # Test with asc sort
            args = parse_arguments(["--sort", "asc"])
            assert args.sort == "asc"

            # Test with desc sort
            args = parse_arguments(["--sort", "desc"])
            assert args.sort == "desc"

    def test_sort_with_sort_field(self):
        """Test combining sort and sort-field arguments."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            args = parse_arguments(["--sort", "desc", "--sort-field", "cost"])
            assert args.sort == "desc"
            assert args.sort_field == "cost"

    def test_sort_arguments_defaults(self):
        """Test that sort arguments use correct defaults."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            args = parse_arguments([])
            assert args.sort == "desc"  # Default is now desc
            assert args.sort_field == "date"  # Defaults to 'date' when not specified

    def test_sort_only_specified(self):
        """Test behavior when only --sort is specified."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            args = parse_arguments(["--sort", "asc"])
            assert args.sort == "asc"
            assert args.sort_field == "date"  # Should default to 'date' when sort is specified

    def test_sort_field_only_specified(self):
        """Test behavior when only --sort-field is specified."""
        with patch("claude_code_cost_collector.cli.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            mock_path.return_value.expanduser.return_value.is_dir.return_value = True
            mock_path.return_value.expanduser.return_value.resolve.return_value = Path("/mocked/path")

            args = parse_arguments(["--sort-field", "cost"])
            assert args.sort == "desc"  # Should use default
            assert args.sort_field == "cost"

    def test_help_message_includes_new_options(self):
        """Test that help message includes new sort options."""
        parser = create_parser()
        help_text = parser.format_help()

        # Check that new options are mentioned in help
        assert "--sort ORDER" in help_text or "--sort" in help_text
        assert "--sort-field FIELD" in help_text or "--sort-field" in help_text
        assert "asc" in help_text and "desc" in help_text
        assert "date" in help_text  # New date field option

        # Check that old option is not mentioned
        assert "--sort-desc" not in help_text


if __name__ == "__main__":
    pytest.main([__file__])
