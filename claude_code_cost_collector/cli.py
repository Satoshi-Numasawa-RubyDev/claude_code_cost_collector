"""
Command Line Interface module for Claude Code Cost Collector.

This module provides argument parsing functionality using argparse to handle
command-line options for cost aggregation, output formatting, and other features.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import __version__


def validate_date(date_string: str) -> str:
    """
    Validate date format (YYYY-MM-DD).

    Args:
        date_string: Date string to validate

    Returns:
        Validated date string

    Raises:
        argparse.ArgumentTypeError: If date format is invalid
    """
    # Check exact format first (YYYY-MM-DD with leading zeros)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_string):
        raise argparse.ArgumentTypeError(f"Invalid date format: '{date_string}'. Expected format: YYYY-MM-DD")

    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return date_string
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: '{date_string}'. Expected format: YYYY-MM-DD")


def validate_directory(path_string: str) -> Path:
    """
    Validate directory path and expand user home directory.

    Args:
        path_string: Directory path string

    Returns:
        Validated Path object

    Raises:
        argparse.ArgumentTypeError: If directory doesn't exist
    """
    path = Path(path_string).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path is not a directory: {path}")
    return path


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="claude-code-cost-collector",
        description="Analyze and aggregate Claude API usage costs from log files.",
        epilog="Example: %(prog)s --granularity monthly --output json --currency EUR",
    )

    # Version option
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Directory option
    parser.add_argument(
        "-d",
        "--directory",
        type=validate_directory,
        default=Path("~/.claude/projects/").expanduser(),
        help="Directory to search for log files (default: %(default)s)",
        metavar="PATH",
    )

    # Granularity option
    parser.add_argument(
        "-g",
        "--granularity",
        choices=["daily", "monthly", "project", "session", "all"],
        default="daily",
        help="Aggregation granularity (default: %(default)s)",
    )

    # Output format option
    parser.add_argument(
        "-o",
        "--output",
        choices=["text", "json", "yaml", "csv"],
        default="text",
        help="Output format (default: %(default)s)",
    )

    # Currency conversion option
    parser.add_argument(
        "--currency",
        help="Convert costs to specified currency (e.g., EUR, GBP, CAD)",
        metavar="CURRENCY_CODE",
    )

    # Date range options
    parser.add_argument(
        "--start-date",
        type=validate_date,
        help="Start date for aggregation (format: YYYY-MM-DD)",
        metavar="YYYY-MM-DD",
    )

    parser.add_argument(
        "--end-date",
        type=validate_date,
        help="End date for aggregation (format: YYYY-MM-DD)",
        metavar="YYYY-MM-DD",
    )

    # Config file option
    parser.add_argument("--config", type=Path, help="Path to configuration file", metavar="PATH")

    # Exchange rate API key option
    parser.add_argument(
        "--exchange-rate-api-key",
        help="API key for exchange rate service",
        metavar="KEY",
    )

    # Debug option
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with detailed processing information",
    )

    # Timezone option
    parser.add_argument(
        "--timezone",
        help="Timezone for date calculation (auto, UTC, or timezone name like Asia/Tokyo)",
        metavar="TZ",
    )

    # Option to show all data (disable default date range)
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Show all available data (disable default date range filtering)",
    )

    # Limit output for performance
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of results shown (for large datasets)",
        metavar="N",
    )

    # Sort order option
    parser.add_argument(
        "--sort",
        choices=["asc", "desc"],
        default="desc",
        help="Sort order: ascending or descending (default: %(default)s)",
        metavar="ORDER",
    )

    # Sort field option
    parser.add_argument(
        "--sort-field",
        choices=["input", "output", "total", "cost", "date"],
        help="Sort results by specified field (input/output tokens, total tokens, cost, or date)",
        metavar="FIELD",
    )

    return parser


def parse_arguments(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: List of arguments to parse. If None, uses sys.argv

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If argument parsing fails or help is requested
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Additional validation
    if parsed_args.start_date and parsed_args.end_date:
        start_date = datetime.strptime(parsed_args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(parsed_args.end_date, "%Y-%m-%d")
        if start_date > end_date:
            parser.error("Start date must be before or equal to end date")

    # Set default sort field to 'date' if not specified
    if parsed_args.sort_field is None:
        parsed_args.sort_field = "date"

    # Currency validation
    if parsed_args.currency:
        # Validate currency code format (3 uppercase letters)
        currency_code = parsed_args.currency.upper()
        if not re.match(r"^[A-Z]{3}$", currency_code):
            parser.error(f"Invalid currency code: '{parsed_args.currency}'. Must be 3 letters (e.g., EUR, GBP, CAD)")
        parsed_args.currency = currency_code

    # Validate config file exists if provided
    if parsed_args.config:
        config_path = Path(parsed_args.config).expanduser().resolve()
        if not config_path.exists():
            parser.error(f"Config file does not exist: {config_path}")
        parsed_args.config = config_path

    return parsed_args


if __name__ == "__main__":
    # For testing purposes
    args = parse_arguments()
    print(f"Parsed arguments: {args}")
