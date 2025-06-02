"""Claude Code Cost Collector main entry point.

This module serves as the main entry point for the application.
Controls the overall processing flow from command-line argument parsing to output.
"""

import logging
import sys
from typing import List, Optional

from .aggregator import AggregationError, aggregate_data
from .cli import parse_arguments
from .collector import collect_log_files
from .config import ConfigError, load_config, validate_config
from .exchange import ExchangeRateError, get_exchange_rate
from .formatter import FormatterError, format_data
from .models import ProcessedLogEntry
from .parser import LogParseError, parse_multiple_log_files


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    if verbose:
        # Debug mode: Show detailed logs
        level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        # Normal mode: Show important messages only
        level = logging.WARNING
        format_str = "%(message)s"
    logging.basicConfig(level=level, format=format_str, datefmt="%Y-%m-%d %H:%M:%S")


def filter_entries_by_date_range(
    entries: List[ProcessedLogEntry], start_date: Optional[str] = None, end_date: Optional[str] = None
) -> List[ProcessedLogEntry]:
    """Filter entries by date range."""
    if not start_date and not end_date:
        return entries

    filtered_entries = []
    for entry in entries:
        entry_date = entry.date_str

        # Check start date
        if start_date and entry_date < start_date:
            continue

        # Check end date
        if end_date and entry_date > end_date:
            continue

        filtered_entries.append(entry)

    return filtered_entries


def get_exchange_rate_for_currency(
    target_currency: str, api_key: Optional[str] = None, config: Optional[dict] = None
) -> Optional[float]:
    """Get exchange rate for specified currency display."""
    try:
        # Get API key from config file (command line argument takes priority)
        if not api_key and config:
            api_key = config.get("exchange_rate_api_key")

        return get_exchange_rate(from_currency="USD", to_currency=target_currency, api_key=api_key)
    except ExchangeRateError as e:
        logging.warning(f"Failed to get exchange rate ({target_currency}): {e}")
        return None


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0: success, 1: error)
    """
    logger = None
    try:
        # 1. Parse command line arguments
        args = parse_arguments()

        # Set up logging
        setup_logging(verbose=args.debug)

        logger = logging.getLogger(__name__)
        if args.debug:
            logger.info("Starting Claude Code Cost Collector")

        # 2. Load configuration file
        try:
            config = load_config(config_path=str(args.config) if args.config else None)
            validate_config(config)
            logger.debug("Configuration file loaded successfully")
        except ConfigError as e:
            logger.error(f"Configuration error: {e}")
            return 1

        # 3. Collect log files
        try:
            log_directory = args.directory
            if args.debug:
                logger.info(f"Collecting log files: {log_directory}")
            log_files = collect_log_files(log_directory)

            if not log_files:
                if args.debug:
                    logger.warning(f"No log files found: {log_directory}")
                print("No log files found.")
                return 0

            if args.debug:
                logger.info(f"Found {len(log_files)} log files")
        except (FileNotFoundError, PermissionError, ValueError) as e:
            logger.error(f"Log file collection error: {e}")
            print(f"Error: {e}")
            return 1

        # 4. Parse log data (considering timezone)
        timezone = args.timezone or config.get("timezone", "auto")
        try:
            if args.debug:
                logger.info(f"Parsing log files... (timezone: {timezone})")
            all_entries = parse_multiple_log_files(log_files, target_timezone=timezone)

            if not all_entries:
                if args.debug:
                    logger.warning("No parseable log entries found")
                print("No parseable log entries found.")
                return 0

            if args.debug:
                logger.info(f"Parsed {len(all_entries)} log entries")
        except LogParseError as e:
            logger.error(f"Log parsing error: {e}")
            print(f"Log parsing error: {e}")
            return 1

        # 5. Filter by date range
        # Apply default date range
        start_date = args.start_date
        end_date = args.end_date
        if not start_date and not end_date and not args.all_data:
            # Apply default date range (when --all-data is not specified)
            default_range_days = config.get("default_date_range_days", 30)
            if default_range_days > 0:
                from datetime import date, timedelta

                end_date = date.today().strftime("%Y-%m-%d")
                start_date = (date.today() - timedelta(days=default_range_days)).strftime("%Y-%m-%d")
                if args.debug:
                    logger.info(f"Applied default date range: {start_date} to {end_date} ({default_range_days} days)")

        if args.all_data and args.debug:
            logger.info("Displaying all data (date range filter disabled)")
        if start_date or end_date:
            if args.debug:
                logger.info("Filtering by date range...")
            filtered_entries = filter_entries_by_date_range(all_entries, start_date, end_date)
            if args.debug:
                logger.info(f"After filtering: {len(filtered_entries)} entries")
            all_entries = filtered_entries

            if not all_entries:
                if args.debug:
                    logger.warning("No log entries found in the specified date range")
                print("No log entries found in the specified date range.")
                return 0

        # 6. Get exchange rate for currency conversion
        exchange_rate = None
        target_currency = None
        if args.currency:
            target_currency = args.currency
            if args.debug:
                logger.info(f"Getting exchange rate... (USD -> {target_currency})")
            exchange_rate = get_exchange_rate_for_currency(
                target_currency=target_currency, api_key=args.exchange_rate_api_key, config=config
            )
            if exchange_rate is None:
                if args.debug:
                    logger.warning(f"Failed to get exchange rate. {target_currency} display will be disabled.")
                target_currency = None
            else:
                if args.debug:
                    logger.info(f"Exchange rate USD/{target_currency}: {exchange_rate}")

        # 7. Data aggregation
        try:
            if args.debug:
                logger.info(f"Aggregating data by {args.granularity}...")
            aggregated_data = aggregate_data(
                all_entries,
                granularity=args.granularity,
                exchange_rate=exchange_rate,
                target_currency=target_currency,
                sort_by=args.sort_field,
                sort_desc=(args.sort == "desc"),
            )
            if args.debug:
                logger.info("Data aggregation completed")
        except (AggregationError, ValueError) as e:
            logger.error(f"Data aggregation error: {e}")
            print(f"Data aggregation error: {e}")
            return 1

        # 8. Output formatting
        try:
            if args.debug:
                logger.info(f"Generating output in {args.output} format...")
            output = format_data(
                data=aggregated_data,
                output_format=args.output,
                granularity=args.granularity,
                target_currency=target_currency,
                limit=args.limit,
                sort_by=args.sort_field,
                sort_desc=(args.sort == "desc"),
            )

            # Output results
            print(output)
            if args.debug:
                logger.info("Processing completed successfully")

        except FormatterError as e:
            logger.error(f"Output formatting error: {e}")
            print(f"Output formatting error: {e}")
            return 1

        return 0

    except KeyboardInterrupt:
        if logger:
            logger.info("Processing interrupted by user")
        print("\nProcessing interrupted.")
        return 1

    except Exception as e:
        if logger:
            logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
