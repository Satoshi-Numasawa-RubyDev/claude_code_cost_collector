"""Output formatting module for Claude Code Cost Collector.

This module provides functions to format aggregated data and individual log entries
into various output formats including text tables, JSON, YAML, and CSV.
"""

import csv
import json
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import AggregatedData, ProcessedLogEntries


class FormatterError(Exception):
    """Custom exception for formatting errors."""

    pass


def format_data(
    data: Union[ProcessedLogEntries, AggregatedData],
    output_format: str = "text",
    granularity: str = "daily",
    target_currency: Optional[str] = None,
    limit: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = False,
) -> str:
    """Main formatting function that dispatches to appropriate formatter.

    Args:
        data: Either a list of ProcessedLogEntry objects or aggregated data dictionary
        output_format: Output format ('text', 'json', 'yaml', 'csv')
        granularity: Aggregation granularity ('daily', 'monthly', 'project', 'session', 'all')
        target_currency: Target currency code for conversion display (e.g., "USD", "EUR")
        limit: Maximum number of results to display (None for no limit)
        sort_by: Sort field ('input', 'output', 'total', 'cost') - if None, use existing order
        sort_desc: Sort in descending order (default: False)

    Returns:
        Formatted string ready for output

    Raises:
        FormatterError: If output format is not supported or formatting fails
    """
    supported_formats = {"text", "json", "yaml", "csv"}

    if output_format not in supported_formats:
        raise FormatterError(f"Unsupported output format: {output_format}. Supported formats: {', '.join(supported_formats)}")

    # Apply limit if specified, but preserve sort order from aggregator
    if limit is not None:
        data = _apply_limit(data, limit, granularity, sort_by)

    try:
        if output_format == "text":
            return _format_text_table(data, granularity, target_currency, sort_by)
        elif output_format == "json":
            return _format_json(data, granularity, target_currency)
        elif output_format == "yaml":
            return _format_yaml(data, granularity, target_currency)
        elif output_format == "csv":
            return _format_csv(data, granularity, target_currency)
        else:
            # This should never be reached due to the check above
            raise FormatterError(f"Unknown output format: {output_format}")
    except Exception as e:
        raise FormatterError(f"Failed to format data as {output_format}: {str(e)}")


def _format_text_table(
    data: Union[ProcessedLogEntries, AggregatedData],
    granularity: str,
    target_currency: Optional[str] = None,
    sort_by: Optional[str] = None,
) -> str:
    """Format data as a text table using rich library.

    Args:
        data: Either a list of ProcessedLogEntry objects or aggregated data dictionary
        granularity: Aggregation granularity for title and column headers
        target_currency: Target currency code for conversion display
        sort_by: Sort field specified by user - if provided, preserve existing order

    Returns:
        Formatted text table as string
    """
    if isinstance(data, list):
        return _format_individual_entries_as_text(data, target_currency)
    else:
        return _format_aggregated_data_as_text(data, granularity, target_currency, sort_by)


def _format_json(
    data: Union[ProcessedLogEntries, AggregatedData], granularity: str, target_currency: Optional[str] = None
) -> str:
    """Format data as JSON.

    Args:
        data: Either a list of ProcessedLogEntry objects or aggregated data dictionary
        granularity: Aggregation granularity for metadata
        target_currency: Target currency code for conversion display

    Returns:
        Formatted JSON string with metadata and data sections

    Raises:
        ValueError: If data cannot be serialized to JSON
    """
    try:
        if isinstance(data, list):
            # Individual log entries
            json_data: Union[List[Dict[str, Any]], Dict[str, Any]] = [entry.to_dict() for entry in data]
            total_entries = len(data)

            # Calculate summary for individual entries
            if data:
                total_tokens = sum(entry.total_tokens for entry in data)
                total_cost_usd = sum(entry.cost_usd for entry in data)
                total_converted_cost = sum(entry.converted_cost or 0.0 for entry in data) if target_currency else None
            else:
                total_tokens = 0
                total_cost_usd = 0.0
                total_converted_cost = 0.0 if target_currency else None

            summary = {
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
            }
            if target_currency and total_converted_cost is not None:
                summary[f"total_cost_{target_currency.lower()}"] = total_converted_cost

        else:
            # Aggregated data
            json_data = data
            total_entries = len(data)

            # Calculate summary for aggregated data
            total_tokens = sum(values.get("total_tokens", 0) for values in data.values())
            total_cost_usd = sum(values.get("total_cost_usd", 0.0) for values in data.values())

            summary = {
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
            }
            if target_currency:
                total_converted_cost = sum(values.get("total_converted_cost", 0.0) for values in data.values())
                summary[f"total_cost_{target_currency.lower()}"] = total_converted_cost

        output = {
            "metadata": {
                "format_version": "1.0",
                "granularity": granularity,
                "target_currency": target_currency,
                "generated_at": datetime.now().isoformat(),
                "data_type": "individual_entries" if isinstance(data, list) else "aggregated",
                "total_entries": total_entries,
            },
            "summary": summary,
            "data": json_data,
        }

        return json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True)

    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize data to JSON: {str(e)}")


def _format_yaml(
    data: Union[ProcessedLogEntries, AggregatedData], granularity: str, target_currency: Optional[str] = None
) -> str:
    """Format data as YAML.

    Args:
        data: Either a list of ProcessedLogEntry objects or aggregated data dictionary
        granularity: Aggregation granularity for metadata
        target_currency: Target currency code for conversion display

    Returns:
        Formatted YAML string
    """
    if isinstance(data, list):
        yaml_data: Union[List[Dict[str, Any]], Dict[str, Any]] = [entry.to_dict() for entry in data]
    else:
        yaml_data = data

    output = {
        "metadata": {
            "granularity": granularity,
            "target_currency": target_currency,
            "generated_at": datetime.now().isoformat(),
            "total_entries": (len(yaml_data) if isinstance(yaml_data, list) else len(yaml_data)),
        },
        "data": yaml_data,
    }

    result = yaml.dump(
        output,
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True,
        indent=2,
        width=120,
    )
    return str(result)


def _format_csv(data: Union[ProcessedLogEntries, AggregatedData], granularity: str, target_currency: Optional[str] = None) -> str:
    """Format data as CSV.

    This is a stub implementation that will be detailed in task 6.4.
    """
    output = StringIO()
    writer = csv.writer(output)

    if isinstance(data, list):
        headers = [
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
        if target_currency:
            headers.append(f"cost_{target_currency.lower()}")
        writer.writerow(headers)

        for entry in data:
            row = [
                entry.timestamp.isoformat(),
                entry.date_str,
                entry.project_name,
                entry.session_id,
                entry.input_tokens,
                entry.output_tokens,
                entry.total_tokens,
                entry.cost_usd,
                entry.model,
            ]
            if target_currency and hasattr(entry, "converted_cost") and entry.converted_cost is not None:
                row.append(entry.converted_cost)
            elif target_currency:
                row.append("")
            writer.writerow(row)
    else:
        # Aggregated data - use different headers based on granularity
        granularity_column_map = {
            "daily": "date",
            "monthly": "month",
            "project": "project",
            "session": "session_id",
            "all": "key",
        }

        key_column = granularity_column_map.get(granularity, "key")

        headers = [
            key_column,
            "entry_count",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_cost_usd",
        ]
        if target_currency:
            headers.append(f"total_cost_{target_currency.lower()}")
        writer.writerow(headers)

        # Sort aggregated data by key for consistent output
        for key, values in sorted(data.items()):
            row = [
                key,
                values.get("entry_count", 0),
                values.get("total_input_tokens", 0),
                values.get("total_output_tokens", 0),
                values.get("total_tokens", 0),
                values.get("total_cost_usd", 0.0),
            ]
            if target_currency:
                row.append(values.get("total_converted_cost", 0.0) if values.get("total_converted_cost") is not None else "")
            writer.writerow(row)

    return output.getvalue()


def _format_individual_entries_as_text(entries: ProcessedLogEntries, target_currency: Optional[str] = None) -> str:
    """Helper function to format individual log entries as text table.

    Args:
        entries: List of ProcessedLogEntry objects
        target_currency: Target currency code for conversion display

    Returns:
        Formatted text table as string
    """
    console = Console(width=120, force_terminal=False)
    table = Table(title="Individual Log Entries", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Project", style="green")
    table.add_column("Session ID", style="yellow", no_wrap=True)
    table.add_column("Model", style="blue")
    table.add_column("Input", justify="right", style="bright_blue")
    table.add_column("Output", justify="right", style="bright_blue")
    table.add_column("Total", justify="right", style="bold bright_blue")
    table.add_column("Cost USD", justify="right", style="bold green")

    # Add currency conversion column if available
    show_converted_currency = bool(target_currency)
    if show_converted_currency:
        table.add_column(f"Cost {target_currency}", justify="right", style="bold green")

    # Add data rows
    total_tokens = 0
    total_cost_usd = 0.0
    total_converted_cost = 0.0

    for entry in entries:
        total_tokens += entry.total_tokens
        total_cost_usd += entry.cost_usd

        # Handle converted currency costs
        if target_currency and hasattr(entry, "converted_cost") and entry.converted_cost:
            total_converted_cost += entry.converted_cost

        row_data = [
            entry.date_str,
            (entry.project_name[:20] + "..." if len(entry.project_name) > 20 else entry.project_name),
            entry.session_id[:8] + "...",
            entry.model,
            f"{entry.input_tokens:,}",
            f"{entry.output_tokens:,}",
            f"{entry.total_tokens:,}",
            f"${entry.cost_usd:.4f}",
        ]

        if target_currency:
            converted_cost = 0.0
            if hasattr(entry, "converted_cost") and entry.converted_cost:
                converted_cost = entry.converted_cost

            # Format converted currency with generic format
            currency_str = f"{converted_cost:.2f} {target_currency}" if converted_cost > 0 else "N/A"
            row_data.append(currency_str)

        table.add_row(*row_data)

    # Add total row
    total_row: List[Union[str, Text]] = [
        Text("TOTAL", style="bold red"),
        "",
        "",
        "",
        f"{sum(e.input_tokens for e in entries):,}",
        f"{sum(e.output_tokens for e in entries):,}",
        f"{total_tokens:,}",
        f"${total_cost_usd:.4f}",
    ]

    if target_currency:
        currency_total_str = f"{total_converted_cost:.2f} {target_currency}" if total_converted_cost > 0 else "N/A"
        total_row.append(currency_total_str)

    table.add_section()
    table.add_row(*total_row, style="bold")

    # Capture console output
    with console.capture() as capture:
        console.print(table)

    return capture.get()


def _format_aggregated_data_as_text(
    data: AggregatedData, granularity: str, target_currency: Optional[str] = None, sort_by: Optional[str] = None
) -> str:
    """Helper function to format aggregated data as text table.

    Args:
        data: Aggregated data dictionary
        granularity: Aggregation granularity for title and column headers
        target_currency: Target currency code for conversion display
        sort_by: Sort field specified by user - if provided, preserve existing order

    Returns:
        Formatted text table as string
    """
    console = Console(width=120, force_terminal=False)

    # Determine title and key column name based on granularity
    title_mapping = {
        "daily": "Daily Cost Summary",
        "monthly": "Monthly Cost Summary",
        "project": "Project Cost Summary",
        "session": "Session Cost Summary",
    }

    key_column_mapping = {
        "daily": "Date",
        "monthly": "Month",
        "project": "Project",
        "session": "Session ID",
    }

    title = title_mapping.get(granularity, f"{granularity.title()} Cost Summary")
    key_column = key_column_mapping.get(granularity, granularity.title())

    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column(key_column, style="cyan", no_wrap=True)
    table.add_column("Input", justify="right", style="bright_blue")
    table.add_column("Output", justify="right", style="bright_blue")
    table.add_column("Cache Create", justify="right", style="yellow")
    table.add_column("Cache Read", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="bold bright_blue")
    table.add_column("Cost USD", justify="right", style="bold green")

    # Add currency conversion column if available
    show_converted_currency = bool(target_currency)
    if show_converted_currency:
        table.add_column(f"Cost {target_currency}", justify="right", style="bold green")

    # Use data as-is if user specified sort, otherwise apply default sorting
    if sort_by is not None:
        # User specified sort - preserve the order from aggregator
        sorted_data = list(data.items())
    elif granularity in ["daily", "monthly"]:
        # No user sort - apply default date sorting for readability (newest first)
        sorted_data = sorted(data.items(), reverse=True)
    else:
        # For project/session without user sort, maintain natural order
        sorted_data = list(data.items())

    # Calculate totals
    total_input = 0
    total_output = 0
    total_cache_creation = 0
    total_cache_read = 0
    total_tokens = 0
    total_cost_usd = 0.0
    total_converted_cost = 0.0

    # Add data rows
    for key, values in sorted_data:
        input_tokens = values.get("total_input_tokens", 0)
        output_tokens = values.get("total_output_tokens", 0)
        cache_creation_tokens = values.get("total_cache_creation_tokens", 0)
        cache_read_tokens = values.get("total_cache_read_tokens", 0)
        tokens = values.get("total_tokens", 0)
        cost_usd = values.get("total_cost_usd", 0.0)
        # Get converted currency cost
        converted_cost = 0.0
        if target_currency:
            converted_cost = values.get("total_converted_cost", 0.0)

        total_input += input_tokens
        total_output += output_tokens
        total_cache_creation += cache_creation_tokens
        total_cache_read += cache_read_tokens
        total_tokens += tokens
        total_cost_usd += cost_usd
        total_converted_cost += converted_cost

        # Format key for display (truncate if too long)
        display_key = key[:30] + "..." if len(key) > 30 else key

        row_data = [
            display_key,
            f"{input_tokens:,}",
            f"{output_tokens:,}",
            f"{cache_creation_tokens:,}",
            f"{cache_read_tokens:,}",
            f"{tokens:,}",
            f"${cost_usd:.4f}",
        ]

        if target_currency:
            # Format converted currency with generic format
            currency_str = f"{converted_cost:.2f} {target_currency}" if converted_cost > 0 else "N/A"
            row_data.append(currency_str)

        table.add_row(*row_data)

    # Add total row
    total_row: List[Union[str, Text]] = [
        Text("TOTAL", style="bold red"),
        f"{total_input:,}",
        f"{total_output:,}",
        f"{total_cache_creation:,}",
        f"{total_cache_read:,}",
        f"{total_tokens:,}",
        f"${total_cost_usd:.4f}",
    ]

    if target_currency:
        currency_total_str = f"{total_converted_cost:.2f} {target_currency}" if total_converted_cost > 0 else "N/A"
        total_row.append(currency_total_str)

    table.add_section()
    table.add_row(*total_row, style="bold")

    # Capture console output
    with console.capture() as capture:
        console.print(table)

    return capture.get()


def _apply_limit(
    data: Union[ProcessedLogEntries, AggregatedData], limit: int, granularity: str, sort_by: Optional[str] = None
) -> Union[ProcessedLogEntries, AggregatedData]:
    """Apply limit to data by sorting appropriately and taking top N entries.

    Args:
        data: Either a list of ProcessedLogEntry objects or aggregated data dictionary
        limit: Maximum number of results to return (0 returns empty data)
        granularity: Aggregation granularity to determine sort order
        sort_by: Sort field specified by user - if provided, preserve existing order

    Returns:
        Limited data of the same type as input
    """
    if limit <= 0:
        # Return empty data of the same type
        if isinstance(data, list):
            return []
        else:
            return {}

    if isinstance(data, list):
        # For individual entries, preserve order if user specified sort, otherwise sort by timestamp
        if sort_by is not None:
            # User specified sort - preserve the order from aggregator
            return data[:limit]
        else:
            # No user sort - use default timestamp sort (newest first)
            sorted_entries = sorted(data, key=lambda entry: entry.timestamp, reverse=True)
            return sorted_entries[:limit]
    else:
        # For aggregated data
        if sort_by is not None:
            # User specified sort - preserve the order from aggregator
            # Just take the first N items in the existing order
            return dict(list(data.items())[:limit])
        else:
            # No user sort - apply default sorting based on granularity
            if granularity in ["daily", "monthly"]:
                # Sort by date/time key (newest first), then take top N
                sorted_items = sorted(data.items(), key=lambda item: item[0], reverse=True)
            elif granularity in ["project", "session"]:
                # Sort by cost (highest first) for projects and sessions
                sorted_items = sorted(data.items(), key=lambda item: item[1].get("total_cost_usd", 0.0), reverse=True)
            else:  # granularity == "all"
                # Sort by timestamp for individual entries in "all" mode
                sorted_items = sorted(data.items(), key=lambda item: item[0], reverse=True)

            return dict(sorted_items[:limit])


def get_supported_formats() -> List[str]:
    """Get list of supported output formats.

    Returns:
        List of supported format strings
    """
    return ["text", "json", "yaml", "csv"]


def format_summary_statistics(
    data: Union[ProcessedLogEntries, AggregatedData], target_currency: Optional[str] = None
) -> Dict[str, Any]:
    """Generate summary statistics for the data.

    Args:
        data: Data to analyze
        target_currency: Target currency code for conversion statistics

    Returns:
        Dictionary containing summary statistics
    """
    if isinstance(data, list):
        total_entries = len(data)
        total_tokens = sum(entry.total_tokens for entry in data)
        total_cost_usd = sum(entry.cost_usd for entry in data)

        stats = {
            "total_entries": total_entries,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "average_tokens_per_entry": (total_tokens / total_entries if total_entries > 0 else 0),
            "average_cost_per_entry": (total_cost_usd / total_entries if total_entries > 0 else 0),
        }

        if target_currency:
            total_converted_cost = sum(entry.converted_cost or 0.0 for entry in data if hasattr(entry, "converted_cost"))
            stats[f"total_cost_{target_currency.lower()}"] = total_converted_cost
            stats[f"average_cost_per_entry_{target_currency.lower()}"] = (
                total_converted_cost / total_entries if total_entries > 0 else 0
            )

    else:
        total_entries = len(data)
        total_tokens = sum(values.get("total_tokens", 0) for values in data.values())
        total_cost_usd = sum(values.get("total_cost_usd", 0.0) for values in data.values())

        stats = {
            "total_aggregation_groups": total_entries,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
        }

        if target_currency:
            total_converted_cost = sum(values.get("total_converted_cost", 0.0) for values in data.values())
            stats[f"total_cost_{target_currency.lower()}"] = total_converted_cost

    return stats
