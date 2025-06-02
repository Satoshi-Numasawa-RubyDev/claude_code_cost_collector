"""
Data aggregation module for Claude Code Cost Collector.

This module provides functionality to aggregate ProcessedLogEntry data
according to different granularities (daily, monthly, project, session,
individual).
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

from .models import AggregatedData, ProcessedLogEntries, ProcessedLogEntry

logger = logging.getLogger(__name__)


class AggregationError(Exception):
    """Exception raised when aggregation fails."""

    pass


def aggregate_data(entries: ProcessedLogEntries, granularity: str, **kwargs: Any) -> AggregatedData:
    """
    Aggregate processed log entries according to specified granularity.

    This is the main entry point for data aggregation. It dispatches to
    appropriate helper functions based on the granularity parameter.

    Args:
        entries: List of ProcessedLogEntry objects to aggregate
        granularity: Aggregation granularity ('daily', 'monthly', 'project',
                    'session', 'all')
        **kwargs: Additional keyword arguments for specific aggregation methods.
                  Supported kwargs:
                  - exchange_rate: Exchange rate for currency conversion
                  - target_currency: Target currency code for conversion (e.g., "EUR", "GBP")
                  - sort_by: Sort field ('input', 'output', 'total', 'cost')
                  - sort_desc: Sort in descending order (default: False)

    Returns:
        Dictionary containing aggregated data

    Raises:
        AggregationError: If aggregation fails or granularity is invalid
        ValueError: If input parameters are invalid
    """
    # Validate input parameters first
    if not isinstance(entries, list):
        raise ValueError("entries must be a list of ProcessedLogEntry objects")

    # Validate granularity
    valid_granularities = ["daily", "monthly", "project", "session", "all"]
    if granularity not in valid_granularities:
        raise ValueError(f"Invalid granularity '{granularity}'. Must be one of: {valid_granularities}")

    if not entries:
        logger.warning("No entries provided for aggregation")
        return {}

    logger.info(f"Aggregating {len(entries)} entries by {granularity}")

    try:
        # Extract currency conversion kwargs
        exchange_rate = kwargs.get("exchange_rate")
        target_currency = kwargs.get("target_currency")

        # If currency conversion should be done
        if target_currency and exchange_rate:
            entries = _add_converted_costs_to_entries(entries, exchange_rate, target_currency)

        # Dispatch to appropriate aggregation function
        if granularity == "daily":
            result = _aggregate_by_daily(entries, **kwargs)
        elif granularity == "monthly":
            result = _aggregate_by_monthly(entries, **kwargs)
        elif granularity == "project":
            result = _aggregate_by_project(entries, **kwargs)
        elif granularity == "session":
            result = _aggregate_by_session(entries, **kwargs)
        elif granularity == "all":
            result = _prepare_individual_entries(entries, **kwargs)
        else:
            # This should not happen due to validation above, but just in case
            raise AggregationError(f"Unimplemented granularity: {granularity}")

        # Apply sorting if requested
        sort_by = kwargs.get("sort_by")
        sort_desc = kwargs.get("sort_desc", False)
        if sort_by:
            result = _sort_aggregated_data(result, sort_by, sort_desc)

        return result

    except Exception as e:
        logger.error(f"Failed to aggregate data by {granularity}: {e}")
        raise AggregationError(f"Aggregation failed: {e}") from e


def _aggregate_by_daily(entries: ProcessedLogEntries, **kwargs: Any) -> AggregatedData:
    """
    Aggregate entries by daily granularity.

    Groups entries by date_str and aggregates metrics for each day.

    Args:
        entries: List of ProcessedLogEntry objects

    Returns:
        Dictionary with date strings as keys and aggregated data as values
    """
    logger.debug(f"Performing daily aggregation on {len(entries)} entries")

    daily_groups = defaultdict(list)
    for entry in entries:
        daily_groups[entry.date_str].append(entry)

    result = {}
    for date_str, date_entries in daily_groups.items():
        result[date_str] = _create_aggregate_entry(date_str, date_entries)

    logger.info(f"Created {len(result)} daily aggregation groups")
    return result


def _aggregate_by_monthly(entries: ProcessedLogEntries, **kwargs: Any) -> AggregatedData:
    """
    Aggregate entries by monthly granularity.

    Groups entries by month_str and aggregates metrics for each month.

    Args:
        entries: List of ProcessedLogEntry objects

    Returns:
        Dictionary with month strings as keys and aggregated data as values
    """
    logger.debug(f"Performing monthly aggregation on {len(entries)} entries")

    monthly_groups = defaultdict(list)
    for entry in entries:
        monthly_groups[entry.month_str].append(entry)

    result = {}
    for month_str, month_entries in monthly_groups.items():
        result[month_str] = _create_aggregate_entry(month_str, month_entries)

    logger.info(f"Created {len(result)} monthly aggregation groups")
    return result


def _aggregate_by_project(entries: ProcessedLogEntries, **kwargs: Any) -> AggregatedData:
    """
    Aggregate entries by project granularity.

    Groups entries by project_name and aggregates metrics for each project.

    Args:
        entries: List of ProcessedLogEntry objects

    Returns:
        Dictionary with project names as keys and aggregated data as values
    """
    logger.debug(f"Performing project aggregation on {len(entries)} entries")

    project_groups = defaultdict(list)
    for entry in entries:
        project_groups[entry.project_name].append(entry)

    result = {}
    for project_name, project_entries in project_groups.items():
        result[project_name] = _create_aggregate_entry(project_name, project_entries)

    logger.info(f"Created {len(result)} project aggregation groups")
    return result


def _aggregate_by_session(entries: ProcessedLogEntries, **kwargs: Any) -> AggregatedData:
    """
    Aggregate entries by session granularity.

    Groups entries by session_id and aggregates metrics for each session.

    Args:
        entries: List of ProcessedLogEntry objects

    Returns:
        Dictionary with session IDs as keys and aggregated data as values
    """
    logger.debug(f"Performing session aggregation on {len(entries)} entries")

    session_groups = defaultdict(list)
    for entry in entries:
        session_groups[entry.session_id].append(entry)

    result = {}
    for session_id, session_entries in session_groups.items():
        result[session_id] = _create_aggregate_entry(session_id, session_entries)

    logger.info(f"Created {len(result)} session aggregation groups")
    return result


def _prepare_individual_entries(entries: ProcessedLogEntries, **kwargs: Any) -> AggregatedData:
    """
    Prepare individual entries for display without aggregation.

    Converts each ProcessedLogEntry to a dictionary suitable for output
    formatting.

    Args:
        entries: List of ProcessedLogEntry objects

    Returns:
        Dictionary with entry indices as keys and entry data as values
    """
    logger.debug(f"Preparing {len(entries)} individual entries")

    result = {}
    for i, entry in enumerate(entries):
        # Create a unique key for each entry (using index + some entry info)
        key = f"{i:06d}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}_{entry.session_id[:8]}"

        # Convert entry to dictionary and add some computed fields
        entry_dict = entry.to_dict()
        entry_dict.update(
            {
                "entry_index": i,
                "total_cost_usd": entry.cost_usd,
                "total_input_tokens": entry.input_tokens,
                "total_output_tokens": entry.output_tokens,
                "total_tokens": entry.total_tokens,
                "entry_count": 1,
            }
        )

        result[key] = entry_dict

    logger.info(f"Prepared {len(result)} individual entries")
    return result


def _create_aggregate_entry(key: str, entries: List[ProcessedLogEntry]) -> Dict[str, Any]:
    """
    Create an aggregated entry from a list of ProcessedLogEntry objects.

    This utility function performs the common aggregation logic:
    - Sums tokens and costs
    - Collects metadata
    - Handles currency conversion if available

    Args:
        key: The aggregation key (date, project name, etc.)
        entries: List of ProcessedLogEntry objects to aggregate

    Returns:
        Dictionary containing aggregated data
    """
    if not entries:
        raise ValueError("Cannot create aggregate entry from empty list")

    # Initialize aggregation variables
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    total_converted_cost = 0.0
    has_converted_data = False
    converted_currency = None

    # Collect metadata
    models = set()
    projects = set()
    sessions = set()
    earliest_timestamp = None
    latest_timestamp = None

    # Aggregate all entries
    for entry in entries:
        total_input_tokens += entry.input_tokens
        total_output_tokens += entry.output_tokens
        total_cache_creation_tokens += entry.cache_creation_tokens or 0
        total_cache_read_tokens += entry.cache_read_tokens or 0
        total_tokens += entry.total_tokens
        total_cost_usd += entry.cost_usd

        if entry.converted_cost is not None:
            total_converted_cost += entry.converted_cost
            has_converted_data = True
            if converted_currency is None:
                converted_currency = entry.target_currency

        models.add(entry.model)
        projects.add(entry.project_name)
        sessions.add(entry.session_id)

        if earliest_timestamp is None or entry.timestamp < earliest_timestamp:
            earliest_timestamp = entry.timestamp
        if latest_timestamp is None or entry.timestamp > latest_timestamp:
            latest_timestamp = entry.timestamp

    # Build aggregated result
    result = {
        "key": key,
        "entry_count": len(entries),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_creation_tokens": total_cache_creation_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost_usd, 8),  # Round to avoid floating point precision issues
        "models": sorted(list(models)),
        "projects": sorted(list(projects)),
        "sessions": sorted(list(sessions)),
        "earliest_timestamp": (earliest_timestamp.isoformat() if earliest_timestamp else None),
        "latest_timestamp": (latest_timestamp.isoformat() if latest_timestamp else None),
    }

    # Add converted currency data if available
    if has_converted_data and converted_currency:
        result["total_converted_cost"] = round(total_converted_cost, 2)
        result["target_currency"] = converted_currency

    return result


def get_supported_granularities() -> List[str]:
    """
    Get list of supported aggregation granularities.

    Returns:
        List of supported granularity strings
    """
    return ["daily", "monthly", "project", "session", "all"]


def validate_aggregation_input(entries: ProcessedLogEntries, granularity: str) -> None:
    """
    Validate input parameters for aggregation.

    Args:
        entries: List of ProcessedLogEntry objects
        granularity: Aggregation granularity string

    Raises:
        ValueError: If input parameters are invalid
    """
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")

    if not all(isinstance(entry, ProcessedLogEntry) for entry in entries):
        raise ValueError("All entries must be ProcessedLogEntry objects")

    if granularity not in get_supported_granularities():
        raise ValueError(f"Invalid granularity '{granularity}'. Must be one of: {get_supported_granularities()}")


def _add_converted_costs_to_entries(
    entries: ProcessedLogEntries, exchange_rate: float, target_currency: str
) -> ProcessedLogEntries:
    """
    Add converted currency costs to entries using the provided exchange rate.

    Args:
        entries: List of ProcessedLogEntry objects
        exchange_rate: USD to target currency exchange rate
        target_currency: Target currency code (e.g., "EUR", "GBP")

    Returns:
        List of ProcessedLogEntry objects with converted currency costs calculated
    """
    from .exchange import convert_currency

    logger.debug(f"Adding {target_currency} costs to {len(entries)} entries using exchange rate {exchange_rate}")

    updated_entries = []
    for entry in entries:
        # Create a new entry with converted cost if it doesn't have one for this currency
        needs_conversion = entry.converted_cost is None or entry.target_currency != target_currency

        if needs_conversion:
            # Calculate converted cost using the exchange rate
            converted_cost = convert_currency(
                amount=entry.cost_usd, from_currency="USD", to_currency=target_currency, exchange_rate=exchange_rate
            )

            # Create a new ProcessedLogEntry with the converted cost
            entry_dict = entry.to_dict()
            entry_dict["converted_cost"] = converted_cost
            entry_dict["target_currency"] = target_currency

            updated_entry = ProcessedLogEntry.from_dict(entry_dict)
            updated_entries.append(updated_entry)
        else:
            # Entry already has the correct converted cost, use as-is
            updated_entries.append(entry)

    logger.debug(f"Updated {len(updated_entries)} entries with {target_currency} costs")
    return updated_entries


def _sort_aggregated_data(data: AggregatedData, sort_by: str, sort_desc: bool = False) -> AggregatedData:
    """
    Sort aggregated data by specified field.

    Args:
        data: Dictionary containing aggregated data
        sort_by: Sort field ('input', 'output', 'total', 'cost')
        sort_desc: Sort in descending order (default: False)

    Returns:
        Sorted dictionary containing aggregated data
    """
    if not data:
        return data

    # Map sort_by parameter to actual field names in aggregated data
    field_mapping = {
        "input": "total_input_tokens",
        "output": "total_output_tokens",
        "total": "total_tokens",
        "cost": None,  # Will be determined based on available cost fields
        "date": None,  # Special case: sort by key (date string)
    }

    field_name = field_mapping.get(sort_by)

    # Handle special cases
    if field_name is None and sort_by == "cost":
        # For cost, prefer converted cost if available, otherwise use USD cost
        sample_entry = next(iter(data.values()))
        if "total_converted_cost" in sample_entry:
            field_name = "total_converted_cost"
        else:
            field_name = "total_cost_usd"
    elif field_name is None and sort_by == "date":
        # For date, sort by key (date string) directly
        logger.debug(f"Sorting data by date key (desc={sort_desc})")
        try:
            sorted_items = sorted(data.items(), key=lambda item: item[0], reverse=sort_desc)
            result = dict(sorted_items)
            logger.info(f"Sorted {len(result)} items by date")
            return result
        except Exception as e:
            logger.error(f"Failed to sort data by date: {e}")
            return data
    elif field_name is None:
        logger.warning(f"Invalid sort field: {sort_by}. Returning unsorted data.")
        return data

    logger.debug(f"Sorting data by {field_name} (desc={sort_desc})")

    try:
        # Sort items by the specified field
        sorted_items = sorted(data.items(), key=lambda item: item[1].get(field_name, 0), reverse=sort_desc)

        # Convert back to dictionary
        result = dict(sorted_items)
        logger.info(f"Sorted {len(result)} items by {field_name}")
        return result

    except Exception as e:
        logger.error(f"Failed to sort data by {field_name}: {e}")
        return data
