"""
Log data parser for Claude Code Cost Collector.

This module parses JSON log files and converts them into standardized
ProcessedLogEntry objects for further processing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cost_calculator import CostCalculator, TokenUsage, create_default_cost_calculator
from .exceptions import LogParseError
from .format_detector import FormatDetector, LogFormat
from .models import ProcessedLogEntry

logger = logging.getLogger(__name__)


class LogParser:
    """
    Enhanced log parser with format detection and cost calculation capabilities.

    This class provides automatic format detection and cost estimation for
    Claude CLI logs that may not contain direct cost information.
    """

    def __init__(self, cost_calculator: Optional[CostCalculator] = None):
        """
        Initialize the log parser.

        Args:
            cost_calculator: Optional cost calculator. If None, a default one will be created.
        """
        self.cost_calculator = cost_calculator or create_default_cost_calculator()

    def parse_log_entry(
        self, raw_entry: Dict[str, Any], project_name: str, file_path: Path, target_timezone: Optional[str] = None
    ) -> Optional[ProcessedLogEntry]:
        """
        Parse a single log entry with automatic format detection.

        Args:
            raw_entry: Raw log entry dictionary
            project_name: Project name extracted from file path
            file_path: Source file path for context
            target_timezone: Target timezone for date calculation

        Returns:
            ProcessedLogEntry or None if entry should be skipped

        Raises:
            LogParseError: If required fields are missing or invalid
        """
        # Detect format
        format_type = FormatDetector.detect_entry_format(raw_entry)

        # Parse based on detected format
        if format_type == LogFormat.LEGACY:
            return self._parse_legacy_format(raw_entry, project_name, file_path, target_timezone)
        elif format_type == LogFormat.V1_0_9:
            return self._parse_v1_0_9_format(raw_entry, project_name, file_path, target_timezone)
        else:
            # Unknown format - try legacy format as fallback
            try:
                return self._parse_legacy_format(raw_entry, project_name, file_path, target_timezone)
            except LogParseError:
                # If legacy parsing fails, skip this entry
                logger.debug(f"Skipping entry with unknown format: {raw_entry.get('type', 'unknown')}")
                return None

    def _parse_legacy_format(
        self, raw_entry: Dict[str, Any], project_name: str, file_path: Path, target_timezone: Optional[str] = None
    ) -> Optional[ProcessedLogEntry]:
        """Parse entry using legacy format logic (existing implementation)."""
        return _parse_single_log_entry(raw_entry, project_name, file_path, target_timezone)

    def _parse_v1_0_9_format(
        self, raw_entry: Dict[str, Any], project_name: str, file_path: Path, target_timezone: Optional[str] = None
    ) -> Optional[ProcessedLogEntry]:
        """
        Parse entry using v1.0.9 format logic with cost estimation.

        Args:
            raw_entry: Raw log entry dictionary
            project_name: Project name extracted from file path
            file_path: Source file path for context
            target_timezone: Target timezone for date calculation

        Returns:
            ProcessedLogEntry or None if entry should be skipped

        Raises:
            LogParseError: If required fields are missing or invalid
        """
        # Only process assistant entries
        if raw_entry.get("type") != "assistant":
            return None

        # Skip API error messages (they use synthetic models and have no meaningful usage data)
        if raw_entry.get("isApiErrorMessage", False):
            logger.debug(f"Skipping API error message entry: {raw_entry.get('message', {}).get('content', 'N/A')}")
            return None

        # Check for required fields (costUSD is not required for v1.0.9)
        required_fields = ["timestamp", "sessionId"]
        missing_fields = [field for field in required_fields if field not in raw_entry]

        # Check for model field in message
        if "message" not in raw_entry or "model" not in raw_entry.get("message", {}):
            missing_fields.append("message.model")

        if missing_fields:
            raise LogParseError(f"Missing required fields for v1.0.9 format: {missing_fields}")

        # Parse timestamp
        try:
            timestamp = _parse_timestamp(raw_entry["timestamp"])
        except ValueError as e:
            raise LogParseError(f"Invalid timestamp format: {e}")

        # Extract usage information - v1.0.9 specific extraction
        usage_info = self._extract_v1_0_9_usage_info(raw_entry)

        # Generate date strings for aggregation with timezone consideration
        display_timestamp = _convert_to_target_timezone(timestamp, target_timezone)
        date_str = display_timestamp.strftime("%Y-%m-%d")
        month_str = display_timestamp.strftime("%Y-%m")

        # Calculate total tokens
        cache_creation_tokens = usage_info["cache_creation_tokens"] or 0
        cache_read_tokens = usage_info["cache_read_tokens"] or 0
        input_tokens = usage_info["input_tokens"] or 0
        output_tokens = usage_info["output_tokens"] or 0
        total_tokens = input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens

        # Estimate cost using cost calculator
        model = str(raw_entry["message"]["model"])
        try:
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            )
            estimated_cost = self.cost_calculator.calculate_cost(model, token_usage)

            # Log cost estimation for transparency
            logger.debug(
                f"Estimated cost for model '{model}': ${estimated_cost:.6f} "
                f"(input: {input_tokens}, output: {output_tokens}, "
                f"cache_creation: {cache_creation_tokens}, cache_read: {cache_read_tokens})"
            )
        except Exception as e:
            logger.warning(f"Failed to estimate cost for model '{model}': {e}")
            estimated_cost = 0.0

        # Extract additional v1.0.9 fields if available
        uuid = raw_entry.get("uuid")
        version = raw_entry.get("version")
        ttft_ms = raw_entry.get("message", {}).get("ttftMs") if isinstance(raw_entry.get("message"), dict) else None

        try:
            processed_entry = ProcessedLogEntry(
                timestamp=timestamp,
                date_str=date_str,
                month_str=month_str,
                project_name=project_name,
                session_id=str(raw_entry["sessionId"]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=estimated_cost,
                model=model,
                cache_creation_tokens=usage_info["cache_creation_tokens"],
                cache_read_tokens=usage_info["cache_read_tokens"],
                raw_data=raw_entry,
                cost_estimated=True,
                cost_confidence="medium",
            )

            # Store additional v1.0.9 fields in raw_data for potential future use
            if uuid and processed_entry.raw_data is not None:
                processed_entry.raw_data["_parsed_uuid"] = uuid
            if version and processed_entry.raw_data is not None:
                processed_entry.raw_data["_parsed_version"] = version
            if ttft_ms and processed_entry.raw_data is not None:
                processed_entry.raw_data["_parsed_ttftMs"] = ttft_ms

            return processed_entry

        except (ValueError, TypeError) as e:
            raise LogParseError(f"Failed to create ProcessedLogEntry for v1.0.9 format: {e}")

    def _extract_v1_0_9_usage_info(self, raw_entry: Dict[str, Any]) -> Dict[str, int | None]:
        """
        Extract token usage information from v1.0.9 format entry.

        Args:
            raw_entry: Raw log entry dictionary

        Returns:
            Dictionary containing usage information
        """
        usage_info: Dict[str, int | None] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": None,
            "cache_read_tokens": None,
        }

        try:
            # Extract from message.usage (primary location for v1.0.9)
            if "message" in raw_entry and isinstance(raw_entry["message"], dict):
                usage = raw_entry["message"].get("usage", {})
                if isinstance(usage, dict):
                    # Standard token fields
                    usage_info["input_tokens"] = int(usage.get("input_tokens", 0))
                    usage_info["output_tokens"] = int(usage.get("output_tokens", 0))

                    # Cache-related fields with proper v1.0.9 field names
                    cache_creation = usage.get("cache_creation_input_tokens")
                    if cache_creation is not None:
                        usage_info["cache_creation_tokens"] = int(cache_creation)

                    cache_read = usage.get("cache_read_input_tokens")
                    if cache_read is not None:
                        usage_info["cache_read_tokens"] = int(cache_read)

            # Fallback to top-level fields if message.usage is missing or incomplete
            if usage_info["input_tokens"] == 0 and "input_tokens" in raw_entry:
                usage_info["input_tokens"] = int(raw_entry.get("input_tokens", 0))
            if usage_info["output_tokens"] == 0 and "output_tokens" in raw_entry:
                usage_info["output_tokens"] = int(raw_entry.get("output_tokens", 0))

        except (ValueError, TypeError) as e:
            logger.debug(f"Error extracting usage info from v1.0.9 entry: {e}")
            # Use fallback values if extraction fails
            pass

        return usage_info


def parse_log_file(log_file_path: Path, target_timezone: Optional[str] = None) -> List[ProcessedLogEntry]:
    """
    Parse a single JSON log file and return processed log entries.

    Args:
        log_file_path: Path to the JSON log file
        target_timezone: Target timezone for date calculation ("auto", "UTC", or timezone name)

    Returns:
        List of ProcessedLogEntry objects

    Raises:
        LogParseError: If the log file cannot be parsed
        FileNotFoundError: If the log file doesn't exist
    """
    if not log_file_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    if not log_file_path.is_file():
        raise LogParseError(f"Path is not a file: {log_file_path}")

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            # Check if it's a JSONL file (one JSON object per line)
            if log_file_path.suffix.lower() == ".jsonl":
                raw_entries = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            raw_entries.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse line {line_num} in {log_file_path}: {e}")
                            continue
            else:
                # Traditional JSON file
                raw_data = json.load(f)
                # Handle both single objects and arrays
                if isinstance(raw_data, dict):
                    raw_entries = [raw_data]
                elif isinstance(raw_data, list):
                    raw_entries = raw_data
                else:
                    raise LogParseError(f"Invalid JSON structure in {log_file_path}: expected object or array")

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LogParseError(f"Failed to parse JSON from {log_file_path}: {e}")
    except Exception as e:
        raise LogParseError(f"Unexpected error reading {log_file_path}: {e}")

    processed_entries = []
    project_name = _extract_project_name_from_path(log_file_path)

    # Create parser instance with automatic format detection
    parser = LogParser()

    for i, raw_entry in enumerate(raw_entries):
        try:
            processed_entry = parser.parse_log_entry(raw_entry, project_name, log_file_path, target_timezone)
            if processed_entry:
                processed_entries.append(processed_entry)
        except Exception as e:
            logger.debug(f"Failed to parse entry {i} in {log_file_path}: {e}")
            continue

    logger.debug(f"Parsed {len(processed_entries)} entries from {log_file_path}")
    return processed_entries


def _parse_single_log_entry(
    raw_entry: Dict[str, Any], project_name: str, file_path: Path, target_timezone: Optional[str] = None
) -> Optional[ProcessedLogEntry]:
    """
    Parse a single log entry dictionary into a ProcessedLogEntry.

    Args:
        raw_entry: Raw log entry dictionary
        project_name: Project name extracted from file path
        file_path: Source file path for context
        target_timezone: Target timezone for date calculation

    Returns:
        ProcessedLogEntry or None if entry should be skipped

    Raises:
        LogParseError: If required fields are missing or invalid
    """
    # Only process assistant entries (they have costUSD)
    if raw_entry.get("type") != "assistant":
        return None  # Skip user entries and other types

    # Check for required fields
    required_fields = ["timestamp", "sessionId", "costUSD"]
    missing_fields = [field for field in required_fields if field not in raw_entry]

    # Check for model field in message
    if "message" not in raw_entry or "model" not in raw_entry.get("message", {}):
        missing_fields.append("message.model")

    if missing_fields:
        raise LogParseError(f"Missing required fields: {missing_fields}")

    # Parse timestamp
    try:
        timestamp = _parse_timestamp(raw_entry["timestamp"])
    except ValueError as e:
        raise LogParseError(f"Invalid timestamp format: {e}")

    # Extract usage information
    usage_info = _extract_usage_info(raw_entry)

    # Generate date strings for aggregation with timezone consideration
    display_timestamp = _convert_to_target_timezone(timestamp, target_timezone)
    date_str = display_timestamp.strftime("%Y-%m-%d")
    month_str = display_timestamp.strftime("%Y-%m")

    # Calculate total tokens
    cache_creation_tokens = usage_info["cache_creation_tokens"] or 0
    cache_read_tokens = usage_info["cache_read_tokens"] or 0
    input_tokens = usage_info["input_tokens"] or 0
    output_tokens = usage_info["output_tokens"] or 0
    total_tokens = input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens

    try:
        return ProcessedLogEntry(
            timestamp=timestamp,
            date_str=date_str,
            month_str=month_str,
            project_name=project_name,
            session_id=str(raw_entry["sessionId"]),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=float(raw_entry["costUSD"]),
            model=str(raw_entry["message"]["model"]),
            cache_creation_tokens=usage_info["cache_creation_tokens"],
            cache_read_tokens=usage_info["cache_read_tokens"],
            raw_data=raw_entry,
        )
    except (ValueError, TypeError) as e:
        raise LogParseError(f"Failed to create ProcessedLogEntry: {e}")


def _parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO 8601 timestamp string into datetime object.

    Args:
        timestamp_str: ISO 8601 timestamp string

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        # Handle Z suffix (UTC timezone)
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"

        return datetime.fromisoformat(timestamp_str)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format '{timestamp_str}': {e}")


def _convert_to_target_timezone(timestamp: datetime, target_timezone: Optional[str] = None) -> datetime:
    """
    Convert timestamp to target timezone for date calculation.

    Args:
        timestamp: UTC datetime object
        target_timezone: Target timezone ("auto", "UTC", timezone name, or None)

    Returns:
        Datetime object in target timezone
    """
    if target_timezone is None or target_timezone == "UTC":
        return timestamp

    if target_timezone == "auto":
        # Use system local timezone
        return timestamp.astimezone()

    # Handle specific timezone names
    try:
        import zoneinfo

        target_tz = zoneinfo.ZoneInfo(target_timezone)
        return timestamp.astimezone(target_tz)
    except ImportError:
        # Fallback for older Python versions without zoneinfo
        logger.warning("zoneinfo not available, using system timezone")
        return timestamp.astimezone()
    except Exception as e:
        logger.warning(f"Invalid timezone '{target_timezone}': {e}, using system timezone")
        return timestamp.astimezone()


def _extract_usage_info(raw_entry: Dict[str, Any]) -> Dict[str, int | None]:
    """
    Extract token usage information from raw log entry.

    Args:
        raw_entry: Raw log entry dictionary

    Returns:
        Dictionary containing usage information
    """
    usage_info: Dict[str, int | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": None,
        "cache_read_tokens": None,
    }

    # Try to extract from message.usage first
    if "message" in raw_entry and isinstance(raw_entry["message"], dict):
        usage = raw_entry["message"].get("usage", {})
        if isinstance(usage, dict):
            usage_info["input_tokens"] = usage.get("input_tokens", 0)
            usage_info["output_tokens"] = usage.get("output_tokens", 0)
            usage_info["cache_creation_tokens"] = usage.get("cache_creation_input_tokens")
            usage_info["cache_read_tokens"] = usage.get("cache_read_input_tokens")

    # Fallback to top-level fields if available
    if usage_info["input_tokens"] == 0:
        usage_info["input_tokens"] = raw_entry.get("input_tokens", 0)
    if usage_info["output_tokens"] == 0:
        usage_info["output_tokens"] = raw_entry.get("output_tokens", 0)

    return usage_info


def _extract_project_name_from_path(log_file_path: Path) -> str:
    """
    Extract project name from log file path.

    Based on the assumption that logs are stored in ~/.claude/projects/<project_name>/...

    Args:
        log_file_path: Path to the log file

    Returns:
        Project name extracted from path
    """
    path_parts = log_file_path.parts

    # Look for 'projects' directory in the path
    try:
        projects_index = path_parts.index("projects")
        if projects_index + 1 < len(path_parts):
            project_dir = path_parts[projects_index + 1]
            # Clean up the project name (remove special characters, etc.)
            return _clean_project_name(project_dir)
    except ValueError:
        pass

    # Fallback: use the parent directory name
    return _clean_project_name(log_file_path.parent.name)


def _clean_project_name(project_name: str) -> str:
    """
    Clean up project name by replacing special characters.

    Args:
        project_name: Raw project name

    Returns:
        Cleaned project name
    """
    # Replace common path separators and special characters
    cleaned = project_name.replace("-Users-", "").replace("-", "_")

    # Remove leading/trailing underscores
    cleaned = cleaned.strip("_")

    # If empty after cleaning, use default
    if not cleaned:
        cleaned = "unknown_project"

    return cleaned


def parse_multiple_log_files(log_file_paths: List[Path], target_timezone: Optional[str] = None) -> List[ProcessedLogEntry]:
    """
    Parse multiple log files and return all processed entries.

    Args:
        log_file_paths: List of paths to log files
        target_timezone: Target timezone for date calculation

    Returns:
        List of all ProcessedLogEntry objects from all files
    """
    all_entries = []

    for log_file_path in log_file_paths:
        try:
            entries = parse_log_file(log_file_path, target_timezone)
            all_entries.extend(entries)
        except Exception as e:
            logger.error(f"Failed to parse {log_file_path}: {e}")
            continue

    logger.debug(f"Parsed total of {len(all_entries)} entries from {len(log_file_paths)} files")
    return all_entries
