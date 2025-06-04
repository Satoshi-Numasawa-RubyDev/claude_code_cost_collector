"""
Log format detection module for Claude Code Cost Collector.

This module provides functionality to detect different Claude CLI log formats
and determine the appropriate parsing strategy based on format characteristics.
"""

import json
from enum import Enum
from typing import Any, Dict, List


class LogFormat(Enum):
    """
    Enumeration of supported Claude CLI log formats.

    LEGACY: Traditional format with costUSD field present
    V1_0_9: New format (v1.0.9+) without costUSD field, with ttftMs and uuid
    UNKNOWN: Format cannot be determined
    """

    LEGACY = "legacy"
    V1_0_9 = "v1.0.9"
    UNKNOWN = "unknown"


class FormatDetector:
    """
    Detects Claude CLI log format based on entry characteristics.

    This class analyzes log entries to determine which format they conform to,
    enabling appropriate parsing strategies for different Claude CLI versions.
    """

    @staticmethod
    def detect_entry_format(entry: Dict[str, Any]) -> LogFormat:
        """
        Detect the format of a single log entry.

        Args:
            entry: Dictionary representing a single log entry

        Returns:
            LogFormat enum indicating the detected format
        """
        if not isinstance(entry, dict):
            return LogFormat.UNKNOWN

        # Skip non-assistant entries for format detection
        if entry.get("type") != "assistant":
            return LogFormat.UNKNOWN

        # Check for v1.0.9 indicators
        if FormatDetector._is_v1_0_9_format(entry):
            return LogFormat.V1_0_9

        # Check for legacy format indicators
        if FormatDetector._is_legacy_format(entry):
            return LogFormat.LEGACY

        return LogFormat.UNKNOWN

    @staticmethod
    def detect_file_format(file_path: str) -> LogFormat:
        """
        Detect the format of an entire log file.

        Analyzes multiple entries in a file to determine the predominant format.
        Uses a sampling approach for performance with large files.

        Args:
            file_path: Path to the log file to analyze

        Returns:
            LogFormat enum indicating the detected format
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                format_votes = {LogFormat.LEGACY: 0, LogFormat.V1_0_9: 0, LogFormat.UNKNOWN: 0}

                entries_checked = 0
                max_entries_to_check = 10  # Sample first 10 assistant entries

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "assistant":
                            format_result = FormatDetector.detect_entry_format(entry)
                            format_votes[format_result] += 1
                            entries_checked += 1

                            if entries_checked >= max_entries_to_check:
                                break

                    except json.JSONDecodeError:
                        continue

                # Determine predominant format
                if format_votes[LogFormat.V1_0_9] > 0:
                    return LogFormat.V1_0_9
                elif format_votes[LogFormat.LEGACY] > 0:
                    return LogFormat.LEGACY
                else:
                    return LogFormat.UNKNOWN

        except (FileNotFoundError, IOError, UnicodeDecodeError):
            return LogFormat.UNKNOWN

    @staticmethod
    def detect_files_format(file_paths: List[str]) -> LogFormat:
        """
        Detect the format across multiple log files.

        Analyzes multiple files and returns the most common format.
        Useful for determining the format when processing multiple files.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            LogFormat enum indicating the predominant format
        """
        if not file_paths:
            return LogFormat.UNKNOWN

        format_votes = {LogFormat.LEGACY: 0, LogFormat.V1_0_9: 0, LogFormat.UNKNOWN: 0}

        for file_path in file_paths:
            file_format = FormatDetector.detect_file_format(file_path)
            format_votes[file_format] += 1

        # Return the format with the most votes
        # Priority: V1_0_9 > LEGACY > UNKNOWN
        if format_votes[LogFormat.V1_0_9] > 0:
            return LogFormat.V1_0_9
        elif format_votes[LogFormat.LEGACY] > 0:
            return LogFormat.LEGACY
        else:
            return LogFormat.UNKNOWN

    @staticmethod
    def _is_v1_0_9_format(entry: Dict[str, Any]) -> bool:
        """
        Check if entry matches v1.0.9 format characteristics.

        v1.0.9 format indicators:
        - Missing costUSD field
        - Present ttftMs field in message
        - Present uuid field
        - Version field with "1.0.9" or higher

        Args:
            entry: Log entry to check

        Returns:
            True if entry matches v1.0.9 format
        """
        # Primary indicator: missing costUSD field
        has_cost_usd = "costUSD" in entry

        # Secondary indicators
        message = entry.get("message", {})
        has_ttft = isinstance(message, dict) and "ttftMs" in message
        has_uuid = "uuid" in entry

        # Version indicator (if present)
        version = entry.get("version", "")
        is_new_version = False
        if version:
            try:
                # Simple version comparison for "1.0.9" or higher
                version_parts = version.split(".")
                if len(version_parts) >= 3:
                    major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                    is_new_version = (major > 1) or (major == 1 and minor > 0) or (major == 1 and minor == 0 and patch >= 9)
            except (ValueError, IndexError):
                pass

        # Strong indicators for v1.0.9
        strong_indicators = sum(
            [
                not has_cost_usd,  # Primary: missing costUSD
                has_ttft,  # Secondary: ttftMs present
                has_uuid,  # Secondary: uuid present
                is_new_version,  # Tertiary: version 1.0.9+
            ]
        )

        # Require at least 2 strong indicators, with missing costUSD being mandatory
        return not has_cost_usd and strong_indicators >= 2

    @staticmethod
    def _is_legacy_format(entry: Dict[str, Any]) -> bool:
        """
        Check if entry matches legacy format characteristics.

        Legacy format indicators:
        - Present costUSD field
        - Required fields: timestamp, sessionId, message.model
        - Absence of v1.0.9 specific fields

        Args:
            entry: Log entry to check

        Returns:
            True if entry matches legacy format
        """
        # Primary indicator: presence of costUSD field
        has_cost_usd = "costUSD" in entry

        # Required fields for legacy format
        required_fields = ["timestamp", "sessionId"]
        has_required = all(field in entry for field in required_fields)

        # Message structure validation
        message = entry.get("message", {})
        has_model = isinstance(message, dict) and "model" in message

        # Legacy format should have costUSD and required fields
        return has_cost_usd and has_required and has_model

    @staticmethod
    def get_format_confidence(entry: Dict[str, Any]) -> Dict[LogFormat, float]:
        """
        Get confidence scores for each format.

        Provides detailed confidence scores for format detection,
        useful for debugging and fine-tuning detection logic.

        Args:
            entry: Log entry to analyze

        Returns:
            Dictionary mapping LogFormat to confidence score (0.0-1.0)
        """
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            return {LogFormat.LEGACY: 0.0, LogFormat.V1_0_9: 0.0, LogFormat.UNKNOWN: 1.0}

        # Calculate confidence scores
        legacy_score = FormatDetector._calculate_legacy_confidence(entry)
        v1_0_9_score = FormatDetector._calculate_v1_0_9_confidence(entry)

        # Normalize scores
        total_score = legacy_score + v1_0_9_score
        if total_score == 0:
            return {LogFormat.LEGACY: 0.0, LogFormat.V1_0_9: 0.0, LogFormat.UNKNOWN: 1.0}

        unknown_score = max(0.0, 1.0 - total_score)

        return {LogFormat.LEGACY: legacy_score, LogFormat.V1_0_9: v1_0_9_score, LogFormat.UNKNOWN: unknown_score}

    @staticmethod
    def _calculate_legacy_confidence(entry: Dict[str, Any]) -> float:
        """Calculate confidence score for legacy format."""
        score = 0.0

        # Strong indicators
        if "costUSD" in entry:
            score += 0.5

        # Required fields
        required_fields = ["timestamp", "sessionId"]
        if all(field in entry for field in required_fields):
            score += 0.3

        # Message structure
        message = entry.get("message", {})
        if isinstance(message, dict) and "model" in message:
            score += 0.2

        return min(1.0, score)

    @staticmethod
    def _calculate_v1_0_9_confidence(entry: Dict[str, Any]) -> float:
        """Calculate confidence score for v1.0.9 format."""
        score = 0.0

        # Primary indicator: missing costUSD
        if "costUSD" not in entry:
            score += 0.4

        # Secondary indicators
        message = entry.get("message", {})
        if isinstance(message, dict) and "ttftMs" in message:
            score += 0.3

        if "uuid" in entry:
            score += 0.2

        # Version check
        version = entry.get("version", "")
        if version:
            try:
                version_parts = version.split(".")
                if len(version_parts) >= 3:
                    major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                    if (major > 1) or (major == 1 and minor > 0) or (major == 1 and minor == 0 and patch >= 9):
                        score += 0.1
            except (ValueError, IndexError):
                pass

        return min(1.0, score)
