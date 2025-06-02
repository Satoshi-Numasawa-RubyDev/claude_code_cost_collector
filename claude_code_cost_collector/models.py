"""
Core data models for Claude Code Cost Collector.

This module defines the primary internal data structures used across
the parser, aggregator, and formatter modules.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ProcessedLogEntry:
    """
    Processed log entry representing a single Claude API interaction.

    This is the core data structure used throughout the application for
    representing parsed log data in a standardized format.

    Attributes:
        timestamp: Parsed datetime object from the original log timestamp
        date_str: Date string in YYYY-MM-DD format for daily aggregation
        month_str: Year-month string in YYYY-MM format for monthly aggregation
        project_name: Project name derived from the log file directory structure
        session_id: Session ID from the original log entry
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens (input + output + cache tokens if applicable)
        cost_usd: Cost in USD as recorded in the original log
        model: Model name used for the API call
        converted_cost: Optional cost in target currency
        target_currency: Currency code for converted cost
        cache_creation_tokens: Optional cache creation tokens (included in cost)
        cache_read_tokens: Optional cache read tokens (typically not charged)
        raw_data: Optional dictionary containing additional raw log data
    """

    timestamp: datetime
    date_str: str
    month_str: str
    project_name: str
    session_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str
    converted_cost: Optional[float] = None
    target_currency: Optional[str] = None
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate and normalize data after initialization."""
        # Ensure total_tokens is correctly calculated if not explicitly set
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
            if self.cache_creation_tokens:
                self.total_tokens += self.cache_creation_tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ProcessedLogEntry to a dictionary.

        Returns:
            Dictionary representation suitable for serialization
        """
        result = {
            "timestamp": self.timestamp.isoformat(),
            "date_str": self.date_str,
            "month_str": self.month_str,
            "project_name": self.project_name,
            "session_id": self.session_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "model": self.model,
        }

        # Add optional fields if they exist
        if self.converted_cost is not None:
            result["converted_cost"] = self.converted_cost
        if self.target_currency is not None:
            result["target_currency"] = self.target_currency
        if self.cache_creation_tokens is not None:
            result["cache_creation_tokens"] = self.cache_creation_tokens
        if self.cache_read_tokens is not None:
            result["cache_read_tokens"] = self.cache_read_tokens
        if self.raw_data is not None:
            result["raw_data"] = self.raw_data

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedLogEntry":
        """
        Create a ProcessedLogEntry from a dictionary.

        Args:
            data: Dictionary containing log entry data

        Returns:
            ProcessedLogEntry instance
        """
        # Parse timestamp if it's a string
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return cls(
            timestamp=timestamp,
            date_str=data["date_str"],
            month_str=data["month_str"],
            project_name=data["project_name"],
            session_id=data["session_id"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
            cost_usd=data["cost_usd"],
            model=data["model"],
            converted_cost=data.get("converted_cost"),
            target_currency=data.get("target_currency"),
            cache_creation_tokens=data.get("cache_creation_tokens"),
            cache_read_tokens=data.get("cache_read_tokens"),
            raw_data=data.get("raw_data"),
        )


# Type aliases for commonly used data structures
ProcessedLogEntries = list[ProcessedLogEntry]
AggregatedData = Dict[str, Dict[str, Any]]


def create_sample_processed_log_entry() -> ProcessedLogEntry:
    """
    Create a sample ProcessedLogEntry for testing and documentation purposes.

    Returns:
        Sample ProcessedLogEntry with realistic data
    """
    return ProcessedLogEntry(
        timestamp=datetime(2025, 5, 9, 12, 3, 20),
        date_str="2025-05-09",
        month_str="2025-05",
        project_name="code_review",
        session_id="9d2a9923-5653-4bec-bc95-4ffd838a6736",
        input_tokens=4,
        output_tokens=111,
        total_tokens=24514,  # Includes cache creation tokens
        cost_usd=0.09317325,
        model="claude-3-7-sonnet-20250219",
        cache_creation_tokens=24399,
        cache_read_tokens=0,
    )
