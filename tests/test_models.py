"""
Tests for the models module.
"""

from datetime import datetime

import pytest

from claude_code_cost_collector.models import (
    ProcessedLogEntry,
    create_sample_processed_log_entry,
)


class TestProcessedLogEntry:
    """Test the ProcessedLogEntry dataclass."""

    def test_create_minimal_entry(self):
        """Test creating ProcessedLogEntry with minimal required fields."""
        entry = ProcessedLogEntry(
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

        assert entry.timestamp == datetime(2025, 5, 9, 12, 0, 0)
        assert entry.date_str == "2025-05-09"
        assert entry.month_str == "2025-05"
        assert entry.project_name == "test_project"
        assert entry.session_id == "test_session"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.total_tokens == 150
        assert entry.cost_usd == 0.01
        assert entry.model == "claude-3"

        # Optional fields should have default values
        assert entry.cache_creation_tokens is None
        assert entry.cache_read_tokens is None

    def test_create_full_entry(self):
        """Test creating ProcessedLogEntry with all fields."""
        entry = ProcessedLogEntry(
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
            cache_creation_tokens=1000,
            cache_read_tokens=500,
        )

        assert entry.cache_creation_tokens == 1000
        assert entry.cache_read_tokens == 500

    def test_entry_equality(self):
        """Test that two identical entries are equal."""
        entry1 = ProcessedLogEntry(
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

        entry2 = ProcessedLogEntry(
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

        assert entry1 == entry2

    def test_entry_inequality(self):
        """Test that different entries are not equal."""
        entry1 = ProcessedLogEntry(
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

        entry2 = ProcessedLogEntry(
            timestamp=datetime(2025, 5, 9, 12, 0, 0),
            date_str="2025-05-09",
            month_str="2025-05",
            project_name="test_project",
            session_id="test_session",
            input_tokens=200,  # Different input tokens
            output_tokens=50,
            total_tokens=250,
            cost_usd=0.01,
            model="claude-3",
        )

        assert entry1 != entry2

    def test_entry_representation(self):
        """Test string representation of ProcessedLogEntry."""
        entry = ProcessedLogEntry(
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

        repr_str = repr(entry)
        assert "ProcessedLogEntry" in repr_str
        assert "test_project" in repr_str
        assert "test_session" in repr_str

    def test_entry_hashing(self):
        """Test that ProcessedLogEntry instances are unhashable by default."""
        entry = ProcessedLogEntry(
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

        # Dataclasses are unhashable by default
        with pytest.raises(TypeError, match="unhashable type"):
            hash(entry)

    def test_entry_field_types(self):
        """Test that entry fields have correct types."""
        entry = ProcessedLogEntry(
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
            cache_creation_tokens=1000,
            cache_read_tokens=500,
        )

        assert isinstance(entry.timestamp, datetime)
        assert isinstance(entry.date_str, str)
        assert isinstance(entry.month_str, str)
        assert isinstance(entry.project_name, str)
        assert isinstance(entry.session_id, str)
        assert isinstance(entry.input_tokens, int)
        assert isinstance(entry.output_tokens, int)
        assert isinstance(entry.total_tokens, int)
        assert isinstance(entry.cost_usd, float)
        assert isinstance(entry.model, str)
        assert isinstance(entry.cache_creation_tokens, int)
        assert isinstance(entry.cache_read_tokens, int)

    def test_optional_fields_none(self):
        """Test that optional fields can be None."""
        entry = ProcessedLogEntry(
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
            cache_creation_tokens=None,
            cache_read_tokens=None,
        )

        assert entry.cache_creation_tokens is None
        assert entry.cache_read_tokens is None


class TestCreateSampleProcessedLogEntry:
    """Test the create_sample_processed_log_entry function."""

    def test_create_sample_entry(self):
        """Test creating a sample entry."""
        entry = create_sample_processed_log_entry()

        assert isinstance(entry, ProcessedLogEntry)
        assert isinstance(entry.timestamp, datetime)
        assert isinstance(entry.date_str, str)
        assert isinstance(entry.month_str, str)
        assert isinstance(entry.project_name, str)
        assert isinstance(entry.session_id, str)
        assert isinstance(entry.input_tokens, int)
        assert isinstance(entry.output_tokens, int)
        assert isinstance(entry.total_tokens, int)
        assert isinstance(entry.cost_usd, float)
        assert isinstance(entry.model, str)

    def test_sample_entry_values(self):
        """Test that sample entry has expected values."""
        entry = create_sample_processed_log_entry()

        assert entry.timestamp == datetime(2025, 5, 9, 12, 3, 20)
        assert entry.date_str == "2025-05-09"
        assert entry.month_str == "2025-05"
        assert entry.project_name == "code_review"
        assert entry.session_id == "9d2a9923-5653-4bec-bc95-4ffd838a6736"
        assert entry.input_tokens == 4
        assert entry.output_tokens == 111
        assert entry.cache_creation_tokens == 24399
        assert entry.cache_read_tokens == 0
        assert entry.total_tokens == 24514  # 4 + 111 + 24399
        assert entry.cost_usd == 0.09317325
        assert entry.model == "claude-3-7-sonnet-20250219"

    def test_sample_entry_consistency(self):
        """Test that sample entry is consistent across calls."""
        entry1 = create_sample_processed_log_entry()
        entry2 = create_sample_processed_log_entry()

        assert entry1 == entry2

    def test_sample_entry_total_tokens_calculation(self):
        """Test that total tokens is correctly calculated."""
        entry = create_sample_processed_log_entry()

        expected_total = entry.input_tokens + entry.output_tokens + entry.cache_creation_tokens
        assert entry.total_tokens == expected_total

    def test_sample_entry_optional_fields(self):
        """Test that sample entry has appropriate optional field values."""
        entry = create_sample_processed_log_entry()

        # Cache creation and read tokens should be set
        assert entry.cache_creation_tokens is not None
        assert entry.cache_read_tokens is not None

    def test_sample_entry_realistic_data(self):
        """Test that sample entry contains realistic data."""
        entry = create_sample_processed_log_entry()

        # Validate timestamp is reasonable
        assert entry.timestamp.year >= 2024
        assert 1 <= entry.timestamp.month <= 12
        assert 1 <= entry.timestamp.day <= 31

        # Validate tokens are positive
        assert entry.input_tokens >= 0
        assert entry.output_tokens >= 0
        assert entry.total_tokens >= 0

        # Validate cost is positive
        assert entry.cost_usd >= 0

        # Validate strings are non-empty
        assert len(entry.project_name) > 0
        assert len(entry.session_id) > 0
        assert len(entry.model) > 0

    def test_sample_entry_date_strings_match_timestamp(self):
        """Test that date strings match the timestamp."""
        entry = create_sample_processed_log_entry()

        expected_date_str = entry.timestamp.strftime("%Y-%m-%d")
        expected_month_str = entry.timestamp.strftime("%Y-%m")

        assert entry.date_str == expected_date_str
        assert entry.month_str == expected_month_str

    def test_sample_entry_session_id_format(self):
        """Test that session ID has expected UUID format."""
        entry = create_sample_processed_log_entry()

        # Basic validation that it looks like a UUID
        assert len(entry.session_id) == 36
        assert entry.session_id.count("-") == 4

    def test_sample_entry_model_format(self):
        """Test that model name has expected format."""
        entry = create_sample_processed_log_entry()

        # Should start with "claude"
        assert entry.model.startswith("claude")
        assert "sonnet" in entry.model.lower()


if __name__ == "__main__":
    pytest.main([__file__])
