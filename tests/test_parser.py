"""
Unit tests for the parser module.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_code_cost_collector.models import ProcessedLogEntry
from claude_code_cost_collector.parser import (
    LogParseError,
    _clean_project_name,
    _extract_project_name_from_path,
    _extract_usage_info,
    _parse_single_log_entry,
    _parse_timestamp,
    parse_log_file,
    parse_multiple_log_files,
)


class TestParseLogFile:
    """Test parse_log_file function."""

    def test_parse_valid_single_entry_file(self):
        """Test parsing a valid JSON file with a single log entry."""
        log_data = {
            "type": "assistant",
            "uuid": "9959f193-81ad-4d35-8f3e-5eb25def1cbd",
            "sessionId": "9d2a9923-5653-4bec-bc95-4ffd838a6736",
            "timestamp": "2025-05-09T12:03:20.000Z",
            "cwd": "/Users/testuser/source/1000_project/code_review",
            "costUSD": 0.09317325,
            "message": {
                "model": "claude-3-7-sonnet-20250219",
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 24399,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 111,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            # Create a path structure that simulates ~/.claude/projects/test_project/

            with tempfile.TemporaryDirectory() as temp_dir:
                projects_dir = Path(temp_dir) / "projects" / "test_project"
                projects_dir.mkdir(parents=True)
                log_file = projects_dir / "logs.json"

                # Copy the temp file content to the proper location
                with open(temp_path, "r", encoding="utf-8") as src:
                    with open(log_file, "w", encoding="utf-8") as dst:
                        dst.write(src.read())

                entries = parse_log_file(log_file)

            assert len(entries) == 1
            entry = entries[0]

            assert isinstance(entry, ProcessedLogEntry)
            assert entry.session_id == "9d2a9923-5653-4bec-bc95-4ffd838a6736"
            assert entry.timestamp == datetime(2025, 5, 9, 12, 3, 20, tzinfo=timezone.utc)
            assert entry.date_str == "2025-05-09"
            assert entry.month_str == "2025-05"
            assert entry.project_name == "test_project"
            assert entry.input_tokens == 4
            assert entry.output_tokens == 111
            assert entry.cache_creation_tokens == 24399
            assert entry.cache_read_tokens == 0
            assert entry.total_tokens == 24514  # 4 + 111 + 24399
            assert entry.cost_usd == 0.09317325
            assert entry.model == "claude-3-7-sonnet-20250219"

        finally:
            temp_path.unlink()

    def test_parse_array_of_entries(self):
        """Test parsing a JSON file with an array of log entries."""
        log_data = [
            {
                "type": "assistant",
                "sessionId": "session1",
                "timestamp": "2025-05-09T12:03:20.000Z",
                "costUSD": 0.1,
                "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
            },
            {
                "type": "assistant",
                "sessionId": "session2",
                "timestamp": "2025-05-09T13:03:20.000Z",
                "costUSD": 0.2,
                "message": {"model": "claude-3-haiku", "usage": {"input_tokens": 15, "output_tokens": 25}},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            # Create a path structure that simulates ~/.claude/projects/multi_project/
            with tempfile.TemporaryDirectory() as temp_dir:
                projects_dir = Path(temp_dir) / "projects" / "multi_project"
                projects_dir.mkdir(parents=True)
                log_file = projects_dir / "logs.json"

                # Copy the temp file content to the proper location
                with open(temp_path, "r", encoding="utf-8") as src:
                    with open(log_file, "w", encoding="utf-8") as dst:
                        dst.write(src.read())

                entries = parse_log_file(log_file)

            assert len(entries) == 2
            assert entries[0].session_id == "session1"
            assert entries[1].session_id == "session2"
            assert entries[0].project_name == "multi_project"
            assert entries[1].project_name == "multi_project"

        finally:
            temp_path.unlink()

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        non_existent_path = Path("/non/existent/file.json")

        with pytest.raises(FileNotFoundError):
            parse_log_file(non_existent_path)

    def test_invalid_json(self):
        """Test handling of invalid JSON files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_path = Path(f.name)

        try:
            with pytest.raises(LogParseError, match="Failed to parse JSON"):
                parse_log_file(temp_path)
        finally:
            temp_path.unlink()

    def test_missing_required_fields(self):
        """Test handling of log entries missing required fields."""
        log_data = {
            "sessionId": "test-session",
            # Missing timestamp, costUSD, model
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            # Should not raise exception but log warning and skip entry
            entries = parse_log_file(temp_path)
            assert len(entries) == 0
        finally:
            temp_path.unlink()


class TestParseSingleLogEntry:
    """Test _parse_single_log_entry function."""

    def test_minimal_valid_entry(self):
        """Test parsing a minimal valid log entry."""
        raw_entry = {
            "type": "assistant",
            "sessionId": "test-session",
            "timestamp": "2025-05-09T12:03:20.000Z",
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }

        entry = _parse_single_log_entry(raw_entry, "test_project", Path("/test/path"))

        assert entry is not None
        assert entry.session_id == "test-session"
        assert entry.project_name == "test_project"
        assert entry.input_tokens == 10
        assert entry.output_tokens == 20
        assert entry.total_tokens == 30
        assert entry.cost_usd == 0.1
        assert entry.model == "claude-3-sonnet"

    def test_entry_with_cache_tokens(self):
        """Test parsing entry with cache creation and read tokens."""
        raw_entry = {
            "type": "assistant",
            "sessionId": "test-session",
            "timestamp": "2025-05-09T12:03:20.000Z",
            "costUSD": 0.1,
            "message": {
                "model": "claude-3-sonnet",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 50,
                },
            },
        }

        entry = _parse_single_log_entry(raw_entry, "test_project", Path("/test/path"))

        assert entry.cache_creation_tokens == 100
        assert entry.cache_read_tokens == 50
        assert entry.total_tokens == 180  # 10 + 20 + 100 + 50

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        raw_entry = {
            "type": "assistant",
            "sessionId": "test-session",
            # Missing timestamp, costUSD, message.model
        }

        with pytest.raises(LogParseError, match="Missing required fields"):
            _parse_single_log_entry(raw_entry, "test_project", Path("/test/path"))

    def test_skip_user_entries(self):
        """Test that user entries are skipped."""
        raw_entry = {
            "type": "user",
            "sessionId": "test-session",
            "timestamp": "2025-05-09T12:03:20.000Z",
            "message": {"role": "user", "content": "Hello"},
        }

        entry = _parse_single_log_entry(raw_entry, "test_project", Path("/test/path"))
        assert entry is None


class TestParseTimestamp:
    """Test _parse_timestamp function."""

    def test_utc_timestamp_with_z(self):
        """Test parsing UTC timestamp with Z suffix."""
        timestamp_str = "2025-05-09T12:03:20.000Z"
        result = _parse_timestamp(timestamp_str)

        expected = datetime(2025, 5, 9, 12, 3, 20, tzinfo=timezone.utc)
        assert result == expected

    def test_timestamp_with_timezone(self):
        """Test parsing timestamp with timezone offset."""
        timestamp_str = "2025-05-09T12:03:20.000+09:00"
        result = _parse_timestamp(timestamp_str)

        # Should preserve timezone information
        assert result.year == 2025
        assert result.month == 5
        assert result.day == 9
        assert result.hour == 12

    def test_invalid_timestamp(self):
        """Test handling of invalid timestamp formats."""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            _parse_timestamp("invalid-timestamp")


class TestExtractUsageInfo:
    """Test _extract_usage_info function."""

    def test_usage_from_message(self):
        """Test extracting usage info from message.usage."""
        raw_entry = {
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 50,
                }
            }
        }

        usage_info = _extract_usage_info(raw_entry)

        assert usage_info["input_tokens"] == 10
        assert usage_info["output_tokens"] == 20
        assert usage_info["cache_creation_tokens"] == 100
        assert usage_info["cache_read_tokens"] == 50

    def test_usage_fallback_to_top_level(self):
        """Test fallback to top-level fields."""
        raw_entry = {"input_tokens": 15, "output_tokens": 25}

        usage_info = _extract_usage_info(raw_entry)

        assert usage_info["input_tokens"] == 15
        assert usage_info["output_tokens"] == 25
        assert usage_info["cache_creation_tokens"] is None
        assert usage_info["cache_read_tokens"] is None

    def test_usage_no_data(self):
        """Test handling when no usage data is available."""
        raw_entry = {}

        usage_info = _extract_usage_info(raw_entry)

        assert usage_info["input_tokens"] == 0
        assert usage_info["output_tokens"] == 0
        assert usage_info["cache_creation_tokens"] is None
        assert usage_info["cache_read_tokens"] is None


class TestExtractProjectName:
    """Test _extract_project_name_from_path function."""

    def test_projects_directory_in_path(self):
        """Test extracting project name when 'projects' is in path."""
        path = Path("/Users/test/.claude/projects/my_project/subdirectory/logs.json")
        result = _extract_project_name_from_path(path)

        assert result == "my_project"

    def test_complex_project_name(self):
        """Test extracting and cleaning complex project names."""
        path = Path("/Users/test/.claude/projects/-Users-testuser-source-1000-project-code-review/logs.json")
        result = _extract_project_name_from_path(path)

        assert result == "testuser_source_1000_project_code_review"

    def test_fallback_to_parent_directory(self):
        """Test fallback to parent directory when 'projects' not found."""
        path = Path("/some/other/path/my_project_dir/logs.json")
        result = _extract_project_name_from_path(path)

        assert result == "my_project_dir"


class TestCleanProjectName:
    """Test _clean_project_name function."""

    def test_remove_users_prefix(self):
        """Test removing -Users- prefix."""
        result = _clean_project_name("-Users-testuser-project-name")
        assert result == "testuser_project_name"

    def test_replace_hyphens_with_underscores(self):
        """Test replacing hyphens with underscores."""
        result = _clean_project_name("my-project-name")
        assert result == "my_project_name"

    def test_strip_underscores(self):
        """Test stripping leading/trailing underscores."""
        result = _clean_project_name("_my_project_")
        assert result == "my_project"

    def test_empty_string_fallback(self):
        """Test fallback for empty strings."""
        result = _clean_project_name("")
        assert result == "unknown_project"

        result = _clean_project_name("___")
        assert result == "unknown_project"


class TestParseMultipleLogFiles:
    """Test parse_multiple_log_files function."""

    def test_parse_multiple_files(self):
        """Test parsing multiple log files."""
        # Create two temporary files
        log_data1 = {
            "type": "assistant",
            "sessionId": "session1",
            "timestamp": "2025-05-09T12:03:20.000Z",
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }
        log_data2 = {
            "type": "assistant",
            "sessionId": "session2",
            "timestamp": "2025-05-09T13:03:20.000Z",
            "costUSD": 0.2,
            "message": {"model": "claude-3-haiku", "usage": {"input_tokens": 15, "output_tokens": 25}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(log_data1, f1)
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(log_data2, f2)
            temp_path2 = Path(f2.name)

        try:
            entries = parse_multiple_log_files([temp_path1, temp_path2])

            assert len(entries) == 2
            session_ids = [entry.session_id for entry in entries]
            assert "session1" in session_ids
            assert "session2" in session_ids

        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_handle_invalid_files(self):
        """Test that invalid files are skipped without stopping processing."""
        # Create one valid and one invalid file
        valid_log_data = {
            "type": "assistant",
            "sessionId": "session1",
            "timestamp": "2025-05-09T12:03:20.000Z",
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(valid_log_data, f1)
            valid_path = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            f2.write("invalid json")
            invalid_path = Path(f2.name)

        try:
            entries = parse_multiple_log_files([valid_path, invalid_path])

            # Should get one valid entry, invalid file should be skipped
            assert len(entries) == 1
            assert entries[0].session_id == "session1"

        finally:
            valid_path.unlink()
            invalid_path.unlink()


class TestTimezoneHandling:
    """Test timezone handling in date calculations."""

    def test_utc_timezone_setting(self):
        """Test that UTC timezone preserves original date."""
        log_data = {
            "type": "assistant",
            "sessionId": "session1",
            "timestamp": "2025-05-31T23:00:00.000Z",  # UTC 23:00
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            entries = parse_log_file(temp_path, target_timezone="UTC")
            assert len(entries) == 1
            assert entries[0].date_str == "2025-05-31"

        finally:
            temp_path.unlink()

    def test_jst_timezone_setting(self):
        """Test that JST timezone converts date appropriately."""
        log_data = {
            "type": "assistant",
            "sessionId": "session1",
            "timestamp": "2025-05-31T23:00:00.000Z",  # UTC 23:00 = JST 08:00 next day
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            entries = parse_log_file(temp_path, target_timezone="Asia/Tokyo")
            assert len(entries) == 1
            assert entries[0].date_str == "2025-06-01"

        finally:
            temp_path.unlink()

    def test_auto_timezone_setting(self):
        """Test that auto timezone uses system timezone."""
        log_data = {
            "type": "assistant",
            "sessionId": "session1",
            "timestamp": "2025-05-31T10:00:00.000Z",  # UTC 10:00 = JST 19:00 same day
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            # With auto, should use system timezone (JST in this environment)
            entries = parse_log_file(temp_path, target_timezone="auto")
            assert len(entries) == 1
            assert entries[0].date_str == "2025-05-31"  # Same day in JST

        finally:
            temp_path.unlink()

    def test_none_timezone_defaults_to_utc(self):
        """Test that None timezone defaults to UTC behavior."""
        log_data = {
            "type": "assistant",
            "sessionId": "session1",
            "timestamp": "2025-05-31T23:00:00.000Z",  # UTC 23:00
            "costUSD": 0.1,
            "message": {"model": "claude-3-sonnet", "usage": {"input_tokens": 10, "output_tokens": 20}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f)
            temp_path = Path(f.name)

        try:
            entries = parse_log_file(temp_path, target_timezone=None)
            assert len(entries) == 1
            assert entries[0].date_str == "2025-05-31"  # UTC behavior

        finally:
            temp_path.unlink()
