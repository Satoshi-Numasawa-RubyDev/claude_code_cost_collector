"""
Tests for the format_detector module.

This module tests the log format detection functionality to ensure
accurate identification of different Claude CLI log formats.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from claude_code_cost_collector.format_detector import FormatDetector, LogFormat


class TestFormatDetector:
    """Test cases for FormatDetector class."""

    def test_detect_v1_0_9_format(self) -> None:
        """Test detection of v1.0.9 format entries."""
        # Sample v1.0.9 entry (based on actual new format)
        v1_0_9_entry = {
            "parentUuid": "aa87c72a-a2b7-4db0-93f3-0892afe18d40",
            "isSidechain": False,
            "userType": "external",
            "cwd": "/Users/test/project",
            "sessionId": "0da08427-d1b9-4055-8314-83d3f147b157",
            "version": "1.0.9",
            "message": {
                "id": "msg_bdrk_01D6C9sdmn7o8sNa3ZCqrBeV",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "Test response"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 6094,
                    "cache_read_input_tokens": 13558,
                    "output_tokens": 252,
                },
                "ttftMs": 5742,
            },
            "type": "assistant",
            "uuid": "13f77474-e274-4692-ad57-772d224a5e06",
            "timestamp": "2025-06-03T12:35:51.791Z",
        }

        result = FormatDetector.detect_entry_format(v1_0_9_entry)
        assert result == LogFormat.V1_0_9

    def test_detect_legacy_format(self) -> None:
        """Test detection of legacy format entries."""
        # Sample legacy entry (based on current format spec)
        legacy_entry = {
            "uuid": "test-uuid-1",
            "parentUuid": "test-parent-1",
            "sessionId": "session-1",
            "timestamp": "2025-05-09T10:30:00.000Z",
            "type": "assistant",
            "cwd": "/Users/test/project1",
            "originalCwd": "/Users/test/project1",
            "userType": "external",
            "version": "0.2.104",
            "isSidechain": False,
            "costUSD": 0.045,
            "durationMs": 5000,
            "message": {
                "id": "msg_test_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-sonnet-20241022",
                "content": ["Test response content"],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 300,
                },
            },
            "isApiErrorMessage": False,
            "model": "claude-3-sonnet-20241022",
        }

        result = FormatDetector.detect_entry_format(legacy_entry)
        assert result == LogFormat.LEGACY

    def test_detect_unknown_format(self) -> None:
        """Test detection of unknown format entries."""
        # Invalid entry
        invalid_entry = {"type": "unknown_type", "timestamp": "2025-06-03T12:35:51.791Z"}

        result = FormatDetector.detect_entry_format(invalid_entry)
        assert result == LogFormat.UNKNOWN

    def test_detect_non_assistant_entry(self) -> None:
        """Test handling of non-assistant entries."""
        user_entry = {
            "type": "user",
            "message": {"role": "user", "content": "Test message"},
            "timestamp": "2025-06-03T12:35:42.663Z",
        }

        result = FormatDetector.detect_entry_format(user_entry)
        assert result == LogFormat.UNKNOWN

    def test_detect_malformed_entry(self) -> None:
        """Test handling of malformed entries."""
        # Non-dict entry
        result = FormatDetector.detect_entry_format("not a dict")  # type: ignore[arg-type]
        assert result == LogFormat.UNKNOWN

        # None entry
        result = FormatDetector.detect_entry_format(None)  # type: ignore[arg-type]
        assert result == LogFormat.UNKNOWN

    def test_v1_0_9_edge_cases(self) -> None:
        """Test edge cases for v1.0.9 format detection."""
        # Entry with ttftMs but has costUSD (transitional format)
        transitional_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "costUSD": 0.045,  # Has cost (legacy indicator)
            "uuid": "test-uuid",  # Has uuid (v1.0.9 indicator)
            "message": {
                "model": "claude-sonnet-4-20250514",
                "ttftMs": 5742,  # Has ttftMs (v1.0.9 indicator)
            },
        }

        # Should be detected as legacy due to presence of costUSD
        result = FormatDetector.detect_entry_format(transitional_entry)
        assert result == LogFormat.LEGACY

    def test_legacy_edge_cases(self) -> None:
        """Test edge cases for legacy format detection."""
        # Minimal legacy entry
        minimal_legacy = {
            "type": "assistant",
            "timestamp": "2025-05-09T10:30:00.000Z",
            "sessionId": "session-minimal",
            "costUSD": 0.001,
            "message": {"model": "claude-3-sonnet-20241022"},
        }

        result = FormatDetector.detect_entry_format(minimal_legacy)
        assert result == LogFormat.LEGACY

    def test_version_parsing(self) -> None:
        """Test version parsing for format detection."""
        # Test various version formats
        test_cases: List[Tuple[Dict[str, Any], LogFormat]] = [
            # (entry, expected_format)
            (
                {
                    "type": "assistant",
                    "timestamp": "2025-06-03T12:35:51.791Z",
                    "sessionId": "test",
                    "version": "1.0.9",
                    "uuid": "test-uuid",
                    "message": {"model": "test", "ttftMs": 1000},
                },
                LogFormat.V1_0_9,
            ),
            (
                {
                    "type": "assistant",
                    "timestamp": "2025-06-03T12:35:51.791Z",
                    "sessionId": "test",
                    "version": "1.1.0",
                    "uuid": "test-uuid",
                    "message": {"model": "test", "ttftMs": 1000},
                },
                LogFormat.V1_0_9,
            ),
            (
                {
                    "type": "assistant",
                    "timestamp": "2025-06-03T12:35:51.791Z",
                    "sessionId": "test",
                    "version": "0.2.104",
                    "costUSD": 0.001,
                    "message": {"model": "test"},
                },
                LogFormat.LEGACY,
            ),
        ]

        for entry, expected in test_cases:
            result = FormatDetector.detect_entry_format(entry)
            assert result == expected, f"Failed for version {entry.get('version')}"

    def test_detect_file_format(self) -> None:
        """Test file-level format detection."""
        # Create temporary files with different formats
        v1_0_9_data = [
            {"type": "summary", "summary": "Test summary", "leafUuid": "test-uuid"},
            {
                "type": "assistant",
                "timestamp": "2025-06-03T12:35:51.791Z",
                "sessionId": "test-session",
                "version": "1.0.9",
                "uuid": "test-uuid",
                "message": {
                    "model": "claude-sonnet-4-20250514",
                    "ttftMs": 5742,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        ]

        legacy_data = [
            {
                "type": "assistant",
                "timestamp": "2025-05-09T10:30:00.000Z",
                "sessionId": "test-session",
                "costUSD": 0.045,
                "message": {"model": "claude-3-sonnet-20241022", "usage": {"input_tokens": 100, "output_tokens": 50}},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in v1_0_9_data:
                f.write(json.dumps(entry) + "\n")
            v1_0_9_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in legacy_data:
                f.write(json.dumps(entry) + "\n")
            legacy_file = f.name

        try:
            # Test v1.0.9 file
            result = FormatDetector.detect_file_format(v1_0_9_file)
            assert result == LogFormat.V1_0_9

            # Test legacy file
            result = FormatDetector.detect_file_format(legacy_file)
            assert result == LogFormat.LEGACY

            # Test non-existent file
            result = FormatDetector.detect_file_format("/non/existent/file.jsonl")
            assert result == LogFormat.UNKNOWN
        finally:
            # Cleanup
            Path(v1_0_9_file).unlink(missing_ok=True)
            Path(legacy_file).unlink(missing_ok=True)

    def test_detect_files_format(self) -> None:
        """Test multi-file format detection."""
        # Create multiple temporary files
        files_data: List[Tuple[List[Dict[str, Any]], LogFormat]] = [
            (
                [
                    {
                        "type": "assistant",
                        "costUSD": 0.1,
                        "timestamp": "2025-01-01T00:00:00Z",
                        "sessionId": "s1",
                        "message": {"model": "test"},
                    }
                ],
                LogFormat.LEGACY,
            ),
            (
                [
                    {
                        "type": "assistant",
                        "uuid": "u1",
                        "timestamp": "2025-01-01T00:00:00Z",
                        "sessionId": "s1",
                        "version": "1.0.9",
                        "message": {"model": "test", "ttftMs": 1000},
                    }
                ],
                LogFormat.V1_0_9,
            ),
        ]

        temp_files = []
        try:
            for data, _ in files_data:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for entry in data:
                        f.write(json.dumps(entry) + "\n")
                    temp_files.append(f.name)

            # Test mixed files - should return V1_0_9 (priority order)
            result = FormatDetector.detect_files_format(temp_files)
            assert result == LogFormat.V1_0_9

            # Test empty list
            result = FormatDetector.detect_files_format([])
            assert result == LogFormat.UNKNOWN
        finally:
            for file_path in temp_files:
                Path(file_path).unlink(missing_ok=True)

    def test_format_confidence_scores(self) -> None:
        """Test confidence scoring functionality."""
        # v1.0.9 entry
        v1_0_9_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "version": "1.0.9",
            "uuid": "test-uuid",
            "message": {"model": "claude-sonnet-4-20250514", "ttftMs": 5742},
        }

        confidence = FormatDetector.get_format_confidence(v1_0_9_entry)
        assert confidence[LogFormat.V1_0_9] > confidence[LogFormat.LEGACY]
        assert confidence[LogFormat.V1_0_9] > 0.5

        # Legacy entry
        legacy_entry = {
            "type": "assistant",
            "timestamp": "2025-05-09T10:30:00.000Z",
            "sessionId": "test-session",
            "costUSD": 0.045,
            "message": {"model": "claude-3-sonnet-20241022"},
        }

        confidence = FormatDetector.get_format_confidence(legacy_entry)
        assert confidence[LogFormat.LEGACY] > confidence[LogFormat.V1_0_9]
        assert confidence[LogFormat.LEGACY] > 0.5

    def test_confidence_unknown_entry(self) -> None:
        """Test confidence scores for unknown entries."""
        unknown_entry = {"type": "unknown", "data": "test"}

        confidence = FormatDetector.get_format_confidence(unknown_entry)
        assert confidence[LogFormat.UNKNOWN] == 1.0
        assert confidence[LogFormat.LEGACY] == 0.0
        assert confidence[LogFormat.V1_0_9] == 0.0

    def test_format_detection_comprehensive(self) -> None:
        """Comprehensive test with various real-world scenarios."""
        test_cases: List[Tuple[Dict[str, Any], LogFormat]] = [
            # Real v1.0.9 format samples
            (
                {
                    "parentUuid": "aa87c72a-a2b7-4db0-93f3-0892afe18d40",
                    "isSidechain": False,
                    "userType": "external",
                    "cwd": "/Users/test/project",
                    "sessionId": "0da08427-d1b9-4055-8314-83d3f147b157",
                    "version": "1.0.9",
                    "message": {
                        "id": "msg_bdrk_01D6C9sdmn7o8sNa3ZCqrBeV",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "usage": {
                            "input_tokens": 4,
                            "cache_creation_input_tokens": 6094,
                            "cache_read_input_tokens": 13558,
                            "output_tokens": 252,
                        },
                        "ttftMs": 5742,
                    },
                    "type": "assistant",
                    "uuid": "13f77474-e274-4692-ad57-772d224a5e06",
                    "timestamp": "2025-06-03T12:35:51.791Z",
                },
                LogFormat.V1_0_9,
            ),
            # Real legacy format samples
            (
                {
                    "type": "assistant",
                    "sessionId": "9d2a9923-5653-4bec-bc95-4ffd838a6736",
                    "timestamp": "2025-05-09T12:03:20.000Z",
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
                },
                LogFormat.LEGACY,
            ),
            # Edge case: new format without version
            (
                {
                    "type": "assistant",
                    "timestamp": "2025-06-03T12:35:51.791Z",
                    "sessionId": "test-session",
                    "uuid": "test-uuid",
                    "message": {"model": "claude-sonnet-4-20250514", "ttftMs": 5742},
                },
                LogFormat.V1_0_9,
            ),
        ]

        for entry, expected_format in test_cases:
            result = FormatDetector.detect_entry_format(entry)
            assert result == expected_format, f"Failed for entry: {entry.get('uuid', 'no-uuid')}"

    def test_private_methods(self) -> None:
        """Test private helper methods."""
        # Test _is_v1_0_9_format
        v1_0_9_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "version": "1.0.9",
            "uuid": "test-uuid",
            "message": {"model": "test", "ttftMs": 5742},
        }
        assert FormatDetector._is_v1_0_9_format(v1_0_9_entry) is True

        # Test _is_legacy_format
        legacy_entry = {
            "type": "assistant",
            "timestamp": "2025-05-09T10:30:00.000Z",
            "sessionId": "test-session",
            "costUSD": 0.045,
            "message": {"model": "test"},
        }
        assert FormatDetector._is_legacy_format(legacy_entry) is True

    def test_file_format_with_empty_lines_and_invalid_json(self) -> None:
        """Test file format detection with empty lines and invalid JSON."""
        # Test file with empty lines and invalid JSON entries
        mixed_data = [
            "",  # Empty line
            "invalid json line",  # Invalid JSON
            (
                '{"type": "assistant", "costUSD": 0.1, "timestamp": "2025-01-01T00:00:00Z", '
                '"sessionId": "s1", "message": {"model": "test"}}'
            ),
            "",  # Another empty line
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in mixed_data:
                f.write(line + "\n")
            test_file = f.name

        try:
            result = FormatDetector.detect_file_format(test_file)
            assert result == LogFormat.LEGACY
        finally:
            Path(test_file).unlink(missing_ok=True)

    def test_format_detection_ambiguous_entry(self) -> None:
        """Test entries that don't clearly match any format."""
        # Entry that has neither costUSD nor strong v1.0.9 indicators
        ambiguous_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "message": {"model": "test"},
            # No costUSD, no ttftMs, no uuid, no version
        }

        result = FormatDetector.detect_entry_format(ambiguous_entry)
        assert result == LogFormat.UNKNOWN

    def test_file_format_reaching_max_entries(self) -> None:
        """Test file format detection when reaching max_entries_to_check limit."""
        # Create file with more than 10 assistant entries to test limit
        many_entries = []

        # Add 15 legacy assistant entries to test the max_entries_to_check limit
        for i in range(15):
            entry = {
                "type": "assistant",
                "timestamp": f"2025-01-0{i % 9 + 1}T00:00:00Z",
                "sessionId": f"session-{i}",
                "costUSD": 0.1 + i * 0.01,
                "message": {"model": "test"},
            }
            many_entries.append(json.dumps(entry))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry_json in many_entries:
                f.write(entry_json + "\n")
            test_file = f.name

        try:
            result = FormatDetector.detect_file_format(test_file)
            assert result == LogFormat.LEGACY
        finally:
            Path(test_file).unlink(missing_ok=True)

    def test_version_parsing_edge_cases(self) -> None:
        """Test edge cases in version parsing."""
        # Test invalid version format
        invalid_version_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "version": "invalid.version.format",
            "uuid": "test-uuid",
            "message": {"model": "test", "ttftMs": 5742},
        }

        # Should still detect as v1.0.9 due to other indicators
        result = FormatDetector.detect_entry_format(invalid_version_entry)
        assert result == LogFormat.V1_0_9

        # Test version with non-numeric parts
        non_numeric_version_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "version": "1.x.9",
            "uuid": "test-uuid",
            "message": {"model": "test", "ttftMs": 5742},
        }

        result = FormatDetector.detect_entry_format(non_numeric_version_entry)
        assert result == LogFormat.V1_0_9

    def test_confidence_score_edge_cases(self) -> None:
        """Test confidence score calculation edge cases."""
        # Test entry with no version but v1.0.9 indicators
        no_version_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "uuid": "test-uuid",
            "message": {"model": "test", "ttftMs": 5742},
        }

        confidence = FormatDetector.get_format_confidence(no_version_entry)
        assert confidence[LogFormat.V1_0_9] > 0.5

        # Test entry with version but invalid format
        invalid_version_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "version": "not.a.version",
            "uuid": "test-uuid",
            "message": {"model": "test", "ttftMs": 5742},
        }

        confidence = FormatDetector.get_format_confidence(invalid_version_entry)
        assert confidence[LogFormat.V1_0_9] > 0.5

    def test_file_format_detection_edge_cases(self) -> None:
        """Test edge cases in file format detection."""
        # Test file with only unknown entries (no assistant entries)
        unknown_only_data = [
            '{"type": "user", "message": "hello"}',
            '{"type": "system", "message": "system msg"}',
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in unknown_only_data:
                f.write(line + "\n")
            test_file = f.name

        try:
            result = FormatDetector.detect_file_format(test_file)
            assert result == LogFormat.UNKNOWN  # Line 108 coverage
        finally:
            Path(test_file).unlink(missing_ok=True)

    def test_detect_files_format_edge_cases(self) -> None:
        """Test edge cases in multi-file format detection."""
        # Test files where only UNKNOWN format is detected across all files
        unknown_files_data = [
            ['{"type": "user", "message": "hello"}'],
            ['{"type": "system", "message": "system"}'],
        ]

        temp_files = []
        try:
            for data in unknown_files_data:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for line in data:
                        f.write(line + "\n")
                    temp_files.append(f.name)

            result = FormatDetector.detect_files_format(temp_files)
            assert result == LogFormat.UNKNOWN  # Lines 140-143 coverage
        finally:
            for file_path in temp_files:
                Path(file_path).unlink(missing_ok=True)

    def test_confidence_score_zero_total(self) -> None:
        """Test confidence score calculation when total score is zero."""
        # Entry that has conflicting indicators or minimal data
        conflicting_entry = {
            "type": "assistant",
            "costUSD": 0.1,  # Legacy indicator
            "uuid": "test",  # v1.0.9 indicator
            # No other strong indicators for either format
        }

        confidence = FormatDetector.get_format_confidence(conflicting_entry)

        # Should have some score for legacy due to costUSD
        assert confidence[LogFormat.LEGACY] > 0.0

        # Test truly minimal entry that gets zero total score
        # Use a non-assistant type entry to trigger the early return
        non_assistant_entry = {"type": "user", "message": "hello"}

        confidence = FormatDetector.get_format_confidence(non_assistant_entry)

        # Should return all zeros for legacy/v1_0_9 and 1.0 for unknown
        assert confidence[LogFormat.LEGACY] == 0.0
        assert confidence[LogFormat.V1_0_9] == 0.0
        assert confidence[LogFormat.UNKNOWN] == 1.0  # Line 250 coverage (early return)

    def test_real_v1_0_9_sample_data(self) -> None:
        """Test with actual v1.0.9 sample data from new format."""
        # Real v1.0.9 assistant entry from actual sample data
        real_v1_0_9_entry = {
            "parentUuid": "aa87c72a-a2b7-4db0-93f3-0892afe18d40",
            "isSidechain": False,
            "userType": "external",
            "cwd": "/Users/satoshi_numasawa/source/2000_self/claude_code_cost_collector",
            "sessionId": "0da08427-d1b9-4055-8314-83d3f147b157",
            "version": "1.0.9",
            "message": {
                "id": "msg_bdrk_01D6C9sdmn7o8sNa3ZCqrBeV",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "このプロジェクトについて詳しく調べて説明しますね。まず基本的な情報を収集します。"},
                    {
                        "type": "tool_use",
                        "id": "toolu_bdrk_01Mi74W2jzUsgrm3Lj67wdFs",
                        "name": "Read",
                        "input": {"file_path": "/Users/satoshi_numasawa/source/2000_self/claude_code_cost_collector/README.md"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 6094,
                    "cache_read_input_tokens": 13558,
                    "output_tokens": 252,
                },
                "ttftMs": 5742,
            },
            "type": "assistant",
            "uuid": "13f77474-e274-4692-ad57-772d224a5e06",
            "timestamp": "2025-06-03T12:35:51.791Z",
        }

        # Should be detected as v1.0.9 format
        result = FormatDetector.detect_entry_format(real_v1_0_9_entry)
        assert result == LogFormat.V1_0_9

        # Confidence should be high for v1.0.9
        confidence = FormatDetector.get_format_confidence(real_v1_0_9_entry)
        assert confidence[LogFormat.V1_0_9] > 0.8
        assert confidence[LogFormat.V1_0_9] > confidence[LogFormat.LEGACY]

    def test_new_format_token_usage_structure(self) -> None:
        """Test recognition of new format token usage structure."""
        # Entry with new token usage structure (cache_creation_input_tokens, cache_read_input_tokens)
        new_usage_entry = {
            "type": "assistant",
            "timestamp": "2025-06-03T12:35:51.791Z",
            "sessionId": "test-session",
            "version": "1.0.9",
            "uuid": "test-uuid",
            "message": {
                "model": "claude-sonnet-4-20250514",
                "ttftMs": 5742,
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 6094,
                    "cache_read_input_tokens": 13558,
                    "output_tokens": 252,
                },
            },
        }

        result = FormatDetector.detect_entry_format(new_usage_entry)
        assert result == LogFormat.V1_0_9

        # Should have high confidence
        confidence = FormatDetector.get_format_confidence(new_usage_entry)
        assert confidence[LogFormat.V1_0_9] >= 0.9  # Should have very high confidence

    def test_mixed_format_file_detection(self) -> None:
        """Test file detection with mixed entry types including real sample patterns."""
        # Create a file with summary entries and v1.0.9 assistant entries
        mixed_entries = [
            (
                '{"type":"summary","summary":"Cost Tracking for Claude Code Command",'
                '"leafUuid":"dbc7b016-0d3e-4e53-a87d-e2ea175a15cd"}'
            ),
            '{"type":"user","message":{"role":"user","content":"test"},"uuid":"aa87c72a-a2b7-4db0-93f3-0892afe18d40","timestamp":"2025-06-03T12:35:42.663Z"}',
            '{"type":"assistant","version":"1.0.9","uuid":"13f77474-e274-4692-ad57-772d224a5e06","timestamp":"2025-06-03T12:35:51.791Z","sessionId":"0da08427-d1b9-4055-8314-83d3f147b157","message":{"model":"claude-sonnet-4-20250514","ttftMs":5742,"usage":{"input_tokens":4,"output_tokens":252}}}',
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in mixed_entries:
                f.write(entry + "\n")
            test_file = f.name

        try:
            result = FormatDetector.detect_file_format(test_file)
            assert result == LogFormat.V1_0_9
        finally:
            Path(test_file).unlink(missing_ok=True)


class TestFormatDetectorPerformance:
    """Performance tests for FormatDetector."""

    def test_large_file_detection_performance(self) -> None:
        """Test performance with large files."""
        import time

        # Create a large file with many entries
        large_entries = []
        num_entries = 1000

        for i in range(num_entries):
            if i % 2 == 0:
                # Legacy entries
                entry = {
                    "type": "assistant",
                    "timestamp": f"2025-01-{i % 28 + 1:02d}T00:00:00Z",
                    "sessionId": f"session-{i}",
                    "costUSD": 0.001 * (i + 1),
                    "message": {"model": "claude-3-sonnet"},
                }
            else:
                # v1.0.9 entries
                entry = {
                    "type": "assistant",
                    "timestamp": f"2025-01-{i % 28 + 1:02d}T00:00:00Z",
                    "sessionId": f"session-{i}",
                    "version": "1.0.9",
                    "uuid": f"uuid-{i}",
                    "message": {"model": "claude-sonnet-4", "ttftMs": 1000},
                }
            large_entries.append(json.dumps(entry))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in large_entries:
                f.write(entry + "\n")
            test_file = f.name

        try:
            start_time = time.time()
            result = FormatDetector.detect_file_format(test_file)
            end_time = time.time()

            # Should complete within reasonable time (< 1 second for 1000 entries)
            detection_time = end_time - start_time
            assert detection_time < 1.0

            # Should detect v1.0.9 format (priority)
            assert result == LogFormat.V1_0_9

        finally:
            Path(test_file).unlink(missing_ok=True)

    def test_confidence_calculation_performance(self) -> None:
        """Test performance of confidence score calculations."""
        import time

        # Create entries with various complexities
        test_entries = [
            {
                "type": "assistant",
                "timestamp": "2025-01-01T00:00:00Z",
                "sessionId": "session-1",
                "version": "1.0.9",
                "uuid": "uuid-1",
                "message": {
                    "model": "claude-sonnet-4",
                    "ttftMs": 1000,
                    "usage": {"input_tokens": 100, "output_tokens": 200},
                },
            }
            for _ in range(100)
        ]

        start_time = time.time()
        for entry in test_entries:
            confidence = FormatDetector.get_format_confidence(entry)
            assert isinstance(confidence, dict)
            assert len(confidence) == 3  # Three format types
        end_time = time.time()

        # Should process 100 entries quickly (< 0.1 second)
        processing_time = end_time - start_time
        assert processing_time < 0.1

    def test_multi_file_detection_performance(self) -> None:
        """Test performance of multi-file format detection."""
        import time

        # Create multiple files
        num_files = 20
        temp_files = []

        try:
            for i in range(num_files):
                entries = []
                for j in range(50):  # 50 entries per file
                    entry = {
                        "type": "assistant",
                        "timestamp": f"2025-01-{j % 28 + 1:02d}T00:00:00Z",
                        "sessionId": f"session-{i}-{j}",
                        "version": "1.0.9",
                        "uuid": f"uuid-{i}-{j}",
                        "message": {"model": "claude-sonnet-4", "ttftMs": 1000},
                    }
                    entries.append(json.dumps(entry))

                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for entry in entries:
                        f.write(entry + "\n")
                    temp_files.append(f.name)

            start_time = time.time()
            result = FormatDetector.detect_files_format(temp_files)
            end_time = time.time()

            # Should complete within reasonable time (< 2 seconds for 20 files)
            detection_time = end_time - start_time
            assert detection_time < 2.0

            # Should detect v1.0.9 format
            assert result == LogFormat.V1_0_9

        finally:
            for file_path in temp_files:
                Path(file_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
