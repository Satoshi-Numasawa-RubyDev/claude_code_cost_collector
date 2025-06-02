"""Log file collection functionality.

Module for exploring and collecting Claude API log files.
Recursively searches for JSON log files from specified directory.
"""

import logging
import os
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)


def collect_log_files(directory: Union[str, Path]) -> List[Path]:
    """Recursively collect JSON log files from specified directory.

    Args:
        directory: Target directory path to search

    Returns:
        List of Path objects for found JSON files

    Raises:
        FileNotFoundError: When specified directory does not exist
        PermissionError: When directory access permission is denied
        ValueError: When invalid path is specified
    """
    # Convert to Path object
    dir_path = Path(directory)

    # Basic validation
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {dir_path}")

    # Check directory access permission
    if not os.access(dir_path, os.R_OK):
        raise PermissionError(f"No read permission for directory: {dir_path}")

    try:
        # Recursively search for JSON and JSONL files
        json_files = list(dir_path.rglob("*.json"))
        jsonl_files = list(dir_path.rglob("*.jsonl"))
        json_files.extend(jsonl_files)

        logger.info(f"Found {len(json_files)} JSON/JSONL files in {dir_path}")

        # Sort by file modification time (newest first)
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return json_files

    except PermissionError as e:
        logger.error(f"Permission denied while accessing {dir_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while collecting log files: {e}")
        raise
