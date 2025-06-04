"""Log file collection functionality.

Module for exploring and collecting Claude API log files.
Recursively searches for JSON log files from specified directory.
"""

import logging
import os
from pathlib import Path
from typing import List, Union

from .constants import SUPPORTED_LOG_EXTENSIONS

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
        # Recursively search for supported log files
        log_files: List[Path] = []
        for ext in SUPPORTED_LOG_EXTENSIONS:
            pattern = f"*{ext}"
            log_files.extend(dir_path.rglob(pattern))

        logger.info(f"Found {len(log_files)} log files in {dir_path}")

        # Sort by file modification time (newest first)
        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return log_files

    except PermissionError as e:
        logger.error(f"Permission denied while accessing {dir_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while collecting log files: {e}")
        raise
