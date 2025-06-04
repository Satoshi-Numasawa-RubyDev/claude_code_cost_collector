"""Constants for Claude Code Cost Collector."""

# Granularity options for aggregation
GRANULARITIES = ["daily", "monthly", "project", "session", "all"]

# Output format options
OUTPUT_FORMATS = ["text", "json", "yaml", "csv"]

# Sort field options
SORT_FIELDS = ["input", "output", "total", "cost", "date"]

# Confidence levels for cost estimation
CONFIDENCE_LEVELS = ["high", "medium", "low"]

# Default values
DEFAULT_GRANULARITY = "daily"
DEFAULT_OUTPUT_FORMAT = "text"
DEFAULT_SORT_BY = "date"
DEFAULT_CONFIDENCE_THRESHOLD = "medium"

# Currency and formatting
DEFAULT_CURRENCY = "USD"
COST_PRECISION = 6  # Number of decimal places for cost display

# File extensions
SUPPORTED_LOG_EXTENSIONS = [".json", ".jsonl"]
SUPPORTED_CONFIG_EXTENSIONS = [".yaml", ".yml", ".json"]

# API and model constants
DEFAULT_MODEL_FALLBACK = "claude-3-sonnet"
TOKENS_PER_MILLION = 1_000_000

# Time and date formatting
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DATE_FORMAT = "%Y-%m-%d"
MONTH_FORMAT = "%Y-%m"

# Performance limits
MAX_FILES_PER_BATCH = 1000
MAX_ENTRIES_PER_FILE = 10000
