# Claude Code Cost Collector (CCCC)

**This repository's code was created with [Claude Code](https://claude.ai/code)**

A command-line tool for aggregating and displaying costs associated with Claude API usage.

## Overview

Claude Code Cost Collector (CCCC) is a tool designed to provide transparency into the costs of using Anthropic's Claude API, enabling users to easily track and manage their expenses.

The tool analyzes Claude CLI usage log files and aggregates costs across various dimensions (daily, monthly, per-project, per-session), displaying the results in user-friendly formats.

<img width="1622" alt="スクリーンショット 2025-06-01 10 15 54" src="https://github.com/user-attachments/assets/6f38e5e8-bf32-47b1-9385-e400225f2882" />



## Key Features

### Cost Aggregation Units
- **Daily (Default)**: Aggregates costs by day
- **Monthly**: Aggregates costs by month  
- **Per-Project**: Aggregates costs by project
- **Per-Session**: Aggregates costs by session
- **Individual Log Entry Display**: Shows individual log entries without aggregation

### Output Formats
- **Formatted Text Table (Default)**: Clean tabular display
- **JSON**: Format suitable for programmatic processing
- **YAML**: Human-readable structured data format
- **CSV**: Format compatible with spreadsheet software

### Currency Display
- **US Dollar (USD)**: Default display currency

### Filtering & Sorting Features
- **Date Range Filtering**: Filter data by specifying start and end dates
- **Default Period Limitation**: Shows only the last 30 days by default for performance
- **Sort Functionality**: Sort by input tokens, output tokens, total tokens, cost, or date in ascending/descending order
- **Smart Default Sorting**: When only sort direction is specified, automatically sorts by date (most intuitive for time-series data)

### Timezone Support
- **Automatic Timezone Detection**: Calculates dates based on system timezone (default)
- **Custom Timezone**: Specify any timezone (e.g., `Asia/Tokyo`, `UTC`)
- **Accurate Date Aggregation**: Precise daily cost aggregation considering timezones

## Installation

### From PyPI (Recommended)

```bash
pip install claude-code-cost-collector
```

### System Requirements

- Python 3.13 or higher

## Usage

### Basic Usage

```bash
# After installation via pip
cccc

# Or run directly with Python
python -m claude_code_cost_collector
```

**Note**: By default, only the last 30 days of data are displayed. Use the `--all-data` option to show all data.

### Command Line Options

#### Basic Options

```bash
# Show help
cccc --help

# Specify custom directory
cccc --directory /path/to/claude/logs

# Change aggregation unit
cccc --granularity monthly    # Monthly
cccc --granularity project    # Per-project
cccc --granularity session    # Per-session
cccc --granularity all        # Individual display (no aggregation)

# Change output format
cccc --output json           # JSON format
cccc --output yaml           # YAML format
cccc --output csv            # CSV format

# Sort functionality
cccc --sort asc                       # Sort by date (ascending) - smart default
cccc --sort desc                      # Sort by date (descending) - smart default
cccc --sort asc --sort-field input    # Sort by input tokens (ascending)
cccc --sort desc --sort-field cost    # Sort by cost (descending)
cccc --sort asc --sort-field total    # Sort by total tokens (ascending)
cccc --sort desc --sort-field output  # Sort by output tokens (descending)
cccc --sort asc --sort-field date     # Sort by date (ascending) - explicit
```

#### Date Range and Data Control

```bash
# Specify a specific period
cccc --start-date 2024-01-01 --end-date 2024-01-31

# Specify start date only
cccc --start-date 2024-01-01

# Specify end date only
cccc --end-date 2024-01-31

# Show all data (disable default 30-day limit)
cccc --all-data

# Limit results for large datasets
cccc --all-data --limit 100
```

#### Timezone Settings

```bash
# Use system timezone (default)
cccc --timezone auto

# Aggregate in UTC time
cccc --timezone UTC

# Aggregate in Japan time
cccc --timezone Asia/Tokyo

# Other timezone examples
cccc --timezone America/New_York
cccc --timezone Europe/London
```

#### Sort Functionality

```bash
# Smart default sorting (sorts by date when only direction specified)
cccc --sort asc                        # Sort by date (ascending) - oldest first
cccc --sort desc                       # Sort by date (descending) - newest first

# Sort by specific fields
cccc --sort asc --sort-field input     # Sort by input tokens (ascending)
cccc --sort desc --sort-field cost     # Sort by cost (descending)

# Combined with other options
cccc --granularity monthly --sort desc --sort-field total  # Monthly total tokens (descending)
cccc --granularity project --sort asc --sort-field output  # Project output tokens (ascending)

# Explicit date sorting (same as smart default)
cccc --sort desc --sort-field date     # Sort by date (descending) - show latest data first

# Available sort fields:
# - input:  Input token count
# - output: Output token count
# - total:  Total token count
# - cost:   Cost (uses currency converted if available, otherwise USD)
# - date:   Date/time (for time-based aggregations) - used as default when field not specified
```

### Configuration File

You can persist settings using a configuration file:

```yaml
# config.yaml
timezone: "Asia/Tokyo"              # Default timezone
default_date_range_days: 30         # Default period (0 = all data)
default_granularity: "daily"        # Default aggregation unit
default_output_format: "text"       # Default output format
```

```bash
# Use configuration file
cccc --config /path/to/config.yaml
```

## Data Source

The tool analyzes JSON/JSONL format log files from the following locations:

- **Default**: `~/.claude/projects/`
- **Custom**: Specified with `--directory` option

### Log File Structure

The following information is extracted from JSON/JSONL log files generated by Claude CLI:
- Timestamp (timezone-aware)
- Session ID
- Project name (determined from directory structure)
- Model used
- Token counts (input, output, cache)
- Cost (USD)

### Performance Optimization

To efficiently process large amounts of data, the following features are provided:
- **Default 30-day limit**: Automatically processes only the last 30 days of data
- **Custom period specification**: Specify any period with `--start-date`/`--end-date`
- **Full data access**: Use `--all-data` to disable the limit
- **Result limiting**: Limit display count with `--limit`

## Output Examples

### Text Format (Default)

```
# Default output (sorted by date, descending - newest first)
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Date       ┃ Input      ┃ Output     ┃ Total       ┃ Cost (USD) ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 2024-01-02 │ 2,345      │ 890        │ 3,235       │ $2.45      │
│ 2024-01-01 │ 1,234      │ 567        │ 1,801       │ $1.23      │
└────────────┴────────────┴────────────┴─────────────┴────────────┘
Total: 3,579 input, 1,457 output, 5,036 total tokens, $3.68

# With --sort asc (sorted by date, ascending - oldest first)
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Date       ┃ Input      ┃ Output     ┃ Total       ┃ Cost (USD) ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 2024-01-01 │ 1,234      │ 567        │ 1,801       │ $1.23      │
│ 2024-01-02 │ 2,345      │ 890        │ 3,235       │ $2.45      │
└────────────┴────────────┴────────────┴─────────────┴────────────┘
Total: 3,579 input, 1,457 output, 5,036 total tokens, $3.68
```

### JSON Format

```json
[
  {
    "key": "2024-01-01",
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801,
    "cost_usd": 1.23
  }
]
```

## Troubleshooting

### Common Issues

#### Log files not found
```
No log files found.
```
**Solution**:
- Verify that you have used Claude CLI before
- Check log file location: `~/.claude/projects/`
- Specify the correct path with `--directory` option


#### Date format error
```
error: argument --start-date: Invalid date format
```
**Solution**:
- Ensure date format is `YYYY-MM-DD`
- Example: `2024-01-01`

#### No data displayed (default 30-day limit)
```
No log entries found in the specified date range.
```
**Solution**:
- By default, only the last 30 days are displayed
- Use `--all-data` option to show all data
- Explicitly specify `--start-date` to check older data

#### Timezone issues
When cost aggregation results differ from other tools:
**Solution**:
- Explicitly specify Japan time with `--timezone Asia/Tokyo`
- Use UTC time with `--timezone UTC`
- Set default timezone in configuration file

## Advanced Usage Examples

### Combined Multi-Condition Usage Examples

```bash
# Monthly aggregation for specific period with JSON output
cccc --granularity monthly \
    --output json \
    --start-date 2024-01-01 \
    --end-date 2024-06-30

# Per-project aggregation in CSV format (cost descending)
cccc --granularity project --output csv --sort desc --sort-field cost > project_costs.csv

# Individual display of all logs in specific directory
cccc --directory /custom/logs --granularity all --all-data

# Last 7 days data in Japan time (chronological order, newest first)
cccc --timezone Asia/Tokyo --start-date $(date -v-7d '+%Y-%m-%d') --sort desc

# Last 7 days data in Japan time (input tokens descending)
cccc --timezone Asia/Tokyo --start-date $(date -v-7d '+%Y-%m-%d') --sort desc --sort-field input

# Performance-focused: show only top 50 results (total tokens descending)
cccc --all-data --limit 50 --granularity session --sort desc --sort-field total

# TOP 10 most expensive projects
cccc --granularity project --sort desc --sort-field cost --limit 10

# Monthly aggregation showing highest token usage months first
cccc --granularity monthly --sort desc --sort-field total --all-data
```

### Automation Usage

```bash
#!/bin/bash
# Monthly report generation script example

# Get last month's data
LAST_MONTH=$(date -v-1m '+%Y-%m')
START_DATE="${LAST_MONTH}-01"
END_DATE=$(date -v-1m -v+1m -v-1d '+%Y-%m-%d')

# Generate monthly report
cccc --granularity monthly \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --output json > "claude_code_costs_${LAST_MONTH}.json"
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Copyright 2025 Claude Code Cost Collector Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Development & Contributing

For contributing to development, please see [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd cccc

# Using uv (recommended)
uv sync

# Or using traditional pip
pip install -e .
```

### Dependencies
- `requests`: For HTTP requests (exchange rate retrieval)
- `PyYAML`: For YAML format output
- `rich`: For formatted text display

### Development Commands

```bash
# Run tests
uv run python -m pytest

# Code formatting
uv run python -m black .

# Linting
uv run python -m flake8

# Type checking
uv run python -m mypy claude_code_cost_collector
```
