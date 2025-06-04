"""Exception classes for Claude Code Cost Collector."""


class ClaudeCodeCostCollectorError(Exception):
    """Base exception for Claude Code Cost Collector."""

    pass


class AggregationError(ClaudeCodeCostCollectorError):
    """Exception raised when aggregation fails."""

    pass


class ConfigError(ClaudeCodeCostCollectorError):
    """Exception raised for configuration-related errors."""

    pass


class CostCalculationError(ClaudeCodeCostCollectorError):
    """Exception raised when cost calculation fails."""

    pass


class ExchangeRateError(ClaudeCodeCostCollectorError):
    """Exception raised when exchange rate operations fail."""

    pass


class FormatterError(ClaudeCodeCostCollectorError):
    """Exception raised when formatting fails."""

    pass


class LogParseError(ClaudeCodeCostCollectorError):
    """Exception raised when log parsing fails."""

    pass


class PricingError(ClaudeCodeCostCollectorError):
    """Exception raised when pricing operations fail."""

    pass
