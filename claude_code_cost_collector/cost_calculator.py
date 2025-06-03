"""
Cost calculation functionality for Claude Code Cost Collector.

This module provides functionality to estimate costs from token usage
when cost information is not directly available in log files (e.g., Claude CLI v1.0.9 format).
"""

from dataclasses import dataclass
from typing import Tuple

from .model_pricing import ModelPricingManager, PricingError


class CostCalculationError(Exception):
    """Exception raised for cost calculation errors."""

    pass


@dataclass
class TokenUsage:
    """
    Token usage information for a Claude API call.

    Represents the different types of tokens consumed during an API call,
    including regular input/output tokens and cache-related tokens.

    Attributes:
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        cache_creation_tokens: Number of tokens used for cache creation (charged)
        cache_read_tokens: Number of tokens read from cache (typically free)
    """

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def __post_init__(self) -> None:
        """Validate token usage data after initialization."""
        self._validate_tokens()

    def _validate_tokens(self) -> None:
        """
        Validate that token counts are reasonable.

        Raises:
            CostCalculationError: If token values are invalid.
        """
        # Check for negative values
        fields = [
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("cache_creation_tokens", self.cache_creation_tokens),
            ("cache_read_tokens", self.cache_read_tokens),
        ]

        for field_name, value in fields:
            if value < 0:
                raise CostCalculationError(f"{field_name} cannot be negative: {value}")

        # Basic sanity check: at least some tokens should be used
        total_tokens = self.input_tokens + self.output_tokens + self.cache_creation_tokens
        if total_tokens == 0:
            raise CostCalculationError("At least one of input_tokens, output_tokens, or cache_creation_tokens must be positive")

    @property
    def total_tokens(self) -> int:
        """
        Calculate total tokens including cache creation tokens.

        Note: cache_read_tokens are typically not included in cost calculations
        as they are usually free.

        Returns:
            Total number of tokens that contribute to cost.
        """
        return self.input_tokens + self.output_tokens + self.cache_creation_tokens

    @property
    def total_tokens_including_cache_read(self) -> int:
        """
        Calculate total tokens including all types of tokens.

        Returns:
            Total number of tokens including cache read tokens.
        """
        return self.input_tokens + self.output_tokens + self.cache_creation_tokens + self.cache_read_tokens

    def to_dict(self) -> dict[str, int]:
        """
        Convert token usage to a dictionary.

        Returns:
            Dictionary representation of token usage.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }


class CostCalculator:
    """
    Calculator for estimating costs from token usage.

    Uses model pricing information to calculate costs when direct cost information
    is not available in log files. Provides confidence levels for cost estimates.
    """

    def __init__(self, pricing_manager: ModelPricingManager) -> None:
        """
        Initialize the cost calculator.

        Args:
            pricing_manager: Manager for model pricing information.
        """
        self.pricing_manager = pricing_manager

    def calculate_cost(self, model: str, usage: TokenUsage) -> float:
        """
        Calculate cost from token usage for a specific model.

        Args:
            model: Name of the Claude model used.
            usage: Token usage information.

        Returns:
            Estimated cost in USD.

        Raises:
            CostCalculationError: If cost calculation fails.
        """
        try:
            # Get pricing information (with fallback if model not found)
            pricing = self.pricing_manager.get_pricing_or_fallback(model)

            # Calculate individual components
            input_cost = (usage.input_tokens / 1_000_000) * pricing.input_price_per_million
            output_cost = (usage.output_tokens / 1_000_000) * pricing.output_price_per_million
            cache_creation_cost = (usage.cache_creation_tokens / 1_000_000) * pricing.cache_creation_price_per_million
            cache_read_cost = (usage.cache_read_tokens / 1_000_000) * pricing.cache_read_price_per_million

            # Total cost
            total_cost = input_cost + output_cost + cache_creation_cost + cache_read_cost

            return total_cost

        except (PricingError, ZeroDivisionError) as e:
            raise CostCalculationError(f"Failed to calculate cost for model '{model}': {e}")

    def estimate_cost_with_confidence(self, model: str, usage: TokenUsage) -> Tuple[float, str]:
        """
        Calculate cost and provide confidence level for the estimate.

        Args:
            model: Name of the Claude model used.
            usage: Token usage information.

        Returns:
            Tuple of (estimated_cost_usd, confidence_level).
            Confidence levels: "high", "medium", "low".

        Raises:
            CostCalculationError: If cost calculation fails.
        """
        # Calculate the cost
        cost = self.calculate_cost(model, usage)

        # Determine confidence level based on pricing availability
        confidence = self.pricing_manager.get_confidence_level(model)

        return cost, confidence

    def calculate_cost_breakdown(self, model: str, usage: TokenUsage) -> dict[str, float]:
        """
        Calculate detailed cost breakdown by token type.

        Args:
            model: Name of the Claude model used.
            usage: Token usage information.

        Returns:
            Dictionary with cost breakdown by component.

        Raises:
            CostCalculationError: If cost calculation fails.
        """
        try:
            pricing = self.pricing_manager.get_pricing_or_fallback(model)

            breakdown = {
                "input_cost": (usage.input_tokens / 1_000_000) * pricing.input_price_per_million,
                "output_cost": (usage.output_tokens / 1_000_000) * pricing.output_price_per_million,
                "cache_creation_cost": (usage.cache_creation_tokens / 1_000_000) * pricing.cache_creation_price_per_million,
                "cache_read_cost": (usage.cache_read_tokens / 1_000_000) * pricing.cache_read_price_per_million,
            }

            breakdown["total_cost"] = sum(breakdown.values())
            return breakdown

        except (PricingError, ZeroDivisionError) as e:
            raise CostCalculationError(f"Failed to calculate cost breakdown for model '{model}': {e}")

    def is_cost_reliable(self, model: str, confidence_threshold: str = "medium") -> bool:
        """
        Check if cost calculation for a model meets reliability threshold.

        Args:
            model: Name of the Claude model.
            confidence_threshold: Minimum confidence level required ("high", "medium", "low").

        Returns:
            True if the model's pricing confidence meets the threshold.
        """
        confidence_levels = {"low": 1, "medium": 2, "high": 3}

        model_confidence = self.pricing_manager.get_confidence_level(model)
        model_level = confidence_levels.get(model_confidence, 0)
        threshold_level = confidence_levels.get(confidence_threshold, 2)

        return model_level >= threshold_level

    def get_supported_models(self) -> list[str]:
        """
        Get list of models with explicit pricing support.

        Returns:
            List of model names with high confidence pricing.
        """
        return self.pricing_manager.get_all_supported_models()


def create_default_cost_calculator() -> CostCalculator:
    """
    Create a CostCalculator with default pricing manager.

    Returns:
        CostCalculator instance ready for use.
    """
    pricing_manager = ModelPricingManager()
    return CostCalculator(pricing_manager)
