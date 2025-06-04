"""
Model pricing management for Claude Code Cost Collector.

This module provides functionality to manage pricing information for different
Claude API models, including the ability to calculate costs from token usage
when cost information is not directly available in the log files (e.g., in v1.0.9 format).
"""

from dataclasses import dataclass
from typing import Dict, Optional


class PricingError(Exception):
    """Exception raised for pricing-related errors."""

    pass


@dataclass
class ModelPricing:
    """
    Pricing information for a specific Claude model.

    Represents the cost structure for a model, including input tokens, output tokens,
    and cache-related pricing. All prices are specified per million tokens in USD.

    Attributes:
        input_price_per_million: Cost per million input tokens in USD
        output_price_per_million: Cost per million output tokens in USD
        cache_creation_price_per_million: Cost per million cache creation tokens in USD
        cache_read_price_per_million: Cost per million cache read tokens in USD (typically 0)
    """

    input_price_per_million: float
    output_price_per_million: float
    cache_creation_price_per_million: float
    cache_read_price_per_million: float = 0.0

    def __post_init__(self) -> None:
        """Validate pricing data after initialization."""
        self._validate_pricing()

    def _validate_pricing(self) -> None:
        """
        Validate that pricing values are reasonable.

        Raises:
            PricingError: If pricing values are invalid.
        """
        # Check for negative values
        fields = [
            ("input_price_per_million", self.input_price_per_million),
            ("output_price_per_million", self.output_price_per_million),
            ("cache_creation_price_per_million", self.cache_creation_price_per_million),
            ("cache_read_price_per_million", self.cache_read_price_per_million),
        ]

        for field_name, value in fields:
            if value < 0:
                raise PricingError(f"{field_name} cannot be negative: {value}")

        # Sanity check: output tokens are typically more expensive than input tokens
        if self.output_price_per_million < self.input_price_per_million:
            # This is a warning-level issue, not an error
            pass

        # Sanity check: cache creation should be similar to input pricing
        if self.cache_creation_price_per_million < 0:
            raise PricingError(f"cache_creation_price_per_million cannot be negative: {self.cache_creation_price_per_million}")

    def to_dict(self) -> Dict[str, float]:
        """
        Convert pricing information to a dictionary.

        Returns:
            Dictionary representation of the pricing data.
        """
        return {
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "cache_creation_price_per_million": self.cache_creation_price_per_million,
            "cache_read_price_per_million": self.cache_read_price_per_million,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ModelPricing":
        """
        Create a ModelPricing instance from a dictionary.

        Args:
            data: Dictionary containing pricing data.

        Returns:
            ModelPricing instance.

        Raises:
            PricingError: If required fields are missing or invalid.
        """
        required_fields = ["input_price_per_million", "output_price_per_million", "cache_creation_price_per_million"]

        for field in required_fields:
            if field not in data:
                raise PricingError(f"Missing required field: {field}")

        return cls(
            input_price_per_million=data["input_price_per_million"],
            output_price_per_million=data["output_price_per_million"],
            cache_creation_price_per_million=data["cache_creation_price_per_million"],
            cache_read_price_per_million=data.get("cache_read_price_per_million", 0.0),
        )


class ModelPricingManager:
    """
    Manager class for model pricing information.

    Handles storage, retrieval, and validation of pricing data for different
    Claude models. Provides default pricing for known models and supports
    custom pricing configuration.
    """

    def __init__(self) -> None:
        """Initialize the pricing manager with default pricing data."""
        self._pricing_data: Dict[str, ModelPricing] = {}
        self._initialize_default_pricing()

    def _initialize_default_pricing(self) -> None:
        """Initialize pricing data for known Claude models."""
        # Default pricing based on Anthropic's pricing information
        # These are approximate values and should be updated with current pricing
        default_models = {
            "claude-sonnet-4": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,  # 1.25x input price
                cache_read_price_per_million=0.3,  # 0.1x input price
            ),
            "claude-sonnet-4-20250514": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=0.3,
            ),
            "claude-opus-4": ModelPricing(
                input_price_per_million=15.0,
                output_price_per_million=75.0,
                cache_creation_price_per_million=18.75,
                cache_read_price_per_million=1.5,
            ),
            "claude-opus-4-20250514": ModelPricing(
                input_price_per_million=15.0,
                output_price_per_million=75.0,
                cache_creation_price_per_million=18.75,
                cache_read_price_per_million=1.5,
            ),
            "claude-3.7-sonnet": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=0.3,
            ),
            "claude-3-7-sonnet-20250219": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=0.3,
            ),
            "claude-3.5-sonnet": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=0.3,
            ),
            "claude-3-5-sonnet-20241022": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=0.3,
            ),
            "claude-3-sonnet": ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=0.3,
            ),
            "claude-3-haiku": ModelPricing(
                input_price_per_million=0.25,
                output_price_per_million=1.25,
                cache_creation_price_per_million=0.3,
                cache_read_price_per_million=0.03,
            ),
            "claude-3-haiku-20240307": ModelPricing(
                input_price_per_million=0.25,
                output_price_per_million=1.25,
                cache_creation_price_per_million=0.3,
                cache_read_price_per_million=0.03,
            ),
        }

        for model_name, pricing in default_models.items():
            self._pricing_data[model_name] = pricing

    def get_pricing(self, model_name: str) -> Optional[ModelPricing]:
        """
        Get pricing information for a specific model.

        Args:
            model_name: Name of the model to get pricing for.

        Returns:
            ModelPricing instance if the model is known, None otherwise.
        """
        return self._pricing_data.get(model_name)

    def get_pricing_or_fallback(self, model_name: str) -> ModelPricing:
        """
        Get pricing information for a model, with fallback to a default pricing.

        If the specific model is not found, returns a conservative fallback pricing
        based on claude-3-sonnet rates.

        Args:
            model_name: Name of the model to get pricing for.

        Returns:
            ModelPricing instance (either specific or fallback).
        """
        pricing = self.get_pricing(model_name)
        if pricing is not None:
            return pricing

        # Fallback to claude-3-sonnet pricing as a conservative estimate
        fallback_pricing = self._pricing_data.get("claude-3-sonnet")
        if fallback_pricing is not None:
            return fallback_pricing

        # Ultimate fallback if even claude-3-sonnet is not available
        return ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
            cache_read_price_per_million=0.3,
        )

    def update_pricing(self, model_name: str, pricing: ModelPricing) -> None:
        """
        Update or add pricing information for a model.

        Args:
            model_name: Name of the model to update pricing for.
            pricing: New pricing information.

        Raises:
            PricingError: If the model name is invalid or pricing is invalid.
        """
        if not model_name or not model_name.strip():
            raise PricingError("Model name cannot be empty")

        # Validate the pricing (will raise PricingError if invalid)
        pricing._validate_pricing()

        self._pricing_data[model_name.strip()] = pricing

    def is_supported_model(self, model_name: str) -> bool:
        """
        Check if a model has explicit pricing information.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if the model has explicit pricing, False otherwise.
        """
        return model_name in self._pricing_data

    def get_all_supported_models(self) -> list[str]:
        """
        Get a list of all models with explicit pricing information.

        Returns:
            List of supported model names.
        """
        return list(self._pricing_data.keys())

    def load_pricing_from_dict(self, pricing_dict: Dict[str, Dict[str, float]]) -> None:
        """
        Load pricing information from a dictionary.

        This method can be used to load custom pricing from configuration files.

        Args:
            pricing_dict: Dictionary mapping model names to pricing data dictionaries.

        Raises:
            PricingError: If the pricing data is invalid.
        """
        for model_name, pricing_data in pricing_dict.items():
            try:
                pricing = ModelPricing.from_dict(pricing_data)
                self.update_pricing(model_name, pricing)
            except (KeyError, TypeError, ValueError) as e:
                raise PricingError(f"Invalid pricing data for model '{model_name}': {e}")

    def get_confidence_level(self, model_name: str) -> str:
        """
        Get the confidence level for pricing of a specific model.

        Args:
            model_name: Name of the model to check.

        Returns:
            Confidence level: "high" for explicitly supported models,
            "medium" for fallback pricing, "low" for ultimate fallback.
        """
        if self.is_supported_model(model_name):
            return "high"
        elif self.get_pricing("claude-3-sonnet") is not None:
            return "medium"
        else:
            return "low"


def create_sample_pricing_manager() -> ModelPricingManager:
    """
    Create a sample ModelPricingManager for testing and documentation purposes.

    Returns:
        ModelPricingManager instance with default pricing loaded.
    """
    return ModelPricingManager()
