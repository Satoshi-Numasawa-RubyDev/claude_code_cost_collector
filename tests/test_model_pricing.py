"""
Tests for the model_pricing module.
"""

import pytest

from claude_code_cost_collector.model_pricing import (
    ModelPricing,
    ModelPricingManager,
    PricingError,
    create_sample_pricing_manager,
)


class TestModelPricing:
    """Test the ModelPricing dataclass."""

    def test_create_minimal_pricing(self):
        """Test creating ModelPricing with minimal required fields."""
        pricing = ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
        )

        assert pricing.input_price_per_million == 3.0
        assert pricing.output_price_per_million == 15.0
        assert pricing.cache_creation_price_per_million == 3.75
        assert pricing.cache_read_price_per_million == 0.0  # Default value

    def test_create_full_pricing(self):
        """Test creating ModelPricing with all fields."""
        pricing = ModelPricing(
            input_price_per_million=15.0,
            output_price_per_million=75.0,
            cache_creation_price_per_million=18.75,
            cache_read_price_per_million=1.5,
        )

        assert pricing.input_price_per_million == 15.0
        assert pricing.output_price_per_million == 75.0
        assert pricing.cache_creation_price_per_million == 18.75
        assert pricing.cache_read_price_per_million == 1.5

    def test_validation_negative_values(self):
        """Test that negative values raise PricingError."""
        with pytest.raises(PricingError, match="input_price_per_million cannot be negative"):
            ModelPricing(
                input_price_per_million=-1.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
            )

        with pytest.raises(PricingError, match="output_price_per_million cannot be negative"):
            ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=-1.0,
                cache_creation_price_per_million=3.75,
            )

        with pytest.raises(PricingError, match="cache_creation_price_per_million cannot be negative"):
            ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=-1.0,
            )

        with pytest.raises(PricingError, match="cache_read_price_per_million cannot be negative"):
            ModelPricing(
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                cache_creation_price_per_million=3.75,
                cache_read_price_per_million=-1.0,
            )

    def test_to_dict(self):
        """Test converting ModelPricing to dictionary."""
        pricing = ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
            cache_read_price_per_million=0.3,
        )

        expected = {
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_creation_price_per_million": 3.75,
            "cache_read_price_per_million": 0.3,
        }

        assert pricing.to_dict() == expected

    def test_from_dict_minimal(self):
        """Test creating ModelPricing from dictionary with minimal fields."""
        data = {
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_creation_price_per_million": 3.75,
        }

        pricing = ModelPricing.from_dict(data)

        assert pricing.input_price_per_million == 3.0
        assert pricing.output_price_per_million == 15.0
        assert pricing.cache_creation_price_per_million == 3.75
        assert pricing.cache_read_price_per_million == 0.0

    def test_from_dict_full(self):
        """Test creating ModelPricing from dictionary with all fields."""
        data = {
            "input_price_per_million": 15.0,
            "output_price_per_million": 75.0,
            "cache_creation_price_per_million": 18.75,
            "cache_read_price_per_million": 1.5,
        }

        pricing = ModelPricing.from_dict(data)

        assert pricing.input_price_per_million == 15.0
        assert pricing.output_price_per_million == 75.0
        assert pricing.cache_creation_price_per_million == 18.75
        assert pricing.cache_read_price_per_million == 1.5

    def test_from_dict_missing_fields(self):
        """Test that missing required fields raise PricingError."""
        with pytest.raises(PricingError, match="Missing required field: input_price_per_million"):
            ModelPricing.from_dict(
                {
                    "output_price_per_million": 15.0,
                    "cache_creation_price_per_million": 3.75,
                }
            )

        with pytest.raises(PricingError, match="Missing required field: output_price_per_million"):
            ModelPricing.from_dict(
                {
                    "input_price_per_million": 3.0,
                    "cache_creation_price_per_million": 3.75,
                }
            )

        with pytest.raises(PricingError, match="Missing required field: cache_creation_price_per_million"):
            ModelPricing.from_dict(
                {
                    "input_price_per_million": 3.0,
                    "output_price_per_million": 15.0,
                }
            )


class TestModelPricingManager:
    """Test the ModelPricingManager class."""

    def test_initialization(self):
        """Test that manager initializes with default models."""
        manager = ModelPricingManager()

        # Check that some known models are loaded
        assert manager.is_supported_model("claude-sonnet-4")
        assert manager.is_supported_model("claude-3-sonnet")
        assert manager.is_supported_model("claude-3-haiku")

        # Check that we have a reasonable number of default models
        supported_models = manager.get_all_supported_models()
        assert len(supported_models) >= 6  # At least the 6 models mentioned in requirements

    def test_get_pricing_existing_model(self):
        """Test getting pricing for an existing model."""
        manager = ModelPricingManager()

        pricing = manager.get_pricing("claude-3-sonnet")
        assert pricing is not None
        assert pricing.input_price_per_million == 3.0
        assert pricing.output_price_per_million == 15.0

    def test_get_pricing_nonexistent_model(self):
        """Test getting pricing for a model that doesn't exist."""
        manager = ModelPricingManager()

        pricing = manager.get_pricing("non-existent-model")
        assert pricing is None

    def test_get_pricing_or_fallback_existing_model(self):
        """Test fallback method with existing model."""
        manager = ModelPricingManager()

        pricing = manager.get_pricing_or_fallback("claude-3-sonnet")
        assert pricing.input_price_per_million == 3.0
        assert pricing.output_price_per_million == 15.0

    def test_get_pricing_or_fallback_nonexistent_model(self):
        """Test fallback method with non-existent model."""
        manager = ModelPricingManager()

        pricing = manager.get_pricing_or_fallback("non-existent-model")
        # Should return fallback pricing (claude-3-sonnet rates)
        assert pricing.input_price_per_million == 3.0
        assert pricing.output_price_per_million == 15.0

    def test_update_pricing(self):
        """Test updating pricing for a model."""
        manager = ModelPricingManager()

        new_pricing = ModelPricing(
            input_price_per_million=5.0,
            output_price_per_million=25.0,
            cache_creation_price_per_million=6.25,
            cache_read_price_per_million=0.5,
        )

        manager.update_pricing("custom-model", new_pricing)

        retrieved_pricing = manager.get_pricing("custom-model")
        assert retrieved_pricing is not None
        assert retrieved_pricing.input_price_per_million == 5.0
        assert retrieved_pricing.output_price_per_million == 25.0

    def test_update_pricing_empty_name(self):
        """Test that empty model name raises PricingError."""
        manager = ModelPricingManager()

        new_pricing = ModelPricing(
            input_price_per_million=5.0,
            output_price_per_million=25.0,
            cache_creation_price_per_million=6.25,
        )

        with pytest.raises(PricingError, match="Model name cannot be empty"):
            manager.update_pricing("", new_pricing)

        with pytest.raises(PricingError, match="Model name cannot be empty"):
            manager.update_pricing("   ", new_pricing)

    def test_is_supported_model(self):
        """Test checking if a model is supported."""
        manager = ModelPricingManager()

        assert manager.is_supported_model("claude-3-sonnet") is True
        assert manager.is_supported_model("non-existent-model") is False

    def test_get_all_supported_models(self):
        """Test getting list of all supported models."""
        manager = ModelPricingManager()

        models = manager.get_all_supported_models()
        assert isinstance(models, list)
        assert "claude-3-sonnet" in models
        assert "claude-3-haiku" in models

    def test_load_pricing_from_dict(self):
        """Test loading pricing from dictionary."""
        manager = ModelPricingManager()

        pricing_dict = {
            "custom-model-1": {
                "input_price_per_million": 10.0,
                "output_price_per_million": 50.0,
                "cache_creation_price_per_million": 12.5,
                "cache_read_price_per_million": 1.0,
            },
            "custom-model-2": {
                "input_price_per_million": 1.0,
                "output_price_per_million": 5.0,
                "cache_creation_price_per_million": 1.25,
            },
        }

        manager.load_pricing_from_dict(pricing_dict)

        # Check first model
        pricing1 = manager.get_pricing("custom-model-1")
        assert pricing1 is not None
        assert pricing1.input_price_per_million == 10.0
        assert pricing1.output_price_per_million == 50.0

        # Check second model (missing cache_read_price should default to 0.0)
        pricing2 = manager.get_pricing("custom-model-2")
        assert pricing2 is not None
        assert pricing2.input_price_per_million == 1.0
        assert pricing2.cache_read_price_per_million == 0.0

    def test_load_pricing_from_dict_invalid_data(self):
        """Test that invalid pricing data raises PricingError."""
        manager = ModelPricingManager()

        # Missing required field
        invalid_dict = {
            "bad-model": {
                "input_price_per_million": 10.0,
                # Missing output_price_per_million
                "cache_creation_price_per_million": 12.5,
            }
        }

        with pytest.raises(PricingError, match="Missing required field: output_price_per_million"):
            manager.load_pricing_from_dict(invalid_dict)

    def test_get_confidence_level(self):
        """Test confidence level calculation."""
        manager = ModelPricingManager()

        # Explicitly supported model should have high confidence
        assert manager.get_confidence_level("claude-3-sonnet") == "high"

        # Non-existent model should have medium confidence (fallback to claude-3-sonnet)
        assert manager.get_confidence_level("non-existent-model") == "medium"

    def test_specific_model_pricing_values(self):
        """Test that specific models have expected pricing values."""
        manager = ModelPricingManager()

        # Test claude-sonnet-4 pricing
        sonnet4_pricing = manager.get_pricing("claude-sonnet-4")
        assert sonnet4_pricing is not None
        assert sonnet4_pricing.input_price_per_million == 15.0
        assert sonnet4_pricing.output_price_per_million == 75.0

        # Test claude-3-haiku pricing
        haiku_pricing = manager.get_pricing("claude-3-haiku")
        assert haiku_pricing is not None
        assert haiku_pricing.input_price_per_million == 0.25
        assert haiku_pricing.output_price_per_million == 1.25

        # Test claude-3-sonnet pricing
        sonnet_pricing = manager.get_pricing("claude-3-sonnet")
        assert sonnet_pricing is not None
        assert sonnet_pricing.input_price_per_million == 3.0
        assert sonnet_pricing.output_price_per_million == 15.0


class TestCreateSamplePricingManager:
    """Test the create_sample_pricing_manager function."""

    def test_create_sample_manager(self):
        """Test creating a sample pricing manager."""
        manager = create_sample_pricing_manager()

        assert isinstance(manager, ModelPricingManager)
        assert manager.is_supported_model("claude-3-sonnet")
        assert len(manager.get_all_supported_models()) >= 6


class TestPricingIntegration:
    """Integration tests for pricing functionality."""

    def test_round_trip_dict_conversion(self):
        """Test converting pricing to dict and back preserves data."""
        original = ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
            cache_read_price_per_million=0.3,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = ModelPricing.from_dict(data)

        assert restored.input_price_per_million == original.input_price_per_million
        assert restored.output_price_per_million == original.output_price_per_million
        assert restored.cache_creation_price_per_million == original.cache_creation_price_per_million
        assert restored.cache_read_price_per_million == original.cache_read_price_per_million

    def test_manager_update_and_retrieve(self):
        """Test updating manager pricing and retrieving it."""
        manager = ModelPricingManager()

        # Create custom pricing
        custom_pricing = ModelPricing(
            input_price_per_million=7.5,
            output_price_per_million=37.5,
            cache_creation_price_per_million=9.375,
            cache_read_price_per_million=0.75,
        )

        # Update manager
        manager.update_pricing("test-model", custom_pricing)

        # Retrieve and verify
        retrieved = manager.get_pricing("test-model")
        assert retrieved is not None
        assert retrieved.input_price_per_million == 7.5
        assert retrieved.output_price_per_million == 37.5

        # Verify it's now a supported model
        assert manager.is_supported_model("test-model")
        assert "test-model" in manager.get_all_supported_models()
