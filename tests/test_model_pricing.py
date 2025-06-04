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
        assert sonnet4_pricing.input_price_per_million == 3.0
        assert sonnet4_pricing.output_price_per_million == 15.0

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


class TestModelPricingEdgeCases:
    """Test edge cases and boundary conditions for ModelPricing."""

    def test_unicode_model_names(self):
        """Test handling of Unicode model names."""
        manager = ModelPricingManager()
        pricing = ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
        )

        # Test with Unicode characters
        unicode_model = "claude-日本語-モデル"
        manager.update_pricing(unicode_model, pricing)

        retrieved = manager.get_pricing(unicode_model)
        assert retrieved is not None
        assert retrieved.input_price_per_million == 3.0

    def test_very_long_model_names(self):
        """Test handling of very long model names."""
        manager = ModelPricingManager()
        pricing = ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
        )

        # Test with very long model name
        long_model = "claude-" + "x" * 1000 + "-model"
        manager.update_pricing(long_model, pricing)

        retrieved = manager.get_pricing(long_model)
        assert retrieved is not None
        assert manager.is_supported_model(long_model)

    def test_extreme_pricing_values(self):
        """Test handling of extreme pricing values."""
        # Very high pricing (enterprise-level)
        high_pricing = ModelPricing(
            input_price_per_million=100.0,
            output_price_per_million=500.0,
            cache_creation_price_per_million=125.0,
            cache_read_price_per_million=10.0,
        )
        assert high_pricing.input_price_per_million == 100.0
        assert high_pricing.output_price_per_million == 500.0

        # Very low pricing (free tier)
        low_pricing = ModelPricing(
            input_price_per_million=0.001,
            output_price_per_million=0.005,
            cache_creation_price_per_million=0.00125,
            cache_read_price_per_million=0.0001,
        )
        assert low_pricing.input_price_per_million == 0.001
        assert low_pricing.output_price_per_million == 0.005

        # Zero pricing (free model)
        zero_pricing = ModelPricing(
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            cache_creation_price_per_million=0.0,
            cache_read_price_per_million=0.0,
        )
        assert zero_pricing.input_price_per_million == 0.0
        assert zero_pricing.output_price_per_million == 0.0

    def test_float_precision_handling(self):
        """Test handling of floating point precision."""
        # Test very precise pricing values
        precise_pricing = ModelPricing(
            input_price_per_million=3.123456789,
            output_price_per_million=15.987654321,
            cache_creation_price_per_million=3.75000001,
            cache_read_price_per_million=0.300000001,
        )

        # Verify precision is preserved
        assert abs(precise_pricing.input_price_per_million - 3.123456789) < 1e-9
        assert abs(precise_pricing.output_price_per_million - 15.987654321) < 1e-9

        # Test round-trip conversion maintains precision
        data = precise_pricing.to_dict()
        restored = ModelPricing.from_dict(data)
        assert abs(restored.input_price_per_million - precise_pricing.input_price_per_million) < 1e-9

    def test_pricing_validation_edge_cases(self):
        """Test validation with edge case values."""
        # Test validation passes with minimal valid values
        minimal_pricing = ModelPricing(
            input_price_per_million=0.0,
            output_price_per_million=0.001,  # Very small but positive
            cache_creation_price_per_million=0.0,
        )
        assert minimal_pricing.output_price_per_million == 0.001

        # Test inverse pricing (output cheaper than input - unusual but not invalid)
        inverse_pricing = ModelPricing(
            input_price_per_million=10.0,
            output_price_per_million=5.0,  # Cheaper than input
            cache_creation_price_per_million=8.0,
        )
        # Should not raise an error (warning case, not error)
        assert inverse_pricing.output_price_per_million == 5.0


class TestModelPricingManagerEdgeCases:
    """Test edge cases for ModelPricingManager."""

    def test_ultimate_fallback_scenario(self):
        """Test ultimate fallback when even claude-3-sonnet is missing."""
        manager = ModelPricingManager()

        # Remove the claude-3-sonnet fallback
        manager._pricing_data.clear()

        # Should use ultimate fallback pricing
        pricing = manager.get_pricing_or_fallback("unknown-model")
        assert pricing.input_price_per_million == 3.0
        assert pricing.output_price_per_million == 15.0
        assert pricing.cache_creation_price_per_million == 3.75
        assert pricing.cache_read_price_per_million == 0.3

        # Confidence should be low
        confidence = manager.get_confidence_level("unknown-model")
        assert confidence == "low"

    def test_empty_manager_operations(self):
        """Test operations on empty pricing manager."""
        manager = ModelPricingManager()
        manager._pricing_data.clear()

        # Test all operations work with empty manager
        assert manager.get_pricing("any-model") is None
        assert not manager.is_supported_model("any-model")
        assert manager.get_all_supported_models() == []
        assert manager.get_confidence_level("any-model") == "low"

        # Fallback should still work
        fallback_pricing = manager.get_pricing_or_fallback("any-model")
        assert fallback_pricing is not None

    def test_large_batch_operations(self):
        """Test operations with large numbers of models."""
        manager = ModelPricingManager()

        # Add many models
        num_models = 100
        for i in range(num_models):
            pricing = ModelPricing(
                input_price_per_million=float(i + 1),
                output_price_per_million=float((i + 1) * 5),
                cache_creation_price_per_million=float((i + 1) * 1.25),
            )
            manager.update_pricing(f"test-model-{i}", pricing)

        # Verify all models are present
        all_models = manager.get_all_supported_models()
        test_models = [f"test-model-{i}" for i in range(num_models)]
        for model in test_models:
            assert model in all_models

        # Test retrieval of specific models
        pricing_50 = manager.get_pricing("test-model-50")
        assert pricing_50 is not None
        assert pricing_50.input_price_per_million == 51.0

    def test_whitespace_model_names(self):
        """Test handling of model names with whitespace."""
        manager = ModelPricingManager()
        pricing = ModelPricing(
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cache_creation_price_per_million=3.75,
        )

        # Names with leading/trailing whitespace should be trimmed
        manager.update_pricing("  test-model  ", pricing)
        assert manager.is_supported_model("test-model")
        assert not manager.is_supported_model("  test-model  ")

        # Retrieve with trimmed name
        retrieved = manager.get_pricing("test-model")
        assert retrieved is not None

    def test_invalid_pricing_data_handling(self):
        """Test handling of various invalid pricing data formats."""
        manager = ModelPricingManager()

        # Test with non-numeric values
        invalid_dict_1 = {
            "model-with-string-price": {
                "input_price_per_million": "not-a-number",
                "output_price_per_million": 15.0,
                "cache_creation_price_per_million": 3.75,
            }
        }
        with pytest.raises(PricingError):
            manager.load_pricing_from_dict(invalid_dict_1)

        # Test with None values
        invalid_dict_2 = {
            "model-with-none": {
                "input_price_per_million": None,
                "output_price_per_million": 15.0,
                "cache_creation_price_per_million": 3.75,
            }
        }
        with pytest.raises(PricingError):
            manager.load_pricing_from_dict(invalid_dict_2)

        # Test with nested invalid structure
        invalid_dict_3 = {
            "model-nested-invalid": {
                "pricing": {
                    "input_price_per_million": 3.0,
                    "output_price_per_million": 15.0,
                }
            }
        }
        with pytest.raises(PricingError):
            manager.load_pricing_from_dict(invalid_dict_3)


class TestModelPricingPerformance:
    """Performance and scalability tests for ModelPricing."""

    def test_pricing_calculation_performance(self):
        """Test that pricing operations perform well with realistic loads."""
        import time

        # Create a manager with many models
        manager = ModelPricingManager()
        num_models = 1000

        # Time model addition
        start_time = time.time()
        for i in range(num_models):
            pricing = ModelPricing(
                input_price_per_million=float(i % 100 + 1),
                output_price_per_million=float((i % 100 + 1) * 5),
                cache_creation_price_per_million=float((i % 100 + 1) * 1.25),
            )
            manager.update_pricing(f"perf-model-{i}", pricing)
        add_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second)
        assert add_time < 1.0

        # Time model retrieval
        start_time = time.time()
        for i in range(0, num_models, 10):  # Sample every 10th model
            pricing = manager.get_pricing(f"perf-model-{i}")
            assert pricing is not None
        retrieval_time = time.time() - start_time

        # Retrieval should be very fast (< 0.1 second)
        assert retrieval_time < 0.1

    def test_large_dict_loading_performance(self):
        """Test performance of loading large pricing dictionaries."""
        import time

        # Create large pricing dictionary
        large_dict = {}
        for i in range(500):
            large_dict[f"bulk-model-{i}"] = {
                "input_price_per_million": float(i % 50 + 1),
                "output_price_per_million": float((i % 50 + 1) * 5),
                "cache_creation_price_per_million": float((i % 50 + 1) * 1.25),
                "cache_read_price_per_million": float((i % 50 + 1) * 0.1),
            }

        manager = ModelPricingManager()
        start_time = time.time()
        manager.load_pricing_from_dict(large_dict)
        load_time = time.time() - start_time

        # Should load efficiently (< 0.5 second)
        assert load_time < 0.5

        # Verify all models loaded correctly
        all_models = manager.get_all_supported_models()
        bulk_models = [f"bulk-model-{i}" for i in range(500)]
        for model in bulk_models:
            assert model in all_models


class TestNewFormatIntegration:
    """Test integration with new v1.0.9 format requirements."""

    def test_v1_0_9_model_support(self):
        """Test that all models from v1.0.9 sample data are supported."""
        manager = ModelPricingManager()

        # Models that appear in v1.0.9 sample data
        v1_0_9_models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
        ]

        for model in v1_0_9_models:
            pricing = manager.get_pricing(model)
            if pricing is not None:
                # If explicitly supported, should have high confidence
                confidence = manager.get_confidence_level(model)
                assert confidence == "high"
            else:
                # If not explicitly supported, should use fallback
                fallback_pricing = manager.get_pricing_or_fallback(model)
                assert fallback_pricing is not None
                confidence = manager.get_confidence_level(model)
                assert confidence in ["medium", "low"]

    def test_cost_estimation_reliability(self):
        """Test reliability indicators for cost estimation."""
        manager = ModelPricingManager()

        # Test with known model
        assert manager.get_confidence_level("claude-3-sonnet") == "high"
        assert manager.get_confidence_level("claude-sonnet-4") == "high"

        # Test with unknown model (should use fallback)
        assert manager.get_confidence_level("future-model-2026") in ["medium", "low"]

        # Test pricing reliability
        known_pricing = manager.get_pricing_or_fallback("claude-3-sonnet")
        unknown_pricing = manager.get_pricing_or_fallback("future-model-2026")

        # Both should return valid pricing
        assert known_pricing.input_price_per_million > 0
        assert unknown_pricing.input_price_per_million > 0

    def test_token_cost_calculation_accuracy(self):
        """Test accuracy of token-based cost calculations."""
        manager = ModelPricingManager()

        # Test with claude-sonnet-4 pricing (common in v1.0.9)
        pricing = manager.get_pricing_or_fallback("claude-sonnet-4-20250514")

        # Sample token usage from v1.0.9 format
        input_tokens = 4
        output_tokens = 252
        cache_creation_tokens = 6094
        cache_read_tokens = 13558

        # Calculate expected cost
        expected_input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_million
        expected_output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_million
        expected_cache_creation_cost = (cache_creation_tokens / 1_000_000) * pricing.cache_creation_price_per_million
        expected_cache_read_cost = (cache_read_tokens / 1_000_000) * pricing.cache_read_price_per_million

        total_expected = expected_input_cost + expected_output_cost + expected_cache_creation_cost + expected_cache_read_cost

        # Verify calculation is reasonable (should be small positive amount)
        assert total_expected > 0
        assert total_expected < 1.0  # Should be less than $1 for this sample

    def test_v1_0_9_specific_models_pricing(self):
        """Test pricing for models specifically mentioned in v1.0.9 samples."""
        manager = ModelPricingManager()

        # Models that appear in actual v1.0.9 sample data
        v1_0_9_sample_models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
        ]

        for model in v1_0_9_sample_models:
            pricing = manager.get_pricing_or_fallback(model)
            assert pricing is not None

            # All should have reasonable pricing values
            assert pricing.input_price_per_million > 0
            assert pricing.output_price_per_million > 0
            assert pricing.cache_creation_price_per_million >= 0
            assert pricing.cache_read_price_per_million >= 0

            # Output should typically be more expensive than input
            if model in ["claude-opus-4-20250514", "claude-opus-4"]:
                # Opus models should have higher pricing
                assert pricing.input_price_per_million >= 10.0
            elif model in ["claude-sonnet-4-20250514", "claude-sonnet-4"]:
                # Sonnet-4 models should have reasonable pricing
                assert pricing.input_price_per_million >= 3.0

            # Check confidence level
            confidence = manager.get_confidence_level(model)
            assert confidence in ["high", "medium", "low"]

    def test_real_sample_cost_estimation_accuracy(self):
        """Test cost estimation accuracy with real sample token usage."""
        manager = ModelPricingManager()

        # Real token usage from v1.0.9 sample
        sample_token_usage = {
            "input_tokens": 4,
            "cache_creation_input_tokens": 6094,
            "cache_read_input_tokens": 13558,
            "output_tokens": 252,
        }

        # Test with actual model from sample
        model = "claude-sonnet-4-20250514"
        pricing = manager.get_pricing_or_fallback(model)

        # Calculate cost manually
        input_cost = (sample_token_usage["input_tokens"] / 1_000_000) * pricing.input_price_per_million
        output_cost = (sample_token_usage["output_tokens"] / 1_000_000) * pricing.output_price_per_million
        cache_creation_cost = (sample_token_usage["cache_creation_input_tokens"] / 1_000_000) * pricing.cache_creation_price_per_million
        cache_read_cost = (sample_token_usage["cache_read_input_tokens"] / 1_000_000) * pricing.cache_read_price_per_million

        total_cost = input_cost + output_cost + cache_creation_cost + cache_read_cost

        # Cost should be reasonable for this usage pattern
        assert total_cost > 0
        assert total_cost < 1.0  # Should be under $1 for this small usage

        # Most cost should come from cache creation due to large token count
        assert cache_creation_cost > input_cost
        assert cache_creation_cost > output_cost

        # Confidence should be high for supported model
        confidence = manager.get_confidence_level(model)
        assert confidence == "high"
