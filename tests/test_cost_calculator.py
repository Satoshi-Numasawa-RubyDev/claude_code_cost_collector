"""
Tests for the cost_calculator module.

This module tests the cost calculation functionality including token usage validation,
cost estimation accuracy, confidence level determination, and error handling.
"""

import pytest

from claude_code_cost_collector.cost_calculator import (
    CostCalculationError,
    CostCalculator,
    TokenUsage,
    create_default_cost_calculator,
)
from claude_code_cost_collector.model_pricing import ModelPricingManager


class TestTokenUsage:
    """Test cases for TokenUsage data class."""

    def test_valid_token_usage(self):
        """Test creation of valid TokenUsage objects."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=200,
            cache_read_tokens=1500,
        )

        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.cache_creation_tokens == 200
        assert usage.cache_read_tokens == 1500

    def test_token_usage_defaults(self):
        """Test TokenUsage with default cache values."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.cache_creation_tokens == 0
        assert usage.cache_read_tokens == 0

    def test_negative_tokens_raise_error(self):
        """Test that negative token counts raise errors."""
        with pytest.raises(CostCalculationError, match="input_tokens cannot be negative"):
            TokenUsage(input_tokens=-1, output_tokens=500)

        with pytest.raises(CostCalculationError, match="output_tokens cannot be negative"):
            TokenUsage(input_tokens=1000, output_tokens=-1)

        with pytest.raises(CostCalculationError, match="cache_creation_tokens cannot be negative"):
            TokenUsage(input_tokens=1000, output_tokens=500, cache_creation_tokens=-1)

        with pytest.raises(CostCalculationError, match="cache_read_tokens cannot be negative"):
            TokenUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=-1)

    def test_zero_tokens_raise_error(self):
        """Test that all-zero token counts raise error."""
        with pytest.raises(
            CostCalculationError, match="At least one of input_tokens, output_tokens, or cache_creation_tokens must be positive"
        ):
            TokenUsage(input_tokens=0, output_tokens=0, cache_creation_tokens=0)

    def test_total_tokens_calculation(self):
        """Test total tokens calculation (excluding cache read)."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=200,
            cache_read_tokens=1500,  # Not included in cost total
        )

        assert usage.total_tokens == 1700  # 1000 + 500 + 200
        assert usage.total_tokens_including_cache_read == 3200  # 1000 + 500 + 200 + 1500

    def test_to_dict(self):
        """Test conversion to dictionary."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=200,
            cache_read_tokens=1500,
        )

        expected = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_tokens": 200,
            "cache_read_tokens": 1500,
        }

        assert usage.to_dict() == expected


class TestCostCalculator:
    """Test cases for CostCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pricing_manager = ModelPricingManager()
        self.calculator = CostCalculator(self.pricing_manager)

    def test_calculate_cost_known_model(self):
        """Test cost calculation for a known model."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        # Test with claude-3-sonnet (known pricing)
        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)

        # Expected: (1000/1M * $3) + (500/1M * $15) = $0.003 + $0.0075 = $0.0105
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10

    def test_calculate_cost_with_cache_tokens(self):
        """Test cost calculation with cache tokens."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=2000,
            cache_read_tokens=5000,
        )

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)

        # Expected calculation:
        # Input: 1000/1M * $3 = $0.003
        # Output: 500/1M * $15 = $0.0075
        # Cache creation: 2000/1M * $3.75 = $0.0075
        # Cache read: 5000/1M * $0.3 = $0.0015
        # Total: $0.018
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0 + (2000 / 1_000_000) * 3.75 + (5000 / 1_000_000) * 0.3
        assert abs(cost - expected_cost) < 1e-10

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model uses fallback."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        # Test with unknown model
        cost = self.calculator.calculate_cost("unknown-model", usage)

        # Should use claude-3-sonnet fallback pricing
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10

    def test_estimate_cost_with_confidence_known_model(self):
        """Test cost estimation with confidence for known model."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        cost, confidence = self.calculator.estimate_cost_with_confidence("claude-3-sonnet", usage)

        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10
        assert confidence == "high"

    def test_estimate_cost_with_confidence_unknown_model(self):
        """Test cost estimation with confidence for unknown model."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        cost, confidence = self.calculator.estimate_cost_with_confidence("unknown-model", usage)

        # Should use fallback pricing with medium confidence
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10
        assert confidence == "medium"

    def test_calculate_cost_breakdown(self):
        """Test detailed cost breakdown calculation."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=2000,
            cache_read_tokens=5000,
        )

        breakdown = self.calculator.calculate_cost_breakdown("claude-3-sonnet", usage)

        assert "input_cost" in breakdown
        assert "output_cost" in breakdown
        assert "cache_creation_cost" in breakdown
        assert "cache_read_cost" in breakdown
        assert "total_cost" in breakdown

        # Verify individual calculations
        assert abs(breakdown["input_cost"] - (1000 / 1_000_000) * 3.0) < 1e-10
        assert abs(breakdown["output_cost"] - (500 / 1_000_000) * 15.0) < 1e-10
        assert abs(breakdown["cache_creation_cost"] - (2000 / 1_000_000) * 3.75) < 1e-10
        assert abs(breakdown["cache_read_cost"] - (5000 / 1_000_000) * 0.3) < 1e-10

        # Verify total
        expected_total = sum(
            [breakdown["input_cost"], breakdown["output_cost"], breakdown["cache_creation_cost"], breakdown["cache_read_cost"]]
        )
        assert abs(breakdown["total_cost"] - expected_total) < 1e-10

    def test_is_cost_reliable(self):
        """Test cost reliability checking."""
        # Known model should be reliable at all levels
        assert self.calculator.is_cost_reliable("claude-3-sonnet", "high")
        assert self.calculator.is_cost_reliable("claude-3-sonnet", "medium")
        assert self.calculator.is_cost_reliable("claude-3-sonnet", "low")

        # Unknown model should only be reliable at medium/low levels
        assert not self.calculator.is_cost_reliable("unknown-model", "high")
        assert self.calculator.is_cost_reliable("unknown-model", "medium")
        assert self.calculator.is_cost_reliable("unknown-model", "low")

    def test_get_supported_models(self):
        """Test getting list of supported models."""
        supported_models = self.calculator.get_supported_models()

        assert isinstance(supported_models, list)
        assert "claude-3-sonnet" in supported_models
        assert "claude-3-haiku" in supported_models
        assert "claude-sonnet-4" in supported_models

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        usage = TokenUsage(
            input_tokens=10_000_000,  # 10M tokens
            output_tokens=5_000_000,  # 5M tokens
        )

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)

        # Expected: (10M/1M * $3) + (5M/1M * $15) = $30 + $75 = $105
        expected_cost = 10.0 * 3.0 + 5.0 * 15.0
        assert abs(cost - expected_cost) < 1e-6

    def test_small_token_counts(self):
        """Test cost calculation with very small token counts."""
        usage = TokenUsage(input_tokens=1, output_tokens=1)

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)

        # Expected: (1/1M * $3) + (1/1M * $15) = $0.000003 + $0.000015 = $0.000018
        expected_cost = (1 / 1_000_000) * 3.0 + (1 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10


class TestErrorHandling:
    """Test error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pricing_manager = ModelPricingManager()
        self.calculator = CostCalculator(self.pricing_manager)

    def test_invalid_token_usage_in_calculate_cost(self):
        """Test that invalid token usage raises appropriate errors."""
        # This should be caught during TokenUsage creation, not in calculate_cost
        with pytest.raises(CostCalculationError):
            invalid_usage = TokenUsage.__new__(TokenUsage)
            invalid_usage.input_tokens = -1
            invalid_usage.output_tokens = 500
            invalid_usage.__post_init__()

    def test_empty_model_name(self):
        """Test handling of empty model names."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        # Empty model name should still work with fallback
        cost = self.calculator.calculate_cost("", usage)
        assert cost > 0

    def test_none_model_name(self):
        """Test handling of None model names."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        # None model name should use fallback pricing (no error)
        cost = self.calculator.calculate_cost(None, usage)
        assert cost > 0  # Should calculate using fallback pricing

    def test_pricing_error_in_calculate_cost(self):
        """Test error handling when PricingError occurs in calculate_cost."""
        # Create a mock pricing manager that raises PricingError
        from unittest.mock import Mock

        from claude_code_cost_collector.model_pricing import PricingError

        mock_manager = Mock()
        mock_manager.get_pricing_or_fallback.side_effect = PricingError("Mock pricing error")

        calculator = CostCalculator(mock_manager)
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        with pytest.raises(CostCalculationError, match="Failed to calculate cost for model 'test-model': Mock pricing error"):
            calculator.calculate_cost("test-model", usage)

    def test_pricing_error_in_calculate_cost_breakdown(self):
        """Test error handling when PricingError occurs in calculate_cost_breakdown."""
        # Create a mock pricing manager that raises PricingError
        from unittest.mock import Mock

        from claude_code_cost_collector.model_pricing import PricingError

        mock_manager = Mock()
        mock_manager.get_pricing_or_fallback.side_effect = PricingError("Mock pricing error")

        calculator = CostCalculator(mock_manager)
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        with pytest.raises(
            CostCalculationError, match="Failed to calculate cost breakdown for model 'test-model': Mock pricing error"
        ):
            calculator.calculate_cost_breakdown("test-model", usage)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pricing_manager = ModelPricingManager()
        self.calculator = CostCalculator(self.pricing_manager)

    def test_zero_cache_tokens(self):
        """Test calculation with zero cache tokens."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10

    def test_only_cache_tokens(self):
        """Test calculation with only cache tokens."""
        usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1000,
            cache_read_tokens=500,
        )

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)
        expected_cost = (1000 / 1_000_000) * 3.75 + (500 / 1_000_000) * 0.3
        assert abs(cost - expected_cost) < 1e-10

    def test_only_input_tokens(self):
        """Test calculation with only input tokens."""
        usage = TokenUsage(input_tokens=1000, output_tokens=0)

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)
        expected_cost = (1000 / 1_000_000) * 3.0
        assert abs(cost - expected_cost) < 1e-10

    def test_only_output_tokens(self):
        """Test calculation with only output tokens."""
        usage = TokenUsage(input_tokens=0, output_tokens=1000)

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)
        expected_cost = (1000 / 1_000_000) * 15.0
        assert abs(cost - expected_cost) < 1e-10

    def test_confidence_threshold_edge_cases(self):
        """Test confidence threshold edge cases."""
        calculator = self.calculator

        # Test with invalid confidence threshold (defaults to medium)
        # Unknown model has medium confidence, so should pass medium threshold
        assert calculator.is_cost_reliable("unknown-model", "invalid-threshold")

        # Test with edge case thresholds
        assert calculator.is_cost_reliable("claude-3-sonnet", "low")
        assert calculator.is_cost_reliable("claude-3-sonnet", "medium")
        assert calculator.is_cost_reliable("claude-3-sonnet", "high")

        # Test reliability with unknown model at different thresholds
        assert not calculator.is_cost_reliable("unknown-model", "high")
        assert calculator.is_cost_reliable("unknown-model", "medium")
        assert calculator.is_cost_reliable("unknown-model", "low")

        # Test threshold boundary - create empty manager to force low confidence
        from claude_code_cost_collector.model_pricing import ModelPricingManager

        empty_manager = ModelPricingManager()
        empty_manager._pricing_data.clear()  # Remove all models
        empty_calculator = CostCalculator(empty_manager)

        # Should have low confidence and fail medium/high thresholds
        assert not empty_calculator.is_cost_reliable("any-model", "medium")
        assert not empty_calculator.is_cost_reliable("any-model", "high")
        assert empty_calculator.is_cost_reliable("any-model", "low")

    def test_extremely_large_token_values(self):
        """Test with extremely large token values."""
        usage = TokenUsage(
            input_tokens=999_999_999,  # Near 1 billion tokens
            output_tokens=999_999_999,
            cache_creation_tokens=999_999_999,
            cache_read_tokens=999_999_999,
        )

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)

        # Should handle large values without overflow
        assert cost > 0
        assert cost < float("inf")

        # Verify calculation is approximately correct
        expected_input = (999_999_999 / 1_000_000) * 3.0
        expected_output = (999_999_999 / 1_000_000) * 15.0
        expected_cache_creation = (999_999_999 / 1_000_000) * 3.75
        expected_cache_read = (999_999_999 / 1_000_000) * 0.3
        expected_total = expected_input + expected_output + expected_cache_creation + expected_cache_read

        assert abs(cost - expected_total) < 1e-3  # Allow small floating point errors

    def test_fractional_token_handling(self):
        """Test that integer token values work correctly with fractional results."""
        # Test with values that result in fractional costs
        usage = TokenUsage(input_tokens=1, output_tokens=1)

        cost = self.calculator.calculate_cost("claude-3-sonnet", usage)

        # Very small cost should be handled properly
        expected_cost = (1 / 1_000_000) * 3.0 + (1 / 1_000_000) * 15.0  # 0.000018
        assert abs(cost - expected_cost) < 1e-10
        assert 0 < cost < 0.001  # Should be a very small positive value


class TestIntegration:
    """Integration tests with real-world scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = create_default_cost_calculator()

    def test_create_default_cost_calculator(self):
        """Test creating default cost calculator."""
        calculator = create_default_cost_calculator()
        assert isinstance(calculator, CostCalculator)
        assert isinstance(calculator.pricing_manager, ModelPricingManager)

    def test_realistic_claude_sonnet_4_usage(self):
        """Test with realistic Claude Sonnet 4 usage."""
        # Based on the sample data from new format analysis
        usage = TokenUsage(
            input_tokens=4,
            output_tokens=252,
            cache_creation_tokens=6094,
            cache_read_tokens=13558,
        )

        cost, confidence = self.calculator.estimate_cost_with_confidence("claude-sonnet-4-20250514", usage)

        # Verify the calculation
        # Input: 4/1M * $3.0 = $0.000012
        # Output: 252/1M * $15.0 = $0.00378
        # Cache creation: 6094/1M * $3.75 = $0.0228525
        # Cache read: 13558/1M * $0.3 = $0.0040674
        expected_cost = (4 / 1_000_000) * 3.0 + (252 / 1_000_000) * 15.0 + (6094 / 1_000_000) * 3.75 + (13558 / 1_000_000) * 0.3

        assert abs(cost - expected_cost) < 1e-6
        assert confidence == "high"

    def test_confidence_levels_across_models(self):
        """Test confidence levels for different model types."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        # Known specific model should have high confidence
        _, confidence1 = self.calculator.estimate_cost_with_confidence("claude-3-sonnet", usage)
        assert confidence1 == "high"

        # Known versioned model should have high confidence
        _, confidence2 = self.calculator.estimate_cost_with_confidence("claude-sonnet-4-20250514", usage)
        assert confidence2 == "high"

        # Unknown model should have medium confidence (fallback)
        _, confidence3 = self.calculator.estimate_cost_with_confidence("future-claude-model", usage)
        assert confidence3 == "medium"

    def test_cost_breakdown_totals_match(self):
        """Test that cost breakdown totals match direct calculation."""
        usage = TokenUsage(
            input_tokens=1500,
            output_tokens=750,
            cache_creation_tokens=3000,
            cache_read_tokens=8000,
        )

        direct_cost = self.calculator.calculate_cost("claude-3-haiku", usage)
        breakdown = self.calculator.calculate_cost_breakdown("claude-3-haiku", usage)

        assert abs(direct_cost - breakdown["total_cost"]) < 1e-10

    def test_real_v1_0_9_sample_cost_calculation(self):
        """Test cost calculation with real v1.0.9 sample data."""
        # Real token usage from v1.0.9 sample data
        usage = TokenUsage(
            input_tokens=4,
            output_tokens=252,
            cache_creation_tokens=6094,
            cache_read_tokens=13558,
        )

        # Test with claude-sonnet-4-20250514 (the actual model from sample)
        cost, confidence = self.calculator.estimate_cost_with_confidence("claude-sonnet-4-20250514", usage)

        # Should have high confidence for supported model
        assert confidence == "high"

        # Calculate expected cost based on known pricing
        pricing = self.calculator.pricing_manager.get_pricing("claude-sonnet-4-20250514")
        assert pricing is not None

        expected_cost = (
            (4 / 1_000_000) * pricing.input_price_per_million
            + (252 / 1_000_000) * pricing.output_price_per_million
            + (6094 / 1_000_000) * pricing.cache_creation_price_per_million
            + (13558 / 1_000_000) * pricing.cache_read_price_per_million
        )

        assert abs(cost - expected_cost) < 1e-6
        assert cost > 0  # Should be a reasonable positive cost

        # Verify breakdown matches
        breakdown = self.calculator.calculate_cost_breakdown("claude-sonnet-4-20250514", usage)
        assert abs(breakdown["total_cost"] - cost) < 1e-10


class TestCostCalculatorPerformance:
    """Performance tests for CostCalculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = create_default_cost_calculator()

    def test_bulk_cost_calculation_performance(self):
        """Test performance with bulk cost calculations."""
        import time

        # Create many token usage scenarios
        usage_scenarios = []
        for i in range(1000):
            usage = TokenUsage(
                input_tokens=100 + i,
                output_tokens=50 + i // 2,
                cache_creation_tokens=i * 10,
                cache_read_tokens=i * 20,
            )
            usage_scenarios.append(usage)

        models = ["claude-3-sonnet", "claude-3-haiku", "claude-sonnet-4"]

        start_time = time.time()
        total_cost = 0.0

        for i, usage in enumerate(usage_scenarios):
            model = models[i % len(models)]
            cost = self.calculator.calculate_cost(model, usage)
            total_cost += cost

        end_time = time.time()

        # Should complete 1000 calculations quickly (< 0.5 seconds)
        calculation_time = end_time - start_time
        assert calculation_time < 0.5

        # Total cost should be reasonable
        assert total_cost > 0
        assert total_cost < 10000  # Sanity check

    def test_confidence_calculation_performance(self):
        """Test performance of confidence calculations."""
        import time

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        models = [f"test-model-{i}" for i in range(100)]

        start_time = time.time()

        for model in models:
            cost, confidence = self.calculator.estimate_cost_with_confidence(model, usage)
            assert cost > 0
            assert confidence in ["high", "medium", "low"]

        end_time = time.time()

        # Should complete 100 confidence calculations quickly (< 0.1 seconds)
        calculation_time = end_time - start_time
        assert calculation_time < 0.1

    def test_cost_breakdown_performance(self):
        """Test performance of detailed cost breakdown calculations."""
        import time

        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=2000,
            cache_read_tokens=5000,
        )

        models = ["claude-3-sonnet", "claude-3-haiku", "claude-sonnet-4"]

        start_time = time.time()

        for _ in range(100):
            for model in models:
                breakdown = self.calculator.calculate_cost_breakdown(model, usage)
                assert len(breakdown) == 5  # 4 cost components + total
                assert breakdown["total_cost"] > 0

        end_time = time.time()

        # Should complete 300 breakdown calculations quickly (< 0.2 seconds)
        calculation_time = end_time - start_time
        assert calculation_time < 0.2

    def test_reliability_check_performance(self):
        """Test performance of reliability checks."""
        import time

        models = [f"test-model-{i}" for i in range(200)]
        thresholds = ["low", "medium", "high"]

        start_time = time.time()

        for model in models:
            for threshold in thresholds:
                reliable = self.calculator.is_cost_reliable(model, threshold)
                assert isinstance(reliable, bool)

        end_time = time.time()

        # Should complete 600 reliability checks quickly (< 0.1 seconds)
        calculation_time = end_time - start_time
        assert calculation_time < 0.1
