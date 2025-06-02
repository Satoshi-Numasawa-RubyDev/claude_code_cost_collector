"""
Exchange rate functionality for Claude Code Cost Collector.

This module provides functions to fetch currency exchange rates
from external APIs for cost conversion purposes.
"""

import json
import logging
from datetime import date
from typing import Any, Dict, Optional

import requests


class ExchangeRateError(Exception):
    """Exception raised when exchange rate retrieval fails."""

    pass


class ExchangeRateCache:
    """Simple in-memory cache for exchange rates to avoid repeated API calls."""

    def __init__(self) -> None:
        self._cache: Dict[str, float] = {}
        self._cache_date: Optional[date] = None

    def get_rate(self, currency_pair: str) -> Optional[float]:
        """Get cached exchange rate if available and still valid (same day)."""
        today = date.today()
        if self._cache_date == today and currency_pair in self._cache:
            return self._cache[currency_pair]
        return None

    def set_rate(self, currency_pair: str, rate: float) -> None:
        """Cache exchange rate for the current day."""
        today = date.today()
        if self._cache_date != today:
            # Clear cache if date changed
            self._cache.clear()
            self._cache_date = today
        self._cache[currency_pair] = rate


# Global cache instance
_rate_cache = ExchangeRateCache()


def get_exchange_rate(
    from_currency: str = "USD",
    to_currency: str = "EUR",
    api_key: Optional[str] = None,
    timeout: int = 10,
    use_cache: bool = True,
    fallback_rate: Optional[float] = None,
) -> float:
    """
    Get exchange rate between two currencies from external API.

    Args:
        from_currency: Base currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR", "GBP", "CAD")
        api_key: API key for exchangerate-api.com (optional for free tier)
        timeout: Request timeout in seconds
        use_cache: Whether to use cached rate if available
        fallback_rate: Default rate to use if API fails (auto-determined if None)

    Returns:
        Exchange rate as float (from_currency to to_currency)

    Raises:
        ExchangeRateError: If API request fails and no fallback is provided
    """
    # Validate currency codes
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    # If same currency, return 1.0
    if from_currency == to_currency:
        return 1.0

    currency_pair = f"{from_currency}_{to_currency}"
    # Set default fallback rates for common currency pairs
    if fallback_rate is None:
        fallback_rates = {
            "USD_EUR": 0.85,
            "USD_GBP": 0.75,
            "USD_CAD": 1.35,
            "USD_AUD": 1.50,
        }
        fallback_rate = fallback_rates.get(currency_pair, 1.0)

    # Check cache first
    if use_cache:
        cached_rate = _rate_cache.get_rate(currency_pair)
        if cached_rate is not None:
            logging.debug(f"Using cached exchange rate {currency_pair}: {cached_rate}")
            return cached_rate

    try:
        # Use exchangerate-api.com free tier endpoint
        if api_key:
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency}"
        else:
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"

        logging.debug(f"Fetching exchange rate from: {url} for {currency_pair}")

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        # Extract target currency rate from response
        try:
            if api_key:
                # v6 API format
                if "conversion_rates" in data and to_currency in data["conversion_rates"]:
                    rate = float(data["conversion_rates"][to_currency])
                else:
                    raise ValueError(f"{to_currency} rate not found in API response")
            else:
                # v4 API format
                if "rates" in data and to_currency in data["rates"]:
                    rate = float(data["rates"][to_currency])
                else:
                    raise ValueError(f"{to_currency} rate not found in API response")

            # Basic sanity check - rate should be positive
            if rate <= 0:
                raise ValueError(f"Exchange rate {rate} is not positive")
            # Additional sanity checks for common currency pairs
            if currency_pair == "USD_EUR" and (rate < 0.5 or rate > 1.5):
                raise ValueError(f"USD/EUR exchange rate {rate} seems unrealistic")
            elif currency_pair == "USD_GBP" and (rate < 0.5 or rate > 1.5):
                raise ValueError(f"USD/GBP exchange rate {rate} seems unrealistic")

        except ValueError as e:
            logging.warning(f"Error validating exchange rate: {e}")
            if fallback_rate > 0:
                logging.warning(f"Using fallback exchange rate: {fallback_rate}")
                return fallback_rate
            raise ExchangeRateError(f"Invalid exchange rate data: {e}")

        # Cache the rate
        if use_cache:
            _rate_cache.set_rate(currency_pair, rate)

        logging.info(f"Successfully fetched {currency_pair} exchange rate: {rate}")
        return rate

    except requests.exceptions.RequestException as e:
        logging.warning(f"Network error while fetching exchange rate: {e}")
        if fallback_rate > 0:
            logging.warning(f"Using fallback exchange rate: {fallback_rate}")
            return fallback_rate
        raise ExchangeRateError(f"Failed to fetch exchange rate: {e}")

    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logging.warning(f"Error parsing exchange rate response: {e}")
        if fallback_rate > 0:
            logging.warning(f"Using fallback exchange rate: {fallback_rate}")
            return fallback_rate
        raise ExchangeRateError(f"Failed to parse exchange rate response: {e}")


def convert_currency(
    amount: float,
    from_currency: str = "USD",
    to_currency: str = "EUR",
    api_key: Optional[str] = None,
    exchange_rate: Optional[float] = None,
    **kwargs: Any,
) -> float:
    """
    Convert amount from one currency to another.

    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR", "GBP", "CAD")
        api_key: API key for exchange rate service
        exchange_rate: Pre-fetched exchange rate (if provided, skips API call)
        **kwargs: Additional arguments passed to get_exchange_rate()

    Returns:
        Converted amount in target currency
    """
    if exchange_rate is None:
        exchange_rate = get_exchange_rate(from_currency=from_currency, to_currency=to_currency, api_key=api_key, **kwargs)

    converted_amount = amount * exchange_rate
    return round(converted_amount, 2)
