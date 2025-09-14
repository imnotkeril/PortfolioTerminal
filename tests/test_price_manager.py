"""
Unit tests for PriceManager and related functionality.

This module tests market data fetching, caching, and price management
functionality to ensure reliable data operations.
"""

import pytest
import time
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from core.data_manager import (
    PriceManager, YahooFinanceProvider, PriceCache, Quote,
    MarketStatus, CompanyInfo, Portfolio, Asset, AssetClass
)


class TestPriceCache:
    """Test price caching functionality"""

    def setup_method(self):
        """Setup cache for testing"""
        self.cache = PriceCache(ttl_seconds=1)  # 1 second TTL for testing

    def test_cache_basic_operations(self):
        """Test basic cache set/get operations"""
        # Set value
        self.cache.set("AAPL", 150.0)

        # Get value (should exist)
        value = self.cache.get("AAPL")
        assert value == 150.0

        # Get non-existent value
        value = self.cache.get("NONEXISTENT")
        assert value is None

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        # Set value
        self.cache.set("AAPL", 150.0)

        # Should be available immediately
        assert self.cache.get("AAPL") == 150.0

        # Wait for expiration (TTL is 1 second)
        time.sleep(1.1)

        # Should be expired now
        assert self.cache.get("AAPL") is None

    def test_cache_clear(self):
        """Test cache clearing"""
        # Set multiple values
        self.cache.set("AAPL", 150.0)
        self.cache.set("MSFT", 300.0)

        # Verify they exist
        assert self.cache.get("AAPL") == 150.0
        assert self.cache.get("MSFT") == 300.0

        # Clear cache
        self.cache.clear()

        # Should be empty
        assert self.cache.get("AAPL") is None
        assert self.cache.get("MSFT") is None

    def test_cache_stats(self):
        """Test cache statistics"""
        # Add some data
        self.cache.set("AAPL", 150.0)
        self.cache.set("MSFT", 300.0)

        stats = self.cache.get_stats()

        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 2
        assert stats['expired_entries'] == 0
        assert stats['cache_size_mb'] > 0

    def test_clear_expired(self):
        """Test clearing only expired entries"""
        # Set values with different timing
        self.cache.set("AAPL", 150.0)
        time.sleep(0.5)  # Half TTL
        self.cache.set("MSFT", 300.0)

        # Wait for first to expire
        time.sleep(0.6)

        # Clear expired
        self.cache.clear_expired()

        # AAPL should be gone, MSFT should remain
        assert self.cache.get("AAPL") is None
        assert self.cache.get("MSFT") == 300.0


class TestYahooFinanceProvider:
    """Test Yahoo Finance data provider"""

    def setup_method(self):
        """Setup provider for testing"""
        self.provider = YahooFinanceProvider()

    @pytest.mark.integration
    def test_get_current_price(self):
        """Test fetching single current price (integration test)"""
        # Test with a stable, well-known ticker
        price = self.provider.get_current_price("AAPL")

        if price is not None:  # If internet available
            assert isinstance(price, float)
            assert price > 0
        else:
            pytest.skip("Yahoo Finance not available")

    @pytest.mark.integration
    def test_get_current_prices_batch(self):
        """Test batch price fetching (integration test)"""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices = self.provider.get_current_prices(tickers)

        if prices:  # If any prices returned
            for ticker, price in prices.items():
                assert ticker in tickers
                assert isinstance(price, float)
                assert price > 0
        else:
            pytest.skip("Yahoo Finance not available")

    def test_invalid_ticker(self):
        """Test handling of invalid ticker"""
        price = self.provider.get_current_price("INVALID_TICKER_12345")
        assert price is None

    @pytest.mark.integration
    def test_historical_data(self):
        """Test historical data fetching (integration test)"""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        data = self.provider.get_historical_data("AAPL", start, end)

        if not data.empty:  # If data available
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in expected_columns if col in data.columns]
            assert len(available_columns) > 0
            assert len(data) > 0
        else:
            pytest.skip("Historical data not available")

    @pytest.mark.integration
    def test_company_info(self):
        """Test company information fetching (integration test)"""
        info = self.provider.get_company_info("AAPL")

        if info is not None:  # If data available
            assert isinstance(info, CompanyInfo)
            assert info.ticker == "AAPL"
            assert len(info.name) > 0
        else:
            pytest.skip("Company info not available")

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Test that rate limiting doesn't break functionality
        provider = YahooFinanceProvider()

        # Make multiple quick requests
        for i in range(3):
            price = provider.get_current_price("AAPL")
            # Should not crash, may return None if rate limited

        # No assertion needed - just testing it doesn't crash

    def test_provider_availability(self):
        """Test provider availability check"""
        # This is more of a connectivity test
        is_available = self.provider.is_available()
        # Result depends on internet connectivity
        assert isinstance(is_available, bool)


class TestPriceManager:
    """Test main PriceManager functionality"""

    def setup_method(self):
        """Setup price manager for testing"""
        # Use short cache TTL for testing
        self.price_manager = PriceManager(cache_ttl=1, enable_cache=True)

    def test_price_manager_initialization(self):
        """Test price manager initialization"""
        assert self.price_manager.primary_provider is not None
        assert self.price_manager.cache is not None
        assert self.price_manager.enable_cache == True

    def test_cache_disabled_mode(self):
        """Test price manager with cache disabled"""
        no_cache_manager = PriceManager(enable_cache=False)
        assert no_cache_manager.cache is None
        assert no_cache_manager.enable_cache == False

    @patch('core.data_manager.price_manager.YahooFinanceProvider')
    def test_get_current_price_with_mock(self, mock_provider_class):
        """Test current price fetching with mocked provider"""
        # Setup mock
        mock_provider = Mock()
        mock_provider.get_current_price.return_value = 150.0
        mock_provider_class.return_value = mock_provider

        # Create manager with mocked provider
        manager = PriceManager(primary_provider=mock_provider)

        # Test price fetching
        price = manager.get_current_price("AAPL")
        assert price == 150.0

        # Verify mock was called
        mock_provider.get_current_price.assert_called_with("AAPL")

    @patch('core.data_manager.price_manager.YahooFinanceProvider')
    def test_get_current_prices_batch_with_mock(self, mock_provider_class):
        """Test batch price fetching with mocked provider"""
        # Setup mock
        mock_provider = Mock()
        mock_provider.get_current_prices.return_value = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0
        }
        mock_provider_class.return_value = mock_provider

        # Create manager with mocked provider
        manager = PriceManager(primary_provider=mock_provider)

        # Test batch fetching
        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices = manager.get_current_prices(tickers)

        assert len(prices) == 3
        assert prices["AAPL"] == 150.0
        assert prices["MSFT"] == 300.0
        assert prices["GOOGL"] == 2500.0

    def test_cache_functionality(self):
        """Test caching behavior with mocked provider"""
        # Setup mock provider
        mock_provider = Mock()
        mock_provider.get_current_price.return_value = 150.0

        # Create manager with mock
        manager = PriceManager(primary_provider=mock_provider, cache_ttl=10)

        # First call should hit provider
        price1 = manager.get_current_price("AAPL")
        assert price1 == 150.0
        assert mock_provider.get_current_price.call_count == 1

        # Second call should hit cache
        price2 = manager.get_current_price("AAPL")
        assert price2 == 150.0
        assert mock_provider.get_current_price.call_count == 1  # Still 1, not called again

    def test_fallback_provider(self):
        """Test fallback to backup provider"""
        # Setup primary provider that fails
        primary_mock = Mock()
        primary_mock.get_current_price.return_value = None

        # Setup backup provider that succeeds
        backup_mock = Mock()
        backup_mock.get_current_price.return_value = 150.0

        # Create manager and add backup
        manager = PriceManager(primary_provider=primary_mock)
        manager.add_backup_provider(backup_mock)

        # Test fetching - should use backup
        price = manager._fetch_price_with_fallback("AAPL")
        assert price == 150.0

        # Verify both providers were called
        primary_mock.get_current_price.assert_called_with("AAPL")
        backup_mock.get_current_price.assert_called_with("AAPL")

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        manager = PriceManager()

        # Reset rate limits for testing
        manager.request_count = 0
        manager.request_limit = 5  # Low limit for testing
        manager.request_reset_time = time.time() + 3600

        # Make requests up to limit
        for i in range(5):
            assert manager._check_rate_limit() == True

        # Next request should be rate limited
        assert manager._check_rate_limit() == False

    def test_market_status(self):
        """Test market status functionality"""
        status = self.price_manager.get_market_status()

        assert isinstance(status, MarketStatus)
        assert isinstance(status.is_open, bool)

        if status.next_open:
            assert isinstance(status.next_open, datetime)

        if status.next_close:
            assert isinstance(status.next_close, datetime)

    def test_market_indices(self):
        """Test market indices fetching"""
        # Mock the provider
        mock_provider = Mock()
        mock_provider.get_current_prices.return_value = {
            "^GSPC": 4500.0,
            "^IXIC": 15000.0,
            "^DJI": 35000.0
        }

        manager = PriceManager(primary_provider=mock_provider)
        indices = manager.get_market_indices()

        assert isinstance(indices, dict)
        if indices:  # If any data returned
            for name, value in indices.items():
                assert isinstance(name, str)
                assert isinstance(value, (int, float))
                assert value > 0

    def test_ticker_validation(self):
        """Test ticker validation functionality"""
        # Mock provider for validation
        mock_provider = Mock()

        def mock_price_function(ticker):
            valid_tickers = ["AAPL", "MSFT", "GOOGL"]
            return 150.0 if ticker in valid_tickers else None

        mock_provider.get_current_price.side_effect = mock_price_function

        manager = PriceManager(primary_provider=mock_provider)

        tickers = ["AAPL", "MSFT", "INVALID"]
        results = manager.validate_tickers(tickers)

        assert results["AAPL"] == True
        assert results["MSFT"] == True
        assert results["INVALID"] == False

    def test_cache_stats(self):
        """Test cache statistics"""
        stats = self.price_manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert 'cache_enabled' in stats
        assert stats['cache_enabled'] == True

        if stats['cache_enabled']:
            assert 'valid_entries' in stats
            assert 'expired_entries' in stats
            assert 'cache_size_mb' in stats

    def test_warm_cache(self):
        """Test cache warming functionality"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.get_current_prices.return_value = {
            "AAPL": 150.0,
            "MSFT": 300.0
        }

        manager = PriceManager(primary_provider=mock_provider)

        tickers = ["AAPL", "MSFT"]
        manager.warm_cache(tickers)

        # Verify provider was called
        mock_provider.get_current_prices.assert_called()


class TestPortfolioIntegration:
    """Test integration between PriceManager and Portfolio"""

    def setup_method(self):
        """Setup portfolio and price manager"""
        self.assets = [
            Asset(ticker="AAPL", weight=0.5, shares=100),
            Asset(ticker="MSFT", weight=0.5, shares=50)
        ]

        self.portfolio = Portfolio(
            name="Test Portfolio",
            assets=self.assets
        )

        # Mock provider
        self.mock_provider = Mock()
        self.mock_provider.get_current_prices.return_value = {
            "AAPL": 150.0,
            "MSFT": 300.0
        }

        self.price_manager = PriceManager(primary_provider=self.mock_provider)

    def test_update_portfolio_prices(self):
        """Test updating all portfolio asset prices"""
        # Update prices
        updated_portfolio = self.price_manager.update_portfolio_prices(self.portfolio)

        # Check that prices were updated
        aapl_asset = updated_portfolio.get_asset("AAPL")
        msft_asset = updated_portfolio.get_asset("MSFT")

        assert aapl_asset.current_price == 150.0
        assert msft_asset.current_price == 300.0

        # Verify provider was called with correct tickers
        called_tickers = self.mock_provider.get_current_prices.call_args[0][0]
        assert "AAPL" in called_tickers
        assert "MSFT" in called_tickers

    def test_portfolio_performance_calculation(self):
        """Test portfolio performance calculation with historical data"""
        # Mock historical data
        historical_data = pd.DataFrame({
            'AAPL': [140.0, 145.0, 150.0],
            'MSFT': [290.0, 295.0, 300.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        self.mock_provider.get_historical_prices = Mock(return_value=historical_data)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)

        performance = self.price_manager.get_portfolio_performance(
            self.portfolio, start_date, end_date
        )

        assert not performance.empty
        assert 'portfolio_value' in performance.columns
        assert len(performance) == len(historical_data)

    def test_portfolio_returns_calculation(self):
        """Test portfolio returns calculation"""
        # Mock historical data
        historical_data = pd.DataFrame({
            'AAPL': [140.0, 145.0, 150.0],
            'MSFT': [290.0, 295.0, 300.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        self.mock_provider.get_historical_prices = Mock(return_value=historical_data)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)

        returns = self.price_manager.get_portfolio_returns(
            self.portfolio, start_date, end_date
        )

        # Returns should be calculated from portfolio performance
        if not returns.empty:
            assert isinstance(returns, pd.Series)
            # Should have daily returns (one less than performance points)
            assert len(returns) <= len(historical_data) - 1


class TestQuote:
    """Test Quote data structure"""

    def test_quote_creation(self):
        """Test Quote object creation"""
        quote = Quote(
            ticker="AAPL",
            price=150.0,
            change=5.0,
            change_percent=3.45,
            volume=1000000
        )

        assert quote.ticker == "AAPL"
        assert quote.price == 150.0
        assert quote.change == 5.0
        assert quote.change_percent == 3.45
        assert quote.volume == 1000000
        assert isinstance(quote.timestamp, datetime)

    def test_quote_with_optional_fields(self):
        """Test Quote with optional market data"""
        quote = Quote(
            ticker="AAPL",
            price=150.0,
            change=5.0,
            change_percent=3.45,
            volume=1000000,
            market_cap=2_500_000_000_000,  # 2.5T
            pe_ratio=25.5,
            dividend_yield=0.0052
        )

        assert quote.market_cap == 2_500_000_000_000
        assert quote.pe_ratio == 25.5
        assert quote.dividend_yield == 0.0052


class TestMarketStatus:
    """Test MarketStatus functionality"""

    def test_market_status_creation(self):
        """Test MarketStatus object creation"""
        now = datetime.now()
        next_open = now + timedelta(hours=1)

        status = MarketStatus(
            is_open=True,
            next_open=next_open,
            timezone="US/Eastern"
        )

        assert status.is_open == True
        assert status.next_open == next_open
        assert status.timezone == "US/Eastern"
        assert isinstance(status.market_hours, dict)


class TestCompanyInfo:
    """Test CompanyInfo data structure"""

    def test_company_info_creation(self):
        """Test CompanyInfo object creation"""
        info = CompanyInfo(
            ticker="AAPL",
            name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=2_500_000_000_000,
            employees=164_000,
            website="https://www.apple.com",
            country="United States",
            exchange="NASDAQ"
        )

        assert info.ticker == "AAPL"
        assert info.name == "Apple Inc."
        assert info.sector == "Technology"
        assert info.market_cap == 2_500_000_000_000
        assert info.employees == 164_000

    def test_company_info_serialization(self):
        """Test CompanyInfo serialization"""
        info = CompanyInfo(
            ticker="AAPL",
            name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics"
        )

        info_dict = info.to_dict()

        assert isinstance(info_dict, dict)
        assert info_dict['ticker'] == "AAPL"
        assert info_dict['name'] == "Apple Inc."
        assert info_dict['sector'] == "Technology"


# Performance and stress tests
class TestPriceManagerPerformance:
    """Test price manager performance with larger datasets"""

    def test_batch_price_performance(self):
        """Test performance of batch price fetching"""
        # Mock provider that simulates delays
        mock_provider = Mock()

        def mock_batch_fetch(tickers):
            # Simulate some processing time
            time.sleep(0.01 * len(tickers))  # 10ms per ticker
            return {ticker: 100.0 for ticker in tickers}

        mock_provider.get_current_prices.side_effect = mock_batch_fetch

        manager = PriceManager(primary_provider=mock_provider)

        # Test with medium-sized batch
        tickers = [f"STOCK{i:03d}" for i in range(50)]

        start_time = time.time()
        prices = manager.get_current_prices(tickers)
        end_time = time.time()

        # Should complete in reasonable time (< 2 seconds)
        assert (end_time - start_time) < 2.0
        assert len(prices) == len(tickers)

    def test_cache_performance(self):
        """Test cache performance with many entries"""
        cache = PriceCache(ttl_seconds=60)

        # Add many entries
        num_entries = 1000

        start_time = time.time()
        for i in range(num_entries):
            cache.set(f"TICKER{i:04d}", float(i))
        end_time = time.time()

        # Should be fast to add entries
        assert (end_time - start_time) < 1.0

        # Test retrieval performance
        start_time = time.time()
        for i in range(num_entries):
            value = cache.get(f"TICKER{i:04d}")
            assert value == float(i)
        end_time = time.time()

        # Should be fast to retrieve entries
        assert (end_time - start_time) < 1.0


# Error handling tests
class TestErrorHandling:
    """Test error handling in price management"""

    def test_network_error_handling(self):
        """Test handling of network errors"""
        # Mock provider that raises exceptions
        mock_provider = Mock()
        mock_provider.get_current_price.side_effect = Exception("Network error")

        manager = PriceManager(primary_provider=mock_provider)

        # Should not crash, should return None
        price = manager.get_current_price("AAPL")
        assert price is None

    def test_invalid_data_handling(self):
        """Test handling of invalid data from provider"""
        # Mock provider that returns invalid data
        mock_provider = Mock()
        mock_provider.get_current_price.return_value = -100.0  # Invalid negative price

        manager = PriceManager(primary_provider=mock_provider)

        # Manager should handle invalid data gracefully
        price = manager.get_current_price("AAPL")
        # Depending on implementation, might return None or the invalid value
        # The key is that it shouldn't crash

    def test_empty_response_handling(self):
        """Test handling of empty responses"""
        mock_provider = Mock()
        mock_provider.get_current_prices.return_value = {}  # Empty dict

        manager = PriceManager(primary_provider=mock_provider)

        prices = manager.get_current_prices(["AAPL", "MSFT"])
        assert isinstance(prices, dict)
        # Should return empty dict, not crash


# Integration test markers
pytestmark = pytest.mark.unit


# Add integration test marker for tests that need internet
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may need internet)"
    )


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])