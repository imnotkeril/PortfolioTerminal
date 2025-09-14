"""
Price Manager - Market data management with caching and multiple providers.

This module handles fetching real-time and historical market data from various
providers, with intelligent caching and error handling.
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Generator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from .models import Portfolio, Asset

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Real-time quote data"""
    ticker: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketStatus:
    """Market status information"""
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    timezone: str = "US/Eastern"
    market_hours: Dict[str, str] = None

    def __post_init__(self):
        if self.market_hours is None:
            self.market_hours = {
                "pre_market": "04:00-09:30",
                "regular": "09:30-16:00",
                "after_hours": "16:00-20:00"
            }


@dataclass
class CompanyInfo:
    """Company fundamental information"""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    website: Optional[str] = None
    description: Optional[str] = None
    country: Optional[str] = None
    currency: str = "USD"
    exchange: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'name': self.name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'employees': self.employees,
            'website': self.website,
            'description': self.description,
            'country': self.country,
            'currency': self.currency,
            'exchange': self.exchange
        }


@dataclass
class PriceUpdate:
    """Real-time price update"""
    ticker: str
    price: float
    timestamp: datetime
    volume: int = 0


class DataProvider(ABC):
    """Abstract base class for market data providers"""

    @abstractmethod
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for single ticker"""
        pass

    @abstractmethod
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for multiple tickers"""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data"""
        pass

    @abstractmethod
    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Get company information"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

    def get_rate_limit_delay(self) -> float:
        """Get recommended delay between requests"""
        return 0.1  # Default 100ms


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider implementation"""

    def __init__(self):
        self.name = "Yahoo Finance"
        self.base_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        self._info_cache = {}  # Cache company info
        self._cache_ttl = 3600  # 1 hour TTL for company info

    def _respect_rate_limit(self):
        """Ensure we respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.base_delay:
            time.sleep(self.base_delay - time_since_last)

        self.last_request_time = time.time()

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for single ticker"""
        try:
            self._respect_rate_limit()

            stock = yf.Ticker(ticker)
            info = stock.info

            # Try different price fields
            price = (info.get('currentPrice') or
                    info.get('regularMarketPrice') or
                    info.get('previousClose'))

            if price and price > 0:
                return float(price)

            return None

        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return None

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for multiple tickers"""
        prices = {}

        # Use yfinance download for batch requests
        try:
            # Split into chunks to avoid overwhelming the API
            chunk_size = 50
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i + chunk_size]

                self._respect_rate_limit()

                # Download latest data
                data = yf.download(
                    chunk,
                    period="1d",
                    interval="1m",
                    progress=False,
                    show_errors=False
                )

                if not data.empty:
                    if len(chunk) == 1:
                        # Single ticker
                        if 'Close' in data.columns:
                            latest_price = data['Close'].iloc[-1]
                            if not pd.isna(latest_price):
                                prices[chunk[0]] = float(latest_price)
                    else:
                        # Multiple tickers
                        if 'Close' in data.columns:
                            for ticker in chunk:
                                try:
                                    if ticker in data['Close'].columns:
                                        latest_price = data['Close'][ticker].iloc[-1]
                                        if not pd.isna(latest_price):
                                            prices[ticker] = float(latest_price)
                                except Exception as e:
                                    logger.debug(f"Error getting price for {ticker}: {e}")
                                    continue

                # Add delay between chunks
                if i + chunk_size < len(tickers):
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in batch price fetch: {e}")

            # Fallback to individual requests
            for ticker in tickers:
                price = self.get_current_price(ticker)
                if price:
                    prices[ticker] = price

        return prices

    def get_historical_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data"""
        try:
            self._respect_rate_limit()

            # Convert datetime to string format
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')

            # Download data
            data = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval=interval,
                progress=False,
                show_errors=False
            )

            if data.empty:
                logger.warning(f"No historical data found for {ticker}")
                return pd.DataFrame()

            # Clean data
            data = data.dropna()

            # Ensure we have the expected columns
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in expected_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns for {ticker}: {missing_columns}")

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Get company information with caching"""
        # Check cache first
        cache_key = f"{ticker}_info"
        current_time = time.time()

        if cache_key in self._info_cache:
            cached_data, cache_time = self._info_cache[cache_key]
            if current_time - cache_time < self._cache_ttl:
                return cached_data

        try:
            self._respect_rate_limit()

            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or len(info) < 5:  # Basic check for valid data
                return None

            company_info = CompanyInfo(
                ticker=ticker,
                name=info.get('longName', ticker),
                sector=info.get('sector', ''),
                industry=info.get('industry', ''),
                market_cap=info.get('marketCap'),
                employees=info.get('fullTimeEmployees'),
                website=info.get('website'),
                description=info.get('longBusinessSummary'),
                country=info.get('country'),
                currency=info.get('currency', 'USD'),
                exchange=info.get('exchange')
            )

            # Cache the result
            self._info_cache[cache_key] = (company_info, current_time)

            return company_info

        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            # Try to fetch a known ticker
            test_price = self.get_current_price("AAPL")
            return test_price is not None
        except Exception:
            return False


class PriceCache:
    """Simple in-memory cache for price data"""

    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default TTL
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # Expired, remove from cache
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp"""
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()

    def clear_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = sum(
            1 for _, timestamp in self.cache.values()
            if current_time - timestamp < self.ttl
        )

        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'cache_size_mb': len(str(self.cache)) / (1024 * 1024)
        }


class PriceManager:
    """
    Main price manager with caching and multiple provider support
    """

    def __init__(
        self,
        primary_provider: Optional[DataProvider] = None,
        cache_ttl: int = 300,  # 5 minutes
        enable_cache: bool = True
    ):
        """
        Initialize price manager

        Args:
            primary_provider: Primary data provider
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable caching
        """
        self.primary_provider = primary_provider or YahooFinanceProvider()
        self.backup_providers: List[DataProvider] = []

        # Cache setup
        self.enable_cache = enable_cache
        self.cache = PriceCache(cache_ttl) if enable_cache else None

        # Rate limiting
        self.request_count = 0
        self.request_limit = 1000  # Per hour
        self.request_reset_time = time.time() + 3600

        logger.info(f"PriceManager initialized with provider: {self.primary_provider.name}")

    def add_backup_provider(self, provider: DataProvider) -> None:
        """Add backup data provider"""
        self.backup_providers.append(provider)
        logger.info(f"Added backup provider: {provider.name}")

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for single ticker with caching

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price or None if unavailable
        """
        ticker = ticker.upper().strip()

        # Check cache first
        if self.cache:
            cache_key = f"price_{ticker}"
            cached_price = self.cache.get(cache_key)
            if cached_price is not None:
                logger.debug(f"Cache hit for {ticker}: ${cached_price}")
                return cached_price

        # Check rate limits
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, using cached data only")
            return None

        # Try primary provider
        price = self._fetch_price_with_fallback(ticker)

        # Cache the result
        if price and self.cache:
            cache_key = f"price_{ticker}"
            self.cache.set(cache_key, price)

        return price

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary of ticker -> price
        """
        tickers = [t.upper().strip() for t in tickers]
        prices = {}
        uncached_tickers = []

        # Check cache for each ticker
        if self.cache:
            for ticker in tickers:
                cache_key = f"price_{ticker}"
                cached_price = self.cache.get(cache_key)
                if cached_price is not None:
                    prices[ticker] = cached_price
                else:
                    uncached_tickers.append(ticker)
        else:
            uncached_tickers = tickers

        # Fetch uncached prices
        if uncached_tickers:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, returning cached data only")
                return prices

            # Use batch fetch from primary provider
            try:
                batch_prices = self.primary_provider.get_current_prices(uncached_tickers)
                prices.update(batch_prices)

                # Cache the new prices
                if self.cache:
                    for ticker, price in batch_prices.items():
                        cache_key = f"price_{ticker}"
                        self.cache.set(cache_key, price)

            except Exception as e:
                logger.error(f"Error in batch price fetch: {e}")

                # Fallback to individual requests
                for ticker in uncached_tickers:
                    price = self._fetch_price_with_fallback(ticker)
                    if price:
                        prices[ticker] = price

        return prices

    def get_quote(self, ticker: str) -> Optional[Quote]:
        """
        Get detailed quote information

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote object with detailed information
        """
        ticker = ticker.upper().strip()

        try:
            if not self._check_rate_limit():
                return None

            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            # Get current and previous prices
            current_price = (info.get('currentPrice') or
                           info.get('regularMarketPrice') or
                           info.get('previousClose'))

            previous_close = info.get('previousClose')

            if not current_price:
                return None

            # Calculate change
            change = 0.0
            change_percent = 0.0
            if previous_close and previous_close > 0:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100

            quote = Quote(
                ticker=ticker,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=info.get('volume', 0),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                timestamp=datetime.now()
            )

            return quote

        except Exception as e:
            logger.error(f"Error fetching quote for {ticker}: {e}")
            return None

    def get_historical_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data"""
        return self.primary_provider.get_historical_data(ticker, start, end, interval)

    def get_historical_prices(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical prices for multiple tickers

        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date
            frequency: Data frequency ('1d', '1wk', '1mo')

        Returns:
            DataFrame with tickers as columns and dates as index
        """
        logger.info(f"Fetching historical data for {len(tickers)} tickers from {start} to {end}")

        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded")
                return pd.DataFrame()

            # Use yfinance download for multiple tickers
            data = yf.download(
                tickers,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval=frequency,
                progress=False,
                show_errors=False,
                threads=True
            )

            if data.empty:
                logger.warning("No historical data returned")
                return pd.DataFrame()

            # Handle single vs multiple tickers
            if len(tickers) == 1:
                if 'Close' in data.columns:
                    result = pd.DataFrame({tickers[0]: data['Close']})
                else:
                    result = pd.DataFrame()
            else:
                # Multiple tickers - extract Close prices
                if ('Close' in data.columns.get_level_values(0) if
                    hasattr(data.columns, 'get_level_values') else 'Close' in data.columns):

                    if hasattr(data.columns, 'get_level_values'):
                        # MultiIndex columns
                        result = data['Close']
                    else:
                        # Single level columns
                        result = data[['Close']].copy()
                        result.columns = [tickers[0]]
                else:
                    result = pd.DataFrame()

            # Clean data
            result = result.dropna()

            logger.info(f"Retrieved {len(result)} data points for {len(result.columns)} tickers")
            return result

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_returns(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Get returns data for tickers

        Args:
            tickers: List of ticker symbols
            period: Time period ('1y', '6mo', '3mo', etc.)

        Returns:
            DataFrame with daily returns
        """
        # Convert period to start date
        if period == "1y":
            start = datetime.now() - timedelta(days=365)
        elif period == "6mo":
            start = datetime.now() - timedelta(days=180)
        elif period == "3mo":
            start = datetime.now() - timedelta(days=90)
        elif period == "1mo":
            start = datetime.now() - timedelta(days=30)
        else:
            start = datetime.now() - timedelta(days=365)  # Default to 1 year

        end = datetime.now()

        # Get historical prices
        prices = self.get_historical_prices(tickers, start, end)

        if prices.empty:
            return pd.DataFrame()

        # Calculate returns
        returns = prices.pct_change().dropna()

        return returns

    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Get company information"""
        return self.primary_provider.get_company_info(ticker)

    def update_portfolio_prices(self, portfolio: Portfolio) -> Portfolio:
        """
        Update all asset prices in portfolio

        Args:
            portfolio: Portfolio to update

        Returns:
            Updated portfolio
        """
        logger.info(f"Updating prices for portfolio: {portfolio.name}")

        if not portfolio.assets:
            logger.warning("No assets in portfolio to update")
            return portfolio

        tickers = [asset.ticker for asset in portfolio.assets]
        current_prices = self.get_current_prices(tickers)

        # Update asset prices
        updated_count = 0
        for asset in portfolio.assets:
            if asset.ticker in current_prices:
                asset.update_price(current_prices[asset.ticker])
                updated_count += 1

        logger.info(f"Updated prices for {updated_count}/{len(portfolio.assets)} assets")
        return portfolio

    def get_market_status(self) -> MarketStatus:
        """Get current market status"""
        # Simple implementation - check if it's a weekday and within trading hours
        now = datetime.now()

        # Check if it's a weekday (Monday = 0, Sunday = 6)
        is_weekday = now.weekday() < 5

        # Check trading hours (9:30 AM - 4:00 PM EST)
        trading_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        trading_end = now.replace(hour=16, minute=0, second=0, microsecond=0)

        is_trading_hours = trading_start <= now <= trading_end
        is_open = is_weekday and is_trading_hours

        # Calculate next open/close
        next_open = None
        next_close = None

        if is_open:
            next_close = trading_end
        else:
            # Next trading day
            days_ahead = 1
            if now.weekday() >= 5:  # Weekend
                days_ahead = 7 - now.weekday()  # Days until Monday

            next_open = (now + timedelta(days=days_ahead)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )

        return MarketStatus(
            is_open=is_open,
            next_open=next_open,
            next_close=next_close
        )

    # ================================
    # PORTFOLIO-SPECIFIC METHODS
    # ================================

    def get_portfolio_performance(
        self,
        portfolio: Portfolio,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get portfolio performance over time period

        Args:
            portfolio: Portfolio object
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with portfolio value over time
        """
        if not portfolio.assets:
            return pd.DataFrame()

        # Get historical data for all assets
        tickers = [asset.ticker for asset in portfolio.assets]
        historical_prices = self.get_historical_prices(tickers, start_date, end_date)

        if historical_prices.empty:
            return pd.DataFrame()

        # Calculate portfolio value over time
        portfolio_values = []

        for date, row in historical_prices.iterrows():
            daily_value = 0.0

            for asset in portfolio.assets:
                if asset.ticker in row and not pd.isna(row[asset.ticker]):
                    # Calculate asset value based on weight and total portfolio value
                    asset_value = portfolio.initial_value * asset.weight * (
                        row[asset.ticker] / historical_prices[asset.ticker].iloc[0]
                    )
                    daily_value += asset_value

            portfolio_values.append({
                'date': date,
                'portfolio_value': daily_value
            })

        result_df = pd.DataFrame(portfolio_values)
        result_df.set_index('date', inplace=True)

        return result_df

    def get_portfolio_returns(
        self,
        portfolio: Portfolio,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Get portfolio returns over time period

        Args:
            portfolio: Portfolio object
            start_date: Start date
            end_date: End date

        Returns:
            Series with daily returns
        """
        performance_df = self.get_portfolio_performance(portfolio, start_date, end_date)

        if performance_df.empty:
            return pd.Series()

        # Calculate daily returns
        returns = performance_df['portfolio_value'].pct_change().dropna()

        return returns

    # ================================
    # PRIVATE METHODS
    # ================================

    def _fetch_price_with_fallback(self, ticker: str) -> Optional[float]:
        """Fetch price with fallback to backup providers"""
        # Try primary provider
        price = self.primary_provider.get_current_price(ticker)
        if price is not None:
            return price

        # Try backup providers
        for provider in self.backup_providers:
            try:
                price = provider.get_current_price(ticker)
                if price is not None:
                    logger.info(f"Used backup provider for {ticker}: {provider.name}")
                    return price
            except Exception as e:
                logger.debug(f"Backup provider {provider.name} failed for {ticker}: {e}")
                continue

        logger.warning(f"Could not fetch price for {ticker} from any provider")
        return None

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()

        # Reset counter if hour has passed
        if current_time > self.request_reset_time:
            self.request_count = 0
            self.request_reset_time = current_time + 3600

        # Check limit
        if self.request_count >= self.request_limit:
            return False

        self.request_count += 1
        return True

    # ================================
    # CACHE MANAGEMENT
    # ================================

    def clear_cache(self) -> None:
        """Clear price cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Price cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            stats = self.cache.get_stats()
            stats['cache_enabled'] = True
            return stats
        else:
            return {'cache_enabled': False}

    def warm_cache(self, tickers: List[str]) -> None:
        """
        Pre-populate cache with ticker prices

        Args:
            tickers: List of tickers to cache
        """
        logger.info(f"Warming cache for {len(tickers)} tickers")

        # Fetch in chunks to avoid overwhelming API
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            self.get_current_prices(chunk)

            # Add delay between chunks
            if i + chunk_size < len(tickers):
                time.sleep(1)

        logger.info("Cache warming completed")

    # ================================
    # UTILITY METHODS
    # ================================

    def validate_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Validate list of ticker symbols by attempting to fetch prices

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary of ticker -> valid (boolean)
        """
        logger.info(f"Validating {len(tickers)} tickers")

        validation_results = {}

        # Use ThreadPoolExecutor for parallel validation
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks
            future_to_ticker = {
                executor.submit(self.get_current_price, ticker): ticker
                for ticker in tickers
            }

            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    price = future.result(timeout=10)  # 10 second timeout
                    validation_results[ticker] = price is not None
                except Exception as e:
                    logger.debug(f"Validation failed for {ticker}: {e}")
                    validation_results[ticker] = False

        valid_count = sum(validation_results.values())
        logger.info(f"Validation complete: {valid_count}/{len(tickers)} tickers are valid")

        return validation_results

    def get_market_indices(self) -> Dict[str, float]:
        """Get current values for major market indices"""
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX'
        }

        current_values = self.get_current_prices(list(indices.keys()))

        # Map to friendly names
        result = {}
        for symbol, name in indices.items():
            if symbol in current_values:
                result[name] = current_values[symbol]

        return result

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        status = self.get_market_status()
        return status.is_open

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all data providers"""
        status = {
            'primary': self.primary_provider.is_available()
        }

        for i, provider in enumerate(self.backup_providers):
            status[f'backup_{i+1}'] = provider.is_available()

        return status