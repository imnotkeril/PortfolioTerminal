"""
Data Manager Module - Portfolio and asset data management.

This module provides the core data structures and management functionality
for the portfolio management system.
"""

from typing import List, Dict, Any

from .models import (
    # Core data models
    Portfolio,
    Asset,
    PortfolioSettings,
    ESGSettings,
    GeographicConstraints,
    PortfolioStats,
    Trade,

    # Enums
    AssetClass,
    PortfolioType,
    RiskLevel,

    # Validation models
    ValidationResult,
    ConstraintViolation,
    Suggestion,

    # Exceptions
    PortfolioError,
    ValidationError,
    DataError,
    CalculationError,
    OptimizationError
)

from .validators import (
    TickerValidator,
    WeightValidator,
    PortfolioValidator,
    DataValidator,
    ImportValidator,
    TextParser,
    validate_all_inputs,
    is_valid_weight,
    is_valid_ticker,
    normalize_ticker,
    validate_portfolio_quick,
    get_validation_summary
)

from .portfolio_manager import PortfolioManager

from .price_manager import (
    PriceManager,
    DataProvider,
    YahooFinanceProvider,
    PriceCache,
    Quote,
    MarketStatus,
    CompanyInfo,
    PriceUpdate
)

# Version info
__version__ = "1.0.0"
__author__ = "WMC Development Team"

# Module-level convenience functions
def create_portfolio(
    name: str,
    assets_data: List[Dict[str, Any]],
    storage_path: str = "data/portfolios"
) -> Portfolio:
    """
    Convenience function to create portfolio

    Args:
        name: Portfolio name
        assets_data: List of asset data dictionaries
        storage_path: Storage directory

    Returns:
        Created Portfolio object
    """
    manager = PortfolioManager(storage_path)

    # Convert asset data to Asset objects
    assets = []
    for data in assets_data:
        asset = Asset(
            ticker=data['ticker'],
            weight=data.get('weight', 0.0),
            name=data.get('name', ''),
            shares=data.get('shares', 0.0),
            purchase_price=data.get('purchase_price'),
            sector=data.get('sector', ''),
            asset_class=AssetClass(data.get('asset_class', 'stock'))
        )
        assets.append(asset)

    return manager.create_portfolio(name=name, assets=assets)


def load_portfolio(portfolio_id: str, storage_path: str = "data/portfolios") -> Portfolio:
    """
    Convenience function to load portfolio

    Args:
        portfolio_id: Portfolio ID
        storage_path: Storage directory

    Returns:
        Portfolio object
    """
    manager = PortfolioManager(storage_path)
    return manager.get_portfolio(portfolio_id)


def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Convenience function to get current prices

    Args:
        tickers: List of ticker symbols

    Returns:
        Dictionary of ticker -> price
    """
    price_manager = PriceManager()
    return price_manager.get_current_prices(tickers)


# Export main classes for easy import
__all__ = [
    # Core models
    'Portfolio',
    'Asset',
    'PortfolioSettings',
    'ESGSettings',
    'GeographicConstraints',
    'PortfolioStats',
    'Trade',

    # Enums
    'AssetClass',
    'PortfolioType',
    'RiskLevel',

    # Validation
    'ValidationResult',
    'ConstraintViolation',
    'Suggestion',

    # Exceptions
    'PortfolioError',
    'ValidationError',
    'DataError',
    'CalculationError',
    'OptimizationError',

    # Validators
    'TickerValidator',
    'WeightValidator',
    'PortfolioValidator',
    'DataValidator',
    'ImportValidator',
    'TextParser',

    # Managers
    'PortfolioManager',
    'PriceManager',

    # Data providers
    'DataProvider',
    'YahooFinanceProvider',
    'PriceCache',

    # Market data
    'Quote',
    'MarketStatus',
    'CompanyInfo',
    'PriceUpdate',

    # Convenience functions
    'create_portfolio',
    'load_portfolio',
    'get_current_prices',
    'validate_all_inputs',
    'is_valid_weight',
    'is_valid_ticker',
    'normalize_ticker',
    'validate_portfolio_quick',
    'get_validation_summary'
]