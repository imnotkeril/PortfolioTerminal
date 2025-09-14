"""
Core Module - Portfolio Management System.

This is the main core module that provides all business logic and data management
functionality for the portfolio management system.
"""

from typing import Dict, Any, List

# Import data management functionality
from .data_manager import (
    # Core models
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

    # Validation
    ValidationResult,
    ConstraintViolation,
    Suggestion,

    # Exceptions
    PortfolioError,
    ValidationError,
    DataError,
    CalculationError,
    OptimizationError,

    # Managers
    PortfolioManager,
    PriceManager,

    # Data providers
    DataProvider,
    YahooFinanceProvider,

    # Market data
    Quote,
    MarketStatus,
    CompanyInfo,

    # Validators
    TickerValidator,
    PortfolioValidator,

    # Convenience functions
    create_portfolio,
    load_portfolio,
    get_current_prices
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Wild Market Capital Development Team"
__description__ = "Professional Portfolio Management System"

# Phase completion status
PHASE_STATUS = {
    1: "✅ Complete - Data Foundation",
    2: "🚧 Planned - Analytics Engine",
    3: "🚧 Planned - Risk Engine",
    4: "🚧 Planned - Optimization Engine",
    5: "🚧 Planned - Scenario Engine",
    6: "🚧 Planned - Reporting Engine"
}

def get_system_info() -> dict:
    """Get system information and status"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'phase_status': PHASE_STATUS,
        'current_phase': 1,
        'modules_available': [
            'data_manager'
        ],
        'modules_planned': [
            'analytics_engine',
            'risk_engine',
            'optimization_engine',
            'scenario_engine',
            'reporting_engine'
        ]
    }

# Export main functionality
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

    # Managers
    'PortfolioManager',
    'PriceManager',

    # Data providers
    'DataProvider',
    'YahooFinanceProvider',

    # Market data
    'Quote',
    'MarketStatus',
    'CompanyInfo',

    # Validators
    'TickerValidator',
    'PortfolioValidator',

    # Convenience functions
    'create_portfolio',
    'load_portfolio',
    'get_current_prices',

    # System info
    'get_system_info',
    'PHASE_STATUS'
]