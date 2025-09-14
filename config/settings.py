"""
Application Settings and Configuration.

This module contains all configuration settings for the portfolio management system.
Settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings:
    """Application settings with environment variable support."""

    # ================================
    # APPLICATION SETTINGS
    # ================================

    # Application metadata
    APP_NAME = "Wild Market Capital Portfolio Manager"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Professional Portfolio Management and Analytics Platform"

    # Debug and development
    DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # development, staging, production

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = os.getenv("LOG_FILE", "logs/portfolio_manager.log")

    # ================================
    # DATA STORAGE SETTINGS
    # ================================

    # Portfolio storage
    PORTFOLIO_STORAGE_PATH = os.getenv(
        "PORTFOLIO_STORAGE_PATH",
        str(PROJECT_ROOT / "data" / "portfolios")
    )

    # Export directory
    EXPORT_PATH = os.getenv("EXPORT_PATH", str(PROJECT_ROOT / "exports"))

    # Cache directory
    CACHE_PATH = os.getenv("CACHE_PATH", str(PROJECT_ROOT / "cache"))

    # Backup settings
    BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "true").lower() in ("true", "1", "yes")
    BACKUP_PATH = os.getenv("BACKUP_PATH", str(PROJECT_ROOT / "backups"))
    BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))

    # ================================
    # PRICE DATA SETTINGS
    # ================================

    # Data providers
    YAHOO_FINANCE_ENABLED = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() in ("true", "1", "yes")
    ALPHA_VANTAGE_ENABLED = os.getenv("ALPHA_VANTAGE_ENABLED", "false").lower() in ("true", "1", "yes")
    POLYGON_ENABLED = os.getenv("POLYGON_ENABLED", "false").lower() in ("true", "1", "yes")
    IEX_CLOUD_ENABLED = os.getenv("IEX_CLOUD_ENABLED", "false").lower() in ("true", "1", "yes")

    # API keys (keep empty if not used)
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    IEX_CLOUD_API_KEY = os.getenv("IEX_CLOUD_API_KEY", "")

    # Cache settings
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes default
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))  # Max cached items

    # Rate limiting
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() in ("true", "1", "yes")
    REQUESTS_PER_HOUR = int(os.getenv("REQUESTS_PER_HOUR", "1000"))
    REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "100"))

    # Request timeouts
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
    BATCH_REQUEST_DELAY = float(os.getenv("BATCH_REQUEST_DELAY", "0.1"))  # 100ms between requests

    # ================================
    # STREAMLIT SETTINGS
    # ================================

    # Server settings
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    STREAMLIT_HEADLESS = os.getenv("STREAMLIT_HEADLESS", "true").lower() in ("true", "1", "yes")

    # Theme settings (based on TradingView design)
    THEME_BASE = os.getenv("THEME_BASE", "dark")
    THEME_PRIMARY_COLOR = os.getenv("THEME_PRIMARY_COLOR", "#BF9FFB")
    THEME_BACKGROUND_COLOR = os.getenv("THEME_BACKGROUND_COLOR", "#0D1015")
    THEME_SECONDARY_BACKGROUND_COLOR = os.getenv("THEME_SECONDARY_BACKGROUND_COLOR", "#2A2E39")
    THEME_TEXT_COLOR = os.getenv("THEME_TEXT_COLOR", "#FFFFFF")

    # UI settings
    PAGE_TITLE = os.getenv("PAGE_TITLE", "WMC Portfolio Manager")
    PAGE_ICON = os.getenv("PAGE_ICON", "📊")
    LAYOUT = os.getenv("LAYOUT", "wide")

    # ================================
    # BUSINESS LOGIC SETTINGS
    # ================================

    # Default portfolio settings
    DEFAULT_INITIAL_VALUE = float(os.getenv("DEFAULT_INITIAL_VALUE", "100000.0"))
    DEFAULT_MIN_POSITION_SIZE = float(os.getenv("DEFAULT_MIN_POSITION_SIZE", "1000.0"))
    DEFAULT_MAX_POSITION_SIZE = float(os.getenv("DEFAULT_MAX_POSITION_SIZE", "50000.0"))
    DEFAULT_MAX_POSITIONS = int(os.getenv("DEFAULT_MAX_POSITIONS", "50"))

    # Risk management
    DEFAULT_MAX_DRAWDOWN = float(os.getenv("DEFAULT_MAX_DRAWDOWN", "0.20"))  # 20%
    DEFAULT_STOP_LOSS = float(os.getenv("DEFAULT_STOP_LOSS", "0.10"))  # 10%
    DEFAULT_TAX_RATE = float(os.getenv("DEFAULT_TAX_RATE", "0.25"))  # 25%

    # Validation settings
    WEIGHT_TOLERANCE = float(os.getenv("WEIGHT_TOLERANCE", "0.001"))  # 0.1% tolerance
    MAX_PORTFOLIO_NAME_LENGTH = int(os.getenv("MAX_PORTFOLIO_NAME_LENGTH", "100"))
    MAX_DESCRIPTION_LENGTH = int(os.getenv("MAX_DESCRIPTION_LENGTH", "1000"))

    # Concentration limits
    DEFAULT_MAX_SINGLE_POSITION = float(os.getenv("DEFAULT_MAX_SINGLE_POSITION", "0.30"))  # 30%
    DEFAULT_MAX_SECTOR_ALLOCATION = float(os.getenv("DEFAULT_MAX_SECTOR_ALLOCATION", "0.40"))  # 40%

    # ================================
    # PERFORMANCE SETTINGS
    # ================================

    # Threading and concurrency
    MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", "10"))
    ENABLE_ASYNC_OPERATIONS = os.getenv("ENABLE_ASYNC_OPERATIONS", "true").lower() in ("true", "1", "yes")

    # Memory management
    MAX_MEMORY_USAGE_MB = int(os.getenv("MAX_MEMORY_USAGE_MB", "1000"))  # 1GB
    ENABLE_MEMORY_MONITORING = os.getenv("ENABLE_MEMORY_MONITORING", "false").lower() in ("true", "1", "yes")

    # ================================
    # SECURITY SETTINGS
    # ================================

    # Data encryption (for future phases)
    ENCRYPT_PORTFOLIO_DATA = os.getenv("ENCRYPT_PORTFOLIO_DATA", "false").lower() in ("true", "1", "yes")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

    # Session management (for future API)
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))

    # ================================
    # ANALYTICS SETTINGS (Future Phases)
    # ================================

    # Performance calculation
    TRADING_DAYS_PER_YEAR = int(os.getenv("TRADING_DAYS_PER_YEAR", "252"))
    RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.03"))  # 3% default

    # Benchmarks
    DEFAULT_BENCHMARK = os.getenv("DEFAULT_BENCHMARK", "^GSPC")  # S&P 500
    AVAILABLE_BENCHMARKS = [
        "^GSPC",  # S&P 500
        "^IXIC",  # NASDAQ
        "^DJI",  # Dow Jones
        "^RUT"  # Russell 2000
    ]

    # ================================
    # FEATURE FLAGS (Future Phases)
    # ================================

    # Phase enablement
    ANALYTICS_ENGINE_ENABLED = os.getenv("ANALYTICS_ENGINE_ENABLED", "false").lower() in ("true", "1", "yes")
    RISK_ENGINE_ENABLED = os.getenv("RISK_ENGINE_ENABLED", "false").lower() in ("true", "1", "yes")
    OPTIMIZATION_ENGINE_ENABLED = os.getenv("OPTIMIZATION_ENGINE_ENABLED", "false").lower() in ("true", "1", "yes")
    SCENARIO_ENGINE_ENABLED = os.getenv("SCENARIO_ENGINE_ENABLED", "false").lower() in ("true", "1", "yes")
    REPORTING_ENGINE_ENABLED = os.getenv("REPORTING_ENGINE_ENABLED", "false").lower() in ("true", "1", "yes")

    # API features (Phase 7)
    API_ENABLED = os.getenv("API_ENABLED", "false").lower() in ("true", "1", "yes")
    API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "true").lower() in ("true", "1", "yes")

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        settings = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                settings[attr_name] = getattr(cls, attr_name)
        return settings

    @classmethod
    def validate_settings(cls) -> List[str]:
        """Validate current settings and return any issues."""
        issues = []

        # Check required directories exist
        required_dirs = [
            cls.PORTFOLIO_STORAGE_PATH,
            cls.EXPORT_PATH,
            cls.CACHE_PATH
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")

        # Check API keys if providers are enabled
        if cls.ALPHA_VANTAGE_ENABLED and not cls.ALPHA_VANTAGE_API_KEY:
            issues.append("Alpha Vantage enabled but no API key provided")

        if cls.POLYGON_ENABLED and not cls.POLYGON_API_KEY:
            issues.append("Polygon enabled but no API key provided")

        if cls.IEX_CLOUD_ENABLED and not cls.IEX_CLOUD_API_KEY:
            issues.append("IEX Cloud enabled but no API key provided")

        # Validate numeric settings
        if cls.CACHE_TTL_SECONDS < 30:
            issues.append("Cache TTL too low (minimum 30 seconds recommended)")

        if cls.REQUESTS_PER_HOUR < 100:
            issues.append("Rate limit too restrictive (minimum 100 requests/hour recommended)")

        # Validate percentages
        percentage_settings = [
            ("DEFAULT_MAX_DRAWDOWN", cls.DEFAULT_MAX_DRAWDOWN),
            ("DEFAULT_STOP_LOSS", cls.DEFAULT_STOP_LOSS),
            ("DEFAULT_TAX_RATE", cls.DEFAULT_TAX_RATE),
            ("DEFAULT_MAX_SINGLE_POSITION", cls.DEFAULT_MAX_SINGLE_POSITION),
            ("DEFAULT_MAX_SECTOR_ALLOCATION", cls.DEFAULT_MAX_SECTOR_ALLOCATION)
        ]

        for name, value in percentage_settings:
            if not (0.0 <= value <= 1.0):
                issues.append(f"{name} should be between 0.0 and 1.0 (got {value})")

        return issues

    @classmethod
    def create_env_template(cls) -> str:
        """Create a template .env file content."""
        template = """# Wild Market Capital Portfolio Manager Configuration
# Copy this file to .env and customize as needed

# =================================
# APPLICATION SETTINGS
# =================================

# Environment: development, staging, production
ENVIRONMENT=production

# Debug mode (true/false)
DEBUG=false

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# =================================
# DATA STORAGE
# =================================

# Portfolio storage directory
PORTFOLIO_STORAGE_PATH=data/portfolios

# Export directory
EXPORT_PATH=exports

# Cache directory
CACHE_PATH=cache

# =================================
# PRICE DATA PROVIDERS
# =================================

# Yahoo Finance (free, enabled by default)
YAHOO_FINANCE_ENABLED=true

# Alpha Vantage (requires API key)
ALPHA_VANTAGE_ENABLED=false
ALPHA_VANTAGE_API_KEY=

# Polygon.io (requires API key)
POLYGON_ENABLED=false
POLYGON_API_KEY=

# IEX Cloud (requires API key)
IEX_CLOUD_ENABLED=false
IEX_CLOUD_API_KEY=

# =================================
# CACHING SETTINGS
# =================================

# Enable price caching
CACHE_ENABLED=true

# Cache time-to-live in seconds (300 = 5 minutes)
CACHE_TTL_SECONDS=300

# Maximum number of cached items
CACHE_MAX_SIZE=1000

# =================================
# RATE LIMITING
# =================================

# Enable rate limiting
ENABLE_RATE_LIMITING=true

# Requests per hour limit
REQUESTS_PER_HOUR=1000

# Requests per minute limit
REQUESTS_PER_MINUTE=100

# =================================
# STREAMLIT UI SETTINGS
# =================================

# Server port
STREAMLIT_PORT=8501

# Server host (0.0.0.0 for all interfaces)
STREAMLIT_HOST=0.0.0.0

# Run headless (no browser popup)
STREAMLIT_HEADLESS=true

# Theme colors (TradingView-inspired)
THEME_PRIMARY_COLOR=#BF9FFB
THEME_BACKGROUND_COLOR=#0D1015
THEME_SECONDARY_BACKGROUND_COLOR=#2A2E39
THEME_TEXT_COLOR=#FFFFFF

# =================================
# BUSINESS SETTINGS
# =================================

# Default portfolio settings
DEFAULT_INITIAL_VALUE=100000.0
DEFAULT_MIN_POSITION_SIZE=1000.0
DEFAULT_MAX_POSITION_SIZE=50000.0
DEFAULT_MAX_POSITIONS=50

# Risk management defaults
DEFAULT_MAX_DRAWDOWN=0.20
DEFAULT_STOP_LOSS=0.10
DEFAULT_TAX_RATE=0.25

# =================================
# PERFORMANCE SETTINGS
# =================================

# Maximum worker threads
MAX_WORKER_THREADS=10

# Enable async operations
ENABLE_ASYNC_OPERATIONS=true

# Maximum memory usage in MB
MAX_MEMORY_USAGE_MB=1000
"""
        return template


# Create global settings instance
settings = Settings()

# Validate settings on import
validation_issues = settings.validate_settings()
if validation_issues:
    import warnings

    for issue in validation_issues:
        warnings.warn(f"Settings issue: {issue}")


# Helper functions
def get_streamlit_config() -> Dict[str, Any]:
    """Get Streamlit configuration dictionary."""
    return {
        "server.port": settings.STREAMLIT_PORT,
        "server.address": settings.STREAMLIT_HOST,
        "server.headless": settings.STREAMLIT_HEADLESS,
        "browser.gatherUsageStats": False,
        "theme.base": settings.THEME_BASE,
        "theme.primaryColor": settings.THEME_PRIMARY_COLOR,
        "theme.backgroundColor": settings.THEME_BACKGROUND_COLOR,
        "theme.secondaryBackgroundColor": settings.THEME_SECONDARY_BACKGROUND_COLOR,
        "theme.textColor": settings.THEME_TEXT_COLOR
    }


def get_data_provider_config() -> Dict[str, Any]:
    """Get data provider configuration."""
    return {
        "yahoo_finance": {
            "enabled": settings.YAHOO_FINANCE_ENABLED,
            "api_key": None  # Yahoo Finance doesn't require API key
        },
        "alpha_vantage": {
            "enabled": settings.ALPHA_VANTAGE_ENABLED,
            "api_key": settings.ALPHA_VANTAGE_API_KEY
        },
        "polygon": {
            "enabled": settings.POLYGON_ENABLED,
            "api_key": settings.POLYGON_API_KEY
        },
        "iex_cloud": {
            "enabled": settings.IEX_CLOUD_ENABLED,
            "api_key": settings.IEX_CLOUD_API_KEY
        }
    }


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.ENVIRONMENT.lower() == "development" or settings.DEBUG


def is_production() -> bool:
    """Check if running in production mode."""
    return settings.ENVIRONMENT.lower() == "production" and not settings.DEBUG