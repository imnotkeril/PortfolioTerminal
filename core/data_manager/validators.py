"""
Data validation module for portfolio management system.

This module provides comprehensive validation for portfolios, assets,
and other data structures to ensure data integrity and business rule compliance.
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

from .models import (
    Portfolio, Asset, PortfolioSettings, ValidationResult,
    ConstraintViolation, Suggestion, AssetClass, PortfolioType
)


class TickerValidator:
    """Validate stock ticker symbols"""

    # Common ticker patterns for different exchanges
    TICKER_PATTERNS = {
        'US': r'^[A-Z]{1,5}$',  # 1-5 uppercase letters
        'US_CLASS': r'^[A-Z]{1,4}\.[A-Z]$',  # Class shares (e.g., BRK.A)
        'NASDAQ': r'^[A-Z]{1,5}$',
        'NYSE': r'^[A-Z]{1,4}$',
        'OTC': r'^[A-Z]{4,5}$',
        'ETF': r'^[A-Z]{2,4}$',
        'CRYPTO': r'^[A-Z]{3,4}-USD$',  # e.g., BTC-USD
        'FOREX': r'^[A-Z]{3}[A-Z]{3}=X$',  # e.g., EURUSD=X
        'INDEX': r'^\^[A-Z]{2,5}$',  # e.g., ^GSPC
        'INTERNATIONAL': r'^[A-Z0-9]{1,6}\.[A-Z]{1,3}$'  # e.g., 7203.T (Toyota)
    }

    # Known invalid patterns
    INVALID_PATTERNS = [
        r'.*[^A-Z0-9\.\-\^=].*',  # Contains invalid characters
        r'^[0-9]+$',  # Numbers only
        r'^.*\s+.*$',  # Contains spaces
    ]

    # Common ticker symbols for validation
    KNOWN_VALID_TICKERS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'NFLX', 'DIS', 'PYPL',
        'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'BND', 'GLD', 'TLT', 'VEA'
    }

    @classmethod
    def validate_ticker(cls, ticker: str) -> bool:
        """Validate if ticker format is correct"""
        if not ticker:
            return False

        ticker = ticker.upper().strip()

        # Check against invalid patterns
        for pattern in cls.INVALID_PATTERNS:
            if re.match(pattern, ticker):
                return False

        # Check against valid patterns
        for pattern in cls.TICKER_PATTERNS.values():
            if re.match(pattern, ticker):
                return True

        return False

    @classmethod
    def suggest_corrections(cls, ticker: str) -> List[str]:
        """Suggest corrections for invalid tickers"""
        suggestions = []

        if not ticker:
            return suggestions

        original = ticker.upper().strip()

        # Remove common formatting issues
        cleaned = re.sub(r'[^A-Z0-9\.\-\^=]', '', original)
        if cleaned != original and cls.validate_ticker(cleaned):
            suggestions.append(cleaned)

        # Check against known tickers (fuzzy matching)
        for known_ticker in cls.KNOWN_VALID_TICKERS:
            if cls._calculate_similarity(original, known_ticker) > 0.8:
                suggestions.append(known_ticker)

        return suggestions[:3]  # Return top 3 suggestions

    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()


class WeightValidator:
    """Validate portfolio weights and allocations"""

    @staticmethod
    def validate_weights(weights: List[float], tolerance: float = 0.001) -> ValidationResult:
        """Validate that weights are valid and sum to 1.0"""
        result = ValidationResult(is_valid=True)

        # Check individual weights
        for i, weight in enumerate(weights):
            if weight < 0:
                result.add_error(f"Weight {i+1} is negative: {weight}")
            elif weight > 1:
                result.add_error(f"Weight {i+1} exceeds 100%: {weight}")

        # Check total weight
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > tolerance:
            if total_weight == 0:
                result.add_error("All weights are zero")
            else:
                result.add_warning(
                    f"Weights sum to {total_weight:.4f}, not 1.0 "
                    f"(difference: {total_weight - 1.0:+.4f})"
                )

        return result

    @staticmethod
    def normalize_weights(weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.0"""
        total = sum(weights)
        if total == 0:
            # Equal weighting if all zeros
            return [1.0 / len(weights)] * len(weights)

        return [w / total for w in weights]

    @staticmethod
    def validate_concentration(weights: Dict[str, float], max_concentration: float = 0.3) -> ValidationResult:
        """Check for over-concentration in single positions"""
        result = ValidationResult(is_valid=True)

        for ticker, weight in weights.items():
            if weight > max_concentration:
                result.add_warning(
                    f"{ticker} weight ({weight:.1%}) exceeds concentration limit ({max_concentration:.1%})"
                )

        return result


class PortfolioValidator:
    """Comprehensive portfolio validation"""

    def __init__(self):
        self.ticker_validator = TickerValidator()
        self.weight_validator = WeightValidator()

    def validate_portfolio(self, portfolio: Portfolio) -> ValidationResult:
        """Comprehensive portfolio validation"""
        result = ValidationResult(is_valid=True)

        # Basic information validation
        if not portfolio.name or len(portfolio.name.strip()) == 0:
            result.add_error("Portfolio name is required")

        if len(portfolio.name) > 100:
            result.add_error("Portfolio name too long (max 100 characters)")

        # Asset validation
        if len(portfolio.assets) == 0:
            result.add_error("Portfolio must contain at least one asset")

        # Check for duplicate tickers
        tickers = [asset.ticker for asset in portfolio.assets]
        duplicates = set([t for t in tickers if tickers.count(t) > 1])
        if duplicates:
            result.add_error(f"Duplicate tickers found: {', '.join(duplicates)}")

        # Validate individual assets
        for i, asset in enumerate(portfolio.assets):
            asset_result = self.validate_asset(asset)
            if not asset_result.is_valid:
                for error in asset_result.errors:
                    result.add_error(f"Asset {i+1} ({asset.ticker}): {error}")

        # Weight validation
        weights = [asset.weight for asset in portfolio.assets]
        weight_result = self.weight_validator.validate_weights(weights)

        for error in weight_result.errors:
            result.add_error(f"Weight validation: {error}")
        for warning in weight_result.warnings:
            result.add_warning(f"Weight validation: {warning}")

        # Concentration validation
        weights_dict = {asset.ticker: asset.weight for asset in portfolio.assets}
        concentration_result = self.weight_validator.validate_concentration(weights_dict)

        for warning in concentration_result.warnings:
            result.add_warning(f"Concentration check: {warning}")

        # Settings validation
        settings_result = self.validate_settings(portfolio.settings)
        for error in settings_result.errors:
            result.add_error(f"Settings: {error}")
        for warning in settings_result.warnings:
            result.add_warning(f"Settings: {warning}")

        return result

    def validate_asset(self, asset: Asset) -> ValidationResult:
        """Validate individual asset"""
        result = ValidationResult(is_valid=True)

        # Ticker validation
        if not self.ticker_validator.validate_ticker(asset.ticker):
            result.add_error(f"Invalid ticker format: {asset.ticker}")
            suggestions = self.ticker_validator.suggest_corrections(asset.ticker)
            if suggestions:
                result.add_warning(f"Did you mean: {', '.join(suggestions)}?")

        # Weight validation
        if asset.weight < 0:
            result.add_error(f"Negative weight: {asset.weight}")
        elif asset.weight > 1:
            result.add_error(f"Weight exceeds 100%: {asset.weight}")

        # Shares validation
        if asset.shares < 0:
            result.add_error(f"Negative shares: {asset.shares}")

        # Price validation
        if asset.purchase_price is not None and asset.purchase_price <= 0:
            result.add_error(f"Invalid purchase price: {asset.purchase_price}")

        if asset.current_price is not None and asset.current_price <= 0:
            result.add_error(f"Invalid current price: {asset.current_price}")

        # Date validation
        if asset.purchase_date and asset.purchase_date > datetime.now():
            result.add_error("Purchase date cannot be in the future")

        return result

    def validate_settings(self, settings: PortfolioSettings) -> ValidationResult:
        """Validate portfolio settings"""
        result = ValidationResult(is_valid=True)

        # Rebalancing settings
        valid_frequencies = ['monthly', 'quarterly', 'annual', 'manual']
        if settings.rebalancing_frequency not in valid_frequencies:
            result.add_error(f"Invalid rebalancing frequency: {settings.rebalancing_frequency}")

        if not 0 < settings.rebalancing_threshold <= 1:
            result.add_error(f"Rebalancing threshold must be between 0 and 1: {settings.rebalancing_threshold}")

        # Position constraints
        if settings.min_position_size < 0:
            result.add_error(f"Minimum position size cannot be negative: {settings.min_position_size}")

        if settings.max_position_size <= settings.min_position_size:
            result.add_error("Maximum position size must be greater than minimum")

        if settings.max_positions <= 0:
            result.add_error(f"Maximum positions must be positive: {settings.max_positions}")

        # Risk management
        if settings.max_drawdown <= 0 or settings.max_drawdown > 1:
            result.add_error(f"Max drawdown must be between 0 and 1: {settings.max_drawdown}")

        if settings.stop_loss is not None and (settings.stop_loss <= 0 or settings.stop_loss > 1):
            result.add_error(f"Stop loss must be between 0 and 1: {settings.stop_loss}")

        if settings.take_profit is not None and settings.take_profit <= 0:
            result.add_error(f"Take profit must be positive: {settings.take_profit}")

        # Tax settings
        if not 0 <= settings.tax_rate <= 1:
            result.add_error(f"Tax rate must be between 0 and 1: {settings.tax_rate}")

        # Leverage settings
        if settings.max_leverage <= 0:
            result.add_error(f"Max leverage must be positive: {settings.max_leverage}")

        if settings.max_leverage > 3.0:
            result.add_warning(f"High leverage detected: {settings.max_leverage}x")

        return result

    def check_constraints(self, portfolio: Portfolio) -> List[ConstraintViolation]:
        """Check portfolio against defined constraints"""
        violations = []
        settings = portfolio.settings

        # Position size constraints
        for asset in portfolio.assets:
            market_value = asset.market_value

            if market_value > 0:  # Only check if we have market value
                if market_value < settings.min_position_size:
                    violations.append(ConstraintViolation(
                        constraint_type="position_size",
                        description=f"{asset.ticker} position (${market_value:,.0f}) below minimum (${settings.min_position_size:,.0f})",
                        severity="warning",
                        suggested_fix=f"Increase position size or remove asset"
                    ))

                if market_value > settings.max_position_size:
                    violations.append(ConstraintViolation(
                        constraint_type="position_size",
                        description=f"{asset.ticker} position (${market_value:,.0f}) exceeds maximum (${settings.max_position_size:,.0f})",
                        severity="error",
                        suggested_fix=f"Reduce position size to ${settings.max_position_size:,.0f}"
                    ))

        # Number of positions constraint
        if len(portfolio.assets) > settings.max_positions:
            violations.append(ConstraintViolation(
                constraint_type="position_count",
                description=f"Portfolio has {len(portfolio.assets)} positions, exceeds limit of {settings.max_positions}",
                severity="error",
                suggested_fix="Remove some positions or increase limit"
            ))

        # Sector constraints
        sector_allocation = portfolio.get_sector_allocation()
        for sector, (min_weight, max_weight) in settings.sector_limits.items():
            current_weight = sector_allocation.get(sector, 0.0)

            if current_weight < min_weight:
                violations.append(ConstraintViolation(
                    constraint_type="sector_allocation",
                    description=f"{sector} allocation ({current_weight:.1%}) below minimum ({min_weight:.1%})",
                    severity="warning",
                    suggested_fix=f"Increase {sector} allocation"
                ))

            if current_weight > max_weight:
                violations.append(ConstraintViolation(
                    constraint_type="sector_allocation",
                    description=f"{sector} allocation ({current_weight:.1%}) exceeds maximum ({max_weight:.1%})",
                    severity="error",
                    suggested_fix=f"Reduce {sector} allocation to {max_weight:.1%}"
                ))

        # ESG constraints
        if settings.esg_settings.enabled:
            esg_violations = self._check_esg_constraints(portfolio, settings.esg_settings)
            violations.extend(esg_violations)

        # Geographic constraints
        geo_violations = self._check_geographic_constraints(portfolio, settings.geographic_constraints)
        violations.extend(geo_violations)

        return violations

    def _check_esg_constraints(self, portfolio: Portfolio, esg_settings) -> List[ConstraintViolation]:
        """Check ESG constraints"""
        violations = []

        for asset in portfolio.assets:
            # Check exclusions
            if esg_settings.exclude_tobacco and asset.sector.lower() == 'tobacco':
                violations.append(ConstraintViolation(
                    constraint_type="esg_exclusion",
                    description=f"{asset.ticker} is in excluded tobacco sector",
                    severity="error",
                    suggested_fix="Remove tobacco assets"
                ))

            if esg_settings.exclude_weapons and asset.sector.lower() in ['defense', 'weapons']:
                violations.append(ConstraintViolation(
                    constraint_type="esg_exclusion",
                    description=f"{asset.ticker} is in excluded weapons/defense sector",
                    severity="error",
                    suggested_fix="Remove weapons/defense assets"
                ))

            if esg_settings.exclude_fossil_fuels and asset.sector.lower() in ['energy', 'oil', 'gas']:
                violations.append(ConstraintViolation(
                    constraint_type="esg_exclusion",
                    description=f"{asset.ticker} is in excluded fossil fuels sector",
                    severity="error",
                    suggested_fix="Remove fossil fuel assets"
                ))

            # Check minimum ESG score
            if (esg_settings.min_esg_score is not None and
                asset.esg_score is not None and
                asset.esg_score < esg_settings.min_esg_score):
                violations.append(ConstraintViolation(
                    constraint_type="esg_score",
                    description=f"{asset.ticker} ESG score ({asset.esg_score}) below minimum ({esg_settings.min_esg_score})",
                    severity="warning",
                    suggested_fix="Replace with higher ESG rated assets"
                ))

        return violations

    def _check_geographic_constraints(self, portfolio: Portfolio, geo_constraints) -> List[ConstraintViolation]:
        """Check geographic constraints"""
        violations = []

        for asset in portfolio.assets:
            # This would need market data to determine asset geography
            # For now, we'll use exchange information as a proxy
            pass  # Implementation would require additional market data

        return violations

    def suggest_improvements(self, portfolio: Portfolio) -> List[Suggestion]:
        """Generate improvement suggestions for portfolio"""
        suggestions = []

        # Diversification suggestions
        sector_allocation = portfolio.get_sector_allocation()
        max_sector_weight = max(sector_allocation.values()) if sector_allocation else 0

        if max_sector_weight > 0.4:
            suggestions.append(Suggestion(
                suggestion_type="diversification",
                description=f"High sector concentration detected ({max_sector_weight:.1%})",
                impact="medium",
                implementation="Add assets from underrepresented sectors"
            ))

        # Position size suggestions
        weights = [asset.weight for asset in portfolio.assets]
        if weights:
            max_weight = max(weights)
            if max_weight > 0.25:
                suggestions.append(Suggestion(
                    suggestion_type="concentration",
                    description=f"Large single position detected ({max_weight:.1%})",
                    impact="medium",
                    implementation="Consider reducing position size"
                ))

        # Number of positions
        if len(portfolio.assets) < 10:
            suggestions.append(Suggestion(
                suggestion_type="diversification",
                description=f"Portfolio has only {len(portfolio.assets)} positions",
                impact="low",
                implementation="Consider adding more positions for better diversification"
            ))
        elif len(portfolio.assets) > 50:
            suggestions.append(Suggestion(
                suggestion_type="complexity",
                description=f"Portfolio has {len(portfolio.assets)} positions",
                impact="low",
                implementation="Consider consolidating similar positions"
            ))

        # Asset class diversification
        asset_classes = portfolio.get_asset_class_allocation()
        if len(asset_classes) == 1:
            suggestions.append(Suggestion(
                suggestion_type="diversification",
                description="Portfolio is concentrated in single asset class",
                impact="high",
                implementation="Consider adding bonds, REITs, or other asset classes"
            ))

        return suggestions


class DataValidator:
    """General data validation utilities"""

    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> ValidationResult:
        """Validate date range"""
        result = ValidationResult(is_valid=True)

        if start_date >= end_date:
            result.add_error("Start date must be before end date")

        if end_date > datetime.now():
            result.add_warning("End date is in the future")

        if (end_date - start_date).days < 30:
            result.add_warning("Date range is less than 30 days")

        return result

    @staticmethod
    def validate_percentage(value: float, field_name: str) -> ValidationResult:
        """Validate percentage values"""
        result = ValidationResult(is_valid=True)

        if value < 0:
            result.add_error(f"{field_name} cannot be negative: {value}")

        if value > 1:
            result.add_error(f"{field_name} cannot exceed 100%: {value}")

        return result

    @staticmethod
    def validate_positive_number(value: float, field_name: str) -> ValidationResult:
        """Validate positive number"""
        result = ValidationResult(is_valid=True)

        if value <= 0:
            result.add_error(f"{field_name} must be positive: {value}")

        return result

    @staticmethod
    def clean_ticker(ticker: str) -> str:
        """Clean and standardize ticker symbol"""
        if not ticker:
            return ""

        # Remove whitespace and convert to uppercase
        cleaned = ticker.strip().upper()

        # Remove common prefixes/suffixes that might be added incorrectly
        # (This would be expanded based on actual data patterns)

        return cleaned


class ImportValidator:
    """Validate imported data from various sources"""

    @staticmethod
    def validate_csv_data(df: pd.DataFrame) -> ValidationResult:
        """Validate CSV import data"""
        result = ValidationResult(is_valid=True)

        # Check required columns
        required_columns = ['ticker']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            result.add_error(f"Missing required columns: {', '.join(missing_columns)}")

        # Check for empty data
        if df.empty:
            result.add_error("No data found in file")

        # Check for duplicate tickers
        if 'ticker' in df.columns:
            duplicates = df['ticker'].duplicated()
            if duplicates.any():
                duplicate_tickers = df.loc[duplicates, 'ticker'].unique()
                result.add_error(f"Duplicate tickers in file: {', '.join(duplicate_tickers)}")

        # Validate weights if present
        if 'weight' in df.columns:
            weight_issues = df['weight'].apply(lambda x: x < 0 or x > 1 if pd.notnull(x) else False)
            if weight_issues.any():
                result.add_error("Some weights are outside valid range (0-1)")

        return result

    @staticmethod
    def validate_text_input(text: str) -> ValidationResult:
        """Validate text input for portfolio creation"""
        result = ValidationResult(is_valid=True)

        if not text or len(text.strip()) == 0:
            result.add_error("No input text provided")
            return result

        # Try to parse the text to check format
        try:
            parsed_data = TextParser.parse_text_input(text)
            if not parsed_data:
                result.add_error("Could not parse any valid ticker-weight pairs")
        except Exception as e:
            result.add_error(f"Error parsing text input: {str(e)}")

        return result


class TextParser:
    """Parse text input for portfolio creation"""

    # Regex patterns for different formats
    PATTERNS = {
        'ticker_percent': r'([A-Z]{1,5})\s+(\d+(?:\.\d+)?)%',  # AAPL 30%
        'ticker_decimal': r'([A-Z]{1,5})\s+(\d+(?:\.\d+)?)',   # AAPL 0.30
        'ticker_colon': r'([A-Z]{1,5}):(\d+(?:\.\d+)?)',       # AAPL:30
        'ticker_only': r'([A-Z]{1,5})',                        # AAPL (for equal weight)
    }

    @classmethod
    def parse_text_input(cls, text: str) -> List[Dict[str, Any]]:
        """Parse text input and return list of ticker-weight pairs"""
        if not text:
            return []

        text = text.upper().strip()
        results = []

        # Try different patterns
        for pattern_name, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text)

            if matches:
                if pattern_name == 'ticker_only':
                    # Equal weighting
                    tickers = [match for match in matches if TickerValidator.validate_ticker(match)]
                    if tickers:
                        equal_weight = 1.0 / len(tickers)
                        results = [{'ticker': ticker, 'weight': equal_weight} for ticker in tickers]
                        break
                else:
                    # Has weights
                    for ticker, weight_str in matches:
                        if TickerValidator.validate_ticker(ticker):
                            weight = float(weight_str)

                            # Convert percentage to decimal if needed
                            if pattern_name == 'ticker_percent':
                                weight = weight / 100.0
                            elif pattern_name in ['ticker_decimal', 'ticker_colon'] and weight > 1:
                                # Assume it's a percentage if > 1
                                weight = weight / 100.0

                            results.append({'ticker': ticker, 'weight': weight})

                    if results:
                        break

        return results


def validate_all_inputs(
    name: str,
    assets_data: List[Dict[str, Any]],
    settings: Optional[PortfolioSettings] = None
) -> ValidationResult:
    """Validate all inputs for portfolio creation"""
    result = ValidationResult(is_valid=True)

    # Validate name
    if not name or len(name.strip()) == 0:
        result.add_error("Portfolio name is required")
    elif len(name) > 100:
        result.add_error("Portfolio name too long (max 100 characters)")

    # Validate assets data
    if not assets_data:
        result.add_error("At least one asset is required")
        return result

    # Check for required fields
    for i, asset_data in enumerate(assets_data):
        if 'ticker' not in asset_data:
            result.add_error(f"Asset {i+1}: ticker is required")
        elif not TickerValidator.validate_ticker(asset_data['ticker']):
            result.add_error(f"Asset {i+1}: invalid ticker format '{asset_data['ticker']}'")

        if 'weight' not in asset_data:
            result.add_error(f"Asset {i+1}: weight is required")
        elif not isinstance(asset_data['weight'], (int, float)):
            result.add_error(f"Asset {i+1}: weight must be numeric")
        elif asset_data['weight'] < 0 or asset_data['weight'] > 1:
            result.add_error(f"Asset {i+1}: weight must be between 0 and 1")

    # Validate total weights
    if result.is_valid:
        total_weight = sum(asset_data.get('weight', 0) for asset_data in assets_data)
        if abs(total_weight - 1.0) > 0.001:
            result.add_warning(
                f"Weights sum to {total_weight:.3f}, not 1.0. "
                "They will be automatically normalized."
            )

    # Validate settings if provided
    if settings:
        validator = PortfolioValidator()
        settings_result = validator.validate_settings(settings)

        for error in settings_result.errors:
            result.add_error(f"Settings: {error}")
        for warning in settings_result.warnings:
            result.add_warning(f"Settings: {warning}")

    return result


# Utility functions for common validation tasks
def is_valid_weight(weight: float) -> bool:
    """Check if weight is valid (0-1)"""
    return 0 <= weight <= 1


def is_valid_ticker(ticker: str) -> bool:
    """Check if ticker format is valid"""
    return TickerValidator.validate_ticker(ticker)


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol"""
    return DataValidator.clean_ticker(ticker)


def validate_portfolio_quick(portfolio: Portfolio) -> bool:
    """Quick validation check for portfolio"""
    return portfolio.validate()


def get_validation_summary(validation_result: ValidationResult) -> str:
    """Get human-readable validation summary"""
    if validation_result.is_valid and not validation_result.warnings:
        return "✅ All validations passed"

    summary = []

    if validation_result.errors:
        summary.append(f"❌ {len(validation_result.errors)} error(s):")
        for error in validation_result.errors:
            summary.append(f"  • {error}")

    if validation_result.warnings:
        summary.append(f"⚠️ {len(validation_result.warnings)} warning(s):")
        for warning in validation_result.warnings:
            summary.append(f"  • {warning}")

    return "\n".join(summary)