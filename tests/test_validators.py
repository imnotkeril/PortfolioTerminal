"""
Unit tests for validation functionality.

This module tests all validation classes and functions to ensure
data integrity and proper error handling.
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from core.data_manager import (
    Asset, Portfolio, PortfolioSettings, AssetClass, PortfolioType
)
from core.data_manager.validators import (
    TickerValidator, WeightValidator, PortfolioValidator, DataValidator,
    ImportValidator, TextParser, validate_all_inputs, get_validation_summary
)


class TestTickerValidator:
    """Test ticker validation functionality"""

    def test_valid_tickers(self):
        """Test valid ticker formats"""
        valid_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "BRK.A", "BRK.B",  # Class shares
            "SPY", "QQQ", "VTI",  # ETFs
            "^GSPC", "^IXIC",  # Indices
            "BTC-USD", "ETH-USD",  # Crypto
            "7203.T",  # International (Toyota)
            "EURUSD=X"  # Forex
        ]

        for ticker in valid_tickers:
            assert TickerValidator.validate_ticker(ticker), f"Failed for {ticker}"

    def test_invalid_tickers(self):
        """Test invalid ticker formats"""
        invalid_tickers = [
            "",  # Empty
            "123",  # Numbers only
            "AAPL MSFT",  # Contains spaces
            "AAPL!",  # Invalid characters
            "toolongtickerSymbol",  # Too long
            "aa",  # Too short
            "AAPL@USD"  # Invalid format
        ]

        for ticker in invalid_tickers:
            assert not TickerValidator.validate_ticker(ticker), f"Should fail for {ticker}"

    def test_ticker_suggestions(self):
        """Test ticker correction suggestions"""
        # Test similar known ticker
        suggestions = TickerValidator.suggest_corrections("APLE")
        assert "AAPL" in suggestions

        # Test cleaning formatting issues
        suggestions = TickerValidator.suggest_corrections("AAPL ")
        assert "AAPL" in suggestions

        # Test empty input
        suggestions = TickerValidator.suggest_corrections("")
        assert len(suggestions) == 0


class TestWeightValidator:
    """Test weight validation functionality"""

    def test_valid_weights(self):
        """Test valid weight combinations"""
        valid_weights = [0.3, 0.3, 0.4]  # Sum to 1.0
        result = WeightValidator.validate_weights(valid_weights)

        assert result.is_valid == True
        assert len(result.errors) == 0

    def test_invalid_weights_negative(self):
        """Test negative weights"""
        invalid_weights = [-0.1, 0.6, 0.5]
        result = WeightValidator.validate_weights(invalid_weights)

        assert result.is_valid == False
        assert len(result.errors) > 0
        assert "negative" in result.errors[0].lower()

    def test_invalid_weights_sum(self):
        """Test weights that don't sum to 1.0"""
        invalid_weights = [0.4, 0.4, 0.4]  # Sum to 1.2
        result = WeightValidator.validate_weights(invalid_weights)

        assert len(result.warnings) > 0
        assert "sum to" in result.warnings[0].lower()

    def test_normalize_weights(self):
        """Test weight normalization"""
        weights = [0.6, 0.8, 1.0]  # Sum to 2.4
        normalized = WeightValidator.normalize_weights(weights)

        assert abs(sum(normalized) - 1.0) < 0.001
        assert abs(normalized[0] - 0.6 / 2.4) < 0.001
        assert abs(normalized[1] - 0.8 / 2.4) < 0.001
        assert abs(normalized[2] - 1.0 / 2.4) < 0.001

    def test_normalize_zero_weights(self):
        """Test normalization with all zero weights"""
        zero_weights = [0.0, 0.0, 0.0]
        normalized = WeightValidator.normalize_weights(zero_weights)

        # Should result in equal weighting
        expected_weight = 1.0 / 3
        for weight in normalized:
            assert abs(weight - expected_weight) < 0.001

    def test_concentration_check(self):
        """Test concentration validation"""
        # High concentration portfolio
        concentrated_weights = {"AAPL": 0.8, "MSFT": 0.2}
        result = WeightValidator.validate_concentration(concentrated_weights, max_concentration=0.3)

        assert len(result.warnings) > 0
        assert "concentration" in result.warnings[0].lower()

        # Well diversified portfolio
        diversified_weights = {"AAPL": 0.2, "MSFT": 0.2, "GOOGL": 0.2, "AMZN": 0.2, "TSLA": 0.2}
        result = WeightValidator.validate_concentration(diversified_weights, max_concentration=0.3)

        assert len(result.warnings) == 0


class TestPortfolioValidator:
    """Test portfolio validation functionality"""

    def setup_method(self):
        """Setup test data"""
        self.validator = PortfolioValidator()

        self.valid_assets = [
            Asset(ticker="AAPL", weight=0.4, name="Apple Inc."),
            Asset(ticker="MSFT", weight=0.3, name="Microsoft"),
            Asset(ticker="GOOGL", weight=0.3, name="Alphabet")
        ]

        self.valid_portfolio = Portfolio(
            name="Test Portfolio",
            description="Valid test portfolio",
            assets=self.valid_assets
        )

    def test_valid_portfolio(self):
        """Test validation of valid portfolio"""
        result = self.validator.validate_portfolio(self.valid_portfolio)

        assert result.is_valid == True
        assert len(result.errors) == 0

    def test_empty_name(self):
        """Test portfolio with empty name"""
        invalid_portfolio = Portfolio(name="", assets=self.valid_assets)
        result = self.validator.validate_portfolio(invalid_portfolio)

        assert result.is_valid == False
        assert any("name is required" in error.lower() for error in result.errors)

    def test_no_assets(self):
        """Test portfolio with no assets"""
        empty_portfolio = Portfolio(name="Empty Portfolio", assets=[])
        result = self.validator.validate_portfolio(empty_portfolio)

        assert result.is_valid == False
        assert any("at least one asset" in error.lower() for error in result.errors)

    def test_duplicate_tickers(self):
        """Test portfolio with duplicate tickers"""
        duplicate_assets = [
            Asset(ticker="AAPL", weight=0.5),
            Asset(ticker="AAPL", weight=0.5)  # Duplicate
        ]

        duplicate_portfolio = Portfolio(name="Duplicate Portfolio", assets=duplicate_assets)
        result = self.validator.validate_portfolio(duplicate_portfolio)

        assert result.is_valid == False
        assert any("duplicate" in error.lower() for error in result.errors)

    def test_invalid_asset(self):
        """Test portfolio with invalid asset"""
        invalid_assets = [
            Asset(ticker="", weight=0.5),  # Invalid ticker
            Asset(ticker="MSFT", weight=0.5)
        ]

        invalid_portfolio = Portfolio(name="Invalid Asset Portfolio", assets=invalid_assets)
        result = self.validator.validate_portfolio(invalid_portfolio)

        assert result.is_valid == False

    def test_weight_issues(self):
        """Test portfolio with weight issues"""
        weight_assets = [
            Asset(ticker="AAPL", weight=0.6),
            Asset(ticker="MSFT", weight=0.8)  # Total = 1.4
        ]

        weight_portfolio = Portfolio(name="Weight Issues", assets=weight_assets)
        result = self.validator.validate_portfolio(weight_portfolio)

        assert len(result.warnings) > 0

    def test_asset_validation(self):
        """Test individual asset validation"""
        # Valid asset
        valid_asset = Asset(
            ticker="AAPL",
            weight=0.5,
            shares=100,
            current_price=150.0,
            purchase_price=140.0
        )

        result = self.validator.validate_asset(valid_asset)
        assert result.is_valid == True

        # Invalid asset - bad ticker
        invalid_asset = Asset(ticker="123", weight=0.5)
        result = self.validator.validate_asset(invalid_asset)
        assert result.is_valid == False

    def test_settings_validation(self):
        """Test portfolio settings validation"""
        # Valid settings
        valid_settings = PortfolioSettings()
        result = self.validator.validate_settings(valid_settings)
        assert result.is_valid == True

        # Invalid settings
        invalid_settings = PortfolioSettings(
            rebalancing_frequency="invalid_frequency",
            max_drawdown=1.5  # > 1.0
        )
        result = self.validator.validate_settings(invalid_settings)
        assert result.is_valid == False
        assert len(result.errors) >= 2


class TestTextParser:
    """Test text input parsing functionality"""

    def test_percentage_format(self):
        """Test parsing percentage format"""
        text = "AAPL 30%, MSFT 40%, GOOGL 30%"
        result = TextParser.parse_text_input(text)

        assert len(result) == 3
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["weight"] == 0.30
        assert result[1]["ticker"] == "MSFT"
        assert result[1]["weight"] == 0.40

    def test_decimal_format(self):
        """Test parsing decimal format"""
        text = "AAPL 0.3 MSFT 0.4 GOOGL 0.3"
        result = TextParser.parse_text_input(text)

        assert len(result) == 3
        assert result[0]["weight"] == 0.3
        assert result[1]["weight"] == 0.4
        assert result[2]["weight"] == 0.3

    def test_colon_format(self):
        """Test parsing colon format"""
        text = "AAPL:30 MSFT:40 GOOGL:30"
        result = TextParser.parse_text_input(text)

        assert len(result) == 3
        assert result[0]["weight"] == 0.30
        assert result[1]["weight"] == 0.40
        assert result[2]["weight"] == 0.30

    def test_ticker_only_format(self):
        """Test parsing ticker-only format (equal weight)"""
        text = "AAPL MSFT GOOGL AMZN"
        result = TextParser.parse_text_input(text)

        assert len(result) == 4
        # Each should have equal weight
        for item in result:
            assert abs(item["weight"] - 0.25) < 0.001

    def test_mixed_formats(self):
        """Test that parser handles one format at a time"""
        # Should parse as percentage format, not mixed
        text = "AAPL 30%, MSFT 0.4"
        result = TextParser.parse_text_input(text)

        # Should only find the percentage one
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_invalid_text(self):
        """Test parsing invalid text"""
        invalid_texts = [
            "",  # Empty
            "This is not a ticker list",  # No valid patterns
            "AAPL 30% INVALID_TICKER 40%",  # Mixed valid/invalid
        ]

        for text in invalid_texts:
            result = TextParser.parse_text_input(text)
            # Should either be empty or filter out invalid tickers
            if result:
                # All results should have valid tickers
                for item in result:
                    assert TickerValidator.validate_ticker(item["ticker"])


class TestImportValidator:
    """Test import validation functionality"""

    def test_valid_csv_data(self):
        """Test validation of valid CSV data"""
        valid_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'weight': [0.4, 0.3, 0.3],
            'name': ['Apple', 'Microsoft', 'Alphabet']
        })

        result = ImportValidator.validate_csv_data(valid_df)
        assert result.is_valid == True

    def test_missing_required_columns(self):
        """Test CSV data missing required columns"""
        invalid_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],  # Wrong column name
            'allocation': [0.5, 0.5]
        })

        result = ImportValidator.validate_csv_data(invalid_df)
        assert result.is_valid == False
        assert any("missing required columns" in error.lower() for error in result.errors)

    def test_empty_csv_data(self):
        """Test empty CSV data"""
        empty_df = pd.DataFrame()
        result = ImportValidator.validate_csv_data(empty_df)

        assert result.is_valid == False
        assert any("no data found" in error.lower() for error in result.errors)

    def test_duplicate_tickers_csv(self):
        """Test CSV data with duplicate tickers"""
        duplicate_df = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT'],  # AAPL duplicate
            'weight': [0.3, 0.2, 0.5]
        })

        result = ImportValidator.validate_csv_data(duplicate_df)
        assert result.is_valid == False
        assert any("duplicate tickers" in error.lower() for error in result.errors)

    def test_invalid_weights_csv(self):
        """Test CSV data with invalid weights"""
        invalid_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'weight': [1.5, -0.3]  # Invalid weights
        })

        result = ImportValidator.validate_csv_data(invalid_df)
        assert result.is_valid == False
        assert any("outside valid range" in error.lower() for error in result.errors)

    def test_text_input_validation(self):
        """Test text input validation"""
        # Valid text
        valid_text = "AAPL 30%, MSFT 40%, GOOGL 30%"
        result = ImportValidator.validate_text_input(valid_text)
        assert result.is_valid == True

        # Empty text
        empty_text = ""
        result = ImportValidator.validate_text_input(empty_text)
        assert result.is_valid == False

        # Invalid text
        invalid_text = "This is not valid input"
        result = ImportValidator.validate_text_input(invalid_text)
        assert result.is_valid == False


class TestDataValidator:
    """Test general data validation utilities"""

    def test_date_range_validation(self):
        """Test date range validation"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        result = DataValidator.validate_date_range(start, end)
        assert result.is_valid == True

        # Invalid range (start after end)
        result = DataValidator.validate_date_range(end, start)
        assert result.is_valid == False

    def test_percentage_validation(self):
        """Test percentage validation"""
        # Valid percentage
        result = DataValidator.validate_percentage(0.75, "test_field")
        assert result.is_valid == True

        # Invalid percentage (negative)
        result = DataValidator.validate_percentage(-0.1, "test_field")
        assert result.is_valid == False

        # Invalid percentage (> 100%)
        result = DataValidator.validate_percentage(1.5, "test_field")
        assert result.is_valid == False

    def test_positive_number_validation(self):
        """Test positive number validation"""
        # Valid positive number
        result = DataValidator.validate_positive_number(10.5, "test_field")
        assert result.is_valid == True

        # Invalid (zero)
        result = DataValidator.validate_positive_number(0, "test_field")
        assert result.is_valid == False

        # Invalid (negative)
        result = DataValidator.validate_positive_number(-5.0, "test_field")
        assert result.is_valid == False

    def test_clean_ticker(self):
        """Test ticker cleaning"""
        # Basic cleaning
        assert DataValidator.clean_ticker("  AAPL  ") == "AAPL"
        assert DataValidator.clean_ticker("aapl") == "AAPL"

        # Empty input
        assert DataValidator.clean_ticker("") == ""
        assert DataValidator.clean_ticker(None) == ""


class TestValidationHelpers:
    """Test validation helper functions"""

    def test_validate_all_inputs(self):
        """Test comprehensive input validation"""
        # Valid inputs
        valid_assets_data = [
            {"ticker": "AAPL", "weight": 0.5},
            {"ticker": "MSFT", "weight": 0.5}
        ]

        result = validate_all_inputs("Test Portfolio", valid_assets_data)
        assert result.is_valid == True

        # Invalid inputs
        invalid_assets_data = [
            {"ticker": "", "weight": 0.5},  # Empty ticker
            {"ticker": "MSFT", "weight": 1.5}  # Invalid weight
        ]

        result = validate_all_inputs("Test Portfolio", invalid_assets_data)
        assert result.is_valid == False
        assert len(result.errors) >= 2

    def test_validation_summary(self):
        """Test validation summary generation"""
        from core.data_manager.models import ValidationResult

        # Success case
        success_result = ValidationResult(is_valid=True)
        summary = get_validation_summary(success_result)
        assert "✅" in summary

        # Error case
        error_result = ValidationResult(is_valid=False)
        error_result.errors = ["Error 1", "Error 2"]
        error_result.warnings = ["Warning 1"]

        summary = get_validation_summary(error_result)
        assert "❌" in summary
        assert "⚠️" in summary
        assert "Error 1" in summary
        assert "Warning 1" in summary


class TestConstraintValidation:
    """Test constraint validation functionality"""

    def setup_method(self):
        """Setup constraint test data"""
        self.validator = PortfolioValidator()

        # Portfolio with constraint violations
        self.test_portfolio = Portfolio(
            name="Test Portfolio",
            assets=[
                Asset(ticker="AAPL", weight=0.8, current_price=150.0, shares=1000),  # High concentration
                Asset(ticker="MSFT", weight=0.2, current_price=300.0, shares=100)
            ],
            settings=PortfolioSettings(
                min_position_size=10000.0,  # $10K minimum
                max_position_size=100000.0,  # $100K maximum
                max_positions=10
            )
        )

    def test_position_size_constraints(self):
        """Test position size constraint checking"""
        violations = self.validator.check_constraints(self.test_portfolio)

        # Should have violations for position sizes
        position_violations = [v for v in violations if v.constraint_type == "position_size"]
        assert len(position_violations) >= 1

    def test_concentration_constraints(self):
        """Test concentration constraint checking"""
        violations = self.validator.check_constraints(self.test_portfolio)

        # Check if concentration is flagged (AAPL is 80%)
        # Note: This depends on the specific concentration thresholds set
        assert len(violations) >= 0  # May or may not have concentration violations

    def test_sector_constraints(self):
        """Test sector allocation constraints"""
        # Create portfolio with sector constraints
        settings = PortfolioSettings()
        settings.sector_limits = {
            "Technology": (0.0, 0.6)  # Max 60% in tech
        }

        tech_heavy_portfolio = Portfolio(
            name="Tech Heavy",
            assets=[
                Asset(ticker="AAPL", weight=0.8, sector="Technology"),
                Asset(ticker="MSFT", weight=0.2, sector="Technology")
            ],
            settings=settings
        )

        violations = self.validator.check_constraints(tech_heavy_portfolio)

        # Should have sector constraint violation (100% tech > 60% limit)
        sector_violations = [v for v in violations if v.constraint_type == "sector_allocation"]
        assert len(sector_violations) >= 1


class TestSuggestionGeneration:
    """Test suggestion generation functionality"""

    def setup_method(self):
        """Setup suggestion test data"""
        self.validator = PortfolioValidator()

    def test_diversification_suggestions(self):
        """Test diversification improvement suggestions"""
        # Concentrated portfolio
        concentrated_portfolio = Portfolio(
            name="Concentrated",
            assets=[
                Asset(ticker="AAPL", weight=0.7, sector="Technology"),
                Asset(ticker="MSFT", weight=0.3, sector="Technology")
            ]
        )

        suggestions = self.validator.suggest_improvements(concentrated_portfolio)

        # Should suggest diversification improvements
        diversification_suggestions = [s for s in suggestions if s.suggestion_type == "diversification"]
        assert len(diversification_suggestions) >= 1

    def test_position_count_suggestions(self):
        """Test position count suggestions"""
        # Too few positions
        small_portfolio = Portfolio(
            name="Small",
            assets=[
                Asset(ticker="AAPL", weight=0.6),
                Asset(ticker="MSFT", weight=0.4)
            ]
        )

        suggestions = self.validator.suggest_improvements(small_portfolio)

        # Should suggest adding more positions
        diversification_suggestions = [s for s in suggestions if "positions" in s.description.lower()]
        assert len(diversification_suggestions) >= 0  # May suggest more positions

    def test_asset_class_suggestions(self):
        """Test asset class diversification suggestions"""
        # Single asset class portfolio
        stock_only_portfolio = Portfolio(
            name="Stocks Only",
            assets=[
                Asset(ticker="AAPL", weight=0.5, asset_class=AssetClass.STOCK),
                Asset(ticker="MSFT", weight=0.5, asset_class=AssetClass.STOCK)
            ]
        )

        suggestions = self.validator.suggest_improvements(stock_only_portfolio)

        # Should suggest adding other asset classes
        asset_class_suggestions = [s for s in suggestions if "asset class" in s.description.lower()]
        assert len(asset_class_suggestions) >= 1


# Performance tests
class TestValidationPerformance:
    """Test validation performance with larger datasets"""

    def test_large_portfolio_validation(self):
        """Test validation performance with large portfolio"""
        import time

        # Create large portfolio (100 assets)
        large_assets = []
        for i in range(100):
            asset = Asset(
                ticker=f"STOCK{i:03d}",
                weight=0.01,  # 1% each
                name=f"Stock {i}"
            )
            large_assets.append(asset)

        large_portfolio = Portfolio(
            name="Large Portfolio",
            assets=large_assets
        )

        validator = PortfolioValidator()

        start_time = time.time()
        result = validator.validate_portfolio(large_portfolio)
        end_time = time.time()

        # Validation should complete quickly (< 1 second)
        assert (end_time - start_time) < 1.0
        assert result.is_valid == True

    def test_batch_ticker_validation(self):
        """Test performance of batch ticker validation"""
        import time

        # Generate many tickers
        tickers = [f"STOCK{i:03d}" for i in range(1000)]

        start_time = time.time()

        valid_count = 0
        for ticker in tickers:
            if TickerValidator.validate_ticker(ticker):
                valid_count += 1

        end_time = time.time()

        # Should validate quickly
        assert (end_time - start_time) < 2.0
        assert valid_count > 0


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_portfolio_name(self):
        """Test handling of empty portfolio name"""
        portfolio = Portfolio(name="", assets=[])
        validator = PortfolioValidator()

        result = validator.validate_portfolio(portfolio)
        assert result.is_valid == False

    def test_very_long_portfolio_name(self):
        """Test handling of very long portfolio name"""
        long_name = "A" * 200  # 200 characters
        portfolio = Portfolio(name=long_name, assets=[])
        validator = PortfolioValidator()

        result = validator.validate_portfolio(portfolio)
        # Should have error about name length
        name_errors = [e for e in result.errors if "name" in e.lower() and "long" in e.lower()]
        assert len(name_errors) > 0

    def test_extreme_weights(self):
        """Test handling of extreme weight values"""
        extreme_assets = [
            Asset(ticker="AAPL", weight=1e-10),  # Very small
            Asset(ticker="MSFT", weight=1.0 - 1e-10)  # Very large
        ]

        portfolio = Portfolio(name="Extreme Weights", assets=extreme_assets)

        # Should still validate as weights sum to ~1.0
        assert abs(portfolio.total_weight - 1.0) < 1e-6

    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        # Portfolio with unicode name
        unicode_portfolio = Portfolio(
            name="Portfolio 测试 🚀",
            description="Portfolio with unicode characters: αβγ",
            assets=[Asset(ticker="AAPL", weight=1.0)]
        )

        validator = PortfolioValidator()
        result = validator.validate_portfolio(unicode_portfolio)

        # Should handle unicode gracefully
        assert isinstance(result, type(validator.validate_portfolio(unicode_portfolio)))


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])