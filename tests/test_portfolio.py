"""
Unit tests for Portfolio class and related functionality.

This module contains comprehensive tests for the portfolio data structures
and basic operations to ensure data integrity and business logic correctness.
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Add core module to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from core.data_manager import (
    Portfolio, Asset, PortfolioSettings, PortfolioManager,
    AssetClass, PortfolioType, ValidationError
)


class TestPortfolio:
    """Test Portfolio class functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.sample_assets = [
            Asset(ticker="AAPL", name="Apple Inc.", weight=0.3, asset_class=AssetClass.STOCK),
            Asset(ticker="MSFT", name="Microsoft", weight=0.3, asset_class=AssetClass.STOCK),
            Asset(ticker="GOOGL", name="Alphabet", weight=0.4, asset_class=AssetClass.STOCK)
        ]

        self.sample_portfolio = Portfolio(
            name="Test Portfolio",
            description="Test portfolio for unit testing",
            assets=self.sample_assets.copy()
        )

    def test_portfolio_creation(self):
        """Test basic portfolio creation"""
        portfolio = Portfolio(name="Test Portfolio")

        assert portfolio.name == "Test Portfolio"
        assert portfolio.id is not None
        assert len(portfolio.id) > 0
        assert portfolio.assets == []
        assert portfolio.total_weight == 0.0

    def test_portfolio_with_assets(self):
        """Test portfolio creation with assets"""
        portfolio = self.sample_portfolio

        assert len(portfolio.assets) == 3
        assert portfolio.total_weight == 1.0  # 0.3 + 0.3 + 0.4
        assert portfolio.asset_count == 3
        assert set(portfolio.tickers) == {"AAPL", "MSFT", "GOOGL"}

    def test_portfolio_validation(self):
        """Test portfolio validation"""
        # Valid portfolio
        assert self.sample_portfolio.validate() == True

        # Invalid portfolio - weights don't sum to 1
        invalid_portfolio = Portfolio(
            name="Invalid",
            assets=[
                Asset(ticker="AAPL", weight=0.5),
                Asset(ticker="MSFT", weight=0.7)  # Total = 1.2
            ]
        )

        assert invalid_portfolio.validate() == False

    def test_add_asset(self):
        """Test adding assets to portfolio"""
        portfolio = Portfolio(name="Test")

        asset = Asset(ticker="AMZN", weight=1.0)
        portfolio.add_asset(asset)

        assert len(portfolio.assets) == 1
        assert portfolio.assets[0].ticker == "AMZN"

    def test_add_duplicate_asset(self):
        """Test adding duplicate asset (should fail)"""
        portfolio = self.sample_portfolio

        duplicate_asset = Asset(ticker="AAPL", weight=0.1)

        with pytest.raises(ValueError, match="already exists"):
            portfolio.add_asset(duplicate_asset)

    def test_remove_asset(self):
        """Test removing assets from portfolio"""
        portfolio = self.sample_portfolio

        portfolio.remove_asset("AAPL")

        assert len(portfolio.assets) == 2
        assert "AAPL" not in portfolio.tickers

    def test_remove_nonexistent_asset(self):
        """Test removing non-existent asset (should fail)"""
        portfolio = self.sample_portfolio

        with pytest.raises(ValueError, match="not found"):
            portfolio.remove_asset("TSLA")

    def test_update_asset(self):
        """Test updating asset properties"""
        portfolio = self.sample_portfolio

        portfolio.update_asset("AAPL", {"weight": 0.5, "name": "Apple Inc. Updated"})

        apple_asset = portfolio.get_asset("AAPL")
        assert apple_asset.weight == 0.5
        assert apple_asset.name == "Apple Inc. Updated"

    def test_normalize_weights(self):
        """Test weight normalization"""
        portfolio = Portfolio(
            name="Test",
            assets=[
                Asset(ticker="AAPL", weight=0.6),
                Asset(ticker="MSFT", weight=0.8)  # Total = 1.4
            ]
        )

        portfolio.normalize_weights()

        # Weights should now sum to 1.0
        assert abs(portfolio.total_weight - 1.0) < 0.001

        # Individual weights should be proportional
        apple_asset = portfolio.get_asset("AAPL")
        msft_asset = portfolio.get_asset("MSFT")

        assert abs(apple_asset.weight - 0.6 / 1.4) < 0.001
        assert abs(msft_asset.weight - 0.8 / 1.4) < 0.001

    def test_calculate_value(self):
        """Test portfolio value calculation"""
        portfolio = Portfolio(
            name="Test",
            assets=[
                Asset(ticker="AAPL", weight=0.5, shares=100, current_price=150.0),
                Asset(ticker="MSFT", weight=0.5, shares=50, current_price=300.0)
            ]
        )

        expected_value = (100 * 150.0) + (50 * 300.0)  # 15,000 + 15,000 = 30,000
        assert portfolio.calculate_value() == expected_value

    def test_calculate_value_with_prices(self):
        """Test portfolio value calculation with provided prices"""
        portfolio = Portfolio(
            name="Test",
            assets=[
                Asset(ticker="AAPL", weight=0.5, shares=100),
                Asset(ticker="MSFT", weight=0.5, shares=50)
            ]
        )

        prices = {"AAPL": 160.0, "MSFT": 320.0}
        expected_value = (100 * 160.0) + (50 * 320.0)  # 16,000 + 16,000 = 32,000

        assert portfolio.calculate_value(prices) == expected_value

    def test_get_statistics(self):
        """Test portfolio statistics calculation"""
        portfolio = Portfolio(
            name="Test",
            assets=[
                Asset(
                    ticker="AAPL",
                    weight=0.5,
                    shares=100,
                    current_price=150.0,
                    purchase_price=140.0
                )
            ]
        )

        stats = portfolio.get_statistics()

        assert stats.asset_count == 1
        assert stats.total_value == 15000.0  # 100 * 150
        assert stats.total_cost == 14000.0  # 100 * 140
        assert stats.unrealized_pnl == 1000.0  # 15000 - 14000

    def test_sector_allocation(self):
        """Test sector allocation calculation"""
        portfolio = Portfolio(
            name="Test",
            assets=[
                Asset(ticker="AAPL", weight=0.4, sector="Technology"),
                Asset(ticker="MSFT", weight=0.3, sector="Technology"),
                Asset(ticker="JPM", weight=0.3, sector="Finance")
            ]
        )

        sectors = portfolio.get_sector_allocation()

        assert sectors["Technology"] == 0.7  # 0.4 + 0.3
        assert sectors["Finance"] == 0.3

    def test_asset_class_allocation(self):
        """Test asset class allocation"""
        portfolio = Portfolio(
            name="Test",
            assets=[
                Asset(ticker="AAPL", weight=0.6, asset_class=AssetClass.STOCK),
                Asset(ticker="BND", weight=0.4, asset_class=AssetClass.BOND)
            ]
        )

        allocation = portfolio.get_asset_class_allocation()

        assert allocation["stock"] == 0.6
        assert allocation["bond"] == 0.4

    def test_portfolio_serialization(self):
        """Test portfolio to_dict and from_dict"""
        original = self.sample_portfolio

        # Convert to dict and back
        portfolio_dict = original.to_dict()
        restored = Portfolio.from_dict(portfolio_dict)

        # Check equality
        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.assets) == len(original.assets)
        assert restored.portfolio_type == original.portfolio_type

        # Check asset details
        for orig_asset, rest_asset in zip(original.assets, restored.assets):
            assert orig_asset.ticker == rest_asset.ticker
            assert orig_asset.weight == rest_asset.weight
            assert orig_asset.name == rest_asset.name

    def test_portfolio_copy(self):
        """Test portfolio copying"""
        original = self.sample_portfolio
        copied = original.copy()

        # Should be different objects
        assert copied is not original
        assert copied.id != original.id

        # But same content
        assert copied.name == original.name
        assert len(copied.assets) == len(original.assets)


class TestAsset:
    """Test Asset class functionality"""

    def test_asset_creation(self):
        """Test basic asset creation"""
        asset = Asset(ticker="AAPL", weight=0.5)

        assert asset.ticker == "AAPL"
        assert asset.weight == 0.5
        assert asset.asset_class == AssetClass.STOCK

    def test_asset_validation(self):
        """Test asset validation"""
        # Valid asset
        valid_asset = Asset(ticker="AAPL", weight=0.5, shares=100, current_price=150.0)
        assert valid_asset.validate() == True

        # Invalid weight
        invalid_asset = Asset(ticker="AAPL", weight=1.5)  # > 1.0
        assert invalid_asset.validate() == False

        # Negative shares
        invalid_asset2 = Asset(ticker="AAPL", weight=0.5, shares=-10)
        assert invalid_asset2.validate() == False

    def test_market_value_calculation(self):
        """Test market value calculation"""
        asset = Asset(ticker="AAPL", shares=100, current_price=150.0)

        assert asset.market_value == 15000.0  # 100 * 150

    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation"""
        asset = Asset(
            ticker="AAPL",
            shares=100,
            current_price=150.0,
            purchase_price=140.0
        )

        assert asset.unrealized_pnl == 1000.0  # (150 - 140) * 100
        assert abs(asset.unrealized_pnl_percent - (10.0 / 140.0)) < 0.001

    def test_update_price(self):
        """Test price update"""
        asset = Asset(ticker="AAPL")

        asset.update_price(150.0)
        assert asset.current_price == 150.0

        # Should fail with invalid price
        with pytest.raises(ValueError):
            asset.update_price(-10.0)

    def test_asset_serialization(self):
        """Test asset serialization"""
        asset = Asset(
            ticker="AAPL",
            name="Apple Inc.",
            weight=0.5,
            purchase_date=datetime.now()
        )

        # Convert to dict and back
        asset_dict = asset.to_dict()
        restored = Asset.from_dict(asset_dict)

        assert restored.ticker == asset.ticker
        assert restored.name == asset.name
        assert restored.weight == asset.weight
        assert restored.purchase_date == asset.purchase_date


class TestPortfolioManager:
    """Test PortfolioManager functionality"""

    def setup_method(self):
        """Setup test environment"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PortfolioManager(storage_path=self.temp_dir)

        # Sample assets for testing
        self.test_assets = [
            Asset(ticker="AAPL", name="Apple Inc.", weight=0.4),
            Asset(ticker="MSFT", name="Microsoft", weight=0.6)
        ]

    def teardown_method(self):
        """Cleanup test environment"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_create_portfolio(self):
        """Test portfolio creation through manager"""
        portfolio = self.manager.create_portfolio(
            name="Test Portfolio",
            description="Test description",
            assets=self.test_assets
        )

        assert portfolio.name == "Test Portfolio"
        assert len(portfolio.assets) == 2
        assert portfolio.validate() == True

    def test_create_invalid_portfolio(self):
        """Test creation of invalid portfolio (should fail)"""
        invalid_assets = [
            Asset(ticker="AAPL", weight=0.7),
            Asset(ticker="MSFT", weight=0.8)  # Total = 1.5
        ]

        with pytest.raises(ValueError):
            self.manager.create_portfolio(
                name="Invalid Portfolio",
                assets=invalid_assets
            )

    def test_save_and_load_portfolio(self):
        """Test portfolio persistence"""
        # Create and save portfolio
        original = self.manager.create_portfolio(
            name="Test Save/Load",
            assets=self.test_assets
        )

        # Load portfolio
        loaded = self.manager.get_portfolio(original.id)

        assert loaded.name == original.name
        assert len(loaded.assets) == len(original.assets)
        assert loaded.validate() == True

    def test_update_portfolio(self):
        """Test portfolio updates"""
        # Create portfolio
        portfolio = self.manager.create_portfolio(
            name="Original Name",
            assets=self.test_assets
        )

        # Update portfolio
        updated = self.manager.update_portfolio(
            portfolio.id,
            {"name": "Updated Name", "description": "Updated description"}
        )

        assert updated.name == "Updated Name"
        assert updated.description == "Updated description"
        assert updated.last_modified > portfolio.last_modified

    def test_delete_portfolio(self):
        """Test portfolio deletion"""
        # Create portfolio
        portfolio = self.manager.create_portfolio(
            name="To Delete",
            assets=self.test_assets
        )

        portfolio_id = portfolio.id

        # Delete portfolio
        success = self.manager.delete_portfolio(portfolio_id)
        assert success == True

        # Try to load deleted portfolio (should fail)
        with pytest.raises(FileNotFoundError):
            self.manager.get_portfolio(portfolio_id)

    def test_list_portfolios(self):
        """Test portfolio listing"""
        # Create multiple portfolios
        portfolio1 = self.manager.create_portfolio("Portfolio 1", assets=self.test_assets)
        portfolio2 = self.manager.create_portfolio("Portfolio 2", assets=self.test_assets)

        # List portfolios
        portfolios = self.manager.list_portfolios()

        assert len(portfolios) == 2
        portfolio_names = [p.name for p in portfolios]
        assert "Portfolio 1" in portfolio_names
        assert "Portfolio 2" in portfolio_names

    def test_create_from_text_percentage(self):
        """Test creating portfolio from text with percentages"""
        text = "AAPL 40%, MSFT 35%, GOOGL 25%"

        portfolio = self.manager.create_from_text("Text Portfolio", text)

        assert len(portfolio.assets) == 3
        assert abs(portfolio.total_weight - 1.0) < 0.001

        # Check individual weights
        aapl = portfolio.get_asset("AAPL")
        assert abs(aapl.weight - 0.4) < 0.001

    def test_create_from_text_equal_weight(self):
        """Test creating portfolio from text with equal weighting"""
        text = "AAPL, MSFT, GOOGL, AMZN"

        portfolio = self.manager.create_from_text("Equal Weight Portfolio", text)

        assert len(portfolio.assets) == 4
        assert abs(portfolio.total_weight - 1.0) < 0.001

        # Each asset should have 25% weight
        for asset in portfolio.assets:
            assert abs(asset.weight - 0.25) < 0.001

    def test_create_from_text_invalid(self):
        """Test creating portfolio from invalid text"""
        invalid_text = "This is not a valid ticker list"

        with pytest.raises(ValueError):
            self.manager.create_from_text("Invalid Portfolio", invalid_text)

    def test_clone_portfolio(self):
        """Test portfolio cloning"""
        # Create original portfolio
        original = self.manager.create_portfolio(
            name="Original",
            description="Original description",
            assets=self.test_assets
        )

        # Clone portfolio
        cloned = self.manager.clone_portfolio(original.id, "Cloned Portfolio")

        assert cloned.name == "Cloned Portfolio"
        assert cloned.id != original.id
        assert len(cloned.assets) == len(original.assets)
        assert cloned.description == original.description
        assert len(cloned.trade_history) == 0  # Should start empty

    def test_portfolio_search(self):
        """Test portfolio search functionality"""
        # Create portfolios with different characteristics
        tech_portfolio = self.manager.create_portfolio(
            "Tech Growth",
            "Technology focused portfolio",
            self.test_assets
        )
        tech_portfolio.tags = ["tech", "growth"]

        conservative_portfolio = self.manager.create_portfolio(
            "Conservative Income",
            "Low risk income portfolio",
            self.test_assets
        )
        conservative_portfolio.tags = ["conservative", "income"]

        # Search by name
        results = self.manager.search_portfolios("tech")
        assert len(results) == 1
        assert results[0].name == "Tech Growth"

        # Search by description
        results = self.manager.search_portfolios("income")
        assert len(results) == 1
        assert results[0].name == "Conservative Income"

    def test_portfolio_filters(self):
        """Test portfolio filtering"""
        # Create portfolios
        growth_portfolio = self.manager.create_portfolio(
            "Growth Portfolio",
            assets=self.test_assets,
            portfolio_type=PortfolioType.GROWTH
        )

        balanced_portfolio = self.manager.create_portfolio(
            "Balanced Portfolio",
            assets=self.test_assets,
            portfolio_type=PortfolioType.BALANCED
        )

        # Filter by type
        growth_portfolios = self.manager.list_portfolios(
            filters={"type": "growth"}
        )

        assert len(growth_portfolios) == 1
        assert growth_portfolios[0].name == "Growth Portfolio"

        # Filter by minimum assets
        filtered = self.manager.list_portfolios(
            filters={"min_assets": 2}
        )

        assert len(filtered) == 2  # Both have 2 assets


class TestPortfolioSettings:
    """Test PortfolioSettings functionality"""

    def test_default_settings(self):
        """Test default settings creation"""
        settings = PortfolioSettings()

        assert settings.rebalancing_frequency == "quarterly"
        assert settings.auto_rebalance == False
        assert settings.max_drawdown == 0.20
        assert settings.tax_rate == 0.25

    def test_settings_serialization(self):
        """Test settings serialization"""
        settings = PortfolioSettings(
            rebalancing_frequency="monthly",
            auto_rebalance=True,
            max_drawdown=0.15
        )

        # Convert to dict and back
        settings_dict = settings.to_dict()
        restored = PortfolioSettings.from_dict(settings_dict)

        assert restored.rebalancing_frequency == settings.rebalancing_frequency
        assert restored.auto_rebalance == settings.auto_rebalance
        assert restored.max_drawdown == settings.max_drawdown


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""

    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PortfolioManager(storage_path=self.temp_dir)

    def teardown_method(self):
        """Cleanup integration test environment"""
        shutil.rmtree(self.temp_dir)

    def test_complete_portfolio_lifecycle(self):
        """Test complete portfolio creation, modification, and deletion"""
        # Create portfolio
        portfolio = self.manager.create_from_text(
            "Lifecycle Test",
            "AAPL 50%, MSFT 50%"
        )

        assert portfolio.validate() == True
        assert len(portfolio.assets) == 2

        # Update portfolio
        updated = self.manager.update_portfolio(
            portfolio.id,
            {"description": "Updated description"}
        )

        assert updated.description == "Updated description"

        # Add asset
        new_asset = Asset(ticker="GOOGL", weight=0.2)
        updated.add_asset(new_asset)
        updated.normalize_weights()

        assert len(updated.assets) == 3
        assert abs(updated.total_weight - 1.0) < 0.001

        # Save changes
        self.manager.update_portfolio(updated.id, {"assets": updated.assets})

        # Reload and verify
        reloaded = self.manager.get_portfolio(updated.id)
        assert len(reloaded.assets) == 3

        # Delete portfolio
        success = self.manager.delete_portfolio(portfolio.id)
        assert success == True


# Test fixtures and utilities
@pytest.fixture
def sample_portfolio():
    """Fixture providing a sample portfolio"""
    assets = [
        Asset(ticker="AAPL", name="Apple Inc.", weight=0.3),
        Asset(ticker="MSFT", name="Microsoft", weight=0.3),
        Asset(ticker="GOOGL", name="Alphabet", weight=0.4)
    ]

    return Portfolio(
        name="Sample Portfolio",
        description="Sample portfolio for testing",
        assets=assets
    )


@pytest.fixture
def temp_manager():
    """Fixture providing a temporary portfolio manager"""
    temp_dir = tempfile.mkdtemp()
    manager = PortfolioManager(storage_path=temp_dir)

    yield manager

    # Cleanup
    shutil.rmtree(temp_dir)


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__])