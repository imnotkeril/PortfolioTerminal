#!/usr/bin/env python3
"""
Basic Usage Examples for Wild Market Capital Portfolio Manager.

This script demonstrates how to use the core functionality of the portfolio
management system programmatically.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.data_manager import (
    PortfolioManager, PriceManager, Portfolio, Asset, AssetClass,
    PortfolioType, PortfolioSettings, ValidationError
)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n📊 {title}")
    print("-" * 40)


def example_1_basic_portfolio_creation():
    """Example 1: Basic portfolio creation and management."""

    print_header("EXAMPLE 1: Basic Portfolio Creation")

    # Initialize portfolio manager
    manager = PortfolioManager()

    print_section("Creating Assets")

    # Create individual assets
    apple = Asset(
        ticker="AAPL",
        name="Apple Inc.",
        weight=0.30,
        shares=100,
        purchase_price=140.0,
        sector="Technology",
        asset_class=AssetClass.STOCK
    )

    microsoft = Asset(
        ticker="MSFT",
        name="Microsoft Corporation",
        weight=0.25,
        shares=50,
        purchase_price=280.0,
        sector="Technology",
        asset_class=AssetClass.STOCK
    )

    google = Asset(
        ticker="GOOGL",
        name="Alphabet Inc.",
        weight=0.25,
        shares=20,
        purchase_price=2400.0,
        sector="Technology",
        asset_class=AssetClass.STOCK
    )

    amazon = Asset(
        ticker="AMZN",
        name="Amazon.com Inc.",
        weight=0.20,
        shares=30,
        purchase_price=3000.0,
        sector="Consumer Discretionary",
        asset_class=AssetClass.STOCK
    )

    assets = [apple, microsoft, google, amazon]

    print(f"Created {len(assets)} assets:")
    for asset in assets:
        print(f"  • {asset.ticker}: {asset.weight:.1%} - {asset.name}")

    print_section("Creating Portfolio")

    # Create portfolio
    portfolio = manager.create_portfolio(
        name="Tech Growth Portfolio",
        description="A technology-focused growth portfolio",
        assets=assets,
        portfolio_type=PortfolioType.GROWTH,
        initial_value=100000.0
    )

    print(f"✅ Portfolio created: {portfolio.name}")
    print(f"   ID: {portfolio.id}")
    print(f"   Assets: {len(portfolio.assets)}")
    print(f"   Total Weight: {portfolio.total_weight:.1%}")
    print(f"   Type: {portfolio.portfolio_type.value}")

    # Portfolio statistics
    stats = portfolio.get_statistics()
    print(f"   Total Value: ${stats.total_value:,.2f}")
    print(f"   Total Cost: ${stats.total_cost:,.2f}")

    return portfolio


def example_2_text_based_creation():
    """Example 2: Creating portfolio from text input."""

    print_header("EXAMPLE 2: Text-Based Portfolio Creation")

    manager = PortfolioManager()

    print_section("Creating from Text Input")

    # Different text formats
    text_formats = [
        ("Percentage Format", "AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%"),
        ("Decimal Format", "SPY 0.6, BND 0.3, GLD 0.1"),
        ("Equal Weight", "AAPL, MSFT, GOOGL, AMZN")
    ]

    portfolios = []

    for format_name, text_input in text_formats:
        print(f"\n{format_name}:")
        print(f"  Input: {text_input}")

        try:
            portfolio = manager.create_from_text(
                name=f"{format_name} Portfolio",
                text=text_input,
                description=f"Portfolio created using {format_name.lower()}"
            )

            print(f"  ✅ Created: {portfolio.name}")
            print(f"     Assets: {len(portfolio.assets)}")

            for asset in portfolio.assets:
                print(f"     • {asset.ticker}: {asset.weight:.1%}")

            portfolios.append(portfolio)

        except Exception as e:
            print(f"  ❌ Error: {e}")

    return portfolios


def example_3_price_management():
    """Example 3: Working with price data."""

    print_header("EXAMPLE 3: Price Management")

    # Initialize price manager
    price_manager = PriceManager()

    print_section("Fetching Current Prices")

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    try:
        prices = price_manager.get_current_prices(tickers)

        print("Current Prices:")
        for ticker, price in prices.items():
            if price:
                print(f"  • {ticker}: ${price:.2f}")
            else:
                print(f"  • {ticker}: Price not available")

        print_section("Market Status")

        market_status = price_manager.get_market_status()
        status_text = "🟢 Open" if market_status.is_open else "🔴 Closed"
        print(f"Market Status: {status_text}")

        if market_status.next_open:
            print(f"Next Open: {market_status.next_open.strftime('%Y-%m-%d %H:%M')}")

        if market_status.next_close:
            print(f"Next Close: {market_status.next_close.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        print(f"❌ Price fetching failed: {e}")
        print("This may be due to network connectivity or API limits")


def example_4_portfolio_operations():
    """Example 4: Advanced portfolio operations."""

    print_header("EXAMPLE 4: Advanced Portfolio Operations")

    manager = PortfolioManager()

    # Create a base portfolio
    base_portfolio = manager.create_from_text(
        name="Base Portfolio",
        text="AAPL 40%, MSFT 30%, GOOGL 30%"
    )

    print_section("Portfolio Validation")

    validation_result = manager.validate_portfolio(base_portfolio)

    if validation_result.is_valid:
        print("✅ Portfolio is valid")
    else:
        print("❌ Portfolio has issues:")
        for error in validation_result.errors:
            print(f"  • {error}")

    if validation_result.warnings:
        print("⚠️  Warnings:")
        for warning in validation_result.warnings:
            print(f"  • {warning}")

    print_section("Portfolio Modifications")

    # Add a new asset
    try:
        new_asset = Asset(
            ticker="TSLA",
            name="Tesla Inc.",
            weight=0.15,
            sector="Automotive",
            asset_class=AssetClass.STOCK
        )

        base_portfolio.add_asset(new_asset)

        # Normalize weights to maintain 100%
        base_portfolio.normalize_weights()

        print("✅ Added TSLA and normalized weights")
        print("Updated allocation:")
        for asset in base_portfolio.assets:
            print(f"  • {asset.ticker}: {asset.weight:.1%}")

    except Exception as e:
        print(f"❌ Error modifying portfolio: {e}")

    print_section("Portfolio Cloning")

    try:
        cloned_portfolio = manager.clone_portfolio(
            base_portfolio.id,
            "Cloned Portfolio"
        )

        print(f"✅ Cloned portfolio: {cloned_portfolio.name}")
        print(f"   Original ID: {base_portfolio.id}")
        print(f"   Clone ID: {cloned_portfolio.id}")
        print(f"   Assets in clone: {len(cloned_portfolio.assets)}")

    except Exception as e:
        print(f"❌ Error cloning portfolio: {e}")

    return base_portfolio


def example_5_import_export():
    """Example 5: Import and export operations."""

    print_header("EXAMPLE 5: Import/Export Operations")

    manager = PortfolioManager()

    # Create a sample portfolio
    portfolio = manager.create_from_text(
        name="Export Example Portfolio",
        text="SPY 50%, BND 30%, VTI 20%"
    )

    print_section("Export Operations")

    # Define export files
    exports_dir = project_root / "exports"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_file = exports_dir / f"example_portfolio_{timestamp}.csv"
    excel_file = exports_dir / f"example_portfolio_{timestamp}.xlsx"
    json_file = exports_dir / f"example_portfolio_{timestamp}.json"

    try:
        # Export to different formats
        csv_path = manager.export_to_csv(portfolio.id, str(csv_file))
        print(f"✅ CSV Export: {csv_path}")

        excel_path = manager.export_to_excel(portfolio.id, str(excel_file))
        print(f"✅ Excel Export: {excel_path}")

        json_path = manager.export_to_json(portfolio.id, str(json_file))
        print(f"✅ JSON Export: {json_path}")

        # Show file sizes
        for file_path in [csv_file, excel_file, json_file]:
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"   {file_path.name}: {size_kb:.1f} KB")

    except Exception as e:
        print(f"❌ Export failed: {e}")


def example_6_portfolio_settings():
    """Example 6: Working with portfolio settings."""

    print_header("EXAMPLE 6: Portfolio Settings")

    print_section("Creating Custom Settings")

    # Create custom portfolio settings
    custom_settings = PortfolioSettings(
        rebalancing_frequency="monthly",
        rebalancing_threshold=0.03,  # 3%
        auto_rebalance=True,
        min_position_size=5000.0,  # $5K minimum
        max_position_size=25000.0,  # $25K maximum
        max_positions=20,
        max_drawdown=0.15,  # 15% max drawdown
        tax_rate=0.28,  # 28% tax rate
        stop_loss=0.10,  # 10% stop loss
        take_profit=0.50  # 50% take profit
    )

    print("Custom Settings:")
    print(f"  • Rebalancing: {custom_settings.rebalancing_frequency}")
    print(f"  • Threshold: {custom_settings.rebalancing_threshold:.1%}")
    print(f"  • Auto-rebalance: {custom_settings.auto_rebalance}")
    print(f"  • Position size: ${custom_settings.min_position_size:,.0f} - ${custom_settings.max_position_size:,.0f}")
    print(f"  • Max positions: {custom_settings.max_positions}")
    print(f"  • Max drawdown: {custom_settings.max_drawdown:.1%}")

    # Create portfolio with custom settings
    manager = PortfolioManager()

    assets = [
        Asset(ticker="VTI", weight=0.6, name="Total Stock Market ETF"),
        Asset(ticker="BND", weight=0.3, name="Total Bond Market ETF"),
        Asset(ticker="VXUS", weight=0.1, name="Total International Stock ETF")
    ]

    portfolio = manager.create_portfolio(
        name="Conservative Balanced Portfolio",
        description="Balanced portfolio with conservative settings",
        assets=assets,
        settings=custom_settings,
        portfolio_type=PortfolioType.BALANCED
    )

    print(f"\n✅ Created portfolio with custom settings: {portfolio.name}")

    return portfolio


def example_7_error_handling():
    """Example 7: Error handling and validation."""

    print_header("EXAMPLE 7: Error Handling")

    print_section("Validation Errors")

    manager = PortfolioManager()

    # Demonstrate various error conditions
    error_cases = [
        {
            "name": "Invalid Weights",
            "text": "AAPL 60%, MSFT 70%",  # Sums to 130%
            "description": "Weights sum to more than 100%"
        },
        {
            "name": "Invalid Ticker",
            "text": "AAPL 50%, INVALID_TICKER 50%",
            "description": "Contains invalid ticker symbol"
        },
        {
            "name": "Empty Input",
            "text": "",
            "description": "Empty text input"
        }
    ]

    for case in error_cases:
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        print(f"  Input: '{case['text']}'")

        try:
            portfolio = manager.create_from_text(
                name=case['name'],
                text=case['text']
            )
            print(f"  ✅ Unexpectedly succeeded: {portfolio.name}")

        except ValidationError as e:
            print(f"  ❌ Validation Error (expected): {e}")

        except ValueError as e:
            print(f"  ❌ Value Error (expected): {e}")

        except Exception as e:
            print(f"  ❌ Unexpected Error: {e}")


def main():
    """Run all examples."""

    print("🚀 Wild Market Capital Portfolio Manager")
    print("📚 Basic Usage Examples")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Run all examples
        example_1_basic_portfolio_creation()
        example_2_text_based_creation()
        example_3_price_management()
        example_4_portfolio_operations()
        example_5_import_export()
        example_6_portfolio_settings()
        example_7_error_handling()

        print_header("EXAMPLES COMPLETED")
        print("✅ All examples completed successfully!")
        print("\nNext Steps:")
        print("• Launch the web interface: python run.py")
        print("• Explore the Streamlit dashboard")
        print("• Create your own portfolios")
        print("• Check out the test suite: python run.py --test")

    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")

    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()