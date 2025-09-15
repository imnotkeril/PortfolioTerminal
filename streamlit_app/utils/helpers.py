"""
Helper utilities for the Portfolio Management System.

This module contains miscellaneous helper functions used throughout the application.
"""
import streamlit as st
import pandas as pd
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from io import BytesIO
import sys
from pathlib import Path
from ..utils.formatting import format_currency, format_percentage

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import Portfolio, Asset


def display_portfolio_summary(portfolio: Portfolio):
    """
    Display a unified summary card for a portfolio with auto-fetched sectors and remaining cash.

    Args:
        portfolio: Portfolio object to summarize
    """
    from .session_state import get_price_manager

    # Calculate basic metrics
    total_assets = len(portfolio.assets)
    total_value = portfolio.calculate_value()

    # Calculate remaining cash
    invested_value = 0.0
    price_manager = get_price_manager()

    # Try to auto-fetch company info for assets without sectors (but don't save to avoid method errors)
    assets_needing_update = [asset for asset in portfolio.assets if
                             not asset.sector or not asset.name or asset.sector == "Unknown"]

    if assets_needing_update:
        for asset in assets_needing_update:
            try:
                # Try to get company info from API
                company_info = price_manager.get_company_info(asset.ticker)
                if company_info:
                    if not asset.name or asset.name == f"{asset.ticker} Corp":
                        asset.name = company_info.name

                    if not asset.sector or asset.sector == "Unknown":
                        asset.sector = company_info.sector or "Unknown"
                else:
                    # Set defaults if API fails
                    if not asset.name:
                        asset.name = f"{asset.ticker} Corp"
                    if not asset.sector:
                        asset.sector = "Unknown"

            except Exception:
                # Set defaults if everything fails
                if not asset.name:
                    asset.name = f"{asset.ticker} Corp"
                if not asset.sector:
                    asset.sector = "Unknown"

    # Calculate invested value based on current prices and shares
    for asset in portfolio.assets:
        if asset.current_price and asset.shares:
            invested_value += asset.current_price * asset.shares
        elif asset.weight and portfolio.initial_value:
            # Fallback to weight-based calculation
            invested_value += asset.weight * portfolio.initial_value

    remaining_cash = max(0, portfolio.initial_value - invested_value)

    # Create columns for layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Assets", total_assets)

    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")

    with col3:
        st.metric("Portfolio Type", portfolio.portfolio_type.value.title())

    with col4:
        # Show remaining cash instead of age
        st.metric("Remaining Cash", f"${remaining_cash:,.2f}")

    # Show asset breakdown
    if portfolio.assets:
        st.subheader("Top Holdings")

        # Sort assets by weight
        sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
        top_assets = sorted_assets[:5]  # Show top 5

        for asset in top_assets:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                # Display ticker, name, and sector
                display_name = asset.name if asset.name and asset.name != f"{asset.ticker} Corp" else f"{asset.ticker} Corp"
                sector_info = f" • {asset.sector}" if asset.sector and asset.sector != "Unknown" else ""
                st.write(f"**{asset.ticker}** - {display_name}{sector_info}")

            with col2:
                st.write(f"{asset.weight:.1%}")

            with col3:
                if hasattr(asset, 'current_price') and asset.current_price:
                    st.write(f"${asset.current_price:.2f}")
                else:
                    st.write("N/A")


def force_update_company_info(portfolio: Portfolio):
    """Force update company information for all assets"""
    from .session_state import get_price_manager, get_portfolio_manager

    price_manager = get_price_manager()
    updated_count = 0

    with st.spinner("Updating all company information..."):
        for asset in portfolio.assets:
            try:
                company_info = price_manager.get_company_info(asset.ticker)
                if company_info:
                    asset.name = company_info.name
                    asset.sector = company_info.sector or "Unknown"
                    updated_count += 1
                else:
                    # Set defaults
                    asset.name = f"{asset.ticker} Corp"
                    asset.sector = "Unknown"

            except Exception as e:
                st.warning(f"Could not fetch info for {asset.ticker}: {e}")
                asset.name = f"{asset.ticker} Corp"
                asset.sector = "Unknown"

        # Save portfolio
        try:
            portfolio_manager = get_portfolio_manager()
            portfolio_manager.save_portfolio(portfolio)
            st.success(f"✅ Updated and saved information for {updated_count} assets!")
        except Exception as e:
            st.error(f"Error saving portfolio: {e}")

    # Create columns for layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Assets", total_assets)

    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")

    with col3:
        st.metric("Portfolio Type", portfolio.portfolio_type.value.title())

    with col4:
        # Show remaining cash instead of age
        st.metric("Remaining Cash", f"${remaining_cash:,.2f}")

    # Show asset breakdown
    if portfolio.assets:
        st.subheader("Top Holdings")

        # Sort assets by weight
        sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
        top_assets = sorted_assets[:5]  # Show top 5

        for asset in top_assets:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                # Display ticker, name, and sector
                display_name = asset.name if asset.name and asset.name != f"{asset.ticker} Corp" else f"{asset.ticker} Corp"
                sector_info = f" • {asset.sector}" if asset.sector and asset.sector != "Unknown" else ""
                st.write(f"**{asset.ticker}** - {display_name}{sector_info}")

            with col2:
                st.write(f"{asset.weight:.1%}")

            with col3:
                if hasattr(asset, 'current_price') and asset.current_price:
                    st.write(f"${asset.current_price:.2f}")
                else:
                    st.write("N/A")


def force_update_company_info(portfolio: Portfolio):
    """Force update company information for all assets"""
    from .session_state import get_price_manager, get_portfolio_manager

    price_manager = get_price_manager()
    updated_count = 0

    with st.spinner("Updating all company information..."):
        for asset in portfolio.assets:
            try:
                company_info = price_manager.get_company_info(asset.ticker)
                if company_info:
                    asset.name = company_info.name
                    asset.sector = company_info.sector or "Unknown"
                    updated_count += 1
                else:
                    # Set defaults
                    asset.name = f"{asset.ticker} Corp"
                    asset.sector = "Unknown"

            except Exception as e:
                st.warning(f"Could not fetch info for {asset.ticker}: {e}")
                asset.name = f"{asset.ticker} Corp"
                asset.sector = "Unknown"

        # Save portfolio
        try:
            portfolio_manager = get_portfolio_manager()
            portfolio_manager.save_portfolio(portfolio)
            st.success(f"✅ Updated and saved information for {updated_count} assets!")
        except Exception as e:
            st.error(f"Error saving portfolio: {e}")


def update_company_info(portfolio: Portfolio):
    """
    Update company information for portfolio assets using real API data.

    Args:
        portfolio: Portfolio object to update
    """
    from .session_state import get_price_manager

    with st.spinner("Fetching company information..."):
        price_manager = get_price_manager()
        updated_count = 0

        for asset in portfolio.assets:
            try:
                # Get company info from Yahoo Finance API
                company_info = price_manager.get_company_info(asset.ticker)

                if company_info:
                    # Update name if not set
                    if not asset.name or asset.name == f"{asset.ticker} Corp":
                        asset.name = company_info.name

                    # Update sector if not set
                    if not asset.sector or asset.sector == "Unknown":
                        asset.sector = company_info.sector or "Unknown"

                    # Update industry if available
                    if hasattr(asset, 'industry') and company_info.industry:
                        asset.industry = company_info.industry

                    # Update market cap if available
                    if hasattr(asset, 'market_cap') and company_info.market_cap:
                        asset.market_cap = company_info.market_cap

                    # Update country if available
                    if hasattr(asset, 'country') and company_info.country:
                        asset.country = company_info.country

                    updated_count += 1
                else:
                    # Fallback data if API fails
                    if not asset.name:
                        asset.name = f"{asset.ticker} Corp"
                    if not asset.sector:
                        asset.sector = "Unknown"

            except Exception as e:
                st.warning(f"Could not fetch info for {asset.ticker}: {str(e)}")
                # Set fallback values
                if not asset.name:
                    asset.name = f"{asset.ticker} Corp"
                if not asset.sector:
                    asset.sector = "Unknown"

        if updated_count > 0:
            st.success(f"Updated company information for {updated_count} assets")

            # Show what sectors were found
            sectors_found = set(
                asset.sector for asset in portfolio.assets if asset.sector and asset.sector != "Unknown")
            if sectors_found:
                st.info(f"Sectors identified: {', '.join(sorted(sectors_found))}")
        else:
            st.info("No new company information was available")

        time.sleep(1)  # Brief pause for user experience


def update_portfolio_prices(portfolio: Portfolio):
    """
    Update prices for all assets in a portfolio.

    Args:
        portfolio: Portfolio object to update
    """
    from .session_state import get_price_manager, update_last_price_update

    with st.spinner("Updating prices..."):
        try:
            price_manager = get_price_manager()

            # Get all tickers
            tickers = [asset.ticker for asset in portfolio.assets]

            # Fetch current prices
            prices = price_manager.get_current_prices(tickers)

            # Update asset prices
            updated_count = 0
            for asset in portfolio.assets:
                if asset.ticker in prices and prices[asset.ticker]:
                    old_price = getattr(asset, 'current_price', None)
                    asset.current_price = prices[asset.ticker]

                    if old_price != asset.current_price:
                        updated_count += 1

            # Update timestamp
            update_last_price_update()

            if updated_count > 0:
                st.success(f"Updated prices for {updated_count} assets")
            else:
                st.info("All prices are up to date")

        except Exception as e:
            st.error(f"Error updating prices: {e}")


def validate_ticker_input(text: str) -> tuple[bool, str, list]:
    """
    Validate and parse ticker input text.

    Args:
        text: Input text with ticker symbols and weights

    Returns:
        Tuple of (is_valid, error_message, parsed_tickers)
    """

    if not text or not text.strip():
        return False, "No input provided", []

    try:
        from core.data_manager.validators import TextParser
        parsed_data = TextParser.parse_text_input(text)

        if not parsed_data:
            return False, "Could not parse any valid ticker-weight pairs", []

        # Convert to list of tuples for compatibility
        parsed_tickers = [(item['ticker'], item['weight']) for item in parsed_data]

        return True, "", parsed_tickers

    except Exception as e:
        return False, f"Error parsing input: {str(e)}", []


def validate_file_upload(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded file for portfolio import.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (is_valid, error_message)
    """

    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    file_extension = '.' + uploaded_file.name.split('.')[-1].lower()

    if file_extension not in allowed_extensions:
        return False, f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"

    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return False, f"File too large: {uploaded_file.size / (1024*1024):.1f}MB. Maximum: 10MB"

    return True, ""


def generate_sample_data(sample_type: str) -> str:
    """
    Generate sample data for portfolio creation.

    Args:
        sample_type: Type of sample ('tech_focus', 'balanced', 'dividend', etc.)

    Returns:
        Sample text string
    """

    samples = {
        'tech_focus': "AAPL 25%, MSFT 20%, GOOGL 15%, NVDA 15%, META 10%, AMZN 10%, TSLA 5%",
        'balanced': "SPY 40%, BND 25%, VTI 20%, VTIAX 10%, GLD 5%",
        'dividend': "JNJ 15%, PG 15%, KO 10%, PFE 10%, VZ 10%, T 10%, XOM 10%, CVX 10%, IBM 10%",
        'etf': "SPY 30%, QQQ 25%, IWM 15%, EFA 15%, VEA 10%, BND 5%",
        'conservative': "BND 40%, SPY 30%, VTI 15%, VTEB 10%, GLD 5%",
        'growth': "QQQ 35%, SPY 25%, VUG 20%, ARKK 10%, VEA 10%"
    }

    return samples.get(sample_type, samples['balanced'])


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value if division by zero

    Returns:
        Result of division or default value
    """

    if denominator == 0 or pd.isna(denominator):
        return default

    return numerator / denominator


def create_asset_table(assets: List[Asset]) -> pd.DataFrame:
    """
    Create a formatted DataFrame from list of assets.

    Args:
        assets: List of Asset objects

    Returns:
        Formatted pandas DataFrame
    """

    data = []
    for asset in assets:
        data.append({
            'Ticker': asset.ticker,
            'Name': asset.name or 'N/A',
            'Weight': f"{asset.weight:.1%}",
            'Shares': getattr(asset, 'shares', 0),
            'Current Price': f"${getattr(asset, 'current_price', 0):.2f}" if hasattr(asset, 'current_price') else 'N/A',
            'Market Value': f"${asset.market_value:,.2f}" if hasattr(asset, 'market_value') else 'N/A',
            'Sector': getattr(asset, 'sector', 'N/A')
        })

    return pd.DataFrame(data)


def format_currency_short(value: float) -> str:
    """
    Format currency value with K/M/B suffixes.

    Args:
        value: Currency value to format

    Returns:
        Formatted currency string
    """

    if pd.isna(value) or value is None:
        return "$0"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.2f}"


def calculate_portfolio_metrics(portfolio: Portfolio) -> dict:
    """
    Calculate comprehensive portfolio metrics.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with calculated metrics
    """

    metrics = {}

    # Basic metrics
    metrics['total_assets'] = len(portfolio.assets)
    metrics['total_value'] = portfolio.calculate_value()
    metrics['total_weight'] = sum(asset.weight for asset in portfolio.assets)

    # Initialize position metrics
    metrics['largest_position'] = None
    metrics['smallest_position'] = None
    metrics['concentration_risk'] = 0

    # Asset allocation metrics
    if portfolio.assets:
        weights = [asset.weight for asset in portfolio.assets]

        # Concentration metrics
        metrics['max_weight'] = max(weights)
        metrics['min_weight'] = min(weights)
        metrics['weight_std'] = pd.Series(weights).std()

        # Top holdings
        sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
        metrics['top_5_weight'] = sum(asset.weight for asset in sorted_assets[:5])
        metrics['top_10_weight'] = sum(asset.weight for asset in sorted_assets[:10])

        # Largest and smallest positions
        metrics['largest_position'] = {
            'ticker': sorted_assets[0].ticker,
            'weight': sorted_assets[0].weight
        }
        metrics['smallest_position'] = {
            'ticker': sorted_assets[-1].ticker,
            'weight': sorted_assets[-1].weight
        }

        # Concentration risk (sum of top 5 positions)
        metrics['concentration_risk'] = sum(asset.weight for asset in sorted_assets[:5])

        # Sector diversification
        sectors = {}
        for asset in portfolio.assets:
            sector = getattr(asset, 'sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + asset.weight

        metrics['sector_count'] = len(sectors)
        metrics['largest_sector_weight'] = max(sectors.values()) if sectors else 0
        metrics['sector_diversification'] = sectors

    # Value metrics (if prices available)
    total_market_value = 0
    assets_with_prices = 0

    for asset in portfolio.assets:
        if hasattr(asset, 'current_price') and asset.current_price:
            assets_with_prices += 1
            if hasattr(asset, 'shares') and asset.shares:
                total_market_value += asset.shares * asset.current_price

    metrics['assets_with_prices'] = assets_with_prices
    metrics['price_coverage'] = assets_with_prices / len(portfolio.assets) if portfolio.assets else 0
    metrics['current_market_value'] = total_market_value

    return metrics


def get_portfolio_health_score(portfolio: Portfolio) -> dict:
    """
    Calculate portfolio health score and recommendations.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with health score and recommendations
    """

    metrics = calculate_portfolio_metrics(portfolio)
    health_score = 100  # Start with perfect score
    recommendations = []

    # Weight distribution check
    if abs(metrics['total_weight'] - 1.0) > 0.01:
        health_score -= 20
        recommendations.append("Normalize portfolio weights to sum to 100%")

    # Diversification check
    if metrics['max_weight'] > 0.5:
        health_score -= 15
        recommendations.append("Reduce concentration - largest position exceeds 50%")
    elif metrics['max_weight'] > 0.3:
        health_score -= 5
        recommendations.append("Consider reducing largest position (>30%)")

    # Asset count check
    if metrics['total_assets'] < 5:
        health_score -= 10
        recommendations.append("Consider adding more assets for better diversification")
    elif metrics['total_assets'] > 50:
        health_score -= 5
        recommendations.append("Portfolio may be over-diversified - consider consolidating")

    # Price data coverage
    if metrics['price_coverage'] < 0.8:
        health_score -= 10
        recommendations.append("Update price data for better portfolio tracking")

    # Sector diversification
    if metrics['sector_count'] < 3:
        health_score -= 10
        recommendations.append("Add assets from different sectors for better diversification")

    if metrics.get('largest_sector_weight', 0) > 0.6:
        health_score -= 10
        recommendations.append("Reduce sector concentration - largest sector exceeds 60%")

    # Ensure score doesn't go below 0
    health_score = max(0, health_score)

    # Health rating
    if health_score >= 90:
        rating = "Excellent"
    elif health_score >= 80:
        rating = "Good"
    elif health_score >= 70:
        rating = "Fair"
    elif health_score >= 60:
        rating = "Poor"
    else:
        rating = "Critical"

    return {
        'score': health_score,
        'rating': rating,
        'recommendations': recommendations,
        'metrics': metrics
    }


def export_portfolio_data(portfolio: Portfolio, export_format: str = "JSON"):
    """
    Export portfolio data in specified format.

    Args:
        portfolio: Portfolio object to export
        export_format: Format ('JSON', 'CSV', 'Excel')
    """

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{portfolio.name}_{timestamp}"

        if export_format == "JSON":
            # Export as JSON
            portfolio_dict = portfolio.to_dict()
            json_str = json.dumps(portfolio_dict, indent=2, default=str)

            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{filename}.json",
                mime="application/json"
            )

        elif export_format == "CSV":
            # Export assets as CSV
            data = []
            for asset in portfolio.assets:
                data.append({
                    'ticker': asset.ticker,
                    'name': asset.name or '',
                    'weight': asset.weight,
                    'shares': getattr(asset, 'shares', 0),
                    'current_price': getattr(asset, 'current_price', 0),
                    'sector': getattr(asset, 'sector', '')
                })

            df = pd.DataFrame(data)
            csv_str = df.to_csv(index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv_str,
                file_name=f"{filename}.csv",
                mime="text/csv"
            )

        elif export_format == "Excel":
            # Export as Excel
            data = []
            for asset in portfolio.assets:
                data.append({
                    'Ticker': asset.ticker,
                    'Name': asset.name or '',
                    'Weight': asset.weight,
                    'Shares': getattr(asset, 'shares', 0),
                    'Current Price': getattr(asset, 'current_price', 0),
                    'Sector': getattr(asset, 'sector', '')
                })

            df = pd.DataFrame(data)

            # Create Excel file in memory
            excel_buffer = BytesIO()

            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Portfolio', index=False)

                # Add portfolio info sheet
                info_df = pd.DataFrame([
                    ['Name', portfolio.name],
                    ['Description', portfolio.description or ''],
                    ['Type', portfolio.portfolio_type.value],
                    ['Created', portfolio.created_date.strftime('%Y-%m-%d %H:%M:%S')],
                    ['Assets Count', len(portfolio.assets)],
                    ['Total Weight', sum(asset.weight for asset in portfolio.assets)]
                ], columns=['Property', 'Value'])

                info_df.to_excel(writer, sheet_name='Info', index=False)

            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.success(f"Portfolio export ready for download!")

    except Exception as e:
        st.error(f"Export failed: {e}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """

    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text
    """

    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def get_unique_colors(n: int) -> List[str]:
    """
    Generate n unique colors for charts.

    Args:
        n: Number of colors needed

    Returns:
        List of color hex codes
    """

    import plotly.colors as pc

    if n <= 10:
        return pc.qualitative.Set3[:n]
    else:
        # Generate more colors using interpolation
        colors = []
        for i in range(n):
            hue = (i * 360) // n
            colors.append(f"hsl({hue}, 70%, 50%)")
        return colors


def create_metric_card(title: str, value: str, delta: Optional[str] = None,
                      delta_color: str = "normal") -> None:
    """
    Create a styled metric card.

    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta value
        delta_color: Color for delta (normal, inverse)
    """

    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def show_loading_message(message: str = "Loading..."):
    """
    Show a loading message with spinner.

    Args:
        message: Loading message to display
    """

    return st.spinner(message)


def show_success_message(message: str, duration: int = 3):
    """
    Show a success message that auto-disappears.

    Args:
        message: Success message
        duration: Duration in seconds
    """

    success_placeholder = st.success(message)
    time.sleep(duration)
    success_placeholder.empty()


def show_error_message(message: str, details: Optional[str] = None):
    """
    Show an error message with optional details.

    Args:
        message: Error message
        details: Optional detailed error information
    """

    st.error(message)

    if details:
        with st.expander("Error Details"):
            st.code(details)