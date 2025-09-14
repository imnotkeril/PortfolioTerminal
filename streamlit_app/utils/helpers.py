"""
Helper utilities for the Portfolio Management System.

This module contains miscellaneous helper functions used throughout the application.
"""
import streamlit as st
import pandas as pd
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import Portfolio, Asset


def display_portfolio_summary(portfolio: Portfolio):
    """
    Display a summary card for a portfolio.

    Args:
        portfolio: Portfolio object to summarize
    """

    # Calculate basic metrics
    total_assets = len(portfolio.assets)
    total_value = portfolio.calculate_value()

    # Create columns for layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Assets", total_assets)

    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")

    with col3:
        st.metric("Portfolio Type", portfolio.portfolio_type.value.title())

    with col4:
        created_days_ago = (datetime.now() - portfolio.created_date).days
        st.metric("Age (Days)", created_days_ago)

    # Show asset breakdown
    if portfolio.assets:
        st.subheader("Top Holdings")

        # Sort assets by weight
        sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
        top_assets = sorted_assets[:5]  # Show top 5

        for asset in top_assets:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**{asset.ticker}** - {asset.name or 'N/A'}")

            with col2:
                st.write(f"{asset.weight:.1%}")

            with col3:
                if hasattr(asset, 'current_price') and asset.current_price:
                    st.write(f"${asset.current_price:.2f}")
                else:
                    st.write("N/A")


def update_company_info(portfolio: Portfolio):
    """
    Update company information for portfolio assets.

    Args:
        portfolio: Portfolio object to update
    """

    with st.spinner("Fetching company information..."):
        # This is a placeholder for company info fetching
        # In a real implementation, you would fetch from an API

        company_info = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
            'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology'},
            'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
            'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary'},
            'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive'},
            'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology'},
            'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
            'BRK-B': {'name': 'Berkshire Hathaway Inc.', 'sector': 'Financial Services'},
            'V': {'name': 'Visa Inc.', 'sector': 'Financial Services'},
            'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
        }

        for asset in portfolio.assets:
            if asset.ticker in company_info:
                info = company_info[asset.ticker]
                asset.name = info.get('name', asset.name)
                asset.sector = info.get('sector', getattr(asset, 'sector', None))

        time.sleep(1)  # Simulate API delay


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
                st.success(f"✅ Updated prices for {updated_count} assets")
            else:
                st.info("ℹ️ All prices are up to date")

        except Exception as e:
            st.error(f"Error updating prices: {e}")


def export_portfolio_data(portfolio: Portfolio):
    """
    Export portfolio data in various formats.

    Args:
        portfolio: Portfolio object to export
    """

    st.subheader(f"📤 Export {portfolio.name}")

    # Create export data
    export_data = []
    for asset in portfolio.assets:
        export_data.append({
            'Ticker': asset.ticker,
            'Name': asset.name or 'N/A',
            'Weight': asset.weight,
            'Shares': getattr(asset, 'shares', 0),
            'Current Price': getattr(asset, 'current_price', None),
            'Sector': getattr(asset, 'sector', 'N/A'),
            'Asset Class': asset.asset_class.value if hasattr(asset, 'asset_class') else 'stock'
        })

    df = pd.DataFrame(export_data)

    # Export options
    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"{portfolio.name}_portfolio.csv",
            mime="text/csv"
        )

    with col2:
        # JSON export
        json_data = portfolio.to_dict()
        import json
        json_str = json.dumps(json_data, indent=2, default=str)
        st.download_button(
            label="📋 Download JSON",
            data=json_str,
            file_name=f"{portfolio.name}_portfolio.json",
            mime="application/json"
        )

    with col3:
        # Excel would require openpyxl, show placeholder
        st.button("📊 Excel Export", disabled=True, help="Excel export requires additional setup")

    # Show preview
    st.subheader("📋 Export Preview")
    st.dataframe(df, use_container_width=True)


def validate_ticker_input(text: str) -> tuple[bool, str, List[str]]:
    """
    Validate and parse ticker input text.

    Args:
        text: Raw text input containing tickers and weights

    Returns:
        Tuple of (is_valid, error_message, parsed_tickers)
    """

    if not text or not text.strip():
        return False, "Input text cannot be empty", []

    try:
        # Parse the text - handle different formats
        import re

        # Remove extra whitespace and split by common delimiters
        text = re.sub(r'\s+', ' ', text.strip())

        # Try different parsing patterns
        tickers = []

        # Pattern 1: "AAPL 30%, MSFT 25%, GOOGL 45%"
        percentage_pattern = r'([A-Z.-]+)\s+(\d+(?:\.\d+)?)%'
        matches = re.findall(percentage_pattern, text)

        if matches:
            for ticker, weight in matches:
                tickers.append((ticker.upper(), float(weight) / 100))

        # Pattern 2: "AAPL 0.30, MSFT 0.25, GOOGL 0.45"
        elif ',' in text and any(char.isdigit() for char in text):
            decimal_pattern = r'([A-Z.-]+)\s+(\d*\.?\d+)'
            matches = re.findall(decimal_pattern, text)

            if matches:
                for ticker, weight in matches:
                    weight_val = float(weight)
                    # If weight > 1, assume it's percentage
                    if weight_val > 1:
                        weight_val = weight_val / 100
                    tickers.append((ticker.upper(), weight_val))

        # Pattern 3: "AAPL,MSFT,GOOGL" (equal weights)
        elif ',' in text and not any(char.isdigit() for char in text):
            ticker_list = [t.strip().upper() for t in text.split(',') if t.strip()]
            if ticker_list:
                equal_weight = 1.0 / len(ticker_list)
                tickers = [(ticker, equal_weight) for ticker in ticker_list]

        # Pattern 4: "AAPL MSFT GOOGL" (space separated, equal weights)
        else:
            ticker_list = [t.strip().upper() for t in text.split() if t.strip() and t.isalpha()]
            if ticker_list:
                equal_weight = 1.0 / len(ticker_list)
                tickers = [(ticker, equal_weight) for ticker in ticker_list]

        if not tickers:
            return False, "Could not parse any valid tickers from input", []

        # Validate weights sum to approximately 1
        total_weight = sum(weight for _, weight in tickers)
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            return False, f"Weights sum to {total_weight:.2%}, must sum to 100%", []

        # Validate ticker symbols (basic validation)
        invalid_tickers = []
        for ticker, _ in tickers:
            if not ticker or len(ticker) > 10 or not ticker.replace('-', '').replace('.', '').isalpha():
                invalid_tickers.append(ticker)

        if invalid_tickers:
            return False, f"Invalid ticker symbols: {', '.join(invalid_tickers)}", []

        return True, "", tickers

    except Exception as e:
        return False, f"Error parsing input: {str(e)}", []


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


def create_asset_table(assets: List[Asset]) -> pd.DataFrame:
    """
    Create a formatted DataFrame from assets list.

    Args:
        assets: List of Asset objects

    Returns:
        Formatted DataFrame
    """

    if not assets:
        return pd.DataFrame()

    data = []
    for asset in assets:
        row = {
            'Ticker': asset.ticker,
            'Name': asset.name or 'N/A',
            'Weight': asset.weight,
            'Shares': getattr(asset, 'shares', 0),
            'Current Price': getattr(asset, 'current_price', None),
            'Market Value': getattr(asset, 'current_price', 0) * getattr(asset, 'shares', 0) if hasattr(asset, 'current_price') and hasattr(asset, 'shares') else 0,
            'Sector': getattr(asset, 'sector', 'N/A'),
            'Asset Class': asset.asset_class.value if hasattr(asset, 'asset_class') else 'stock'
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by weight descending
    df = df.sort_values('Weight', ascending=False)

    return df


def calculate_portfolio_metrics(portfolio: Portfolio) -> Dict[str, Any]:
    """
    Calculate key portfolio metrics.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with calculated metrics
    """

    metrics = {
        'total_assets': len(portfolio.assets),
        'total_weight': sum(asset.weight for asset in portfolio.assets),
        'total_value': portfolio.calculate_value(),
        'largest_position': None,
        'smallest_position': None,
        'concentration_risk': 0,
        'sector_diversification': {},
    }

    if portfolio.assets:
        # Find largest and smallest positions
        sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
        metrics['largest_position'] = {
            'ticker': sorted_assets[0].ticker,
            'weight': sorted_assets[0].weight
        }
        metrics['smallest_position'] = {
            'ticker': sorted_assets[-1].ticker,
            'weight': sorted_assets[-1].weight
        }

        # Calculate concentration risk (sum of top 5 positions)
        top_5_weights = sum(asset.weight for asset in sorted_assets[:5])
        metrics['concentration_risk'] = top_5_weights

        # Sector diversification
        sector_weights = {}
        for asset in portfolio.assets:
            sector = getattr(asset, 'sector', 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + asset.weight

        metrics['sector_diversification'] = sector_weights

    return metrics


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


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.

    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """

    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


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


def validate_file_upload(uploaded_file) -> tuple[bool, str]:
    """
    Validate an uploaded file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (is_valid, error_message)
    """

    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return False, f"File too large: {format_file_size(uploaded_file.size)}. Maximum size is 10MB."

    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    file_extension = '.' + uploaded_file.name.split('.')[-1].lower()

    if file_extension not in allowed_extensions:
        return False, f"Invalid file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"

    return True, ""


def generate_sample_data() -> str:
    """
    Generate sample ticker input for demonstration.

    Returns:
        Sample ticker string
    """

    samples = [
        "AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%",
        "AAPL 0.25, MSFT 0.20, GOOGL 0.15, NVDA 0.15, META 0.10, V 0.10, JNJ 0.05",
        "AAPL, MSFT, GOOGL, AMZN, TSLA",
        "SPY 40%, QQQ 30%, IWM 20%, EFA 10%"
    ]

    import random
    return random.choice(samples)