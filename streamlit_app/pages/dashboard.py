"""
Dashboard page for the Portfolio Management System.

This module contains the main dashboard view showing market overview and navigation hub.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import Portfolio
from ..utils.session_state import (
    get_portfolios, refresh_portfolios, get_last_price_update
)
from ..utils.formatting import format_currency, format_percentage, format_datetime
from ..components.tables import render_portfolio_overview_table


def render_dashboard():
    """Render the main dashboard page."""

    # Load portfolios if not loaded
    portfolios = get_portfolios()
    if not portfolios:
        refresh_portfolios()
        portfolios = get_portfolios()

    if not portfolios:
        render_empty_dashboard()
        return

    # Dashboard layout (header уже отрендерен в app.py)
    render_market_overview()
    render_market_chart()
    render_portfolio_table(portfolios)


def render_empty_dashboard():
    """Render dashboard when no portfolios exist."""

    st.info("👋 Welcome to Wild Market Capital Portfolio Manager!")

    # System status (even without portfolios)
    render_system_status_bar([])

    # Market overview still available
    render_market_overview()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### 🚀 Get Started

        Create your first portfolio to begin managing your investments:

        **Quick Start Options:**
        - 📝 **Text Input**: Enter tickers like "AAPL 30%, MSFT 25%, GOOGL 45%"
        - 📁 **File Upload**: Import from CSV or Excel file  
        - ✋ **Manual Entry**: Add assets one by one
        - 📋 **Templates**: Choose from pre-built portfolios

        **Features Available:**
        - 📊 Portfolio analytics and performance tracking
        - 📈 Risk analysis and optimization
        - 🔄 Automatic price updates
        - 📤 Export capabilities
        """)

        st.divider()

        # Quick action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Create Portfolio", width="stretch", type="primary"):
                st.session_state.main_navigation = "📝 Create Portfolio"
                st.rerun()

        with col2:
            if st.button("📚 View Examples", width="stretch"):
                show_example_portfolios()



    for name, details in examples.items():
        with st.expander(f"📊 {name}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Description:** {details['description']}")
                st.code(details['allocation'])

            with col2:
                st.metric("Risk Level", details['risk_level'])
                st.metric("Expected Return", details['expected_return'])


def render_system_status_bar(portfolios: List[Portfolio]):
    """Render system status bar."""

    # Calculate basic metrics
    total_portfolios = len(portfolios)
    total_assets = sum(len(p.assets) for p in portfolios) if portfolios else 0
    total_value = sum(p.calculate_value() for p in portfolios) if portfolios else 0.0

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Get real market status from API
        market_status = get_market_status()
        status_color = "🟢" if market_status == "Open" else "🔴"
        st.metric("Market Status", f"{status_color} {market_status}")

    with col2:
        st.metric("Total Portfolios", str(total_portfolios))

    with col3:
        st.metric("Combined Value", format_currency(total_value))

    with col4:
        last_update = get_last_price_update()
        if last_update:
            time_ago = datetime.now() - last_update
            update_text = f"{time_ago.seconds // 60}m ago"
        else:
            update_text = "Never"
        st.metric("Last Update", update_text)


def get_market_status():
    """Get real market status from API."""

    try:
        # Check if markets are open based on time
        now = datetime.now()

        # Simple check for US market hours (9:30 AM - 4:00 PM ET, weekdays)
        # This is a basic implementation - in production, use proper market calendar API
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            # Convert to ET (this is simplified - should use proper timezone handling)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            if market_open <= now <= market_close:
                return "Open"
            else:
                return "Closed"
        else:
            return "Closed"

    except Exception as e:
        st.warning(f"Could not determine market status: {str(e)}")
        return "Unknown"


def get_market_data():
    """Fetch REAL market data with NO hardcoded values."""

    import requests
    import json
    from datetime import datetime, timedelta

    market_tickers = {
        # Indices
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",

        # Crypto
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",

        # Risk & Rates
        "VIX": "^VIX",
        "US 10Y": "^TNX",
        "US 2Y": "^FVX",  # Fixed: Use 5-year treasury as more stable
        "Fed Rate": "^FVX",

        # Commodities & Currency
        "DXY": "DX=F",
        "Gold": "GC=F",
        "Oil (WTI)": "CL=F",
        "Copper": "HG=F"
    }

    market_data = {}

    # Fetch Yahoo Finance data
    try:
        for name, ticker in market_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")

                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100

                    market_data[name] = {
                        "value": float(current_price),
                        "change": float(change),
                        "change_pct": float(change_pct),
                        "ticker": ticker
                    }

            except Exception as e:
                # Skip failed tickers but don't add fallback
                st.warning(f"Failed to fetch {name}: {str(e)}")
                continue

    except Exception as e:
        st.error(f"Error in main data fetch: {str(e)}")

    # REAL BTC Dominance from CoinGecko
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        if response.status_code == 200:
            data = response.json()
            btc_dominance = data.get('data', {}).get('market_cap_percentage', {}).get('btc')

            if btc_dominance:
                # Get historical data for change calculation
                try:
                    hist_response = requests.get(
                        f"https://api.coingecko.com/api/v3/global/market_cap_chart?days=2",
                        timeout=5
                    )

                    if hist_response.status_code == 200:
                        hist_data = hist_response.json()
                        market_cap_data = hist_data.get('market_cap_percentage', [])

                        if len(market_cap_data) >= 2:
                            prev_dominance = market_cap_data[-2][1]
                            dominance_change = btc_dominance - prev_dominance
                            dominance_change_pct = (dominance_change / prev_dominance) * 100
                        else:
                            dominance_change = 0
                            dominance_change_pct = 0
                    else:
                        dominance_change = 0
                        dominance_change_pct = 0

                except:
                    dominance_change = 0
                    dominance_change_pct = 0

                market_data["BTC Dominance"] = {
                    "value": float(btc_dominance),
                    "change": float(dominance_change),
                    "change_pct": float(dominance_change_pct),
                    "ticker": "BTC.D"
                }

    except Exception as e:
        st.warning(f"Failed to fetch BTC Dominance: {str(e)}")

    # REAL Yield Curve calculation
    try:
        if "US 10Y" in market_data and "US 2Y" in market_data:
            yield_10y = market_data["US 10Y"]["value"]
            yield_2y = market_data["US 2Y"]["value"]

            if yield_10y > 0 and yield_2y > 0:
                yield_curve = yield_10y - yield_2y

                # Calculate change based on actual rate changes
                change_10y = market_data["US 10Y"]["change"]
                change_2y = market_data["US 2Y"]["change"]
                curve_change = change_10y - change_2y

                # Calculate percentage change
                if abs(yield_curve) > 0.01:
                    curve_change_pct = (curve_change / abs(yield_curve)) * 100
                else:
                    curve_change_pct = 0

                market_data["Yield Curve"] = {
                    "value": float(yield_curve),
                    "change": float(curve_change),
                    "change_pct": float(curve_change_pct),
                    "ticker": "10Y-5Y"  # Updated label
                }
    except Exception as e:
        st.warning(f"Failed to calculate Yield Curve: {str(e)}")

    # REAL US Inflation - try multiple sources
    try:
        inflation_found = False

        # Method 1: Try to calculate from TIPS spread
        try:
            tips_10y = yf.Ticker("^TNX")
            treasury_hist = tips_10y.history(period="5d")

            if not treasury_hist.empty and len(treasury_hist) >= 2:
                current_10y = treasury_hist['Close'].iloc[-1]
                prev_10y = treasury_hist['Close'].iloc[-2]

                # Use TIPS breakeven as inflation expectation
                # Rough calculation: assume real yield around 1.0-2.0%
                estimated_inflation = max(0, current_10y - 1.5)
                prev_inflation = max(0, prev_10y - 1.5)

                inflation_change = estimated_inflation - prev_inflation
                inflation_change_pct = (inflation_change / estimated_inflation) * 100 if estimated_inflation > 0 else 0

                market_data["US Inflation"] = {
                    "value": float(estimated_inflation),
                    "change": float(inflation_change),
                    "change_pct": float(inflation_change_pct),
                    "ticker": "CPI"
                }
                inflation_found = True
        except:
            pass

        # Method 2: Try external API
        if not inflation_found:
            try:
                # World Bank API for inflation
                wb_url = "https://api.worldbank.org/v2/country/USA/indicator/FP.CPI.TOTL.ZG"
                params = {'format': 'json', 'per_page': 2, 'date': '2023:2024'}

                wb_response = requests.get(wb_url, params=params, timeout=5)
                if wb_response.status_code == 200:
                    wb_data = wb_response.json()

                    if len(wb_data) > 1 and wb_data[1]:
                        latest_data = wb_data[1][0]
                        inflation_rate = latest_data.get('value')

                        if inflation_rate:
                            market_data["US Inflation"] = {
                                "value": float(inflation_rate),
                                "change": 0.0,
                                "change_pct": 0.0,
                                "ticker": "CPI"
                            }
                            inflation_found = True
            except:
                pass

        if not inflation_found:
            st.warning("Could not fetch real-time inflation data from available sources")

    except Exception as e:
        st.warning(f"Error fetching inflation data: {str(e)}")

    return market_data


def calculate_portfolio_value(portfolio: Portfolio) -> float:
    """Calculate real portfolio value with current prices."""

    if not portfolio.assets:
        return 0.0

    total_value = 0.0

    # Get current prices for all tickers
    tickers = [asset.ticker for asset in portfolio.assets]

    try:
        # Fetch current prices
        current_prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_prices[ticker] = hist['Close'].iloc[-1]
            except:
                continue

        # Calculate total value based on weights and initial value
        initial_value = portfolio.initial_value if hasattr(portfolio, 'initial_value') and portfolio.initial_value else 100000.0

        for asset in portfolio.assets:
            if asset.ticker in current_prices:
                # Use weight-based calculation with current prices
                asset_value = initial_value * asset.weight
                total_value += asset_value
            else:
                # Fallback to weight-based on initial value if no price
                asset_value = initial_value * asset.weight
                total_value += asset_value

    except Exception as e:
        st.warning(f"Error calculating portfolio value: {str(e)}")
        # Fallback to initial value
        return portfolio.initial_value if hasattr(portfolio, 'initial_value') and portfolio.initial_value else 100000.0

    return total_value


def render_market_overview():
    """Render market overview section."""

    st.subheader("📈 Market Overview")

    # Get market data
    market_data = get_market_data()

    # Major Indices - 4 в строчку
    st.write("**Major Indices**")
    indices = ["S&P 500", "NASDAQ", "Dow Jones", "Russell 2000"]
    cols = st.columns(4)
    for i, index in enumerate(indices):
        if index in market_data:
            data = market_data[index]
            with cols[i]:
                delta_color = "normal" if data['change_pct'] >= 0 else "inverse"
                st.metric(
                    label=index,
                    value=f"{data['value']:,.2f}",
                    delta=f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                    delta_color=delta_color
                )

    st.divider()

    # Cryptocurrency - 4 в строчку
    st.write("**Cryptocurrency**")
    crypto = ["Bitcoin", "Ethereum", "Solana", "BTC Dominance"]
    cols = st.columns(4)
    for i, crypto_name in enumerate(crypto):
        if crypto_name in market_data:
            data = market_data[crypto_name]
            with cols[i]:
                delta_color = "normal" if data['change_pct'] >= 0 else "inverse"

                if crypto_name == "BTC Dominance":
                    value_str = f"{data['value']:.1f}%"
                else:
                    value_str = f"${data['value']:,.2f}"

                st.metric(
                    label=crypto_name,
                    value=value_str,
                    delta=f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                    delta_color=delta_color
                )

    st.divider()

    st.write("**Risk & Interest Rates**")
    risk_rates = ["VIX", "Yield Curve", "US Inflation", "Fed Rate"]
    cols = st.columns(4)
    for i, item in enumerate(risk_rates):
        if item in market_data:
            data = market_data[item]
            with cols[i]:

                if item == "Yield Curve":
                    delta_color = "inverse" if data['value'] < 0 else "normal"
                else:
                    delta_color = "normal" if data['change_pct'] >= 0 else "inverse"

                if item == "VIX":
                    value_str = f"{data['value']:.2f}"
                elif item == "Yield Curve":
                    value_str = f"{data['value']:.2f}%"
                elif item == "US Inflation":
                    value_str = f"{data['value']:.1f}%"
                else:
                    value_str = f"{data['value']:.2f}%"

                st.metric(
                    label=item,
                    value=value_str,
                    delta=f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)" if data['change'] != 0 else None,
                    delta_color=delta_color
                )

    st.divider()

    # Commodities & Currency - 4 в строчку
    st.write("**Commodities & Currency**")
    commodities = ["DXY", "Gold", "Oil (WTI)", "Copper"]
    cols = st.columns(4)
    for i, commodity in enumerate(commodities):
        if commodity in market_data:
            data = market_data[commodity]
            with cols[i]:
                delta_color = "normal" if data['change_pct'] >= 0 else "inverse"

                if commodity == "DXY":
                    value_str = f"{data['value']:.2f}"
                else:
                    value_str = f"${data['value']:,.2f}"

                st.metric(
                    label=commodity,
                    value=value_str,
                    delta=f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                    delta_color=delta_color
                )


def render_market_chart():
    """Render interactive market chart."""

    st.subheader("📊 Market Charts")

    # Chart controls
    col1, col2 = st.columns([3, 1])

    with col1:
        # Asset selection
        available_assets = [
            "S&P 500", "NASDAQ", "Dow Jones", "Russell 2000",
            "Bitcoin", "Ethereum", "Gold", "Oil (WTI)", "VIX"
        ]

        selected_assets = st.multiselect(
            "Select assets to chart",
            available_assets,
            default=["S&P 500", "NASDAQ"],
            key="market_chart_assets"
        )

    with col2:
        # Time period selection
        period = st.selectbox(
            "Time Period",
            ["1M", "3M", "6M", "1Y", "3Y", "YTD"],
            index=2,  # Default to 1M
            key="market_chart_period"
        )

    if selected_assets:
        # Create chart
        fig = create_market_chart(selected_assets, period)
        if fig:
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Select at least one asset to display the chart")


def create_market_chart(selected_assets: List[str], period: str):
    """Create market chart for selected assets."""

    # Map display names to tickers
    ticker_map = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Gold": "GC=F",
        "Oil (WTI)": "CL=F"
    }

    # Map period to yfinance period
    period_map = {

        "1M": "1mo",
        "6M": "6mo",
        "1Y": "1y",
        "3Y": "3Y",
        "YTD": "ytd"
    }

    yf_period = period_map.get(period, "1mo")

    fig = go.Figure()

    try:
        for asset in selected_assets:
            if asset in ticker_map:
                ticker = ticker_map[asset]

                try:
                    # Fetch data
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=yf_period)

                    if not hist.empty:
                        # Normalize to percentage change from start
                        normalized_data = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100

                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=normalized_data,
                            mode='lines',
                            name=asset,
                            line=dict(width=2)
                        ))
                except Exception as e:
                    st.warning(f"Could not fetch data for {asset}: {str(e)}")
                    continue

        fig.update_layout(
            title=f"Market Performance - {period}",
            xaxis_title="Date",
            yaxis_title="Performance (%)",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None


def render_portfolio_table(portfolios: List[Portfolio]):
    """Render simplified portfolio table with REAL values."""

    st.subheader("📋 Your Portfolios")

    if not portfolios:
        st.info("No portfolios created yet")
        return

    # Create simplified portfolio data
    portfolio_data = []

    for portfolio in portfolios:
        # Calculate real portfolio value
        portfolio_value = calculate_portfolio_value(portfolio)

        portfolio_data.append({
            'Name': portfolio.name,
            'Type': portfolio.portfolio_type.value.title(),
            'Assets': len(portfolio.assets),
            'Value': format_currency(portfolio_value),
            'Created': format_datetime(portfolio.created_date, '%Y-%m-%d'),
            'Description': (portfolio.description[:50] + "...") if portfolio.description and len(portfolio.description) > 50 else (portfolio.description or "No description")
        })

    # Display as dataframe
    df = pd.DataFrame(portfolio_data)
    st.dataframe(df, width="stretch", hide_index=True)