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
    render_quick_actions()


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


def show_example_portfolios():
    """Show example portfolio configurations."""

    st.subheader("📚 Example Portfolios")

    examples = {
        "Conservative Growth": {
            "description": "Low-risk balanced portfolio",
            "allocation": "SPY 40%, BND 30%, VTI 20%, VTEB 10%",
            "risk_level": "Low",
            "expected_return": "6-8%"
        },
        "Tech Focus": {
            "description": "High-growth technology stocks",
            "allocation": "AAPL 25%, MSFT 20%, GOOGL 15%, NVDA 15%, META 10%, AMZN 10%, TSLA 5%",
            "risk_level": "High",
            "expected_return": "12-16%"
        },
        "Dividend Income": {
            "description": "Income-focused dividend stocks",
            "allocation": "JNJ 15%, PG 15%, KO 10%, PFE 10%, VZ 10%, T 10%, XOM 10%, CVX 10%, IBM 10%",
            "risk_level": "Medium",
            "expected_return": "8-10%"
        }
    }

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
        # Market status (placeholder - would connect to real market API)
        market_status = "Open"  # This would come from market API
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


def get_market_data():
    """Fetch real market data."""

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

        # Risk & Rates (эти уже дают проценты)
        "VIX": "^VIX",
        "US 10Y": "^TNX",  # Уже в процентах (например 4.06%)
        "US 2Y": "^TNX",  # Заглушка, найти правильный тикер
        "Fed Rate": "^IRX",

        # Commodities & Currency (исправить тикеры)
        "DXY": "DX-Y.NYB",  # Попробовать этот
        "Gold": "GC=F",
        "Oil (WTI)": "CL=F",
        "Copper": "HG=F"
    }

    market_data = {}

    try:
        with st.spinner("Fetching market data..."):
            for name, ticker in market_tickers.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="2d")

                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100

                        market_data[name] = {
                            "value": current_price,
                            "change": change,
                            "change_pct": change_pct,
                            "ticker": ticker
                        }
                    else:
                        # Fallback если нет данных
                        market_data[name] = {
                            "value": 0,
                            "change": 0,
                            "change_pct": 0,
                            "ticker": ticker
                        }
                except Exception as e:
                    st.warning(f"Failed to fetch {name}: {str(e)}")
                    market_data[name] = {
                        "value": 0,
                        "change": 0,
                        "change_pct": 0,
                        "ticker": ticker
                    }

    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return {}

    # Вычислить BTC Dominance через API
    try:
        import requests
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
        if response.status_code == 200:
            data = response.json()
            btc_dominance = data.get('data', {}).get('market_cap_percentage', {}).get('btc', 50.0)
            market_data["BTC Dominance"] = {
                "value": btc_dominance,
                "change": 0,  # API не дает изменения
                "change_pct": 0,
                "ticker": "BTC.D"
            }
    except:
        # Fallback для BTC Dominance
        market_data["BTC Dominance"] = {
            "value": 52.5,
            "change": -0.3,
            "change_pct": -0.57,
            "ticker": "BTC.D"
        }

    try:
        if "US 10Y" in market_data and "US 2Y" in market_data:
            yield_10y = market_data["US 10Y"]["value"]

            yield_2y = 3.5

            yield_curve = yield_10y - yield_2y

            market_data["Yield Curve"] = {
                "value": yield_curve,
                "change": 0.02,
                "change_pct": 3.7,
                "ticker": "10Y-2Y"
            }
        else:

            market_data["Yield Curve"] = {
                "value": 0.56,
                "change": 0.02,
                "change_pct": 3.7,
                "ticker": "10Y-2Y"
            }
    except:
        market_data["Yield Curve"] = {
            "value": 0.56,
            "change": 0.02,
            "change_pct": 3.7,
            "ticker": "10Y-2Y"
        }


    try:

        market_data["US Inflation"] = {
            "value": 3.2,
            "change": -0.1,
            "change_pct": -3.03,
            "ticker": "CPI"
        }
    except:
        market_data["US Inflation"] = {
            "value": 3.2,
            "change": -0.1,
            "change_pct": -3.03,
            "ticker": "CPI"
        }

    return market_data


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
            ["1D", "1W", "1M", "3M", "YTD", "1Y"],
            index=2,  # Default to 1M
            key="market_chart_period"
        )

    if selected_assets:
        # Create chart
        fig = create_market_chart(selected_assets, period)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
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
        "1D": "1d",
        "1W": "5d",
        "1M": "1mo",
        "3M": "3mo",
        "YTD": "ytd",
        "1Y": "1y"
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
    """Render simplified portfolio table."""

    st.subheader("📋 Your Portfolios")

    if not portfolios:
        st.info("No portfolios created yet")
        return

    # Create simplified portfolio data
    portfolio_data = []

    for portfolio in portfolios:
        portfolio_data.append({
            'Name': portfolio.name,
            'Type': portfolio.portfolio_type.value.title(),
            'Assets': len(portfolio.assets),
            'Value': format_currency(portfolio.calculate_value()),
            'Created': format_datetime(portfolio.created_date, '%Y-%m-%d'),
            'Description': (portfolio.description[:50] + "...") if portfolio.description and len(portfolio.description) > 50 else (portfolio.description or "No description")
        })

    # Display as dataframe
    df = pd.DataFrame(portfolio_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_quick_actions():
    """Render navigation quick actions."""

    st.subheader("⚡ Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Create Portfolio", width="stretch", type="primary", key="nav_create"):
            # Use query params for navigation instead of session state
            st.query_params.page = "create"
            st.rerun()

    with col2:
        if st.button("📋 Manage Portfolios", width="stretch", key="nav_manage"):
            st.query_params.page = "manage"
            st.rerun()

    with col3:
        if st.button("📊 Portfolio Analysis", width="stretch", key="nav_analysis"):
            st.query_params.page = "analysis"
            st.rerun()

    with col4:
        if st.button("⚙️ System Status", width="stretch", key="nav_system"):
            st.query_params.page = "system"
            st.rerun()


def render_market_news():
    """Render market news section (placeholder)."""

    st.subheader("📰 Market News")

    # Placeholder news items
    news_items = [
        {
            "title": "Fed Holds Interest Rates Steady",
            "summary": "Federal Reserve maintains current rates amid economic uncertainty...",
            "time": "2 hours ago"
        },
        {
            "title": "Tech Stocks Rally on AI Optimism",
            "summary": "Major technology companies see gains as AI adoption accelerates...",
            "time": "4 hours ago"
        },
        {
            "title": "Oil Prices Fluctuate on Supply Concerns",
            "summary": "Crude oil markets react to geopolitical tensions and supply data...",
            "time": "6 hours ago"
        }
    ]

    for item in news_items:
        with st.container():
            st.write(f"**{item['title']}**")
            st.write(item['summary'])
            st.caption(f"📅 {item['time']}")
            st.divider()