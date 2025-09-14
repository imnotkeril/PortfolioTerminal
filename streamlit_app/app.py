"""
Main Streamlit Application for Portfolio Management System.

This is the entry point for the web interface implementing the design
specification with TradingView-inspired styling and comprehensive functionality.
"""
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import time

# Add core module to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# Import core modules
from core.data_manager import (
    PortfolioManager, PriceManager, Portfolio, Asset, AssetClass,
    PortfolioType, ValidationResult, get_validation_summary
)

# ================================
# PAGE CONFIGURATION
# ================================

st.set_page_config(
    page_title="WMC Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.wildmarketcapital.com',
        'Report a bug': 'https://github.com/wmc/portfolio-manager/issues',
        'About': """
        # Wild Market Capital Portfolio Manager

        Professional-grade portfolio management and analytics platform.

        **Version:** 1.0.0  
        **Build:** Phase 1 - Data Foundation
        """
    }
)


# ================================
# CUSTOM CSS STYLING
# ================================

def load_custom_css():
    """Load custom CSS based on TradingView design specification"""
    st.markdown("""
    <style>
    /* Main theme colors from specification */
    :root {
        --bg-color: #0D1015;
        --primary-color: #BF9FFB;
        --text-color: #FFFFFF;
        --positive-color: #74F174;
        --negative-color: #FAA1A4;
        --neutral-color: #D1D4DC;
        --blue-accent: #90BFF9;
        --yellow-accent: #FFF59D;
        --border-color: #2A2E39;
        --crosshair-color: #9598A1;
    }

    /* Dark theme background */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, var(--bg-color) 0%, var(--primary-color) 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
    }

    /* Metric cards */
    .metric-card {
        background-color: var(--bg-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Positive/negative colors */
    .positive {
        color: var(--positive-color);
    }

    .negative {
        color: var(--negative-color);
    }

    /* Success messages */
    .stSuccess {
        background-color: rgba(116, 241, 116, 0.1);
        border: 1px solid var(--positive-color);
    }

    /* Error messages */
    .stError {
        background-color: rgba(250, 161, 164, 0.1);
        border: 1px solid var(--negative-color);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-color);
        border-right: 1px solid var(--border-color);
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: var(--bg-color);
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--blue-accent);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(191, 159, 251, 0.3);
    }

    /* DataFrames */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Hide hamburger menu */
    .css-14xtw13.e8zbici0 {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


# Apply custom CSS
load_custom_css()


# ================================
# SESSION STATE INITIALIZATION
# ================================

def initialize_session_state():
    """Initialize session state variables"""

    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()

    if 'price_manager' not in st.session_state:
        st.session_state.price_manager = PriceManager()

    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = []

    if 'selected_portfolio' not in st.session_state:
        st.session_state.selected_portfolio = None

    if 'last_price_update' not in st.session_state:
        st.session_state.last_price_update = None


# Initialize session state
initialize_session_state()


# ================================
# UTILITY FUNCTIONS
# ================================

def format_currency(value: float) -> str:
    """Format value as currency"""
    if pd.isna(value) or value is None:
        return "$0.00"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    if pd.isna(value) or value is None:
        return "0.00%"
    return f"{value * 100:.2f}%"


def get_color_class(value: float) -> str:
    """Get CSS class for positive/negative values"""
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return ""


def refresh_portfolios():
    """Refresh portfolio list from storage"""
    try:
        st.session_state.portfolios = st.session_state.portfolio_manager.list_portfolios()
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")
        st.session_state.portfolios = []


# ================================
# HEADER SECTION
# ================================

def render_header():
    """Render main application header"""

    st.markdown("""
    <div class="main-header">
        <h1>🚀 Wild Market Capital - Portfolio Manager</h1>
        <p>Professional Portfolio Management & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Market status indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        market_status = st.session_state.price_manager.get_market_status()
        status_color = "🟢" if market_status.is_open else "🔴"
        st.metric("Market Status", f"{status_color} {'Open' if market_status.is_open else 'Closed'}")

    with col2:
        portfolio_count = st.session_state.portfolio_manager.get_portfolio_count()
        st.metric("Total Portfolios", portfolio_count)

    with col3:
        cache_stats = st.session_state.price_manager.get_cache_stats()
        if cache_stats.get('cache_enabled'):
            st.metric("Cache Hits", f"{cache_stats.get('valid_entries', 0)}")
        else:
            st.metric("Cache", "Disabled")

    with col4:
        if st.session_state.last_price_update:
            time_ago = datetime.now() - st.session_state.last_price_update
            st.metric("Last Update", f"{time_ago.seconds // 60}m ago")
        else:
            st.metric("Last Update", "Never")


# ================================
# SIDEBAR NAVIGATION
# ================================

def render_sidebar():
    """Render sidebar navigation"""

    with st.sidebar:
        st.title("📊 Navigation")

        # Main navigation
        page = st.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "📝 Create Portfolio",
                "📋 Manage Portfolios",
                "📊 Portfolio Analysis",
                "⚙️ System Status"
            ],
            key="main_navigation"
        )

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            refresh_portfolios()
            st.session_state.price_manager.clear_cache()
            st.rerun()

        if st.button("📥 Import Portfolio", use_container_width=True):
            st.session_state.main_navigation = "📝 Create Portfolio"
            st.rerun()

        st.divider()

        # Portfolio selector
        if st.session_state.portfolios:
            st.subheader("Select Portfolio")
            portfolio_names = ["None"] + [p.name for p in st.session_state.portfolios]

            selected_name = st.selectbox(
                "Current Portfolio",
                portfolio_names,
                key="portfolio_selector"
            )

            if selected_name != "None":
                selected_portfolio = next(
                    (p for p in st.session_state.portfolios if p.name == selected_name),
                    None
                )
                st.session_state.selected_portfolio = selected_portfolio

                if selected_portfolio:
                    st.success(f"✅ {selected_portfolio.name}")
                    st.caption(f"Assets: {len(selected_portfolio.assets)}")
                    st.caption(f"Value: {format_currency(selected_portfolio.calculate_value())}")
            else:
                st.session_state.selected_portfolio = None
        else:
            st.info("No portfolios created yet")

    return page


# ================================
# PAGE: DASHBOARD
# ================================

def render_dashboard():
    """Render main dashboard page"""

    st.header("🏠 Portfolio Dashboard")

    # Load portfolios if not loaded
    if not st.session_state.portfolios:
        refresh_portfolios()

    if not st.session_state.portfolios:
        st.info("👋 Welcome! Create your first portfolio to get started.")

        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("📝 Create First Portfolio", type="primary", use_container_width=True):
                st.session_state.main_navigation = "📝 Create Portfolio"
                st.rerun()

        return

    # Portfolio overview cards
    st.subheader("📊 Portfolio Overview")

    total_portfolios = len(st.session_state.portfolios)
    total_assets = sum(len(p.assets) for p in st.session_state.portfolios)
    total_value = sum(p.calculate_value() for p in st.session_state.portfolios)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Portfolios", total_portfolios)

    with col2:
        st.metric("Total Assets", total_assets)

    with col3:
        st.metric("Total Value", format_currency(total_value))

    with col4:
        # Calculate simple return (placeholder)
        st.metric("Est. Return", "+8.5%", delta="+1.2%")

    # Portfolio list
    st.subheader("📋 Your Portfolios")

    # Create portfolio summary data
    portfolio_data = []
    for portfolio in st.session_state.portfolios:
        stats = portfolio.get_statistics()

        portfolio_data.append({
            'Name': portfolio.name,
            'Type': portfolio.portfolio_type.value.title(),
            'Assets': len(portfolio.assets),
            'Value': format_currency(stats.total_value),
            'P&L': format_currency(stats.unrealized_pnl),
            'P&L %': format_percentage(stats.unrealized_pnl_percent),
            'Created': portfolio.created_date.strftime('%Y-%m-%d'),
            'Last Modified': portfolio.last_modified.strftime('%Y-%m-%d %H:%M')
        })

    if portfolio_data:
        df = pd.DataFrame(portfolio_data)

        # Display with custom styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Name': st.column_config.TextColumn('Portfolio Name', width="medium"),
                'Type': st.column_config.TextColumn('Type', width="small"),
                'Assets': st.column_config.NumberColumn('Assets', format="%d"),
                'Value': st.column_config.TextColumn('Value', width="medium"),
                'P&L': st.column_config.TextColumn('P&L', width="medium"),
                'P&L %': st.column_config.TextColumn('P&L %', width="small"),
            }
        )

        # Action buttons
        st.subheader("🔧 Portfolio Actions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📊 Analyze Selected", use_container_width=True):
                if st.session_state.selected_portfolio:
                    st.session_state.main_navigation = "📊 Portfolio Analysis"
                    st.rerun()
                else:
                    st.warning("Please select a portfolio first")

        with col2:
            if st.button("💰 Update Prices", use_container_width=True):
                update_all_prices()

        with col3:
            if st.button("📄 Export Data", use_container_width=True):
                if st.session_state.selected_portfolio:
                    export_portfolio_data()
                else:
                    st.warning("Please select a portfolio first")

        with col4:
            if st.button("📝 Create New", use_container_width=True):
                st.session_state.main_navigation = "📝 Create Portfolio"
                st.rerun()


# ================================
# PAGE: CREATE PORTFOLIO
# ================================

def render_create_portfolio():
    """Render portfolio creation page"""

    st.header("📝 Create New Portfolio")

    # Creation method tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Text Input",
        "📁 CSV/Excel Upload",
        "✋ Manual Entry",
        "🎯 Templates"
    ])

    with tab1:
        render_text_input_creation()

    with tab2:
        render_file_upload_creation()

    with tab3:
        render_manual_creation()

    with tab4:
        render_template_creation()


def render_text_input_creation():
    """Text input portfolio creation"""

    st.subheader("📝 Create from Text")

    # Instructions
    with st.expander("📖 Supported Formats", expanded=False):
        st.markdown("""
        **Supported input formats:**

        1. **Percentage format:** `AAPL 30%, MSFT 25%, GOOGL 45%`
        2. **Decimal format:** `AAPL 0.30, MSFT 0.25, GOOGL 0.45`
        3. **Colon format:** `AAPL:30 MSFT:25 GOOGL:45`
        4. **Equal weight:** `AAPL, MSFT, GOOGL` (automatically assigns equal weights)

        **Examples:**
        - `AAPL 20%, MSFT 15%, GOOGL 10%, AMZN 25%, TSLA 30%`
        - `SPY 60%, BND 30%, GLD 10%`
        - `AAPL, MSFT, GOOGL, AMZN, TSLA` (20% each)
        """)

    # Input form
    with st.form("text_input_form"):
        portfolio_name = st.text_input(
            "Portfolio Name *",
            placeholder="e.g., Tech Growth Portfolio",
            help="Enter a unique name for your portfolio"
        )

        portfolio_description = st.text_area(
            "Description (Optional)",
            placeholder="Brief description of your investment strategy...",
            height=100
        )

        text_input = st.text_area(
            "Enter Tickers and Weights *",
            placeholder="AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%",
            height=150,
            help="Enter ticker symbols with weights in any supported format"
        )

        # Portfolio settings
        col1, col2 = st.columns(2)

        with col1:
            portfolio_type = st.selectbox(
                "Portfolio Type",
                options=[t.value for t in PortfolioType],
                format_func=lambda x: x.title()
            )

        with col2:
            initial_value = st.number_input(
                "Initial Value ($)",
                min_value=1000.0,
                value=100000.0,
                step=1000.0,
                format="%.0f"
            )

        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                auto_normalize = st.checkbox("Auto-normalize weights", value=True)
                fetch_company_info = st.checkbox("Fetch company information", value=True)

            with col2:
                min_position_size = st.number_input("Min Position Size ($)", value=1000.0)
                max_position_size = st.number_input("Max Position Size ($)", value=50000.0)

        # Submit button
        submit_button = st.form_submit_button(
            "🚀 Create Portfolio",
            type="primary",
            use_container_width=True
        )

        if submit_button:
            create_portfolio_from_text(
                portfolio_name,
                portfolio_description,
                text_input,
                portfolio_type,
                initial_value,
                auto_normalize,
                fetch_company_info
            )


def create_portfolio_from_text(
        name: str,
        description: str,
        text: str,
        portfolio_type: str,
        initial_value: float,
        auto_normalize: bool,
        fetch_info: bool
):
    """Create portfolio from text input"""

    if not name or not text:
        st.error("Portfolio name and ticker input are required")
        return

    try:
        with st.spinner("Creating portfolio..."):
            # Create portfolio using manager
            portfolio = st.session_state.portfolio_manager.create_from_text(
                name=name,
                text=text,
                description=description,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Calculate shares for each asset based on initial_value
            for asset in portfolio.assets:
                if asset.current_price and asset.current_price > 0:
                    allocation = asset.weight * initial_value
                    asset.shares = int(allocation / asset.current_price)  # Округляем вниз

            # Fetch company information if requested
            if fetch_info:
                update_company_info(portfolio)

            # Refresh portfolio list
            refresh_portfolios()

            # Set as selected portfolio
            st.session_state.selected_portfolio = portfolio

            st.success(f"✅ Portfolio '{name}' created successfully!")

            # Show portfolio summary
            display_portfolio_summary(portfolio)

        # Calculate shares and set current prices for each asset
        with st.spinner("Fetching prices and calculating shares..."):
            tickers = [asset.ticker for asset in portfolio.assets]
            prices = st.session_state.price_manager.get_current_prices(tickers)

            for asset in portfolio.assets:
                if asset.ticker in prices and prices[asset.ticker]:
                    asset.current_price = prices[asset.ticker]
                    allocation = asset.weight * initial_value
                    asset.shares = int(allocation / asset.current_price)

        # Calculate shares for each asset
        for asset in portfolio.assets:
            if hasattr(asset, 'current_price') and asset.current_price and asset.current_price > 0:
                allocation = asset.weight * initial_value
                asset.shares = int(allocation / asset.current_price)
            else:
                asset.shares = 0

    except Exception as e:
        st.error(f"Error creating portfolio: {str(e)}")


def render_file_upload_creation():
    """File upload portfolio creation"""

    st.subheader("📁 Import from File")

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with ticker symbols and weights"
    )

    if uploaded_file is not None:
        try:
            # Show file info
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")

            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Show preview
            st.subheader("📋 File Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column mapping
            st.subheader("🗂️ Column Mapping")

            col1, col2, col3 = st.columns(3)

            with col1:
                ticker_column = st.selectbox(
                    "Ticker Column",
                    options=df.columns.tolist(),
                    help="Column containing ticker symbols"
                )

            with col2:
                weight_column = st.selectbox(
                    "Weight Column",
                    options=["None"] + df.columns.tolist(),
                    help="Column containing weights (optional for equal weighting)"
                )

            with col3:
                name_column = st.selectbox(
                    "Name Column (Optional)",
                    options=["None"] + df.columns.tolist(),
                    help="Column containing company names"
                )

            # Portfolio details
            with st.form("file_import_form"):
                portfolio_name = st.text_input("Portfolio Name *")
                portfolio_description = st.text_area("Description")

                col1, col2 = st.columns(2)
                with col1:
                    equal_weight = st.checkbox(
                        "Equal Weight All Assets",
                        value=(weight_column == "None")
                    )
                with col2:
                    normalize_weights = st.checkbox("Normalize Weights", value=True)

                if st.form_submit_button("📥 Import Portfolio", type="primary"):
                    import_portfolio_from_file(
                        df, portfolio_name, portfolio_description,
                        ticker_column, weight_column, name_column,
                        equal_weight, normalize_weights
                    )

        except Exception as e:
            st.error(f"Error reading file: {e}")


def import_portfolio_from_file(
        df: pd.DataFrame,
        name: str,
        description: str,
        ticker_col: str,
        weight_col: str,
        name_col: str,
        equal_weight: bool,
        normalize: bool
):
    """Import portfolio from DataFrame"""

    try:
        with st.spinner("Importing portfolio..."):
            # Prepare asset data
            assets = []

            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).upper().strip()

                if not ticker or pd.isna(ticker):
                    continue

                # Get weight
                if equal_weight or weight_col == "None":
                    weight = 1.0 / len(df)  # Equal weight
                else:
                    weight = float(row[weight_col])
                    # Convert percentage to decimal if needed
                    if weight > 1:
                        weight = weight / 100.0

                # Get name
                asset_name = ""
                if name_col != "None" and name_col in row:
                    asset_name = str(row[name_col])

                asset = Asset(
                    ticker=ticker,
                    name=asset_name,
                    weight=weight,
                    asset_class=AssetClass.STOCK
                )

                assets.append(asset)

            if not assets:
                st.error("No valid assets found in file")
                return

            # Normalize weights if requested
            if normalize:
                total_weight = sum(asset.weight for asset in assets)
                if total_weight > 0:
                    for asset in assets:
                        asset.weight = asset.weight / total_weight

            # Create portfolio
            portfolio = st.session_state.portfolio_manager.create_portfolio(
                name=name,
                description=description,
                assets=assets,
                portfolio_type=PortfolioType.BALANCED
            )

            # Refresh and select
            refresh_portfolios()
            st.session_state.selected_portfolio = portfolio

            st.success(f"✅ Portfolio '{name}' imported successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error importing portfolio: {e}")


def render_manual_creation():
    """Manual portfolio creation interface with autocomplete and auto price fetch"""

    st.subheader("✋ Manual Entry")

    # Initialize asset list in session state
    if 'manual_assets' not in st.session_state:
        st.session_state.manual_assets = []

    # Portfolio basic info
    col1, col2 = st.columns(2)

    with col1:
        manual_name = st.text_input("Portfolio Name *", key="manual_name")

    with col2:
        manual_type = st.selectbox(
            "Portfolio Type",
            options=[t.value for t in PortfolioType],
            format_func=lambda x: x.title(),
            key="manual_type"
        )

    manual_description = st.text_area("Description", key="manual_description")

    # Portfolio value setting
    st.subheader("Portfolio Value")
    initial_value = st.number_input(
        "Total Portfolio Value ($)",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        format="%.2f",
        help="Enter the total value you want to invest in this portfolio"
    )

    # Asset entry section
    st.subheader("Add Assets")

    with st.form("add_asset_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            new_ticker = st.text_input(
                "Ticker Symbol *",
                placeholder="Enter ticker",
                key="ticker_input"
            ).upper()

            # Live price preview
            if new_ticker and len(new_ticker) >= 2:
                # Check if it's a valid ticker format
                import re
                if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', new_ticker):
                    try:
                        preview_price = st.session_state.price_manager.get_current_price(new_ticker)
                        if preview_price:
                            st.success(f"✅ {new_ticker}: ${preview_price:.2f}")
                        else:
                            st.warning(f"⚠️ {new_ticker}: Price not found")
                    except:
                        st.info(f"🔍 {new_ticker}: Checking...")
                else:
                    if len(new_ticker) > 0:
                        st.info("💡 Enter a valid ticker symbol")

        with col2:
            new_weight = st.number_input(
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.2f",
                help="Percentage allocation for this asset"
            )

        with col3:
            new_shares = st.number_input(
                "Shares (Optional)",
                min_value=0.0,
                step=0.001,
                format="%.3f",
                help="Number of shares you want to buy"
            )

        # Add asset button with price fetching
        if st.form_submit_button("➕ Add Asset"):
            if new_ticker and new_weight > 0:
                add_manual_asset_with_price_fetch(new_ticker, new_weight, new_shares, initial_value)
            else:
                st.error("Please enter both ticker symbol and weight")

    # Display current assets
    if st.session_state.manual_assets:
        st.subheader("📋 Current Assets")

        # Create DataFrame for display with calculated values
        manual_df_data = []
        total_weight = 0.0

        for i, asset_data in enumerate(st.session_state.manual_assets):
            # Calculate dollar allocation
            dollar_allocation = (asset_data['weight'] / 100) * initial_value

            manual_df_data.append({
                'Index': i,
                'Ticker': asset_data['ticker'],
                'Weight %': f"{asset_data['weight']:.2f}%",
                'Current Price': f"${asset_data.get('current_price', 0):.2f}" if asset_data.get(
                    'current_price') else "Fetching...",
                'Dollar Allocation': f"${dollar_allocation:,.2f}",
                'Estimated Shares': f"{int(dollar_allocation / asset_data.get('current_price', 1))}" if asset_data.get(
                    'current_price') else "TBD",
                'Cash Remainder': f"${(dollar_allocation % asset_data.get('current_price', 1)):.2f}" if asset_data.get(
                    'current_price') else "TBD",
                'Manual Shares': f"{asset_data.get('shares', 0):.0f}" if asset_data.get('shares') else "-"
            })
            total_weight += asset_data['weight']

        manual_df = pd.DataFrame(manual_df_data)
        st.dataframe(manual_df, use_container_width=True, hide_index=True)

        # Weight and value summary
        col1, col2, col3 = st.columns(3)

        with col1:
            weight_color = "🟢" if abs(total_weight - 100.0) < 0.1 else "🔴"
            st.markdown(f"**Total Weight:** {weight_color} {total_weight:.2f}%")

        with col2:

            total_invested = sum(
                int((asset_data['weight'] / 100) * initial_value / asset_data.get('current_price', 1)) * asset_data.get('current_price', 0)
                for asset_data in st.session_state.manual_assets
                if asset_data.get('current_price')
            )
            st.markdown(f"**Invested:** ${total_invested:,.2f}")

        with col3:
            remaining_cash = initial_value - total_invested
            st.markdown(f"**Remaining Cash:** ${remaining_cash:,.2f}")

        # Management buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⚖️ Normalize Weights"):
                normalize_manual_weights()
                st.rerun()

        with col2:
            if st.button("💰 Update Prices"):
                update_manual_asset_prices()
                st.rerun()

        with col3:
            if st.button("🗑️ Clear All"):
                st.session_state.manual_assets = []
                st.rerun()

        # Asset removal
        if manual_df_data:
            st.subheader("🔧 Manage Assets")
            remove_index = st.selectbox(
                "Remove Asset",
                options=[-1] + list(range(len(st.session_state.manual_assets))),
                format_func=lambda x: "Select asset to remove" if x == -1 else
                f"{st.session_state.manual_assets[x]['ticker']} ({st.session_state.manual_assets[x]['weight']:.1f}%)"
            )

            if remove_index >= 0:
                if st.button(f"🗑️ Remove {st.session_state.manual_assets[remove_index]['ticker']}"):
                    del st.session_state.manual_assets[remove_index]
                    st.rerun()

        # Create portfolio button
        if manual_df_data and manual_name:
            st.subheader("🚀 Create Portfolio")

            # Validation warnings
            if abs(total_weight - 100.0) > 0.1:
                st.warning(f"⚠️ Total weight is {total_weight:.2f}%, not 100%. Consider normalizing weights.")

            # Check for missing prices
            missing_prices = [asset for asset in st.session_state.manual_assets if not asset.get('current_price')]
            if missing_prices:
                st.warning(f"⚠️ Missing prices for: {', '.join([a['ticker'] for a in missing_prices])}")

            if st.button("🚀 Create Portfolio", type="primary", use_container_width=True):
                create_manual_portfolio(manual_name, manual_description, manual_type, initial_value)


def add_manual_asset(ticker: str, weight_percent: float, shares: float, price: float):
    """Add asset to manual creation list"""

    if not ticker:
        st.error("Ticker is required")
        return

    ticker = ticker.upper().strip()

    # Check for duplicates
    if any(asset['ticker'] == ticker for asset in st.session_state.manual_assets):
        st.error(f"Asset {ticker} already exists")
        return

    # Validate ticker format
    from core.data_manager.validators import TickerValidator
    if not TickerValidator.validate_ticker(ticker):
        st.error(f"Invalid ticker format: {ticker}")
        return

    # Add to list
    asset_data = {
        'ticker': ticker,
        'weight': weight_percent,
        'shares': shares if shares > 0 else None,
        'price': price if price > 0 else None
    }

    st.session_state.manual_assets.append(asset_data)
    st.success(f"✅ Added {ticker}")


def normalize_manual_weights():
    """Normalize weights in manual asset list"""
    if not st.session_state.manual_assets:
        return

    total_weight = sum(asset['weight'] for asset in st.session_state.manual_assets)

    if total_weight > 0:
        for asset in st.session_state.manual_assets:
            asset['weight'] = (asset['weight'] / total_weight) * 100

        st.success("✅ Weights normalized to 100%")


def create_manual_portfolio(name: str, description: str, portfolio_type: str, initial_value: float):
    """Create portfolio from manual asset list"""

    try:
        with st.spinner("Creating portfolio..."):
            # Convert to Asset objects
            assets = []
            for asset_data in st.session_state.manual_assets:
                asset = Asset(
                    ticker=asset_data['ticker'],
                    weight=asset_data['weight'] / 100.0,  # Convert to decimal
                    shares=asset_data.get('shares', 0.0),
                    purchase_price=asset_data.get('price'),
                    asset_class=AssetClass.STOCK
                )
                assets.append(asset)

            # Create portfolio
            portfolio = st.session_state.portfolio_manager.create_portfolio(
                name=name,
                description=description,
                assets=assets,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Clear manual assets
            st.session_state.manual_assets = []

            # Refresh and select
            refresh_portfolios()
            st.session_state.selected_portfolio = portfolio

            st.success(f"✅ Portfolio '{name}' created successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating portfolio: {e}")


def add_manual_asset_with_price_fetch(ticker: str, weight_percent: float, shares: float, portfolio_value: float):
    """Add asset to manual creation list with automatic price fetching"""

    if not ticker:
        st.error("Ticker is required")
        return

    ticker = ticker.upper().strip()

    # Check for duplicates
    if any(asset['ticker'] == ticker for asset in st.session_state.manual_assets):
        st.error(f"Asset {ticker} already exists")
        return

    # Validate ticker format
    from core.data_manager.validators import TickerValidator
    if not TickerValidator.validate_ticker(ticker):
        st.error(f"Invalid ticker format: {ticker}")
        return

    # Fetch current price
    current_price = None
    try:
        with st.spinner(f"Fetching price for {ticker}..."):
            current_price = st.session_state.price_manager.get_current_price(ticker)

        if current_price is None:
            st.warning(f"Could not fetch price for {ticker}. You can add it anyway and update prices later.")
        else:
            st.success(f"✅ Fetched {ticker}: ${current_price:.2f}")

    except Exception as e:
        st.warning(f"Error fetching price for {ticker}: {e}")

    # Add to list
    asset_data = {
        'ticker': ticker,
        'weight': weight_percent,
        'shares': shares if shares > 0 else None,
        'current_price': current_price
    }

    st.session_state.manual_assets.append(asset_data)
    st.success(f"✅ Added {ticker} ({weight_percent:.1f}%)")


def update_manual_asset_prices():
    """Update prices for all manual assets"""
    if not st.session_state.manual_assets:
        return

    tickers = [asset['ticker'] for asset in st.session_state.manual_assets]

    try:
        with st.spinner("Updating prices..."):
            prices = st.session_state.price_manager.get_current_prices(tickers)

        updated_count = 0
        for asset in st.session_state.manual_assets:
            ticker = asset['ticker']
            if ticker in prices and prices[ticker]:
                asset['current_price'] = prices[ticker]
                updated_count += 1

        if updated_count > 0:
            st.success(f"✅ Updated {updated_count} prices")
        else:
            st.warning("No prices were updated")

    except Exception as e:
        st.error(f"Error updating prices: {e}")


def update_portfolio_prices(portfolio: Portfolio):
    """Update prices for specific portfolio"""

    with st.spinner(f"Updating prices for {portfolio.name}..."):
        try:
            updated_portfolio = st.session_state.price_manager.update_portfolio_prices(portfolio)
            st.session_state.last_price_update = datetime.now()
            st.success("✅ Prices updated successfully!")

            # Show price update summary
            prices_found = sum(1 for asset in updated_portfolio.assets if asset.current_price)
            total_assets = len(updated_portfolio.assets)
            st.info(f"📊 Updated {prices_found}/{total_assets} asset prices")

        except Exception as e:
            st.error(f"Error updating prices: {e}")

def render_template_creation():
    """Template-based portfolio creation"""

    st.subheader("🎯 Portfolio Templates")

    # Define templates
    templates = {
        "Conservative": {
            "description": "Low-risk portfolio focused on stability",
            "assets": [
                ("SPY", 40, "S&P 500 ETF"),
                ("BND", 40, "Bond ETF"),
                ("VTI", 10, "Total Stock Market"),
                ("GLD", 10, "Gold ETF")
            ]
        },
        "Balanced": {
            "description": "Moderate risk with balanced growth and income",
            "assets": [
                ("SPY", 50, "S&P 500 ETF"),
                ("QQQ", 20, "NASDAQ ETF"),
                ("BND", 20, "Bond ETF"),
                ("VEA", 10, "International ETF")
            ]
        },
        "Growth": {
            "description": "Higher risk focused on capital appreciation",
            "assets": [
                ("QQQ", 30, "NASDAQ ETF"),
                ("SPY", 25, "S&P 500 ETF"),
                ("VUG", 20, "Growth ETF"),
                ("ARKK", 15, "Innovation ETF"),
                ("VEA", 10, "International ETF")
            ]
        },
        "Tech Focus": {
            "description": "Technology-heavy growth portfolio",
            "assets": [
                ("AAPL", 20, "Apple Inc."),
                ("MSFT", 18, "Microsoft"),
                ("GOOGL", 15, "Alphabet"),
                ("AMZN", 12, "Amazon"),
                ("TSLA", 10, "Tesla"),
                ("NVDA", 10, "NVIDIA"),
                ("META", 8, "Meta Platforms"),
                ("NFLX", 7, "Netflix")
            ]
        },
        "Dividend Income": {
            "description": "Focus on dividend-paying stocks",
            "assets": [
                ("VYM", 25, "Vanguard High Dividend"),
                ("SCHD", 20, "Schwab Dividend ETF"),
                ("JNJ", 10, "Johnson & Johnson"),
                ("PG", 10, "Procter & Gamble"),
                ("KO", 8, "Coca-Cola"),
                ("T", 8, "AT&T"),
                ("VZ", 7, "Verizon"),
                ("XOM", 7, "ExxonMobil"),
                ("CVX", 5, "Chevron")
            ]
        }
    }

    # Template selection
    selected_template = st.selectbox(
        "Choose Template",
        options=list(templates.keys()),
        format_func=lambda x: f"{x} - {templates[x]['description']}"
    )

    if selected_template:
        template = templates[selected_template]

        # Show template details
        st.subheader(f"📋 {selected_template} Template")
        st.write(template['description'])

        # Display template assets
        template_df = pd.DataFrame([
            {
                'Ticker': ticker,
                'Weight %': f"{weight:.1f}%",
                'Description': desc
            }
            for ticker, weight, desc in template['assets']
        ])

        st.dataframe(template_df, use_container_width=True, hide_index=True)

        # Creation form
        with st.form("template_form"):
            template_name = st.text_input(
                "Portfolio Name *",
                value=f"{selected_template} Portfolio",
                placeholder="Enter portfolio name"
            )

            template_description = st.text_area(
                "Description",
                value=template['description'],
                help="Customize the description if needed"
            )

            col1, col2 = st.columns(2)
            with col1:
                template_initial_value = st.number_input(
                    "Initial Value ($)",
                    min_value=1000.0,
                    value=100000.0,
                    step=1000.0
                )

            with col2:
                fetch_prices = st.checkbox("Fetch Current Prices", value=True)

            if st.form_submit_button("🚀 Create from Template", type="primary"):
                create_template_portfolio(
                    template_name,
                    template_description,
                    template,
                    template_initial_value,
                    fetch_prices
                )


def create_template_portfolio(
        name: str,
        description: str,
        template: Dict,
        initial_value: float,
        fetch_prices: bool
):
    """Create portfolio from template"""

    try:
        with st.spinner("Creating portfolio from template..."):
            # Create assets
            assets = []
            for ticker, weight, desc in template['assets']:
                asset = Asset(
                    ticker=ticker,
                    name=desc,
                    weight=weight / 100.0,  # Convert to decimal
                    asset_class=AssetClass.STOCK
                )
                assets.append(asset)

            # Create portfolio
            portfolio = st.session_state.portfolio_manager.create_portfolio(
                name=name,
                description=description,
                assets=assets,
                initial_value=initial_value
            )

            # Fetch current prices if requested
            if fetch_prices:
                update_portfolio_prices(portfolio)

            # Refresh and select
            refresh_portfolios()
            st.session_state.selected_portfolio = portfolio

            st.success(f"✅ Portfolio '{name}' created from template!")
            display_portfolio_summary(portfolio)

        # Calculate shares and fetch prices
        with st.spinner("Fetching prices and calculating shares..."):
            tickers = [asset.ticker for asset in portfolio.assets]
            prices = st.session_state.price_manager.get_current_prices(tickers)

            for asset in portfolio.assets:
                if asset.ticker in prices and prices[asset.ticker]:
                    asset.current_price = prices[asset.ticker]
                    allocation = asset.weight * initial_value
                    asset.shares = int(allocation / asset.current_price)

        # Calculate shares for each asset
        for asset in portfolio.assets:
            if hasattr(asset, 'current_price') and asset.current_price and asset.current_price > 0:
                allocation = asset.weight * initial_value
                asset.shares = int(allocation / asset.current_price)
            else:
                asset.shares = 0

        # Calculate shares based on prices and allocation
        if fetch_prices:
            with st.spinner("Calculating shares..."):
                tickers = [asset.ticker for asset in portfolio.assets]
                prices = st.session_state.price_manager.get_current_prices(tickers)

                for asset in portfolio.assets:
                    if asset.ticker in prices and prices[asset.ticker]:
                        asset.current_price = prices[asset.ticker]
                        allocation = asset.weight * initial_value
                        asset.shares = int(allocation / asset.current_price)

    except Exception as e:
        st.error(f"Error creating template portfolio: {e}")


# ================================
# PAGE: MANAGE PORTFOLIOS
# ================================

def render_manage_portfolios():
    """Render portfolio management page"""

    st.header("📋 Manage Portfolios")

    if not st.session_state.portfolios:
        refresh_portfolios()

    if not st.session_state.portfolios:
        st.info("No portfolios found. Create your first portfolio!")
        return

    # Portfolio selection for management
    portfolio_options = {p.name: p for p in st.session_state.portfolios}
    selected_name = st.selectbox(
        "Select Portfolio to Manage",
        options=list(portfolio_options.keys()),
        key="manage_portfolio_selector"
    )

    if selected_name:
        selected_portfolio = portfolio_options[selected_name]

        # Portfolio actions
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📊 View Details", use_container_width=True):
                display_portfolio_details(selected_portfolio)

        with col2:
            if st.button("💰 Update Prices", use_container_width=True):
                update_portfolio_prices(selected_portfolio)

        with col3:
            if st.button("📄 Export", use_container_width=True):
                export_portfolio_data(selected_portfolio)

        with col4:
            if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                delete_portfolio_confirmation(selected_portfolio)


def display_portfolio_details(portfolio: Portfolio):
    """Display detailed portfolio information"""

    st.subheader(f"📊 {portfolio.name} - Details")

    # Basic info
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Basic Information:**")
        st.write(f"- **Name:** {portfolio.name}")
        st.write(f"- **Type:** {portfolio.portfolio_type.value.title()}")
        st.write(f"- **Created:** {portfolio.created_date.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"- **Last Modified:** {portfolio.last_modified.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"- **Assets:** {len(portfolio.assets)}")

    with col2:
        stats = portfolio.get_statistics()
        st.write("**Portfolio Statistics:**")
        st.write(f"- **Total Value:** {format_currency(stats.total_value)}")
        st.write(f"- **Total Cost:** {format_currency(stats.total_cost)}")
        st.write(f"- **Unrealized P&L:** {format_currency(stats.unrealized_pnl)}")
        st.write(f"- **P&L %:** {format_percentage(stats.unrealized_pnl_percent)}")

    if portfolio.description:
        st.write(f"**Description:** {portfolio.description}")

    if portfolio.tags:
        st.write(f"**Tags:** {', '.join(portfolio.tags)}")

    # Assets table
    st.subheader("📋 Holdings")

    if portfolio.assets:
        holdings_data = []
        for asset in portfolio.assets:
            holdings_data.append({
                'Ticker': asset.ticker,
                'Name': asset.name or 'N/A',
                'Weight %': f"{asset.weight * 100:.2f}%",
                'Shares': f"{asset.shares:.3f}" if asset.shares > 0 else "N/A",
                'Current Price': f"${asset.current_price:.2f}" if asset.current_price else "N/A",
                'Market Value': format_currency(asset.market_value),
                'P&L': format_currency(asset.unrealized_pnl),
                'P&L %': format_percentage(asset.unrealized_pnl_percent),
                'Sector': asset.sector or 'N/A'
            })

        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

        # Allocation charts
        col1, col2 = st.columns(2)

        with col1:
            # Weight allocation pie chart
            if len(portfolio.assets) > 0:
                fig_pie = px.pie(
                    values=[asset.weight for asset in portfolio.assets],
                    names=[asset.ticker for asset in portfolio.assets],
                    title="Asset Allocation"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Sector allocation
            sectors = portfolio.get_sector_allocation()
            if sectors:
                fig_sector = px.pie(
                    values=list(sectors.values()),
                    names=list(sectors.keys()),
                    title="Sector Allocation"
                )
                fig_sector.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.info("No sector information available")


# ================================
# PAGE: PORTFOLIO ANALYSIS
# ================================

def render_portfolio_analysis():
    """Render portfolio analysis page"""

    st.header("📊 Portfolio Analysis")

    if not st.session_state.selected_portfolio:
        st.warning("⚠️ Please select a portfolio first")
        return

    portfolio = st.session_state.selected_portfolio

    # Analysis controls
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            options=["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"],
            index=4  # Default to 1Y
        )

    with col2:
        benchmark = st.selectbox(
            "Benchmark",
            options=["None", "S&P 500 (^GSPC)", "NASDAQ (^IXIC)", "Custom"],
            index=1  # Default to S&P 500
        )

    with col3:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            update_portfolio_prices(portfolio)
            st.rerun()

    # Basic metrics display
    st.subheader("📈 Key Metrics")

    # Placeholder metrics (will be calculated by analytics engine in Phase 2)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_value = portfolio.calculate_value()
        st.metric("Portfolio Value", format_currency(current_value))

    with col2:
        # Placeholder calculation
        daily_change = 0.012  # +1.2%
        st.metric("Today's Change", "+1.2%", delta="$1,234")

    with col3:
        # Placeholder YTD return
        st.metric("YTD Return", "+18.5%", delta="+2.3%")

    with col4:
        # Placeholder volatility
        st.metric("Volatility", "16.2%", delta="-0.8%")

    # Holdings analysis
    st.subheader("📋 Holdings Analysis")

    if portfolio.assets:
        # Update prices first
        with st.spinner("Fetching current prices..."):
            updated_portfolio = st.session_state.price_manager.update_portfolio_prices(portfolio)

        # Create enhanced holdings table
        holdings_data = []
        for asset in updated_portfolio.assets:
            holdings_data.append({
                'Ticker': asset.ticker,
                'Name': asset.name[:30] + "..." if len(asset.name) > 30 else asset.name,
                'Weight': asset.weight,
                'Weight %': f"{asset.weight * 100:.2f}%",
                'Current Price': asset.current_price or 0,
                'Market Value': asset.market_value,
                'Sector': asset.sector or 'Unknown'
            })

        holdings_df = pd.DataFrame(holdings_data)

        # Display interactive table
        st.dataframe(
            holdings_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Weight': st.column_config.ProgressColumn(
                    'Weight',
                    min_value=0,
                    max_value=max(holdings_df['Weight']) if len(holdings_df) > 0 else 1,
                    format="%.3f"
                ),
                'Current Price': st.column_config.NumberColumn(
                    'Current Price',
                    format="$%.2f"
                ),
                'Market Value': st.column_config.NumberColumn(
                    'Market Value',
                    format="$%.0f"
                )
            }
        )

        # Visualization section
        st.subheader("📊 Portfolio Visualization")

        tab1, tab2, tab3 = st.tabs(["Allocation", "Performance", "Risk"])

        with tab1:
            render_allocation_charts(updated_portfolio)

        with tab2:
            render_performance_charts(updated_portfolio)

        with tab3:
            render_risk_overview(updated_portfolio)


def render_allocation_charts(portfolio: Portfolio):
    """Render allocation visualization charts"""

    col1, col2 = st.columns(2)

    with col1:
        # Asset allocation donut chart
        fig_assets = go.Figure(data=[go.Pie(
            labels=[asset.ticker for asset in portfolio.assets],
            values=[asset.weight for asset in portfolio.assets],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])

        fig_assets.update_layout(
            title="Asset Allocation",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig_assets, use_container_width=True)

    with col2:
        # Sector allocation
        sectors = portfolio.get_sector_allocation()
        if sectors:
            fig_sectors = go.Figure(data=[go.Pie(
                labels=list(sectors.keys()),
                values=list(sectors.values()),
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(colors=px.colors.qualitative.Pastel)
            )])

            fig_sectors.update_layout(
                title="Sector Allocation",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig_sectors, use_container_width=True)
        else:
            st.info("Sector information not available")


def render_performance_charts(portfolio: Portfolio):
    """Render performance charts (placeholder for Phase 2)"""

    st.info("📈 Detailed performance charts will be available in Phase 2 (Analytics Engine)")

    # Simple placeholder chart
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    values = [100000 * (1 + 0.0003 * i + 0.01 * np.random.randn()) for i in range(len(dates))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#BF9FFB', width=2)
    ))

    fig.update_layout(
        title="Portfolio Performance (Simulated)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_overview(portfolio: Portfolio):
    """Render risk overview (placeholder for Phase 3)"""

    st.info("🛡️ Comprehensive risk analysis will be available in Phase 3 (Risk Engine)")

    # Basic risk metrics placeholder
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Level", "MODERATE", help="Based on portfolio composition")

    with col2:
        st.metric("Estimated VaR (95%)", "$12,345", help="Value at Risk calculation")

    with col3:
        st.metric("Diversification Score", "7.2/10", help="Portfolio diversification rating")


# ================================
# PAGE: SYSTEM STATUS
# ================================

def render_system_status():
    """Render system status and diagnostics"""

    st.header("⚙️ System Status")

    # Module status
    st.subheader("📦 Module Status")

    modules = {
        "Data Manager": "✅ Active",
        "Price Manager": "✅ Active",
        "Analytics Engine": "🚧 Phase 2",
        "Risk Engine": "🚧 Phase 3",
        "Optimization Engine": "🚧 Phase 4",
        "Scenario Engine": "🚧 Phase 5",
        "Reporting Engine": "🚧 Phase 6"
    }

    col1, col2 = st.columns(2)

    for i, (module, status) in enumerate(modules.items()):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.write(f"**{module}:** {status}")

    # Data provider status
    st.subheader("📡 Data Provider Status")

    provider_status = st.session_state.price_manager.get_provider_status()

    for provider, is_available in provider_status.items():
        status_icon = "✅" if is_available else "❌"
        status_text = "Available" if is_available else "Unavailable"
        st.write(f"**{provider.title()} Provider:** {status_icon} {status_text}")

    # Cache statistics
    st.subheader("💾 Cache Statistics")

    cache_stats = st.session_state.price_manager.get_cache_stats()

    if cache_stats.get('cache_enabled'):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Valid Entries", cache_stats.get('valid_entries', 0))

        with col2:
            st.metric("Expired Entries", cache_stats.get('expired_entries', 0))

        with col3:
            cache_size = cache_stats.get('cache_size_mb', 0)
            st.metric("Cache Size", f"{cache_size:.2f} MB")

        if st.button("🗑️ Clear Cache"):
            st.session_state.price_manager.clear_cache()
            st.success("Cache cleared successfully!")
            st.rerun()
    else:
        st.info("Caching is disabled")

    # Storage statistics
    st.subheader("💿 Storage Information")

    storage_stats = st.session_state.portfolio_manager.get_cache_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Cached Portfolios", storage_stats.get('cached_portfolios', 0))

    with col2:
        cache_size = storage_stats.get('cache_size_mb', 0)
        st.metric("Memory Usage", f"{cache_size:.2f} MB")


# ================================
# UTILITY FUNCTIONS
# ================================

def display_portfolio_summary(portfolio: Portfolio):
    """Display portfolio creation summary"""

    st.subheader("📊 Portfolio Summary")

    # Calculate actual invested amount
    total_invested = 0
    for asset in portfolio.assets:
        if asset.current_price and asset.shares:
            total_invested += asset.shares * asset.current_price

    # Calculate remaining cash
    remaining_cash = portfolio.initial_value - total_invested

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Assets", len(portfolio.assets))

    with col2:
        st.metric("Total Weight", f"{portfolio.total_weight:.1%}")

    with col3:
        st.metric("Initial Value", format_currency(portfolio.initial_value))

    # Calculate actual current value based on shares * current_price
    current_value = sum(asset.shares * (asset.current_price or 0) for asset in portfolio.assets)

    with col4:
        st.metric("Current Value", format_currency(current_value))

    # Asset breakdown with shares
    if portfolio.assets:
        st.write("**Asset Breakdown:**")
        for asset in portfolio.assets:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"• **{asset.ticker}** - {asset.name or 'Unknown'}")
            with col2:
                st.write(f"{asset.weight:.1%}")
            with col3:
                if asset.current_price:
                    st.write(f"${asset.current_price:.2f}")
                else:
                    st.write("Price N/A")
            with col4:
                if asset.shares > 0:
                    st.write(f"{int(asset.shares)} shares")
                else:
                    st.write("0 shares")

    # Show remaining cash
    if remaining_cash > 0:
        st.info(f"💰 Remaining cash: {format_currency(remaining_cash)}")
    else:
        st.success("✅ Fully invested")


def update_all_prices():
    """Update prices for all portfolios"""

    with st.spinner("Updating all portfolio prices..."):
        updated_count = 0

        for portfolio in st.session_state.portfolios:
            try:
                st.session_state.price_manager.update_portfolio_prices(portfolio)
                updated_count += 1
            except Exception as e:
                st.error(f"Error updating {portfolio.name}: {e}")

        st.session_state.last_price_update = datetime.now()

        st.success(f"✅ Updated prices for {updated_count} portfolios")


def update_portfolio_prices(portfolio: Portfolio):
    """Update prices for specific portfolio"""

    with st.spinner(f"Updating prices for {portfolio.name}..."):
        try:
            updated_portfolio = st.session_state.price_manager.update_portfolio_prices(portfolio)
            st.session_state.last_price_update = datetime.now()
            st.success("✅ Prices updated successfully!")

            # Show price update summary
            prices_found = sum(1 for asset in updated_portfolio.assets if asset.current_price)
            total_assets = len(updated_portfolio.assets)
            st.info(f"📊 Updated {prices_found}/{total_assets} asset prices")

        except Exception as e:
            st.error(f"Error updating prices: {e}")


def update_company_info(portfolio: Portfolio):
    """Fetch and update company information for assets"""

    for asset in portfolio.assets:
        if not asset.name:  # Only fetch if name is missing
            try:
                company_info = st.session_state.price_manager.get_company_info(asset.ticker)
                if company_info:
                    asset.name = company_info.name
                    asset.sector = company_info.sector
            except Exception as e:
                logger.warning(f"Could not fetch info for {asset.ticker}: {e}")


def export_portfolio_data(portfolio: Optional[Portfolio] = None):
    """Export portfolio data"""

    if not portfolio:
        portfolio = st.session_state.selected_portfolio

    if not portfolio:
        st.error("No portfolio selected for export")
        return

    # Export options
    export_format = st.selectbox(
        "Export Format",
        options=["CSV", "Excel", "JSON"],
        key=f"export_format_{portfolio.id}"
    )

    if st.button(f"📄 Export as {export_format}"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{portfolio.name}_{timestamp}"

            if export_format == "CSV":
                file_path = f"exports/{filename}.csv"
                st.session_state.portfolio_manager.export_to_csv(portfolio.id, file_path)
            elif export_format == "Excel":
                file_path = f"exports/{filename}.xlsx"
                st.session_state.portfolio_manager.export_to_excel(portfolio.id, file_path)
            elif export_format == "JSON":
                file_path = f"exports/{filename}.json"
                st.session_state.portfolio_manager.export_to_json(portfolio.id, file_path)

            st.success(f"✅ Portfolio exported to: {file_path}")

        except Exception as e:
            st.error(f"Export failed: {e}")


def delete_portfolio_confirmation(portfolio: Portfolio):
    """Show delete confirmation dialog"""

    # Use session state to track confirmation
    confirm_key = f"delete_confirm_{portfolio.id}"

    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False

    if not st.session_state[confirm_key]:
        st.error(f"⚠️ Are you sure you want to delete '{portfolio.name}'?")
        st.write("This action cannot be undone.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("❌ Yes, Delete", key=f"confirm_delete_{portfolio.id}"):
                st.session_state[confirm_key] = True
                st.rerun()

        with col2:
            if st.button("✅ Cancel", key=f"cancel_delete_{portfolio.id}"):
                st.rerun()
    else:
        # Actually delete the portfolio
        try:
            import os
            # Get portfolio file path and delete it directly
            portfolio_file = f"data/portfolios/{portfolio.id}.json"
            if os.path.exists(portfolio_file):
                os.remove(portfolio_file)

            # Also try portfolio manager delete
            st.session_state.portfolio_manager.delete_portfolio(portfolio.id)

            refresh_portfolios()
            if st.session_state.selected_portfolio and st.session_state.selected_portfolio.id == portfolio.id:
                st.session_state.selected_portfolio = None

            # Clear confirmation state
            del st.session_state[confirm_key]

            st.success(f"✅ Portfolio '{portfolio.name}' deleted successfully!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Error deleting portfolio: {e}")
            del st.session_state[confirm_key]


# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application entry point"""

    # Render header
    render_header()

    # Render sidebar and get selected page
    current_page = render_sidebar()

    # Refresh portfolios on startup
    if not st.session_state.portfolios:
        refresh_portfolios()

    # Route to appropriate page
    if current_page == "🏠 Dashboard":
        render_dashboard()

    elif current_page == "📝 Create Portfolio":
        render_create_portfolio()

    elif current_page == "📋 Manage Portfolios":
        render_manage_portfolios()

    elif current_page == "📊 Portfolio Analysis":
        render_portfolio_analysis()

    elif current_page == "⚙️ System Status":
        render_system_status()

    # Footer
    st.divider()
    st.caption("🚀 Wild Market Capital Portfolio Manager v1.0.0 - Phase 1: Data Foundation")


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    main()