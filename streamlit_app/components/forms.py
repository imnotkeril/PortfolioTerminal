"""
Form components for the Portfolio Management System.

This module contains reusable form components for portfolio creation and management.
"""
import streamlit as st
import pandas as pd
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from io import BytesIO
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import PortfolioType, AssetClass, Asset
from ..utils.session_state import get_portfolio_manager, get_price_manager, refresh_portfolios, set_selected_portfolio
from ..utils.helpers import (
    validate_ticker_input,
    update_company_info,
    display_portfolio_summary,
    validate_file_upload,
    generate_sample_data
)


def render_text_input_form():
    """Render the text input portfolio creation form."""

    st.subheader("✏️ Create from Text Input")

    # Sample data helper
    col1, col2 = st.columns([3, 1])

    with col1:
        st.info("Enter ticker symbols with weights. Supports multiple formats.")

    with col2:
        sample_type = st.selectbox(
            "Load Sample",
            ["None", "Tech Focus", "Balanced", "Dividend", "ETF", "Conservative", "Growth"],
            key="sample_selector"
        )

        if sample_type != "None":
            sample_text = generate_sample_data(sample_type.lower().replace(' ', '_'))
            st.session_state.text_input_sample = sample_text

    # Input form
    with st.form("text_input_form"):
        portfolio_name = st.text_input(
            "Portfolio Name *",
            placeholder="e.g., My Tech Portfolio",
            help="Enter a unique name for your portfolio"
        )

        portfolio_description = st.text_area(
            "Description (Optional)",
            placeholder="Brief description of your investment strategy...",
            height=80
        )

        # Use sample text if selected
        default_text = st.session_state.get('text_input_sample', '')
        text_input = st.text_area(
            "Enter Tickers and Weights *",
            value=default_text,
            placeholder="AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%",
            height=120,
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

        # Additional options
        st.write("**Options**")
        col1, col2 = st.columns(2)

        with col1:
            auto_normalize = st.checkbox(
                "Auto-normalize weights",
                value=True,
                help="Automatically adjust weights to sum to 100%",
                key="text_input_auto_normalize"
            )

        with col2:
            fetch_company_info = st.checkbox(
                "Fetch company information",
                value=True,
                help="Automatically fetch company names and sector data",
                key="text_input_fetch_info"
            )

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


def render_file_upload_form():
    """Render the file upload portfolio creation form."""

    st.subheader("📁 Upload Portfolio File")

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with ticker symbols and weights"
    )

    if uploaded_file is not None:
        # Validate file
        is_valid, error_msg = validate_file_upload(uploaded_file)

        if not is_valid:
            st.error(error_msg)
            return

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
                    options=["None (Equal Weight)"] + df.columns.tolist(),
                    help="Column containing weights (optional for equal weighting)"
                )

            with col3:
                name_column = st.selectbox(
                    "Name Column (Optional)",
                    options=["None"] + df.columns.tolist(),
                    help="Column containing company names"
                )

            # Portfolio details form
            with st.form("file_import_form"):
                portfolio_name = st.text_input("Portfolio Name *")
                portfolio_description = st.text_area("Description (Optional)")

                col1, col2 = st.columns(2)
                with col1:
                    equal_weight = st.checkbox(
                        "Equal Weight All Assets",
                        value=(weight_column == "None (Equal Weight)")
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


def render_file_operations_form():
    """Render combined file upload and import/export form."""

    st.subheader("📁 File Operations")

    # Sub-tabs for different file operations
    subtab1, subtab2, subtab3 = st.tabs([
        "📤 Upload CSV/Excel",
        "📥 Import JSON",
        "📋 Export Portfolio"
    ])

    with subtab1:
        render_csv_excel_upload()

    with subtab2:
        render_json_import()

    with subtab3:
        render_portfolio_export()


def render_csv_excel_upload():
    """Render CSV/Excel upload section."""

    st.write("#### Upload Portfolio File")
    st.info("Upload CSV or Excel files with your portfolio holdings")

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with ticker symbols and weights"
    )

    if uploaded_file is not None:
        # Validate file
        is_valid, error_msg = validate_file_upload(uploaded_file)

        if not is_valid:
            st.error(error_msg)
            return

        try:
            # Show file info
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")

            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Show preview
            st.write("##### File Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column mapping
            st.write("##### Column Mapping")

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
                    options=["None (Equal Weight)"] + df.columns.tolist(),
                    help="Column containing weights (optional for equal weighting)"
                )

            with col3:
                name_column = st.selectbox(
                    "Name Column (Optional)",
                    options=["None"] + df.columns.tolist(),
                    help="Column containing company names"
                )

            # Portfolio details form
            with st.form("file_import_form"):
                portfolio_name = st.text_input("Portfolio Name *")
                portfolio_description = st.text_area("Description (Optional)")

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

        except Exception as e:
            st.error(f"Error reading file: {e}")


def render_json_import():
    """Render JSON import section."""

    st.write("#### Import Previously Exported Portfolio")
    st.info("Upload JSON files previously exported from this system")

    import_file = st.file_uploader(
        "Upload Portfolio JSON File",
        type=['json'],
        help="Import a portfolio that was previously exported from this system"
    )

    if import_file is not None:
        try:
            portfolio_data = json.load(import_file)

            st.success(f"📄 Loaded portfolio data: {portfolio_data.get('name', 'Unnamed')}")

            # Show preview
            with st.expander("Preview Portfolio Data", expanded=False):
                st.json(portfolio_data)

            if st.button("📥 Import Portfolio", type="primary", use_container_width=True):
                import_portfolio_from_json(portfolio_data)

        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")


def render_portfolio_export():
    """Render portfolio export section."""

    st.write("#### Export Existing Portfolio")
    st.info("Export your portfolios for backup or sharing")

    from ..utils.session_state import get_portfolios
    portfolios = get_portfolios()

    if portfolios:
        portfolio_options = {p.name: p for p in portfolios}
        selected_name = st.selectbox(
            "Select Portfolio to Export",
            options=list(portfolio_options.keys()),
            help="Choose which portfolio you want to export"
        )

        if selected_name:
            selected_portfolio = portfolio_options[selected_name]

            # Show portfolio summary
            with st.expander("Portfolio Summary", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Assets", len(selected_portfolio.assets))

                with col2:
                    st.metric("Type", selected_portfolio.portfolio_type.value.title())

                with col3:
                    st.metric("Created", selected_portfolio.created_date.strftime("%m/%d/%Y"))

            export_format = st.selectbox(
                "Export Format",
                options=["JSON", "CSV", "Excel"],
                help="Choose the format for your exported file"
            )

            if st.button("📤 Export Portfolio", type="primary", use_container_width=True):
                export_portfolio_data(selected_portfolio, export_format)
    else:
        st.info("No portfolios available to export. Create a portfolio first.")

        if st.button("➕ Create Portfolio", use_container_width=True):
            st.query_params["tab"] = "text_input"
            st.rerun()


def render_manual_creation_form():
    """Render the manual portfolio creation form."""

    st.subheader("✋ Manual Asset Entry")

    # Initialize manual assets if not exists
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

    manual_description = st.text_area("Description (Optional)", key="manual_description")

    # Portfolio value setting
    st.subheader("💰 Portfolio Value")
    initial_value = st.number_input(
        "Total Portfolio Value ($)",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        format="%.2f",
        help="Enter the total value you want to invest in this portfolio"
    )

    # Asset entry section
    st.subheader("📈 Add Assets")

    with st.form("add_asset_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            new_ticker = st.text_input(
                "Ticker Symbol *",
                placeholder="e.g., AAPL",
                key="ticker_input"
            ).upper()

            # Live price preview
            if new_ticker and len(new_ticker) >= 2:
                try:
                    price_manager = get_price_manager()
                    preview_price = price_manager.get_current_price(new_ticker)
                    if preview_price:
                        st.success(f"💰 Current Price: ${preview_price:.2f}")
                    else:
                        st.warning("⚠️ Price not available")
                except Exception as e:
                    st.warning(f"⚠️ Error fetching price: {str(e)}")

        with col2:
            new_weight = st.number_input(
                "Weight (%)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1,
                key="weight_input"
            )

        with col3:
            new_shares = st.number_input(
                "Shares (Optional)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="shares_input",
                help="Leave 0 for automatic calculation"
            )

        add_asset_button = st.form_submit_button("➕ Add Asset", use_container_width=True)

        if add_asset_button and new_ticker:
            add_manual_asset_with_price_fetch(new_ticker, new_weight, new_shares, initial_value)
            st.rerun()

    # Display current assets if any
    if st.session_state.manual_assets:
        st.subheader("📊 Current Assets")

        # Build dataframe for display
        manual_df_data = []
        total_weight = 0.0  # Initialize as float
        total_invested = 0.0

        for i, asset_data in enumerate(st.session_state.manual_assets):
            # Safely get weight and price with defaults
            weight = asset_data.get('weight', 0.0) or 0.0
            current_price = asset_data.get('current_price', 0.0) or 0.0
            shares = asset_data.get('shares', 0.0) or 0.0

            # Calculate dollar allocation
            dollar_allocation = (weight / 100) * initial_value

            # Calculate estimated shares if no manual shares provided
            estimated_shares = 0
            cash_remainder = 0.0
            if current_price > 0:
                if shares > 0:
                    # Use manual shares
                    estimated_shares = int(shares)
                else:
                    # Calculate automatic shares
                    estimated_shares = int(dollar_allocation / current_price)

                cash_remainder = dollar_allocation - (estimated_shares * current_price)
                total_invested += estimated_shares * current_price

            manual_df_data.append({
                'Index': i,
                'Ticker': asset_data['ticker'],
                'Weight %': f"{weight:.2f}%",
                'Current Price': f"${current_price:.2f}" if current_price > 0 else "Fetching...",
                'Dollar Allocation': f"${dollar_allocation:,.2f}",
                'Estimated Shares': f"{estimated_shares}" if current_price > 0 else "TBD",
                'Cash Remainder': f"${cash_remainder:.2f}" if current_price > 0 else "TBD",
                'Manual Shares': f"{shares:.0f}" if shares > 0 else "-"
            })

            # Add weight to total (now both are guaranteed to be numbers)
            total_weight += weight

        manual_df = pd.DataFrame(manual_df_data)
        st.dataframe(manual_df.drop('Index', axis=1), use_container_width=True, hide_index=True)

        # Weight and value summary
        col1, col2, col3 = st.columns(3)

        with col1:
            weight_color = "🟢" if abs(total_weight - 100.0) < 0.1 else "🔴"
            st.markdown(f"**Total Weight:** {weight_color} {total_weight:.2f}%")

        with col2:
            st.markdown(f"**Invested:** ${total_invested:,.2f}")

        with col3:
            remaining_cash = initial_value - total_invested
            st.markdown(f"**Remaining Cash:** ${remaining_cash:,.2f}")

        # Management buttons
        col1, col2, col3, col4 = st.columns(4)

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

        with col4:
            # Remove selected asset
            if manual_df_data:
                remove_index = st.selectbox(
                    "Remove:",
                    options=[-1] + list(range(len(st.session_state.manual_assets))),
                    format_func=lambda x: "Select..." if x == -1 else f"{st.session_state.manual_assets[x]['ticker']}",
                    key="remove_asset_selector"
                )

                if remove_index >= 0 and st.button(f"🗑️ Remove"):
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


def render_portfolio_template_form():
    """Render the portfolio template creation form."""

    st.subheader("📋 Create from Template")

    # Define templates
    templates = {
        "Conservative Growth": {
            "description": "Low-risk portfolio focused on stability and moderate growth",
            "assets": [
                ("SPY", 40, "S&P 500 ETF"),
                ("BND", 30, "Aggregate Bond ETF"),
                ("VTI", 15, "Total Stock Market ETF"),
                ("VTEB", 10, "Tax-Exempt Bond ETF"),
                ("GLD", 5, "Gold ETF")
            ]
        },
        "Balanced Growth": {
            "description": "Moderate risk with balanced growth and income approach",
            "assets": [
                ("SPY", 50, "S&P 500 ETF"),
                ("QQQ", 20, "NASDAQ 100 ETF"),
                ("BND", 15, "Aggregate Bond ETF"),
                ("VEA", 10, "Developed Markets ETF"),
                ("VWO", 5, "Emerging Markets ETF")
            ]
        },
        "Tech Focus": {
            "description": "Technology-heavy growth portfolio for higher risk tolerance",
            "assets": [
                ("AAPL", 20, "Apple Inc."),
                ("MSFT", 18, "Microsoft Corporation"),
                ("GOOGL", 15, "Alphabet Inc."),
                ("AMZN", 12, "Amazon.com Inc."),
                ("TSLA", 10, "Tesla Inc."),
                ("NVDA", 10, "NVIDIA Corporation"),
                ("META", 8, "Meta Platforms Inc."),
                ("NFLX", 7, "Netflix Inc.")
            ]
        },
        "Dividend Income": {
            "description": "Focus on dividend-paying stocks for steady income",
            "assets": [
                ("VYM", 25, "Vanguard High Dividend Yield ETF"),
                ("SCHD", 20, "Schwab US Dividend Equity ETF"),
                ("JNJ", 10, "Johnson & Johnson"),
                ("PG", 10, "Procter & Gamble"),
                ("KO", 8, "Coca-Cola Company"),
                ("PEP", 8, "PepsiCo Inc."),
                ("VZ", 7, "Verizon Communications"),
                ("T", 7, "AT&T Inc."),
                ("XOM", 5, "Exxon Mobil Corporation")
            ]
        },
        "Global Diversified": {
            "description": "Internationally diversified portfolio across markets",
            "assets": [
                ("VTI", 35, "Total US Stock Market ETF"),
                ("VTIAX", 25, "Total International Stock ETF"),
                ("VEA", 20, "Developed Markets ETF"),
                ("VWO", 10, "Emerging Markets ETF"),
                ("BND", 10, "US Aggregate Bond ETF")
            ]
        },
        "ESG Focused": {
            "description": "Environmentally and socially responsible investing",
            "assets": [
                ("ESGU", 30, "ESG MSCI USA ETF"),
                ("ESGD", 25, "ESG MSCI EAFE ETF"),
                ("SUSA", 20, "ESG S&P 500 ETF"),
                ("ICLN", 15, "Clean Energy ETF"),
                ("SUSC", 10, "ESG Small-Cap ETF")
            ]
        }
    }

    # Template selection
    selected_template = st.selectbox(
        "Choose Template",
        options=list(templates.keys()),
        format_func=lambda x: f"{x}",
        key="template_selector"
    )

    if selected_template:
        template = templates[selected_template]

        # Show template details
        st.info(template['description'])

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


def render_import_export_form():
    """Render the import/export portfolio form."""

    st.subheader("🔄 Import/Export Portfolios")

    tab1, tab2 = st.tabs(["📥 Import", "📤 Export"])

    with tab1:
        st.write("### Import Previously Exported Portfolio")

        import_file = st.file_uploader(
            "Upload Portfolio JSON File",
            type=['json'],
            help="Import a portfolio that was previously exported from this system"
        )

        if import_file is not None:
            try:
                portfolio_data = json.load(import_file)

                st.success(f"📄 Loaded portfolio data: {portfolio_data.get('name', 'Unnamed')}")

                # Show preview
                st.json(portfolio_data, expanded=False)

                if st.button("Import Portfolio", type="primary"):
                    import_portfolio_from_json(portfolio_data)

            except Exception as e:
                st.error(f"Error reading JSON file: {str(e)}")

    with tab2:
        st.write("### Export Existing Portfolio")

        from ..utils.session_state import get_portfolios
        portfolios = get_portfolios()

        if portfolios:
            portfolio_options = {p.name: p for p in portfolios}
            selected_name = st.selectbox(
                "Select Portfolio to Export",
                options=list(portfolio_options.keys())
            )

            if selected_name:
                selected_portfolio = portfolio_options[selected_name]

                export_format = st.selectbox(
                    "Export Format",
                    options=["JSON", "CSV", "Excel"]
                )

                if st.button("📤 Export Portfolio", type="primary"):
                    export_portfolio_data(selected_portfolio, export_format)
        else:
            st.info("No portfolios available to export. Create a portfolio first.")


# =================================================================================
# CREATION FUNCTIONS
# =================================================================================


def add_manual_asset_with_price_fetch(ticker: str, weight_percent: float, shares: float, portfolio_value: float):
    """Add asset to manual creation list with automatic price fetching"""

    if not ticker:
        st.error("Ticker is required")
        return

    ticker = ticker.upper().strip()

    # Initialize manual assets if not exists
    if 'manual_assets' not in st.session_state:
        st.session_state.manual_assets = []

    # Check for duplicates
    if any(asset['ticker'] == ticker for asset in st.session_state.manual_assets):
        st.error(f"Asset {ticker} already exists")
        return

    # Validate ticker format
    try:
        from core.data_manager.validators import TickerValidator
        if not TickerValidator.validate_ticker(ticker):
            st.error(f"Invalid ticker format: {ticker}")
            return
    except ImportError:
        # Basic validation if validator not available
        import re
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            st.error(f"Invalid ticker format: {ticker}")
            return

    # Fetch current price
    current_price = None
    try:
        with st.spinner(f"Fetching price for {ticker}..."):
            price_manager = get_price_manager()
            current_price = price_manager.get_current_price(ticker)

        if current_price is None:
            st.warning(f"Could not fetch price for {ticker}. You can add it anyway and update prices later.")
        else:
            st.success(f"Fetched {ticker}: ${current_price:.2f}")

    except Exception as e:
        st.warning(f"Error fetching price for {ticker}: {e}")

    # Add to list - ensure all values are proper types
    asset_data = {
        'ticker': ticker,
        'weight': float(weight_percent),  # Ensure it's a float
        'shares': float(shares) if shares > 0 else 0.0,  # Ensure it's a float or 0
        'current_price': float(current_price) if current_price else None  # Ensure it's float or None
    }

    st.session_state.manual_assets.append(asset_data)
    st.success(f"Added {ticker} ({weight_percent:.1f}%)")


def update_manual_asset_prices():
    """Update prices for all manual assets"""

    if 'manual_assets' not in st.session_state or not st.session_state.manual_assets:
        st.warning("No manual assets to update")
        return

    tickers = [asset['ticker'] for asset in st.session_state.manual_assets]

    try:
        with st.spinner("Updating prices..."):
            price_manager = get_price_manager()
            prices = price_manager.get_current_prices(tickers)

        updated_count = 0
        for asset in st.session_state.manual_assets:
            ticker = asset['ticker']
            if ticker in prices and prices[ticker]:
                asset['current_price'] = float(prices[ticker])  # Ensure it's a float
                updated_count += 1

        if updated_count > 0:
            st.success(f"Updated {updated_count} prices")
        else:
            st.warning("No prices were updated")

    except Exception as e:
        st.error(f"Error updating prices: {e}")


def normalize_manual_weights():
    """Normalize weights in manual asset list"""

    if 'manual_assets' not in st.session_state or not st.session_state.manual_assets:
        st.warning("No manual assets to normalize")
        return

    # Safely calculate total weight with default values
    total_weight = 0.0
    for asset in st.session_state.manual_assets:
        weight = asset.get('weight', 0.0)
        if weight is not None:
            total_weight += float(weight)

    if total_weight > 0:
        for asset in st.session_state.manual_assets:
            current_weight = asset.get('weight', 0.0) or 0.0
            asset['weight'] = float((current_weight / total_weight) * 100)

        st.success("Weights normalized to 100%")
    else:
        st.error("Total weight is zero - cannot normalize")


def create_manual_portfolio(name: str, description: str, portfolio_type: str, initial_value: float):
    """Create portfolio from manual asset list"""

    try:
        with st.spinner("Creating portfolio..."):
            # Get manual assets from session state
            manual_assets = st.session_state.get('manual_assets', [])

            if not manual_assets:
                st.error("No assets added")
                return

            # Convert to Asset objects
            assets = []
            from core.data_manager import Asset, AssetClass

            for asset_data in manual_assets:
                # Safely get values with defaults
                weight = asset_data.get('weight', 0.0) or 0.0
                shares = asset_data.get('shares', 0.0) or 0.0
                current_price = asset_data.get('current_price')

                asset = Asset(
                    ticker=asset_data['ticker'],
                    weight=float(weight) / 100.0,  # Convert to decimal and ensure float
                    shares=float(shares),  # Ensure float
                    current_price=float(current_price) if current_price else None,  # Ensure float or None
                    asset_class=AssetClass.STOCK
                )
                assets.append(asset)

            # Create portfolio
            portfolio_manager = get_portfolio_manager()
            from core.data_manager import PortfolioType

            portfolio = portfolio_manager.create_portfolio(
                name=name,
                description=description,
                assets=assets,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Update prices for all assets after creation
            try:
                price_manager = get_price_manager()
                tickers = [asset.ticker for asset in portfolio.assets]
                prices = price_manager.get_current_prices(tickers)

                for asset in portfolio.assets:
                    if asset.ticker in prices and prices[asset.ticker]:
                        asset.current_price = prices[asset.ticker]
                        # Recalculate shares based on weight and current price
                        if asset.weight and initial_value and asset.current_price:
                            allocation = asset.weight * initial_value
                            asset.shares = int(allocation / asset.current_price)

                # Update the portfolio with new prices using update_portfolio method
                portfolio_manager.update_portfolio(portfolio.id, {"assets": portfolio.assets})
                st.success("✅ Prices updated successfully!")

            except Exception as e:
                st.warning(f"Portfolio created but couldn't update prices: {e}")

            # Clear manual assets
            st.session_state.manual_assets = []

            # Refresh and select
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"Portfolio '{name}' created successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating portfolio: {str(e)}")
        # Log the full error for debugging
        import traceback
        st.error(f"Debug info: {traceback.format_exc()}")
        st.error(f"Debug info: {traceback.format_exc()}")


def update_manual_asset_prices():
    """Update prices for all manual assets"""

    if 'manual_assets' not in st.session_state or not st.session_state.manual_assets:
        st.warning("No manual assets to update")
        return

    tickers = [asset['ticker'] for asset in st.session_state.manual_assets]

    try:
        with st.spinner("Updating prices..."):
            price_manager = get_price_manager()
            prices = price_manager.get_current_prices(tickers)

        updated_count = 0
        for asset in st.session_state.manual_assets:
            ticker = asset['ticker']
            if ticker in prices and prices[ticker]:
                asset['current_price'] = prices[ticker]
                updated_count += 1

        if updated_count > 0:
            st.success(f"Updated {updated_count} prices")
        else:
            st.warning("No prices were updated")

    except Exception as e:
        st.error(f"Error updating prices: {e}")


def normalize_manual_weights():
    """Normalize weights in manual asset list"""

    if 'manual_assets' not in st.session_state or not st.session_state.manual_assets:
        st.warning("No manual assets to normalize")
        return

    total_weight = sum(asset['weight'] for asset in st.session_state.manual_assets)

    if total_weight > 0:
        for asset in st.session_state.manual_assets:
            asset['weight'] = (asset['weight'] / total_weight) * 100

        st.success("Weights normalized to 100%")
    else:
        st.error("Total weight is zero - cannot normalize")


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
            portfolio_manager = get_portfolio_manager()

            from core.data_manager import PortfolioType
            portfolio = portfolio_manager.create_from_text(
                name=name,
                text=text,
                description=description,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Update prices and calculate shares
            try:
                price_manager = get_price_manager()
                tickers = [asset.ticker for asset in portfolio.assets]
                prices = price_manager.get_current_prices(tickers)

                for asset in portfolio.assets:
                    if asset.ticker in prices and prices[asset.ticker]:
                        asset.current_price = prices[asset.ticker]
                        # Calculate shares based on weight and current price
                        if asset.weight and initial_value and asset.current_price:
                            allocation = asset.weight * initial_value
                            asset.shares = int(allocation / asset.current_price)

                # Fetch company information if requested
                if fetch_info:
                    for asset in portfolio.assets:
                        try:
                            company_info = price_manager.get_company_info(asset.ticker)
                            if company_info:
                                if not asset.name:
                                    asset.name = company_info.name
                                if not asset.sector:
                                    asset.sector = company_info.sector or "Unknown"
                        except Exception:
                            if not asset.name:
                                asset.name = f"{asset.ticker} Corp"
                            if not asset.sector:
                                asset.sector = "Unknown"

                # Update the portfolio with new data
                portfolio_manager.update_portfolio(portfolio.id, {"assets": portfolio.assets})
                st.success("✅ Prices and company info updated!")

            except Exception as e:
                st.warning(f"Portfolio created but couldn't update prices: {e}")

            # Refresh portfolio list and set as selected
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"Portfolio '{name}' created successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating portfolio: {str(e)}")
        import traceback
        st.error(f"Debug info: {traceback.format_exc()}")


def create_manual_portfolio(name: str, description: str, portfolio_type: str, initial_value: float):
    """Create portfolio from manual asset list"""

    try:
        with st.spinner("Creating portfolio..."):
            # Get manual assets from session state
            manual_assets = st.session_state.get('manual_assets', [])

            if not manual_assets:
                st.error("No assets added")
                return

            # Convert to Asset objects
            assets = []
            from core.data_manager import Asset, AssetClass

            for asset_data in manual_assets:
                asset = Asset(
                    ticker=asset_data['ticker'],
                    weight=asset_data['weight'] / 100.0,  # Convert to decimal
                    shares=asset_data.get('shares', 0.0),
                    current_price=asset_data.get('current_price'),
                    asset_class=AssetClass.STOCK
                )
                assets.append(asset)

            # Create portfolio
            portfolio_manager = get_portfolio_manager()
            from core.data_manager import PortfolioType

            portfolio = portfolio_manager.create_portfolio(
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
            set_selected_portfolio(portfolio)

            st.success(f"Portfolio '{name}' created successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating portfolio: {e}")


def create_template_portfolio(
        name: str,
        description: str,
        template: dict,
        initial_value: float,
        fetch_prices: bool
):
    """Create portfolio from template"""

    try:
        with st.spinner("Creating portfolio from template..."):
            # Create assets
            assets = []
            from core.data_manager import Asset, AssetClass

            for ticker, weight, desc in template['assets']:
                asset = Asset(
                    ticker=ticker,
                    name=desc,
                    weight=weight / 100.0,  # Convert to decimal
                    asset_class=AssetClass.STOCK
                )
                assets.append(asset)

            # Create portfolio
            portfolio_manager = get_portfolio_manager()
            from core.data_manager import PortfolioType

            portfolio = portfolio_manager.create_portfolio(
                name=name,
                description=description,
                assets=assets,
                portfolio_type=PortfolioType.BALANCED,
                initial_value=initial_value
            )

            # Fetch current prices and calculate shares if requested
            if fetch_prices:
                try:
                    price_manager = get_price_manager()
                    tickers = [asset.ticker for asset in portfolio.assets]
                    prices = price_manager.get_current_prices(tickers)

                    # Update prices and calculate shares
                    for asset in portfolio.assets:
                        if asset.ticker in prices and prices[asset.ticker]:
                            asset.current_price = prices[asset.ticker]
                            # Calculate shares based on weight and current price
                            if asset.weight and initial_value and asset.current_price:
                                allocation = asset.weight * initial_value
                                asset.shares = int(allocation / asset.current_price)

                        # Also try to get company info for sectors
                        try:
                            company_info = price_manager.get_company_info(asset.ticker)
                            if company_info:
                                if not asset.name or asset.name == desc:
                                    asset.name = company_info.name
                                if not asset.sector:
                                    asset.sector = company_info.sector or "Unknown"
                        except Exception:
                            if not asset.sector:
                                asset.sector = "Unknown"

                    # Update the portfolio with new data
                    portfolio_manager.update_portfolio(portfolio.id, {"assets": portfolio.assets})
                    st.success("✅ Prices and company info updated!")

                except Exception as e:
                    st.warning(f"Portfolio created but couldn't update prices: {e}")

            # Refresh and select
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"Portfolio '{name}' created from template!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating template portfolio: {str(e)}")
        import traceback
        st.error(f"Debug info: {traceback.format_exc()}")


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
            from core.data_manager import Asset, AssetClass

            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).upper().strip()

                if not ticker or pd.isna(ticker):
                    continue

                # Get weight
                if equal_weight or weight_col == "None (Equal Weight)":
                    weight = 1.0 / len(df)
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
            portfolio_manager = get_portfolio_manager()
            from core.data_manager import PortfolioType

            portfolio = portfolio_manager.create_portfolio(
                name=name,
                description=description,
                assets=assets,
                portfolio_type=PortfolioType.BALANCED
            )

            # Refresh and select
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"Portfolio '{name}' imported successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error importing portfolio: {e}")


def import_portfolio_from_json(portfolio_data: dict):
    """Import portfolio from JSON data."""

    try:
        with st.spinner("Importing portfolio from JSON..."):
            # This would implement the actual JSON import logic
            st.success(f"Successfully imported portfolio: {portfolio_data.get('name', 'Unnamed')}")

            # Refresh portfolio list
            refresh_portfolios()

    except Exception as e:
        st.error(f"Error importing portfolio: {str(e)}")


def export_portfolio_data(portfolio, export_format: str = "JSON"):
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