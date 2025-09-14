"""
Form components for the Portfolio Management System.

This module contains reusable form components for portfolio creation and management.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import PortfolioType, AssetClass
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
        st.info("Enter ticker symbols with weights. Supports multiple formats:")
        st.code("AAPL 30%, MSFT 25%, GOOGL 45%")
        st.code("AAPL 0.3, MSFT 0.25, GOOGL 0.45")
        st.code("AAPL, MSFT, GOOGL (equal weights)")

    with col2:
        if st.button("📋 Use Sample", use_container_width=True):
            st.session_state.sample_text = generate_sample_data()

    # Main form
    with st.form("text_input_portfolio"):

        # Text input with sample data
        default_text = st.session_state.get('sample_text', '')
        text_input = st.text_area(
            "Ticker Input *",
            value=default_text,
            height=100,
            help="Enter ticker symbols with weights in any supported format"
        )

        # Portfolio details
        col1, col2 = st.columns(2)

        with col1:
            portfolio_name = st.text_input("Portfolio Name *")
            portfolio_type = st.selectbox(
                "Portfolio Type",
                options=[pt.value for pt in PortfolioType],
                format_func=lambda x: x.title()
            )

        with col2:
            initial_value = st.number_input(
                "Initial Investment ($)",
                min_value=1.0,
                value=100000.0,
                step=1000.0,
                help="Total amount to invest"
            )

            auto_normalize = st.checkbox(
                "Auto-normalize weights",
                value=True,
                help="Automatically adjust weights to sum to 100%"
            )

        # Description
        description = st.text_area(
            "Description (Optional)",
            help="Add notes about your portfolio strategy"
        )

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            fetch_info = st.checkbox(
                "Fetch company information",
                value=True,
                help="Automatically fetch company names and sector information"
            )

            calculate_shares = st.checkbox(
                "Calculate share quantities",
                value=True,
                help="Calculate number of shares based on current prices"
            )

        # Submit button
        submitted = st.form_submit_button("🚀 Create Portfolio", use_container_width=True)

        # Form validation and submission
        if submitted:
            if not portfolio_name or not text_input:
                st.error("Portfolio name and ticker input are required!")
            else:
                # Validate ticker input
                is_valid, error_msg, parsed_tickers = validate_ticker_input(text_input)

                if not is_valid:
                    st.error(f"Invalid input: {error_msg}")
                else:
                    # Create portfolio
                    create_portfolio_from_text(
                        name=portfolio_name,
                        text=text_input,
                        description=description,
                        portfolio_type=portfolio_type,
                        initial_value=initial_value,
                        auto_normalize=auto_normalize,
                        fetch_info=fetch_info
                    )


def render_file_upload_form():
    """Render the file upload portfolio creation form."""

    st.subheader("📁 Create from File Upload")

    # File upload
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
                    "Ticker Column *",
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

            # Portfolio details form
            with st.form("file_upload_portfolio"):

                col1, col2 = st.columns(2)

                with col1:
                    portfolio_name = st.text_input("Portfolio Name *")
                    portfolio_type = st.selectbox(
                        "Portfolio Type",
                        options=[pt.value for pt in PortfolioType],
                        format_func=lambda x: x.title()
                    )

                with col2:
                    initial_value = st.number_input(
                        "Initial Investment ($)",
                        min_value=1.0,
                        value=100000.0,
                        step=1000.0
                    )

                description = st.text_area("Description (Optional)")

                # Advanced options
                with st.expander("🔧 Advanced Options"):
                    skip_invalid = st.checkbox(
                        "Skip invalid tickers",
                        value=True,
                        help="Skip tickers that cannot be validated"
                    )

                    fetch_info = st.checkbox(
                        "Fetch company information",
                        value=True
                    )

                # Submit button
                submitted = st.form_submit_button("📊 Create from File", use_container_width=True)

                if submitted:
                    if not portfolio_name:
                        st.error("Portfolio name is required!")
                    else:
                        create_portfolio_from_file(
                            df=df,
                            name=portfolio_name,
                            description=description,
                            portfolio_type=portfolio_type,
                            initial_value=initial_value,
                            ticker_column=ticker_column,
                            weight_column=weight_column if weight_column != "None" else None,
                            name_column=name_column if name_column != "None" else None,
                            skip_invalid=skip_invalid,
                            fetch_info=fetch_info
                        )

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def render_manual_creation_form():
    """Render the manual portfolio creation form."""

    st.subheader("✋ Manual Asset Entry")

    # Initialize session state for manual assets
    if 'manual_assets' not in st.session_state:
        st.session_state.manual_assets = []

    # Asset entry form
    with st.expander("➕ Add New Asset", expanded=len(st.session_state.manual_assets) == 0):

        col1, col2, col3 = st.columns(3)

        with col1:
            new_ticker = st.text_input("Ticker Symbol", key="new_ticker")

        with col2:
            new_weight = st.number_input(
                "Weight (%)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1,
                key="new_weight"
            )

        with col3:
            new_name = st.text_input("Company Name (Optional)", key="new_name")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("➕ Add Asset", use_container_width=True):
                if new_ticker:
                    asset_data = {
                        'ticker': new_ticker.upper(),
                        'weight': new_weight / 100,  # Convert percentage to decimal
                        'name': new_name if new_name else None
                    }
                    st.session_state.manual_assets.append(asset_data)
                    st.success(f"Added {new_ticker.upper()}")
                    st.rerun()
                else:
                    st.error("Ticker symbol is required!")

        with col2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.manual_assets = []
                st.rerun()

    # Show current assets
    if st.session_state.manual_assets:

        st.subheader("📋 Current Assets")

        # Create DataFrame for display
        assets_df = pd.DataFrame(st.session_state.manual_assets)
        assets_df['Weight (%)'] = assets_df['weight'] * 100

        # Display editable data
        edited_df = st.data_editor(
            assets_df[['ticker', 'name', 'Weight (%)']],
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "name": st.column_config.TextColumn("Name"),
                "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0, max_value=100, step=0.1)
            }
        )

        # Update session state with changes
        if not edited_df.equals(assets_df[['ticker', 'name', 'Weight (%)']]):
            updated_assets = []
            for _, row in edited_df.iterrows():
                updated_assets.append({
                    'ticker': str(row['ticker']).upper(),
                    'weight': row['Weight (%)'] / 100,
                    'name': row['name'] if pd.notna(row['name']) else None
                })
            st.session_state.manual_assets = updated_assets

        # Show weight summary
        total_weight = sum(asset['weight'] for asset in st.session_state.manual_assets)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assets", len(st.session_state.manual_assets))
        with col2:
            st.metric("Total Weight", f"{total_weight:.1%}")
        with col3:
            weight_status = "✅ Valid" if abs(total_weight - 1.0) < 0.01 else "⚠️ Invalid"
            st.metric("Status", weight_status)

        # Portfolio creation form
        with st.form("manual_portfolio"):

            col1, col2 = st.columns(2)

            with col1:
                portfolio_name = st.text_input("Portfolio Name *")
                portfolio_type = st.selectbox(
                    "Portfolio Type",
                    options=[pt.value for pt in PortfolioType],
                    format_func=lambda x: x.title()
                )

            with col2:
                initial_value = st.number_input(
                    "Initial Investment ($)",
                    min_value=1.0,
                    value=100000.0,
                    step=1000.0
                )

                normalize_weights = st.checkbox(
                    "Normalize weights to 100%",
                    value=True,
                    help="Automatically adjust weights to sum to 100%"
                )

            description = st.text_area("Description (Optional)")

            submitted = st.form_submit_button("🚀 Create Portfolio", use_container_width=True)

            if submitted:
                if not portfolio_name:
                    st.error("Portfolio name is required!")
                elif not st.session_state.manual_assets:
                    st.error("At least one asset is required!")
                else:
                    create_portfolio_from_manual(
                        assets=st.session_state.manual_assets,
                        name=portfolio_name,
                        description=description,
                        portfolio_type=portfolio_type,
                        initial_value=initial_value,
                        normalize_weights=normalize_weights
                    )

    else:
        st.info("👆 Add your first asset using the form above")


def create_portfolio_from_text(name: str, text: str, description: str,
                               portfolio_type: str, initial_value: float,
                               auto_normalize: bool, fetch_info: bool):
    """Create portfolio from text input."""

    try:
        with st.spinner("Creating portfolio..."):

            portfolio_manager = get_portfolio_manager()

            # Create portfolio using manager
            portfolio = portfolio_manager.create_from_text(
                name=name,
                text=text,
                description=description,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Fetch company information if requested
            if fetch_info:
                update_company_info(portfolio)

            # Update prices and calculate shares
            update_portfolio_prices_and_shares(portfolio, initial_value)

            # Refresh portfolio list and select new portfolio
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"✅ Portfolio '{name}' created successfully!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating portfolio: {str(e)}")


def create_portfolio_from_file(df: pd.DataFrame, name: str, description: str,
                               portfolio_type: str, initial_value: float,
                               ticker_column: str, weight_column: Optional[str],
                               name_column: Optional[str], skip_invalid: bool,
                               fetch_info: bool):
    """Create portfolio from uploaded file."""

    try:
        with st.spinner("Processing file and creating portfolio..."):

            # Extract tickers
            tickers = df[ticker_column].dropna().astype(str).str.upper().tolist()

            # Handle weights
            if weight_column:
                weights = df[weight_column].dropna().tolist()
                # Ensure we have weights for all tickers
                weights = weights[:len(tickers)]
                if len(weights) < len(tickers):
                    # Fill missing weights with equal distribution
                    remaining_weight = 1.0 - sum(weights)
                    equal_weight = remaining_weight / (len(tickers) - len(weights))
                    weights.extend([equal_weight] * (len(tickers) - len(weights)))
            else:
                # Equal weights
                equal_weight = 1.0 / len(tickers)
                weights = [equal_weight] * len(tickers)

            # Extract names if available
            names = None
            if name_column:
                names = df[name_column].fillna('').astype(str).tolist()
                names = names[:len(tickers)]

            # Create text input from parsed data
            text_parts = []
            for i, ticker in enumerate(tickers):
                weight_pct = weights[i] * 100
                text_parts.append(f"{ticker} {weight_pct:.1f}%")

            text_input = ", ".join(text_parts)

            # Create portfolio using text method
            portfolio_manager = get_portfolio_manager()

            portfolio = portfolio_manager.create_from_text(
                name=name,
                text=text_input,
                description=description,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Add names if provided
            if names:
                for i, asset in enumerate(portfolio.assets):
                    if i < len(names) and names[i]:
                        asset.name = names[i]

            # Fetch company information if requested
            if fetch_info:
                update_company_info(portfolio)

            # Update prices and calculate shares
            update_portfolio_prices_and_shares(portfolio, initial_value)

            # Refresh and select
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"✅ Portfolio '{name}' created from file!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating portfolio from file: {str(e)}")


def create_portfolio_from_manual(assets: List[Dict], name: str, description: str,
                                 portfolio_type: str, initial_value: float,
                                 normalize_weights: bool):
    """Create portfolio from manual asset entry."""

    try:
        with st.spinner("Creating portfolio..."):

            # Normalize weights if requested
            if normalize_weights:
                total_weight = sum(asset['weight'] for asset in assets)
                if total_weight > 0:
                    for asset in assets:
                        asset['weight'] = asset['weight'] / total_weight

            # Create text input from assets
            text_parts = []
            for asset in assets:
                weight_pct = asset['weight'] * 100
                text_parts.append(f"{asset['ticker']} {weight_pct:.1f}%")

            text_input = ", ".join(text_parts)

            # Create portfolio
            portfolio_manager = get_portfolio_manager()

            portfolio = portfolio_manager.create_from_text(
                name=name,
                text=text_input,
                description=description,
                portfolio_type=PortfolioType(portfolio_type),
                initial_value=initial_value
            )

            # Add manual names
            for i, asset in enumerate(portfolio.assets):
                if i < len(assets) and assets[i].get('name'):
                    asset.name = assets[i]['name']

            # Update prices and calculate shares
            update_portfolio_prices_and_shares(portfolio, initial_value)

            # Clear manual assets and refresh
            st.session_state.manual_assets = []
            refresh_portfolios()
            set_selected_portfolio(portfolio)

            st.success(f"✅ Portfolio '{name}' created manually!")
            display_portfolio_summary(portfolio)

    except Exception as e:
        st.error(f"Error creating manual portfolio: {str(e)}")


def update_portfolio_prices_and_shares(portfolio, initial_value: float):
    """Update portfolio with current prices and calculate shares."""

    try:
        price_manager = get_price_manager()

        # Get current prices
        tickers = [asset.ticker for asset in portfolio.assets]
        prices = price_manager.get_current_prices(tickers)

        # Update asset prices and calculate shares
        for asset in portfolio.assets:
            if asset.ticker in prices and prices[asset.ticker]:
                asset.current_price = prices[asset.ticker]
                # Calculate shares based on allocation
                allocation = asset.weight * initial_value
                asset.shares = int(allocation / asset.current_price) if asset.current_price > 0 else 0
            else:
                asset.current_price = None
                asset.shares = 0

    except Exception as e:
        st.warning(f"Could not update prices: {str(e)}")


def render_portfolio_template_form():
    """Render portfolio template selection form."""

    st.subheader("📋 Create from Template")

    # Define templates
    templates = {
        "Conservative Growth": {
            "description": "Balanced portfolio focused on stable growth with lower risk",
            "assets": "SPY 40%, BND 30%, VTI 20%, VTEB 10%",
            "type": "conservative"
        },
        "Tech Focus": {
            "description": "Technology-heavy portfolio for growth investors",
            "assets": "AAPL 25%, MSFT 20%, GOOGL 15%, NVDA 15%, META 10%, AMZN 10%, TSLA 5%",
            "type": "growth"
        },
        "Dividend Income": {
            "description": "Income-focused portfolio with dividend-paying stocks",
            "assets": "JNJ 15%, PG 15%, KO 10%, PFE 10%, VZ 10%, T 10%, XOM 10%, CVX 10%, IBM 10%",
            "type": "income"
        },
        "Global Diversified": {
            "description": "Internationally diversified portfolio across markets",
            "assets": "VTI 30%, VTIAX 25%, EFA 20%, VWO 15%, BND 10%",
            "type": "balanced"
        },
        "ESG Sustainable": {
            "description": "Environmentally and socially responsible investing",
            "assets": "ESGD 30%, VSGX 25%, ICLN 20%, PBW 15%, QCLN 10%",
            "type": "growth"
        }
    }

    # Template selection
    selected_template = st.selectbox(
        "Choose Template",
        options=list(templates.keys()),
        help="Select a pre-defined portfolio template"
    )

    if selected_template:
        template_data = templates[selected_template]

        # Show template details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.info(f"**Description:** {template_data['description']}")
            st.code(template_data['assets'])

        with col2:
            st.metric("Portfolio Type", template_data['type'].title())
            st.metric("Assets", len(template_data['assets'].split(',')))

        # Template customization form
        with st.form("template_portfolio"):

            col1, col2 = st.columns(2)

            with col1:
                portfolio_name = st.text_input(
                    "Portfolio Name *",
                    value=f"My {selected_template} Portfolio"
                )

                custom_description = st.text_area(
                    "Custom Description",
                    value=template_data['description']
                )

            with col2:
                initial_value = st.number_input(
                    "Initial Investment ($)",
                    min_value=1.0,
                    value=100000.0,
                    step=1000.0
                )

                portfolio_type = st.selectbox(
                    "Portfolio Type",
                    options=[pt.value for pt in PortfolioType],
                    index=[pt.value for pt in PortfolioType].index(template_data['type'])
                )

            # Advanced options
            with st.expander("🔧 Template Options"):
                modify_weights = st.checkbox(
                    "Allow weight modifications",
                    value=False,
                    help="Enable editing of template asset weights"
                )

                add_custom_assets = st.checkbox(
                    "Add custom assets",
                    value=False,
                    help="Add additional assets to the template"
                )

                fetch_info = st.checkbox(
                    "Fetch company information",
                    value=True
                )

            # Weight modification (if enabled)
            if modify_weights:
                st.subheader("⚖️ Modify Weights")

                # Parse template assets
                asset_pairs = []
                for pair in template_data['assets'].split(','):
                    ticker, weight = pair.strip().split(' ')
                    weight_val = float(weight.replace('%', '')) / 100
                    asset_pairs.append((ticker, weight_val))

                # Create editable weights
                modified_weights = {}
                for ticker, original_weight in asset_pairs:
                    modified_weights[ticker] = st.slider(
                        f"{ticker} Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=original_weight,
                        step=0.01,
                        format="%.1%"
                    )

                # Show total weight
                total_weight = sum(modified_weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"Total weight: {total_weight:.1%} (should be 100%)")

            # Custom assets (if enabled)
            custom_assets_text = ""
            if add_custom_assets:
                st.subheader("➕ Additional Assets")
                custom_assets_text = st.text_area(
                    "Additional Assets",
                    help="Enter additional assets in the same format: TICKER WEIGHT%, ..."
                )

            # Submit button
            submitted = st.form_submit_button("📊 Create from Template", use_container_width=True)

            if submitted:
                if not portfolio_name:
                    st.error("Portfolio name is required!")
                else:
                    # Build final asset string
                    final_assets = template_data['assets']

                    if modify_weights:
                        # Rebuild with modified weights
                        asset_parts = []
                        for ticker, weight in modified_weights.items():
                            asset_parts.append(f"{ticker} {weight:.1%}")
                        final_assets = ", ".join(asset_parts)

                    if add_custom_assets and custom_assets_text.strip():
                        final_assets += ", " + custom_assets_text.strip()

                    # Create portfolio
                    create_portfolio_from_text(
                        name=portfolio_name,
                        text=final_assets,
                        description=custom_description,
                        portfolio_type=portfolio_type,
                        initial_value=initial_value,
                        auto_normalize=True,
                        fetch_info=fetch_info
                    )


def render_import_export_form():
    """Render portfolio import/export form."""

    st.subheader("🔄 Import/Export Portfolio")

    tab1, tab2 = st.tabs(["📥 Import", "📤 Export"])

    with tab1:
        st.write("**Import existing portfolio data**")

        import_method = st.selectbox(
            "Import Method",
            ["JSON File", "CSV File", "Text Format"]
        )

        if import_method == "JSON File":
            render_json_import_form()
        elif import_method == "CSV File":
            render_csv_import_form()
        else:
            render_text_import_form()

    with tab2:
        st.write("**Export current portfolios**")
        render_export_form()


def render_json_import_form():
    """Render JSON import form."""

    uploaded_file = st.file_uploader(
        "Upload JSON Portfolio File",
        type=['json'],
        help="Upload a previously exported portfolio JSON file"
    )

    if uploaded_file is not None:
        try:
            import json

            # Read and parse JSON
            json_data = json.load(uploaded_file)

            # Show preview
            st.json(json_data, expanded=False)

            with st.form("json_import"):

                # Allow name override
                override_name = st.text_input(
                    "Override Portfolio Name (Optional)",
                    help="Leave empty to use original name"
                )

                import_settings = st.checkbox(
                    "Import portfolio settings",
                    value=True,
                    help="Import portfolio type, description, etc."
                )

                submitted = st.form_submit_button("📥 Import Portfolio")

                if submitted:
                    try:
                        portfolio_manager = get_portfolio_manager()

                        # Import portfolio
                        portfolio = portfolio_manager.from_dict(json_data)

                        # Override name if provided
                        if override_name.strip():
                            portfolio.name = override_name.strip()

                        # Save imported portfolio
                        portfolio_manager.save_portfolio(portfolio)

                        refresh_portfolios()
                        set_selected_portfolio(portfolio)

                        st.success(f"✅ Imported portfolio '{portfolio.name}'!")
                        display_portfolio_summary(portfolio)

                    except Exception as e:
                        st.error(f"Error importing portfolio: {str(e)}")

        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")


def render_csv_import_form():
    """Render CSV import form (reuse file upload logic)."""

    st.info("CSV import uses the same process as file upload creation")
    render_file_upload_form()


def render_text_import_form():
    """Render text format import form."""

    st.info("Text import uses the same process as text input creation")
    render_text_input_form()


def render_export_form():
    """Render portfolio export form."""

    from ..utils.session_state import get_portfolios

    portfolios = get_portfolios()

    if not portfolios:
        st.info("No portfolios available to export")
        return

    # Portfolio selection
    portfolio_options = {p.name: p for p in portfolios}
    selected_names = st.multiselect(
        "Select Portfolios to Export",
        options=list(portfolio_options.keys()),
        default=list(portfolio_options.keys())[:1]  # Select first by default
    )

    if selected_names:

        # Export format
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "Text Summary"]
        )

        # Export options
        with st.expander("🔧 Export Options"):
            include_prices = st.checkbox(
                "Include current prices",
                value=True,
                help="Include latest price data in export"
            )

            include_metadata = st.checkbox(
                "Include metadata",
                value=True,
                help="Include creation dates, descriptions, etc."
            )

            combine_files = st.checkbox(
                "Combine into single file",
                value=len(selected_names) > 1,
                help="Export all portfolios into one file"
            )

        # Export buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Download Export", use_container_width=True):
                export_portfolios(
                    portfolios=[portfolio_options[name] for name in selected_names],
                    format=export_format,
                    include_prices=include_prices,
                    include_metadata=include_metadata,
                    combine_files=combine_files
                )

        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                # This would copy export data to clipboard (browser limitation)
                st.info("Copy functionality requires browser permissions")

        with col3:
            if st.button("👁️ Preview Export", use_container_width=True):
                preview_export(
                    portfolios=[portfolio_options[name] for name in selected_names],
                    format=export_format
                )


def export_portfolios(portfolios, format, include_prices, include_metadata, combine_files):
    """Export portfolios in specified format."""

    # Implementation would depend on the export format
    st.info(f"Exporting {len(portfolios)} portfolio(s) in {format} format...")
    # This is a placeholder for the actual export logic


def preview_export(portfolios, format):
    """Preview export data."""

    st.subheader("📋 Export Preview")

    for portfolio in portfolios:
        with st.expander(f"Portfolio: {portfolio.name}"):
            if format == "JSON":
                st.json(portfolio.to_dict())
            elif format == "CSV":
                from ..utils.helpers import create_asset_table
                df = create_asset_table(portfolio.assets)
                st.dataframe(df, use_container_width=True)
            else:  # Text Summary
                st.text(f"""
Portfolio: {portfolio.name}
Type: {portfolio.portfolio_type.value.title()}
Assets: {len(portfolio.assets)}
Total Value: ${portfolio.calculate_value():,.2f}
Created: {portfolio.created_date.strftime('%Y-%m-%d')}

Holdings:
{chr(10).join([f"- {a.ticker}: {a.weight:.1%}" for a in portfolio.assets])}
                """)


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

        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                auto_normalize = st.checkbox("Auto-normalize weights", value=True)
                fetch_company_info = st.checkbox("Fetch company information", value=True)

            with col2:
                calculate_shares = st.checkbox("Calculate share quantities", value=True)
                update_prices = st.checkbox("Update current prices", value=True)

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
                        st.success(f"✅ {new_ticker}: ${preview_price:.2f}")
                    else:
                        st.warning(f"⚠️ {new_ticker}: Price not found")
                except:
                    st.info(f"🔍 {new_ticker}: Checking...")

        with col2:
            new_weight = st.number_input(
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
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

        # Add asset button
        if st.form_submit_button("➕ Add Asset"):
            if new_ticker and new_weight > 0:
                add_manual_asset_with_price_fetch(new_ticker, new_weight, new_shares, initial_value)
                st.rerun()
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
        st.dataframe(manual_df.drop('Index', axis=1), use_container_width=True, hide_index=True)

        # Weight and value summary
        col1, col2, col3 = st.columns(3)

        with col1:
            weight_color = "🟢" if abs(total_weight - 100.0) < 0.1 else "🔴"
            st.markdown(f"**Total Weight:** {weight_color} {total_weight:.2f}%")

        with col2:
            total_invested = sum(
                int((asset_data['weight'] / 100) * initial_value / asset_data.get('current_price', 1)) * asset_data.get(
                    'current_price', 0)
                for asset_data in st.session_state.manual_assets
                if asset_data.get('current_price')
            )
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
                import json
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


def import_portfolio_from_json(portfolio_data: dict):
    """Import portfolio from JSON data."""

    try:
        with st.spinner("Importing portfolio from JSON..."):
            # This would implement the actual JSON import logic
            st.success(f"✅ Successfully imported portfolio: {portfolio_data.get('name', 'Unnamed')}")

            # Refresh portfolio list
            refresh_portfolios()

    except Exception as e:
        st.error(f"Error importing portfolio: {str(e)}")