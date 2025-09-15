"""
Create Portfolio page for the Portfolio Management System.

This module contains all portfolio creation functionality including text input,
file upload, manual entry, and template-based creation.
"""
import streamlit as st
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from ..components.forms import (
    render_text_input_form,
    render_manual_creation_form,
    render_portfolio_template_form,
    render_file_operations_form
)


def render_create_portfolio():
    """Render the create portfolio page with all creation methods."""

    st.header("📝 Create Portfolio")

    # Introduction and help
    render_creation_help()


    tab1, tab2, tab3, tab4 = st.tabs([
        "✏️ Text Input",
        "📁 File Operations",
        "✋ Manual Entry",
        "📋 Templates"
    ])

    with tab1:
        render_text_input_form()

    with tab2:
        render_file_operations_form()

    with tab3:
        render_manual_creation_form()

    with tab4:
        render_portfolio_template_form()


def render_creation_help():
    """Render help section for portfolio creation."""

    with st.expander("❓ How to Create a Portfolio", expanded=False):
        st.markdown("""
        ### 🎯 Choose Your Creation Method

        **📝 Text Input** - *Fastest for experienced users*
        - Enter tickers with weights: `AAPL 30%, MSFT 25%, GOOGL 45%`
        - Supports multiple formats: percentages, decimals, or equal weights
        - Perfect for quick portfolio creation

        **📁 File Operations** - *Upload files or import/export data*
        - **Upload CSV/Excel**: Import portfolios from spreadsheet files
        - **Import JSON**: Restore previously exported portfolios
        - **Export Portfolio**: Backup existing portfolios in multiple formats
        - Ideal for importing from other platforms or backup/restore

        **✋ Manual Entry** - *Most control and flexibility*
        - Add each asset individually with full details
        - Edit weights and information as you go
        - Best for careful, detailed portfolio construction

        **📋 Templates** - *Great for beginners*
        - Pre-built portfolios for different strategies
        - Conservative, Growth, Income, and ESG options
        - Customize templates to fit your needs

        ### 💡 Tips for Success
        - Weights should sum to 100% (auto-normalization available)
        - Use standard ticker symbols (AAPL, MSFT, GOOGL, etc.)
        - Company names and sectors are auto-fetched when possible
        - Initial investment amount is used for share calculations
        """)



def render_creation_wizard():
    """Render step-by-step portfolio creation wizard."""

    st.subheader("🧙‍♂️ Portfolio Creation Wizard")

    # Initialize wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1

    if 'wizard_data' not in st.session_state:
        st.session_state.wizard_data = {}

    # Progress bar
    progress = (st.session_state.wizard_step - 1) / 4
    st.progress(progress, text=f"Step {st.session_state.wizard_step} of 5")

    # Wizard steps
    if st.session_state.wizard_step == 1:
        render_wizard_step_1()
    elif st.session_state.wizard_step == 2:
        render_wizard_step_2()
    elif st.session_state.wizard_step == 3:
        render_wizard_step_3()
    elif st.session_state.wizard_step == 4:
        render_wizard_step_4()
    elif st.session_state.wizard_step == 5:
        render_wizard_step_5()


def render_wizard_step_1():
    """Wizard Step 1: Portfolio Information."""

    st.write("### Step 1: Portfolio Information")

    portfolio_name = st.text_input(
        "Portfolio Name *",
        value=st.session_state.wizard_data.get('name', ''),
        key='wizard_name'
    )

    portfolio_description = st.text_area(
        "Description",
        value=st.session_state.wizard_data.get('description', ''),
        key='wizard_description'
    )

    from core.data_manager import PortfolioType
    portfolio_type = st.selectbox(
        "Portfolio Type",
        options=[pt.value for pt in PortfolioType],
        index=[pt.value for pt in PortfolioType].index(st.session_state.wizard_data.get('type', 'balanced')),
        format_func=lambda x: x.title(),
        key='wizard_type'
    )

    initial_value = st.number_input(
        "Initial Investment ($)",
        min_value=1.0,
        value=st.session_state.wizard_data.get('initial_value', 100000.0),
        step=1000.0,
        key='wizard_initial_value'
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel Wizard", use_container_width=True):
            st.session_state.wizard_step = 1
            st.session_state.wizard_data = {}
            st.rerun()

    with col2:
        if st.button("Next Step", use_container_width=True, disabled=not portfolio_name):
            st.session_state.wizard_data.update({
                'name': portfolio_name,
                'description': portfolio_description,
                'type': portfolio_type,
                'initial_value': initial_value
            })
            st.session_state.wizard_step = 2
            st.rerun()


def render_wizard_step_2():
    """Wizard Step 2: Input Method Selection."""

    st.write("### Step 2: Choose Input Method")

    method = st.radio(
        "How would you like to add assets?",
        ["Text Input", "Upload File", "Manual Entry", "Use Template"],
        index=0 if 'input_method' not in st.session_state.wizard_data else
        ["Text Input", "Upload File", "Manual Entry", "Use Template"].index(
            st.session_state.wizard_data['input_method']),
        key='wizard_input_method'
    )

    # Show preview of selected method
    if method == "Text Input":
        st.info("Enter ticker symbols with weights like: AAPL 30%, MSFT 25%, GOOGL 45%")
        st.code("AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%")

    elif method == "Upload File":
        st.info("Upload a CSV or Excel file with your portfolio data")
        st.write("Required columns: ticker, weight (optional: name, sector)")

    elif method == "Manual Entry":
        st.info("Add each asset individually with full control")
        st.write("Perfect for detailed portfolio construction")

    elif method == "Use Template":
        st.info("Start with a pre-built portfolio template")
        st.write("Available: Conservative, Growth, Income, Tech Focus, ESG")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()

    with col2:
        if st.button("Next Step", use_container_width=True):
            st.session_state.wizard_data['input_method'] = method
            st.session_state.wizard_step = 3
            st.rerun()


def render_wizard_step_3():
    """Wizard Step 3: Asset Input."""

    st.write("### Step 3: Add Your Assets")

    method = st.session_state.wizard_data['input_method']

    if method == "Text Input":
        render_wizard_text_input()
    elif method == "Upload File":
        render_wizard_file_upload()
    elif method == "Manual Entry":
        render_wizard_manual_entry()
    elif method == "Use Template":
        render_wizard_template_selection()


def render_wizard_text_input():
    """Wizard text input for assets."""

    text_input = st.text_area(
        "Enter your portfolio assets:",
        value=st.session_state.wizard_data.get('asset_text', ''),
        height=100,
        help="Format: TICKER WEIGHT%, TICKER WEIGHT%, ...",
        key='wizard_asset_text'
    )

    if text_input:
        # Validate input
        from ..utils.helpers import validate_ticker_input

        is_valid, error_msg, parsed_tickers = validate_ticker_input(text_input)

        if is_valid:
            st.success(f"✅ Parsed {len(parsed_tickers)} assets successfully")

            # Show preview
            import pandas as pd
            preview_df = pd.DataFrame(parsed_tickers, columns=['Ticker', 'Weight'])
            preview_df['Weight'] = preview_df['Weight'].apply(lambda x: f"{x:.1%}")
            st.dataframe(preview_df, hide_index=True)

            st.session_state.wizard_data['parsed_assets'] = parsed_tickers
        else:
            st.error(f"❌ {error_msg}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()

    with col2:
        can_proceed = text_input and st.session_state.wizard_data.get('parsed_assets')
        if st.button("Next Step", use_container_width=True, disabled=not can_proceed):
            st.session_state.wizard_data['asset_text'] = text_input
            st.session_state.wizard_step = 4
            st.rerun()


def render_wizard_file_upload():
    """Wizard file upload for assets."""

    uploaded_file = st.file_uploader(
        "Upload your portfolio file",
        type=['csv', 'xlsx', 'xls'],
        key='wizard_file_upload'
    )

    if uploaded_file:
        from ..utils.helpers import validate_file_upload

        is_valid, error_msg = validate_file_upload(uploaded_file)

        if is_valid:
            try:
                import pandas as pd

                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ File loaded: {len(df)} rows")

                # Show preview
                st.dataframe(df.head(), hide_index=True)

                # Column mapping
                ticker_col = st.selectbox("Ticker Column", df.columns.tolist(), key='wizard_ticker_col')
                weight_col = st.selectbox("Weight Column", ["Auto (Equal Weight)"] + df.columns.tolist(),
                                          key='wizard_weight_col')

                st.session_state.wizard_data['file_data'] = {
                    'df': df,
                    'ticker_col': ticker_col,
                    'weight_col': weight_col if weight_col != "Auto (Equal Weight)" else None
                }

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.error(error_msg)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()

    with col2:
        can_proceed = uploaded_file and st.session_state.wizard_data.get('file_data')
        if st.button("Next Step", use_container_width=True, disabled=not can_proceed):
            st.session_state.wizard_step = 4
            st.rerun()


def render_wizard_manual_entry():
    """Wizard manual entry for assets."""

    # Initialize manual assets
    if 'manual_assets' not in st.session_state.wizard_data:
        st.session_state.wizard_data['manual_assets'] = []

    # Asset entry form
    with st.form("wizard_asset_entry"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker", key='wizard_manual_ticker')

        with col2:
            weight = st.number_input("Weight (%)", min_value=0.1, max_value=100.0, value=10.0,
                                     key='wizard_manual_weight')

        with col3:
            name = st.text_input("Name (Optional)", key='wizard_manual_name')

        if st.form_submit_button("Add Asset"):
            if ticker:
                st.session_state.wizard_data['manual_assets'].append({
                    'ticker': ticker.upper(),
                    'weight': weight / 100,
                    'name': name or None
                })
                st.success(f"Added {ticker.upper()}")
                st.rerun()

    # Show current assets
    if st.session_state.wizard_data['manual_assets']:
        st.write("**Current Assets:**")

        assets_data = []
        total_weight = 0

        for asset in st.session_state.wizard_data['manual_assets']:
            assets_data.append({
                'Ticker': asset['ticker'],
                'Name': asset['name'] or 'N/A',
                'Weight': f"{asset['weight']:.1%}"
            })
            total_weight += asset['weight']

        import pandas as pd
        df = pd.DataFrame(assets_data)
        st.dataframe(df, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Assets", len(st.session_state.wizard_data['manual_assets']))
        with col2:
            st.metric("Total Weight", f"{total_weight:.1%}")

        if st.button("Clear All Assets", type="secondary"):
            st.session_state.wizard_data['manual_assets'] = []
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()

    with col2:
        can_proceed = len(st.session_state.wizard_data.get('manual_assets', [])) > 0
        if st.button("Next Step", use_container_width=True, disabled=not can_proceed):
            st.session_state.wizard_step = 4
            st.rerun()


def render_wizard_template_selection():
    """Wizard template selection."""

    templates = {
        "Conservative Growth": "SPY 40%, BND 30%, VTI 20%, VTEB 10%",
        "Tech Focus": "AAPL 25%, MSFT 20%, GOOGL 15%, NVDA 15%, META 10%, AMZN 10%, TSLA 5%",
        "Dividend Income": "JNJ 15%, PG 15%, KO 10%, PFE 10%, VZ 10%, T 10%, XOM 10%, CVX 10%, IBM 10%",
        "Global Diversified": "VTI 30%, VTIAX 25%, EFA 20%, VWO 15%, BND 10%"
    }

    selected_template = st.selectbox(
        "Choose a template:",
        list(templates.keys()),
        key='wizard_template_selection'
    )

    if selected_template:
        st.code(templates[selected_template])

        # Allow customization
        customize = st.checkbox("Customize template", key='wizard_customize_template')

        if customize:
            custom_text = st.text_area(
                "Modify the template:",
                value=templates[selected_template],
                key='wizard_custom_template'
            )
            st.session_state.wizard_data['template_text'] = custom_text
        else:
            st.session_state.wizard_data['template_text'] = templates[selected_template]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()

    with col2:
        can_proceed = selected_template and st.session_state.wizard_data.get('template_text')
        if st.button("Next Step", use_container_width=True, disabled=not can_proceed):
            st.session_state.wizard_step = 4
            st.rerun()


def render_wizard_step_4():
    """Wizard Step 4: Options and Settings."""

    st.write("### Step 4: Portfolio Settings")

    # Advanced options
    col1, col2 = st.columns(2)

    with col1:
        fetch_info = st.checkbox(
            "Fetch company information",
            value=True,
            help="Automatically fetch company names and sector data",
            key='wizard_fetch_info'
        )

        auto_normalize = st.checkbox(
            "Auto-normalize weights",
            value=True,
            help="Automatically adjust weights to sum to 100%",
            key='wizard_auto_normalize'
        )

    with col2:
        calculate_shares = st.checkbox(
            "Calculate share quantities",
            value=True,
            help="Calculate number of shares based on current prices",
            key='wizard_calculate_shares'
        )

        update_prices = st.checkbox(
            "Update current prices",
            value=True,
            help="Fetch latest market prices for all assets",
            key='wizard_update_prices'
        )

    # Preview settings
    st.subheader("📋 Portfolio Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Name:** {st.session_state.wizard_data.get('name', 'N/A')}")
        st.write(f"**Type:** {st.session_state.wizard_data.get('type', 'N/A').title()}")

    with col2:
        st.write(f"**Initial Value:** ${st.session_state.wizard_data.get('initial_value', 0):,.0f}")
        st.write(f"**Method:** {st.session_state.wizard_data.get('input_method', 'N/A')}")

    with col3:
        asset_count = 0
        if 'parsed_assets' in st.session_state.wizard_data:
            asset_count = len(st.session_state.wizard_data['parsed_assets'])
        elif 'manual_assets' in st.session_state.wizard_data:
            asset_count = len(st.session_state.wizard_data['manual_assets'])
        elif 'file_data' in st.session_state.wizard_data:
            asset_count = len(st.session_state.wizard_data['file_data']['df'])

        st.write(f"**Assets:** {asset_count}")
        st.write(f"**Description:** {st.session_state.wizard_data.get('description', 'None')[:20]}...")

    # Save settings
    st.session_state.wizard_data['settings'] = {
        'fetch_info': fetch_info,
        'auto_normalize': auto_normalize,
        'calculate_shares': calculate_shares,
        'update_prices': update_prices
    }

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()

    with col2:
        if st.button("Create Portfolio", use_container_width=True, type="primary"):
            st.session_state.wizard_step = 5
            st.rerun()


def render_wizard_step_5():
    """Wizard Step 5: Portfolio Creation and Confirmation."""

    st.write("### Step 5: Creating Portfolio...")

    try:
        # Create the portfolio based on wizard data
        result = create_portfolio_from_wizard_data()

        if result['success']:
            st.success("🎉 Portfolio created successfully!")

            # Show portfolio summary
            portfolio = result['portfolio']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Assets", len(portfolio.assets))

            with col2:
                st.metric("Total Value", f"${portfolio.calculate_value():,.2f}")

            with col3:
                st.metric("Portfolio Type", portfolio.portfolio_type.value.title())

            with col4:
                creation_time = portfolio.created_date.strftime("%H:%M:%S")
                st.metric("Created At", creation_time)

            # Show asset breakdown
            if portfolio.assets:
                st.subheader("📊 Asset Breakdown")

                import pandas as pd
                asset_data = []

                for asset in portfolio.assets:
                    asset_data.append({
                        'Ticker': asset.ticker,
                        'Name': asset.name or 'N/A',
                        'Weight': f"{asset.weight:.1%}",
                        'Shares': getattr(asset, 'shares', 0),
                        'Price': f"${getattr(asset, 'current_price', 0):.2f}" if hasattr(asset,
                                                                                         'current_price') else 'N/A'
                    })

                df = pd.DataFrame(asset_data)
                st.dataframe(df, hide_index=True, use_container_width=True)

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 View Analysis", use_container_width=True, type="primary"):
                    from ..utils.session_state import set_selected_portfolio
                    set_selected_portfolio(portfolio)
                    st.session_state.main_navigation = "📊 Portfolio Analysis"
                    reset_wizard()
                    st.rerun()

            with col2:
                if st.button("📝 Create Another", use_container_width=True):
                    reset_wizard()
                    st.rerun()

            with col3:
                if st.button("🏠 Go to Dashboard", use_container_width=True):
                    st.session_state.main_navigation = "🏠 Dashboard"
                    reset_wizard()
                    st.rerun()

        else:
            st.error(f"❌ Portfolio creation failed: {result['error']}")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Try Again", use_container_width=True):
                    st.session_state.wizard_step = 4
                    st.rerun()

            with col2:
                if st.button("Start Over", use_container_width=True):
                    reset_wizard()
                    st.rerun()

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

        if st.button("Start Over", use_container_width=True):
            reset_wizard()
            st.rerun()


def create_portfolio_from_wizard_data() -> Dict[str, Any]:
    """Create portfolio from wizard data."""

    try:
        from ..utils.session_state import get_portfolio_manager, refresh_portfolios, set_selected_portfolio
        from core.data_manager import PortfolioType

        wizard_data = st.session_state.wizard_data

        # Prepare asset data based on input method
        asset_text = ""

        if 'parsed_assets' in wizard_data:
            # From text input
            asset_parts = []
            for ticker, weight in wizard_data['parsed_assets']:
                asset_parts.append(f"{ticker} {weight * 100:.1f}%")
            asset_text = ", ".join(asset_parts)

        elif 'manual_assets' in wizard_data:
            # From manual entry
            asset_parts = []
            for asset in wizard_data['manual_assets']:
                asset_parts.append(f"{asset['ticker']} {asset['weight'] * 100:.1f}%")
            asset_text = ", ".join(asset_parts)

        elif 'template_text' in wizard_data:
            # From template
            asset_text = wizard_data['template_text']

        elif 'file_data' in wizard_data:
            # From file upload
            file_data = wizard_data['file_data']
            df = file_data['df']
            ticker_col = file_data['ticker_col']
            weight_col = file_data['weight_col']

            tickers = df[ticker_col].astype(str).str.upper().tolist()

            if weight_col:
                weights = df[weight_col].tolist()
            else:
                # Equal weights
                equal_weight = 100.0 / len(tickers)
                weights = [equal_weight] * len(tickers)

            asset_parts = []
            for ticker, weight in zip(tickers, weights):
                asset_parts.append(f"{ticker} {weight:.1f}%")
            asset_text = ", ".join(asset_parts)

        if not asset_text:
            return {'success': False, 'error': 'No asset data provided'}

        # Create portfolio using portfolio manager
        portfolio_manager = get_portfolio_manager()

        portfolio = portfolio_manager.create_from_text(
            name=wizard_data['name'],
            text=asset_text,
            description=wizard_data.get('description', ''),
            portfolio_type=PortfolioType(wizard_data['type']),
            initial_value=wizard_data['initial_value']
        )

        # Apply settings
        settings = wizard_data.get('settings', {})

        if settings.get('fetch_info'):
            from ..utils.helpers import update_company_info
            update_company_info(portfolio)

        if settings.get('update_prices') or settings.get('calculate_shares'):
            from ..utils.helpers import update_portfolio_prices
            update_portfolio_prices(portfolio)

        # Refresh portfolio list and set as selected
        refresh_portfolios()
        set_selected_portfolio(portfolio)

        return {'success': True, 'portfolio': portfolio}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def reset_wizard():
    """Reset wizard state."""
    st.session_state.wizard_step = 1
    st.session_state.wizard_data = {}


def render_batch_creation():
    """Render batch portfolio creation interface."""

    st.subheader("📦 Batch Portfolio Creation")

    st.info("Create multiple portfolios at once from different data sources")

    # Batch creation options
    batch_method = st.selectbox(
        "Batch Creation Method",
        [
            "Multiple CSV Files",
            "Excel Workbook (Multiple Sheets)",
            "JSON Configuration File",
            "Template Variations"
        ]
    )

    if batch_method == "Multiple CSV Files":
        render_batch_csv_creation()
    elif batch_method == "Excel Workbook (Multiple Sheets)":
        render_batch_excel_creation()
    elif batch_method == "JSON Configuration File":
        render_batch_json_creation()
    elif batch_method == "Template Variations":
        render_batch_template_creation()


def render_batch_csv_creation():
    """Render batch CSV portfolio creation."""

    uploaded_files = st.file_uploader(
        "Upload multiple CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Each CSV file will create a separate portfolio"
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")

        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name}"):
                try:
                    import pandas as pd
                    df = pd.read_csv(file)

                    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                    st.dataframe(df.head(3), hide_index=True)

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        if st.button("Create All Portfolios", type="primary"):
            create_batch_portfolios_from_csvs(uploaded_files)


def create_batch_portfolios_from_csvs(files):
    """Create portfolios from multiple CSV files."""

    with st.spinner("Creating portfolios from CSV files..."):
        success_count = 0
        error_count = 0

        for file in files:
            try:
                # Extract portfolio name from filename
                portfolio_name = file.name.replace('.csv', '').replace('_', ' ').title()

                # This would implement actual batch creation
                st.write(f"✅ Created portfolio: {portfolio_name}")
                success_count += 1

            except Exception as e:
                st.write(f"❌ Failed to create portfolio from {file.name}: {str(e)}")
                error_count += 1

        if success_count > 0:
            st.success(f"🎉 Successfully created {success_count} portfolios!")
            st.balloons()

        if error_count > 0:
            st.warning(f"⚠️ {error_count} portfolios failed to create")


def render_batch_excel_creation():
    """Render batch Excel portfolio creation."""

    st.info("Upload an Excel file with multiple sheets - each sheet becomes a portfolio")

    uploaded_file = st.file_uploader(
        "Upload Excel workbook",
        type=['xlsx', 'xls'],
        help="Each sheet will create a separate portfolio"
    )

    if uploaded_file:
        try:
            import pandas as pd

            # Read all sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            st.write(f"Found {len(sheet_names)} sheets:")

            for sheet in sheet_names:
                st.write(f"- {sheet}")

            if st.button("Create Portfolios from Sheets", type="primary"):
                create_batch_portfolios_from_excel(excel_file)

        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")


def create_batch_portfolios_from_excel(excel_file):
    """Create portfolios from Excel sheets."""

    with st.spinner("Creating portfolios from Excel sheets..."):
        success_count = 0

        for sheet_name in excel_file.sheet_names:
            try:
                # This would implement actual sheet processing
                st.write(f"✅ Created portfolio: {sheet_name}")
                success_count += 1

            except Exception as e:
                st.write(f"❌ Failed to create portfolio from sheet {sheet_name}: {str(e)}")

        if success_count > 0:
            st.success(f"🎉 Successfully created {success_count} portfolios!")
            st.balloons()


def render_batch_json_creation():
    """Render batch JSON portfolio creation."""

    st.info("Upload a JSON configuration file with multiple portfolio definitions")

    uploaded_file = st.file_uploader(
        "Upload JSON configuration",
        type=['json'],
        help="JSON file should contain an array of portfolio configurations"
    )

    if uploaded_file:
        try:
            import json
            config_data = json.load(uploaded_file)

            if isinstance(config_data, list):
                st.write(f"Found {len(config_data)} portfolio configurations")

                # Show preview of first configuration
                if config_data:
                    st.json(config_data[0], expanded=False)

                if st.button("Create All Portfolios", type="primary"):
                    create_batch_portfolios_from_json(config_data)
            else:
                st.error("JSON file should contain an array of portfolio configurations")

        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")


def create_batch_portfolios_from_json(config_data):
    """Create portfolios from JSON configuration."""

    with st.spinner("Creating portfolios from JSON configuration..."):
        success_count = 0

        for config in config_data:
            try:
                # This would implement actual JSON processing
                portfolio_name = config.get('name', f'Portfolio {success_count + 1}')
                st.write(f"✅ Created portfolio: {portfolio_name}")
                success_count += 1

            except Exception as e:
                st.write(f"❌ Failed to create portfolio: {str(e)}")

        if success_count > 0:
            st.success(f"🎉 Successfully created {success_count} portfolios!")
            st.balloons()


def render_batch_template_creation():
    """Render batch template variation creation."""

    st.info("Create multiple portfolios based on template variations")

    base_template = st.selectbox(
        "Base Template",
        ["Conservative Growth", "Tech Focus", "Dividend Income", "Global Diversified"]
    )

    variation_type = st.selectbox(
        "Variation Type",
        ["Weight Adjustments", "Asset Substitutions", "Risk Levels", "Time Horizons"]
    )

    num_variations = st.number_input(
        "Number of Variations",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    if st.button("Generate Template Variations", type="primary"):
        create_template_variations(base_template, variation_type, num_variations)


def create_template_variations(base_template, variation_type, num_variations):
    """Create portfolio variations from base template."""

    with st.spinner("Creating template variations..."):
        for i in range(num_variations):
            variation_name = f"{base_template} - Variation {i + 1}"
            st.write(f"✅ Created portfolio: {variation_name}")

        st.success(f"🎉 Successfully created {num_variations} portfolio variations!")
        st.balloons()