"""
Manage Portfolios page for the Portfolio Management System.

Clean, structured implementation for portfolio management operations.
Fixed version addressing all errors.
"""
import streamlit as st
import pandas as pd
import time
from typing import List, Optional
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import Portfolio
from ..utils.session_state import (
    get_portfolios, refresh_portfolios, get_selected_portfolio,
    set_selected_portfolio, get_portfolio_manager
)
from ..utils.helpers import update_portfolio_prices, export_portfolio_data
from ..utils.formatting import format_currency, format_datetime, format_percentage
from ..components.charts import create_portfolio_allocation_chart, create_portfolio_summary_cards


def render_manage_portfolios():
    """Main entry point for portfolio management page."""

    # Load portfolios
    portfolios = get_portfolios()
    if not portfolios:
        refresh_portfolios()
        portfolios = get_portfolios()

    if not portfolios:
        render_empty_state()
        return

    # Page header
    st.header("📋 Manage Portfolios")
    st.caption(f"Manage and analyze your {len(portfolios)} portfolio(s)")

    # Main layout
    render_portfolio_list(portfolios)

    # Selected portfolio details
    selected_portfolio = get_selected_portfolio()
    if selected_portfolio:
        render_portfolio_details(selected_portfolio)


def render_empty_state():
    """Render empty state when no portfolios exist."""

    st.info("No portfolios available to manage.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 📝 Create Your First Portfolio
        
        Start by creating a portfolio to access management features:
        
        - Edit portfolio details and holdings
        - Update asset weights and allocations  
        - Add or remove assets
        - Export portfolio data
        - Monitor performance metrics
        """)

        # Используем callback для навигации вместо изменения session_state
        if st.button("📝 Create Portfolio", width="stretch", type="primary", key="create_from_empty"):
            # Будем использовать query params для навигации
            st.switch_page("pages/2_📝_Create_Portfolio.py")


def render_portfolio_list(portfolios: List[Portfolio]):
    """Render compact portfolio list with action buttons."""

    st.subheader("📊 Portfolio Overview")

    # Bulk operations
    with st.expander("⚡ Bulk Operations"):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Update All Prices", width="stretch", key="bulk_update_prices"):
                update_all_portfolio_prices(portfolios)

        with col2:
            if st.button("📤 Export All Data", width="stretch", key="bulk_export_all"):
                export_all_portfolios(portfolios)

        with col3:
            if st.button("📊 Compare Portfolios", width="stretch", key="bulk_compare"):
                # Используем switch_page вместо изменения session_state
                st.switch_page("pages/4_📊_Portfolio_Analysis.py")

    # Portfolio grid
    for i, portfolio in enumerate(portfolios):
        render_portfolio_card(portfolio, i)


def render_portfolio_card(portfolio: Portfolio, index: int):
    """Render individual portfolio card with actions."""

    with st.container():
        # Используем st.container border для визуального эффекта
        with st.container(border=True):
            # Portfolio info row
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])

            with col1:
                st.markdown(f"**{portfolio.name}**")
                st.caption(f"{portfolio.portfolio_type.value.title()} • {len(portfolio.assets)} assets")
                if portfolio.description:
                    st.caption(f"Description: {portfolio.description}")

            with col2:
                total_value = portfolio.calculate_value()
                st.metric("Value", format_currency(total_value))
                st.caption(f"Modified: {format_datetime(portfolio.last_modified, '%m/%d/%Y')}")

            with col3:
                if st.button("👁️ View", key=f"view_portfolio_{index}_{portfolio.id}", width="stretch"):
                    set_selected_portfolio(portfolio)
                    st.rerun()

            with col4:
                if st.button("✏️ Edit", key=f"edit_portfolio_{index}_{portfolio.id}", width="stretch"):
                    set_selected_portfolio(portfolio)
                    st.session_state.edit_mode = True
                    st.rerun()

            with col5:
                if st.button("🗑️ Delete", key=f"delete_portfolio_{index}_{portfolio.id}",
                            width="stretch", type="secondary"):
                    st.session_state[f"confirm_delete_{portfolio.id}"] = True
                    st.rerun()

            # Delete confirmation
            if st.session_state.get(f"confirm_delete_{portfolio.id}", False):
                render_delete_confirmation(portfolio, index)


def render_delete_confirmation(portfolio: Portfolio, index: int):
    """Render delete confirmation dialog."""

    st.warning(f"⚠️ Delete portfolio '{portfolio.name}'?")
    st.write("This action cannot be undone.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete", key=f"confirm_delete_{index}_{portfolio.id}", type="primary"):
            delete_portfolio(portfolio)

    with col2:
        if st.button("✅ Cancel", key=f"cancel_delete_{index}_{portfolio.id}"):
            del st.session_state[f"confirm_delete_{portfolio.id}"]
            st.rerun()


def delete_portfolio(portfolio: Portfolio):
    """Delete a portfolio with proper cleanup."""

    try:
        portfolio_manager = get_portfolio_manager()

        # ИСПРАВЛЕНО: используем правильный метод delete_portfolio
        portfolio_manager.delete_portfolio(portfolio.id)

        # Clear selection if this was selected
        if get_selected_portfolio() and get_selected_portfolio().id == portfolio.id:
            set_selected_portfolio(None)

        # Clear confirmation state
        if f"confirm_delete_{portfolio.id}" in st.session_state:
            del st.session_state[f"confirm_delete_{portfolio.id}"]

        # Clear edit mode if active
        if "edit_mode" in st.session_state:
            del st.session_state.edit_mode

        refresh_portfolios()
        st.success(f"✅ Portfolio '{portfolio.name}' deleted successfully!")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"Error deleting portfolio: {e}")


def render_portfolio_details(portfolio: Portfolio):
    """Render detailed view of selected portfolio."""

    st.divider()

    # Header with back button
    col1, col2 = st.columns([1, 10])

    with col1:
        if st.button("← Back", key="back_to_portfolio_list"):
            set_selected_portfolio(None)
            if "edit_mode" in st.session_state:
                del st.session_state.edit_mode
            st.rerun()

    with col2:
        mode = "Edit" if st.session_state.get("edit_mode", False) else "View"
        st.subheader(f"{mode}: {portfolio.name}")

    # Mode toggle
    if not st.session_state.get("edit_mode", False):
        if st.button("✏️ Switch to Edit Mode", type="primary", key="toggle_to_edit"):
            st.session_state.edit_mode = True
            st.rerun()
    else:
        if st.button("👁️ Switch to View Mode", key="toggle_to_view"):
            st.session_state.edit_mode = False
            st.rerun()

    # Render appropriate mode
    if st.session_state.get("edit_mode", False):
        render_edit_mode(portfolio)
    else:
        render_view_mode(portfolio)


def render_view_mode(portfolio: Portfolio):
    """Render portfolio in view mode with analytics."""

    # Summary metrics
    st.write("### 📊 Portfolio Summary")

    total_value = portfolio.calculate_value()
    total_assets = len(portfolio.assets)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Value", format_currency(total_value))

    with col2:
        st.metric("Number of Assets", total_assets)

    with col3:
        if portfolio.assets:
            avg_weight = 1.0 / len(portfolio.assets)
            st.metric("Average Weight", format_percentage(avg_weight))
        else:
            st.metric("Average Weight", "0%")

    with col4:
        if portfolio.assets:
            largest_weight = max(asset.weight for asset in portfolio.assets)
            st.metric("Largest Position", format_percentage(largest_weight))
        else:
            st.metric("Largest Position", "0%")

    # Asset allocation chart
    if portfolio.assets:
        st.write("### 📈 Asset Allocation")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_portfolio_allocation_chart(portfolio, "pie")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Top Holdings**")
            sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
            for i, asset in enumerate(sorted_assets[:5]):
                st.write(f"{i+1}. **{asset.ticker}** - {format_percentage(asset.weight)}")

    # Holdings table
    render_holdings_table(portfolio)

    # Action buttons
    st.write("### ⚡ Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Update Prices", width="stretch", key="view_update_prices"):
            update_portfolio_prices(portfolio)

    with col2:
        if st.button("📤 Export Data", width="stretch", key="view_export_data"):
            export_portfolio_data(portfolio)

    with col3:
        if st.button("📊 Analyze", width="stretch", key="view_analyze"):
            # ИСПРАВЛЕНО: используем switch_page вместо изменения session_state
            st.switch_page("pages/4_📊_Portfolio_Analysis.py")


def render_edit_mode(portfolio: Portfolio):
    """Render portfolio in edit mode."""

    # Edit tabs
    tab1, tab2, tab3 = st.tabs(["📝 Basic Info", "📊 Assets", "⚙️ Settings"])

    with tab1:
        render_edit_basic_info(portfolio)

    with tab2:
        render_edit_assets(portfolio)

    with tab3:
        render_edit_settings(portfolio)


def render_edit_basic_info(portfolio: Portfolio):
    """Render basic info editing."""

    st.write("### ✏️ Edit Portfolio Information")

    with st.form(f"edit_basic_{portfolio.id}"):

        col1, col2 = st.columns(2)

        with col1:
            new_name = st.text_input("Portfolio Name", value=portfolio.name)

            from core.data_manager import PortfolioType
            type_options = [pt.value for pt in PortfolioType]
            current_index = type_options.index(portfolio.portfolio_type.value)
            new_type = st.selectbox("Portfolio Type", type_options, index=current_index)

        with col2:
            new_description = st.text_area(
                "Description",
                value=portfolio.description or "",
                height=100
            )

        if st.form_submit_button("💾 Save Changes", width="stretch"):
            update_basic_info(portfolio, new_name, new_description, new_type)


def update_basic_info(portfolio: Portfolio, name: str, description: str, portfolio_type: str):
    """Update portfolio basic information."""

    try:
        from core.data_manager import PortfolioType
        from datetime import datetime

        # ИСПРАВЛЕНО: используем правильный метод update_portfolio
        portfolio_manager = get_portfolio_manager()

        updates = {
            'name': name,
            'description': description,
            'portfolio_type': PortfolioType(portfolio_type),
            'last_modified': datetime.now()
        }

        # Используем update_portfolio вместо save_portfolio
        portfolio_manager.update_portfolio(portfolio.id, updates)

        refresh_portfolios()
        st.success("✅ Portfolio information updated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error updating portfolio: {e}")


def render_edit_assets(portfolio: Portfolio):
    """Render assets editing interface."""

    st.write("### 📊 Manage Assets")

    if not portfolio.assets:
        st.info("No assets in this portfolio")
        return

    # Create editable dataframe
    assets_data = []
    for asset in portfolio.assets:
        assets_data.append({
            'Ticker': asset.ticker,
            'Name': asset.name or '',
            'Weight (%)': round(asset.weight * 100, 2),
            'Shares': getattr(asset, 'shares', 0),
            'Current Price': getattr(asset, 'current_price', 0.0)
        })

    df = pd.DataFrame(assets_data)

    # Display editable table
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        key=f"edit_assets_{portfolio.id}",
        column_config={
            "Weight (%)": st.column_config.NumberColumn(
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.2f"
            ),
            "Current Price": st.column_config.NumberColumn(
                "Current Price",
                min_value=0.0,
                step=0.01,
                format="$%.2f"
            ),
            "Shares": st.column_config.NumberColumn(
                "Shares",
                min_value=0,
                step=1
            )
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Asset Changes", width="stretch", key=f"save_assets_{portfolio.id}"):
            update_assets_from_dataframe(portfolio, edited_df)

    with col2:
        if st.button("⚖️ Normalize Weights", width="stretch", key=f"normalize_weights_{portfolio.id}"):
            normalize_portfolio_weights(portfolio, edited_df)


def update_assets_from_dataframe(portfolio: Portfolio, df: pd.DataFrame):
    """Update portfolio assets from edited DataFrame."""

    try:
        # Validate total weight
        total_weight = df['Weight (%)'].sum()
        if abs(total_weight - 100) > 0.01:
            st.warning(f"Total weight is {total_weight:.2f}%. Consider normalizing to 100%.")

        # Update assets
        for i, asset in enumerate(portfolio.assets):
            if i < len(df):
                asset.ticker = df.iloc[i]['Ticker']
                asset.name = df.iloc[i]['Name'] if df.iloc[i]['Name'] else None
                asset.weight = df.iloc[i]['Weight (%)'] / 100

                if 'Shares' in df.columns and df.iloc[i]['Shares']:
                    asset.shares = int(df.iloc[i]['Shares'])

                if 'Current Price' in df.columns and df.iloc[i]['Current Price']:
                    asset.current_price = float(df.iloc[i]['Current Price'])

        # ИСПРАВЛЕНО: используем правильный метод
        from datetime import datetime
        portfolio_manager = get_portfolio_manager()

        updates = {
            'assets': portfolio.assets,
            'last_modified': datetime.now()
        }

        portfolio_manager.update_portfolio(portfolio.id, updates)

        refresh_portfolios()
        st.success("✅ Assets updated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error updating assets: {e}")


def normalize_portfolio_weights(portfolio: Portfolio, df: pd.DataFrame):
    """Normalize portfolio weights to sum to 100%."""

    try:
        total_weight = df['Weight (%)'].sum()
        if total_weight > 0:
            # Normalize weights
            for i, asset in enumerate(portfolio.assets):
                if i < len(df):
                    normalized_weight = (df.iloc[i]['Weight (%)'] / total_weight)
                    asset.weight = normalized_weight

            # ИСПРАВЛЕНО: используем правильный метод
            from datetime import datetime
            portfolio_manager = get_portfolio_manager()

            updates = {
                'assets': portfolio.assets,
                'last_modified': datetime.now()
            }

            portfolio_manager.update_portfolio(portfolio.id, updates)

            refresh_portfolios()
            st.success("✅ Weights normalized to 100%!")
            st.rerun()

    except Exception as e:
        st.error(f"Error normalizing weights: {e}")


def render_edit_settings(portfolio: Portfolio):
    """Render portfolio settings editing."""

    st.write("### ⚙️ Portfolio Settings")
    st.info("Advanced settings will be available in Phase 2")

    # Basic settings for now
    with st.form(f"settings_{portfolio.id}"):

        enable_auto_rebalance = st.checkbox(
            "Enable Auto-Rebalancing",
            value=False,
            help="Automatically rebalance when weights drift"
        )

        rebalance_threshold = st.slider(
            "Rebalance Threshold (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Trigger rebalancing when weights drift by this amount"
        )

        if st.form_submit_button("💾 Save Settings"):
            st.success("✅ Settings saved! (Feature coming in Phase 2)")


def render_holdings_table(portfolio: Portfolio):
    """Render detailed holdings table with enhanced information including cash."""

    if not portfolio.assets:
        st.info("No assets in this portfolio")
        return

    st.write("### 📋 Top Holdings")

    # Create enhanced holdings dataframe
    holdings_data = []
    total_value = portfolio.calculate_value()
    total_allocated_weight = sum(asset.weight for asset in portfolio.assets)
    cash_percentage = 1.0 - total_allocated_weight

    # Add regular assets
    for asset in portfolio.assets:
        current_price = getattr(asset, 'current_price', None)
        shares = getattr(asset, 'shares', None)
        sector = getattr(asset, 'sector', 'N/A')
        purchase_price = getattr(asset, 'purchase_price', None)

        # Calculate market value
        if current_price and shares:
            market_value = current_price * shares
        else:
            market_value = asset.weight * total_value

        # Calculate P&L if purchase price available
        pnl = None
        pnl_percent = None
        if current_price and purchase_price and shares:
            pnl = (current_price - purchase_price) * shares
            pnl_percent = (current_price - purchase_price) / purchase_price * 100

        holdings_data.append({
            'Ticker': asset.ticker,
            'Company Name': asset.name or 'N/A',
            'Sector': sector,
            'Weight': f"{asset.weight * 100:.1f}%",
            'Price': f"${current_price:.2f}" if current_price else 'N/A',
            'Shares': f"{shares:,.0f}" if shares else 'N/A',
            'Total Value': format_currency(market_value),
            'Purchase Price': f"${purchase_price:.2f}" if purchase_price else 'N/A',
            'P&L': format_currency(pnl) if pnl is not None else 'N/A',
            'P&L %': f"{pnl_percent:.1f}%" if pnl_percent is not None else 'N/A'
        })

    # Add cash row if significant
    if cash_percentage > 0.001:  # Show if > 0.1%
        cash_value = cash_percentage * total_value
        holdings_data.append({
            'Ticker': '💰 CASH',
            'Company Name': 'Cash & Cash Equivalents',
            'Sector': 'Cash',
            'Weight': f"{cash_percentage * 100:.1f}%",
            'Price': '$1.00',
            'Shares': f"{cash_value:,.0f}",
            'Total Value': format_currency(cash_value),
            'Purchase Price': '$1.00',
            'P&L': '$0.00',
            'P&L %': '0.0%'
        })

    # Sort by weight (descending)
    df = pd.DataFrame(holdings_data)
    df = df.sort_values('Weight', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)

    # Enhanced dataframe display with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Company Name": st.column_config.TextColumn("Company Name", width="medium"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Weight": st.column_config.TextColumn("Weight", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Shares": st.column_config.TextColumn("Shares", width="small"),
            "Total Value": st.column_config.TextColumn("Total Value", width="medium"),
            "Purchase Price": st.column_config.TextColumn("Purchase Price", width="small"),
            "P&L": st.column_config.TextColumn("P&L", width="small"),
            "P&L %": st.column_config.TextColumn("P&L %", width="small")
        }
    )


def update_all_portfolio_prices(portfolios: List[Portfolio]):
    """Update prices for all portfolios."""

    with st.spinner("Updating prices for all portfolios..."):
        try:
            from ..utils.session_state import get_price_manager

            price_manager = get_price_manager()
            updated_count = 0

            # Collect unique tickers
            all_tickers = set()
            for portfolio in portfolios:
                for asset in portfolio.assets:
                    all_tickers.add(asset.ticker)

            if all_tickers:
                # ИСПРАВЛЕНО: убираем show_errors параметр
                prices = price_manager.get_current_prices(list(all_tickers))

                # Update portfolios
                for portfolio in portfolios:
                    for asset in portfolio.assets:
                        if asset.ticker in prices and prices[asset.ticker]:
                            asset.current_price = prices[asset.ticker]
                            updated_count += 1

                if updated_count > 0:
                    st.success(f"✅ Updated {updated_count} asset prices across all portfolios")
                else:
                    st.info("ℹ️ All prices are already up to date")
            else:
                st.warning("⚠️ No assets found to update")

        except Exception as e:
            st.error(f"Error updating prices: {e}")


def export_all_portfolios(portfolios: List[Portfolio]):
    """Export all portfolios."""

    with st.spinner("Exporting all portfolios..."):
        try:
            for portfolio in portfolios:
                export_portfolio_data(portfolio)

            st.success(f"✅ Exported {len(portfolios)} portfolios successfully!")

        except Exception as e:
            st.error(f"Error exporting portfolios: {e}")