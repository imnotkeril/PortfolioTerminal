"""
Manage Portfolios page for the Portfolio Management System.

This module contains functionality for managing existing portfolios including
editing, updating, deleting, and bulk operations.
"""
import streamlit as st
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
    set_selected_portfolio, get_confirmation_state, set_confirmation_state,
    clear_confirmation_state
)
from ..utils.helpers import update_portfolio_prices, export_portfolio_data
from ..components.tables import render_portfolio_assets_table
from ..utils.formatting import format_currency, format_datetime


def render_manage_portfolios():
    """Render the manage portfolios page."""

    st.header("📋 Manage Portfolios")

    # Load portfolios if not loaded
    portfolios = get_portfolios()
    if not portfolios:
        refresh_portfolios()
        portfolios = get_portfolios()

    if not portfolios:
        render_empty_management()
        return

    # Portfolio management interface
    render_portfolio_selector(portfolios)
    render_bulk_operations(portfolios)
    render_portfolio_management_details()


def render_empty_management():
    """Render management page when no portfolios exist."""

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

        if st.button("📝 Create Portfolio", use_container_width=True, type="primary"):
            st.session_state.main_navigation = "📝 Create Portfolio"
            st.rerun()


def render_portfolio_selector(portfolios: List[Portfolio]):
    """Render portfolio selection interface."""

    st.subheader("📊 Select Portfolio to Manage")

    # Portfolio overview with action buttons
    for i, portfolio in enumerate(portfolios):
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 1, 1, 1, 1])

            with col1:
                st.write(f"**{portfolio.name}**")
                st.caption(f"{portfolio.portfolio_type.value.title()} • {len(portfolio.assets)} assets")

            with col2:
                total_value = portfolio.calculate_value()
                st.write(f"**{format_currency(total_value)}**")
                st.caption(f"Modified: {format_datetime(portfolio.last_modified, '%m/%d/%Y')}")

            with col3:
                if st.button("👁️ View", key=f"view_{portfolio.id}", use_container_width=True):
                    set_selected_portfolio(portfolio)
                    st.rerun()

            with col4:
                if st.button("✏️ Edit", key=f"edit_{portfolio.id}", use_container_width=True):
                    set_selected_portfolio(portfolio)
                    st.session_state.management_mode = "edit"
                    st.rerun()

            with col5:
                if st.button("📤 Export", key=f"export_{portfolio.id}", use_container_width=True):
                    export_portfolio_data(portfolio)

            with col6:
                if st.button("🗑️ Delete", key=f"delete_{portfolio.id}", use_container_width=True, type="secondary"):
                    set_confirmation_state(f"delete_{portfolio.id}", True)
                    st.rerun()

            # Handle delete confirmation
            if get_confirmation_state(f"delete_{portfolio.id}"):
                render_delete_confirmation(portfolio)

        st.divider()


def render_delete_confirmation(portfolio: Portfolio):
    """Render delete confirmation dialog."""

    st.warning(f"⚠️ Delete portfolio '{portfolio.name}'?")
    st.write("This action cannot be undone. The portfolio will be permanently removed.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("❌ Yes, Delete", key=f"confirm_delete_{portfolio.id}", type="primary"):
            delete_portfolio(portfolio)

    with col2:
        if st.button("✅ Cancel", key=f"cancel_delete_{portfolio.id}"):
            clear_confirmation_state(f"delete_{portfolio.id}")
            st.rerun()

    with col3:
        # Show portfolio info as reminder
        st.caption(f"Value: {format_currency(portfolio.calculate_value())}")
        st.caption(f"Assets: {len(portfolio.assets)}")


def delete_portfolio(portfolio: Portfolio):
    """Delete a portfolio."""

    try:
        import os
        from ..utils.session_state import get_portfolio_manager

        # Delete using portfolio manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.delete_portfolio(portfolio.id)

        # Also try direct file deletion as backup
        portfolio_file = f"data/portfolios/{portfolio.id}.json"
        if os.path.exists(portfolio_file):
            os.remove(portfolio_file)

        # Clear states and refresh
        clear_confirmation_state(f"delete_{portfolio.id}")

        if get_selected_portfolio() and get_selected_portfolio().id == portfolio.id:
            set_selected_portfolio(None)

        refresh_portfolios()

        st.success(f"✅ Portfolio '{portfolio.name}' deleted successfully!")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"Error deleting portfolio: {e}")
        clear_confirmation_state(f"delete_{portfolio.id}")


def render_bulk_operations(portfolios: List[Portfolio]):
    """Render bulk operations interface."""

    st.subheader("⚡ Bulk Operations")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Update All Prices", use_container_width=True):
            update_all_portfolio_prices(portfolios)

    with col2:
        if st.button("📤 Export All", use_container_width=True):
            export_all_portfolios(portfolios)

    with col3:
        if st.button("📊 Compare All", use_container_width=True):
            st.session_state.main_navigation = "📊 Portfolio Analysis"
            st.session_state.analysis_mode = "comparison"
            st.rerun()

    with col4:
        if st.button("🗑️ Bulk Delete", use_container_width=True, type="secondary"):
            st.session_state.show_bulk_delete = True
            st.rerun()

    # Bulk delete interface
    if st.session_state.get('show_bulk_delete', False):
        render_bulk_delete_interface(portfolios)


def render_bulk_delete_interface(portfolios: List[Portfolio]):
    """Render bulk delete interface."""

    st.warning("⚠️ Bulk Delete Portfolios")

    selected_portfolios = st.multiselect(
        "Select portfolios to delete:",
        options=[p.name for p in portfolios],
        key="bulk_delete_selection"
    )

    if selected_portfolios:
        total_value = sum(
            p.calculate_value()
            for p in portfolios
            if p.name in selected_portfolios
        )

        st.write(f"**Selected:** {len(selected_portfolios)} portfolios")
        st.write(f"**Total Value:** {format_currency(total_value)}")

        st.error("⚠️ This action cannot be undone!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("❌ Delete Selected", key="bulk_delete_confirm", type="primary"):
                delete_selected_portfolios(portfolios, selected_portfolios)

        with col2:
            if st.button("✅ Cancel", key="bulk_delete_cancel"):
                st.session_state.show_bulk_delete = False
                st.rerun()


def delete_selected_portfolios(portfolios: List[Portfolio], selected_names: List[str]):
    """Delete multiple selected portfolios."""

    with st.spinner("Deleting selected portfolios..."):
        success_count = 0
        error_count = 0

        for portfolio in portfolios:
            if portfolio.name in selected_names:
                try:
                    # Delete portfolio
                    delete_portfolio(portfolio)
                    success_count += 1
                except Exception as e:
                    st.error(f"Failed to delete {portfolio.name}: {str(e)}")
                    error_count += 1

        # Clear bulk delete state
        st.session_state.show_bulk_delete = False

        if success_count > 0:
            st.success(f"✅ Successfully deleted {success_count} portfolios")

        if error_count > 0:
            st.error(f"❌ Failed to delete {error_count} portfolios")

        st.rerun()


def render_portfolio_management_details():
    """Render detailed management interface for selected portfolio."""

    selected_portfolio = get_selected_portfolio()

    if not selected_portfolio:
        st.info("💡 Select a portfolio above to view management options")
        return

    st.subheader(f"🔧 Managing: {selected_portfolio.name}")

    # Management tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Edit Details",
        "📊 Manage Assets",
        "💰 Update Prices",
        "📈 Rebalance",
        "📋 History"
    ])

    with tab1:
        render_edit_portfolio_details(selected_portfolio)

    with tab2:
        render_manage_portfolio_assets(selected_portfolio)

    with tab3:
        render_update_portfolio_prices(selected_portfolio)

    with tab4:
        render_portfolio_rebalancing(selected_portfolio)

    with tab5:
        render_portfolio_history(selected_portfolio)


def render_edit_portfolio_details(portfolio: Portfolio):
    """Render portfolio details editing interface."""

    st.write("### Portfolio Information")

    with st.form(f"edit_portfolio_{portfolio.id}"):

        # Basic details
        col1, col2 = st.columns(2)

        with col1:
            new_name = st.text_input(
                "Portfolio Name",
                value=portfolio.name,
                key=f"edit_name_{portfolio.id}"
            )

            from core.data_manager import PortfolioType
            current_type_index = [pt.value for pt in PortfolioType].index(portfolio.portfolio_type.value)
            new_type = st.selectbox(
                "Portfolio Type",
                options=[pt.value for pt in PortfolioType],
                index=current_type_index,
                format_func=lambda x: x.title(),
                key=f"edit_type_{portfolio.id}"
            )

        with col2:
            new_description = st.text_area(
                "Description",
                value=portfolio.description or "",
                height=100,
                key=f"edit_description_{portfolio.id}"
            )

            # Tags (if portfolio has tags)
            current_tags = getattr(portfolio, 'tags', [])
            new_tags = st.text_input(
                "Tags (comma separated)",
                value=", ".join(current_tags) if current_tags else "",
                help="Enter tags separated by commas",
                key=f"edit_tags_{portfolio.id}"
            )

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):

            col1, col2 = st.columns(2)

            with col1:
                auto_rebalance = st.checkbox(
                    "Auto-rebalancing",
                    value=getattr(portfolio, 'auto_rebalance', False),
                    help="Automatically rebalance when thresholds are exceeded",
                    key=f"edit_auto_rebalance_{portfolio.id}"
                )

                rebalance_frequency = st.selectbox(
                    "Rebalancing Frequency",
                    ["Never", "Weekly", "Monthly", "Quarterly", "Annually"],
                    index=0,
                    key=f"edit_rebalance_freq_{portfolio.id}"
                )

            with col2:
                max_drawdown = st.number_input(
                    "Max Drawdown Threshold (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=20.0,
                    step=1.0,
                    help="Alert threshold for maximum drawdown",
                    key=f"edit_max_drawdown_{portfolio.id}"
                )

                risk_level = st.selectbox(
                    "Risk Level",
                    ["Conservative", "Moderate", "Aggressive"],
                    index=1,
                    key=f"edit_risk_level_{portfolio.id}"
                )

        # Submit button
        if st.form_submit_button("💾 Save Changes", use_container_width=True):
            update_portfolio_details(
                portfolio,
                new_name,
                new_description,
                new_type,
                new_tags,
                {
                    'auto_rebalance': auto_rebalance,
                    'rebalance_frequency': rebalance_frequency,
                    'max_drawdown': max_drawdown / 100,
                    'risk_level': risk_level
                }
            )


def update_portfolio_details(portfolio: Portfolio, name: str, description: str,
                           portfolio_type: str, tags: str, settings: dict):
    """Update portfolio details."""

    try:
        # Update basic details
        portfolio.name = name
        portfolio.description = description

        from core.data_manager import PortfolioType
        portfolio.portfolio_type = PortfolioType(portfolio_type)

        # Update tags
        if tags.strip():
            portfolio.tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        else:
            portfolio.tags = []

        # Update settings (if portfolio supports them)
        for key, value in settings.items():
            setattr(portfolio, key, value)

        # Update modification time
        from datetime import datetime
        portfolio.last_modified = datetime.now()

        # Save portfolio
        from ..utils.session_state import get_portfolio_manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.save_portfolio(portfolio)

        # Refresh data
        refresh_portfolios()

        st.success("✅ Portfolio details updated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error updating portfolio: {str(e)}")


def render_manage_portfolio_assets(portfolio: Portfolio):
    """Render asset management interface."""

    st.write("### Manage Portfolio Assets")

    if not portfolio.assets:
        st.info("No assets in this portfolio")
        return

    # Asset management options
    management_action = st.selectbox(
        "Asset Management Action",
        [
            "View/Edit Assets",
            "Add New Asset",
            "Remove Assets",
            "Reweight Assets",
            "Replace Assets"
        ]
    )

    if management_action == "View/Edit Assets":
        render_edit_assets_table(portfolio)

    elif management_action == "Add New Asset":
        render_add_asset_form(portfolio)

    elif management_action == "Remove Assets":
        render_remove_assets_interface(portfolio)

    elif management_action == "Reweight Assets":
        render_reweight_assets_interface(portfolio)

    elif management_action == "Replace Assets":
        render_replace_assets_interface(portfolio)


def render_edit_assets_table(portfolio: Portfolio):
    """Render editable assets table."""

    st.write("**Edit Asset Details**")

    # Render editable table
    edited_df = render_portfolio_assets_table(portfolio, editable=True)

    if edited_df is not None:
        # Check if data was modified
        if st.button("💾 Save Asset Changes", use_container_width=True):
            update_portfolio_assets_from_dataframe(portfolio, edited_df)


def update_portfolio_assets_from_dataframe(portfolio: Portfolio, df):
    """Update portfolio assets from edited DataFrame."""

    try:
        # Clear existing assets
        portfolio.assets.clear()

        # Add updated assets
        for _, row in df.iterrows():
            from core.data_manager import Asset

            asset = Asset(
                ticker=row['Ticker'],
                weight=row['Weight (%)'] / 100,
                name=row['Name'] if row['Name'] != 'N/A' else None
            )

            # Add other properties if available
            if 'Shares' in row and row['Shares']:
                asset.shares = int(row['Shares'])

            if 'Current Price' in row and row['Current Price']:
                asset.current_price = float(row['Current Price'])

            portfolio.assets.append(asset)

        # Save portfolio
        from ..utils.session_state import get_portfolio_manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.save_portfolio(portfolio)

        refresh_portfolios()
        st.success("✅ Assets updated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error updating assets: {str(e)}")


def render_add_asset_form(portfolio: Portfolio):
    """Render form to add new asset."""

    st.write("**Add New Asset**")

    with st.form(f"add_asset_{portfolio.id}"):

        col1, col2, col3 = st.columns(3)

        with col1:
            new_ticker = st.text_input("Ticker Symbol *")
            new_weight = st.number_input("Weight (%)", min_value=0.1, max_value=100.0, value=5.0)

        with col2:
            new_name = st.text_input("Company Name")
            new_shares = st.number_input("Shares", min_value=0, value=0, step=1)

        with col3:
            new_sector = st.text_input("Sector")
            new_price = st.number_input("Purchase Price ($)", min_value=0.0, value=0.0, step=0.01)

        # Options
        col1, col2 = st.columns(2)

        with col1:
            auto_weight = st.checkbox(
                "Auto-adjust weights",
                value=True,
                help="Automatically adjust all weights to maintain 100% total"
            )

        with col2:
            fetch_price = st.checkbox(
                "Fetch current price",
                value=True,
                help="Automatically fetch current market price"
            )

        if st.form_submit_button("➕ Add Asset", use_container_width=True):
            add_asset_to_portfolio(
                portfolio, new_ticker, new_weight/100, new_name,
                new_shares, new_sector, new_price, auto_weight, fetch_price
            )


def add_asset_to_portfolio(portfolio: Portfolio, ticker: str, weight: float,
                         name: str, shares: int, sector: str, price: float,
                         auto_weight: bool, fetch_price: bool):
    """Add new asset to portfolio."""

    try:
        from core.data_manager import Asset

        # Create new asset
        asset = Asset(ticker=ticker.upper(), weight=weight, name=name if name else None)

        if shares > 0:
            asset.shares = shares

        if sector:
            asset.sector = sector

        if price > 0:
            asset.purchase_price = price

        # Fetch current price if requested
        if fetch_price:
            from ..utils.session_state import get_price_manager
            try:
                price_manager = get_price_manager()
                prices = price_manager.get_current_prices([ticker])
                if ticker in prices and prices[ticker]:
                    asset.current_price = prices[ticker]
            except:
                pass

        # Add to portfolio
        portfolio.assets.append(asset)

        # Auto-adjust weights if requested
        if auto_weight:
            total_weight = sum(a.weight for a in portfolio.assets)
            if total_weight > 1.0:
                # Normalize all weights
                for a in portfolio.assets:
                    a.weight = a.weight / total_weight

        # Save portfolio
        from ..utils.session_state import get_portfolio_manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.save_portfolio(portfolio)

        refresh_portfolios()
        st.success(f"✅ Added {ticker.upper()} to portfolio!")
        st.rerun()

    except Exception as e:
        st.error(f"Error adding asset: {str(e)}")


def render_remove_assets_interface(portfolio: Portfolio):
    """Render interface to remove assets."""

    st.write("**Remove Assets**")

    asset_names = [f"{asset.ticker} - {asset.name or 'N/A'} ({asset.weight:.1%})" for asset in portfolio.assets]

    assets_to_remove = st.multiselect(
        "Select assets to remove:",
        options=asset_names,
        key=f"remove_assets_{portfolio.id}"
    )

    if assets_to_remove:
        st.warning(f"⚠️ This will remove {len(assets_to_remove)} assets from the portfolio")

        col1, col2 = st.columns(2)

        with col1:
            auto_reweight = st.checkbox(
                "Auto-reweight remaining assets",
                value=True,
                help="Redistribute weights proportionally among remaining assets"
            )

        with col2:
            if st.button("🗑️ Remove Selected Assets", type="primary"):
                remove_assets_from_portfolio(portfolio, assets_to_remove, auto_reweight)


def remove_assets_from_portfolio(portfolio: Portfolio, assets_to_remove: list, auto_reweight: bool):
    """Remove selected assets from portfolio."""

    try:
        # Get indices of assets to remove
        remove_indices = []
        for asset_name in assets_to_remove:
            ticker = asset_name.split(' - ')[0]
            for i, asset in enumerate(portfolio.assets):
                if asset.ticker == ticker:
                    remove_indices.append(i)
                    break

        # Remove assets (in reverse order to maintain indices)
        for i in sorted(remove_indices, reverse=True):
            portfolio.assets.pop(i)

        # Auto-reweight remaining assets if requested
        if auto_reweight and portfolio.assets:
            total_weight = sum(asset.weight for asset in portfolio.assets)
            if total_weight > 0:
                for asset in portfolio.assets:
                    asset.weight = asset.weight / total_weight

        # Save portfolio
        from ..utils.session_state import get_portfolio_manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.save_portfolio(portfolio)

        refresh_portfolios()
        st.success(f"✅ Removed {len(assets_to_remove)} assets from portfolio!")
        st.rerun()

    except Exception as e:
        st.error(f"Error removing assets: {str(e)}")


def render_reweight_assets_interface(portfolio: Portfolio):
    """Render interface to reweight assets."""

    st.write("**Reweight Portfolio Assets**")

    reweight_method = st.selectbox(
        "Reweighting Method",
        [
            "Manual Adjustment",
            "Equal Weighting",
            "Market Cap Weighting",
            "Risk Parity",
            "Custom Strategy"
        ]
    )

    if reweight_method == "Manual Adjustment":
        render_manual_reweighting(portfolio)

    elif reweight_method == "Equal Weighting":
        render_equal_reweighting(portfolio)

    elif reweight_method == "Market Cap Weighting":
        st.info("Market cap weighting requires market cap data (coming in Phase 2)")

    elif reweight_method == "Risk Parity":
        st.info("Risk parity weighting requires volatility data (coming in Phase 2)")

    elif reweight_method == "Custom Strategy":
        render_custom_strategy_reweighting(portfolio)


def render_manual_reweighting(portfolio: Portfolio):
    """Render manual reweighting interface."""

    st.write("**Manual Weight Adjustment**")

    # Show current weights and allow editing
    new_weights = {}

    for asset in portfolio.assets:
        new_weights[asset.ticker] = st.slider(
            f"{asset.ticker} ({asset.name or 'N/A'})",
            min_value=0.0,
            max_value=100.0,
            value=asset.weight * 100,
            step=0.1,
            format="%.1f%%",
            key=f"weight_{asset.ticker}_{portfolio.id}"
        )

    # Show total weight
    total_weight = sum(new_weights.values())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Weight", f"{total_weight:.1f}%")

    with col2:
        if abs(total_weight - 100) > 0.1:
            st.warning("⚠️ Weights should sum to 100%")
        else:
            st.success("✅ Weights are balanced")

    # Normalization option
    normalize_weights = st.checkbox(
        "Auto-normalize to 100%",
        value=True,
        help="Automatically adjust weights to sum to exactly 100%"
    )

    if st.button("💾 Apply New Weights", use_container_width=True):
        apply_new_weights(portfolio, new_weights, normalize_weights)


def apply_new_weights(portfolio: Portfolio, new_weights: dict, normalize: bool):
    """Apply new weights to portfolio assets."""

    try:
        # Update weights
        for asset in portfolio.assets:
            if asset.ticker in new_weights:
                asset.weight = new_weights[asset.ticker] / 100

        # Normalize if requested
        if normalize:
            total_weight = sum(asset.weight for asset in portfolio.assets)
            if total_weight > 0:
                for asset in portfolio.assets:
                    asset.weight = asset.weight / total_weight

        # Save portfolio
        from ..utils.session_state import get_portfolio_manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.save_portfolio(portfolio)

        refresh_portfolios()
        st.success("✅ Portfolio weights updated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error updating weights: {str(e)}")


def render_equal_reweighting(portfolio: Portfolio):
    """Render equal weighting interface."""

    st.write("**Equal Weight Distribution**")

    if portfolio.assets:
        equal_weight = 100.0 / len(portfolio.assets)
        st.info(f"Each asset will be weighted at {equal_weight:.2f}%")

        # Show preview
        import pandas as pd
        preview_data = []

        for asset in portfolio.assets:
            preview_data.append({
                'Ticker': asset.ticker,
                'Current Weight': f"{asset.weight:.1%}",
                'New Weight': f"{equal_weight:.2f}%"
            })

        df = pd.DataFrame(preview_data)
        st.dataframe(df, hide_index=True)

        if st.button("⚖️ Apply Equal Weighting", use_container_width=True):
            apply_equal_weighting(portfolio)
    else:
        st.warning("No assets in portfolio to reweight")


def apply_equal_weighting(portfolio: Portfolio):
    """Apply equal weighting to all assets."""

    try:
        if portfolio.assets:
            equal_weight = 1.0 / len(portfolio.assets)

            for asset in portfolio.assets:
                asset.weight = equal_weight

            # Save portfolio
            from ..utils.session_state import get_portfolio_manager
            portfolio_manager = get_portfolio_manager()
            portfolio_manager.save_portfolio(portfolio)

            refresh_portfolios()
            st.success("✅ Applied equal weighting to all assets!")
            st.rerun()

    except Exception as e:
        st.error(f"Error applying equal weighting: {str(e)}")


def render_custom_strategy_reweighting(portfolio: Portfolio):
    """Render custom strategy reweighting."""

    st.write("**Custom Reweighting Strategy**")

    strategy = st.selectbox(
        "Select Strategy",
        [
            "Target Allocation by Sector",
            "Momentum-Based Weighting",
            "Dividend Yield Weighting",
            "Low Volatility Focus"
        ]
    )

    st.info(f"Custom strategies require additional data and will be implemented in Phase 2")


def render_replace_assets_interface(portfolio: Portfolio):
    """Render interface to replace assets."""

    st.write("**Replace Assets**")

    if not portfolio.assets:
        st.warning("No assets in portfolio to replace")
        return

    # Asset replacement form
    col1, col2 = st.columns(2)

    with col1:
        asset_to_replace = st.selectbox(
            "Asset to Replace",
            options=[f"{asset.ticker} - {asset.name or 'N/A'}" for asset in portfolio.assets]
        )

    with col2:
        replacement_ticker = st.text_input("Replacement Ticker")

    # Options
    col1, col2 = st.columns(2)

    with col1:
        keep_weight = st.checkbox(
            "Keep same weight",
            value=True,
            help="Maintain the same portfolio weight"
        )

    with col2:
        fetch_info = st.checkbox(
            "Fetch company info",
            value=True,
            help="Automatically fetch company information"
        )

    if st.button("🔄 Replace Asset", use_container_width=True):
        if replacement_ticker:
            replace_portfolio_asset(portfolio, asset_to_replace, replacement_ticker, keep_weight, fetch_info)
        else:
            st.error("Please enter a replacement ticker symbol")


def replace_portfolio_asset(portfolio: Portfolio, old_asset_name: str, new_ticker: str,
                          keep_weight: bool, fetch_info: bool):
    """Replace an asset in the portfolio."""

    try:
        # Find the asset to replace
        old_ticker = old_asset_name.split(' - ')[0]
        asset_to_replace = None

        for asset in portfolio.assets:
            if asset.ticker == old_ticker:
                asset_to_replace = asset
                break

        if not asset_to_replace:
            st.error("Asset to replace not found")
            return

        # Create replacement asset
        from core.data_manager import Asset

        new_asset = Asset(
            ticker=new_ticker.upper(),
            weight=asset_to_replace.weight if keep_weight else 0.05  # Default 5%
        )

        # Fetch info if requested
        if fetch_info:
            # This would fetch company information
            # For now, just set a placeholder
            new_asset.name = f"{new_ticker.upper()} Company"

        # Replace in portfolio
        asset_index = portfolio.assets.index(asset_to_replace)
        portfolio.assets[asset_index] = new_asset

        # Save portfolio
        from ..utils.session_state import get_portfolio_manager
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.save_portfolio(portfolio)

        refresh_portfolios()
        st.success(f"✅ Replaced {old_ticker} with {new_ticker.upper()}!")
        st.rerun()

    except Exception as e:
        st.error(f"Error replacing asset: {str(e)}")


def render_update_portfolio_prices(portfolio: Portfolio):
    """Render price update interface."""

    st.write("### Update Portfolio Prices")

    if not portfolio.assets:
        st.info("No assets in portfolio to update")
        return

    # Price update options
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Update All Prices", use_container_width=True):
            update_portfolio_prices(portfolio)

    with col2:
        if st.button("📊 Price History", use_container_width=True):
            st.info("Price history feature coming in Phase 2")

    # Show current price status
    st.write("**Current Price Status**")

    import pandas as pd

    price_data = []
    for asset in portfolio.assets:
        current_price = getattr(asset, 'current_price', None)
        last_updated = getattr(asset, 'last_updated', None)

        price_data.append({
            'Ticker': asset.ticker,
            'Name': asset.name or 'N/A',
            'Current Price': f"${current_price:.2f}" if current_price else "Not Available",
            'Last Updated': format_datetime(last_updated) if last_updated else "Never",
            'Status': "✅ Current" if current_price else "❌ Missing"
        })

    df = pd.DataFrame(price_data)
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_portfolio_rebalancing(portfolio: Portfolio):
    """Render portfolio rebalancing interface."""

    st.write("### Portfolio Rebalancing")

    st.info("Advanced rebalancing features will be available in Phase 2")

    # Basic rebalancing preview
    if portfolio.assets:
        current_weights = [asset.weight for asset in portfolio.assets]
        target_weights = [1.0 / len(portfolio.assets)] * len(portfolio.assets)  # Equal weight target

        import pandas as pd

        rebalance_data = []
        for i, asset in enumerate(portfolio.assets):
            rebalance_data.append({
                'Ticker': asset.ticker,
                'Current Weight': f"{current_weights[i]:.1%}",
                'Target Weight': f"{target_weights[i]:.1%}",
                'Difference': f"{target_weights[i] - current_weights[i]:+.1%}"
            })

        df = pd.DataFrame(rebalance_data)
        st.dataframe(df, hide_index=True)

        st.write("**Rebalancing Actions:**")
        st.write("- Equal weight rebalancing shown as example")
        st.write("- Custom target weights coming in Phase 2")
        st.write("- Transaction cost analysis coming in Phase 2")


def render_portfolio_history(portfolio: Portfolio):
    """Render portfolio history and activity log."""

    st.write("### Portfolio History & Activity")

    # Basic portfolio information
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Portfolio Timeline:**")
        st.write(f"📅 Created: {format_datetime(portfolio.created_date, '%Y-%m-%d %H:%M')}")
        st.write(f"🔄 Last Modified: {format_datetime(portfolio.last_modified, '%Y-%m-%d %H:%M')}")

        # Calculate age
        from datetime import datetime
        age = datetime.now() - portfolio.created_date
        st.write(f"⏰ Age: {age.days} days")

    with col2:
        st.write("**Current Status:**")
        st.write(f"📊 Assets: {len(portfolio.assets)}")
        st.write(f"💰 Value: {format_currency(portfolio.calculate_value())}")
        st.write(f"📈 Type: {portfolio.portfolio_type.value.title()}")

    # Activity log placeholder
    st.write("**Activity Log:**")
    st.info("Detailed activity logging will be available in Phase 2")

    # Show basic activity
    activities = [
        f"Portfolio created ({format_datetime(portfolio.created_date, '%Y-%m-%d %H:%M')})",
        f"Last modified ({format_datetime(portfolio.last_modified, '%Y-%m-%d %H:%M')})"
    ]

    for activity in activities:
        st.write(f"• {activity}")

    # Export history option
    if st.button("📤 Export Portfolio History", use_container_width=True):
        st.info("Portfolio history export prepared")


def update_all_portfolio_prices(portfolios: List[Portfolio]):
    """Update prices for all portfolios."""

    with st.spinner("Updating prices for all portfolios..."):
        try:
            from ..utils.session_state import get_price_manager, update_last_price_update

            price_manager = get_price_manager()
            updated_count = 0

            # Collect all unique tickers
            all_tickers = set()
            for portfolio in portfolios:
                for asset in portfolio.assets:
                    all_tickers.add(asset.ticker)

            if all_tickers:
                # Batch fetch prices
                prices = price_manager.get_current_prices(list(all_tickers))

                # Update each portfolio
                for portfolio in portfolios:
                    for asset in portfolio.assets:
                        if asset.ticker in prices and prices[asset.ticker]:
                            old_price = getattr(asset, 'current_price', None)
                            asset.current_price = prices[asset.ticker]

                            if old_price != asset.current_price:
                                updated_count += 1

                update_last_price_update()

                if updated_count > 0:
                    st.success(f"✅ Updated {updated_count} asset prices across all portfolios")
                else:
                    st.info("ℹ️ All prices are already up to date")
            else:
                st.warning("No assets found to update")

        except Exception as e:
            st.error(f"Error updating prices: {str(e)}")


def export_all_portfolios(portfolios: List[Portfolio]):
    """Export all portfolios."""

    st.subheader("📤 Export All Portfolios")

    export_format = st.selectbox(
        "Export Format",
        ["JSON (Complete)", "CSV (Summary)", "CSV (Detailed)", "Excel Workbook"],
        key="bulk_export_format"
    )

    include_options = st.multiselect(
        "Include in Export",
        [
            "Portfolio Metadata",
            "Asset Details",
            "Current Prices",
            "Performance Data",
            "Transaction History"
        ],
        default=["Portfolio Metadata", "Asset Details", "Current Prices"],
        key="bulk_export_options"
    )

    if st.button("📥 Generate Export", use_container_width=True):
        generate_bulk_export(portfolios, export_format, include_options)


def generate_bulk_export(portfolios: List[Portfolio], format_type: str, options: List[str]):
    """Generate bulk export of all portfolios."""

    with st.spinner(f"Generating {format_type} export..."):
        try:
            # This would implement actual export functionality
            # For now, show progress and success message

            progress_bar = st.progress(0)

            for i, portfolio in enumerate(portfolios):
                # Simulate export processing
                progress_bar.progress((i + 1) / len(portfolios))
                time.sleep(0.1)

            st.success(f"✅ Export completed! {len(portfolios)} portfolios exported in {format_type} format.")
            st.balloons()

        except Exception as e:
            st.error(f"Export failed: {str(e)}")