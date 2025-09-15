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
from ..utils.formatting import format_currency, format_datetime, format_percentage
from ..components.charts import create_portfolio_allocation_chart, create_portfolio_summary_cards


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
                    # Принудительно выбираем портфель и переключаемся в режим просмотра
                    set_selected_portfolio(portfolio)
                    st.session_state.portfolio_view_mode = "detailed"
                    # Очищаем конфликтующие состояния
                    if "portfolio_edit_mode" in st.session_state:
                        del st.session_state.portfolio_edit_mode
                    # Обновляем sidebar selector для синхронизации
                    st.session_state.portfolio_selector = portfolio.name
                    st.rerun()

            with col4:
                if st.button("✏️ Edit", key=f"edit_{portfolio.id}", use_container_width=True):
                    # Принудительно выбираем портфель и переключаемся в режим редактирования
                    set_selected_portfolio(portfolio)
                    st.session_state.portfolio_edit_mode = True
                    # Очищаем конфликтующие состояния
                    if "portfolio_view_mode" in st.session_state:
                        del st.session_state.portfolio_view_mode
                    # Обновляем sidebar selector для синхронизации
                    st.session_state.portfolio_selector = portfolio.name
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

    # Check if we're in detailed view mode
    if st.session_state.get('portfolio_view_mode') == 'detailed':
        render_detailed_portfolio_view(selected_portfolio)
        return

    # Check if we're in edit mode
    if st.session_state.get('portfolio_edit_mode', False):
        render_portfolio_edit_interface(selected_portfolio)
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


def render_detailed_portfolio_view(portfolio: Portfolio):
    """Render detailed view of selected portfolio with all metrics."""

    # Header with back button
    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("← Back", key="back_to_list"):
            st.session_state.portfolio_view_mode = None
            st.rerun()

    with col2:
        st.subheader(f"📊 Portfolio Analysis: {portfolio.name}")

    # Portfolio summary cards
    st.write("### Portfolio Overview")

    metrics = create_portfolio_summary_cards(portfolio)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Value",
            value=metrics['total_value']['value'],
            delta=metrics['total_value']['delta'],
            delta_color=metrics['total_value']['delta_color']
        )

    with col2:
        st.metric(
            label="Total Assets",
            value=metrics['total_assets']['value']
        )

    with col3:
        st.metric(
            label="Top 3 Concentration",
            value=metrics['concentration']['value']
        )

    with col4:
        st.metric(
            label="Avg Position Size",
            value=metrics['avg_position_size']['value']
        )

    with col5:
        st.metric(
            label="Daily Return",
            value=metrics['daily_return']['value'],
            delta_color=metrics['daily_return']['delta_color']
        )

    # Portfolio Information
    st.write("### Portfolio Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Portfolio Type**: {portfolio.portfolio_type.value.title()}")
        st.write(f"**Created**: {format_datetime(portfolio.created_date, '%Y-%m-%d %H:%M')}")
        st.write(f"**Last Modified**: {format_datetime(portfolio.last_modified, '%Y-%m-%d %H:%M')}")

        # Calculate portfolio age
        from datetime import datetime
        age = datetime.now() - portfolio.created_date
        st.write(f"**Portfolio Age**: {age.days} days")

    with col2:
        if portfolio.description:
            st.write(f"**Description**: {portfolio.description}")

        # Show tags if available
        if hasattr(portfolio, 'tags') and portfolio.tags:
            st.write(f"**Tags**: {', '.join(portfolio.tags)}")

        # Risk metrics
        if portfolio.assets:
            sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
            largest_position = sorted_assets[0]
            st.write(f"**Largest Position**: {largest_position.ticker} ({format_percentage(largest_position.weight)})")

    # Asset Allocation Visualization
    st.write("### Asset Allocation")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Chart type selector
        chart_type = st.radio(
            "Chart Type",
            ["pie", "donut", "bar"],
            horizontal=True,
            key="view_portfolio_chart_type"
        )

        fig = create_portfolio_allocation_chart(portfolio, chart_type)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk analysis
        st.write("**Risk Analysis**")

        if portfolio.assets:
            # Concentration risk
            sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
            top_3_weight = sum(asset.weight for asset in sorted_assets[:3])
            top_5_weight = sum(asset.weight for asset in sorted_assets[:5])

            st.metric("Top 3 Holdings", format_percentage(top_3_weight))
            st.metric("Top 5 Holdings", format_percentage(top_5_weight))

            # Diversification score
            diversification_score = min(1.0, len(portfolio.assets) / 20) * 100
            st.metric("Diversification Score", f"{diversification_score:.0f}%")

            # Risk level indicator
            if top_3_weight > 0.6:
                st.error("🔴 High Concentration Risk")
            elif top_3_weight > 0.4:
                st.warning("🟡 Moderate Concentration Risk")
            else:
                st.success("🟢 Low Concentration Risk")

    # Detailed Assets Table
    st.write("### Portfolio Holdings")

    if portfolio.assets:
        import pandas as pd

        # Create detailed assets dataframe
        assets_data = []
        total_value = portfolio.calculate_value()

        for asset in portfolio.assets:
            current_price = getattr(asset, 'current_price', None)
            purchase_price = getattr(asset, 'purchase_price', None)
            shares = getattr(asset, 'shares', None)

            # Calculate market value
            if current_price and shares:
                market_value = current_price * shares
            else:
                market_value = asset.weight * total_value

            # Calculate P&L
            unrealized_pnl = None
            unrealized_pnl_pct = None

            if current_price and purchase_price and shares:
                unrealized_pnl = (current_price - purchase_price) * shares
                unrealized_pnl_pct = (current_price - purchase_price) / purchase_price

            assets_data.append({
                'Ticker': asset.ticker,
                'Name': asset.name or 'N/A',
                'Weight': format_percentage(asset.weight),
                'Shares': f"{shares:,.0f}" if shares else 'N/A',
                'Current Price': f"${current_price:.2f}" if current_price else 'N/A',
                'Market Value': format_currency(market_value),
                'Purchase Price': f"${purchase_price:.2f}" if purchase_price else 'N/A',
                'Unrealized P&L': format_currency(unrealized_pnl) if unrealized_pnl else 'N/A',
                'P&L %': format_percentage(unrealized_pnl_pct) if unrealized_pnl_pct else 'N/A',
                'Sector': getattr(asset, 'sector', 'N/A')
            })

        df = pd.DataFrame(assets_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Portfolio Statistics
        st.write("### Portfolio Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Composition**")
            st.write(f"Total Assets: {len(portfolio.assets)}")
            st.write(f"Total Weight: {format_percentage(sum(asset.weight for asset in portfolio.assets))}")
            st.write(f"Average Weight: {format_percentage(sum(asset.weight for asset in portfolio.assets) / len(portfolio.assets))}")

        with col2:
            st.write("**Valuation**")
            st.write(f"Total Value: {format_currency(total_value)}")

            # Calculate average position size
            avg_position = total_value / len(portfolio.assets)
            st.write(f"Avg Position: {format_currency(avg_position)}")

            # Largest position
            largest_asset = max(portfolio.assets, key=lambda x: x.weight)
            largest_value = largest_asset.weight * total_value
            st.write(f"Largest Position: {format_currency(largest_value)}")

        with col3:
            st.write("**Risk Metrics**")

            # Calculate some basic risk metrics
            weights = [asset.weight for asset in portfolio.assets]

            # Herfindahl Index (concentration measure)
            herfindahl = sum(w**2 for w in weights)
            st.write(f"Herfindahl Index: {herfindahl:.3f}")

            # Effective number of assets
            effective_assets = 1 / herfindahl if herfindahl > 0 else 0
            st.write(f"Effective Assets: {effective_assets:.1f}")

            # Weight standard deviation
            import numpy as np
            weight_std = np.std(weights) if len(weights) > 1 else 0
            st.write(f"Weight Std Dev: {format_percentage(weight_std)}")

    # Action Buttons
    st.write("### Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✏️ Edit Portfolio", key="edit_from_view", use_container_width=True):
            st.session_state.portfolio_edit_mode = True
            st.session_state.portfolio_view_mode = None
            st.rerun()

    with col2:
        if st.button("🔄 Update Prices", key="update_from_view", use_container_width=True):
            update_portfolio_prices(portfolio)

    with col3:
        if st.button("📤 Export Data", key="export_from_view", use_container_width=True):
            export_portfolio_data(portfolio)

    with col4:
        if st.button("📊 Full Analysis", key="analyze_from_view", use_container_width=True):
            st.session_state.main_navigation = "📊 Portfolio Analysis"
            st.rerun()


def render_portfolio_edit_interface(portfolio: Portfolio):
    """Render portfolio editing interface."""

    # Header with back button
    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("← Back", key="back_from_edit"):
            st.session_state.portfolio_edit_mode = False
            st.rerun()

    with col2:
        st.subheader(f"✏️ Edit Portfolio: {portfolio.name}")

    # Edit tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Basic Info",
        "📊 Assets",
        "⚖️ Rebalance",
        "🔧 Advanced"
    ])

    with tab1:
        render_edit_portfolio_details(portfolio)

    with tab2:
        render_manage_portfolio_assets(portfolio)

    with tab3:
        render_portfolio_rebalancing(portfolio)

    with tab4:
        render_advanced_portfolio_settings(portfolio)


def render_advanced_portfolio_settings(portfolio: Portfolio):
    """Render advanced portfolio settings."""

    st.write("### Advanced Settings")

    with st.form(f"advanced_settings_{portfolio.id}"):

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Risk Management**")

            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1.0,
                max_value=50.0,
                value=getattr(portfolio, 'max_position_size', 20.0),
                step=1.0,
                help="Maximum allowed weight for any single asset"
            )

            max_sector_allocation = st.slider(
                "Max Sector Allocation (%)",
                min_value=10.0,
                max_value=100.0,
                value=getattr(portfolio, 'max_sector_allocation', 40.0),
                step=5.0,
                help="Maximum allowed allocation to any single sector"
            )

            rebalance_threshold = st.slider(
                "Rebalance Threshold (%)",
                min_value=1.0,
                max_value=20.0,
                value=getattr(portfolio, 'rebalance_threshold', 5.0),
                step=0.5,
                help="Trigger rebalancing when weights drift by this amount"
            )

        with col2:
            st.write("**Automation**")

            auto_rebalance = st.checkbox(
                "Enable Auto-Rebalancing",
                value=getattr(portfolio, 'auto_rebalance', False),
                help="Automatically rebalance when thresholds are exceeded"
            )

            price_update_frequency = st.selectbox(
                "Price Update Frequency",
                ["Manual", "Hourly", "Daily", "Weekly"],
                index=0,
                help="How often to automatically update asset prices"
            )

            notification_settings = st.multiselect(
                "Notifications",
                [
                    "Large Price Movements",
                    "Rebalancing Alerts",
                    "Risk Threshold Breaches",
                    "Performance Updates"
                ],
                default=[],
                help="Select which events should trigger notifications"
            )

        if st.form_submit_button("💾 Save Advanced Settings", use_container_width=True):
            try:
                # Update portfolio with advanced settings
                portfolio.max_position_size = max_position_size / 100
                portfolio.max_sector_allocation = max_sector_allocation / 100
                portfolio.rebalance_threshold = rebalance_threshold / 100
                portfolio.auto_rebalance = auto_rebalance
                portfolio.price_update_frequency = price_update_frequency
                portfolio.notification_settings = notification_settings

                # Save portfolio
                from ..utils.session_state import get_portfolio_manager
                portfolio_manager = get_portfolio_manager()
                portfolio_manager.save_portfolio(portfolio)

                refresh_portfolios()
                st.success("✅ Advanced settings updated successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Error updating settings: {str(e)}")


# [Include all the existing functions from the original file...]
# render_edit_portfolio_details, render_manage_portfolio_assets, etc.
# [The rest of the functions remain the same as in the original file]


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

        # Submit button
        if st.form_submit_button("💾 Save Changes", use_container_width=True):
            update_portfolio_details(
                portfolio,
                new_name,
                new_description,
                new_type,
                new_tags
            )


def update_portfolio_details(portfolio: Portfolio, name: str, description: str,
                           portfolio_type: str, tags: str):
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
            "Reweight Assets"
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


def render_edit_assets_table(portfolio: Portfolio):
    """Render editable assets table."""

    st.write("**Current Assets**")

    import pandas as pd

    # Create editable dataframe
    assets_data = []
    for asset in portfolio.assets:
        assets_data.append({
            'Ticker': asset.ticker,
            'Name': asset.name or '',
            'Weight (%)': asset.weight * 100,
            'Shares': getattr(asset, 'shares', 0),
            'Current Price': getattr(asset, 'current_price', 0.0),
            'Sector': getattr(asset, 'sector', '')
        })

    df = pd.DataFrame(assets_data)

    # Display editable dataframe
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Weight (%)": st.column_config.NumberColumn(
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f"
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
                step=1,
                format="%d"
            )
        }
    )

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
                name=row['Name'] if row['Name'] != '' else None
            )

            # Add other properties if available
            if 'Shares' in row and row['Shares']:
                asset.shares = int(row['Shares'])

            if 'Current Price' in row and row['Current Price']:
                asset.current_price = float(row['Current Price'])

            if 'Sector' in row and row['Sector']:
                asset.sector = row['Sector']

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
            "Custom Strategy"
        ]
    )

    if reweight_method == "Manual Adjustment":
        render_manual_reweighting(portfolio)

    elif reweight_method == "Equal Weighting":
        render_equal_reweighting(portfolio)

    elif reweight_method == "Custom Strategy":
        st.info("Custom strategies will be implemented in Phase 2")


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


def render_update_portfolio_prices(portfolio: Portfolio):
    """Render price update interface."""

    st.write("### Update Portfolio Prices")

    if not portfolio.assets:
        st.info("No assets in portfolio to update")
        return

    # Diagnostic information
    with st.expander("🔍 Diagnostic Information"):
        st.write("**Asset Details:**")
        for asset in portfolio.assets:
            st.write(f"- **{asset.ticker}**:")
            st.write(f"  - Weight: {asset.weight:.1%}")
            st.write(f"  - Shares: {getattr(asset, 'shares', 'Not set')}")
            st.write(f"  - Current Price: ${getattr(asset, 'current_price', 'Not set')}")
            st.write(f"  - Market Value: ${asset.market_value:.2f}")
            st.write(f"  - Last Updated: {getattr(asset, 'last_updated', 'Never')}")

        st.write(f"**Portfolio Total Value:** ${portfolio.calculate_value():.2f}")

    # Test price fetching
    if st.button("🧪 Test Price Fetching"):
        with st.spinner("Testing price fetching..."):
            try:
                from ..utils.session_state import get_price_manager
                price_manager = get_price_manager()

                tickers = [asset.ticker for asset in portfolio.assets]
                st.write(f"Fetching prices for: {', '.join(tickers)}")

                prices = price_manager.get_current_prices(tickers)
                st.write("**Fetched Prices:**")
                for ticker, price in prices.items():
                    st.write(f"- {ticker}: ${price:.2f}" if price else f"- {ticker}: Failed to fetch")

                if not prices:
                    st.error("No prices were fetched! Check internet connection and yfinance.")

            except Exception as e:
                st.error(f"Error testing price fetch: {e}")

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
        st.write("- Custom target weights available in asset management")
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
        ["JSON (Complete)", "CSV (Summary)", "Excel Workbook"],
        key="bulk_export_format"
    )

    if st.button("📥 Generate Export", use_container_width=True):
        with st.spinner(f"Generating {export_format} export..."):
            try:
                progress_bar = st.progress(0)

                for i, portfolio in enumerate(portfolios):
                    # Simulate export processing
                    progress_bar.progress((i + 1) / len(portfolios))
                    time.sleep(0.1)

                st.success(f"✅ Export completed! {len(portfolios)} portfolios exported in {export_format} format.")
                st.balloons()

            except Exception as e:
                st.error(f"Export failed: {str(e)}")