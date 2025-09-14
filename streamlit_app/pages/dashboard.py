"""
Dashboard page for the Portfolio Management System.

This module contains the main dashboard view showing portfolio overview and key metrics.
"""
import streamlit as st
from typing import List, Optional
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import Portfolio
from ..utils.session_state import (
    get_portfolios, refresh_portfolios, get_selected_portfolio,
    set_selected_portfolio, get_last_price_update
)
from ..utils.formatting import format_currency, format_percentage, format_datetime
from ..utils.helpers import calculate_portfolio_metrics
from ..components.tables import render_portfolio_overview_table



def render_dashboard():
    """Render the main dashboard page."""

    st.header("🏠 Portfolio Dashboard")

    # Load portfolios if not loaded
    portfolios = get_portfolios()
    if not portfolios:
        refresh_portfolios()
        portfolios = get_portfolios()

    if not portfolios:
        render_empty_dashboard()
        return

    # Dashboard layout
    render_dashboard_header(portfolios)
    render_portfolio_overview(portfolios)
    render_selected_portfolio_details()
    render_market_summary()


def render_empty_dashboard():
    """Render dashboard when no portfolios exist."""

    st.info("👋 Welcome to Wild Market Capital Portfolio Manager!")

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
            if st.button("📝 Create Portfolio", use_container_width=True, type="primary"):
                st.session_state.main_navigation = "📝 Create Portfolio"
                st.rerun()

        with col2:
            if st.button("📚 View Examples", use_container_width=True):
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


def render_dashboard_header(portfolios: List[Portfolio]):
    """Render dashboard header with key metrics."""

    # Calculate aggregate metrics
    total_portfolios = len(portfolios)
    total_assets = sum(len(p.assets) for p in portfolios)
    total_value = sum(p.calculate_value() for p in portfolios)

    # Get largest portfolio
    largest_portfolio = max(portfolios, key=lambda p: p.calculate_value()) if portfolios else None

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Portfolios",
            value=str(total_portfolios),
            help="Number of portfolios created"
        )

    with col2:
        st.metric(
            label="Total Assets",
            value=str(total_assets),
            help="Total number of individual assets across all portfolios"
        )

    with col3:
        st.metric(
            label="Combined Value",
            value=format_currency(total_value),
            help="Sum of all portfolio values"
        )

    with col4:
        if largest_portfolio:
            st.metric(
                label="Largest Portfolio",
                value=largest_portfolio.name[:15] + "..." if len(
                    largest_portfolio.name) > 15 else largest_portfolio.name,
                delta=format_currency(largest_portfolio.calculate_value()),
                help="Portfolio with highest total value"
            )
        else:
            st.metric("Largest Portfolio", "N/A")


def render_portfolio_overview(portfolios: List[Portfolio]):
    """Render portfolio overview section."""

    st.subheader("📊 Portfolio Overview")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Table View", "📈 Chart View", "🔍 Comparison"])

    with tab1:
        # Portfolio overview table
        render_portfolio_overview_table(portfolios)

        # Quick action buttons
        if portfolios:
            st.subheader("⚡ Quick Actions")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("🔄 Refresh All Prices", use_container_width=True):
                    refresh_all_portfolio_prices(portfolios)

            with col2:
                if st.button("📤 Export All", use_container_width=True):
                    export_all_portfolios(portfolios)

            with col3:
                if st.button("📊 Compare All", use_container_width=True):
                    st.session_state.compare_portfolios = portfolios[:5]  # Limit to 5 for performance
                    st.rerun()

            with col4:
                if st.button("📈 Generate Report", use_container_width=True):
                    generate_dashboard_report(portfolios)

    with tab2:
        # Chart visualizations
        render_portfolio_charts(portfolios)

    with tab3:
        # Portfolio comparison
        render_portfolio_comparison(portfolios)


def render_portfolio_charts(portfolios: List[Portfolio]):
    """Render portfolio visualization charts."""

    if len(portfolios) < 2:
        st.info("Need at least 2 portfolios for chart comparisons")
        return

    # Chart selection
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Portfolio Values", "Asset Distribution", "Risk-Return", "Performance"]
    )

    if chart_type == "Portfolio Values":
        # Portfolio value comparison bar chart
        portfolio_values = [(p.name, p.calculate_value()) for p in portfolios]
        portfolio_values.sort(key=lambda x: x[1], reverse=True)

        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(
                x=[pv[0] for pv in portfolio_values],
                y=[pv[1] for pv in portfolio_values],
                text=[format_currency(pv[1]) for pv in portfolio_values],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Portfolio Values Comparison",
            xaxis_title="Portfolios",
            yaxis_title="Value ($)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Risk-Return":
        # Risk-return scatter plot
        fig = create_risk_return_scatter(portfolios)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Asset Distribution":
        # Show asset distribution across portfolios
        render_asset_distribution_analysis(portfolios)

    elif chart_type == "Performance":
        st.info("Performance charts require historical data (coming in Phase 2)")


def render_asset_distribution_analysis(portfolios: List[Portfolio]):
    """Render asset distribution analysis."""

    # Collect all unique assets
    all_assets = {}

    for portfolio in portfolios:
        for asset in portfolio.assets:
            if asset.ticker not in all_assets:
                all_assets[asset.ticker] = {
                    'ticker': asset.ticker,
                    'name': asset.name or 'N/A',
                    'portfolios': [],
                    'total_weight': 0,
                    'occurrences': 0
                }

            all_assets[asset.ticker]['portfolios'].append(portfolio.name)
            all_assets[asset.ticker]['total_weight'] += asset.weight
            all_assets[asset.ticker]['occurrences'] += 1

    # Sort by occurrence frequency
    asset_list = sorted(all_assets.values(), key=lambda x: x['occurrences'], reverse=True)

    # Top 10 most common assets
    st.subheader("🔥 Most Common Assets")

    top_assets = asset_list[:10]

    col1, col2 = st.columns(2)

    with col1:
        # Frequency chart
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(
                x=[asset['ticker'] for asset in top_assets],
                y=[asset['occurrences'] for asset in top_assets],
                text=[f"{asset['occurrences']} portfolios" for asset in top_assets],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Asset Frequency Across Portfolios",
            xaxis_title="Assets",
            yaxis_title="Number of Portfolios",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Weight distribution
        fig = go.Figure(data=[
            go.Bar(
                x=[asset['ticker'] for asset in top_assets],
                y=[asset['total_weight'] * 100 for asset in top_assets],
                text=[f"{asset['total_weight']:.1%}" for asset in top_assets],
                textposition='auto',
                marker_color='orange'
            )
        ])

        fig.update_layout(
            title="Total Weight Across Portfolios",
            xaxis_title="Assets",
            yaxis_title="Total Weight (%)",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)


def render_portfolio_comparison(portfolios: List[Portfolio]):
    """Render portfolio comparison section."""

    if len(portfolios) < 2:
        st.info("Need at least 2 portfolios for comparison")
        return

    # Portfolio selection for comparison
    st.subheader("📊 Compare Portfolios")

    selected_portfolios = st.multiselect(
        "Select portfolios to compare",
        options=[p.name for p in portfolios],
        default=[p.name for p in portfolios[:min(3, len(portfolios))]]  # Default to first 3
    )

    if len(selected_portfolios) >= 2:
        # Get selected portfolio objects
        selected_objects = [p for p in portfolios if p.name in selected_portfolios]

        # Comparison metrics
        comparison_data = []
        for portfolio in selected_objects:
            metrics = calculate_portfolio_metrics(portfolio)

            comparison_data.append({
                'Portfolio': portfolio.name,
                'Total Value': format_currency(metrics['total_value']),
                'Assets': metrics['total_assets'],
                'Largest Position': f"{metrics['largest_position']['ticker']} ({format_percentage(metrics['largest_position']['weight'])})" if
                metrics['largest_position'] else 'N/A',
                'Concentration Risk': format_percentage(metrics['concentration_risk']),
                'Creation Date': format_datetime(portfolio.created_date, "%Y-%m-%d")
            })

        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Visual comparison chart
        if len(selected_objects) <= 5:  # Limit for performance
            fig = create_portfolio_comparison_chart(selected_objects)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Select at least 2 portfolios for comparison")


def render_selected_portfolio_details():
    """Render details for the selected portfolio."""

    selected_portfolio = get_selected_portfolio()

    if not selected_portfolio:
        st.info("💡 Select a portfolio from the sidebar to view detailed analysis")
        return

    st.subheader(f"🔍 Portfolio Details: {selected_portfolio.name}")

    # Portfolio metrics cards
    metrics = create_portfolio_summary_cards(selected_portfolio)

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

    # Portfolio allocation chart
    col1, col2 = st.columns([2, 1])

    with col1:
        # Allocation chart
        chart_type = st.radio(
            "Chart Type",
            ["pie", "donut", "bar"],
            horizontal=True,
            key="portfolio_chart_type"
        )

        fig = create_portfolio_allocation_chart(selected_portfolio, chart_type)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Portfolio summary info
        st.write("**Portfolio Information**")
        st.write(f"📊 **Type**: {selected_portfolio.portfolio_type.value.title()}")
        st.write(f"📅 **Created**: {format_datetime(selected_portfolio.created_date, '%Y-%m-%d')}")
        st.write(f"🔄 **Modified**: {format_datetime(selected_portfolio.last_modified, '%Y-%m-%d')}")

        if selected_portfolio.description:
            st.write(f"📝 **Description**: {selected_portfolio.description}")

        # Quick actions for selected portfolio
        st.write("**Quick Actions**")

        if st.button("💰 Update Prices", key="update_selected_prices", use_container_width=True):
            update_selected_portfolio_prices(selected_portfolio)

        if st.button("📊 Detailed Analysis", key="analyze_selected", use_container_width=True):
            st.session_state.main_navigation = "📊 Portfolio Analysis"
            st.rerun()

        if st.button("⚙️ Manage Portfolio", key="manage_selected", use_container_width=True):
            st.session_state.main_navigation = "📋 Manage Portfolios"
            st.rerun()


def render_market_summary():
    """Render market summary and system status."""

    st.subheader("📈 Market Summary")

    col1, col2 = st.columns(2)

    with col1:
        # Market status indicators
        st.write("**Market Status**")

        # Placeholder market data
        import numpy as np

        market_data = {
            "S&P 500": {
                "value": "4,750.20",
                "change": np.random.uniform(-50, 50),
                "change_pct": np.random.uniform(-0.02, 0.02)
            },
            "NASDAQ": {
                "value": "15,240.80",
                "change": np.random.uniform(-100, 100),
                "change_pct": np.random.uniform(-0.03, 0.03)
            },
            "Dow Jones": {
                "value": "37,890.40",
                "change": np.random.uniform(-300, 300),
                "change_pct": np.random.uniform(-0.015, 0.015)
            }
        }

        for index, data in market_data.items():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.metric(
                    label=index,
                    value=data["value"],
                    delta=f"{data['change']:+.2f} ({data['change_pct']:+.2%})"
                )

    with col2:
        # System status
        st.write("**System Status**")

        from ..utils.session_state import get_price_manager

        # Price data status
        last_update = get_last_price_update()
        if last_update:
            st.success(f"✅ Price data updated: {format_datetime(last_update, '%H:%M:%S')}")
        else:
            st.warning("⚠️ No recent price updates")

        # Cache status
        try:
            price_manager = get_price_manager()
            cache_stats = price_manager.get_cache_stats()

            if cache_stats.get('cache_enabled'):
                st.info(f"💾 Cache: {cache_stats.get('valid_entries', 0)} entries")
            else:
                st.warning("💾 Cache: Disabled")
        except:
            st.error("❌ Price manager error")

        # Portfolio data integrity
        portfolios = get_portfolios()
        if portfolios:
            st.success(f"✅ {len(portfolios)} portfolios loaded")
        else:
            st.warning("⚠️ No portfolios loaded")


def refresh_all_portfolio_prices(portfolios: List[Portfolio]):
    """Refresh prices for all portfolios."""

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

            # Batch update prices
            if all_tickers:
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
                    st.success(f"✅ Updated prices for {updated_count} assets across all portfolios")
                else:
                    st.info("ℹ️ All prices are already up to date")
            else:
                st.warning("No assets found to update")

        except Exception as e:
            st.error(f"Error updating prices: {str(e)}")


def update_selected_portfolio_prices(portfolio: Portfolio):
    """Update prices for selected portfolio."""

    from ..utils.helpers import update_portfolio_prices
    update_portfolio_prices(portfolio)


def export_all_portfolios(portfolios: List[Portfolio]):
    """Export all portfolios."""

    st.info("📤 Preparing export for all portfolios...")

    # This would implement actual export functionality
    # For now, show export options

    export_format = st.radio(
        "Export Format",
        ["CSV", "JSON", "Excel"],
        horizontal=True,
        key="bulk_export_format"
    )

    if st.button("Download Export", key="bulk_export_download"):
        # Placeholder for actual export
        st.success(f"Export prepared in {export_format} format")
        st.balloons()


def generate_dashboard_report(portfolios: List[Portfolio]):
    """Generate comprehensive dashboard report."""

    st.subheader("📋 Dashboard Report")

    # Report summary
    total_value = sum(p.calculate_value() for p in portfolios)
    total_assets = sum(len(p.assets) for p in portfolios)

    report_data = {
        "Report Generated": format_datetime(st.session_state.get('current_time', pd.Timestamp.now())),
        "Total Portfolios": len(portfolios),
        "Total Unique Assets": len(set(asset.ticker for p in portfolios for asset in p.assets)),
        "Combined Portfolio Value": format_currency(total_value),
        "Average Portfolio Size": f"{total_assets / len(portfolios):.1f} assets" if portfolios else "0 assets",
        "Most Common Asset": get_most_common_asset(portfolios),
        "Largest Portfolio": max(portfolios, key=lambda p: p.calculate_value()).name if portfolios else "N/A"
    }

    # Display report
    for key, value in report_data.items():
        st.write(f"**{key}**: {value}")

    # Export report option
    if st.button("📤 Download Report", key="dashboard_report_download"):
        st.success("📋 Dashboard report prepared for download")


def get_most_common_asset(portfolios: List[Portfolio]) -> str:
    """Get the most commonly held asset across portfolios."""

    asset_counts = {}

    for portfolio in portfolios:
        for asset in portfolio.assets:
            asset_counts[asset.ticker] = asset_counts.get(asset.ticker, 0) + 1

    if not asset_counts:
        return "None"

    most_common = max(asset_counts.items(), key=lambda x: x[1])
    return f"{most_common[0]} ({most_common[1]} portfolios)"