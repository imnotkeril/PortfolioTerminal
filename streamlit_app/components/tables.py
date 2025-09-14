"""
Table components for the Portfolio Management System.

This module contains reusable table components for displaying portfolio data.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np

from core.data_manager import Portfolio, Asset
from ..utils.formatting import (
    format_currency, format_percentage, format_number,
    format_datetime, format_dataframe_for_display
)
from ..utils.helpers import safe_divide, create_asset_table


def render_portfolio_overview_table(portfolios: List[Portfolio]) -> None:
    """
    Render overview table of all portfolios.

    Args:
        portfolios: List of Portfolio objects
    """

    if not portfolios:
        st.info("No portfolios created yet")
        return

    # Prepare data for table
    portfolio_data = []
    for portfolio in portfolios:
        total_value = portfolio.calculate_value()

        # Calculate additional metrics
        total_assets = len(portfolio.assets)
        largest_position = max(portfolio.assets, key=lambda x: x.weight) if portfolio.assets else None

        row = {
            'Name': portfolio.name,
            'Type': portfolio.portfolio_type.value.title(),
            'Assets': total_assets,
            'Total Value': format_currency(total_value),
            'Largest Position': f"{largest_position.ticker} ({format_percentage(largest_position.weight)})" if largest_position else "N/A",
            'Created': format_datetime(portfolio.created_date, "%Y-%m-%d"),
            'Last Modified': format_datetime(portfolio.last_modified, "%Y-%m-%d"),
            'Description': portfolio.description[:50] + "..." if portfolio.description and len(portfolio.description) > 50 else portfolio.description or "N/A"
        }
        portfolio_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(portfolio_data)

    # Display with custom configuration
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Portfolio Name", width="medium"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Assets": st.column_config.NumberColumn("Assets", width="small"),
            "Total Value": st.column_config.TextColumn("Total Value", width="medium"),
            "Largest Position": st.column_config.TextColumn("Largest Position", width="medium"),
            "Created": st.column_config.TextColumn("Created", width="small"),
            "Last Modified": st.column_config.TextColumn("Last Modified", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large")
        }
    )


def render_portfolio_assets_table(portfolio: Portfolio, editable: bool = False) -> Optional[pd.DataFrame]:
    """
    Render detailed assets table for a portfolio.

    Args:
        portfolio: Portfolio object
        editable: Whether the table should be editable

    Returns:
        Modified DataFrame if editable, None otherwise
    """

    if not portfolio.assets:
        st.info("No assets in this portfolio")
        return None

    # Create detailed asset data
    asset_data = []
    total_value = portfolio.calculate_value()

    for i, asset in enumerate(portfolio.assets):
        # Calculate current market value
        if hasattr(asset, 'current_price') and hasattr(asset, 'shares'):
            if asset.current_price and asset.shares:
                market_value = asset.current_price * asset.shares
            else:
                market_value = asset.weight * total_value
        else:
            market_value = asset.weight * total_value

        # Calculate gain/loss if purchase price available
        gain_loss = 0
        gain_loss_pct = 0
        if hasattr(asset, 'purchase_price') and hasattr(asset, 'current_price'):
            if asset.purchase_price and asset.current_price:
                gain_loss = (asset.current_price - asset.purchase_price) * getattr(asset, 'shares', 0)
                gain_loss_pct = safe_divide(asset.current_price - asset.purchase_price, asset.purchase_price)

        row = {
            'Ticker': asset.ticker,
            'Name': asset.name or 'N/A',
            'Sector': getattr(asset, 'sector', 'N/A'),
            'Asset Class': asset.asset_class.value if hasattr(asset, 'asset_class') else 'stock',
            'Weight (%)': asset.weight * 100,
            'Shares': getattr(asset, 'shares', 0),
            'Current Price': getattr(asset, 'current_price', None),
            'Market Value': market_value,
            'Purchase Price': getattr(asset, 'purchase_price', None),
            'Gain/Loss ($)': gain_loss,
            'Gain/Loss (%)': gain_loss_pct,
            'Last Updated': getattr(asset, 'last_updated', None)
        }
        asset_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(asset_data)

    # Configure columns
    column_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Name": st.column_config.TextColumn("Company Name", width="large"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Asset Class": st.column_config.SelectboxColumn(
            "Asset Class",
            options=["stock", "bond", "etf", "crypto", "commodity"],
            width="small"
        ) if editable else st.column_config.TextColumn("Asset Class", width="small"),
        "Weight (%)": st.column_config.NumberColumn(
            "Weight (%)",
            min_value=0,
            max_value=100,
            step=0.1,
            format="%.1f%%",
            width="small"
        ),
        "Shares": st.column_config.NumberColumn(
            "Shares",
            min_value=0,
            step=1,
            format="%d",
            width="small"
        ),
        "Current Price": st.column_config.NumberColumn(
            "Current Price",
            format="$%.2f",
            width="small"
        ),
        "Market Value": st.column_config.NumberColumn(
            "Market Value",
            format="$%.2f",
            width="medium"
        ),
        "Purchase Price": st.column_config.NumberColumn(
            "Purchase Price",
            format="$%.2f",
            width="small"
        ),
        "Gain/Loss ($)": st.column_config.NumberColumn(
            "Gain/Loss ($)",
            format="$%.2f",
            width="small"
        ),
        "Gain/Loss (%)": st.column_config.NumberColumn(
            "Gain/Loss (%)",
            format="%.2f%%",
            width="small"
        ),
        "Last Updated": st.column_config.DatetimeColumn(
            "Last Updated",
            format="MM/DD/YY",
            width="small"
        )
    }

    if editable:
        # Editable data editor
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config=column_config,
            key=f"assets_table_{portfolio.id}"
        )
        return edited_df
    else:
        # Read-only dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        return None


def render_portfolio_comparison_table(portfolios: List[Portfolio]) -> None:
    """
    Render side-by-side comparison table of portfolios.

    Args:
        portfolios: List of Portfolio objects to compare
    """

    if not portfolios:
        st.info("No portfolios to compare")
        return

    if len(portfolios) < 2:
        st.warning("Select at least 2 portfolios for comparison")
        return

    # Collect all unique metrics for comparison
    comparison_data = []

    # Basic portfolio metrics
    metrics = [
        ('Portfolio Name', lambda p: p.name),
        ('Type', lambda p: p.portfolio_type.value.title()),
        ('Total Assets', lambda p: len(p.assets)),
        ('Total Value', lambda p: format_currency(p.calculate_value())),
        ('Creation Date', lambda p: format_datetime(p.created_date, "%Y-%m-%d")),
        ('Last Modified', lambda p: format_datetime(p.last_modified, "%Y-%m-%d")),
    ]

    # Asset-specific metrics
    for portfolio in portfolios:
        if portfolio.assets:
            # Largest position
            largest_asset = max(portfolio.assets, key=lambda x: x.weight)
            metrics.append((f'Largest Position ({portfolio.name[:10]})',
                          lambda p, la=largest_asset: f"{la.ticker} ({format_percentage(la.weight)})"))

            # Top 3 concentration
            sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
            top_3_weight = sum(asset.weight for asset in sorted_assets[:3])
            metrics.append((f'Top 3 Concentration ({portfolio.name[:10]})',
                          lambda p, w=top_3_weight: format_percentage(w)))

    # Build comparison table
    comparison_rows = []
    for metric_name, metric_func in metrics[:6]:  # Basic metrics only for cleaner display
        row = {'Metric': metric_name}
        for portfolio in portfolios:
            try:
                row[portfolio.name] = metric_func(portfolio)
            except:
                row[portfolio.name] = 'N/A'
        comparison_rows.append(row)

    # Additional calculated metrics
    additional_metrics = []
    for portfolio in portfolios:
        total_value = portfolio.calculate_value()
        asset_count = len(portfolio.assets)
        avg_position_size = safe_divide(total_value, asset_count)

        # Portfolio concentration (Herfindahl index approximation)
        concentration_score = sum(asset.weight ** 2 for asset in portfolio.assets)

        additional_metrics.append({
            'Portfolio': portfolio.name,
            'Avg Position Size': format_currency(avg_position_size),
            'Concentration Score': f"{concentration_score:.3f}",
            'Diversification Level': 'High' if concentration_score < 0.1 else 'Medium' if concentration_score < 0.25 else 'Low'
        })

    # Display basic comparison
    st.subheader("📊 Basic Portfolio Comparison")
    basic_df = pd.DataFrame(comparison_rows)
    st.dataframe(basic_df, use_container_width=True, hide_index=True)

    # Display advanced metrics
    st.subheader("📈 Advanced Metrics")
    advanced_df = pd.DataFrame(additional_metrics)
    st.dataframe(advanced_df, use_container_width=True, hide_index=True)


def render_sector_allocation_table(portfolio: Portfolio) -> None:
    """
    Render sector allocation breakdown table.

    Args:
        portfolio: Portfolio object
    """

    if not portfolio.assets:
        st.info("No assets in portfolio")
        return

    # Aggregate by sector
    sector_data = {}
    total_value = portfolio.calculate_value()

    for asset in portfolio.assets:
        sector = getattr(asset, 'sector', 'Unknown')

        if sector not in sector_data:
            sector_data[sector] = {
                'assets': [],
                'total_weight': 0,
                'total_value': 0,
                'asset_count': 0
            }

        sector_data[sector]['assets'].append(asset)
        sector_data[sector]['total_weight'] += asset.weight
        sector_data[sector]['total_value'] += asset.weight * total_value
        sector_data[sector]['asset_count'] += 1

    # Create sector summary table
    sector_rows = []
    for sector, data in sorted(sector_data.items(), key=lambda x: x[1]['total_weight'], reverse=True):
        row = {
            'Sector': sector,
            'Assets': data['asset_count'],
            'Weight (%)': data['total_weight'] * 100,
            'Value': data['total_value'],
            'Top Holdings': ', '.join([asset.ticker for asset in sorted(data['assets'], key=lambda x: x.weight, reverse=True)[:3]])
        }
        sector_rows.append(row)

    df = pd.DataFrame(sector_rows)

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Assets": st.column_config.NumberColumn("# Assets", width="small"),
            "Weight (%)": st.column_config.NumberColumn(
                "Weight (%)",
                format="%.1f%%",
                width="small"
            ),
            "Value": st.column_config.NumberColumn(
                "Value ($)",
                format="$%.0f",
                width="medium"
            ),
            "Top Holdings": st.column_config.TextColumn("Top Holdings", width="large")
        }
    )


def render_performance_summary_table(portfolio: Portfolio) -> None:
    """
    Render performance metrics summary table.

    Args:
        portfolio: Portfolio object
    """

    # Placeholder performance calculations
    # In real implementation, these would be based on historical data

    import numpy as np

    performance_data = [
        {
            'Period': '1 Day',
            'Return': format_percentage(np.random.uniform(-0.03, 0.03)),
            'Value Change': format_currency(np.random.uniform(-5000, 5000)),
            'Benchmark Return': format_percentage(np.random.uniform(-0.025, 0.025)),
            'Excess Return': format_percentage(np.random.uniform(-0.01, 0.01))
        },
        {
            'Period': '1 Week',
            'Return': format_percentage(np.random.uniform(-0.08, 0.08)),
            'Value Change': format_currency(np.random.uniform(-15000, 15000)),
            'Benchmark Return': format_percentage(np.random.uniform(-0.06, 0.06)),
            'Excess Return': format_percentage(np.random.uniform(-0.03, 0.03))
        },
        {
            'Period': '1 Month',
            'Return': format_percentage(np.random.uniform(-0.15, 0.15)),
            'Value Change': format_currency(np.random.uniform(-30000, 30000)),
            'Benchmark Return': format_percentage(np.random.uniform(-0.12, 0.12)),
            'Excess Return': format_percentage(np.random.uniform(-0.05, 0.05))
        },
        {
            'Period': '3 Months',
            'Return': format_percentage(np.random.uniform(-0.25, 0.25)),
            'Value Change': format_currency(np.random.uniform(-50000, 50000)),
            'Benchmark Return': format_percentage(np.random.uniform(-0.20, 0.20)),
            'Excess Return': format_percentage(np.random.uniform(-0.08, 0.08))
        },
        {
            'Period': '1 Year',
            'Return': format_percentage(np.random.uniform(-0.30, 0.30)),
            'Value Change': format_currency(np.random.uniform(-80000, 80000)),
            'Benchmark Return': format_percentage(np.random.uniform(-0.25, 0.25)),
            'Excess Return': format_percentage(np.random.uniform(-0.10, 0.10))
        }
    ]

    df = pd.DataFrame(performance_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Period": st.column_config.TextColumn("Time Period", width="small"),
            "Return": st.column_config.TextColumn("Portfolio Return", width="medium"),
            "Value Change": st.column_config.TextColumn("Value Change ($)", width="medium"),
            "Benchmark Return": st.column_config.TextColumn("Benchmark Return", width="medium"),
            "Excess Return": st.column_config.TextColumn("Excess Return", width="medium")
        }
    )


def render_risk_metrics_table(portfolio: Portfolio) -> None:
    """
    Render risk metrics table.

    Args:
        portfolio: Portfolio object
    """

    # Placeholder risk calculations
    import numpy as np

    risk_metrics = [
        {
            'Metric': 'Portfolio Beta',
            'Value': f"{np.random.uniform(0.7, 1.3):.2f}",
            'Description': 'Sensitivity to market movements'
        },
        {
            'Metric': 'Value at Risk (95%)',
            'Value': format_percentage(np.random.uniform(0.05, 0.15)),
            'Description': 'Maximum expected loss over 1 day'
        },
        {
            'Metric': 'Sharpe Ratio',
            'Value': f"{np.random.uniform(0.8, 2.0):.2f}",
            'Description': 'Risk-adjusted return measure'
        }
    ]

    df = pd.DataFrame(risk_metrics)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Risk Metric", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large")
        }
    )