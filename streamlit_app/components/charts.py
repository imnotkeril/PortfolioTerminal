"""
Chart and visualization components for the Portfolio Management System.

This module contains reusable chart components for portfolio visualization.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path
import numpy as np
from plotly.subplots import make_subplots
# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import Portfolio, Asset
from ..utils.formatting import format_currency, format_percentage, format_large_number
from ..utils.helpers import get_unique_colors, safe_divide


def create_portfolio_allocation_chart(portfolio: Portfolio, chart_type: str = "pie") -> go.Figure:
    """
    Create portfolio allocation visualization.

    Args:
        portfolio: Portfolio object
        chart_type: Type of chart ('pie', 'donut', 'bar', 'treemap')

    Returns:
        Plotly figure object
    """

    if not portfolio.assets:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No assets in portfolio",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Prepare data
    tickers = [asset.ticker for asset in portfolio.assets]
    weights = [asset.weight for asset in portfolio.assets]
    names = [asset.name or asset.ticker for asset in portfolio.assets]

    # Sort by weight for better visualization
    sorted_data = sorted(zip(tickers, weights, names), key=lambda x: x[1], reverse=True)
    tickers, weights, names = zip(*sorted_data)

    # Create chart based on type
    if chart_type == "pie":
        fig = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights,
            text=[f"{name}<br>{format_percentage(weight)}" for name, weight in zip(names, weights)],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>" +
                          "Weight: %{percent}<br>" +
                          "Value: %{value:.1%}<br>" +
                          "<extra></extra>",
            hole=0
        )])

        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
        )

    elif chart_type == "donut":
        fig = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights,
            text=[f"{name}<br>{format_percentage(weight)}" for name, weight in zip(names, weights)],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>" +
                          "Weight: %{percent}<br>" +
                          "Value: %{value:.1%}<br>" +
                          "<extra></extra>",
            hole=0.4
        )])

        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            annotations=[dict(text=portfolio.name, x=0.5, y=0.5, font_size=12, showarrow=False)]
        )

    elif chart_type == "bar":
        fig = go.Figure(data=[go.Bar(
            x=tickers,
            y=[w * 100 for w in weights],  # Convert to percentage
            text=[format_percentage(w) for w in weights],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                          "Weight: %{y:.1f}%<br>" +
                          "<extra></extra>",
        )])

        fig.update_layout(
            title="Portfolio Allocation",
            xaxis_title="Assets",
            yaxis_title="Weight (%)",
            showlegend=False
        )

    elif chart_type == "treemap":
        fig = go.Figure(go.Treemap(
            labels=tickers,
            values=weights,
            parents=["Portfolio"] * len(tickers),
            text=[f"{ticker}<br>{format_percentage(weight)}" for ticker, weight in zip(tickers, weights)],
            hovertemplate="<b>%{label}</b><br>" +
                          "Weight: %{value:.1%}<br>" +
                          "<extra></extra>",
        ))

        fig.update_layout(title="Portfolio Allocation - TreeMap")

    # Apply common styling
    fig.update_layout(
        height=400,
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_sector_allocation_chart(portfolio: Portfolio) -> go.Figure:
    """
    Create sector allocation chart.

    Args:
        portfolio: Portfolio object

    Returns:
        Plotly figure object
    """

    if not portfolio.assets:
        fig = go.Figure()
        fig.add_annotation(
            text="No assets in portfolio",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    # Aggregate by sector
    sector_weights = {}
    for asset in portfolio.assets:
        sector = getattr(asset, 'sector', 'Unknown')
        sector_weights[sector] = sector_weights.get(sector, 0) + asset.weight

    # Sort by weight
    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
    sectors, weights = zip(*sorted_sectors) if sorted_sectors else ([], [])

    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=weights,
        text=[format_percentage(w) for w in weights],
        textinfo="label+percent",
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>" +
                      "Weight: %{percent}<br>" +
                      "Value: %{value:.1%}<br>" +
                      "<extra></extra>"
    )])

    fig.update_layout(
        title="Sector Allocation",
        height=400,
        showlegend=True,
        annotations=[dict(text="Sectors", x=0.5, y=0.5, font_size=12, showarrow=False)],
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_performance_chart(portfolio: Portfolio, timeframe: str = "1Y") -> go.Figure:
    """
    Create portfolio performance chart.

    Args:
        portfolio: Portfolio object
        timeframe: Time period ('1M', '3M', '6M', '1Y', '2Y', '5Y')

    Returns:
        Plotly figure object
    """

    # This is a placeholder for performance charting
    # In a real implementation, you would fetch historical data

    # Generate sample data for demonstration
    import numpy as np
    from datetime import datetime, timedelta

    # Determine date range
    end_date = datetime.now()
    if timeframe == "1M":
        start_date = end_date - timedelta(days=30)
        periods = 30
    elif timeframe == "3M":
        start_date = end_date - timedelta(days=90)
        periods = 90
    elif timeframe == "6M":
        start_date = end_date - timedelta(days=180)
        periods = 90
    elif timeframe == "1Y":
        start_date = end_date - timedelta(days=365)
        periods = 252
    elif timeframe == "2Y":
        start_date = end_date - timedelta(days=730)
        periods = 504
    else:  # 5Y
        start_date = end_date - timedelta(days=1825)
        periods = 1260

    # Generate sample performance data
    np.random.seed(42)  # For consistent demo data
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)

    # Simulate portfolio performance (random walk with slight upward trend)
    returns = np.random.normal(0.0005, 0.02, periods)  # Daily returns
    cumulative_returns = np.cumprod(1 + returns)
    portfolio_values = cumulative_returns * portfolio.calculate_value()

    # Create benchmark (S&P 500 proxy)
    benchmark_returns = np.random.normal(0.0004, 0.015, periods)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)
    benchmark_values = benchmark_cumulative * portfolio.calculate_value()

    # Create line chart
    fig = go.Figure()

    # Portfolio line
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name=portfolio.name,
        line=dict(color='#1f77b4', width=2),
        hovertemplate="<b>%{fullData.name}</b><br>" +
                      "Date: %{x}<br>" +
                      "Value: %{y:$,.0f}<br>" +
                      "<extra></extra>"
    ))

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_values,
        mode='lines',
        name='S&P 500 (Benchmark)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate="<b>%{fullData.name}</b><br>" +
                      "Date: %{x}<br>" +
                      "Value: %{y:$,.0f}<br>" +
                      "<extra></extra>"
    ))

    fig.update_layout(
        title=f"Portfolio Performance - {timeframe}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_asset_comparison_chart(portfolios: List[Portfolio]) -> go.Figure:
    """
    Create asset comparison chart across multiple portfolios.

    Args:
        portfolios: List of Portfolio objects

    Returns:
        Plotly figure object
    """

    if not portfolios:
        fig = go.Figure()
        fig.add_annotation(text="No portfolios to compare", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Collect all unique tickers
    all_tickers = set()
    for portfolio in portfolios:
        for asset in portfolio.assets:
            all_tickers.add(asset.ticker)

    all_tickers = sorted(list(all_tickers))

    # Create comparison matrix
    comparison_data = []
    for portfolio in portfolios:
        portfolio_weights = {}
        for asset in portfolio.assets:
            portfolio_weights[asset.ticker] = asset.weight

        row = [portfolio.name]
        for ticker in all_tickers:
            row.append(portfolio_weights.get(ticker, 0))

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data, columns=['Portfolio'] + all_tickers)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df[all_tickers].values,
        x=all_tickers,
        y=df['Portfolio'].tolist(),
        colorscale='Blues',
        text=[[format_percentage(val) for val in row] for row in df[all_tickers].values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b><br>" +
                      "Asset: %{x}<br>" +
                      "Weight: %{z:.1%}<br>" +
                      "<extra></extra>"
    ))

    fig.update_layout(
        title="Portfolio Asset Comparison",
        xaxis_title="Assets",
        yaxis_title="Portfolios",
        height=max(200, len(portfolios) * 50),
        font=dict(family="Arial", size=10)
    )

    return fig


def create_risk_return_scatter(portfolios: List[Portfolio]) -> go.Figure:
    """
    Create risk-return scatter plot.

    Args:
        portfolios: List of Portfolio objects

    Returns:
        Plotly figure object
    """

    if not portfolios:
        fig = go.Figure()
        fig.add_annotation(text="No portfolios to analyze", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Calculate risk and return metrics (placeholder implementation)
    portfolio_data = []
    for portfolio in portfolios:
        # Placeholder calculations - in reality, these would be based on historical data
        annual_return = np.random.uniform(0.05, 0.15)  # 5-15% annual return
        volatility = np.random.uniform(0.10, 0.25)  # 10-25% volatility
        sharpe_ratio = safe_divide(annual_return - 0.02, volatility)  # Assuming 2% risk-free rate

        portfolio_data.append({
            'name': portfolio.name,
            'return': annual_return,
            'risk': volatility,
            'sharpe': sharpe_ratio,
            'assets': len(portfolio.assets),
            'value': portfolio.calculate_value()
        })

    # Create scatter plot
    fig = go.Figure()

    for data in portfolio_data:
        fig.add_trace(go.Scatter(
            x=[data['risk']],
            y=[data['return']],
            mode='markers',
            name=data['name'],
            marker=dict(
                size=max(10, min(30, data['assets'] * 2)),  # Size based on number of assets
                opacity=0.7
            ),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Risk (Volatility): %{x:.1%}<br>" +
                          "Return: %{y:.1%}<br>" +
                          "Sharpe Ratio: " + f"{data['sharpe']:.2f}" + "<br>" +
                          "Assets: " + f"{data['assets']}" + "<br>" +
                          "Value: " + format_currency(data['value']) + "<br>" +
                          "<extra></extra>"
        ))

    fig.update_layout(
        title="Risk-Return Analysis",
        xaxis_title="Risk (Volatility)",
        yaxis_title="Expected Return",
        height=400,
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%'),
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_portfolio_metrics_chart(portfolio: Portfolio) -> go.Figure:
    """
    Create portfolio metrics radar chart.

    Args:
        portfolio: Portfolio object

    Returns:
        Plotly figure object
    """

    # Calculate metrics (placeholder implementation)
    metrics = {
        'Diversification': min(1.0, len(portfolio.assets) / 20),  # More assets = better diversification
        'Risk Level': np.random.uniform(0.3, 0.8),
        'Growth Potential': np.random.uniform(0.4, 0.9),
        'Income Generation': np.random.uniform(0.2, 0.7),
        'Liquidity': np.random.uniform(0.6, 0.95),
        'ESG Score': np.random.uniform(0.3, 0.8)
    }

    # Create radar chart
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=categories + [categories[0]],  # Close the shape
        fill='toself',
        name=portfolio.name,
        line_color='rgba(31, 119, 180, 0.8)',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate="<b>%{theta}</b><br>" +
                      "Score: %{r:.1%}<br>" +
                      "<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%'
            )
        ),
        showlegend=True,
        title="Portfolio Metrics Overview",
        height=400,
        font=dict(family="Arial", size=10)
    )

    return fig


def create_asset_weight_chart(portfolio: Portfolio, show_names: bool = True) -> go.Figure:
    """
    Create horizontal bar chart of asset weights.

    Args:
        portfolio: Portfolio object
        show_names: Whether to show company names or just tickers

    Returns:
        Plotly figure object
    """

    if not portfolio.assets:
        fig = go.Figure()
        fig.add_annotation(text="No assets in portfolio", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Sort assets by weight
    sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight)

    # Prepare data
    if show_names:
        labels = [f"{asset.ticker} - {asset.name}" if asset.name else asset.ticker for asset in sorted_assets]
    else:
        labels = [asset.ticker for asset in sorted_assets]

    weights = [asset.weight * 100 for asset in sorted_assets]  # Convert to percentage

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=weights,
        y=labels,
        orientation='h',
        text=[f"{w:.1f}%" for w in weights],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>" +
                      "Weight: %{x:.1f}%<br>" +
                      "<extra></extra>",
        marker_color='rgba(31, 119, 180, 0.8)'
    ))

    fig.update_layout(
        title="Asset Weights",
        xaxis_title="Weight (%)",
        yaxis_title="Assets",
        height=max(300, len(sorted_assets) * 25),
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_portfolio_value_breakdown(portfolio: Portfolio) -> go.Figure:
    """
    Create waterfall chart showing portfolio value breakdown.

    Args:
        portfolio: Portfolio object

    Returns:
        Plotly figure object
    """

    if not portfolio.assets:
        fig = go.Figure()
        fig.add_annotation(text="No assets in portfolio", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Calculate asset values
    total_value = portfolio.calculate_value()
    asset_values = []

    for asset in portfolio.assets:
        if hasattr(asset, 'current_price') and hasattr(asset, 'shares'):
            if asset.current_price and asset.shares:
                value = asset.current_price * asset.shares
            else:
                value = asset.weight * total_value  # Fallback to weight-based calculation
        else:
            value = asset.weight * total_value

        asset_values.append((asset.ticker, value))

    # Sort by value
    asset_values.sort(key=lambda x: x[1], reverse=True)

    # Create waterfall chart
    tickers = [av[0] for av in asset_values]
    values = [av[1] for av in asset_values]

    fig = go.Figure(go.Waterfall(
        name="Portfolio Value",
        orientation="v",
        measure=["relative"] * len(tickers) + ["total"],
        x=tickers + ["Total"],
        textposition="outside",
        text=[format_currency(v) for v in values] + [format_currency(total_value)],
        y=values + [0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="Portfolio Value Breakdown",
        showlegend=False,
        height=400,
        xaxis_title="Assets",
        yaxis_title="Value ($)",
        font=dict(family="Arial", size=10)
    )

    return fig


def create_correlation_heatmap(portfolios: List[Portfolio]) -> go.Figure:
    """
    Create correlation heatmap between portfolios.

    Args:
        portfolios: List of Portfolio objects

    Returns:
        Plotly figure object
    """

    if len(portfolios) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 portfolios for correlation analysis",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # This is a placeholder implementation
    # In reality, you would calculate correlations based on historical returns

    portfolio_names = [p.name for p in portfolios]
    n = len(portfolio_names)

    # Generate placeholder correlation matrix
    np.random.seed(42)
    correlation_matrix = np.random.uniform(0.3, 0.9, (n, n))

    # Make matrix symmetric and diagonal = 1
    for i in range(n):
        for j in range(n):
            if i == j:
                correlation_matrix[i][j] = 1.0
            else:
                correlation_matrix[i][j] = correlation_matrix[j][i] = correlation_matrix[i][j]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=portfolio_names,
        y=portfolio_names,
        colorscale='RdYlBu',
        zmid=0.5,
        text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>" +
                      "Correlation: %{z:.2f}<br>" +
                      "<extra></extra>"
    ))

    fig.update_layout(
        title="Portfolio Correlation Matrix",
        height=max(300, n * 50),
        font=dict(family="Arial", size=10)
    )

    return fig


def create_efficient_frontier_chart(portfolio: Portfolio) -> go.Figure:
    """
    Create efficient frontier chart for portfolio optimization.

    Args:
        portfolio: Portfolio object

    Returns:
        Plotly figure object
    """

    # Placeholder implementation for efficient frontier
    # In reality, this would involve complex optimization calculations

    # Generate sample efficient frontier points
    risk_levels = np.linspace(0.05, 0.30, 50)
    returns = []

    for risk in risk_levels:
        # Simple quadratic relationship for demonstration
        max_return = 0.15 - 2 * (risk - 0.12) ** 2
        returns.append(max(0.02, max_return))

    # Create scatter plot for efficient frontier
    fig = go.Figure()

    # Efficient frontier line
    fig.add_trace(go.Scatter(
        x=risk_levels,
        y=returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=3),
        hovertemplate="Risk: %{x:.1%}<br>" +
                      "Expected Return: %{y:.1%}<br>" +
                      "<extra></extra>"
    ))

    # Current portfolio point (placeholder calculation)
    current_risk = np.random.uniform(0.12, 0.20)
    current_return = np.random.uniform(0.08, 0.12)

    fig.add_trace(go.Scatter(
        x=[current_risk],
        y=[current_return],
        mode='markers',
        name=f'Current: {portfolio.name}',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate="<b>Current Portfolio</b><br>" +
                      "Risk: %{x:.1%}<br>" +
                      "Expected Return: %{y:.1%}<br>" +
                      "<extra></extra>"
    ))

    fig.update_layout(
        title="Efficient Frontier Analysis",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Expected Return",
        height=400,
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%'),
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_portfolio_summary_cards(portfolio: Portfolio) -> Dict[str, Any]:
    """
    Create summary metrics cards for portfolio dashboard with improved metrics.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with metric values and formatting
    """
    from ..utils.session_state import get_price_manager

    total_value = portfolio.calculate_value()
    total_assets = len(portfolio.assets)

    # Auto-fetch company info for assets without sectors
    price_manager = get_price_manager()
    assets_to_update = [asset for asset in portfolio.assets if not asset.sector or not asset.name]

    if assets_to_update:
        for asset in assets_to_update:
            try:
                company_info = price_manager.get_company_info(asset.ticker)
                if company_info:
                    if not asset.name:
                        asset.name = company_info.name
                    if not asset.sector:
                        asset.sector = company_info.sector or "Unknown"
            except Exception:
                # Set defaults if fetch fails
                if not asset.name:
                    asset.name = f"{asset.ticker} Corp"
                if not asset.sector:
                    asset.sector = "Unknown"

    # Calculate remaining cash instead of daily return
    invested_value = 0.0
    for asset in portfolio.assets:
        if asset.current_price and asset.shares:
            invested_value += asset.current_price * asset.shares
        elif asset.weight and portfolio.initial_value:
            # Fallback to weight-based calculation
            invested_value += asset.weight * portfolio.initial_value

    remaining_cash = max(0, portfolio.initial_value - invested_value)

    # Portfolio concentration (weight of top 3 holdings)
    sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
    top_3_concentration = sum(asset.weight for asset in sorted_assets[:3])

    # Placeholder performance metrics for other cards
    daily_change = np.random.uniform(-0.03, 0.03)
    daily_change_value = total_value * daily_change

    return {
        'total_value': {
            'value': format_currency(total_value),
            'delta': format_currency(daily_change_value),
            'delta_color': 'normal' if daily_change >= 0 else 'inverse'
        },
        'total_assets': {
            'value': str(total_assets),
            'delta': None,
            'delta_color': 'normal'
        },
        'concentration': {
            'value': format_percentage(top_3_concentration),
            'delta': None,
            'delta_color': 'normal'
        },
        'avg_position_size': {
            'value': format_currency(total_value / total_assets) if total_assets > 0 else '$0',
            'delta': None,
            'delta_color': 'normal'
        },
        'remaining_cash': {  # Changed from daily_return to remaining_cash
            'value': format_currency(remaining_cash),
            'delta': None,
            'delta_color': 'normal'
        }
    }


def render_chart_controls() -> Dict[str, Any]:
    """
    Render chart control widgets and return selected options.

    Returns:
        Dictionary with selected chart options
    """

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Pie", "Donut", "Bar", "Treemap"],
            key="allocation_chart_type"
        )

    with col2:
        show_values = st.checkbox(
            "Show Values",
            value=True,
            key="show_chart_values"
        )

    with col3:
        show_legend = st.checkbox(
            "Show Legend",
            value=True,
            key="show_chart_legend"
        )

    with col4:
        color_scheme = st.selectbox(
            "Colors",
            ["Default", "Pastel", "Bold", "Monochrome"],
            key="chart_colors"
        )

    return {
        'chart_type': chart_type.lower(),
        'show_values': show_values,
        'show_legend': show_legend,
        'color_scheme': color_scheme.lower()
    }


# Replace the incomplete implementation with this complete one:

def create_portfolio_comparison_chart(portfolios: List[Portfolio]) -> go.Figure:
    """
    Create portfolio comparison chart showing values and allocation.

    Args:
        portfolios: List of Portfolio objects to compare

    Returns:
        Plotly figure object showing portfolio comparison
    """

    if not portfolios:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No portfolios selected for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Prepare data for comparison
    portfolio_names = [p.name for p in portfolios]
    portfolio_values = [p.calculate_value() for p in portfolios]
    asset_counts = [len(p.assets) for p in portfolios]

    # Calculate additional metrics for comparison
    concentration_risks = []
    for portfolio in portfolios:
        if portfolio.assets:
            sorted_assets = sorted(portfolio.assets, key=lambda x: x.weight, reverse=True)
            top_3_concentration = sum(asset.weight for asset in sorted_assets[:3])
            concentration_risks.append(top_3_concentration)
        else:
            concentration_risks.append(0)

    # Create subplot with secondary y-axis
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Values', 'Asset Count Comparison',
                        'Top 3 Concentration Risk', 'Asset Allocation Overview'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Portfolio Values Bar Chart
    fig.add_trace(
        go.Bar(
            x=portfolio_names,
            y=portfolio_values,
            name='Portfolio Value',
            text=[format_currency(val) for val in portfolio_values],
            textposition='auto',
            marker_color='#1f77b4',
            hovertemplate="<b>%{x}</b><br>" +
                          "Value: %{text}<br>" +
                          "<extra></extra>"
        ),
        row=1, col=1
    )

    # 2. Asset Count Comparison
    fig.add_trace(
        go.Bar(
            x=portfolio_names,
            y=asset_counts,
            name='Asset Count',
            text=asset_counts,
            textposition='auto',
            marker_color='#ff7f0e',
            hovertemplate="<b>%{x}</b><br>" +
                          "Assets: %{y}<br>" +
                          "<extra></extra>"
        ),
        row=1, col=2
    )

    # 3. Concentration Risk Comparison
    fig.add_trace(
        go.Bar(
            x=portfolio_names,
            y=[risk * 100 for risk in concentration_risks],
            name='Top 3 Concentration %',
            text=[f"{risk:.1%}" for risk in concentration_risks],
            textposition='auto',
            marker_color='#2ca02c',
            hovertemplate="<b>%{x}</b><br>" +
                          "Top 3 Concentration: %{text}<br>" +
                          "<extra></extra>"
        ),
        row=2, col=1
    )

    # 4. Combined Asset Allocation Pie Chart
    # Aggregate all assets across selected portfolios
    combined_assets = {}
    total_combined_value = sum(portfolio_values)

    for portfolio in portfolios:
        portfolio_value = portfolio.calculate_value()
        for asset in portfolio.assets:
            if asset.ticker not in combined_assets:
                combined_assets[asset.ticker] = 0
            # Weight by portfolio value in the comparison
            combined_assets[asset.ticker] += asset.weight * (
                        portfolio_value / total_combined_value) if total_combined_value > 0 else 0

    # Sort by weight and take top 10 for readability
    sorted_combined = sorted(combined_assets.items(), key=lambda x: x[1], reverse=True)[:10]

    if sorted_combined:
        asset_tickers, asset_weights = zip(*sorted_combined)

        fig.add_trace(
            go.Pie(
                labels=asset_tickers,
                values=asset_weights,
                name="Combined Allocation",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>" +
                              "Combined Weight: %{percent}<br>" +
                              "<extra></extra>",
                hole=0.4  # Make it a donut chart
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        title=f"Portfolio Comparison Dashboard ({len(portfolios)} portfolios)",
        height=700,
        showlegend=False,
        font=dict(family="Arial", size=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Assets", row=1, col=2)
    fig.update_yaxes(title_text="Concentration (%)", row=2, col=1)

    # Update x-axis labels
    fig.update_xaxes(title_text="Portfolios", row=1, col=1)
    fig.update_xaxes(title_text="Portfolios", row=1, col=2)
    fig.update_xaxes(title_text="Portfolios", row=2, col=1)

    return fig

def apply_chart_styling(fig: go.Figure, theme: str = "default") -> go.Figure:
    """
    Apply consistent styling to charts.

    Args:
        fig: Plotly figure object
        theme: Theme name ("default", "dark", "minimal")

    Returns:
        Styled figure object
    """

    if theme == "dark":
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
    elif theme == "minimal":
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
            showlegend=False
        )

    # Common styling
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='closest'
    )

    return fig