"""
Chart Generator Module
Creates advanced financial charts and visualizations for portfolio analysis.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class ChartGenerator:
    """
    Advanced chart generation for financial analytics.

    Creates professional-grade visualizations for portfolio analysis.
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize chart generator.

        Args:
            theme: Plotly theme to use for charts
        """
        self.theme = theme
        self.color_palette = {
            'primary': '#3b82f6',
            'secondary': '#6b7280',
            'success': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'info': '#06b6d4'
        }

    def create_performance_comparison_chart(
        self,
        returns_data: Dict[str, pd.Series],
        title: str = "Portfolio Performance Comparison"
    ) -> go.Figure:
        """Create multi-portfolio performance comparison chart."""

        fig = go.Figure()

        colors = [self.color_palette['primary'], self.color_palette['secondary'],
                 self.color_palette['success'], self.color_palette['danger']]

        for i, (name, returns) in enumerate(returns_data.items()):
            cumulative_returns = (1 + returns).cumprod()

            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name=name,
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                ),
                hovertemplate=f'<b>{name}</b><br>' +
                              'Date: %{x}<br>' +
                              'Value: %{y:.4f}<br>' +
                              '<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            showlegend=True,
            template=self.theme,
            height=500
        )

        return fig

    def create_efficient_frontier_chart(
        self,
        frontier_points: np.ndarray,
        current_portfolio: Optional[Tuple[float, float]] = None,
        optimal_portfolio: Optional[Tuple[float, float]] = None
    ) -> go.Figure:
        """Create efficient frontier visualization."""

        fig = go.Figure()

        # Efficient frontier line
        fig.add_trace(go.Scatter(
            x=frontier_points[:, 1],  # Risk (volatility)
            y=frontier_points[:, 0],  # Return
            mode='lines',
            name='Efficient Frontier',
            line=dict(color=self.color_palette['primary'], width=3),
            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))

        # Current portfolio point
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio[1]],
                y=[current_portfolio[0]],
                mode='markers',
                name='Current Portfolio',
                marker=dict(
                    size=15,
                    color=self.color_palette['warning'],
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Current Portfolio</b><br>' +
                              'Risk: %{x:.2%}<br>' +
                              'Return: %{y:.2%}<br>' +
                              '<extra></extra>'
            ))

        # Optimal portfolio point
        if optimal_portfolio:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio[1]],
                y=[optimal_portfolio[0]],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color=self.color_palette['success'],
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Optimal Portfolio</b><br>' +
                              'Risk: %{x:.2%}<br>' +
                              'Return: %{y:.2%}<br>' +
                              '<extra></extra>'
            ))

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            template=self.theme,
            height=500,
            showlegend=True
        )

        # Format axes as percentages
        fig.update_xaxis(tickformat='.1%')
        fig.update_yaxis(tickformat='.1%')

        return fig

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix"
    ) -> go.Figure:
        """Create correlation heatmap."""

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"]
            )
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            width=500
        )

        return fig

    def create_drawdown_underwater_chart(
        self,
        returns: pd.Series,
        title: str = "Underwater Chart (Drawdowns)"
    ) -> go.Figure:
        """Create underwater chart showing drawdown periods."""

        # Calculate cumulative returns and drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig = go.Figure()

        # Drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color=self.color_palette['danger'], width=1),
            fillcolor=f"rgba(239, 68, 68, 0.3)",
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>'
        ))

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            annotation_text="Break-even"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template=self.theme,
            height=300,
            showlegend=False
        )

        fig.update_yaxis(tickformat='.1%')

        return fig

    def create_rolling_metrics_chart(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: List[str] = ['return', 'volatility', 'sharpe']
    ) -> go.Figure:
        """Create rolling metrics chart."""

        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=[f"Rolling {window}D {metric.title()}" for metric in metrics],
            vertical_spacing=0.08
        )

        colors = [self.color_palette['primary'], self.color_palette['success'], self.color_palette['warning']]

        for i, metric in enumerate(metrics):
            if metric == 'return':
                rolling_values = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
                y_title = "Return"
                y_format = '.1%'
            elif metric == 'volatility':
                rolling_values = returns.rolling(window).std() * np.sqrt(252)
                y_title = "Volatility"
                y_format = '.1%'
            elif metric == 'sharpe':
                rolling_returns = returns.rolling(window).mean() * 252
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                rolling_values = (rolling_returns - 0.02) / rolling_vol  # Assuming 2% risk-free rate
                y_title = "Sharpe Ratio"
                y_format = '.2f'

            fig.add_trace(
                go.Scatter(
                    x=rolling_values.index,
                    y=rolling_values.values,
                    mode='lines',
                    name=f"Rolling {metric.title()}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'Date: %{{x}}<br>{metric.title()}: %{{y}}<extra></extra>'
                ),
                row=i+1, col=1
            )

            fig.update_yaxis(title_text=y_title, tickformat=y_format, row=i+1, col=1)

        fig.update_layout(
            title="Rolling Performance Metrics",
            template=self.theme,
            height=200 * len(metrics),
            showlegend=False
        )

        return fig

    def create_return_distribution_chart(
        self,
        returns: pd.Series,
        title: str = "Return Distribution Analysis"
    ) -> go.Figure:
        """Create return distribution with normal overlay."""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histogram', 'Q-Q Plot', 'Box Plot', 'Time Series'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Histogram with normal distribution overlay
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                histnorm='probability density'
            ),
            row=1, col=1
        )

        # Normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x_normal = np.linspace(returns.min(), returns.max(), 100)
        y_normal = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x_normal-mu)/sigma)**2)

        fig.add_trace(
            go.Scatter(
                x=x_normal,
                y=y_normal,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=self.color_palette['danger'], width=2)
            ),
            row=1, col=1
        )

        # Q-Q Plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=self.color_palette['success'], size=4)
            ),
            row=1, col=2
        )

        # Perfect normal line
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[sample_quantiles.min(), sample_quantiles.max()],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=returns,
                name='Returns',
                marker_color=self.color_palette['warning']
            ),
            row=2, col=1
        )

        # Time series
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                mode='lines',
                name='Returns Over Time',
                line=dict(color=self.color_palette['info'], width=1)
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            showlegend=False
        )

        return fig

    def create_risk_decomposition_chart(
        self,
        risk_metrics: Dict[str, float],
        title: str = "Risk Decomposition"
    ) -> go.Figure:
        """Create risk metrics radar chart."""

        # Normalize metrics for radar chart (0-100 scale)
        normalized_metrics = {}
        for key, value in risk_metrics.items():
            if 'ratio' in key.lower():
                # For ratios, higher is better, scale 0-5 to 0-100
                normalized_metrics[key] = min(max(value * 20, 0), 100)
            elif any(word in key.lower() for word in ['volatility', 'var', 'drawdown']):
                # For risk metrics, lower is better, invert scale
                normalized_metrics[key] = max(100 - abs(value) * 1000, 0)
            else:
                # For other metrics, scale appropriately
                normalized_metrics[key] = min(max(abs(value) * 100, 0), 100)

        categories = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Profile',
            fillcolor=f"rgba(59, 130, 246, 0.3)",
            line=dict(color=self.color_palette['primary'], width=2)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=title,
            template=self.theme,
            height=500
        )

        return fig

    def create_sector_allocation_treemap(
        self,
        allocation_data: Dict[str, Dict[str, float]],
        title: str = "Portfolio Allocation Treemap"
    ) -> go.Figure:
        """Create treemap visualization of portfolio allocation."""

        # Flatten the hierarchical data
        labels = []
        parents = []
        values = []
        colors = []

        # Add sectors
        color_map = px.colors.qualitative.Set3
        for i, (sector, assets) in enumerate(allocation_data.items()):
            labels.append(sector)
            parents.append("")
            values.append(sum(assets.values()))
            colors.append(color_map[i % len(color_map)])

            # Add assets within sector
            for asset, weight in assets.items():
                labels.append(asset)
                parents.append(sector)
                values.append(weight)
                colors.append(color_map[i % len(color_map)])

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Allocation: %{value:.1%}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            height=500
        )

        return fig

    def create_monte_carlo_simulation_chart(
        self,
        simulation_results: np.ndarray,
        percentiles: List[int] = [5, 25, 50, 75, 95],
        title: str = "Monte Carlo Portfolio Simulation"
    ) -> go.Figure:
        """Create Monte Carlo simulation visualization."""

        fig = go.Figure()

        # Calculate percentiles
        percentile_values = {}
        for p in percentiles:
            percentile_values[p] = np.percentile(simulation_results, p, axis=0)

        time_steps = range(simulation_results.shape[1])

        # Add percentile bands
        colors = ['rgba(239,68,68,0.1)', 'rgba(245,158,11,0.2)',
                 'rgba(59,130,246,0.8)', 'rgba(245,158,11,0.2)', 'rgba(239,68,68,0.1)']

        for i, p in enumerate(percentiles):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=percentile_values[p],
                mode='lines',
                name=f'{p}th Percentile',
                line=dict(color=colors[i % len(colors)], width=2 if p == 50 else 1),
                fill='tonexty' if i > 0 else None
            ))

        # Add some individual simulation paths
        n_paths_to_show = min(50, simulation_results.shape[0])
        for i in range(0, n_paths_to_show, 10):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=simulation_results[i],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.1)', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time Steps",
            yaxis_title="Portfolio Value",
            template=self.theme,
            height=500
        )

        return fig

    def create_factor_exposure_chart(
        self,
        factor_loadings: Dict[str, float],
        title: str = "Factor Exposure Analysis"
    ) -> go.Figure:
        """Create factor exposure bar chart."""

        factors = list(factor_loadings.keys())
        loadings = list(factor_loadings.values())

        # Color bars based on positive/negative exposure
        colors = [self.color_palette['success'] if x >= 0 else self.color_palette['danger'] for x in loadings]

        fig = go.Figure(data=[
            go.Bar(
                x=factors,
                y=loadings,
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Loading: %{y:.3f}<extra></extra>'
            )
        ])

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")

        fig.update_layout(
            title=title,
            xaxis_title="Factors",
            yaxis_title="Loading",
            template=self.theme,
            height=400
        )

        return fig

    def create_performance_attribution_chart(
        self,
        attribution_data: Dict[str, Dict[str, float]],
        title: str = "Performance Attribution"
    ) -> go.Figure:
        """Create performance attribution waterfall chart."""

        categories = []
        values = []

        # Starting value
        categories.append("Starting Portfolio")
        values.append(0)

        # Attribution components
        cumulative = 0
        for component, breakdown in attribution_data.items():
            for subcomponent, value in breakdown.items():
                categories.append(f"{component}: {subcomponent}")
                values.append(value)
                cumulative += value

        # Ending value
        categories.append("Total Return")
        values.append(cumulative)

        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(values) - 2) + ["total"],
            x=categories,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.color_palette['success']}},
            decreasing={"marker": {"color": self.color_palette['danger']}},
            totals={"marker": {"color": self.color_palette['primary']}}
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            height=500
        )

        return fig

    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save chart to file."""
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        elif format == 'svg':
            fig.write_image(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_dashboard_summary(
        self,
        portfolio_data: Dict,
        returns: pd.Series,
        metrics: Dict[str, float]
    ) -> go.Figure:
        """Create comprehensive dashboard summary chart."""

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Performance', 'Drawdown', 'Allocation',
                           'Risk Metrics', 'Return Distribution', 'Rolling Sharpe'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Performance chart
        cumulative = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative.index, y=cumulative.values, name="Performance"),
            row=1, col=1
        )

        # 2. Drawdown chart
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, fill='tonexty', name="Drawdown"),
            row=1, col=2
        )

        # 3. Allocation pie chart
        if 'allocation' in portfolio_data:
            fig.add_trace(
                go.Pie(labels=list(portfolio_data['allocation'].keys()),
                       values=list(portfolio_data['allocation'].values())),
                row=1, col=3
            )

        # 4. Risk metrics bar chart
        risk_metrics = {k: v for k, v in metrics.items() if any(word in k for word in ['var', 'volatility', 'drawdown'])}
        fig.add_trace(
            go.Bar(x=list(risk_metrics.keys()), y=list(risk_metrics.values())),
            row=2, col=1
        )

        # 5. Return distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=30, name="Returns"),
            row=2, col=2
        )

        # 6. Rolling Sharpe ratio
        if len(returns) > 252:
            rolling_sharpe = (returns.rolling(252).mean() * 252 - 0.02) / (returns.rolling(252).std() * np.sqrt(252))
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Rolling Sharpe"),
                row=2, col=3
            )

        fig.update_layout(
            title="Portfolio Dashboard Summary",
            template=self.theme,
            height=800,
            showlegend=False
        )

        return fig