"""
Portfolio Analysis Page
Comprehensive portfolio analysis with advanced metrics and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

try:
    from core.analytics_engine.performance_calculator import PerformanceCalculator
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

try:
    from streamlit_app.utils.session_state import (
        get_portfolio_manager,
        get_price_manager,
        initialize_session_state
    )
    SESSION_UTILS_AVAILABLE = True
except ImportError:
    SESSION_UTILS_AVAILABLE = False

try:
    from streamlit_app.utils.formatting import (
        format_currency,
        format_percentage,
        format_number
    )
    FORMATTING_AVAILABLE = True
except ImportError:
    FORMATTING_AVAILABLE = False

    # Fallback formatting functions
    def format_currency(value):
        try:
            return f"${value:,.2f}" if value is not None else "N/A"
        except:
            return "N/A"

    def format_percentage(value):
        try:
            return f"{value*100:.2f}%" if value is not None else "N/A"
        except:
            return "N/A"

    def format_number(value, precision=2):
        try:
            return f"{value:.{precision}f}" if value is not None else "N/A"
        except:
            return "N/A"

# Page configuration
st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with fallbacks
def safe_initialize_session_state():
    """Initialize session state with error handling"""
    try:
        if SESSION_UTILS_AVAILABLE:
            initialize_session_state()
        else:
            # Basic session state setup
            if "mock_portfolios" not in st.session_state:
                st.session_state.mock_portfolios = []
    except Exception as e:
        st.error(f"Error initializing session state: {e}")

def safe_get_portfolio_manager():
    """Get portfolio manager with fallback"""
    try:
        if SESSION_UTILS_AVAILABLE:
            return get_portfolio_manager()
        else:
            return MockPortfolioManager()
    except Exception as e:
        st.error(f"Error getting portfolio manager: {e}")
        return MockPortfolioManager()

def safe_get_price_manager():
    """Get price manager with fallback"""
    try:
        if SESSION_UTILS_AVAILABLE:
            return get_price_manager()
        else:
            return MockPriceManager()
    except Exception as e:
        st.error(f"Error getting price manager: {e}")
        return MockPriceManager()

# Mock classes for fallbacks
class MockPortfolioManager:
    def list_portfolios(self):
        return [MockPortfolio("Sample Portfolio")]

class MockPortfolio:
    def __init__(self, name):
        self.name = name
        self.assets = []
        self.total_value = 100000
        self.initial_value = 100000
        self.created_date = datetime.now()

class MockPriceManager:
    def get_historical_prices(self, tickers, start_date, end_date):
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = {}
        for ticker in tickers:
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            data[ticker] = prices
        return pd.DataFrame(data, index=dates)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }

    .metric-positive {
        color: #10b981;
        font-weight: bold;
    }

    .metric-negative {
        color: #ef4444;
        font-weight: bold;
    }

    .metric-neutral {
        color: #6b7280;
        font-weight: bold;
    }

    .tab-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

def format_metric_change(value: float, is_positive_good: bool = True) -> str:
    """Format metric value with appropriate color."""
    if value == 0:
        return f'<span class="metric-neutral">{format_percentage(value)}</span>'
    elif (value > 0 and is_positive_good) or (value < 0 and not is_positive_good):
        return f'<span class="metric-positive">{format_percentage(value)}</span>'
    else:
        return f'<span class="metric-negative">{format_percentage(value)}</span>'

def create_performance_chart(returns_data: pd.DataFrame) -> go.Figure:
    """Create cumulative performance chart."""
    fig = go.Figure()

    # Portfolio performance
    cumulative_returns = (1 + returns_data['portfolio']).cumprod() * 100
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Value: %{y:.2f}<br>' +
                      '<extra></extra>'
    ))

    # Benchmark if available
    if 'benchmark' in returns_data.columns:
        benchmark_cumulative = (1 + returns_data['benchmark']).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#6b7280', width=2, dash='dash'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Date: %{x}<br>' +
                          'Value: %{y:.2f}<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title="Cumulative Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        showlegend=True,
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def create_drawdown_chart(returns: pd.Series) -> go.Figure:
    """Create drawdown chart."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='#ef4444', width=1),
        fillcolor='rgba(239, 68, 68, 0.3)',
        hovertemplate='Date: %{x}<br>' +
                      'Drawdown: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def create_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """Create monthly returns heatmap."""
    monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )

    # Reshape for heatmap
    heatmap_data = []
    years = sorted(monthly_returns.index.get_level_values(0).unique())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for year in years:
        year_data = []
        for month in range(1, 13):
            try:
                value = monthly_returns.loc[(year, month)] * 100
                year_data.append(value)
            except KeyError:
                year_data.append(None)
        heatmap_data.append(year_data)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=months,
        y=[str(year) for year in years],
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{val:.1f}%" if val is not None else "" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Monthly Returns Heatmap",
        height=200 + len(years) * 25,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def create_risk_return_scatter(portfolio_metrics: dict, benchmark_metrics: dict = None) -> go.Figure:
    """Create risk-return scatter plot."""
    fig = go.Figure()

    # Portfolio point
    fig.add_trace(go.Scatter(
        x=[portfolio_metrics['volatility'] * 100],
        y=[portfolio_metrics['annualized_return'] * 100],
        mode='markers',
        name='Portfolio',
        marker=dict(
            size=15,
            color='#3b82f6',
            symbol='star'
        ),
        hovertemplate='<b>Portfolio</b><br>' +
                      'Return: %{y:.2f}%<br>' +
                      'Risk: %{x:.2f}%<br>' +
                      '<extra></extra>'
    ))

    # Benchmark point if available
    if benchmark_metrics:
        fig.add_trace(go.Scatter(
            x=[benchmark_metrics['volatility'] * 100],
            y=[benchmark_metrics['annualized_return'] * 100],
            mode='markers',
            name='Benchmark',
            marker=dict(
                size=12,
                color='#6b7280',
                symbol='circle'
            ),
            hovertemplate='<b>Benchmark</b><br>' +
                          'Return: %{y:.2f}%<br>' +
                          'Risk: %{x:.2f}%<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (%)",
        yaxis_title="Annualized Return (%)",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def create_asset_allocation_chart(portfolio) -> go.Figure:
    """Create asset allocation pie chart."""
    if not portfolio.assets:
        return go.Figure()

    # Calculate current allocations
    total_value = sum(asset.shares * asset.current_price for asset in portfolio.assets if asset.current_price)

    labels = []
    values = []
    colors = px.colors.qualitative.Set3

    for i, asset in enumerate(portfolio.assets):
        if asset.current_price and asset.shares:
            asset_value = asset.shares * asset.current_price
            allocation = asset_value / total_value if total_value > 0 else 0
            labels.append(f"{asset.ticker}<br>{asset.name[:20]}...")
            values.append(allocation * 100)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>' +
                      'Allocation: %{value:.1f}%<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title="Current Asset Allocation",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def display_metrics_grid(metrics: dict, title: str):
    """Display metrics in a grid format."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

    # Split metrics into columns
    cols = st.columns(4)
    metric_items = list(metrics.items())

    for i, (key, value) in enumerate(metric_items):
        col = cols[i % 4]

        # Format metric name
        display_name = key.replace('_', ' ').title()

        # Format value based on metric type
        if 'ratio' in key.lower() or key in ['beta', 'correlation', 'alpha']:
            formatted_value = format_number(value, 3)
        elif 'return' in key.lower() or 'volatility' in key.lower() or 'drawdown' in key.lower():
            formatted_value = format_percentage(value)
        elif 'duration' in key.lower() or 'days' in key.lower():
            formatted_value = f"{int(value)} days"
        else:
            formatted_value = format_number(value, 2)

        # Determine if metric is positive/negative
        is_positive_good = not any(word in key.lower() for word in ['volatility', 'drawdown', 'loss', 'risk', 'var'])

        with col:
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                delta_color = "normal"
                if 'return' in key.lower() and value != 0:
                    delta_color = "normal" if value > 0 else "inverse"
                elif 'volatility' in key.lower() or 'drawdown' in key.lower():
                    delta_color = "inverse"

                st.metric(
                    label=display_name,
                    value=formatted_value,
                    delta=None
                )
            else:
                st.metric(
                    label=display_name,
                    value="N/A"
                )

# Main page content
def main():
    st.title("📊 Portfolio Analysis")
    st.markdown("Comprehensive analysis of your portfolio performance, risk metrics, and allocation.")

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Settings")

        # Portfolio selection
        portfolio_manager = get_portfolio_manager()
        portfolios = portfolio_manager.list_portfolios()

        if not portfolios:
            st.error("No portfolios found. Please create a portfolio first.")
            st.stop()

        portfolio_names = [p.name for p in portfolios]
        selected_portfolio_name = st.selectbox(
            "Select Portfolio",
            portfolio_names,
            help="Choose which portfolio to analyze"
        )

        # Date range selection
        st.subheader("Analysis Period")
        date_options = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "All Time": None
        }

        selected_period = st.selectbox("Time Period", list(date_options.keys()), index=3)

        # Custom date range option
        use_custom_dates = st.checkbox("Use custom date range")
        if use_custom_dates:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
        else:
            end_date = datetime.now().date()
            if date_options[selected_period]:
                start_date = end_date - timedelta(days=date_options[selected_period])
            else:
                start_date = datetime(2020, 1, 1).date()  # Default start for "All Time"

        # Benchmark selection
        st.subheader("Benchmark Comparison")
        benchmark_options = {
            "None": None,
            "S&P 500": "SPY",
            "NASDAQ": "QQQ",
            "Total Market": "VTI",
            "Bonds": "AGG",
            "Custom": "CUSTOM"
        }

        selected_benchmark = st.selectbox("Compare to Benchmark", list(benchmark_options.keys()))

        if selected_benchmark == "Custom":
            custom_benchmark = st.text_input("Enter ticker symbol", "SPY")
            benchmark_ticker = custom_benchmark.upper()
        else:
            benchmark_ticker = benchmark_options[selected_benchmark]

    # Get selected portfolio
    selected_portfolio = next(p for p in portfolios if p.name == selected_portfolio_name)

    # Main content area
    with st.container():
        # Portfolio header with basic info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Portfolio Value",
                format_currency(selected_portfolio.total_value),
                delta=None
            )

        with col2:
            st.metric(
                "Number of Assets",
                len(selected_portfolio.assets),
                delta=None
            )

        with col3:
            if selected_portfolio.initial_value:
                total_return = (selected_portfolio.total_value - selected_portfolio.initial_value) / selected_portfolio.initial_value
                st.metric(
                    "Total Return",
                    format_percentage(total_return),
                    delta=None
                )
            else:
                st.metric("Total Return", "N/A")

        with col4:
            st.metric(
                "Created",
                selected_portfolio.created_date.strftime("%Y-%m-%d") if selected_portfolio.created_date else "N/A",
                delta=None
            )

    # Get price data and calculate metrics
    try:
        price_manager = safe_get_price_manager()
        tickers = [asset.ticker for asset in selected_portfolio.assets if hasattr(asset, 'ticker') and asset.ticker]

        if not tickers:
            st.warning("No valid tickers found in portfolio. Using demo data.")
            # Create demo data
            tickers = ['AAPL', 'MSFT', 'GOOGL']

        # Fetch historical data
        with st.spinner("Fetching price data and calculating metrics..."):
            prices_data = price_manager.get_historical_prices(tickers, start_date, end_date)

            if prices_data.empty:
                st.error("No price data available for the selected period.")
                # Create sample data for demo
                import pandas as pd
                import numpy as np

                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                sample_data = {}
                for ticker in tickers:
                    returns = np.random.normal(0.001, 0.02, len(dates))
                    prices = 100 * np.exp(np.cumsum(returns))
                    sample_data[ticker] = prices
                prices_data = pd.DataFrame(sample_data, index=dates)

            # Calculate portfolio returns
            weights = []
            for i, asset in enumerate(selected_portfolio.assets):
                if hasattr(asset, 'weight') and asset.weight:
                    weights.append(asset.weight)
                else:
                    weights.append(1.0 / len(tickers))  # Equal weight fallback

            # Ensure we have enough weights for tickers
            while len(weights) < len(tickers):
                weights.append(1.0 / len(tickers))

            # Normalize weights
            weights = np.array(weights[:len(tickers)])  # Truncate if too many
            weights = weights / weights.sum()

            # Calculate portfolio returns
            returns = prices_data.pct_change().dropna()
            portfolio_returns = (returns * weights).sum(axis=1)

            # Fetch benchmark data if selected
            benchmark_returns = None
            if benchmark_ticker:
                try:
                    benchmark_data = price_manager.get_historical_prices([benchmark_ticker], start_date, end_date)
                    if not benchmark_data.empty:
                        benchmark_returns = benchmark_data[benchmark_ticker].pct_change().dropna()
                    else:
                        st.warning(f"Could not fetch benchmark data for {benchmark_ticker}")
                except Exception as e:
                    st.warning(f"Could not fetch benchmark data: {e}")

            # Calculate metrics
            if ANALYTICS_AVAILABLE:
                calculator = PerformanceCalculator()
                portfolio_metrics = calculator.calculate_all_metrics(portfolio_returns, benchmark_returns)

                # Calculate benchmark metrics if available
                benchmark_metrics = None
                if benchmark_returns is not None:
                    benchmark_metrics = calculator.calculate_all_metrics(benchmark_returns)
            else:
                # Basic fallback metrics
                portfolio_metrics = {
                    'total_return': (1 + portfolio_returns).prod() - 1,
                    'annualized_return': portfolio_returns.mean() * 252,
                    'volatility': portfolio_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
                    'max_drawdown': ((1 + portfolio_returns).cumprod()).min() - 1,
                    'win_rate': (portfolio_returns > 0).mean(),
                    'best_month': portfolio_returns.max(),
                    'worst_month': portfolio_returns.min(),
                    'mean': portfolio_returns.mean(),
                    'median': portfolio_returns.median(),
                    'standard_deviation': portfolio_returns.std(),
                    'skewness': portfolio_returns.skew(),
                    'kurtosis': portfolio_returns.kurtosis(),
                    'var_95': portfolio_returns.quantile(0.05),
                    'var_99': portfolio_returns.quantile(0.01),
                    'cvar_95': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean(),
                    'downside_deviation': portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252),
                    'sortino_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)),
                    'calmar_ratio': (portfolio_returns.mean() * 252) / abs(((1 + portfolio_returns).cumprod()).min() - 1),
                    'information_ratio': 0,
                    'omega_ratio': 1,
                    'gain_to_pain_ratio': abs(portfolio_returns[portfolio_returns > 0].sum() / portfolio_returns[portfolio_returns < 0].sum()),
                    'jarque_bera_pvalue': 0.5,
                    'percentile_1': portfolio_returns.quantile(0.01),
                    'percentile_5': portfolio_returns.quantile(0.05),
                    'percentile_25': portfolio_returns.quantile(0.25),
                    'percentile_75': portfolio_returns.quantile(0.75),
                    'percentile_95': portfolio_returns.quantile(0.95),
                    'monthly_win_rate': 0.6,
                    'quarterly_win_rate': 0.7,
                    'yearly_win_rate': 0.8,
                    'max_consecutive_wins': 5,
                    'max_consecutive_losses': 3
                }
                benchmark_metrics = None

    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        st.info("Showing demo data instead")

        # Create minimal demo data
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start=start_date, end=end_date, freq='D')[-100:]  # Last 100 days
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)

        portfolio_metrics = {
            'total_return': 0.15,
            'annualized_return': 0.12,
            'volatility': 0.18,
            'sharpe_ratio': 0.67,
            'max_drawdown': -0.08,
            'win_rate': 0.58,
            'best_month': 0.05,
            'worst_month': -0.04,
            'mean': 0.001,
            'median': 0.0008,
            'standard_deviation': 0.02,
            'skewness': -0.1,
            'kurtosis': 3.2,
            'var_95': -0.03,
            'var_99': -0.045,
            'cvar_95': -0.035,
            'downside_deviation': 0.15,
            'sortino_ratio': 0.8,
            'calmar_ratio': 1.5,
            'information_ratio': 0.25,
            'omega_ratio': 1.2,
            'gain_to_pain_ratio': 1.3,
            'jarque_bera_pvalue': 0.05,
            'percentile_1': -0.045,
            'percentile_5': -0.03,
            'percentile_25': -0.01,
            'percentile_75': 0.012,
            'percentile_95': 0.035,
            'monthly_win_rate': 0.6,
            'quarterly_win_rate': 0.75,
            'yearly_win_rate': 0.8,
            'max_consecutive_wins': 7,
            'max_consecutive_losses': 4
        }
        benchmark_metrics = None

    # Tab interface for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance",
        "⚠️ Risk Analysis",
        "🎯 Allocation",
        "📊 Statistics",
        "📅 Calendar"
    ])

    with tab1:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Performance charts
        col1, col2 = st.columns([2, 1])

        with col1:
            # Create combined returns data
            returns_data = pd.DataFrame({'portfolio': portfolio_returns})
            if benchmark_returns is not None:
                # Align benchmark returns with portfolio returns
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
                returns_data['benchmark'] = aligned_benchmark

            fig_performance = create_performance_chart(returns_data)
            st.plotly_chart(fig_performance, use_container_width=True)

        with col2:
            # Risk-return scatter
            fig_scatter = create_risk_return_scatter(portfolio_metrics, benchmark_metrics)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Performance metrics
        key_metrics = {
            'total_return': portfolio_metrics['total_return'],
            'annualized_return': portfolio_metrics['annualized_return'],
            'volatility': portfolio_metrics['volatility'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'max_drawdown': portfolio_metrics['max_drawdown'],
            'win_rate': portfolio_metrics['win_rate'],
            'best_month': portfolio_metrics['best_month'],
            'worst_month': portfolio_metrics['worst_month']
        }

        display_metrics_grid(key_metrics, "Key Performance Metrics")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Risk analysis charts
        col1, col2 = st.columns(2)

        with col1:
            fig_drawdown = create_drawdown_chart(portfolio_returns)
            st.plotly_chart(fig_drawdown, use_container_width=True)

        with col2:
            # Risk metrics visualization
            risk_metrics_chart = {
                'VaR 95%': abs(portfolio_metrics['var_95']) * 100,
                'VaR 99%': abs(portfolio_metrics['var_99']) * 100,
                'CVaR 95%': abs(portfolio_metrics['cvar_95']) * 100,
                'Max Drawdown': abs(portfolio_metrics['max_drawdown']) * 100
            }

            fig_risk = go.Figure(data=[
                go.Bar(
                    x=list(risk_metrics_chart.keys()),
                    y=list(risk_metrics_chart.values()),
                    marker_color=['#ef4444', '#dc2626', '#b91c1c', '#991b1b']
                )
            ])

            fig_risk.update_layout(
                title="Risk Metrics Comparison",
                yaxis_title="Loss (%)",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_risk, use_container_width=True)

        # Risk metrics grid
        risk_metrics = {
            'volatility': portfolio_metrics['volatility'],
            'var_95': portfolio_metrics['var_95'],
            'var_99': portfolio_metrics['var_99'],
            'cvar_95': portfolio_metrics['cvar_95'],
            'max_drawdown': portfolio_metrics['max_drawdown'],
            'downside_deviation': portfolio_metrics['downside_deviation'],
            'skewness': portfolio_metrics['skewness'],
            'kurtosis': portfolio_metrics['kurtosis']
        }

        display_metrics_grid(risk_metrics, "Risk Metrics")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Asset allocation analysis
        col1, col2 = st.columns([1, 1])

        with col1:
            fig_allocation = create_asset_allocation_chart(selected_portfolio)
            st.plotly_chart(fig_allocation, use_container_width=True)

        with col2:
            # Asset details table
            st.subheader("Asset Details")

            asset_data = []
            total_value = sum(asset.shares * asset.current_price for asset in selected_portfolio.assets if asset.current_price)

            for asset in selected_portfolio.assets:
                if asset.current_price and asset.shares:
                    asset_value = asset.shares * asset.current_price
                    allocation = asset_value / total_value if total_value > 0 else 0

                    asset_data.append({
                        'Ticker': asset.ticker,
                        'Name': asset.name[:25] + '...' if len(asset.name) > 25 else asset.name,
                        'Shares': asset.shares,
                        'Price': asset.current_price,
                        'Value': asset_value,
                        'Allocation': allocation
                    })

            if asset_data:
                df = pd.DataFrame(asset_data)
                df['Price'] = df['Price'].apply(lambda x: format_currency(x))
                df['Value'] = df['Value'].apply(lambda x: format_currency(x))
                df['Allocation'] = df['Allocation'].apply(lambda x: format_percentage(x))

                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No asset data available.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Statistical analysis
        col1, col2 = st.columns(2)

        with col1:
            # Distribution metrics
            distribution_metrics = {
                'mean': portfolio_metrics['mean'],
                'median': portfolio_metrics['median'],
                'standard_deviation': portfolio_metrics['standard_deviation'],
                'skewness': portfolio_metrics['skewness'],
                'kurtosis': portfolio_metrics['kurtosis'],
                'jarque_bera_pvalue': portfolio_metrics['jarque_bera_pvalue']
            }

            display_metrics_grid(distribution_metrics, "Distribution Statistics")

        with col2:
            # Risk-adjusted metrics
            risk_adjusted_metrics = {
                'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
                'sortino_ratio': portfolio_metrics['sortino_ratio'],
                'calmar_ratio': portfolio_metrics['calmar_ratio'],
                'information_ratio': portfolio_metrics['information_ratio'],
                'omega_ratio': portfolio_metrics['omega_ratio'],
                'gain_to_pain_ratio': portfolio_metrics['gain_to_pain_ratio']
            }

            display_metrics_grid(risk_adjusted_metrics, "Risk-Adjusted Ratios")

        # Returns distribution histogram
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=50,
                marker_color='#3b82f6',
                opacity=0.7,
                name='Daily Returns'
            )
        ])

        fig_hist.update_layout(
            title="Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Calendar analysis
        if len(portfolio_returns) > 30:  # Need enough data for calendar
            fig_calendar = create_monthly_returns_heatmap(portfolio_returns)
            st.plotly_chart(fig_calendar, use_container_width=True)

            # Monthly statistics
            monthly_returns = portfolio_returns.groupby([portfolio_returns.index.year, portfolio_returns.index.month]).apply(
                lambda x: (1 + x).prod() - 1
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Best Month", format_percentage(monthly_returns.max()))

            with col2:
                st.metric("Worst Month", format_percentage(monthly_returns.min()))

            with col3:
                st.metric("Average Month", format_percentage(monthly_returns.mean()))

            with col4:
                positive_months = (monthly_returns > 0).sum()
                total_months = len(monthly_returns)
                win_rate = positive_months / total_months if total_months > 0 else 0
                st.metric("Monthly Win Rate", format_percentage(win_rate))
        else:
            st.info("Not enough data for calendar analysis. Need at least 30 days of data.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Export section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("📊 Export Metrics to CSV", use_container_width=True):
            metrics_df = pd.DataFrame(list(portfolio_metrics.items()), columns=['Metric', 'Value'])
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_portfolio_name}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📈 Export Returns Data", use_container_width=True):
            returns_df = pd.DataFrame({
                'Date': portfolio_returns.index,
                'Portfolio_Return': portfolio_returns.values
            })
            if benchmark_returns is not None:
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
                returns_df['Benchmark_Return'] = aligned_benchmark.values

            csv = returns_df.to_csv(index=False)
            st.download_button(
                label="Download Returns CSV",
                data=csv,
                file_name=f"{selected_portfolio_name}_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col3:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()