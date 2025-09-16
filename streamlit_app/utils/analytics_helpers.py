"""
Analytics Helper Functions
Utility functions for portfolio analytics and data processing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import streamlit as st


def calculate_portfolio_weights(portfolio, method: str = 'market_value') -> np.ndarray:
    """
    Calculate portfolio weights using different methods.

    Args:
        portfolio: Portfolio object
        method: Weighting method ('market_value', 'equal', 'specified')

    Returns:
        Array of normalized weights
    """
    if method == 'market_value':
        # Calculate weights based on current market values
        weights = []
        total_value = sum(asset.shares * asset.current_price for asset in portfolio.assets if asset.current_price)

        for asset in portfolio.assets:
            if asset.shares and asset.current_price and total_value > 0:
                weight = (asset.shares * asset.current_price) / total_value
                weights.append(weight)
            else:
                weights.append(0)

    elif method == 'equal':
        # Equal weighting
        n_assets = len(portfolio.assets)
        weights = [1.0 / n_assets] * n_assets

    elif method == 'specified':
        # Use specified weights if available
        weights = []
        for asset in portfolio.assets:
            if hasattr(asset, 'weight') and asset.weight:
                weights.append(asset.weight)
            else:
                weights.append(1.0 / len(portfolio.assets))  # Fallback to equal weight

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights to sum to 1
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    return weights


def calculate_portfolio_returns(prices_data: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Calculate portfolio returns from price data and weights.

    Args:
        prices_data: DataFrame with asset prices
        weights: Array of portfolio weights

    Returns:
        Series of portfolio returns
    """
    # Calculate individual asset returns
    returns = prices_data.pct_change().dropna()

    # Calculate weighted portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)

    return portfolio_returns


def align_benchmark_data(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align portfolio and benchmark returns to same time periods.

    Args:
        portfolio_returns: Portfolio returns series
        benchmark_returns: Benchmark returns series

    Returns:
        Tuple of aligned portfolio and benchmark returns
    """
    # Find common date range
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)

    if len(common_dates) == 0:
        raise ValueError("No common dates between portfolio and benchmark data")

    # Align both series
    aligned_portfolio = portfolio_returns.loc[common_dates]
    aligned_benchmark = benchmark_returns.loc[common_dates]

    return aligned_portfolio, aligned_benchmark


def create_performance_summary_table(analysis_results: Dict) -> pd.DataFrame:
    """
    Create a formatted summary table of key performance metrics.

    Args:
        analysis_results: Results from analytics engine

    Returns:
        Formatted DataFrame for display
    """
    from .formatting import format_percentage, format_number

    # Extract key metrics
    perf = analysis_results['performance']
    risk = analysis_results['risk']
    risk_adj = analysis_results['risk_adjusted']

    summary_data = [
        ['Total Return', format_percentage(perf['total_return'])],
        ['Annualized Return', format_percentage(perf['annualized_return'])],
        ['Volatility', format_percentage(perf['volatility'])],
        ['Sharpe Ratio', format_number(perf['sharpe_ratio'], 2)],
        ['Sortino Ratio', format_number(risk_adj['sortino_ratio'], 2)],
        ['Max Drawdown', format_percentage(risk['max_drawdown'])],
        ['VaR (95%)', format_percentage(risk['var_95'])],
        ['Win Rate', format_percentage(perf['win_rate'])],
        ['Best Month', format_percentage(perf['best_month'])],
        ['Worst Month', format_percentage(perf['worst_month'])]
    ]

    return pd.DataFrame(summary_data, columns=['Metric', 'Value'])


def generate_risk_alerts(analysis_results: Dict, thresholds: Dict = None) -> List[str]:
    """
    Generate risk alerts based on portfolio metrics.

    Args:
        analysis_results: Results from analytics engine
        thresholds: Custom thresholds for alerts

    Returns:
        List of alert messages
    """
    if thresholds is None:
        thresholds = {
            'max_drawdown': -0.20,  # Alert if max drawdown > 20%
            'var_95': -0.05,  # Alert if daily VaR > 5%
            'volatility': 0.30,  # Alert if volatility > 30%
            'sharpe_ratio': 0.5,  # Alert if Sharpe < 0.5
            'consecutive_losses': 10  # Alert if > 10 consecutive loss days
        }

    alerts = []
    risk = analysis_results['risk']
    perf = analysis_results['performance']

    # Check max drawdown
    if risk['max_drawdown'] < thresholds['max_drawdown']:
        alerts.append(f"⚠️ High maximum drawdown of {risk['max_drawdown']:.1%}")

    # Check VaR
    if risk['var_95'] < thresholds['var_95']:
        alerts.append(f"⚠️ High daily VaR (95%) of {risk['var_95']:.1%}")

    # Check volatility
    if risk['volatility'] > thresholds['volatility']:
        alerts.append(f"⚠️ High portfolio volatility of {risk['volatility']:.1%}")

    # Check Sharpe ratio
    if perf['sharpe_ratio'] < thresholds['sharpe_ratio']:
        alerts.append(f"⚠️ Low risk-adjusted returns (Sharpe: {perf['sharpe_ratio']:.2f})")

    # Check consecutive losses
    if risk['max_consecutive_losses'] > thresholds['consecutive_losses']:
        alerts.append(f"⚠️ {risk['max_consecutive_losses']} consecutive loss days detected")

    # Check for extreme skewness
    if abs(risk['skewness']) > 2:
        skew_type = "negative" if risk['skewness'] < 0 else "positive"
        alerts.append(f"⚠️ Extreme {skew_type} skewness detected ({risk['skewness']:.2f})")

    # Check for high kurtosis (fat tails)
    if risk['kurtosis'] > 5:
        alerts.append(f"⚠️ High kurtosis indicates fat tail risk ({risk['kurtosis']:.2f})")

    return alerts


def calculate_correlation_matrix(returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for portfolio assets.

    Args:
        returns_data: DataFrame with asset returns

    Returns:
        Correlation matrix DataFrame
    """
    return returns_data.corr()


def calculate_rolling_correlation(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252
) -> pd.Series:
    """
    Calculate rolling correlation between portfolio and benchmark.

    Args:
        portfolio_returns: Portfolio returns series
        benchmark_returns: Benchmark returns series
        window: Rolling window size in days

    Returns:
        Rolling correlation series
    """
    aligned_portfolio, aligned_benchmark = align_benchmark_data(portfolio_returns, benchmark_returns)

    rolling_corr = aligned_portfolio.rolling(window).corr(aligned_benchmark)
    return rolling_corr


def calculate_sector_allocation(portfolio) -> Dict[str, float]:
    """
    Calculate sector allocation for portfolio.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with sector allocations
    """
    sector_allocation = {}
    total_value = sum(asset.shares * asset.current_price for asset in portfolio.assets if asset.current_price)

    for asset in portfolio.assets:
        if asset.shares and asset.current_price:
            sector = getattr(asset, 'sector', 'Unknown')
            value = asset.shares * asset.current_price

            if sector in sector_allocation:
                sector_allocation[sector] += value
            else:
                sector_allocation[sector] = value

    # Convert to percentages
    if total_value > 0:
        sector_allocation = {sector: value / total_value for sector, value in sector_allocation.items()}

    return sector_allocation


def calculate_geographic_allocation(portfolio) -> Dict[str, float]:
    """
    Calculate geographic allocation for portfolio.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with geographic allocations
    """
    geo_allocation = {}
    total_value = sum(asset.shares * asset.current_price for asset in portfolio.assets if asset.current_price)

    # Simplified geographic classification based on ticker patterns
    def get_geography(ticker: str) -> str:
        # This is a simplified classification - real implementation would use market data
        if any(ticker.startswith(prefix) for prefix in ['VTI', 'SPY', 'QQQ', 'IWM']):
            return 'US'
        elif any(ticker.startswith(prefix) for prefix in ['VEA', 'EFA']):
            return 'Developed International'
        elif any(ticker.startswith(prefix) for prefix in ['VWO', 'EEM']):
            return 'Emerging Markets'
        elif '.TO' in ticker or '.TSE' in ticker:
            return 'Canada'
        elif '.L' in ticker or '.LON' in ticker:
            return 'UK'
        else:
            return 'Unknown'

    for asset in portfolio.assets:
        if asset.shares and asset.current_price:
            geography = get_geography(asset.ticker)
            value = asset.shares * asset.current_price

            if geography in geo_allocation:
                geo_allocation[geography] += value
            else:
                geo_allocation[geography] = value

    # Convert to percentages
    if total_value > 0:
        geo_allocation = {geo: value / total_value for geo, value in geo_allocation.items()}

    return geo_allocation


def calculate_concentration_metrics(portfolio) -> Dict[str, float]:
    """
    Calculate portfolio concentration metrics.

    Args:
        portfolio: Portfolio object

    Returns:
        Dictionary with concentration metrics
    """
    weights = calculate_portfolio_weights(portfolio, method='market_value')

    # Herfindahl-Hirschman Index (HHI)
    hhi = sum(w ** 2 for w in weights)

    # Effective number of assets
    effective_assets = 1 / hhi if hhi > 0 else 0

    # Concentration ratio (top 3 holdings)
    sorted_weights = sorted(weights, reverse=True)
    top_3_concentration = sum(sorted_weights[:3])

    # Maximum weight
    max_weight = max(weights) if len(weights) > 0 else 0

    return {
        'herfindahl_index': hhi,
        'effective_number_assets': effective_assets,
        'top_3_concentration': top_3_concentration,
        'max_single_weight': max_weight,
        'number_of_assets': len(weights)
    }


def detect_outliers(returns: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in returns series.

    Args:
        returns: Returns series
        method: Method for outlier detection ('zscore', 'iqr')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(returns.dropna()))
        outliers = pd.Series(z_scores > threshold, index=returns.dropna().index)

    elif method == 'iqr':
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (returns < lower_bound) | (returns > upper_bound)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return outliers.reindex(returns.index, fill_value=False)


def calculate_regime_indicators(returns: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """
    Calculate market regime indicators.

    Args:
        returns: Returns series
        lookback: Lookback period for calculations

    Returns:
        DataFrame with regime indicators
    """
    regime_data = pd.DataFrame(index=returns.index)

    # Rolling volatility regime
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
    vol_median = rolling_vol.median()
    regime_data['high_vol_regime'] = rolling_vol > vol_median

    # Trend regime (based on moving average)
    rolling_return = returns.rolling(lookback).mean() * 252
    regime_data['bull_regime'] = rolling_return > 0

    # Drawdown regime
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    regime_data['drawdown_regime'] = drawdown < -0.05  # 5% drawdown threshold

    # Correlation regime (if benchmark available)
    # This would require benchmark data, placeholder for now
    regime_data['correlation_regime'] = False

    return regime_data


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_and_process_benchmark_data(
        benchmark_ticker: str,
        start_date: datetime,
        end_date: datetime
) -> Optional[pd.Series]:
    """
    Fetch and process benchmark data with caching.

    Args:
        benchmark_ticker: Benchmark ticker symbol
        start_date: Start date
        end_date: End date

    Returns:
        Benchmark returns series or None if failed
    """
    try:
        import yfinance as yf

        # Fetch benchmark data
        benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)

        if benchmark.empty:
            return None

        # Calculate returns
        if 'Adj Close' in benchmark.columns:
            benchmark_prices = benchmark['Adj Close']
        else:
            benchmark_prices = benchmark['Close']

        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_returns.name = benchmark_ticker

        return benchmark_returns

    except Exception as e:
        st.error(f"Failed to fetch benchmark data for {benchmark_ticker}: {e}")
        return None


def create_performance_attribution_data(
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Create performance attribution data.

    Args:
        portfolio_returns: Portfolio returns
        asset_returns: Individual asset returns
        weights: Portfolio weights

    Returns:
        Attribution data dictionary
    """
    attribution = {
        'asset_selection': {},
        'allocation_effect': {},
        'interaction_effect': {}
    }

    # Calculate contribution of each asset
    for i, (asset, weight) in enumerate(zip(asset_returns.columns, weights)):
        if i < len(asset_returns.columns):
            asset_contribution = (asset_returns.iloc[:, i] * weight).sum()
            attribution['asset_selection'][asset] = asset_contribution

    # This is a simplified attribution - real attribution would require benchmark weights
    return attribution


def calculate_style_analysis(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> Dict[str, float]:
    """
    Perform style analysis using factor returns.

    Args:
        portfolio_returns: Portfolio returns series
        factor_returns: DataFrame with factor returns (e.g., size, value, momentum)

    Returns:
        Dictionary with factor loadings
    """
    try:
        from scipy.optimize import minimize

        # Align data
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_dates) < 30:  # Need sufficient data
            return {}

        y = portfolio_returns.loc[common_dates].values
        X = factor_returns.loc[common_dates].values

        # Objective function for style analysis (minimize tracking error)
        def objective(weights):
            predicted = X.dot(weights)
            return np.sum((y - predicted) ** 2)

        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(X.shape[1])]

        # Initial guess
        initial_weights = np.ones(X.shape[1]) / X.shape[1]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            factor_loadings = dict(zip(factor_returns.columns, result.x))
            return factor_loadings
        else:
            return {}

    except Exception:
        return {}


def generate_portfolio_insights(analysis_results: Dict, portfolio_name: str) -> List[str]:
    """
    Generate automated insights from portfolio analysis.

    Args:
        analysis_results: Complete analysis results
        portfolio_name: Name of the portfolio

    Returns:
        List of insight strings
    """
    insights = []

    perf = analysis_results['performance']
    risk = analysis_results['risk']
    risk_adj = analysis_results['risk_adjusted']

    # Performance insights
    total_return = perf['total_return']
    if total_return > 0.20:
        insights.append(f"🚀 {portfolio_name} delivered exceptional returns of {total_return:.1%}")
    elif total_return > 0.10:
        insights.append(f"✅ {portfolio_name} achieved strong returns of {total_return:.1%}")
    elif total_return > 0:
        insights.append(f"📈 {portfolio_name} generated positive returns of {total_return:.1%}")
    else:
        insights.append(f"📉 {portfolio_name} experienced a loss of {abs(total_return):.1%}")

    # Risk-adjusted performance
    sharpe = perf['sharpe_ratio']
    if sharpe > 1.5:
        insights.append(f"⭐ Excellent risk-adjusted performance with Sharpe ratio of {sharpe:.2f}")
    elif sharpe > 1.0:
        insights.append(f"✅ Good risk-adjusted performance with Sharpe ratio of {sharpe:.2f}")
    elif sharpe > 0.5:
        insights.append(f"⚡ Moderate risk-adjusted performance with Sharpe ratio of {sharpe:.2f}")
    else:
        insights.append(f"⚠️ Poor risk-adjusted performance with Sharpe ratio of {sharpe:.2f}")

    # Volatility analysis
    volatility = risk['volatility']
    if volatility < 0.10:
        insights.append(f"🛡️ Low-risk portfolio with {volatility:.1%} annual volatility")
    elif volatility < 0.20:
        insights.append(f"⚖️ Moderate-risk portfolio with {volatility:.1%} annual volatility")
    else:
        insights.append(f"⚡ High-risk portfolio with {volatility:.1%} annual volatility")

    # Drawdown analysis
    max_dd = abs(risk['max_drawdown'])
    if max_dd < 0.05:
        insights.append(f"🛡️ Minimal drawdown risk with maximum decline of {max_dd:.1%}")
    elif max_dd < 0.15:
        insights.append(f"⚖️ Controlled drawdown risk with maximum decline of {max_dd:.1%}")
    else:
        insights.append(f"⚠️ Significant drawdown risk with maximum decline of {max_dd:.1%}")

    # Consistency insights
    win_rate = perf['win_rate']
    if win_rate > 0.65:
        insights.append(f"🎯 Highly consistent with {win_rate:.1%} of periods positive")
    elif win_rate > 0.55:
        insights.append(f"✅ Good consistency with {win_rate:.1%} of periods positive")
    else:
        insights.append(f"⚠️ Low consistency with only {win_rate:.1%} of periods positive")

    # Distribution insights
    skewness = risk['skewness']
    if skewness > 0.5:
        insights.append(
            f"📈 Positive skewness ({skewness:.2f}) suggests more frequent small losses, occasional large gains")
    elif skewness < -0.5:
        insights.append(
            f"📉 Negative skewness ({skewness:.2f}) suggests more frequent small gains, occasional large losses")

    # Kurtosis insights
    kurtosis = risk['kurtosis']
    if kurtosis > 3:
        insights.append(f"⚠️ High kurtosis ({kurtosis:.2f}) indicates elevated tail risk")

    # Benchmark comparison insights
    if 'benchmark_comparison' in analysis_results:
        bench = analysis_results['benchmark_comparison']
        alpha = bench['regression']['alpha']
        beta = bench['regression']['beta']

        if alpha > 0.02:
            insights.append(f"🌟 Strong alpha generation of {alpha:.1%} vs benchmark")
        elif alpha > 0:
            insights.append(f"✅ Positive alpha of {alpha:.1%} vs benchmark")
        else:
            insights.append(f"❌ Negative alpha of {alpha:.1%} vs benchmark")

        if beta < 0.8:
            insights.append(f"🛡️ Lower market risk with beta of {beta:.2f}")
        elif beta > 1.2:
            insights.append(f"⚡ Higher market risk with beta of {beta:.2f}")
        else:
            insights.append(f"⚖️ Market-like risk with beta of {beta:.2f}")

    return insights


def calculate_portfolio_efficiency_metrics(analysis_results: Dict) -> Dict[str, float]:
    """
    Calculate portfolio efficiency metrics.

    Args:
        analysis_results: Complete analysis results

    Returns:
        Dictionary of efficiency metrics
    """
    perf = analysis_results['performance']
    risk = analysis_results['risk']

    # Return per unit of risk
    return_to_risk = perf['annualized_return'] / risk['volatility'] if risk['volatility'] != 0 else 0

    # Return per unit of downside risk
    return_to_downside = perf['annualized_return'] / risk['downside_deviation'] if risk[
                                                                                       'downside_deviation'] != 0 else 0

    # Return per unit of maximum drawdown
    return_to_mdd = perf['annualized_return'] / abs(risk['max_drawdown']) if risk['max_drawdown'] != 0 else 0

    # Win rate efficiency (return per winning period)
    win_efficiency = perf['annualized_return'] / perf['win_rate'] if perf['win_rate'] != 0 else 0

    return {
        'return_to_risk_ratio': return_to_risk,
        'return_to_downside_ratio': return_to_downside,
        'return_to_max_drawdown_ratio': return_to_mdd,
        'win_rate_efficiency': win_efficiency
    }


def create_risk_budget_analysis(returns_data: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """
    Create risk budget analysis for portfolio assets.

    Args:
        returns_data: DataFrame with asset returns
        weights: Portfolio weights

    Returns:
        DataFrame with risk contributions
    """
    # Calculate covariance matrix
    cov_matrix = returns_data.cov() * 252  # Annualize

    # Portfolio variance
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Risk contributions
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / portfolio_volatility

    # Create DataFrame
    risk_budget_df = pd.DataFrame({
        'Asset': returns_data.columns,
        'Weight': weights,
        'Risk_Contribution': risk_contrib,
        'Risk_Contribution_Pct': risk_contrib / risk_contrib.sum()
    })

    risk_budget_df['Risk_vs_Weight_Ratio'] = risk_budget_df['Risk_Contribution_Pct'] / risk_budget_df['Weight']

    return risk_budget_df.sort_values('Risk_Contribution_Pct', ascending=False)


def format_analysis_period(start_date: datetime, end_date: datetime) -> str:
    """
    Format analysis period for display.

    Args:
        start_date: Analysis start date
        end_date: Analysis end date

    Returns:
        Formatted period string
    """
    delta = end_date - start_date
    days = delta.days

    if days < 32:
        return f"{days} days"
    elif days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''}"
    else:
        years = days // 365
        remaining_months = (days % 365) // 30
        if remaining_months > 0:
            return f"{years} year{'s' if years != 1 else ''}, {remaining_months} month{'s' if remaining_months != 1 else ''}"
        else:
            return f"{years} year{'s' if years != 1 else ''}"


def validate_analysis_data(returns: pd.Series, min_observations: int = 30) -> Tuple[bool, str]:
    """
    Validate data quality for analysis.

    Args:
        returns: Returns series to validate
        min_observations: Minimum required observations

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(returns) < min_observations:
        return False, f"Insufficient data: {len(returns)} observations (minimum {min_observations} required)"

    if returns.isna().sum() > len(returns) * 0.1:  # More than 10% missing
        return False, f"Too many missing values: {returns.isna().sum()} out of {len(returns)}"

    if returns.std() == 0:
        return False, "Zero volatility detected - all returns are identical"

    if abs(returns.mean()) > 0.5:  # Daily return > 50% seems unrealistic
        return False, f"Extreme average return detected: {returns.mean():.1%}"

    return True, ""


def export_analysis_to_excel(analysis_results: Dict, portfolio_name: str) -> bytes:
    """
    Export complete analysis to Excel file.

    Args:
        analysis_results: Complete analysis results
        portfolio_name: Portfolio name

    Returns:
        Excel file as bytes
    """
    from io import BytesIO
    import pandas as pd

    # Create Excel writer
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = create_performance_summary_table(analysis_results)
        summary_data.to_excel(writer, sheet_name='Summary', index=False)

        # Performance metrics
        perf_df = pd.DataFrame(list(analysis_results['performance'].items()), columns=['Metric', 'Value'])
        perf_df.to_excel(writer, sheet_name='Performance', index=False)

        # Risk metrics
        risk_df = pd.DataFrame(list(analysis_results['risk'].items()), columns=['Metric', 'Value'])
        risk_df.to_excel(writer, sheet_name='Risk', index=False)

        # Risk-adjusted metrics
        risk_adj_df = pd.DataFrame(list(analysis_results['risk_adjusted'].items()), columns=['Metric', 'Value'])
        risk_adj_df.to_excel(writer, sheet_name='Risk_Adjusted', index=False)

        # VaR analysis
        if 'var_analysis' in analysis_results:
            var_df = pd.DataFrame(list(analysis_results['var_analysis'].items()), columns=['Method', 'VaR'])
            var_df.to_excel(writer, sheet_name='VaR_Analysis', index=False)

        # Benchmark comparison (if available)
        if 'benchmark_comparison' in analysis_results:
            bench_summary = analysis_results['benchmark_comparison']['summary']
            bench_df = pd.DataFrame([
                ['Portfolio', bench_summary['Portfolio']['total_return'],
                 bench_summary['Portfolio']['annualized_return'], bench_summary['Portfolio']['volatility']],
                ['Benchmark', bench_summary['Benchmark']['total_return'],
                 bench_summary['Benchmark']['annualized_return'], bench_summary['Benchmark']['volatility']],
                ['Difference', bench_summary['outperformance']['annualized_excess_return'],
                 bench_summary['outperformance']['annualized_excess_return'],
                 bench_summary['outperformance']['excess_volatility']]
            ], columns=['Type', 'Total Return', 'Annualized Return', 'Volatility'])
            bench_df.to_excel(writer, sheet_name='Benchmark_Comparison', index=False)

    output.seek(0)
    return output.getvalue()


# Streamlit specific helper functions
def display_metric_with_tooltip(label: str, value: str, tooltip: str):
    """Display metric with tooltip in Streamlit."""
    st.metric(label=label, value=value, help=tooltip)


def create_expandable_metrics_section(title: str, metrics_dict: Dict, format_func=None):
    """Create expandable section for metrics display."""
    with st.expander(title, expanded=False):
        cols = st.columns(3)

        for i, (key, value) in enumerate(metrics_dict.items()):
            col = cols[i % 3]

            display_name = key.replace('_', ' ').title()

            if format_func:
                formatted_value = format_func(value)
            else:
                if isinstance(value, float):
                    if 'ratio' in key.lower() or 'beta' in key.lower():
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = f"{value:.2%}"
                else:
                    formatted_value = str(value)

            with col:
                st.metric(display_name, formatted_value)