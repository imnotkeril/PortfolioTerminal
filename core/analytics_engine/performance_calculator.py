"""
Performance Calculator Module
Calculates comprehensive portfolio performance metrics and analytics.

This module provides over 70 performance and risk metrics for portfolio analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf

warnings.filterwarnings('ignore')


class PerformanceCalculator:
    """
    Comprehensive performance calculation engine for portfolio analysis.

    Calculates over 70 performance, risk, and statistical metrics.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_year = 252
        self.months_year = 12

    def calculate_all_metrics(
            self,
            returns: pd.Series,
            benchmark_returns: Optional[pd.Series] = None,
            portfolio_value: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics for a returns series.

        Args:
            returns: Portfolio returns series (daily)
            benchmark_returns: Benchmark returns for comparison
            portfolio_value: Portfolio value over time

        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}

        # Basic Return Metrics (15 metrics)
        metrics.update(self._calculate_return_metrics(returns))

        # Risk Metrics (20 metrics)
        metrics.update(self._calculate_risk_metrics(returns))

        # Risk-Adjusted Metrics (15 metrics)
        metrics.update(self._calculate_risk_adjusted_metrics(returns))

        # Drawdown Metrics (10 metrics)
        metrics.update(self._calculate_drawdown_metrics(returns, portfolio_value))

        # Distribution Metrics (10 metrics)
        metrics.update(self._calculate_distribution_metrics(returns))

        # Benchmark Comparison (15 metrics if benchmark provided)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))

        # Time-based Analysis (10 metrics)
        metrics.update(self._calculate_time_based_metrics(returns))

        return metrics

    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic return metrics."""
        metrics = {}

        # Total returns
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['cumulative_return'] = metrics['total_return']

        # Annualized returns
        years = len(returns) / self.trading_days_year
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / years) - 1
        metrics['cagr'] = metrics['annualized_return']

        # Average returns
        metrics['average_return'] = returns.mean()
        metrics['average_monthly_return'] = returns.mean() * 21  # Approx 21 trading days per month
        metrics['average_annual_return'] = returns.mean() * self.trading_days_year

        # Geometric mean
        metrics['geometric_mean'] = (1 + returns).prod() ** (1 / len(returns)) - 1

        # Best/Worst periods
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()
        metrics['best_month'] = returns.rolling(21).sum().max()
        metrics['worst_month'] = returns.rolling(21).sum().min()
        metrics['best_quarter'] = returns.rolling(63).sum().max()
        metrics['worst_quarter'] = returns.rolling(63).sum().min()

        # Win/Loss statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        metrics['win_rate'] = len(positive_returns) / len(returns)
        metrics['loss_rate'] = len(negative_returns) / len(returns)
        metrics['average_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics['average_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0

        return metrics

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        metrics = {}

        # Volatility metrics
        metrics['volatility'] = returns.std() * np.sqrt(self.trading_days_year)
        metrics['annualized_volatility'] = metrics['volatility']
        metrics['daily_volatility'] = returns.std()
        metrics['monthly_volatility'] = returns.std() * np.sqrt(21)

        # Downside volatility
        downside_returns = returns[returns < 0]
        metrics['downside_volatility'] = downside_returns.std() * np.sqrt(self.trading_days_year)
        metrics['downside_deviation'] = np.sqrt(
            np.mean(np.minimum(returns - self.risk_free_rate / 252, 0) ** 2)) * np.sqrt(self.trading_days_year)

        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['var_95_monthly'] = np.percentile(returns.rolling(21).sum().dropna(), 5)

        # Conditional VaR (Expected Shortfall)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()

        # Semi-deviation
        mean_return = returns.mean()
        negative_deviations = returns[returns < mean_return] - mean_return
        metrics['semi_deviation'] = np.sqrt(np.mean(negative_deviations ** 2)) * np.sqrt(self.trading_days_year)

        # Tracking error (if compared to itself, this will be 0)
        metrics['tracking_error'] = returns.std() * np.sqrt(self.trading_days_year)

        # Skewness and Kurtosis
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        metrics['excess_kurtosis'] = metrics['kurtosis']

        # Tail ratios
        metrics['gain_to_pain_ratio'] = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[
                                                                                                            returns < 0].sum() != 0 else np.inf

        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        metrics['max_consecutive_losses'] = max_consecutive_losses

        return metrics

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        metrics = {}

        # Sharpe ratio
        excess_returns = returns.mean() - self.risk_free_rate / self.trading_days_year
        metrics['sharpe_ratio'] = excess_returns / returns.std() * np.sqrt(self.trading_days_year)

        # Sortino ratio
        downside_returns = returns[returns < self.risk_free_rate / self.trading_days_year]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        metrics['sortino_ratio'] = excess_returns / downside_std * np.sqrt(self.trading_days_year)

        # Calmar ratio
        max_dd = self._calculate_max_drawdown(returns)
        metrics['calmar_ratio'] = (returns.mean() * self.trading_days_year) / abs(max_dd) if max_dd != 0 else np.inf

        # Treynor ratio (using correlation with market as proxy for beta)
        metrics['treynor_ratio'] = metrics['sharpe_ratio']  # Simplified, would need market data for true beta

        # Information ratio
        metrics['information_ratio'] = returns.mean() / returns.std() * np.sqrt(self.trading_days_year)

        # Omega ratio
        threshold = self.risk_free_rate / self.trading_days_year
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        metrics['omega_ratio'] = gains.sum() / losses.sum() if losses.sum() != 0 else np.inf

        # Gain-to-pain ratio
        total_gains = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        metrics['gain_to_pain_ratio'] = total_gains / total_losses if total_losses != 0 else np.inf

        # Sterling ratio
        metrics['sterling_ratio'] = (returns.mean() * self.trading_days_year) / abs(max_dd) if max_dd != 0 else np.inf

        # Burke ratio
        drawdowns = self._calculate_drawdown_series(returns)
        burke_denominator = np.sqrt(np.sum(drawdowns ** 2))
        metrics['burke_ratio'] = (
                                             returns.mean() * self.trading_days_year) / burke_denominator if burke_denominator != 0 else np.inf

        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        metrics['ulcer_index'] = ulcer_index

        # Pain Index
        metrics['pain_index'] = np.mean(abs(drawdowns))

        # Return over Maximum Drawdown (RoMaD)
        metrics['romad'] = (returns.mean() * self.trading_days_year) / abs(max_dd) if max_dd != 0 else np.inf

        # Kappa ratio (similar to Sortino but with nth moment)
        metrics['kappa_ratio'] = metrics['sortino_ratio']  # Simplified

        # Upside potential ratio
        target = self.risk_free_rate / self.trading_days_year
        upside_returns = returns[returns > target] - target
        downside_variance = np.mean(np.minimum(returns - target, 0) ** 2)
        metrics['upside_potential_ratio'] = upside_returns.mean() / np.sqrt(
            downside_variance) if downside_variance != 0 else np.inf

        return metrics

    def _calculate_drawdown_metrics(self, returns: pd.Series, portfolio_value: Optional[pd.Series] = None) -> Dict[
        str, float]:
        """Calculate drawdown-related metrics."""
        metrics = {}

        # Calculate cumulative returns if portfolio value not provided
        if portfolio_value is None:
            cumulative_returns = (1 + returns).cumprod()
        else:
            cumulative_returns = portfolio_value / portfolio_value.iloc[0]

        # Running maximum
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max

        # Maximum drawdown
        metrics['max_drawdown'] = drawdowns.min()
        metrics['maximum_drawdown'] = metrics['max_drawdown']

        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        metrics['average_drawdown'] = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0

        # Drawdown duration analysis
        in_drawdown = drawdowns < -0.01  # Consider 1% threshold for drawdown
        drawdown_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:  # If ending in drawdown
            drawdown_periods.append(current_period)

        if drawdown_periods:
            metrics['max_drawdown_duration'] = max(drawdown_periods)
            metrics['average_drawdown_duration'] = np.mean(drawdown_periods)
            metrics['drawdown_frequency'] = len(drawdown_periods) / len(returns) * self.trading_days_year
        else:
            metrics['max_drawdown_duration'] = 0
            metrics['average_drawdown_duration'] = 0
            metrics['drawdown_frequency'] = 0

        # Recovery time for maximum drawdown
        max_dd_idx = drawdowns.idxmin()
        max_dd_value = drawdowns.min()

        # Find recovery point
        recovery_idx = None
        for idx in drawdowns.index[drawdowns.index > max_dd_idx]:
            if drawdowns[idx] >= -0.01:  # Recovered within 1%
                recovery_idx = idx
                break

        if recovery_idx is not None:
            recovery_days = len(drawdowns.loc[max_dd_idx:recovery_idx]) - 1
            metrics['max_drawdown_recovery_time'] = recovery_days
        else:
            metrics['max_drawdown_recovery_time'] = len(drawdowns) - drawdowns.index.get_loc(max_dd_idx)

        # Drawdown at risk (DaR)
        metrics['drawdown_at_risk_95'] = np.percentile(drawdowns, 5)
        metrics['drawdown_at_risk_99'] = np.percentile(drawdowns, 1)

        return metrics

    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return distribution metrics."""
        metrics = {}

        # Basic statistics
        metrics['mean'] = returns.mean()
        metrics['median'] = returns.median()
        metrics['standard_deviation'] = returns.std()
        metrics['variance'] = returns.var()

        # Distribution shape
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        metrics['jarque_bera_stat'], metrics['jarque_bera_pvalue'] = stats.jarque_bera(returns.dropna())

        # Percentiles
        metrics['percentile_1'] = np.percentile(returns, 1)
        metrics['percentile_5'] = np.percentile(returns, 5)
        metrics['percentile_25'] = np.percentile(returns, 25)
        metrics['percentile_75'] = np.percentile(returns, 75)
        metrics['percentile_95'] = np.percentile(returns, 95)
        metrics['percentile_99'] = np.percentile(returns, 99)

        # Range metrics
        metrics['range'] = returns.max() - returns.min()
        metrics['interquartile_range'] = metrics['percentile_75'] - metrics['percentile_25']

        return metrics

    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        metrics = {}

        # Align the series
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, keys=['portfolio', 'benchmark']).dropna()
        portfolio_ret = aligned_data['portfolio']
        benchmark_ret = aligned_data['benchmark']

        # Beta calculation
        covariance = np.cov(portfolio_ret, benchmark_ret)[0, 1]
        benchmark_variance = np.var(benchmark_ret)
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 1

        # Alpha calculation
        portfolio_annual = portfolio_ret.mean() * self.trading_days_year
        benchmark_annual = benchmark_ret.mean() * self.trading_days_year
        rf_annual = self.risk_free_rate
        metrics['alpha'] = portfolio_annual - (rf_annual + metrics['beta'] * (benchmark_annual - rf_annual))

        # Correlation
        metrics['correlation'] = portfolio_ret.corr(benchmark_ret)

        # Tracking error
        active_returns = portfolio_ret - benchmark_ret
        metrics['tracking_error'] = active_returns.std() * np.sqrt(self.trading_days_year)

        # Information ratio
        metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(
            self.trading_days_year) if active_returns.std() != 0 else 0

        # Up/Down capture ratios
        up_market = benchmark_ret > 0
        down_market = benchmark_ret < 0

        if up_market.sum() > 0:
            up_capture = (portfolio_ret[up_market].mean() / benchmark_ret[up_market].mean())
            metrics['up_capture_ratio'] = up_capture
        else:
            metrics['up_capture_ratio'] = 1

        if down_market.sum() > 0:
            down_capture = (portfolio_ret[down_market].mean() / benchmark_ret[down_market].mean())
            metrics['down_capture_ratio'] = down_capture
        else:
            metrics['down_capture_ratio'] = 1

        # Capture ratio
        if metrics['down_capture_ratio'] != 0:
            metrics['capture_ratio'] = metrics['up_capture_ratio'] / abs(metrics['down_capture_ratio'])
        else:
            metrics['capture_ratio'] = np.inf

        # R-squared
        metrics['r_squared'] = metrics['correlation'] ** 2

        # Jensen's Alpha
        metrics['jensens_alpha'] = metrics['alpha']

        # Treynor ratio
        excess_portfolio = portfolio_annual - rf_annual
        metrics['treynor_ratio'] = excess_portfolio / metrics['beta'] if metrics['beta'] != 0 else np.inf

        # M-squared
        benchmark_vol = benchmark_ret.std() * np.sqrt(self.trading_days_year)
        portfolio_vol = portfolio_ret.std() * np.sqrt(self.trading_days_year)
        if portfolio_vol != 0:
            adjusted_portfolio_return = (portfolio_annual - rf_annual) * (benchmark_vol / portfolio_vol) + rf_annual
            metrics['m_squared'] = adjusted_portfolio_return - benchmark_annual
        else:
            metrics['m_squared'] = 0

        # Active share (simplified - would need holdings data for true calculation)
        metrics['active_share'] = abs(active_returns).mean()

        return metrics

    def _calculate_time_based_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate time-based performance metrics."""
        metrics = {}

        # Rolling performance
        if len(returns) >= 252:  # At least 1 year of data
            metrics['rolling_1y_return'] = returns.rolling(252).apply(lambda x: (1 + x).prod() - 1).iloc[-1]
            metrics['rolling_1y_volatility'] = returns.rolling(252).std().iloc[-1] * np.sqrt(self.trading_days_year)
            metrics['rolling_1y_sharpe'] = (returns.rolling(252).mean().iloc[
                                                -1] - self.risk_free_rate / self.trading_days_year) / \
                                           returns.rolling(252).std().iloc[-1] * np.sqrt(self.trading_days_year)

        # Best/worst periods
        if len(returns) >= 63:  # At least 1 quarter
            quarterly_returns = returns.rolling(63).apply(lambda x: (1 + x).prod() - 1)
            metrics['best_quarter_return'] = quarterly_returns.max()
            metrics['worst_quarter_return'] = quarterly_returns.min()

        if len(returns) >= 21:  # At least 1 month
            monthly_returns = returns.rolling(21).apply(lambda x: (1 + x).prod() - 1)
            metrics['best_month_return'] = monthly_returns.max()
            metrics['worst_month_return'] = monthly_returns.min()

            # Positive months ratio
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns.dropna())
            metrics['positive_months_ratio'] = positive_months / total_months if total_months > 0 else 0

        # Consistency metrics
        monthly_rets = returns.groupby(pd.Grouper(freq='M')).apply(lambda x: (1 + x).prod() - 1)
        if len(monthly_rets) > 0:
            positive_months = (monthly_rets > 0).sum()
            metrics['consistency_score'] = positive_months / len(monthly_rets)

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def get_performance_summary(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Get a structured summary of key performance metrics.

        Returns:
            Dictionary organized by metric categories
        """
        all_metrics = self.calculate_all_metrics(returns)

        summary = {
            'returns': {
                'total_return': all_metrics['total_return'],
                'annualized_return': all_metrics['annualized_return'],
                'average_monthly_return': all_metrics['average_monthly_return'],
                'best_month': all_metrics['best_month'],
                'worst_month': all_metrics['worst_month']
            },
            'risk': {
                'volatility': all_metrics['volatility'],
                'max_drawdown': all_metrics['max_drawdown'],
                'var_95': all_metrics['var_95'],
                'downside_deviation': all_metrics['downside_deviation']
            },
            'risk_adjusted': {
                'sharpe_ratio': all_metrics['sharpe_ratio'],
                'sortino_ratio': all_metrics['sortino_ratio'],
                'calmar_ratio': all_metrics['calmar_ratio'],
                'information_ratio': all_metrics['information_ratio']
            },
            'distribution': {
                'skewness': all_metrics['skewness'],
                'kurtosis': all_metrics['kurtosis'],
                'win_rate': all_metrics['win_rate']
            }
        }

        return summary