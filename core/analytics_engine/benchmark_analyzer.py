"""
Benchmark Analyzer Module
Comprehensive benchmark comparison and analysis tools.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class BenchmarkAnalyzer:
    """
    Advanced benchmark analysis and comparison engine.

    Provides comprehensive comparison metrics between portfolio and benchmarks.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize benchmark analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_year = 252

    def compare_to_benchmark(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series,
            portfolio_name: str = "Portfolio",
            benchmark_name: str = "Benchmark"
    ) -> Dict[str, Dict]:
        """
        Comprehensive comparison between portfolio and benchmark.

        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            portfolio_name: Name for portfolio
            benchmark_name: Name for benchmark

        Returns:
            Dictionary with comprehensive comparison metrics
        """
        # Align the data
        aligned_data = pd.concat(
            [portfolio_returns, benchmark_returns],
            axis=1,
            keys=['portfolio', 'benchmark']
        ).dropna()

        if len(aligned_data) == 0:
            return {}

        port_ret = aligned_data['portfolio']
        bench_ret = aligned_data['benchmark']

        comparison = {
            'summary': self._calculate_summary_metrics(port_ret, bench_ret, portfolio_name, benchmark_name),
            'performance': self._calculate_performance_comparison(port_ret, bench_ret),
            'risk': self._calculate_risk_comparison(port_ret, bench_ret),
            'risk_adjusted': self._calculate_risk_adjusted_comparison(port_ret, bench_ret),
            'regression': self._calculate_regression_analysis(port_ret, bench_ret),
            'capture_ratios': self._calculate_capture_ratios(port_ret, bench_ret),
            'attribution': self._calculate_performance_attribution(port_ret, bench_ret),
            'rolling_analysis': self._calculate_rolling_analysis(port_ret, bench_ret),
            'statistical_tests': self._perform_statistical_tests(port_ret, bench_ret)
        }

        return comparison

    def _calculate_summary_metrics(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series,
            portfolio_name: str,
            benchmark_name: str
    ) -> Dict:
        """Calculate high-level summary metrics."""

        # Annualized returns
        port_annual = (1 + portfolio_returns).prod() ** (self.trading_days_year / len(portfolio_returns)) - 1
        bench_annual = (1 + benchmark_returns).prod() ** (self.trading_days_year / len(benchmark_returns)) - 1

        # Volatilities
        port_vol = portfolio_returns.std() * np.sqrt(self.trading_days_year)
        bench_vol = benchmark_returns.std() * np.sqrt(self.trading_days_year)

        # Sharpe ratios
        port_sharpe = (port_annual - self.risk_free_rate) / port_vol if port_vol != 0 else 0
        bench_sharpe = (bench_annual - self.risk_free_rate) / bench_vol if bench_vol != 0 else 0

        # Maximum drawdowns
        port_dd = self._calculate_max_drawdown(portfolio_returns)
        bench_dd = self._calculate_max_drawdown(benchmark_returns)

        return {
            portfolio_name: {
                'annualized_return': port_annual,
                'volatility': port_vol,
                'sharpe_ratio': port_sharpe,
                'max_drawdown': port_dd,
                'total_return': (1 + portfolio_returns).prod() - 1
            },
            benchmark_name: {
                'annualized_return': bench_annual,
                'volatility': bench_vol,
                'sharpe_ratio': bench_sharpe,
                'max_drawdown': bench_dd,
                'total_return': (1 + benchmark_returns).prod() - 1
            },
            'outperformance': {
                'annualized_excess_return': port_annual - bench_annual,
                'excess_volatility': port_vol - bench_vol,
                'sharpe_difference': port_sharpe - bench_sharpe,
                'relative_max_drawdown': port_dd - bench_dd
            }
        }

    def _calculate_performance_comparison(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate detailed performance comparison metrics."""

        # Cumulative returns
        port_cumulative = (1 + portfolio_returns).cumprod()
        bench_cumulative = (1 + benchmark_returns).cumprod()

        # Active returns
        active_returns = portfolio_returns - benchmark_returns

        # Performance metrics
        performance = {
            'total_return_portfolio': port_cumulative.iloc[-1] - 1,
            'total_return_benchmark': bench_cumulative.iloc[-1] - 1,
            'total_excess_return': (port_cumulative.iloc[-1] - 1) - (bench_cumulative.iloc[-1] - 1),
            'annualized_excess_return': active_returns.mean() * self.trading_days_year,
            'excess_return_volatility': active_returns.std() * np.sqrt(self.trading_days_year),
            'hit_ratio': (portfolio_returns > benchmark_returns).mean(),
            'win_rate': (active_returns > 0).mean(),
            'average_excess_win': active_returns[active_returns > 0].mean() if (active_returns > 0).any() else 0,
            'average_excess_loss': active_returns[active_returns < 0].mean() if (active_returns < 0).any() else 0,
            'best_relative_period': active_returns.max(),
            'worst_relative_period': active_returns.min(),
            'relative_strength': port_cumulative.iloc[-1] / bench_cumulative.iloc[-1]
        }

        # Rolling performance analysis
        if len(portfolio_returns) >= 252:  # Need at least 1 year
            rolling_excess = active_returns.rolling(252).mean() * self.trading_days_year
            performance['rolling_1y_outperformance'] = {
                'current': rolling_excess.iloc[-1],
                'average': rolling_excess.mean(),
                'best': rolling_excess.max(),
                'worst': rolling_excess.min(),
                'positive_periods': (rolling_excess > 0).sum(),
                'total_periods': len(rolling_excess.dropna())
            }

        return performance

    def _calculate_risk_comparison(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate risk comparison metrics."""

        # Basic risk metrics
        port_vol = portfolio_returns.std() * np.sqrt(self.trading_days_year)
        bench_vol = benchmark_returns.std() * np.sqrt(self.trading_days_year)

        # Downside risk
        port_downside = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(self.trading_days_year)
        bench_downside = benchmark_returns[benchmark_returns < 0].std() * np.sqrt(self.trading_days_year)

        # VaR metrics
        port_var95 = np.percentile(portfolio_returns, 5)
        bench_var95 = np.percentile(benchmark_returns, 5)

        # Drawdown analysis
        port_dd = self._calculate_drawdown_series(portfolio_returns)
        bench_dd = self._calculate_drawdown_series(benchmark_returns)

        risk_comparison = {
            'volatility': {
                'portfolio': port_vol,
                'benchmark': bench_vol,
                'difference': port_vol - bench_vol,
                'ratio': port_vol / bench_vol if bench_vol != 0 else np.inf
            },
            'downside_risk': {
                'portfolio': port_downside,
                'benchmark': bench_downside,
                'difference': port_downside - bench_downside
            },
            'var_95': {
                'portfolio': port_var95,
                'benchmark': bench_var95,
                'difference': port_var95 - bench_var95
            },
            'max_drawdown': {
                'portfolio': port_dd.min(),
                'benchmark': bench_dd.min(),
                'difference': port_dd.min() - bench_dd.min()
            },
            'skewness': {
                'portfolio': stats.skew(portfolio_returns),
                'benchmark': stats.skew(benchmark_returns)
            },
            'kurtosis': {
                'portfolio': stats.kurtosis(portfolio_returns),
                'benchmark': stats.kurtosis(benchmark_returns)
            }
        }

        return risk_comparison

    def _calculate_risk_adjusted_comparison(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate risk-adjusted performance comparison."""

        # Sharpe ratios
        port_excess = portfolio_returns.mean() - self.risk_free_rate / self.trading_days_year
        bench_excess = benchmark_returns.mean() - self.risk_free_rate / self.trading_days_year

        port_sharpe = port_excess / portfolio_returns.std() * np.sqrt(self.trading_days_year)
        bench_sharpe = bench_excess / benchmark_returns.std() * np.sqrt(self.trading_days_year)

        # Sortino ratios
        port_downside = portfolio_returns[portfolio_returns < self.risk_free_rate / self.trading_days_year].std()
        bench_downside = benchmark_returns[benchmark_returns < self.risk_free_rate / self.trading_days_year].std()

        port_sortino = port_excess / port_downside * np.sqrt(self.trading_days_year) if port_downside != 0 else np.inf
        bench_sortino = bench_excess / bench_downside * np.sqrt(
            self.trading_days_year) if bench_downside != 0 else np.inf

        # Calmar ratios
        port_calmar = (portfolio_returns.mean() * self.trading_days_year) / abs(
            self._calculate_max_drawdown(portfolio_returns))
        bench_calmar = (benchmark_returns.mean() * self.trading_days_year) / abs(
            self._calculate_max_drawdown(benchmark_returns))

        # Information ratio
        active_returns = portfolio_returns - benchmark_returns
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(
            self.trading_days_year) if active_returns.std() != 0 else 0

        return {
            'sharpe_ratio': {
                'portfolio': port_sharpe,
                'benchmark': bench_sharpe,
                'difference': port_sharpe - bench_sharpe
            },
            'sortino_ratio': {
                'portfolio': port_sortino,
                'benchmark': bench_sortino,
                'difference': port_sortino - bench_sortino
            },
            'calmar_ratio': {
                'portfolio': port_calmar,
                'benchmark': bench_calmar,
                'difference': port_calmar - bench_calmar
            },
            'information_ratio': information_ratio,
            'tracking_error': active_returns.std() * np.sqrt(self.trading_days_year)
        }

    def _calculate_regression_analysis(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate regression-based analysis (Alpha, Beta, etc.)."""

        # Linear regression: Portfolio = Alpha + Beta * Benchmark + Error
        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(benchmark_returns, portfolio_returns)

        # Beta is the slope
        beta = slope

        # Alpha is the intercept annualized
        alpha = intercept * self.trading_days_year

        # R-squared
        r_squared = r_value ** 2

        # Tracking error
        predicted_returns = alpha / self.trading_days_year + beta * benchmark_returns
        tracking_error = (portfolio_returns - predicted_returns).std() * np.sqrt(self.trading_days_year)

        # Treynor ratio
        port_excess = portfolio_returns.mean() * self.trading_days_year - self.risk_free_rate
        treynor_ratio = port_excess / beta if beta != 0 else np.inf

        # Jensen's Alpha (same as regression alpha)
        jensens_alpha = alpha

        # Bull/Bear beta analysis
        bull_periods = benchmark_returns > benchmark_returns.median()
        bear_periods = benchmark_returns <= benchmark_returns.median()

        if bull_periods.sum() > 10 and bear_periods.sum() > 10:
            bull_beta = linregress(benchmark_returns[bull_periods], portfolio_returns[bull_periods])[0]
            bear_beta = linregress(benchmark_returns[bear_periods], portfolio_returns[bear_periods])[0]
        else:
            bull_beta = beta
            bear_beta = beta

        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'treynor_ratio': treynor_ratio,
            'jensens_alpha': jensens_alpha,
            'bull_beta': bull_beta,
            'bear_beta': bear_beta,
            'regression_stats': {
                'p_value': p_value,
                'standard_error': std_err,
                'correlation': r_value
            }
        }

    def _calculate_capture_ratios(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate up/down capture ratios."""

        # Define up and down markets
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0

        # Calculate capture ratios
        if up_market.sum() > 0:
            up_portfolio = portfolio_returns[up_market].mean()
            up_benchmark = benchmark_returns[up_market].mean()
            up_capture = (up_portfolio / up_benchmark) if up_benchmark != 0 else 1
        else:
            up_capture = 1

        if down_market.sum() > 0:
            down_portfolio = portfolio_returns[down_market].mean()
            down_benchmark = benchmark_returns[down_market].mean()
            down_capture = (down_portfolio / down_benchmark) if down_benchmark != 0 else 1
        else:
            down_capture = 1

        # Capture ratio (up capture / down capture)
        capture_ratio = up_capture / abs(down_capture) if down_capture != 0 else np.inf

        # Alternative calculation using different thresholds
        threshold_5pct = np.percentile(benchmark_returns, 5)
        threshold_95pct = np.percentile(benchmark_returns, 95)

        extreme_up = benchmark_returns > threshold_95pct
        extreme_down = benchmark_returns < threshold_5pct

        if extreme_up.sum() > 0:
            extreme_up_capture = portfolio_returns[extreme_up].mean() / benchmark_returns[extreme_up].mean()
        else:
            extreme_up_capture = up_capture

        if extreme_down.sum() > 0:
            extreme_down_capture = portfolio_returns[extreme_down].mean() / benchmark_returns[extreme_down].mean()
        else:
            extreme_down_capture = down_capture

        return {
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'capture_ratio': capture_ratio,
            'extreme_up_capture': extreme_up_capture,
            'extreme_down_capture': extreme_down_capture,
            'up_market_periods': up_market.sum(),
            'down_market_periods': down_market.sum(),
            'up_market_frequency': up_market.mean(),
            'down_market_frequency': down_market.mean()
        }

    def _calculate_performance_attribution(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate performance attribution analysis."""

        active_returns = portfolio_returns - benchmark_returns

        # Basic attribution
        total_return_portfolio = (1 + portfolio_returns).prod() - 1
        total_return_benchmark = (1 + benchmark_returns).prod() - 1
        total_active_return = total_return_portfolio - total_return_benchmark

        # Timing and selection (simplified analysis)
        # This is a basic implementation - real attribution would require holdings data

        # Selection effect (assuming no timing, just stock selection)
        selection_effect = active_returns.mean() * self.trading_days_year

        # Volatility attribution
        port_vol = portfolio_returns.std() * np.sqrt(self.trading_days_year)
        bench_vol = benchmark_returns.std() * np.sqrt(self.trading_days_year)
        vol_effect = port_vol - bench_vol

        # Correlation effect
        correlation = portfolio_returns.corr(benchmark_returns)

        return {
            'total_active_return': total_active_return,
            'annualized_active_return': selection_effect,
            'selection_effect': selection_effect,  # Simplified
            'volatility_effect': vol_effect,
            'correlation_with_benchmark': correlation,
            'attribution_breakdown': {
                'asset_selection': selection_effect * 0.7,  # Rough allocation
                'market_timing': selection_effect * 0.3,  # Rough allocation
                'interaction_effect': 0  # Simplified
            }
        }

    def _calculate_rolling_analysis(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series,
            windows: List[int] = [63, 126, 252]  # Quarter, half-year, year
    ) -> Dict:
        """Calculate rolling analysis metrics."""

        rolling_analysis = {}

        for window in windows:
            if len(portfolio_returns) >= window:
                # Rolling returns
                port_rolling = portfolio_returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
                bench_rolling = benchmark_returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
                active_rolling = port_rolling - bench_rolling

                # Rolling volatility
                port_vol_rolling = portfolio_returns.rolling(window).std() * np.sqrt(self.trading_days_year)
                bench_vol_rolling = benchmark_returns.rolling(window).std() * np.sqrt(self.trading_days_year)

                # Rolling Sharpe ratio
                port_sharpe_rolling = (port_rolling.rolling(
                    window).mean() * self.trading_days_year - self.risk_free_rate) / port_vol_rolling
                bench_sharpe_rolling = (bench_rolling.rolling(
                    window).mean() * self.trading_days_year - self.risk_free_rate) / bench_vol_rolling

                # Rolling beta
                rolling_beta = portfolio_returns.rolling(window).cov(benchmark_returns) / benchmark_returns.rolling(
                    window).var()

                window_days = f"{window}d"
                rolling_analysis[window_days] = {
                    'current_excess_return': active_rolling.iloc[-1] if not active_rolling.empty else 0,
                    'average_excess_return': active_rolling.mean(),
                    'excess_return_volatility': active_rolling.std(),
                    'best_period': active_rolling.max(),
                    'worst_period': active_rolling.min(),
                    'outperformance_frequency': (active_rolling > 0).mean(),
                    'current_beta': rolling_beta.iloc[-1] if not rolling_beta.empty else 1,
                    'average_beta': rolling_beta.mean(),
                    'beta_stability': rolling_beta.std()
                }

        return rolling_analysis

    def _perform_statistical_tests(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series
    ) -> Dict:
        """Perform statistical tests on returns."""

        active_returns = portfolio_returns - benchmark_returns

        statistical_tests = {}

        # T-test for mean active return
        if len(active_returns) > 1:
            t_stat, t_pvalue = stats.ttest_1samp(active_returns, 0)
            statistical_tests['mean_excess_return_ttest'] = {
                'statistic': t_stat,
                'p_value': t_pvalue,
                'significant': t_pvalue < 0.05
            }

        # Jarque-Bera test for normality
        if len(portfolio_returns) > 8:
            jb_stat_port, jb_pvalue_port = stats.jarque_bera(portfolio_returns)
            jb_stat_bench, jb_pvalue_bench = stats.jarque_bera(benchmark_returns)

            statistical_tests['normality_test'] = {
                'portfolio': {
                    'statistic': jb_stat_port,
                    'p_value': jb_pvalue_port,
                    'is_normal': jb_pvalue_port > 0.05
                },
                'benchmark': {
                    'statistic': jb_stat_bench,
                    'p_value': jb_pvalue_bench,
                    'is_normal': jb_pvalue_bench > 0.05
                }
            }

        # Kolmogorov-Smirnov test for distribution comparison
        if len(portfolio_returns) > 8:
            ks_stat, ks_pvalue = stats.ks_2samp(portfolio_returns, benchmark_returns)
            statistical_tests['distribution_comparison'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'distributions_different': ks_pvalue < 0.05
            }

        # Autocorrelation test (Ljung-Box)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat_port = acorr_ljungbox(portfolio_returns, lags=10, return_df=True)
            lb_stat_bench = acorr_ljungbox(benchmark_returns, lags=10, return_df=True)

            statistical_tests['autocorrelation_test'] = {
                'portfolio_ljung_box_pvalue': lb_stat_port['lb_pvalue'].iloc[-1],
                'benchmark_ljung_box_pvalue': lb_stat_bench['lb_pvalue'].iloc[-1],
                'portfolio_has_autocorr': lb_stat_port['lb_pvalue'].iloc[-1] < 0.05,
                'benchmark_has_autocorr': lb_stat_bench['lb_pvalue'].iloc[-1] < 0.05
            }
        except ImportError:
            # Fallback if statsmodels not available
            pass

        return statistical_tests

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown time series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def generate_benchmark_report(
            self,
            portfolio_returns: pd.Series,
            benchmark_returns: pd.Series,
            portfolio_name: str = "Portfolio",
            benchmark_name: str = "Benchmark"
    ) -> str:
        """Generate a text-based benchmark comparison report."""

        comparison = self.compare_to_benchmark(
            portfolio_returns, benchmark_returns, portfolio_name, benchmark_name
        )

        if not comparison:
            return "Insufficient data for benchmark comparison."

        report = f"""
BENCHMARK COMPARISON REPORT
{'=' * 50}

PORTFOLIO: {portfolio_name}
BENCHMARK: {benchmark_name}
ANALYSIS PERIOD: {portfolio_returns.index[0].strftime('%Y-%m-%d')} to {portfolio_returns.index[-1].strftime('%Y-%m-%d')}

PERFORMANCE SUMMARY
{'-' * 20}
                        Portfolio    Benchmark    Difference
Total Return           {comparison['summary'][portfolio_name]['total_return']:>8.2%}     {comparison['summary'][benchmark_name]['total_return']:>8.2%}     {comparison['summary']['outperformance']['annualized_excess_return']:>8.2%}
Annualized Return      {comparison['summary'][portfolio_name]['annualized_return']:>8.2%}     {comparison['summary'][benchmark_name]['annualized_return']:>8.2%}     {comparison['summary']['outperformance']['annualized_excess_return']:>8.2%}
Volatility             {comparison['summary'][portfolio_name]['volatility']:>8.2%}     {comparison['summary'][benchmark_name]['volatility']:>8.2%}     {comparison['summary']['outperformance']['excess_volatility']:>8.2%}
Sharpe Ratio           {comparison['summary'][portfolio_name]['sharpe_ratio']:>8.2f}     {comparison['summary'][benchmark_name]['sharpe_ratio']:>8.2f}     {comparison['summary']['outperformance']['sharpe_difference']:>8.2f}
Max Drawdown           {comparison['summary'][portfolio_name]['max_drawdown']:>8.2%}     {comparison['summary'][benchmark_name]['max_drawdown']:>8.2%}     {comparison['summary']['outperformance']['relative_max_drawdown']:>8.2%}

RISK-ADJUSTED METRICS
{'-' * 22}
Information Ratio:     {comparison['risk_adjusted']['information_ratio']:>8.2f}
Tracking Error:        {comparison['risk_adjusted']['tracking_error']:>8.2%}
Beta:                  {comparison['regression']['beta']:>8.2f}
Alpha:                 {comparison['regression']['alpha']:>8.2%}
R-Squared:             {comparison['regression']['r_squared']:>8.2%}

CAPTURE RATIOS
{'-' * 14}
Up Capture:            {comparison['capture_ratios']['up_capture_ratio']:>8.2%}
Down Capture:          {comparison['capture_ratios']['down_capture_ratio']:>8.2%}
Overall Capture:       {comparison['capture_ratios']['capture_ratio']:>8.2f}

STATISTICAL SIGNIFICANCE
{'-' * 24}
"""

        if 'mean_excess_return_ttest' in comparison['statistical_tests']:
            significance = comparison['statistical_tests']['mean_excess_return_ttest']
            report += f"Alpha Significance:    {'Significant' if significance['significant'] else 'Not Significant'} (p-value: {significance['p_value']:.4f})\n"

        return report

    def calculate_multiple_benchmark_comparison(
            self,
            portfolio_returns: pd.Series,
            benchmark_dict: Dict[str, pd.Series]
    ) -> Dict[str, Dict]:
        """Compare portfolio against multiple benchmarks."""

        comparisons = {}

        for benchmark_name, benchmark_returns in benchmark_dict.items():
            comparisons[benchmark_name] = self.compare_to_benchmark(
                portfolio_returns,
                benchmark_returns,
                "Portfolio",
                benchmark_name
            )

        # Summary comparison across all benchmarks
        summary = {
            'best_benchmark': None,
            'worst_benchmark': None,
            'average_outperformance': 0,
            'consistency_score': 0
        }

        if comparisons:
            # Find best performing benchmark to compare against
            benchmark_returns = {name: comp['summary'][name]['annualized_return']
                                 for name, comp in comparisons.items() if comp}

            if benchmark_returns:
                summary['best_benchmark'] = max(benchmark_returns, key=benchmark_returns.get)
                summary['worst_benchmark'] = min(benchmark_returns, key=benchmark_returns.get)

                # Calculate average outperformance
                outperformances = [comp['summary']['outperformance']['annualized_excess_return']
                                   for comp in comparisons.values() if comp]
                summary['average_outperformance'] = np.mean(outperformances) if outperformances else 0

                # Consistency score (percentage of benchmarks outperformed)
                positive_outperformances = sum(1 for x in outperformances if x > 0)
                summary['consistency_score'] = positive_outperformances / len(outperformances) if outperformances else 0

        comparisons['summary'] = summary
        return comparisons
        '