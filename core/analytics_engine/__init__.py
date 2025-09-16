"""
Analytics Engine Package
Comprehensive portfolio analytics and performance measurement tools.

This package provides advanced analytics capabilities including:
- Performance calculations (70+ metrics)
- Risk analysis and VaR calculations
- Benchmark comparison and analysis
- Advanced charting and visualization
"""

from .performance_calculator import PerformanceCalculator
from .risk_calculator import RiskCalculator
from .benchmark_analyzer import BenchmarkAnalyzer
from .chart_generator import ChartGenerator

__all__ = [
    'PerformanceCalculator',
    'RiskCalculator',
    'BenchmarkAnalyzer',
    'ChartGenerator'
]

__version__ = '1.0.0'


class AnalyticsEngine:
    """
    Main analytics engine class that combines all analytics components.

    This class provides a unified interface to all analytics functionality,
    making it easy to perform comprehensive portfolio analysis.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the analytics engine.

        Args:
            risk_free_rate: Annual risk-free rate for calculations (default 2%)
        """
        self.risk_free_rate = risk_free_rate

        # Initialize all components
        self.performance = PerformanceCalculator(risk_free_rate=risk_free_rate)
        self.risk = RiskCalculator()
        self.benchmark = BenchmarkAnalyzer(risk_free_rate=risk_free_rate)
        self.charts = ChartGenerator()

    def analyze_portfolio(
            self,
            returns,
            benchmark_returns=None,
            portfolio_value=None,
            portfolio_name="Portfolio"
    ):
        """
        Perform comprehensive portfolio analysis.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns for comparison
            portfolio_value: Optional portfolio value series
            portfolio_name: Name for the portfolio

        Returns:
            Dictionary with comprehensive analysis results
        """
        analysis = {
            'portfolio_name': portfolio_name,
            'analysis_period': {
                'start_date': returns.index[0] if len(returns) > 0 else None,
                'end_date': returns.index[-1] if len(returns) > 0 else None,
                'periods': len(returns)
            }
        }

        # Performance analysis
        analysis['performance'] = self.performance.calculate_all_metrics(
            returns, benchmark_returns, portfolio_value
        )

        # Risk analysis
        analysis['risk'] = self.risk.calculate_risk_metrics(returns)
        analysis['var_analysis'] = self.risk.calculate_var_all_methods(returns)
        analysis['risk_adjusted'] = self.risk.calculate_risk_adjusted_performance(
            returns, benchmark_returns
        )

        # Benchmark comparison (if benchmark provided)
        if benchmark_returns is not None:
            analysis['benchmark_comparison'] = self.benchmark.compare_to_benchmark(
                returns, benchmark_returns, portfolio_name, "Benchmark"
            )

        # Summary statistics
        analysis['summary'] = self.performance.get_performance_summary(returns)

        return analysis

    def create_analysis_charts(self, returns, benchmark_returns=None, **kwargs):
        """
        Create comprehensive set of analysis charts.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns
            **kwargs: Additional parameters for chart customization

        Returns:
            Dictionary of plotly figure objects
        """
        charts = {}

        # Performance charts
        if benchmark_returns is not None:
            returns_data = {'Portfolio': returns, 'Benchmark': benchmark_returns}
            charts['performance_comparison'] = self.charts.create_performance_comparison_chart(returns_data)
        else:
            returns_data = {'Portfolio': returns}
            charts['performance_comparison'] = self.charts.create_performance_comparison_chart(returns_data)

        # Drawdown chart
        charts['drawdown'] = self.charts.create_drawdown_underwater_chart(returns)

        # Return distribution
        charts['return_distribution'] = self.charts.create_return_distribution_chart(returns)

        # Rolling metrics
        charts['rolling_metrics'] = self.charts.create_rolling_metrics_chart(returns)

        # Monthly returns heatmap (if enough data)
        if len(returns) > 60:  # At least 2-3 months of daily data
            charts['monthly_heatmap'] = self.charts.create_monthly_returns_heatmap(returns)

        return charts

    def generate_report(
            self,
            returns,
            benchmark_returns=None,
            portfolio_name="Portfolio",
            format="dict"
    ):
        """
        Generate comprehensive portfolio analysis report.

        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns
            portfolio_name: Name for the portfolio
            format: Output format ("dict", "text", "html")

        Returns:
            Report in specified format
        """
        # Perform analysis
        analysis = self.analyze_portfolio(
            returns, benchmark_returns, portfolio_name=portfolio_name
        )

        if format == "dict":
            return analysis

        elif format == "text":
            return self._generate_text_report(analysis)

        elif format == "html":
            return self._generate_html_report(analysis)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_text_report(self, analysis):
        """Generate text-based report."""

        report = f"""
PORTFOLIO ANALYSIS REPORT
{'=' * 50}

Portfolio: {analysis['portfolio_name']}
Period: {analysis['analysis_period']['start_date']} to {analysis['analysis_period']['end_date']}
Observations: {analysis['analysis_period']['periods']}

PERFORMANCE SUMMARY
{'-' * 19}
Total Return:          {analysis['performance']['total_return']:>10.2%}
Annualized Return:     {analysis['performance']['annualized_return']:>10.2%}
Volatility:            {analysis['performance']['volatility']:>10.2%}
Sharpe Ratio:          {analysis['performance']['sharpe_ratio']:>10.2f}
Max Drawdown:          {analysis['risk']['max_drawdown']:>10.2%}

RISK METRICS
{'-' * 12}
VaR (95%):             {analysis['var_analysis']['historical']:>10.2%}
VaR (99%):             {analysis['risk']['var_99']:>10.2%}
Expected Shortfall:    {analysis['var_analysis']['expected_shortfall_historical']:>10.2%}
Downside Deviation:    {analysis['risk']['downside_deviation']:>10.2%}
Skewness:              {analysis['risk']['skewness']:>10.2f}
Kurtosis:              {analysis['risk']['kurtosis']:>10.2f}

RISK-ADJUSTED RATIOS
{'-' * 20}
Sortino Ratio:         {analysis['risk_adjusted']['sortino_ratio']:>10.2f}
Calmar Ratio:          {analysis['risk_adjusted']['calmar_ratio']:>10.2f}
Omega Ratio:           {analysis['risk_adjusted']['omega_ratio']:>10.2f}
"""

        # Add benchmark comparison if available
        if 'benchmark_comparison' in analysis:
            bench_data = analysis['benchmark_comparison']
            report += f"""
BENCHMARK COMPARISON
{'-' * 20}
Alpha:                 {bench_data['regression']['alpha']:>10.2%}
Beta:                  {bench_data['regression']['beta']:>10.2f}
R-Squared:             {bench_data['regression']['r_squared']:>10.2%}
Information Ratio:     {bench_data['risk_adjusted']['information_ratio']:>10.2f}
Tracking Error:        {bench_data['risk_adjusted']['tracking_error']:>10.2%}
Up Capture:            {bench_data['capture_ratios']['up_capture_ratio']:>10.2%}
Down Capture:          {bench_data['capture_ratios']['down_capture_ratio']:>10.2%}
"""

        return report

    def _generate_html_report(self, analysis):
        """Generate HTML-based report."""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; }}
        .metric-table {{ border-collapse: collapse; width: 100%; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        .metric-table th {{ background-color: #f8f9fa; }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
    </style>
</head>
<body>
    <h1 class="header">Portfolio Analysis Report</h1>

    <div class="section">
        <h2>Portfolio Information</h2>
        <p><strong>Portfolio:</strong> {analysis['portfolio_name']}</p>
        <p><strong>Analysis Period:</strong> {analysis['analysis_period']['start_date']} to {analysis['analysis_period']['end_date']}</p>
        <p><strong>Observations:</strong> {analysis['analysis_period']['periods']}</p>
    </div>

    <div class="section">
        <h2>Performance Summary</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Return</td><td class="{'positive' if analysis['performance']['total_return'] > 0 else 'negative'}">{analysis['performance']['total_return']:.2%}</td></tr>
            <tr><td>Annualized Return</td><td class="{'positive' if analysis['performance']['annualized_return'] > 0 else 'negative'}">{analysis['performance']['annualized_return']:.2%}</td></tr>
            <tr><td>Volatility</td><td>{analysis['performance']['volatility']:.2%}</td></tr>
            <tr><td>Sharpe Ratio</td><td>{analysis['performance']['sharpe_ratio']:.2f}</td></tr>
            <tr><td>Max Drawdown</td><td class="negative">{analysis['risk']['max_drawdown']:.2%}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Risk Metrics</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>VaR (95%)</td><td class="negative">{analysis['var_analysis']['historical']:.2%}</td></tr>
            <tr><td>VaR (99%)</td><td class="negative">{analysis['risk']['var_99']:.2%}</td></tr>
            <tr><td>Expected Shortfall</td><td class="negative">{analysis['var_analysis']['expected_shortfall_historical']:.2%}</td></tr>
            <tr><td>Downside Deviation</td><td>{analysis['risk']['downside_deviation']:.2%}</td></tr>
            <tr><td>Skewness</td><td>{analysis['risk']['skewness']:.2f}</td></tr>
            <tr><td>Kurtosis</td><td>{analysis['risk']['kurtosis']:.2f}</td></tr>
        </table>
    </div>
"""

        # Add benchmark comparison if available
        if 'benchmark_comparison' in analysis:
            bench_data = analysis['benchmark_comparison']
            html += f"""
    <div class="section">
        <h2>Benchmark Comparison</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Alpha</td><td class="{'positive' if bench_data['regression']['alpha'] > 0 else 'negative'}">{bench_data['regression']['alpha']:.2%}</td></tr>
            <tr><td>Beta</td><td>{bench_data['regression']['beta']:.2f}</td></tr>
            <tr><td>R-Squared</td><td>{bench_data['regression']['r_squared']:.2%}</td></tr>
            <tr><td>Information Ratio</td><td>{bench_data['risk_adjusted']['information_ratio']:.2f}</td></tr>
            <tr><td>Tracking Error</td><td>{bench_data['risk_adjusted']['tracking_error']:.2%}</td></tr>
            <tr><td>Up Capture</td><td>{bench_data['capture_ratios']['up_capture_ratio']:.2%}</td></tr>
            <tr><td>Down Capture</td><td>{bench_data['capture_ratios']['down_capture_ratio']:.2%}</td></tr>
        </table>
    </div>
"""

        html += """
</body>
</html>
"""
        return html


# Convenience function for quick analysis
def analyze_portfolio_returns(
        returns,
        benchmark_returns=None,
        risk_free_rate=0.02,
        portfolio_name="Portfolio"
):
    """
    Quick portfolio analysis function.

    Args:
        returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        portfolio_name: Name for the portfolio

    Returns:
        Comprehensive analysis dictionary
    """
    engine = AnalyticsEngine(risk_free_rate=risk_free_rate)
    return engine.analyze_portfolio(
        returns=returns,
        benchmark_returns=benchmark_returns,
        portfolio_name=portfolio_name
    )