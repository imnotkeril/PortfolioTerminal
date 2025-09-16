"""
Risk Calculator Module
Advanced risk analysis and Value at Risk (VaR) calculations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class RiskCalculator:
    """
    Comprehensive risk analysis engine.

    Provides multiple VaR methodologies, stress testing, and advanced risk metrics.
    """

    def __init__(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]):
        """
        Initialize risk calculator.

        Args:
            confidence_levels: List of confidence levels for VaR calculation
        """
        self.confidence_levels = confidence_levels
        self.trading_days_year = 252

    def calculate_var_all_methods(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1
    ) -> Dict[str, float]:
        """
        Calculate VaR using all available methods.

        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level for VaR
            holding_period: Holding period in days

        Returns:
            Dictionary with VaR values from different methods
        """
        var_results = {}

        # Historical VaR
        var_results['historical'] = self.historical_var(returns, confidence_level, holding_period)

        # Parametric VaR (Normal distribution)
        var_results['parametric_normal'] = self.parametric_var(returns, confidence_level, holding_period)

        # Parametric VaR (Student's t-distribution)
        var_results['parametric_t'] = self.parametric_var_t(returns, confidence_level, holding_period)

        # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
        var_results['cornish_fisher'] = self.cornish_fisher_var(returns, confidence_level, holding_period)

        # Monte Carlo VaR
        var_results['monte_carlo'] = self.monte_carlo_var(returns, confidence_level, holding_period)

        # Filtered Historical Simulation
        var_results['filtered_hs'] = self.filtered_historical_simulation_var(returns, confidence_level, holding_period)

        # Expected Shortfall (Conditional VaR) for each method
        var_results['expected_shortfall_historical'] = self.expected_shortfall(returns, confidence_level, 'historical')
        var_results['expected_shortfall_parametric'] = self.expected_shortfall(returns, confidence_level, 'parametric')

        return var_results

    def historical_var(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1
    ) -> float:
        """Calculate historical VaR."""
        if len(returns) == 0:
            return np.nan

        # Scale returns for holding period
        if holding_period > 1:
            scaled_returns = returns.rolling(holding_period).sum().dropna()
        else:
            scaled_returns = returns

        # Calculate VaR as percentile
        var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        return var

    def parametric_var(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1
    ) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if len(returns) == 0:
            return np.nan

        mean = returns.mean()
        std = returns.std()

        # Scale for holding period
        mean_scaled = mean * holding_period
        std_scaled = std * np.sqrt(holding_period)

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean_scaled + z_score * std_scaled

        return var

    def parametric_var_t(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1
    ) -> float:
        """Calculate parametric VaR using Student's t-distribution."""
        if len(returns) == 0:
            return np.nan

        # Fit t-distribution to returns
        try:
            params = stats.t.fit(returns)
            df, loc, scale = params

            # Scale for holding period
            mean_scaled = loc * holding_period
            scale_scaled = scale * np.sqrt(holding_period)

            # Calculate VaR using t-distribution
            t_score = stats.t.ppf(1 - confidence_level, df)
            var = mean_scaled + t_score * scale_scaled

            return var
        except:
            # Fallback to normal distribution if fitting fails
            return self.parametric_var(returns, confidence_level, holding_period)

    def cornish_fisher_var(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1
    ) -> float:
        """Calculate Cornish-Fisher VaR (adjusts for skewness and kurtosis)."""
        if len(returns) == 0:
            return np.nan

        mean = returns.mean()
        std = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Scale for holding period
        mean_scaled = mean * holding_period
        std_scaled = std * np.sqrt(holding_period)

        # Normal quantile
        z = stats.norm.ppf(1 - confidence_level)

        # Cornish-Fisher expansion
        z_cf = (z +
                (z ** 2 - 1) * skewness / 6 +
                (z ** 3 - 3 * z) * kurtosis / 24 -
                (2 * z ** 3 - 5 * z) * (skewness ** 2) / 36)

        var = mean_scaled + z_cf * std_scaled
        return var

    def monte_carlo_var(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1,
            n_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR."""
        if len(returns) == 0:
            return np.nan

        # Fit distribution to returns (using normal for simplicity)
        mean = returns.mean()
        std = returns.std()

        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean, std, n_simulations * holding_period)

        # Calculate holding period returns
        if holding_period > 1:
            simulated_returns = simulated_returns.reshape(n_simulations, holding_period).sum(axis=1)

        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return var

    def filtered_historical_simulation_var(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1,
            lambda_decay: float = 0.94
    ) -> float:
        """Calculate Filtered Historical Simulation VaR with EWMA volatility."""
        if len(returns) == 0:
            return np.nan

        # Calculate EWMA volatility
        ewma_var = returns.ewm(alpha=1 - lambda_decay).var()
        current_vol = np.sqrt(ewma_var.iloc[-1])

        # Standardize historical returns
        historical_vol = returns.rolling(252).std().iloc[-1] if len(returns) >= 252 else returns.std()
        if historical_vol == 0:
            return np.nan

        standardized_returns = returns / historical_vol

        # Scale by current volatility
        scaled_returns = standardized_returns * current_vol

        # Calculate VaR
        if holding_period > 1:
            scaled_returns = scaled_returns.rolling(holding_period).sum().dropna()

        var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        return var

    def expected_shortfall(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            method: str = 'historical'
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return np.nan

        if method == 'historical':
            var = self.historical_var(returns, confidence_level)
            # Expected shortfall is the mean of returns below VaR
            tail_returns = returns[returns <= var]
            return tail_returns.mean() if len(tail_returns) > 0 else var

        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            z = stats.norm.ppf(1 - confidence_level)

            # Expected shortfall for normal distribution
            es = mean - std * stats.norm.pdf(z) / (1 - confidence_level)
            return es

        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        metrics = {}

        if len(returns) == 0:
            return metrics

        # Basic risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(self.trading_days_year)
        metrics['daily_volatility'] = returns.std()

        # Downside risk metrics
        negative_returns = returns[returns < 0]
        metrics['downside_volatility'] = negative_returns.std() * np.sqrt(self.trading_days_year) if len(
            negative_returns) > 0 else 0

        target_return = 0  # Can be customized
        downside_deviations = returns[returns < target_return] - target_return
        metrics['downside_deviation'] = np.sqrt(np.mean(downside_deviations ** 2)) * np.sqrt(
            self.trading_days_year) if len(downside_deviations) > 0 else 0

        # Value at Risk for multiple confidence levels
        for conf_level in self.confidence_levels:
            metrics[f'var_{int(conf_level * 100)}'] = self.historical_var(returns, conf_level)
            metrics[f'cvar_{int(conf_level * 100)}'] = self.expected_shortfall(returns, conf_level)

        # Skewness and Kurtosis
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        metrics['excess_kurtosis'] = metrics['kurtosis']

        # Jarque-Bera test for normality
        if len(returns) > 8:  # Minimum sample size for JB test
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
            metrics['jarque_bera_stat'] = jb_stat
            metrics['jarque_bera_pvalue'] = jb_pvalue

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()

        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        metrics['average_drawdown'] = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0

        # Drawdown duration analysis
        in_drawdown = drawdown < -0.001  # 0.1% threshold
        drawdown_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        if drawdown_periods:
            metrics['max_drawdown_duration'] = max(drawdown_periods)
            metrics['average_drawdown_duration'] = np.mean(drawdown_periods)
        else:
            metrics['max_drawdown_duration'] = 0
            metrics['average_drawdown_duration'] = 0

        # Ulcer Index
        drawdown_squared = drawdown ** 2
        metrics['ulcer_index'] = np.sqrt(drawdown_squared.mean())

        # Pain Index (average of all drawdowns)
        metrics['pain_index'] = abs(drawdown).mean()

        # Tail ratios
        positive_returns = returns[returns > 0]
        metrics['gain_to_pain_ratio'] = (positive_returns.sum() / abs(negative_returns.sum())) if len(
            negative_returns) > 0 and negative_returns.sum() != 0 else np.inf

        # Consecutive losses
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

    def stress_test_portfolio(
            self,
            returns: pd.Series,
            scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing on portfolio.

        Args:
            returns: Portfolio returns
            scenarios: Dictionary of stress scenarios

        Returns:
            Dictionary with stress test results
        """
        results = {}

        for scenario_name, scenario_params in scenarios.items():
            scenario_results = {}

            # Apply stress scenario
            if 'market_shock' in scenario_params:
                # Apply market shock
                shocked_returns = returns.copy()
                shock_magnitude = scenario_params['market_shock']
                shocked_returns.iloc[-1] = shock_magnitude

                # Calculate impact
                original_value = (1 + returns).prod()
                shocked_value = (1 + shocked_returns).prod()
                scenario_results['portfolio_impact'] = (shocked_value - original_value) / original_value

            if 'volatility_shock' in scenario_params:
                # Apply volatility shock
                vol_multiplier = scenario_params['volatility_shock']
                current_vol = returns.std()
                new_vol = current_vol * vol_multiplier

                # Scale returns to new volatility
                scaled_returns = returns * (new_vol / current_vol)
                scenario_results['new_volatility'] = new_vol * np.sqrt(self.trading_days_year)
                scenario_results['var_95_new'] = self.historical_var(scaled_returns, 0.95)

            if 'correlation_shock' in scenario_params:
                # This would require multi-asset analysis
                # Placeholder for correlation stress testing
                scenario_results['correlation_impact'] = scenario_params['correlation_shock']

            results[scenario_name] = scenario_results

        return results

    def calculate_portfolio_beta(
            self,
            portfolio_returns: pd.Series,
            market_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate portfolio beta and related metrics."""

        # Align the series
        aligned_data = pd.concat([portfolio_returns, market_returns], axis=1, keys=['portfolio', 'market']).dropna()

        if len(aligned_data) < 30:  # Need sufficient data
            return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}

        port_ret = aligned_data['portfolio']
        mkt_ret = aligned_data['market']

        # Calculate beta using covariance method
        covariance = np.cov(port_ret, mkt_ret)[0, 1]
        market_variance = np.var(mkt_ret)
        beta = covariance / market_variance if market_variance != 0 else np.nan

        # Calculate alpha (Jensen's alpha)
        port_mean = port_ret.mean() * self.trading_days_year
        mkt_mean = mkt_ret.mean() * self.trading_days_year
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        alpha = port_mean - (risk_free_rate + beta * (mkt_mean - risk_free_rate))

        # Calculate R-squared
        correlation = port_ret.corr(mkt_ret)
        r_squared = correlation ** 2

        # Calculate tracking error
        active_returns = port_ret - mkt_ret
        tracking_error = active_returns.std() * np.sqrt(self.trading_days_year)

        # Calculate information ratio
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(
            self.trading_days_year) if active_returns.std() != 0 else np.nan

        return {
            'beta': beta,
            'alpha': alpha,
            'r_squared': r_squared,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }

    def calculate_risk_adjusted_performance(
            self,
            returns: pd.Series,
            benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""

        metrics = {}

        if len(returns) == 0:
            return metrics

        risk_free_rate = 0.02  # Assume 2% annual risk-free rate
        rf_daily = risk_free_rate / self.trading_days_year

        # Sharpe ratio
        excess_returns = returns.mean() - rf_daily
        volatility = returns.std()
        metrics['sharpe_ratio'] = (
                    excess_returns / volatility * np.sqrt(self.trading_days_year)) if volatility != 0 else np.nan

        # Sortino ratio
        negative_returns = returns[returns < rf_daily]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else volatility
        metrics['sortino_ratio'] = (
                    excess_returns / downside_std * np.sqrt(self.trading_days_year)) if downside_std != 0 else np.nan

        # Calmar ratio
        max_drawdown = self.calculate_risk_metrics(returns)['max_drawdown']
        annual_return = returns.mean() * self.trading_days_year
        metrics['calmar_ratio'] = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # Omega ratio
        threshold = rf_daily
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        metrics['omega_ratio'] = gains.sum() / losses.sum() if losses.sum() != 0 else np.inf

        # MAR ratio (Modified Calmar)
        metrics['mar_ratio'] = metrics['calmar_ratio']

        # Sterling ratio
        avg_drawdown = self.calculate_risk_metrics(returns)['average_drawdown']
        metrics['sterling_ratio'] = annual_return / abs(avg_drawdown) if avg_drawdown != 0 else np.inf

        # Burke ratio
        drawdown_series = self._calculate_drawdown_series(returns)
        burke_denominator = np.sqrt(np.sum(drawdown_series ** 2))
        metrics['burke_ratio'] = annual_return / burke_denominator if burke_denominator != 0 else np.inf

        # Treynor ratio (requires benchmark)
        if benchmark_returns is not None:
            beta_metrics = self.calculate_portfolio_beta(returns, benchmark_returns)
            beta = beta_metrics['beta']
            metrics['treynor_ratio'] = excess_returns * self.trading_days_year / beta if beta != 0 and not np.isnan(
                beta) else np.nan

        return metrics

    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def monte_carlo_simulation(
            self,
            returns: pd.Series,
            initial_value: float = 100000,
            time_horizon: int = 252,
            n_simulations: int = 1000,
            method: str = 'normal'
    ) -> np.ndarray:
        """
        Perform Monte Carlo simulation for portfolio value.

        Args:
            returns: Historical returns
            initial_value: Initial portfolio value
            time_horizon: Number of days to simulate
            n_simulations: Number of simulation paths
            method: Distribution method ('normal', 't', 'bootstrap')

        Returns:
            Array of simulation results (n_simulations x time_horizon)
        """
        np.random.seed(42)  # For reproducibility

        if method == 'normal':
            # Parametric approach using normal distribution
            mean = returns.mean()
            std = returns.std()

            # Generate random returns
            random_returns = np.random.normal(mean, std, (n_simulations, time_horizon))

        elif method == 't':
            # Use Student's t-distribution
            try:
                params = stats.t.fit(returns)
                df, loc, scale = params
                random_returns = stats.t.rvs(df, loc, scale, size=(n_simulations, time_horizon))
            except:
                # Fallback to normal
                mean = returns.mean()
                std = returns.std()
                random_returns = np.random.normal(mean, std, (n_simulations, time_horizon))

        elif method == 'bootstrap':
            # Bootstrap from historical returns
            returns_array = returns.values
            random_indices = np.random.choice(len(returns_array), (n_simulations, time_horizon), replace=True)
            random_returns = returns_array[random_indices]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate portfolio values
        portfolio_values = np.zeros((n_simulations, time_horizon + 1))
        portfolio_values[:, 0] = initial_value

        for t in range(time_horizon):
            portfolio_values[:, t + 1] = portfolio_values[:, t] * (1 + random_returns[:, t])

        return portfolio_values

    def calculate_var_backtesting(
            self,
            returns: pd.Series,
            var_estimates: pd.Series,
            confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Perform VaR backtesting using various tests.

        Args:
            returns: Actual returns
            var_estimates: VaR estimates for the same period
            confidence_level: Confidence level used for VaR

        Returns:
            Dictionary with backtesting results
        """
        # Align the series
        aligned_data = pd.concat([returns, var_estimates], axis=1, keys=['returns', 'var']).dropna()
        actual_returns = aligned_data['returns']
        var_values = aligned_data['var']

        # Count violations (returns worse than VaR)
        violations = actual_returns < var_values
        n_violations = violations.sum()
        n_observations = len(actual_returns)

        # Expected number of violations
        expected_violations = n_observations * (1 - confidence_level)
        violation_rate = n_violations / n_observations

        # Kupiec's POF test
        likelihood_ratio = 2 * (
                n_violations * np.log(violation_rate / (1 - confidence_level)) +
                (n_observations - n_violations) * np.log((1 - violation_rate) / confidence_level)
        ) if violation_rate > 0 and violation_rate < 1 else 0

        # Christoffersen's Independence test
        # Count transitions
        v_01 = 0  # No violation followed by violation
        v_10 = 0  # Violation followed by no violation
        v_11 = 0  # Violation followed by violation

        for i in range(1, len(violations)):
            if not violations.iloc[i - 1] and violations.iloc[i]:
                v_01 += 1
            elif violations.iloc[i - 1] and not violations.iloc[i]:
                v_10 += 1
            elif violations.iloc[i - 1] and violations.iloc[i]:
                v_11 += 1

        # Independence test statistic
        n_0 = n_observations - n_violations
        n_1 = n_violations

        if v_01 + v_11 > 0 and v_10 + v_11 > 0:
            pi_01 = v_01 / (v_01 + v_11) if (v_01 + v_11) > 0 else 0
            pi_1 = v_1 / n_observations if n_observations > 0 else 0

            independence_lr = 2 * (
                    v_01 * np.log(pi_01 / pi_1) + v_11 * np.log((1 - pi_01) / (1 - pi_1))
            ) if pi_1 > 0 and pi_1 < 1 and pi_01 > 0 and pi_01 < 1 else 0
        else:
            independence_lr = 0

        return {
            'n_violations': n_violations,
            'expected_violations': expected_violations,
            'violation_rate': violation_rate,
            'expected_violation_rate': 1 - confidence_level,
            'kupiec_lr_stat': likelihood_ratio,
            'independence_lr_stat': independence_lr,
            'var_performance': 'PASS' if abs(violation_rate - (1 - confidence_level)) < 0.05 else 'FAIL'
        }

    def calculate_component_var(
            self,
            returns: pd.DataFrame,
            weights: np.ndarray,
            confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Component VaR for portfolio positions.

        Args:
            returns: Returns matrix (assets as columns)
            weights: Portfolio weights
            confidence_level: Confidence level for VaR

        Returns:
            Dictionary with component VaR for each asset
        """
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Portfolio VaR
        portfolio_var = self.historical_var(portfolio_returns, confidence_level)

        # Calculate marginal VaR for each asset
        component_vars = {}

        for i, asset in enumerate(returns.columns):
            # Calculate marginal VaR using finite differences
            epsilon = 0.01  # Small change in weight

            # Create perturbed weights
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize

            # Calculate perturbed portfolio returns and VaR
            perturbed_portfolio_returns = (returns * perturbed_weights).sum(axis=1)
            perturbed_var = self.historical_var(perturbed_portfolio_returns, confidence_level)

            # Marginal VaR
            marginal_var = (perturbed_var - portfolio_var) / epsilon

            # Component VaR
            component_var = weights[i] * marginal_var
            component_vars[asset] = component_var

        return component_vars

    def extreme_value_theory_var(
            self,
            returns: pd.Series,
            confidence_level: float = 0.95,
            threshold_percentile: float = 0.1
    ) -> float:
        """
        Calculate VaR using Extreme Value Theory (EVT).

        Args:
            returns: Portfolio returns
            confidence_level: Confidence level for VaR
            threshold_percentile: Percentile for threshold selection

        Returns:
            EVT-based VaR estimate
        """
        # Select threshold (e.g., 10th percentile for left tail)
        threshold = np.percentile(returns, threshold_percentile * 100)

        # Extract exceedances (returns below threshold)
        exceedances = returns[returns < threshold] - threshold

        if len(exceedances) < 10:  # Need sufficient exceedances
            return self.historical_var(returns, confidence_level)

        # Fit Generalized Pareto Distribution to exceedances
        try:
            from scipy.stats import genpareto

            # Fit GPD parameters
            shape, loc, scale = genpareto.fit(-exceedances, floc=0)  # Use positive exceedances

            # Calculate VaR using EVT
            n = len(returns)
            n_u = len(exceedances)

            # Probability of exceedance
            p_u = n_u / n

            # Calculate VaR
            p = 1 - confidence_level

            if shape != 0:
                var_evt = threshold - (scale / shape) * (((p / p_u) ** (-shape)) - 1)
            else:
                var_evt = threshold - scale * np.log(p / p_u)

            return var_evt

        except Exception:
            # Fallback to historical VaR if EVT fitting fails
            return self.historical_var(returns, confidence_level)

    def calculate_risk_budget(
            self,
            returns: pd.DataFrame,
            weights: np.ndarray,
            risk_measure: str = 'volatility'
    ) -> Dict[str, float]:
        """
        Calculate risk budgeting/contribution for portfolio positions.

        Args:
            returns: Returns matrix (assets as columns)
            weights: Portfolio weights
            risk_measure: Risk measure ('volatility', 'var', 'cvar')

        Returns:
            Dictionary with risk contributions
        """
        if risk_measure == 'volatility':
            # Covariance matrix
            cov_matrix = returns.cov() * self.trading_days_year

            # Portfolio variance
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)

            # Risk contributions
            risk_contributions = {}
            for i, asset in enumerate(returns.columns):
                marginal_contrib = np.dot(cov_matrix.iloc[i], weights)
                risk_contrib = weights[i] * marginal_contrib / portfolio_vol
                risk_contributions[asset] = risk_contrib

        elif risk_measure == 'var':
            # Use component VaR
            risk_contributions = self.calculate_component_var(returns, weights)

        else:
            raise ValueError(f"Unsupported risk measure: {risk_measure}")

        return risk_contributions