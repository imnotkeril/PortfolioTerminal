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

        # Analysis options
        st.subheader("Analysis Options")
        show_monte_carlo = st.checkbox("Monte Carlo Simulation", value=False)
        show_stress_tests = st.checkbox("Stress Testing", value=False)
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF"])

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

    # Get price data and calculate metrics using new analytics engine
    try:
        price_manager = get_price_manager()
        tickers = [asset.ticker for asset in selected_portfolio.assets if asset.ticker]

        if not tickers:
            st.error("No valid tickers found in portfolio.")
            st.stop()

        # Fetch historical data
        with st.spinner("Fetching price data and calculating comprehensive metrics..."):
            prices_data = price_manager.get_historical_prices(tickers, start_date, end_date)

            if prices_data.empty:
                st.error("No price data available for the selected period.")
                st.stop()

            # Calculate portfolio returns
            weights = []
            for asset in selected_portfolio.assets:
                if asset.weight:
                    weights.append(asset.weight)
                elif asset.shares and asset.current_price and selected_portfolio.total_value:
                    weight = (asset.shares * asset.current_price) / selected_portfolio.total_value
                    weights.append(weight)
                else:
                    weights.append(1.0 / len(selected_portfolio.assets))  # Equal weight fallback

            # Normalize weights
            weights = np.array(weights)
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
                except Exception as e:
                    st.warning(f"Could not fetch benchmark data: {e}")

            # Use new analytics engine for comprehensive analysis
            from core.analytics_engine import AnalyticsEngine
            analytics_engine = AnalyticsEngine(risk_free_rate=0.02)

            # Perform comprehensive analysis
            analysis_results = analytics_engine.analyze_portfolio(
                returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                portfolio_name=selected_portfolio_name
            )

            # Create all charts
            charts = analytics_engine.create_analysis_charts(
                returns=portfolio_returns,
                benchmark_returns=benchmark_returns
            )

    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        st.stop()

    # Enhanced Tab interface with more comprehensive analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Performance",
        "⚠️ Risk Analysis",
        "🎯 Allocation",
        "📊 Advanced Analytics",
        "🔄 Benchmark Comparison",
        "📅 Calendar View"
    ])

    with tab1:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Performance overview
        col1, col2 = st.columns([2, 1])

        with col1:
            # Enhanced performance chart
            st.plotly_chart(charts['performance_comparison'], use_container_width=True)

        with col2:
            # Key performance metrics
            perf_metrics = analysis_results['performance']

            st.metric(
                "Total Return",
                format_percentage(perf_metrics['total_return']),
                delta=format_percentage(perf_metrics['total_return'] - 0) if benchmark_returns is None else None
            )

            st.metric(
                "Annualized Return",
                format_percentage(perf_metrics['annualized_return']),
                delta=None
            )

            st.metric(
                "Sharpe Ratio",
                format_number(perf_metrics['sharpe_ratio'], 2),
                delta=None
            )

            st.metric(
                "Win Rate",
                format_percentage(perf_metrics['win_rate']),
                delta=None
            )

        # Detailed performance metrics
        st.subheader("Detailed Performance Metrics")

        # Create three columns for metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.markdown("**Return Metrics**")
            return_metrics = {
                'Total Return': perf_metrics['total_return'],
                'CAGR': perf_metrics['annualized_return'],
                'Average Monthly': perf_metrics['average_monthly_return'],
                'Best Month': perf_metrics['best_month'],
                'Worst Month': perf_metrics['worst_month']
            }

            for label, value in return_metrics.items():
                st.metric(label, format_percentage(value))

        with metrics_col2:
            st.markdown("**Risk Metrics**")
            risk_metrics = analysis_results['risk']

            risk_display = {
                'Volatility': risk_metrics['volatility'],
                'Max Drawdown': risk_metrics['max_drawdown'],
                'VaR (95%)': risk_metrics['var_95'],
                'Downside Dev.': risk_metrics['downside_deviation']
            }

            for label, value in risk_display.items():
                st.metric(label, format_percentage(value))

        with metrics_col3:
            st.markdown("**Risk-Adjusted**")
            risk_adj = analysis_results['risk_adjusted']

            risk_adj_display = {
                'Sharpe Ratio': risk_adj['sharpe_ratio'],
                'Sortino Ratio': risk_adj['sortino_ratio'],
                'Calmar Ratio': risk_adj['calmar_ratio'],
                'Omega Ratio': min(risk_adj['omega_ratio'], 99.99)  # Cap at 99.99 for display
            }

            for label, value in risk_adj_display.items():
                if 'ratio' in label.lower():
                    st.metric(label, format_number(value, 2))
                else:
                    st.metric(label, format_percentage(value))

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Risk analysis with enhanced VaR analysis
        col1, col2 = st.columns(2)

        with col1:
            # Drawdown chart
            st.plotly_chart(charts['drawdown'], use_container_width=True)

        with col2:
            # VaR comparison chart
            var_data = analysis_results['var_analysis']

            var_methods = ['historical', 'parametric_normal', 'monte_carlo', 'cornish_fisher']
            var_values = [abs(var_data.get(method, 0)) * 100 for method in var_methods]
            var_labels = ['Historical', 'Parametric', 'Monte Carlo', 'Cornish-Fisher']

            fig_var = go.Figure(data=[
                go.Bar(
                    x=var_labels,
                    y=var_values,
                    marker_color=['#ef4444', '#dc2626', '#b91c1c', '#991b1b'],
                    text=[f'{v:.2f}%' for v in var_values],
                    textposition='auto'
                )
            ])

            fig_var.update_layout(
                title="VaR (95%) - Different Methods",
                yaxis_title="Value at Risk (%)",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_var, use_container_width=True)

        # Comprehensive risk metrics display
        st.subheader("Comprehensive Risk Analysis")

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            st.markdown("**Value at Risk**")
            var_metrics = {
                'VaR 90%': analysis_results['risk']['var_90'],
                'VaR 95%': analysis_results['risk']['var_95'],
                'VaR 99%': analysis_results['risk']['var_99'],
                'CVaR 95%': var_data['expected_shortfall_historical']
            }

            for label, value in var_metrics.items():
                st.metric(label, format_percentage(value))

        with risk_col2:
            st.markdown("**Drawdown Analysis**")
            dd_metrics = {
                'Max Drawdown': analysis_results['risk']['max_drawdown'],
                'Avg Drawdown': analysis_results['risk']['average_drawdown'],
                'Max DD Duration': f"{analysis_results['risk']['max_drawdown_duration']:.0f} days",
                'Ulcer Index': analysis_results['risk']['ulcer_index']
            }

            for label, value in dd_metrics.items():
                if 'days' in str(value):
                    st.metric(label, value)
                else:
                    st.metric(label, format_percentage(value))

        with risk_col3:
            st.markdown("**Distribution**")
            dist_metrics = {
                'Skewness': analysis_results['risk']['skewness'],
                'Kurtosis': analysis_results['risk']['kurtosis'],
                'Jarque-Bera p': analysis_results['risk'].get('jarque_bera_pvalue', 0),
                'Max Consec. Losses': analysis_results['risk']['max_consecutive_losses']
            }

            for label, value in dist_metrics.items():
                if 'p' in label:
                    st.metric(label, f"{value:.4f}")
                elif 'losses' in label.lower():
                    st.metric(label, f"{int(value)}")
                else:
                    st.metric(label, format_number(value, 2))

        # Monte Carlo simulation if enabled
        if show_monte_carlo:
            st.subheader("Monte Carlo Simulation")

            with st.spinner("Running Monte Carlo simulation..."):
                # Run simulation using risk calculator
                simulation_results = analytics_engine.risk.monte_carlo_simulation(
                    portfolio_returns,
                    initial_value=selected_portfolio.total_value,
                    time_horizon=252,
                    n_simulations=1000
                )

                # Create Monte Carlo chart
                mc_chart = analytics_engine.charts.create_monte_carlo_simulation_chart(simulation_results)
                st.plotly_chart(mc_chart, use_container_width=True)

                # Simulation statistics
                final_values = simulation_results[:, -1]

                mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)

                with mc_col1:
                    st.metric("Expected Value", format_currency(np.mean(final_values)))

                with mc_col2:
                    st.metric("Worst 5%", format_currency(np.percentile(final_values, 5)))

                with mc_col3:
                    st.metric("Best 5%", format_currency(np.percentile(final_values, 95)))

                with mc_col4:
                    prob_loss = (final_values < selected_portfolio.total_value).mean()
                    st.metric("Prob. of Loss", format_percentage(prob_loss))

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Enhanced asset allocation analysis
        col1, col2 = st.columns([1, 1])

        with col1:
            fig_allocation = create_asset_allocation_chart(selected_portfolio)
            st.plotly_chart(fig_allocation, use_container_width=True)

        with col2:
            # Sector/Geographic breakdown if available
            st.subheader("Allocation Breakdown")

            # Calculate sector allocation
            sector_allocation = {}
            for asset in selected_portfolio.assets:
                sector = getattr(asset, 'sector', 'Unknown')
                if asset.shares and asset.current_price:
                    value = asset.shares * asset.current_price
                    if sector in sector_allocation:
                        sector_allocation[sector] += value
                    else:
                        sector_allocation[sector] = value

            if sector_allocation:
                total_value = sum(sector_allocation.values())
                sector_df = pd.DataFrame([
                    {
                        'Sector': sector,
                        'Value': value,
                        'Allocation': value / total_value
                    }
                    for sector, value in sector_allocation.items()
                ])

                sector_df['Value'] = sector_df['Value'].apply(lambda x: format_currency(x))
                sector_df['Allocation'] = sector_df['Allocation'].apply(lambda x: format_percentage(x))

                st.dataframe(sector_df, use_container_width=True, hide_index=True)

        # Detailed asset table
        st.subheader("Detailed Holdings")

        asset_data = []
        total_value = sum(asset.shares * asset.current_price for asset in selected_portfolio.assets if asset.current_price)

        for asset in selected_portfolio.assets:
            if asset.current_price and asset.shares:
                asset_value = asset.shares * asset.current_price
                allocation = asset_value / total_value if total_value > 0 else 0

                # Calculate individual asset performance if possible
                if asset.ticker in returns.columns:
                    asset_returns = returns[asset.ticker]
                    asset_total_return = (1 + asset_returns).prod() - 1
                    asset_volatility = asset_returns.std() * np.sqrt(252)
                else:
                    asset_total_return = 0
                    asset_volatility = 0

                asset_data.append({
                    'Ticker': asset.ticker,
                    'Name': asset.name[:30] + '...' if len(asset.name) > 30 else asset.name,
                    'Sector': getattr(asset, 'sector', 'Unknown'),
                    'Shares': int(asset.shares),
                    'Price': asset.current_price,
                    'Value': asset_value,
                    'Allocation': allocation,
                    'Return': asset_total_return,
                    'Volatility': asset_volatility
                })

        if asset_data:
            df = pd.DataFrame(asset_data)
            df['Price'] = df['Price'].apply(lambda x: format_currency(x))
            df['Value'] = df['Value'].apply(lambda x: format_currency(x))
            df['Allocation'] = df['Allocation'].apply(lambda x: format_percentage(x))
            df['Return'] = df['Return'].apply(lambda x: format_percentage(x))
            df['Volatility'] = df['Volatility'].apply(lambda x: format_percentage(x))

            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Advanced analytics with distribution analysis
        col1, col2 = st.columns(2)

        with col1:
            # Return distribution chart
            if 'return_distribution' in charts:
                st.plotly_chart(charts['return_distribution'], use_container_width=True)

        with col2:
            # Rolling metrics chart
            if 'rolling_metrics' in charts:
                st.plotly_chart(charts['rolling_metrics'], use_container_width=True)

        # Statistical analysis
        st.subheader("Statistical Analysis")

        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            st.markdown("**Distribution Statistics**")
            dist_stats = {
                'Mean': analysis_results['performance']['mean'],
                'Median': analysis_results['performance']['median'],
                'Std Dev': analysis_results['performance']['standard_deviation'],
                'Skewness': analysis_results['risk']['skewness'],
                'Kurtosis': analysis_results['risk']['kurtosis']
            }

            for label, value in dist_stats.items():
                if label in ['Mean', 'Median', 'Std Dev']:
                    st.metric(label, format_percentage(value))
                else:
                    st.metric(label, format_number(value, 3))

        with stat_col2:
            st.markdown("**Percentiles**")
            percentiles = {
                '1st Percentile': analysis_results['performance']['percentile_1'],
                '5th Percentile': analysis_results['performance']['percentile_5'],
                '25th Percentile': analysis_results['performance']['percentile_25'],
                '75th Percentile': analysis_results['performance']['percentile_75'],
                '95th Percentile': analysis_results['performance']['percentile_95']
            }

            for label, value in percentiles.items():
                st.metric(label, format_percentage(value))

        with stat_col3:
            st.markdown("**Advanced Ratios**")
            advanced_ratios = {
                'Information Ratio': analysis_results['risk_adjusted']['information_ratio'],
                'Gain/Pain Ratio': min(analysis_results['risk_adjusted']['gain_to_pain_ratio'], 99.99),
                'Sterling Ratio': min(analysis_results['risk_adjusted']['sterling_ratio'], 99.99),
                'Burke Ratio': min(analysis_results['risk_adjusted']['burke_ratio'], 99.99),
                'Upside Potential': min(analysis_results['risk_adjusted']['upside_potential_ratio'], 99.99)
            }

            for label, value in advanced_ratios.items():
                st.metric(label, format_number(value, 2))

        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        if benchmark_returns is not None and 'benchmark_comparison' in analysis_results:
            bench_data = analysis_results['benchmark_comparison']

            # Benchmark comparison overview
            st.subheader("Benchmark Comparison Summary")

            comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)

            with comp_col1:
                alpha = bench_data['regression']['alpha']
                st.metric(
                    "Alpha",
                    format_percentage(alpha),
                    delta=format_percentage(alpha) if alpha != 0 else None
                )

            with comp_col2:
                beta = bench_data['regression']['beta']
                st.metric("Beta", format_number(beta, 2))

            with comp_col3:
                info_ratio = bench_data['risk_adjusted']['information_ratio']
                st.metric("Information Ratio", format_number(info_ratio, 2))

            with comp_col4:
                tracking_error = bench_data['risk_adjusted']['tracking_error']
                st.metric("Tracking Error", format_percentage(tracking_error))

            # Detailed comparison table
            st.subheader("Detailed Comparison")

            # Create comparison DataFrame
            portfolio_summary = bench_data['summary']['Portfolio']
            benchmark_summary = bench_data['summary']['Benchmark']
            outperformance = bench_data['summary']['outperformance']

            comparison_data = [
                ['Total Return', portfolio_summary['total_return'], benchmark_summary['total_return'], outperformance['annualized_excess_return']],
                ['Annualized Return', portfolio_summary['annualized_return'], benchmark_summary['annualized_return'], outperformance['annualized_excess_return']],
                ['Volatility', portfolio_summary['volatility'], benchmark_summary['volatility'], outperformance['excess_volatility']],
                ['Sharpe Ratio', portfolio_summary['sharpe_ratio'], benchmark_summary['sharpe_ratio'], outperformance['sharpe_difference']],
                ['Max Drawdown', portfolio_summary['max_drawdown'], benchmark_summary['max_drawdown'], outperformance['relative_max_drawdown']]
            ]

            comparison_df = pd.DataFrame(comparison_data, columns=['Metric', 'Portfolio', 'Benchmark', 'Difference'])

            # Format the DataFrame
            for col in ['Portfolio', 'Benchmark', 'Difference']:
                comparison_df[col] = comparison_df.apply(
                    lambda row: format_percentage(row[col]) if 'Ratio' not in row['Metric'] or row['Metric'] == 'Sharpe Ratio' else format_number(row[col], 2), axis=1
                )

            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Capture ratios
            st.subheader("Capture Ratios")

            capture_col1, capture_col2, capture_col3 = st.columns(3)

            with capture_col1:
                up_capture = bench_data['capture_ratios']['up_capture_ratio']
                st.metric("Up Capture", format_percentage(up_capture))

            with capture_col2:
                down_capture = bench_data['capture_ratios']['down_capture_ratio']
                st.metric("Down Capture", format_percentage(down_capture))

            with capture_col3:
                capture_ratio = bench_data['capture_ratios']['capture_ratio']
                st.metric("Capture Ratio", format_number(capture_ratio, 2))

            # Performance attribution
            if 'attribution' in bench_data:
                st.subheader("Performance Attribution")
                attribution = bench_data['attribution']

                attr_col1, attr_col2, attr_col3 = st.columns(3)

                with attr_col1:
                    st.metric("Total Active Return", format_percentage(attribution['total_active_return']))

                with attr_col2:
                    st.metric("Selection Effect", format_percentage(attribution['selection_effect']))

                with attr_col3:
                    st.metric("Correlation", format_number(attribution['correlation_with_benchmark'], 3))

            # Rolling analysis if available
            if 'rolling_analysis' in bench_data and '252d' in bench_data['rolling_analysis']:
                st.subheader("Rolling Analysis (1 Year)")
                rolling_data = bench_data['rolling_analysis']['252d']

                roll_col1, roll_col2, roll_col3, roll_col4 = st.columns(4)

                with roll_col1:
                    st.metric("Current Excess Return", format_percentage(rolling_data['current_excess_return']))

                with roll_col2:
                    st.metric("Average Excess Return", format_percentage(rolling_data['average_excess_return']))

                with roll_col3:
                    st.metric("Outperformance Frequency", format_percentage(rolling_data['outperformance_frequency']))

                with roll_col4:
                    st.metric("Current Beta", format_number(rolling_data['current_beta'], 2))

        else:
            st.info("Select a benchmark to see detailed comparison analysis.")

            # Show available benchmarks
            st.subheader("Popular Benchmarks")
            benchmark_info = {
                "SPY": "S&P 500 - Large Cap US Stocks",
                "QQQ": "NASDAQ 100 - Technology Heavy",
                "VTI": "Total Stock Market - Broad US Market",
                "AGG": "Aggregate Bond Index - US Bonds",
                "VEA": "Developed Markets - International Stocks",
                "VWO": "Emerging Markets - EM Stocks"
            }

            for ticker, description in benchmark_info.items():
                st.markdown(f"**{ticker}**: {description}")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)

        # Calendar analysis
        if len(portfolio_returns) > 60:  # Need at least 2-3 months of data

            # Monthly returns heatmap
            if 'monthly_heatmap' in charts:
                st.plotly_chart(charts['monthly_heatmap'], use_container_width=True)

            # Monthly statistics
            monthly_returns = portfolio_returns.groupby([portfolio_returns.index.year, portfolio_returns.index.month]).apply(
                lambda x: (1 + x).prod() - 1
            )

            if len(monthly_returns) > 0:
                st.subheader("Monthly Performance Statistics")

                month_col1, month_col2, month_col3, month_col4 = st.columns(4)

                with month_col1:
                    st.metric("Best Month", format_percentage(monthly_returns.max()))

                with month_col2:
                    st.metric("Worst Month", format_percentage(monthly_returns.min()))

                with month_col3:
                    st.metric("Average Month", format_percentage(monthly_returns.mean()))

                with month_col4:
                    positive_months = (monthly_returns > 0).sum()
                    total_months = len(monthly_returns)
                    win_rate = positive_months / total_months if total_months > 0 else 0
                    st.metric("Monthly Win Rate", format_percentage(win_rate))

                # Quarterly analysis if enough data
                if len(portfolio_returns) > 180:  # At least 6 months
                    quarterly_returns = portfolio_returns.groupby([portfolio_returns.index.year, portfolio_returns.index.quarter]).apply(
                        lambda x: (1 + x).prod() - 1
                    )

                    if len(quarterly_returns) > 1:
                        st.subheader("Quarterly Performance")

                        qtr_col1, qtr_col2, qtr_col3, qtr_col4 = st.columns(4)

                        with qtr_col1:
                            st.metric("Best Quarter", format_percentage(quarterly_returns.max()))

                        with qtr_col2:
                            st.metric("Worst Quarter", format_percentage(quarterly_returns.min()))

                        with qtr_col3:
                            st.metric("Average Quarter", format_percentage(quarterly_returns.mean()))

                        with qtr_col4:
                            positive_quarters = (quarterly_returns > 0).sum()
                            total_quarters = len(quarterly_returns)
                            qtr_win_rate = positive_quarters / total_quarters if total_quarters > 0 else 0
                            st.metric("Quarterly Win Rate", format_percentage(qtr_win_rate))

                # Seasonal analysis
                st.subheader("Seasonal Analysis")

                # Group by month name
                monthly_by_name = portfolio_returns.groupby(portfolio_returns.index.month).apply(
                    lambda x: (1 + x).prod() ** (12/len(x)) - 1 if len(x) > 0 else 0
                )

                if len(monthly_by_name) > 0:
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    seasonal_data = []
                    for i in range(1, 13):
                        if i in monthly_by_name.index:
                            seasonal_data.append({
                                'Month': month_names[i-1],
                                'Avg Return': monthly_by_name[i]
                            })

                    if seasonal_data:
                        seasonal_df = pd.DataFrame(seasonal_data)
                        seasonal_df['Avg Return'] = seasonal_df['Avg Return'].apply(lambda x: format_percentage(x))

                        # Create seasonal performance chart
                        fig_seasonal = go.Figure(data=[
                            go.Bar(
                                x=[d['Month'] for d in seasonal_data],
                                y=[monthly_by_name[i+1] * 100 for i in range(len(seasonal_data))],
                                marker_color=['#10b981' if monthly_by_name[i+1] > 0 else '#ef4444' for i in range(len(seasonal_data))]
                            )
                        ])

                        fig_seasonal.update_layout(
                            title="Average Monthly Performance by Month",
                            xaxis_title="Month",
                            yaxis_title="Average Return (%)",
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )

                        st.plotly_chart(fig_seasonal, use_container_width=True)

        else:
            st.info("Not enough data for calendar analysis. Need at least 60 days of data.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Advanced export and report generation section
    st.markdown("---")
    st.subheader("📊 Export & Reports")

    export_col1, export_col2, export_col3, export_col4 = st.columns(4)

    with export_col1:
        if st.button("📈 Export Performance Report", use_container_width=True):
            # Generate comprehensive report
            report_text = analytics_engine.generate_report(
                returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                portfolio_name=selected_portfolio_name,
                format="text"
            )

            st.download_button(
                label="Download Text Report",
                data=report_text,
                file_name=f"{selected_portfolio_name}_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

    with export_col2:
        if st.button("📊 Export All Metrics", use_container_width=True):
            # Create comprehensive metrics CSV
            all_metrics = {**analysis_results['performance'], **analysis_results['risk'], **analysis_results['risk_adjusted']}
            metrics_df = pd.DataFrame(list(all_metrics.items()), columns=['Metric', 'Value'])

            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=csv,
                file_name=f"{selected_portfolio_name}_all_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with export_col3:
        if st.button("📈 Export Returns Data", use_container_width=True):
            # Export returns with benchmark if available
            returns_df = pd.DataFrame({
                'Date': portfolio_returns.index,
                'Portfolio_Return': portfolio_returns.values
            })

            if benchmark_returns is not None:
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
                returns_df['Benchmark_Return'] = aligned_benchmark.values
                returns_df['Active_Return'] = portfolio_returns.values - aligned_benchmark.values

            csv = returns_df.to_csv(index=False)
            st.download_button(
                label="Download Returns CSV",
                data=csv,
                file_name=f"{selected_portfolio_name}_returns_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with export_col4:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()

    # Performance summary at the bottom
    with st.expander("🎯 Analysis Summary", expanded=False):
        st.markdown("### Key Insights")

        # Generate automated insights
        perf = analysis_results['performance']
        risk = analysis_results['risk']

        insights = []

        # Performance insights
        if perf['total_return'] > 0:
            insights.append(f"✅ Portfolio generated positive returns of {format_percentage(perf['total_return'])} over the analysis period")
        else:
            insights.append(f"❌ Portfolio declined {format_percentage(abs(perf['total_return']))} over the analysis period")

        # Risk insights
        if perf['sharpe_ratio'] > 1:
            insights.append(f"✅ Strong risk-adjusted performance with Sharpe ratio of {perf['sharpe_ratio']:.2f}")
        elif perf['sharpe_ratio'] > 0.5:
            insights.append(f"⚡ Moderate risk-adjusted performance with Sharpe ratio of {perf['sharpe_ratio']:.2f}")
        else:
            insights.append(f"⚠️ Low risk-adjusted performance with Sharpe ratio of {perf['sharpe_ratio']:.2f}")

        # Volatility insights
        if risk['volatility'] < 0.15:
            insights.append(f"✅ Low volatility portfolio at {format_percentage(risk['volatility'])} annual volatility")
        elif risk['volatility'] < 0.25:
            insights.append(f"⚡ Moderate volatility at {format_percentage(risk['volatility'])} annual volatility")
        else:
            insights.append(f"⚠️ High volatility portfolio at {format_percentage(risk['volatility'])} annual volatility")

        # Drawdown insights
        if abs(risk['max_drawdown']) < 0.10:
            insights.append(f"✅ Well-controlled downside risk with max drawdown of {format_percentage(risk['max_drawdown'])}")
        elif abs(risk['max_drawdown']) < 0.20:
            insights.append(f"⚡ Moderate downside risk with max drawdown of {format_percentage(risk['max_drawdown'])}")
        else:
            insights.append(f"⚠️ Significant downside risk with max drawdown of {format_percentage(risk['max_drawdown'])}")

        # Win rate insights
        if perf['win_rate'] > 0.6:
            insights.append(f"✅ High consistency with {format_percentage(perf['win_rate'])} win rate")
        elif perf['win_rate'] > 0.5:
            insights.append(f"⚡ Moderate consistency with {format_percentage(perf['win_rate'])} win rate")
        else:
            insights.append(f"⚠️ Low consistency with {format_percentage(perf['win_rate'])} win rate")

        # Benchmark insights
        if benchmark_returns is not None and 'benchmark_comparison' in analysis_results:
            alpha = analysis_results['benchmark_comparison']['regression']['alpha']
            if alpha > 0.02:
                insights.append(f"✅ Strong outperformance with {format_percentage(alpha)} annual alpha")
            elif alpha > 0:
                insights.append(f"⚡ Modest outperformance with {format_percentage(alpha)} annual alpha")
            else:
                insights.append(f"❌ Underperformance with {format_percentage(alpha)} annual alpha")

        for insight in insights:
            st.markdown(f"- {insight}")

if __name__ == "__main__":
    main()"""
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

from core.analytics_engine.performance_calculator import PerformanceCalculator
from streamlit_app.utils.session_state import (
    get_portfolio_manager,
    get_price_manager,
    initialize_session_state
)
from streamlit_app.utils.formatting import (
    format_currency,
    format_percentage,
    format_number
)

# Page configuration
st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

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
        price_manager = get_price_manager()
        tickers = [asset.ticker for asset in selected_portfolio.assets if asset.ticker]

        if not tickers:
            st.error("No valid tickers found in portfolio.")
            st.stop()

        # Fetch historical data
        with st.spinner("Fetching price data and calculating metrics..."):
            prices_data = price_manager.get_historical_prices(tickers, start_date, end_date)

            if prices_data.empty:
                st.error("No price data available for the selected period.")
                st.stop()

            # Calculate portfolio returns
            weights = []
            for asset in selected_portfolio.assets:
                if asset.weight:
                    weights.append(asset.weight)
                elif asset.shares and asset.current_price and selected_portfolio.total_value:
                    weight = (asset.shares * asset.current_price) / selected_portfolio.total_value
                    weights.append(weight)
                else:
                    weights.append(1.0 / len(selected_portfolio.assets))  # Equal weight fallback

            # Normalize weights
            weights = np.array(weights)
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
                except Exception as e:
                    st.warning(f"Could not fetch benchmark data: {e}")

            # Calculate metrics
            calculator = PerformanceCalculator()
            portfolio_metrics = calculator.calculate_all_metrics(portfolio_returns, benchmark_returns)

            # Calculate benchmark metrics if available
            benchmark_metrics = None
            if benchmark_returns is not None:
                benchmark_metrics = calculator.calculate_all_metrics(benchmark_returns)

    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        st.stop()

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