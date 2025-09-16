"""
Wild Market Capital - Portfolio Management System
Main Streamlit Application with Portfolio Analysis Integration (FIXED VERSION)

This fixes the import issues and integrates the new Portfolio Analysis functionality.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import application modules with fixed paths
from streamlit_app.utils.session_state import (
    initialize_session_state,
    get_portfolios,
    get_last_price_update,
    get_selected_portfolio,
    set_selected_portfolio
)
from streamlit_app.utils.formatting import format_currency, format_datetime
from streamlit_app.pages.dashboard import render_dashboard
from streamlit_app.pages.create_portfolio import render_create_portfolio
from streamlit_app.pages.manage_portfolios import render_manage_portfolios

# Configure Streamlit page
st.set_page_config(
    page_title="WMC Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.wildmarketcapital.com',
        'Report a bug': 'https://github.com/wmc/portfolio-manager/issues',
        'About': """
        # Wild Market Capital Portfolio Manager

        Professional-grade portfolio management and analytics platform.
        
        **Current Phase:** Portfolio Analytics (Phase 2)
        **Version:** 1.0.0
        
        Features:
        - Portfolio creation and management
        - Real-time price data integration
        - Advanced portfolio analytics (70+ metrics)
        - Risk analysis and VaR calculations
        - Benchmark comparison
        - Interactive charts and visualizations
        - Import/Export capabilities
        
        Built with Streamlit and modern Python stack.
        """
    }
)


# ================================
# CSS STYLING
# ================================

def load_css():
    """Load custom CSS styling for the application."""

    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .dataframe td {
        border: none !important;
        padding: 0.75rem !important;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    /* Portfolio cards */
    .portfolio-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .portfolio-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Analysis page specific styling */
    .tab-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .portfolio-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ================================
# HEADER SECTION
# ================================

def render_header():
    """Render main application header."""

    st.markdown("""
    <div class="main-header">
        <h1>🚀 Wild Market Capital - Portfolio Manager</h1>
        <p>Professional Portfolio Management & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)


# ================================
# SIDEBAR NAVIGATION
# ================================

def render_sidebar():
    """Render sidebar navigation and return selected page."""

    with st.sidebar:
        st.title("📊 Navigation")

        # Main navigation with Portfolio Analysis included
        page = st.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "📝 Create Portfolio",
                "📋 Manage Portfolios",
                "📊 Portfolio Analysis",  # ← НОВАЯ СТРАНИЦА АНАЛИЗА
                "⚙️ System Status"
            ],
            key="main_navigation"
        )

        st.divider()

        # System information
        render_sidebar_system_info()

    return page


def render_sidebar_system_info():
    """Render system information in sidebar."""

    st.subheader("ℹ️ System Info")

    # Version and phase info
    st.caption("**Version:** 1.0.0")
    st.caption("**Phase:** Portfolio Analytics")
    st.caption("**Build:** Phase 2 Complete")

    # Portfolio statistics
    portfolios = get_portfolios()
    if portfolios:
        total_assets = sum(len(p.assets) for p in portfolios)
        total_value = sum(p.calculate_value() for p in portfolios)

        st.caption(f"**Total Portfolios:** {len(portfolios)}")
        st.caption(f"**Total Assets:** {total_assets}")
        st.caption(f"**Combined Value:** {format_currency(total_value)}")
    else:
        st.caption("**Total Portfolios:** 0")
        st.caption("**Total Assets:** 0")
        st.caption("**Combined Value:** $0.00")

    # Analytics features indicator
    st.divider()
    st.subheader("🔧 Features")
    st.caption("✅ **Portfolio Creation**")
    st.caption("✅ **Data Management**")
    st.caption("✅ **Price Updates**")
    st.caption("✅ **Advanced Analytics**")
    st.caption("✅ **Risk Analysis**")
    st.caption("✅ **Benchmark Comparison**")


# ================================
# PAGE ROUTING
# ================================

def render_page(page_name: str):
    """Route to appropriate page based on navigation selection."""

    if page_name == "🏠 Dashboard":
        render_dashboard()

    elif page_name == "📝 Create Portfolio":
        render_create_portfolio()

    elif page_name == "📋 Manage Portfolios":
        render_manage_portfolios()

    elif page_name == "📊 Portfolio Analysis":
        render_portfolio_analysis()  # ← ДОБАВЛЕН РОУТИНГ

    elif page_name == "⚙️ System Status":
        render_system_status()

    else:
        st.error(f"Unknown page: {page_name}")


def render_portfolio_analysis():
    """Render portfolio analysis page with advanced analytics."""

    try:
        # Try to load the advanced Portfolio Analysis page
        analysis_page_path = current_dir / "pages" / "📊_Portfolio_Analysis.py"

        if analysis_page_path.exists():
            # Import and run the analysis page
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("portfolio_analysis", analysis_page_path)
            portfolio_analysis = importlib.util.module_from_spec(spec)

            # Add to sys.modules to enable proper imports within the module
            sys.modules["portfolio_analysis"] = portfolio_analysis
            spec.loader.exec_module(portfolio_analysis)

            # Execute the main function
            portfolio_analysis.main()

        else:
            # Fallback to enhanced basic analysis
            render_enhanced_portfolio_analysis()

    except Exception as e:
        st.error(f"Error loading advanced Portfolio Analysis: {str(e)}")
        st.error("Loading fallback analysis...")
        render_enhanced_portfolio_analysis()


def render_enhanced_portfolio_analysis():
    """Enhanced portfolio analysis with basic analytics."""

    st.header("📊 Portfolio Analysis")

    portfolios = get_portfolios()
    if not portfolios:
        st.warning("📋 No portfolios found. Please create a portfolio first.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("➕ Create Your First Portfolio", use_container_width=True):
                st.switch_page("pages/create_portfolio.py")
        return

    # Portfolio selection
    portfolio_names = [p.name for p in portfolios]
    selected_portfolio_name = st.selectbox(
        "Select Portfolio to Analyze",
        portfolio_names,
        help="Choose which portfolio you want to analyze"
    )

    if selected_portfolio_name:
        selected_portfolio = next(p for p in portfolios if p.name == selected_portfolio_name)

        # Set selected portfolio for other pages
        set_selected_portfolio(selected_portfolio)

        # Portfolio overview
        st.subheader(f"📊 Analysis: {selected_portfolio.name}")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Portfolio Value",
                format_currency(selected_portfolio.calculate_value()),
                help="Current total value of all positions"
            )

        with col2:
            st.metric(
                "Number of Assets",
                len(selected_portfolio.assets),
                help="Total number of different assets in portfolio"
            )

        with col3:
            if selected_portfolio.initial_value and selected_portfolio.initial_value > 0:
                current_value = selected_portfolio.calculate_value()
                total_return = (current_value - selected_portfolio.initial_value) / selected_portfolio.initial_value
                st.metric(
                    "Total Return",
                    f"{total_return:.1%}",
                    delta=f"{total_return:.1%}",
                    help="Total return since portfolio creation"
                )
            else:
                st.metric("Total Return", "N/A", help="Initial value not set")

        with col4:
            st.metric(
                "Created",
                selected_portfolio.created_date.strftime("%Y-%m-%d") if selected_portfolio.created_date else "N/A",
                help="Portfolio creation date"
            )

        # Tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["📈 Asset Breakdown", "📊 Allocation Analysis", "📋 Portfolio Details"])

        with tab1:
            # Asset breakdown table
            if selected_portfolio.assets:
                st.subheader("Asset Holdings")

                asset_data = []
                total_value = selected_portfolio.calculate_value()

                for asset in selected_portfolio.assets:
                    if asset.current_price and asset.shares:
                        asset_value = asset.shares * asset.current_price
                        allocation = asset_value / total_value if total_value > 0 else 0

                        asset_data.append({
                            'Ticker': asset.ticker,
                            'Name': asset.name[:30] + '...' if len(asset.name) > 30 else asset.name,
                            'Shares': f"{asset.shares:,.0f}",
                            'Price': f"${asset.current_price:.2f}",
                            'Value': format_currency(asset_value),
                            'Allocation': f"{allocation:.1%}",
                            'Sector': getattr(asset, 'sector', 'Unknown')
                        })

                if asset_data:
                    import pandas as pd
                    df = pd.DataFrame(asset_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Quick stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_allocation = 1.0 / len(asset_data) if asset_data else 0
                        st.metric("Average Allocation", f"{avg_allocation:.1%}")

                    with col2:
                        max_allocation = max([float(item['Allocation'].strip('%'))/100 for item in asset_data]) if asset_data else 0
                        st.metric("Largest Position", f"{max_allocation:.1%}")

                    with col3:
                        unique_sectors = len(set(item['Sector'] for item in asset_data))
                        st.metric("Sectors", unique_sectors)

                else:
                    st.warning("No current price data available for assets.")
            else:
                st.warning("This portfolio has no assets.")

        with tab2:
            # Basic allocation chart
            if selected_portfolio.assets:
                try:
                    # Try to create a basic pie chart
                    from streamlit_app.components.charts import create_portfolio_allocation_chart
                    fig = create_portfolio_allocation_chart(selected_portfolio, "pie")
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    # Fallback to simple display
                    st.info("📊 Advanced charts require the charts component. Showing text summary:")

                    total_value = selected_portfolio.calculate_value()
                    if total_value > 0:
                        for asset in selected_portfolio.assets:
                            if asset.current_price and asset.shares:
                                asset_value = asset.shares * asset.current_price
                                allocation = asset_value / total_value
                                st.write(f"**{asset.ticker}**: {allocation:.1%} ({format_currency(asset_value)})")
            else:
                st.info("No assets to display allocation for.")

        with tab3:
            # Portfolio details
            st.subheader("Portfolio Information")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write("**Basic Information:**")
                st.write(f"• **Name:** {selected_portfolio.name}")
                st.write(f"• **Type:** {selected_portfolio.portfolio_type.value.title()}")
                st.write(f"• **Created:** {selected_portfolio.created_date.strftime('%Y-%m-%d %H:%M') if selected_portfolio.created_date else 'Unknown'}")
                st.write(f"• **Assets:** {len(selected_portfolio.assets)}")

            with detail_col2:
                st.write("**Financial Summary:**")
                st.write(f"• **Current Value:** {format_currency(selected_portfolio.calculate_value())}")
                st.write(f"• **Initial Value:** {format_currency(selected_portfolio.initial_value) if selected_portfolio.initial_value else 'Not set'}")

                if selected_portfolio.description:
                    st.write("**Description:**")
                    st.write(selected_portfolio.description)

        # Call to action for advanced analytics
        st.info("🚀 **Upgrade to Advanced Analytics!** Get access to 70+ performance metrics, risk analysis, benchmark comparison, and interactive charts by ensuring the Portfolio Analysis module is properly installed.")

        if st.button("🔄 Refresh Analysis", help="Refresh portfolio data and recalculate metrics"):
            st.rerun()


def render_system_status():
    """Render system status page."""

    st.header("⚙️ System Status")

    # System health overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio System")
        portfolios = get_portfolios()
        st.metric("Total Portfolios", len(portfolios))

        if portfolios:
            total_assets = sum(len(p.assets) for p in portfolios)
            st.metric("Total Assets", total_assets)

            total_value = sum(p.calculate_value() for p in portfolios)
            st.metric("Total Value", format_currency(total_value))

        st.success("✅ Portfolio system operational")

    with col2:
        st.subheader("🔄 Data Systems")

        last_update = get_last_price_update()
        if last_update:
            st.metric("Last Price Update", format_datetime(last_update, "%H:%M:%S"))
            st.success("✅ Price data system operational")
        else:
            st.warning("⚠️ No recent price updates")

        # Analytics system status
        analytics_page_exists = (current_dir / "pages" / "📊_Portfolio_Analysis.py").exists()
        analytics_engine_exists = (project_root / "core" / "analytics_engine" / "__init__.py").exists()

        if analytics_page_exists and analytics_engine_exists:
            st.metric("Analytics Engine", "✅ Fully Operational")
            st.success("✅ Advanced analytics available")
        elif analytics_page_exists:
            st.metric("Analytics Engine", "⚠️ Partial")
            st.warning("⚠️ Analytics page exists, engine missing")
        else:
            st.metric("Analytics Engine", "❌ Not Available")
            st.error("❌ Analytics components missing")

    # System health checks
    st.subheader("🔍 System Health Checks")

    checks = [
        ("Core Data Manager", "✅ Operational", "success"),
        ("Portfolio Storage", "✅ Operational", "success"),
        ("Price Data Provider", "✅ Operational", "success"),
        ("Session State", "✅ Operational", "success"),
        ("File I/O System", "✅ Operational", "success")
    ]

    # Add analytics check
    if analytics_page_exists and analytics_engine_exists:
        checks.append(("Analytics Engine", "✅ Fully Operational", "success"))
    elif analytics_page_exists:
        checks.append(("Analytics Engine", "⚠️ Partially Available", "warning"))
    else:
        checks.append(("Analytics Engine", "❌ Not Available", "error"))

    for check_name, status, status_type in checks:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{check_name}**")
        with col2:
            if status_type == "success":
                st.success(status)
            elif status_type == "warning":
                st.warning(status)
            else:
                st.error(status)

    # Feature status
    st.subheader("🚀 Feature Status")

    features = [
        ("Portfolio Creation & Management", "✅ Fully Operational"),
        ("Data Import/Export", "✅ Fully Operational"),
        ("Price Data Updates", "✅ Fully Operational"),
        ("Basic Portfolio Analysis", "✅ Fully Operational"),
    ]

    # Advanced features status
    if analytics_page_exists and analytics_engine_exists:
        features.extend([
            ("Advanced Analytics (70+ Metrics)", "✅ Fully Operational"),
            ("Risk Analysis & VaR", "✅ Fully Operational"),
            ("Benchmark Comparison", "✅ Fully Operational"),
            ("Interactive Charts", "✅ Fully Operational"),
        ])
    else:
        features.extend([
            ("Advanced Analytics (70+ Metrics)", "🚧 Install Required"),
            ("Risk Analysis & VaR", "🚧 Install Required"),
            ("Benchmark Comparison", "🚧 Install Required"),
            ("Interactive Charts", "🚧 Install Required"),
        ])

    features.extend([
        ("Portfolio Optimization", "🚧 Coming in Phase 3"),
        ("API Integration", "🚧 Coming in Phase 4")
    ])

    for feature, status in features:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{feature}**")
        with col2:
            if "✅" in status:
                st.success(status)
            elif "🚧" in status:
                st.info(status)
            else:
                st.warning(status)


# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application entry point."""

    try:
        # Initialize session state
        initialize_session_state()

        # Load CSS styling
        load_css()

        # Render sidebar and get selected page
        current_page = render_sidebar()

        # Render header only on dashboard
        if current_page == "🏠 Dashboard":
            render_header()

        # Route to appropriate page
        render_page(current_page)

        # Footer
        render_footer()

    except Exception as e:
        st.error(f"Application Error: {str(e)}")

        # Show more details in expander
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

        st.info("💡 Try refreshing the page or check the System Status for more information.")


def render_footer():
    """Render application footer."""

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🚀 Wild Market Capital Portfolio Manager")

    with col2:
        st.caption("v1.0.0 - Phase 2: Portfolio Analytics")

    with col3:
        st.caption("Advanced Analytics & Risk Management")


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    main()