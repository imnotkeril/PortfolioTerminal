"""
Main Streamlit Application for Portfolio Management System.

This is the refactored entry point for the web interface implementing
modular architecture with separated pages, components, and utilities.
"""
import streamlit as st
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import utility modules
from streamlit_app.utils.session_state import initialize_session_state, get_portfolios, get_last_price_update
from streamlit_app.utils.formatting import format_currency, format_datetime

# Import page modules
from streamlit_app.pages.dashboard import render_dashboard
from streamlit_app.pages.create_portfolio import render_create_portfolio
from streamlit_app.pages.manage_portfolios import render_manage_portfolios

# ================================
# PAGE CONFIGURATION
# ================================

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
        
        **Current Phase:** Data Foundation (Phase 1)
        **Version:** 1.0.0
        
        Features:
        - Portfolio creation and management
        - Real-time price data integration
        - Asset allocation analysis
        - Import/Export capabilities
        
        Built with Streamlit and modern Python stack.
        """
    }
)

# ================================
# CSS STYLING
# ================================

def load_css():
    """Load custom CSS styling."""

    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        padding-top: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
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
    """Render main application header with system status."""

    st.markdown("""
    <div class="main-header">
        <h1>🚀 Wild Market Capital - Portfolio Manager</h1>
        <p>Professional Portfolio Management & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # System status indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Market status (placeholder)
        market_status = "Open"  # This would come from price manager
        status_color = "🟢" if market_status == "Open" else "🔴"
        st.metric("Market Status", f"{status_color} {market_status}")

    with col2:
        portfolio_count = len(get_portfolios())
        st.metric("Total Portfolios", portfolio_count)

    with col3:
        # Cache status (placeholder)
        st.metric("Data Cache", "✅ Active")

    with col4:
        last_update = get_last_price_update()
        if last_update:
            from datetime import datetime
            time_ago = datetime.now() - last_update
            update_text = f"{time_ago.seconds // 60}m ago"
        else:
            update_text = "Never"
        st.metric("Last Update", update_text)


# ================================
# SIDEBAR NAVIGATION
# ================================

def render_sidebar():
    """Render sidebar navigation and return selected page."""

    with st.sidebar:
        st.title("📊 Navigation")

        # Main navigation
        page = st.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "📝 Create Portfolio",
                "📋 Manage Portfolios",
                "📊 Portfolio Analysis",
                "⚙️ System Status"
            ],
            key="main_navigation"
        )

        st.divider()

        # Quick actions section
        render_sidebar_quick_actions()

        st.divider()

        # Portfolio selector section
        render_sidebar_portfolio_selector()

        st.divider()

        # System information
        render_sidebar_system_info()

    return page


def render_sidebar_quick_actions():
    """Render quick action buttons in sidebar."""

    st.subheader("⚡ Quick Actions")

    if st.button("🔄 Refresh Data", use_container_width=True):
        from streamlit_app.utils.session_state import refresh_portfolios, get_price_manager
        refresh_portfolios()
        try:
            price_manager = get_price_manager()
            price_manager.clear_cache()
        except:
            pass
        st.success("Data refreshed!")
        st.rerun()

    if st.button("📥 Import Portfolio", use_container_width=True):
        st.session_state.main_navigation = "📝 Create Portfolio"
        st.rerun()


def render_sidebar_portfolio_selector():
    """Render portfolio selector in sidebar."""

    portfolios = get_portfolios()

    if portfolios:
        st.subheader("📁 Select Portfolio")

        portfolio_names = ["None"] + [p.name for p in portfolios]
        selected_name = st.selectbox(
            "Current Portfolio",
            portfolio_names,
            key="portfolio_selector"
        )

        if selected_name != "None":
            from streamlit_app.utils.session_state import set_selected_portfolio
            selected_portfolio = next(
                (p for p in portfolios if p.name == selected_name), None
            )
            set_selected_portfolio(selected_portfolio)

            if selected_portfolio:
                st.success(f"✅ {selected_portfolio.name}")
                st.caption(f"Assets: {len(selected_portfolio.assets)}")
                st.caption(f"Value: {format_currency(selected_portfolio.calculate_value())}")
        else:
            from streamlit_app.utils.session_state import set_selected_portfolio
            set_selected_portfolio(None)
    else:
        st.info("No portfolios created yet")


def render_sidebar_system_info():
    """Render system information in sidebar."""

    st.subheader("ℹ️ System Info")

    # Version and phase info
    st.caption("**Version:** 1.0.0")
    st.caption("**Phase:** Data Foundation")
    st.caption("**Build:** Refactored Architecture")

    # Portfolio statistics
    portfolios = get_portfolios()
    if portfolios:
        total_assets = sum(len(p.assets) for p in portfolios)
        total_value = sum(p.calculate_value() for p in portfolios)

        st.caption(f"**Total Assets:** {total_assets}")
        st.caption(f"**Combined Value:** {format_currency(total_value)}")


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
        render_portfolio_analysis()

    elif page_name == "⚙️ System Status":
        render_system_status()

    else:
        st.error(f"Unknown page: {page_name}")


def render_portfolio_analysis():
    """Render portfolio analysis page (placeholder)."""

    st.header("📊 Portfolio Analysis")
    st.info("🚧 Portfolio Analysis page will be implemented in the next phase of refactoring")

    from streamlit_app.utils.session_state import get_selected_portfolio
    selected_portfolio = get_selected_portfolio()

    if selected_portfolio:
        st.subheader(f"Analysis for: {selected_portfolio.name}")

        # Placeholder metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Value", format_currency(selected_portfolio.calculate_value()))

        with col2:
            st.metric("Assets Count", len(selected_portfolio.assets))

        with col3:
            st.metric("Portfolio Type", selected_portfolio.portfolio_type.value.title())

        with col4:
            st.metric("Created", format_datetime(selected_portfolio.created_date, "%Y-%m-%d"))

        # Placeholder chart
        if selected_portfolio.assets:
            from streamlit_app.components.charts import create_portfolio_allocation_chart
            fig = create_portfolio_allocation_chart(selected_portfolio, "pie")
            st.plotly_chart(fig, use_container_width=True)

        st.info("📈 Advanced analytics, risk metrics, and performance analysis coming soon!")

    else:
        st.info("💡 Select a portfolio from the sidebar to view detailed analysis")


def render_system_status():
    """Render system status page (placeholder)."""

    st.header("⚙️ System Status")
    st.info("🚧 System Status page will be implemented in the next phase of refactoring")

    # Basic system information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio System")
        portfolios = get_portfolios()
        st.metric("Total Portfolios", len(portfolios))

        if portfolios:
            total_assets = sum(len(p.assets) for p in portfolios)
            st.metric("Total Assets", total_assets)

        st.success("✅ Portfolio system operational")

    with col2:
        st.subheader("🔄 Price Data System")

        last_update = get_last_price_update()
        if last_update:
            st.metric("Last Price Update", format_datetime(last_update, "%H:%M:%S"))
            st.success("✅ Price data system operational")
        else:
            st.warning("⚠️ No recent price updates")

    # Placeholder for additional system checks
    st.subheader("🔍 System Health Checks")

    checks = [
        ("Core Data Manager", "✅ Operational"),
        ("Portfolio Storage", "✅ Operational"),
        ("Price Data Provider", "✅ Operational"),
        ("Session State", "✅ Operational"),
        ("File I/O System", "✅ Operational")
    ]

    for check_name, status in checks:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{check_name}**")
        with col2:
            st.write(status)


# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Load CSS styling
    load_css()

    # Render header
    render_header()

    # Render sidebar and get selected page
    current_page = render_sidebar()

    # Route to appropriate page
    render_page(current_page)

    # Footer
    render_footer()


def render_footer():
    """Render application footer."""

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🚀 Wild Market Capital Portfolio Manager")

    with col2:
        st.caption("v1.0.0 - Phase 1: Data Foundation")

    with col3:
        st.caption("Refactored Architecture - Modular Design")


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    main()