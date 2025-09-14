"""
Session state management utilities for the Portfolio Management System.

This module handles Streamlit session state initialization and management.
"""
import streamlit as st
from typing import List, Optional, Any
from datetime import datetime
import sys
from pathlib import Path

# Add core module to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data_manager import PortfolioManager, PriceManager, Portfolio


def initialize_session_state():
    """
    Initialize Streamlit session state with default values.

    This function sets up all necessary session state variables
    if they don't already exist.
    """

    # Initialize core managers
    if "portfolio_manager" not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()

    if "price_manager" not in st.session_state:
        st.session_state.price_manager = PriceManager()

    # Portfolio-related state
    if "portfolios" not in st.session_state:
        st.session_state.portfolios = []

    if "selected_portfolio" not in st.session_state:
        st.session_state.selected_portfolio = None

    # Navigation state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"

    # Data state
    if "last_price_update" not in st.session_state:
        st.session_state.last_price_update = None

    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}

    # UI state
    if "show_advanced_options" not in st.session_state:
        st.session_state.show_advanced_options = False

    if "confirm_delete_states" not in st.session_state:
        st.session_state.confirm_delete_states = {}

    # Form state
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}

    # Error handling state
    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

    # Performance tracking
    if "page_load_times" not in st.session_state:
        st.session_state.page_load_times = {}


def get_portfolio_manager() -> PortfolioManager:
    """
    Get the portfolio manager instance from session state.

    Returns:
        PortfolioManager instance
    """
    initialize_session_state()
    return st.session_state.portfolio_manager


def get_price_manager() -> PriceManager:
    """
    Get the price manager instance from session state.

    Returns:
        PriceManager instance
    """
    initialize_session_state()
    return st.session_state.price_manager


def get_portfolios() -> List[Portfolio]:
    """
    Get the list of portfolios from session state.

    Returns:
        List of Portfolio objects
    """
    initialize_session_state()
    return st.session_state.portfolios


def set_portfolios(portfolios: List[Portfolio]):
    """
    Set the list of portfolios in session state.

    Args:
        portfolios: List of Portfolio objects
    """
    initialize_session_state()
    st.session_state.portfolios = portfolios


def get_selected_portfolio() -> Optional[Portfolio]:
    """
    Get the currently selected portfolio.

    Returns:
        Selected Portfolio object or None
    """
    initialize_session_state()
    return st.session_state.selected_portfolio


def set_selected_portfolio(portfolio: Optional[Portfolio]):
    """
    Set the currently selected portfolio.

    Args:
        portfolio: Portfolio object to select or None to clear selection
    """
    initialize_session_state()
    st.session_state.selected_portfolio = portfolio


def refresh_portfolios():
    """
    Refresh the portfolio list from storage.

    This function reloads all portfolios from the portfolio manager
    and updates the session state.
    """
    try:
        portfolio_manager = get_portfolio_manager()
        portfolios = portfolio_manager.list_portfolios()
        set_portfolios(portfolios)

        # Update last refresh time
        st.session_state.last_portfolio_refresh = datetime.now()

    except Exception as e:
        st.error(f"Error refreshing portfolios: {e}")
        st.session_state.last_error = str(e)
        st.session_state.error_count += 1


def clear_form_data():
    """Clear all form data from session state."""
    if "form_data" in st.session_state:
        st.session_state.form_data.clear()


def set_form_data(key: str, value: Any):
    """
    Set form data in session state.

    Args:
        key: Form field key
        value: Form field value
    """
    initialize_session_state()
    st.session_state.form_data[key] = value


def get_form_data(key: str, default: Any = None) -> Any:
    """
    Get form data from session state.

    Args:
        key: Form field key
        default: Default value if key not found

    Returns:
        Form field value or default
    """
    initialize_session_state()
    return st.session_state.form_data.get(key, default)


def set_confirmation_state(key: str, state: bool):
    """
    Set confirmation dialog state.

    Args:
        key: Confirmation key
        state: Confirmation state (True/False)
    """
    initialize_session_state()
    st.session_state.confirm_delete_states[key] = state


def get_confirmation_state(key: str) -> bool:
    """
    Get confirmation dialog state.

    Args:
        key: Confirmation key

    Returns:
        Confirmation state (True/False)
    """
    initialize_session_state()
    return st.session_state.confirm_delete_states.get(key, False)


def clear_confirmation_state(key: str):
    """
    Clear confirmation dialog state.

    Args:
        key: Confirmation key to clear
    """
    initialize_session_state()
    if key in st.session_state.confirm_delete_states:
        del st.session_state.confirm_delete_states[key]


def update_last_price_update():
    """Update the timestamp of the last price update."""
    st.session_state.last_price_update = datetime.now()


def get_last_price_update() -> Optional[datetime]:
    """
    Get the timestamp of the last price update.

    Returns:
        Datetime of last price update or None
    """
    return st.session_state.get("last_price_update")


def clear_price_cache():
    """Clear the price cache."""
    if "price_cache" in st.session_state:
        st.session_state.price_cache.clear()

    # Also clear the price manager cache
    price_manager = get_price_manager()
    price_manager.clear_cache()


def set_current_page(page: str):
    """
    Set the current page in session state.

    Args:
        page: Page name/identifier
    """
    initialize_session_state()
    st.session_state.current_page = page


def get_current_page() -> str:
    """
    Get the current page from session state.

    Returns:
        Current page name/identifier
    """
    initialize_session_state()
    return st.session_state.current_page


def log_error(error: str):
    """
    Log an error in session state.

    Args:
        error: Error message to log
    """
    initialize_session_state()
    st.session_state.last_error = error
    st.session_state.error_count += 1


def get_last_error() -> Optional[str]:
    """
    Get the last error from session state.

    Returns:
        Last error message or None
    """
    return st.session_state.get("last_error")


def clear_errors():
    """Clear all error states."""
    if "last_error" in st.session_state:
        st.session_state.last_error = None
    if "error_count" in st.session_state:
        st.session_state.error_count = 0


def get_session_stats() -> dict:
    """
    Get session statistics.

    Returns:
        Dictionary with session statistics
    """
    initialize_session_state()

    return {
        "portfolios_count": len(st.session_state.portfolios),
        "selected_portfolio": st.session_state.selected_portfolio.name if st.session_state.selected_portfolio else None,
        "current_page": st.session_state.current_page,
        "last_price_update": st.session_state.last_price_update,
        "error_count": st.session_state.error_count,
        "cache_size": len(st.session_state.price_cache),
        "form_data_keys": list(st.session_state.form_data.keys()) if st.session_state.form_data else []
    }