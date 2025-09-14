"""
Formatting utilities for the Portfolio Management System.

This module contains all formatting functions used throughout the application
for consistent data presentation.
"""
from typing import Optional, Union
import pandas as pd
from datetime import datetime


def format_currency(value: Optional[Union[float, int]], currency: str = "USD") -> str:
    """
    Format a number as currency.

    Args:
        value: The numeric value to format
        currency: Currency symbol (default: USD)

    Returns:
        Formatted currency string

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(-500.00)
        '-$500.00'
        >>> format_currency(None)
        'N/A'
    """
    if value is None:
        return "N/A"

    try:
        value = float(value)
        if currency == "USD":
            if value >= 0:
                return f"${value:,.2f}"
            else:
                return f"-${abs(value):,.2f}"
        else:
            # For other currencies, use generic formatting
            return f"{value:,.2f} {currency}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(value: Optional[Union[float, int]], decimals: int = 2) -> str:
    """
    Format a number as percentage.

    Args:
        value: The numeric value to format (0.15 = 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(0.1567)
        '15.67%'
        >>> format_percentage(-0.05)
        '-5.00%'
        >>> format_percentage(None)
        'N/A'
    """
    if value is None:
        return "N/A"

    try:
        value = float(value)
        return f"{value * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_number(value: Optional[Union[float, int]], decimals: int = 2) -> str:
    """
    Format a number with thousands separators.

    Args:
        value: The numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted number string

    Examples:
        >>> format_number(1234567.89)
        '1,234,567.89'
        >>> format_number(None)
        'N/A'
    """
    if value is None:
        return "N/A"

    try:
        value = float(value)
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_datetime(dt: Optional[datetime], format_str: str = "%Y-%m-%d %H:%M") -> str:
    """
    Format a datetime object.

    Args:
        dt: Datetime object to format
        format_str: Strftime format string

    Returns:
        Formatted datetime string

    Examples:
        >>> format_datetime(datetime(2024, 1, 15, 10, 30))
        '2024-01-15 10:30'
        >>> format_datetime(None)
        'N/A'
    """
    if dt is None:
        return "N/A"

    try:
        return dt.strftime(format_str)
    except (AttributeError, ValueError):
        return "N/A"


def format_large_number(value: Optional[Union[float, int]]) -> str:
    """
    Format large numbers with abbreviations (K, M, B, T).

    Args:
        value: The numeric value to format

    Returns:
        Formatted number string with abbreviation

    Examples:
        >>> format_large_number(1500)
        '1.5K'
        >>> format_large_number(2500000)
        '2.5M'
        >>> format_large_number(1200000000)
        '1.2B'
    """
    if value is None:
        return "N/A"

    try:
        value = float(abs(value))

        if value >= 1_000_000_000_000:  # Trillions
            return f"{value / 1_000_000_000_000:.1f}T"
        elif value >= 1_000_000_000:  # Billions
            return f"{value / 1_000_000_000:.1f}B"
        elif value >= 1_000_000:  # Millions
            return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:  # Thousands
            return f"{value / 1_000:.1f}K"
        else:
            return f"{value:.0f}"
    except (ValueError, TypeError):
        return "N/A"


def get_color_class(value: Optional[Union[float, int]]) -> str:
    """
    Get CSS color class based on positive/negative value.

    Args:
        value: Numeric value to check

    Returns:
        CSS class string ('positive', 'negative', or '')

    Examples:
        >>> get_color_class(10.5)
        'positive'
        >>> get_color_class(-5.2)
        'negative'
        >>> get_color_class(0)
        ''
    """
    if value is None:
        return ""

    try:
        value = float(value)
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return ""
    except (ValueError, TypeError):
        return ""


def format_risk_level(value: Optional[Union[float, int]]) -> tuple[str, str]:
    """
    Format risk level value and return color.

    Args:
        value: Risk level value (0-1 scale)

    Returns:
        Tuple of (formatted_string, color_class)

    Examples:
        >>> format_risk_level(0.15)
        ('Low (15%)', 'success')
        >>> format_risk_level(0.45)
        ('Medium (45%)', 'warning')
        >>> format_risk_level(0.75)
        ('High (75%)', 'danger')
    """
    if value is None:
        return ("N/A", "secondary")

    try:
        value = float(value)
        percentage = value * 100

        if value <= 0.25:
            return (f"Low ({percentage:.0f}%)", "success")
        elif value <= 0.50:
            return (f"Medium ({percentage:.0f}%)", "warning")
        elif value <= 0.75:
            return (f"High ({percentage:.0f}%)", "danger")
        else:
            return (f"Very High ({percentage:.0f}%)", "danger")
    except (ValueError, TypeError):
        return ("N/A", "secondary")


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame for display in Streamlit.

    Args:
        df: DataFrame to format

    Returns:
        Formatted DataFrame
    """
    if df is None or df.empty:
        return df

    # Create a copy to avoid modifying original
    formatted_df = df.copy()

    # Format common column types
    for col in formatted_df.columns:
        col_lower = col.lower()

        # Currency columns
        if any(term in col_lower for term in ['value', 'price', 'cost', 'pnl', 'profit', 'loss']):
            formatted_df[col] = formatted_df[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")

        # Percentage columns
        elif any(term in col_lower for term in ['weight', 'percent', 'return', 'change', 'allocation']):
            formatted_df[col] = formatted_df[col].apply(lambda x: format_percentage(x) if pd.notna(x) else "N/A")

        # Date columns
        elif any(term in col_lower for term in ['date', 'time', 'created', 'modified']):
            formatted_df[col] = formatted_df[col].apply(lambda x: format_datetime(x) if pd.notna(x) else "N/A")

        # Number columns (shares, quantities)
        elif any(term in col_lower for term in ['shares', 'quantity', 'count']):
            formatted_df[col] = formatted_df[col].apply(lambda x: format_number(x, 0) if pd.notna(x) else "N/A")

    return formatted_df