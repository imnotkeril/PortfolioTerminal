"""
Portfolio Manager - CRUD operations and advanced portfolio management.

This module provides the main interface for creating, reading, updating,
and deleting portfolios, as well as import/export functionality.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import logging

from .models import (
    Portfolio, Asset, PortfolioSettings, ValidationResult,
    ConstraintViolation, Suggestion, AssetClass, PortfolioType
)
from .validators import (
    PortfolioValidator, validate_all_inputs, TextParser,
    ImportValidator, DataValidator
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Main portfolio management class providing CRUD operations,
    import/export functionality, and advanced portfolio operations.
    """

    def __init__(self, storage_path: str = "data/portfolios"):
        """
        Initialize portfolio manager

        Args:
            storage_path: Directory to store portfolio data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.validator = PortfolioValidator()

        # Cache for loaded portfolios
        self._portfolio_cache: Dict[str, Portfolio] = {}

        logger.info(f"PortfolioManager initialized with storage path: {self.storage_path}")

    # ================================
    # CRUD OPERATIONS
    # ================================

    def create_portfolio(
        self,
        name: str,
        description: str = "",
        assets: Optional[List[Asset]] = None,
        settings: Optional[PortfolioSettings] = None,
        portfolio_type: PortfolioType = PortfolioType.BALANCED,
        initial_value: float = 100000.0
    ) -> Portfolio:
        """
        Create a new portfolio

        Args:
            name: Portfolio name
            description: Portfolio description
            assets: List of assets (optional)
            settings: Portfolio settings (optional)
            portfolio_type: Type of portfolio
            initial_value: Starting portfolio value

        Returns:
            Created Portfolio object

        Raises:
            ValidationError: If portfolio data is invalid
        """
        logger.info(f"Creating portfolio: {name}")

        # Create portfolio object
        portfolio = Portfolio(
            name=name.strip(),
            description=description.strip(),
            assets=assets or [],
            settings=settings or PortfolioSettings(),
            portfolio_type=portfolio_type,
            initial_value=initial_value
        )

        # Validate portfolio
        validation_result = self.validator.validate_portfolio(portfolio)
        if not validation_result.is_valid:
            error_msg = f"Portfolio validation failed: {'; '.join(validation_result.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log warnings
        for warning in validation_result.warnings:
            logger.warning(f"Portfolio warning: {warning}")

        # Save portfolio
        self._save_portfolio(portfolio)

        # Add to cache
        self._portfolio_cache[portfolio.id] = portfolio

        logger.info(f"Portfolio created successfully: {portfolio.id}")
        return portfolio

    def get_portfolio(self, portfolio_id: str) -> Portfolio:
        """
        Get portfolio by ID

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Portfolio object

        Raises:
            FileNotFoundError: If portfolio not found
        """
        # Check cache first
        if portfolio_id in self._portfolio_cache:
            return self._portfolio_cache[portfolio_id]

        # Load from storage
        portfolio = self._load_portfolio(portfolio_id)

        # Add to cache
        self._portfolio_cache[portfolio_id] = portfolio

        return portfolio

    def update_portfolio(self, portfolio_id: str, updates: Dict[str, Any]) -> Portfolio:
        """
        Update portfolio with new data

        Args:
            portfolio_id: Portfolio ID
            updates: Dictionary of updates

        Returns:
            Updated Portfolio object
        """
        logger.info(f"Updating portfolio: {portfolio_id}")

        # Get current portfolio
        portfolio = self.get_portfolio(portfolio_id)

        # Apply updates
        for key, value in updates.items():
            if hasattr(portfolio, key):
                setattr(portfolio, key, value)

        # Update timestamp
        portfolio.last_modified = datetime.now()

        # Validate updated portfolio
        validation_result = self.validator.validate_portfolio(portfolio)
        if not validation_result.is_valid:
            error_msg = f"Updated portfolio validation failed: {'; '.join(validation_result.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Save updated portfolio
        self._save_portfolio(portfolio)

        # Update cache
        self._portfolio_cache[portfolio_id] = portfolio

        logger.info(f"Portfolio updated successfully: {portfolio_id}")
        return portfolio

    def delete_portfolio(self, portfolio_id: str) -> bool:
        """
        Delete portfolio

        Args:
            portfolio_id: Portfolio ID

        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting portfolio: {portfolio_id}")

        # Remove from cache
        if portfolio_id in self._portfolio_cache:
            del self._portfolio_cache[portfolio_id]

        # Remove file
        file_path = self.storage_path / f"{portfolio_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Portfolio deleted successfully: {portfolio_id}")
            return True

        logger.warning(f"Portfolio file not found: {portfolio_id}")
        return False

    def list_portfolios(self, filters: Optional[Dict[str, Any]] = None) -> List[Portfolio]:
        """
        List all portfolios with optional filtering

        Args:
            filters: Optional filters (name, type, tags, etc.)

        Returns:
            List of Portfolio objects
        """
        portfolios = []

        # Load all portfolio files
        for file_path in self.storage_path.glob("*.json"):
            try:
                portfolio_id = file_path.stem
                portfolio = self.get_portfolio(portfolio_id)

                # Apply filters if provided
                if filters:
                    if not self._matches_filters(portfolio, filters):
                        continue

                portfolios.append(portfolio)

            except Exception as e:
                logger.error(f"Error loading portfolio {file_path}: {e}")
                continue

        # Sort by last modified (newest first)
        portfolios.sort(key=lambda p: p.last_modified, reverse=True)

        return portfolios

    # ================================
    # IMPORT/EXPORT OPERATIONS
    # ================================

    def create_from_text(self, name: str, text: str, **kwargs) -> Portfolio:
        """
        Create portfolio from text input

        Supported formats:
        - "AAPL 30%, MSFT 25%, GOOGL 45%"
        - "AAPL 0.3, MSFT 0.25, GOOGL 0.45"
        - "AAPL:30 MSFT:25 GOOGL:45"
        - "AAPL,MSFT,GOOGL" (equal weight)

        Args:
            name: Portfolio name
            text: Text containing ticker-weight pairs
            **kwargs: Additional portfolio parameters

        Returns:
            Created Portfolio object
        """
        logger.info(f"Creating portfolio from text: {name}")

        # Validate text input
        validation_result = ImportValidator.validate_text_input(text)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid text input: {'; '.join(validation_result.errors)}")

        # Parse text
        parsed_data = TextParser.parse_text_input(text)
        if not parsed_data:
            raise ValueError("Could not parse any valid ticker-weight pairs from text")

        # Create assets
        assets = []
        for data in parsed_data:
            asset = Asset(
                ticker=data['ticker'],
                weight=data['weight'],
                name=data.get('name', ''),  # Will be fetched later
                asset_class=AssetClass.STOCK  # Default, will be updated
            )
            assets.append(asset)

        # Normalize weights
        total_weight = sum(asset.weight for asset in assets)
        if total_weight > 0:
            for asset in assets:
                asset.weight = asset.weight / total_weight

        # Create portfolio
        return self.create_portfolio(
            name=name,
            assets=assets,
            **kwargs
        )

    def import_from_csv(
        self,
        file_path: str,
        name: str,
        mapping: Optional[Dict[str, str]] = None
    ) -> Portfolio:
        """
        Import portfolio from CSV file

        Args:
            file_path: Path to CSV file
            name: Portfolio name
            mapping: Column mapping {'ticker': 'Symbol', 'weight': 'Allocation'}

        Returns:
            Created Portfolio object
        """
        logger.info(f"Importing portfolio from CSV: {file_path}")

        # Read CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Validate CSV data
        validation_result = ImportValidator.validate_csv_data(df)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid CSV data: {'; '.join(validation_result.errors)}")

        # Apply column mapping if provided
        if mapping:
            df = df.rename(columns=mapping)

        # Clean column names (strip whitespace, lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # Create assets from DataFrame
        assets = []
        for _, row in df.iterrows():
            # Get ticker (required)
            ticker = row.get('ticker') or row.get('symbol') or row.get('tick')
            if pd.isna(ticker):
                continue

            ticker = DataValidator.clean_ticker(str(ticker))
            if not ticker:
                continue

            # Get weight (required)
            weight = row.get('weight') or row.get('allocation') or row.get('percent')
            if pd.isna(weight):
                weight = 0.0
            else:
                weight = float(weight)
                # Convert percentage to decimal if needed
                if weight > 1:
                    weight = weight / 100.0

            # Get optional fields
            name = row.get('name') or row.get('company_name') or ''
            if pd.isna(name):
                name = ''

            sector = row.get('sector') or row.get('industry') or ''
            if pd.isna(sector):
                sector = ''

            shares = row.get('shares') or row.get('quantity') or 0.0
            if pd.isna(shares):
                shares = 0.0
            else:
                shares = float(shares)

            purchase_price = row.get('purchase_price') or row.get('cost_basis')
            if pd.isna(purchase_price):
                purchase_price = None
            else:
                purchase_price = float(purchase_price)

            # Create asset
            asset = Asset(
                ticker=ticker,
                name=str(name).strip(),
                weight=weight,
                shares=shares,
                purchase_price=purchase_price,
                sector=str(sector).strip(),
                asset_class=AssetClass.STOCK  # Default
            )

            assets.append(asset)

        if not assets:
            raise ValueError("No valid assets found in CSV file")

        # Normalize weights
        total_weight = sum(asset.weight for asset in assets)
        if total_weight > 0:
            for asset in assets:
                asset.weight = asset.weight / total_weight

        # Create portfolio
        return self.create_portfolio(name=name, assets=assets)

    def import_from_excel(
        self,
        file_path: str,
        name: str,
        sheet_name: str = 0,
        mapping: Optional[Dict[str, str]] = None
    ) -> Portfolio:
        """
        Import portfolio from Excel file

        Args:
            file_path: Path to Excel file
            name: Portfolio name
            sheet_name: Sheet name or index
            mapping: Column mapping

        Returns:
            Created Portfolio object
        """
        logger.info(f"Importing portfolio from Excel: {file_path}, sheet: {sheet_name}")

        try:
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Save as temporary CSV and use CSV import
            temp_csv = self.storage_path / "temp_import.csv"
            df.to_csv(temp_csv, index=False)

            # Use CSV import
            portfolio = self.import_from_csv(str(temp_csv), name, mapping)

            # Clean up temp file
            temp_csv.unlink()

            return portfolio

        except Exception as e:
            raise ValueError(f"Error importing from Excel: {e}")

    def import_from_json(self, file_path: str) -> Portfolio:
        """
        Import portfolio from JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            Portfolio object
        """
        logger.info(f"Importing portfolio from JSON: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            portfolio = Portfolio.from_dict(data)

            # Validate imported portfolio
            validation_result = self.validator.validate_portfolio(portfolio)
            if not validation_result.is_valid:
                raise ValueError(f"Invalid portfolio data: {'; '.join(validation_result.errors)}")

            # Save portfolio (will generate new ID if needed)
            self._save_portfolio(portfolio)

            return portfolio

        except Exception as e:
            raise ValueError(f"Error importing from JSON: {e}")

    def export_to_csv(self, portfolio_id: str, file_path: str) -> str:
        """
        Export portfolio to CSV file

        Args:
            portfolio_id: Portfolio ID
            file_path: Output file path

        Returns:
            Path to created file
        """
        logger.info(f"Exporting portfolio to CSV: {portfolio_id}")

        portfolio = self.get_portfolio(portfolio_id)

        # Create DataFrame
        data = []
        for asset in portfolio.assets:
            data.append({
                'ticker': asset.ticker,
                'name': asset.name,
                'weight': asset.weight,
                'weight_percent': asset.weight * 100,
                'shares': asset.shares,
                'current_price': asset.current_price,
                'market_value': asset.market_value,
                'purchase_price': asset.purchase_price,
                'unrealized_pnl': asset.unrealized_pnl,
                'unrealized_pnl_percent': asset.unrealized_pnl_percent * 100,
                'sector': asset.sector,
                'asset_class': asset.asset_class.value,
                'currency': asset.currency,
                'exchange': asset.exchange
            })

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        logger.info(f"Portfolio exported to CSV: {file_path}")
        return file_path

    def export_to_excel(self, portfolio_id: str, file_path: str) -> str:
        """
        Export portfolio to Excel file with multiple sheets

        Args:
            portfolio_id: Portfolio ID
            file_path: Output file path

        Returns:
            Path to created file
        """
        logger.info(f"Exporting portfolio to Excel: {portfolio_id}")

        portfolio = self.get_portfolio(portfolio_id)

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Holdings sheet
            holdings_data = []
            for asset in portfolio.assets:
                holdings_data.append({
                    'Ticker': asset.ticker,
                    'Name': asset.name,
                    'Weight %': asset.weight * 100,
                    'Shares': asset.shares,
                    'Current Price': asset.current_price,
                    'Market Value': asset.market_value,
                    'Sector': asset.sector,
                    'Asset Class': asset.asset_class.value
                })

            holdings_df = pd.DataFrame(holdings_data)
            holdings_df.to_excel(writer, sheet_name='Holdings', index=False)

            # Portfolio info sheet
            info_data = {
                'Attribute': ['Name', 'Description', 'Created Date', 'Total Assets', 'Total Value'],
                'Value': [
                    portfolio.name,
                    portfolio.description,
                    portfolio.created_date.strftime('%Y-%m-%d'),
                    len(portfolio.assets),
                    portfolio.calculate_value()
                ]
            }
            info_df = pd.DataFrame(info_data)
            info_df.to_excel(writer, sheet_name='Info', index=False)

            # Sector allocation sheet
            sectors = portfolio.get_sector_allocation()
            if sectors:
                sector_df = pd.DataFrame([
                    {'Sector': sector, 'Weight %': weight * 100}
                    for sector, weight in sectors.items()
                ])
                sector_df.to_excel(writer, sheet_name='Sectors', index=False)

        logger.info(f"Portfolio exported to Excel: {file_path}")
        return file_path

    def export_to_json(self, portfolio_id: str, file_path: str) -> str:
        """
        Export portfolio to JSON file

        Args:
            portfolio_id: Portfolio ID
            file_path: Output file path

        Returns:
            Path to created file
        """
        logger.info(f"Exporting portfolio to JSON: {portfolio_id}")

        portfolio = self.get_portfolio(portfolio_id)

        with open(file_path, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2)

        logger.info(f"Portfolio exported to JSON: {file_path}")
        return file_path

    # ================================
    # ADVANCED OPERATIONS
    # ================================

    def clone_portfolio(self, portfolio_id: str, new_name: str) -> Portfolio:
        """
        Clone existing portfolio

        Args:
            portfolio_id: Source portfolio ID
            new_name: New portfolio name

        Returns:
            Cloned Portfolio object
        """
        logger.info(f"Cloning portfolio: {portfolio_id} -> {new_name}")

        # Get source portfolio
        source = self.get_portfolio(portfolio_id)

        # Create copy
        cloned = source.copy()
        cloned.name = new_name
        cloned.created_date = datetime.now()
        cloned.last_modified = datetime.now()
        cloned.trade_history = []  # Start with empty trade history

        # Save cloned portfolio
        self._save_portfolio(cloned)

        logger.info(f"Portfolio cloned successfully: {cloned.id}")
        return cloned

    def merge_portfolios(
        self,
        portfolio_ids: List[str],
        new_name: str,
        merge_method: str = "proportional"
    ) -> Portfolio:
        """
        Merge multiple portfolios into one

        Args:
            portfolio_ids: List of portfolio IDs to merge
            new_name: Name for merged portfolio
            merge_method: How to merge weights ('proportional', 'equal', 'value_weighted')

        Returns:
            Merged Portfolio object
        """
        logger.info(f"Merging portfolios: {portfolio_ids} -> {new_name}")

        if len(portfolio_ids) < 2:
            raise ValueError("At least 2 portfolios required for merging")

        # Load all portfolios
        portfolios = [self.get_portfolio(pid) for pid in portfolio_ids]

        # Collect all unique assets
        all_assets = {}
        total_values = []

        for portfolio in portfolios:
            portfolio_value = portfolio.calculate_value()
            total_values.append(portfolio_value)

            for asset in portfolio.assets:
                if asset.ticker in all_assets:
                    # Asset exists, merge weights
                    existing_asset = all_assets[asset.ticker]

                    if merge_method == "proportional":
                        # Weight by portfolio value
                        portfolio_weight = portfolio_value / sum(total_values) if sum(total_values) > 0 else 0
                        existing_asset.weight += asset.weight * portfolio_weight
                    elif merge_method == "equal":
                        # Equal weighting of portfolios
                        portfolio_weight = 1.0 / len(portfolios)
                        existing_asset.weight += asset.weight * portfolio_weight

                else:
                    # New asset
                    new_asset = Asset(
                        ticker=asset.ticker,
                        name=asset.name,
                        weight=asset.weight,
                        sector=asset.sector,
                        asset_class=asset.asset_class
                    )

                    if merge_method == "proportional":
                        portfolio_weight = portfolio_value / sum(total_values) if sum(total_values) > 0 else 0
                        new_asset.weight *= portfolio_weight
                    elif merge_method == "equal":
                        portfolio_weight = 1.0 / len(portfolios)
                        new_asset.weight *= portfolio_weight

                    all_assets[asset.ticker] = new_asset

        # Convert to list and normalize
        merged_assets = list(all_assets.values())
        total_weight = sum(asset.weight for asset in merged_assets)
        if total_weight > 0:
            for asset in merged_assets:
                asset.weight = asset.weight / total_weight

        # Create merged portfolio
        return self.create_portfolio(
            name=new_name,
            description=f"Merged from: {', '.join([p.name for p in portfolios])}",
            assets=merged_assets
        )

    def split_portfolio(
        self,
        portfolio_id: str,
        split_ratio: float,
        names: Tuple[str, str]
    ) -> Tuple[Portfolio, Portfolio]:
        """
        Split portfolio into two portfolios

        Args:
            portfolio_id: Portfolio to split
            split_ratio: Ratio for first portfolio (0.0 - 1.0)
            names: Names for the two new portfolios

        Returns:
            Tuple of (first_portfolio, second_portfolio)
        """
        logger.info(f"Splitting portfolio: {portfolio_id} at ratio {split_ratio}")

        if not 0 < split_ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1")

        # Get source portfolio
        source = self.get_portfolio(portfolio_id)

        # Create two portfolios with same assets but different weights
        first_assets = []
        second_assets = []

        for asset in source.assets:
            # First portfolio gets split_ratio of the weight
            first_asset = Asset(
                ticker=asset.ticker,
                name=asset.name,
                weight=asset.weight,  # Will be normalized
                sector=asset.sector,
                asset_class=asset.asset_class,
                shares=asset.shares * split_ratio,
                purchase_price=asset.purchase_price,
                current_price=asset.current_price
            )
            first_assets.append(first_asset)

            # Second portfolio gets remaining weight
            second_asset = Asset(
                ticker=asset.ticker,
                name=asset.name,
                weight=asset.weight,  # Will be normalized
                sector=asset.sector,
                asset_class=asset.asset_class,
                shares=asset.shares * (1 - split_ratio),
                purchase_price=asset.purchase_price,
                current_price=asset.current_price
            )
            second_assets.append(second_asset)

        # Create portfolios
        first_portfolio = self.create_portfolio(
            name=names[0],
            assets=first_assets,
            settings=source.settings.copy() if hasattr(source.settings, 'copy') else PortfolioSettings(),
            initial_value=source.initial_value * split_ratio
        )

        second_portfolio = self.create_portfolio(
            name=names[1],
            assets=second_assets,
            settings=source.settings.copy() if hasattr(source.settings, 'copy') else PortfolioSettings(),
            initial_value=source.initial_value * (1 - split_ratio)
        )

        logger.info(f"Portfolio split successfully: {first_portfolio.id}, {second_portfolio.id}")
        return first_portfolio, second_portfolio

    # ================================
    # VALIDATION AND SUGGESTIONS
    # ================================

    def validate_portfolio(self, portfolio: Portfolio) -> ValidationResult:
        """Validate portfolio using comprehensive validator"""
        return self.validator.validate_portfolio(portfolio)

    def check_constraints(self, portfolio: Portfolio) -> List[ConstraintViolation]:
        """Check portfolio constraints"""
        return self.validator.check_constraints(portfolio)

    def suggest_fixes(self, portfolio: Portfolio) -> List[Suggestion]:
        """Generate suggestions for portfolio improvement"""
        return self.validator.suggest_improvements(portfolio)

    # ================================
    # PRIVATE METHODS
    # ================================

    def _save_portfolio(self, portfolio: Portfolio) -> None:
        """Save portfolio to storage"""
        file_path = self.storage_path / f"{portfolio.id}.json"

        try:
            with open(file_path, 'w') as f:
                json.dump(portfolio.to_dict(), f, indent=2)

            logger.debug(f"Portfolio saved: {file_path}")

        except Exception as e:
            logger.error(f"Error saving portfolio {portfolio.id}: {e}")
            raise

    def _load_portfolio(self, portfolio_id: str) -> Portfolio:
        """Load portfolio from storage"""
        file_path = self.storage_path / f"{portfolio_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Portfolio not found: {portfolio_id}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            portfolio = Portfolio.from_dict(data)
            logger.debug(f"Portfolio loaded: {portfolio_id}")

            return portfolio

        except Exception as e:
            logger.error(f"Error loading portfolio {portfolio_id}: {e}")
            raise

    def _matches_filters(self, portfolio: Portfolio, filters: Dict[str, Any]) -> bool:
        """Check if portfolio matches filter criteria"""
        for key, value in filters.items():
            if key == 'name':
                if value.lower() not in portfolio.name.lower():
                    return False

            elif key == 'type':
                if portfolio.portfolio_type.value != value:
                    return False

            elif key == 'tags':
                if isinstance(value, list):
                    # All tags must be present
                    if not all(tag in portfolio.tags for tag in value):
                        return False
                else:
                    # Single tag
                    if value not in portfolio.tags:
                        return False

            elif key == 'min_assets':
                if len(portfolio.assets) < value:
                    return False

            elif key == 'max_assets':
                if len(portfolio.assets) > value:
                    return False

            elif key == 'created_after':
                if portfolio.created_date < value:
                    return False

            elif key == 'created_before':
                if portfolio.created_date > value:
                    return False

        return True

    # ================================
    # UTILITY METHODS
    # ================================

    def get_portfolio_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio summary information"""
        portfolio = self.get_portfolio(portfolio_id)
        stats = portfolio.get_statistics()

        return {
            'id': portfolio.id,
            'name': portfolio.name,
            'description': portfolio.description,
            'type': portfolio.portfolio_type.value,
            'created_date': portfolio.created_date.isoformat(),
            'last_modified': portfolio.last_modified.isoformat(),
            'asset_count': len(portfolio.assets),
            'total_value': stats.total_value,
            'unrealized_pnl': stats.unrealized_pnl,
            'unrealized_pnl_percent': stats.unrealized_pnl_percent,
            'tags': portfolio.tags,
            'sectors': portfolio.get_sector_allocation()
        }

    def search_portfolios(self, query: str) -> List[Portfolio]:
        """Search portfolios by name, description, or tags"""
        all_portfolios = self.list_portfolios()
        query = query.lower().strip()

        matching_portfolios = []

        for portfolio in all_portfolios:
            # Search in name
            if query in portfolio.name.lower():
                matching_portfolios.append(portfolio)
                continue

            # Search in description
            if query in portfolio.description.lower():
                matching_portfolios.append(portfolio)
                continue

            # Search in tags
            if any(query in tag.lower() for tag in portfolio.tags):
                matching_portfolios.append(portfolio)
                continue

            # Search in asset tickers or names
            if any(query in asset.ticker.lower() or query in asset.name.lower()
                   for asset in portfolio.assets):
                matching_portfolios.append(portfolio)
                continue

        return matching_portfolios

    def get_portfolio_count(self) -> int:
        """Get total number of portfolios"""
        return len(list(self.storage_path.glob("*.json")))

    def clear_cache(self) -> None:
        """Clear portfolio cache"""
        self._portfolio_cache.clear()
        logger.info("Portfolio cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_portfolios': len(self._portfolio_cache),
            'cache_size_mb': sum(
                len(str(p.to_dict())) for p in self._portfolio_cache.values()
            ) / (1024 * 1024)
        }