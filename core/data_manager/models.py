"""
Core data models for portfolio management system.

This module contains the fundamental data structures for portfolios, assets,
and related components following the technical specification.
"""

import uuid
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np


class AssetClass(Enum):
    """Asset classification types"""
    STOCK = "stock"
    BOND = "bond"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"
    REAL_ESTATE = "real_estate"
    CASH = "cash"
    OTHER = "other"


class PortfolioType(Enum):
    """Portfolio type classifications"""
    GROWTH = "growth"
    INCOME = "income"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    INDEX = "index"
    CUSTOM = "custom"


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ESGSettings:
    """ESG (Environmental, Social, Governance) settings"""
    enabled: bool = False
    exclude_tobacco: bool = False
    exclude_weapons: bool = False
    exclude_fossil_fuels: bool = False
    min_esg_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ESGSettings':
        return cls(**data)


@dataclass
class GeographicConstraints:
    """Geographic investment constraints"""
    us_only: bool = False
    developed_markets: bool = True
    emerging_markets: bool = False
    frontier_markets: bool = False
    excluded_countries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeographicConstraints':
        return cls(**data)


@dataclass
class PortfolioSettings:
    """Advanced portfolio configuration settings"""

    # Rebalancing Settings
    rebalancing_frequency: str = "quarterly"  # monthly/quarterly/annual/manual
    rebalancing_threshold: float = 0.05  # 5% threshold
    auto_rebalance: bool = False

    # Position Constraints
    min_position_size: float = 1000.0  # Minimum position in USD
    max_position_size: float = 50000.0  # Maximum position in USD
    max_positions: int = 50  # Maximum number of positions
    position_sizing: str = "equal"  # equal/custom/market_cap

    # Risk Management
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage
    max_drawdown: float = 0.20  # Maximum allowed drawdown (20%)

    # Tax Settings
    tax_loss_harvesting: bool = False
    wash_sale_prevention: bool = True
    tax_rate: float = 0.25  # 25% tax rate

    # ESG Settings
    esg_settings: ESGSettings = field(default_factory=ESGSettings)

    # Geographic Constraints
    geographic_constraints: GeographicConstraints = field(default_factory=GeographicConstraints)

    # Sector Constraints (sector_name: (min_weight, max_weight))
    sector_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Advanced Settings
    allow_short_selling: bool = False
    use_leverage: bool = False
    max_leverage: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['esg_settings'] = self.esg_settings.to_dict()
        data['geographic_constraints'] = self.geographic_constraints.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioSettings':
        """Create from dictionary"""
        esg_data = data.pop('esg_settings', {})
        geo_data = data.pop('geographic_constraints', {})

        settings = cls(**data)
        settings.esg_settings = ESGSettings.from_dict(esg_data)
        settings.geographic_constraints = GeographicConstraints.from_dict(geo_data)

        return settings


@dataclass
class Asset:
    """Individual asset in a portfolio"""

    # Basic Information
    ticker: str
    name: str = ""
    weight: float = 0.0  # Portfolio weight (0.0 to 1.0)
    shares: float = 0.0  # Number of shares

    # Price Information
    purchase_price: Optional[float] = None
    current_price: Optional[float] = None
    purchase_date: Optional[datetime] = None

    # Classification
    sector: str = ""
    asset_class: AssetClass = AssetClass.STOCK
    currency: str = "USD"
    exchange: str = ""

    # Identifiers
    cusip: str = ""
    isin: str = ""

    # Additional Data
    market_cap: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None

    # ESG Data
    esg_score: Optional[float] = None

    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate asset data after initialization"""
        if not self.ticker:
            raise ValueError("Ticker symbol is required")

        self.ticker = self.ticker.upper().strip()

        if self.weight < 0 or self.weight > 1:
            raise ValueError("Weight must be between 0 and 1")

    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        if self.current_price is None or self.shares <= 0:
            return 0.0
        return self.current_price * self.shares

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss"""
        if (self.purchase_price is None or
            self.current_price is None or
            self.shares <= 0):
            return 0.0

        return (self.current_price - self.purchase_price) * self.shares

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage"""
        if (self.purchase_price is None or
            self.purchase_price <= 0 or
            self.current_price is None):
            return 0.0

        return (self.current_price - self.purchase_price) / self.purchase_price

    def validate(self) -> bool:
        """Validate asset data integrity"""
        try:
            # Basic validation
            if not self.ticker or len(self.ticker) < 1:
                return False

            if self.weight < 0 or self.weight > 1:
                return False

            if self.shares < 0:
                return False

            if self.purchase_price is not None and self.purchase_price <= 0:
                return False

            if self.current_price is not None and self.current_price <= 0:
                return False

            return True

        except Exception:
            return False

    def update_price(self, price: float) -> None:
        """Update current price"""
        if price <= 0:
            raise ValueError("Price must be positive")

        self.current_price = price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)

        # Handle datetime serialization
        if self.purchase_date:
            data['purchase_date'] = self.purchase_date.isoformat()

        # Handle enum serialization
        data['asset_class'] = self.asset_class.value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Asset':
        """Create Asset from dictionary"""
        # Handle datetime deserialization
        if 'purchase_date' in data and data['purchase_date']:
            data['purchase_date'] = datetime.fromisoformat(data['purchase_date'])

        # Handle enum deserialization
        if 'asset_class' in data:
            data['asset_class'] = AssetClass(data['asset_class'])

        return cls(**data)


@dataclass
class PortfolioStats:
    """Portfolio statistics summary"""
    total_value: float = 0.0
    total_cost: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    asset_count: int = 0
    cash_weight: float = 0.0

    # Performance metrics (will be calculated by analytics engine)
    daily_return: Optional[float] = None
    ytd_return: Optional[float] = None
    total_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trade:
    """Trade execution record"""
    asset_ticker: str
    action: str  # 'buy' or 'sell'
    shares: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    notes: str = ""

    @property
    def value(self) -> float:
        """Trade value (shares * price)"""
        return abs(self.shares * self.price)

    @property
    def total_cost(self) -> float:
        """Total cost including commission"""
        return self.value + self.commission

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Portfolio:
    """Main portfolio data structure with complete functionality"""

    # Basic Information
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Timestamps
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

    # Core Data
    assets: List[Asset] = field(default_factory=list)
    initial_value: float = 100000.0  # Starting portfolio value

    # Configuration
    portfolio_type: PortfolioType = PortfolioType.BALANCED
    settings: PortfolioSettings = field(default_factory=PortfolioSettings)

    # Categorization
    tags: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Trade History
    trade_history: List[Trade] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation"""
        if not self.name:
            self.name = f"Portfolio_{self.id[:8]}"

    @property
    def total_weight(self) -> float:
        """Calculate total weight of all assets"""
        return sum(asset.weight for asset in self.assets)

    @property
    def asset_count(self) -> int:
        """Number of assets in portfolio"""
        return len(self.assets)

    @property
    def tickers(self) -> List[str]:
        """List of all asset tickers"""
        return [asset.ticker for asset in self.assets]

    @property
    def sectors(self) -> Dict[str, float]:
        """Sector allocation as dictionary"""
        sector_weights = {}
        for asset in self.assets:
            if asset.sector:
                if asset.sector in sector_weights:
                    sector_weights[asset.sector] += asset.weight
                else:
                    sector_weights[asset.sector] = asset.weight
        return sector_weights

    def validate(self) -> bool:
        """Validate portfolio data integrity"""
        try:
            # Check basic requirements
            if not self.name or not self.id:
                return False

            # Validate all assets
            for asset in self.assets:
                if not asset.validate():
                    return False

            # Check weight constraints
            total_weight = self.total_weight
            if abs(total_weight - 1.0) > 0.001:  # Allow small rounding errors
                return False

            # Check for duplicate tickers
            tickers = [asset.ticker for asset in self.assets]
            if len(tickers) != len(set(tickers)):
                return False

            return True

        except Exception:
            return False

    def normalize_weights(self) -> None:
        """Normalize asset weights to sum to 1.0"""
        total_weight = self.total_weight

        if total_weight > 0:
            for asset in self.assets:
                asset.weight = asset.weight / total_weight

        self.last_modified = datetime.now()

    def calculate_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate current portfolio value"""
        if not prices:
            # Use stored current prices
            total_value = sum(asset.market_value for asset in self.assets)
        else:
            # Use provided prices
            total_value = 0.0
            for asset in self.assets:
                if asset.ticker in prices and asset.shares > 0:
                    total_value += asset.shares * prices[asset.ticker]

        return total_value

    def get_statistics(self, prices: Optional[Dict[str, float]] = None) -> PortfolioStats:
        """Calculate portfolio statistics"""
        stats = PortfolioStats()

        stats.asset_count = self.asset_count
        stats.total_value = self.calculate_value(prices)
        stats.total_cost = sum(
            (asset.purchase_price or 0) * asset.shares
            for asset in self.assets
        )
        stats.unrealized_pnl = sum(asset.unrealized_pnl for asset in self.assets)

        if stats.total_cost > 0:
            stats.unrealized_pnl_percent = stats.unrealized_pnl / stats.total_cost

        return stats

    def add_asset(self, asset: Asset) -> None:
        """Add asset to portfolio"""
        # Check for duplicate ticker
        if any(a.ticker == asset.ticker for a in self.assets):
            raise ValueError(f"Asset {asset.ticker} already exists in portfolio")

        # Validate asset
        if not asset.validate():
            raise ValueError(f"Invalid asset data for {asset.ticker}")

        self.assets.append(asset)
        self.last_modified = datetime.now()

    def remove_asset(self, ticker: str) -> None:
        """Remove asset from portfolio"""
        ticker = ticker.upper().strip()

        # Find and remove asset
        for i, asset in enumerate(self.assets):
            if asset.ticker == ticker:
                del self.assets[i]
                self.last_modified = datetime.now()
                return

        raise ValueError(f"Asset {ticker} not found in portfolio")

    def update_asset(self, ticker: str, updates: Dict[str, Any]) -> None:
        """Update asset properties"""
        ticker = ticker.upper().strip()

        # Find asset
        for asset in self.assets:
            if asset.ticker == ticker:
                # Update allowed fields
                for key, value in updates.items():
                    if hasattr(asset, key):
                        setattr(asset, key, value)

                # Validate after update
                if not asset.validate():
                    raise ValueError(f"Invalid update for {ticker}")

                self.last_modified = datetime.now()
                return

        raise ValueError(f"Asset {ticker} not found in portfolio")

    def get_asset(self, ticker: str) -> Optional[Asset]:
        """Get asset by ticker"""
        ticker = ticker.upper().strip()
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        return None

    def get_weights_dict(self) -> Dict[str, float]:
        """Get asset weights as dictionary"""
        return {asset.ticker: asset.weight for asset in self.assets}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set asset weights from dictionary"""
        for ticker, weight in weights.items():
            asset = self.get_asset(ticker)
            if asset:
                asset.weight = weight

        self.last_modified = datetime.now()

    def rebalance(self, target_weights: Dict[str, float]) -> List[Trade]:
        """Generate trades for rebalancing to target weights"""
        trades = []
        current_value = self.calculate_value()

        for ticker, target_weight in target_weights.items():
            asset = self.get_asset(ticker)
            if not asset or not asset.current_price:
                continue

            current_weight = asset.weight
            target_value = current_value * target_weight
            current_value_asset = current_value * current_weight

            value_diff = target_value - current_value_asset

            if abs(value_diff) > self.settings.min_position_size:
                shares_diff = value_diff / asset.current_price

                action = "buy" if shares_diff > 0 else "sell"
                trade = Trade(
                    asset_ticker=ticker,
                    action=action,
                    shares=abs(shares_diff),
                    price=asset.current_price,
                    notes=f"Rebalancing: {current_weight:.2%} → {target_weight:.2%}"
                )
                trades.append(trade)

        return trades

    def add_trade(self, trade: Trade) -> None:
        """Add trade to history"""
        self.trade_history.append(trade)

        # Update asset if it exists
        asset = self.get_asset(trade.asset_ticker)
        if asset:
            if trade.action == "buy":
                asset.shares += trade.shares
            else:  # sell
                asset.shares -= trade.shares
                asset.shares = max(0, asset.shares)  # Prevent negative shares

    def get_sector_allocation(self) -> Dict[str, float]:
        """Get allocation by sector"""
        return self.sectors

    def get_asset_class_allocation(self) -> Dict[str, float]:
        """Get allocation by asset class"""
        allocation = {}
        for asset in self.assets:
            asset_class = asset.asset_class.value
            if asset_class in allocation:
                allocation[asset_class] += asset.weight
            else:
                allocation[asset_class] = asset.weight
        return allocation

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary for serialization"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_date': self.created_date.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'initial_value': self.initial_value,
            'portfolio_type': self.portfolio_type.value,
            'tags': self.tags,
            'metadata': self.metadata,
            'assets': [asset.to_dict() for asset in self.assets],
            'settings': self.settings.to_dict(),
            'trade_history': [trade.to_dict() for trade in self.trade_history]
        }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create portfolio from dictionary"""
        # Handle datetime fields
        created_date = datetime.fromisoformat(data['created_date'])
        last_modified = datetime.fromisoformat(data['last_modified'])

        # Handle enum fields
        portfolio_type = PortfolioType(data['portfolio_type'])

        # Handle nested objects
        assets = [Asset.from_dict(asset_data) for asset_data in data.get('assets', [])]
        settings = PortfolioSettings.from_dict(data.get('settings', {}))
        trade_history = [Trade(**trade_data) for trade_data in data.get('trade_history', [])]

        # Create portfolio
        portfolio = cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            created_date=created_date,
            last_modified=last_modified,
            initial_value=data.get('initial_value', 100000.0),
            portfolio_type=portfolio_type,
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            assets=assets,
            settings=settings,
            trade_history=trade_history
        )

        return portfolio

    def copy(self) -> 'Portfolio':
        """Create a deep copy of the portfolio"""
        return Portfolio.from_dict(self.to_dict())

    def __str__(self) -> str:
        """String representation"""
        return f"Portfolio(name='{self.name}', assets={self.asset_count}, value=${self.calculate_value():,.2f})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"Portfolio(id='{self.id}', name='{self.name}', "
                f"assets={self.asset_count}, total_weight={self.total_weight:.3f})")


@dataclass
class ValidationResult:
    """Result of portfolio validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add validation error"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add validation warning"""
        self.warnings.append(message)

    @property
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings"""
        return len(self.errors) > 0 or len(self.warnings) > 0


@dataclass
class ConstraintViolation:
    """Constraint violation record"""
    constraint_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggested_fix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Suggestion:
    """Optimization or improvement suggestion"""
    suggestion_type: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    implementation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Exception Classes
class PortfolioError(Exception):
    """Base exception for portfolio operations"""
    pass


class ValidationError(PortfolioError):
    """Portfolio validation error"""
    pass


class DataError(PortfolioError):
    """Data integrity error"""
    pass


class CalculationError(PortfolioError):
    """Calculation error"""
    pass


class OptimizationError(PortfolioError):
    """Optimization error"""
    pass