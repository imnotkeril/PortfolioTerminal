# 🚀 Wild Market Capital - Portfolio Management System

A professional-grade portfolio management and analytics platform built with Python and Streamlit, featuring comprehensive portfolio analysis, risk management, and optimization capabilities.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Phase](https://img.shields.io/badge/phase-1%20complete-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Development Roadmap](#-development-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### ✅ Phase 1: Data Foundation (COMPLETE)

**Portfolio Management:**
- ✅ Complete CRUD operations for portfolios
- ✅ Multiple portfolio creation methods (text, CSV, Excel, manual)
- ✅ Portfolio templates (Conservative, Growth, Tech, etc.)
- ✅ Asset management with full validation
- ✅ Weight normalization and constraint checking
- ✅ Import/Export in multiple formats (CSV, Excel, JSON)

**Data Management:**
- ✅ Real-time price fetching via Yahoo Finance
- ✅ Historical data retrieval and caching
- ✅ Company information lookup
- ✅ Market status monitoring
- ✅ Intelligent caching with TTL
- ✅ Rate limiting and error handling

**Validation & Safety:**
- ✅ Comprehensive data validation
- ✅ Ticker symbol validation
- ✅ Weight and constraint validation
- ✅ Input sanitization and error handling
- ✅ Business rule enforcement

**User Interface:**
- ✅ Modern Streamlit web interface
- ✅ TradingView-inspired dark theme
- ✅ Responsive design with interactive charts
- ✅ Real-time data updates
- ✅ Intuitive portfolio management

### 🚧 Upcoming Phases

**Phase 2: Analytics Engine** (Planned)
- 70+ performance metrics calculation
- Benchmark comparison and attribution analysis
- Factor analysis (Fama-French models)
- Rolling statistics and time-series analysis

**Phase 3: Risk Engine** (Planned)
- VaR calculation (Historical, Parametric, Monte Carlo)
- Stress testing with 25+ historical scenarios
- Risk monitoring and alerting
- Correlation analysis

**Phase 4: Optimization Engine** (Planned)
- 17+ optimization algorithms
- Efficient frontier calculation
- Constraint-based optimization
- Rebalancing recommendations

**Phase 5: Scenario Analysis** (Planned)
- Custom scenario building
- Chain event modeling
- Historical analogies
- Hedging recommendations

**Phase 6: Reporting Engine** (Planned)
- Professional PDF reports
- Executive summaries
- Custom report templates
- Automated report generation

## 🏗️ Architecture

```
portfolio-management-system/
├── core/                           # Business logic modules
│   └── data_manager/              # Phase 1: Data management
│       ├── models.py              # Core data models
│       ├── validators.py          # Data validation
│       ├── portfolio_manager.py   # Portfolio CRUD operations
│       ├── price_manager.py       # Market data management
│       └── __init__.py
├── streamlit_app/                 # Web interface
│   ├── app.py                     # Main Streamlit application
│   └── components/               # UI components
├── tests/                         # Comprehensive test suite
│   ├── test_portfolio.py         # Portfolio tests
│   ├── test_validators.py        # Validation tests
│   └── test_price_manager.py     # Price management tests
├── data/                          # Data storage
│   └── portfolios/               # Portfolio JSON files
├── exports/                       # Export directory
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Design Principles

- **Modular Architecture**: Clean separation between data, business logic, and presentation
- **Type Safety**: Comprehensive type hints and data validation
- **Performance**: Intelligent caching and optimized data operations
- **Reliability**: Extensive error handling and comprehensive test coverage
- **Scalability**: Designed for easy extension with additional phases
- **User Experience**: Intuitive interface with professional-grade visualization

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Internet connection (for market data)
- 4GB RAM minimum (8GB recommended)

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/wildmarketcapital/portfolio-manager.git
cd portfolio-manager

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/portfolios exports
```

### Option 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/wildmarketcapital/portfolio-manager.git
cd portfolio-manager

# Install in development mode with testing dependencies
pip install -e .
pip install -r requirements-dev.txt

# Run tests to verify installation
pytest
```

## 🎯 Quick Start

### 1. Launch the Application

```bash
# Navigate to project directory
cd portfolio-management-system

# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Launch Streamlit app
streamlit run streamlit_app/app.py
```

The application will open in your browser at `http://localhost:8501`

### 2. Create Your First Portfolio

**Option A: Using Text Input**
1. Navigate to "Create Portfolio" → "Text Input" tab
2. Enter: `AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%`
3. Set portfolio name and click "Create Portfolio"

**Option B: Using Templates**
1. Navigate to "Create Portfolio" → "Templates" tab
2. Select a template (e.g., "Tech Focus")
3. Customize name and settings
4. Click "Create from Template"

### 3. Explore Your Portfolio

- **Dashboard**: View all portfolios and key metrics
- **Analysis**: Deep dive into portfolio performance
- **Management**: Edit, update, and maintain portfolios
- **System Status**: Monitor data providers and system health

## 💡 Usage Examples

### Creating Portfolios Programmatically

```python
from core.data_manager import PortfolioManager, Asset, AssetClass

# Initialize manager
manager = PortfolioManager()

# Create assets
assets = [
    Asset(ticker="AAPL", weight=0.4, name="Apple Inc."),
    Asset(ticker="MSFT", weight=0.3, name="Microsoft"),
    Asset(ticker="GOOGL", weight=0.3, name="Alphabet")
]

# Create portfolio
portfolio = manager.create_portfolio(
    name="My Tech Portfolio",
    description="Technology-focused growth portfolio",
    assets=assets
)

print(f"Created portfolio: {portfolio.name}")
print(f"Total assets: {len(portfolio.assets)}")
print(f"Total weight: {portfolio.total_weight:.1%}")
```

### Working with Price Data

```python
from core.data_manager import PriceManager

# Initialize price manager
price_manager = PriceManager()

# Get current prices
tickers = ["AAPL", "MSFT", "GOOGL"]
current_prices = price_manager.get_current_prices(tickers)

for ticker, price in current_prices.items():
    print(f"{ticker}: ${price:.2f}")

# Get historical data
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

historical_data = price_manager.get_historical_prices(
    tickers, start_date, end_date
)

print(f"Historical data shape: {historical_data.shape}")
```

### Data Validation

```python
from core.data_manager.validators import PortfolioValidator

# Create validator
validator = PortfolioValidator()

# Validate portfolio
result = validator.validate_portfolio(portfolio)

if result.is_valid:
    print("✅ Portfolio is valid!")
else:
    print("❌ Validation errors:")
    for error in result.errors:
        print(f"  • {error}")

# Check constraints
violations = validator.check_constraints(portfolio)
if violations:
    print("⚠️ Constraint violations:")
    for violation in violations:
        print(f"  • {violation.description}")
```

### Import/Export Operations

```python
# Create from text
portfolio = manager.create_from_text(
    name="Text Portfolio",
    text="AAPL 40%, MSFT 30%, GOOGL 30%"
)

# Export to Excel
manager.export_to_excel(portfolio.id, "my_portfolio.xlsx")

# Import from CSV
imported_portfolio = manager.import_from_csv(
    "portfolio_data.csv",
    name="Imported Portfolio"
)
```

## 📚 API Documentation

### Core Classes

#### Portfolio
The main portfolio data structure containing assets and metadata.

```python
class Portfolio:
    id: str                          # Unique identifier
    name: str                        # Portfolio name
    description: str                 # Description
    assets: List[Asset]              # List of assets
    settings: PortfolioSettings      # Advanced settings
    portfolio_type: PortfolioType    # Type classification
    
    # Methods
    def validate() -> bool
    def normalize_weights() -> None
    def calculate_value(prices: Dict = None) -> float
    def add_asset(asset: Asset) -> None
    def remove_asset(ticker: str) -> None
    def get_statistics() -> PortfolioStats
```

#### Asset
Individual asset within a portfolio.

```python
class Asset:
    ticker: str                      # Stock symbol
    name: str                        # Company name
    weight: float                    # Portfolio weight (0.0-1.0)
    shares: float                    # Number of shares
    current_price: Optional[float]   # Current market price
    purchase_price: Optional[float]  # Original purchase price
    sector: str                      # Industry sector
    asset_class: AssetClass          # Asset classification
    
    # Properties
    @property
    def market_value() -> float
    @property
    def unrealized_pnl() -> float
    @property
    def unrealized_pnl_percent() -> float
```

#### PortfolioManager
Main interface for portfolio CRUD operations.

```python
class PortfolioManager:
    def create_portfolio(name: str, assets: List[Asset], **kwargs) -> Portfolio
    def get_portfolio(portfolio_id: str) -> Portfolio
    def update_portfolio(portfolio_id: str, updates: Dict) -> Portfolio
    def delete_portfolio(portfolio_id: str) -> bool
    def list_portfolios(filters: Dict = None) -> List[Portfolio]
    
    # Import/Export
    def create_from_text(name: str, text: str) -> Portfolio
    def import_from_csv(file_path: str, name: str) -> Portfolio
    def export_to_excel(portfolio_id: str, file_path: str) -> str
```

#### PriceManager
Market data management with caching and multiple providers.

```python
class PriceManager:
    def get_current_price(ticker: str) -> Optional[float]
    def get_current_prices(tickers: List[str]) -> Dict[str, float]
    def get_historical_prices(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame
    def get_company_info(ticker: str) -> Optional[CompanyInfo]
    def update_portfolio_prices(portfolio: Portfolio) -> Portfolio
```

### Validation System

The system includes comprehensive validation at multiple levels:

- **Ticker Validation**: Format checking and known ticker validation
- **Weight Validation**: Sum checking, range validation, concentration limits
- **Portfolio Validation**: Complete portfolio integrity checking
- **Data Validation**: Input sanitization and type checking
- **Import Validation**: File format and data structure validation

## 🧪 Testing

The system includes comprehensive test coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov-report=html

# Run specific test categories
pytest tests/test_portfolio.py -v
pytest tests/test_validators.py -v
pytest tests/test_price_manager.py -v

# Run integration tests (requires internet)
pytest -m integration

# Run performance tests
pytest tests/test_performance.py
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Speed and memory usage validation
- **Edge Case Tests**: Error handling and boundary conditions
- **Mock Tests**: Isolated testing with mocked dependencies

### Coverage Requirements

- Minimum 80% code coverage
- 100% coverage for critical paths
- All edge cases tested
- Error handling validated

## 🛣️ Development Roadmap

### ✅ Phase 1: Data Foundation (COMPLETE)
*Duration: 2 weeks*

- [x] Portfolio CRUD operations
- [x] Asset management and validation
- [x] Price data integration with caching
- [x] Comprehensive validation system
- [x] Streamlit user interface
- [x] Import/export functionality
- [x] Complete test suite

### 🔄 Phase 2: Analytics Engine (Q2 2024)
*Duration: 2 weeks*

**Features:**
- 70+ portfolio performance metrics
- Benchmark comparison and analysis
- Factor analysis (Fama-French models)
- Attribution analysis
- Rolling statistics
- Advanced charting and visualization

**Deliverables:**
- `core/analytics_engine/` module
- Performance metrics calculation
- Interactive analytics dashboard
- Comprehensive reporting

### 🔄 Phase 3: Risk Engine (Q2 2024)
*Duration: 2 weeks*

**Features:**
- Value at Risk (5 calculation methods)
- Stress testing (25+ scenarios)
- Monte Carlo simulation
- Risk monitoring and alerts
- Correlation analysis
- Drawdown analysis

**Deliverables:**
- `core/risk_engine/` module
- Risk dashboard interface
- Alert system
- Risk reporting

### 🔄 Phase 4: Optimization Engine (Q3 2024)
*Duration: 2 weeks*

**Features:**
- 17+ optimization algorithms
- Efficient frontier calculation
- Constraint-based optimization
- Rebalancing recommendations
- Backtesting framework
- Strategy optimization

### 🔄 Phase 5: Scenario Analysis (Q3 2024)
*Duration: 2 weeks*

**Features:**
- Custom scenario building
- Chain event modeling
- Historical analogies
- Probability analysis
- Hedging recommendations

### 🔄 Phase 6: Reporting Engine (Q4 2024)
*Duration: 2 weeks*

**Features:**
- Professional PDF reports
- Executive summaries
- Custom templates
- Automated generation
- Multi-format export

### 🔄 Phase 7: API Development (Q4 2024)
*Duration: 2 weeks*

**Features:**
- RESTful API with FastAPI
- Authentication and authorization
- Rate limiting
- API documentation
- Client SDKs

### 🔄 Phase 8: React Frontend (Q1 2025)
*Duration: 2 weeks*

**Features:**
- Modern React interface
- Real-time updates
- Advanced visualizations
- Mobile responsive
- Professional UI/UX

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Data Provider Settings
YAHOO_FINANCE_ENABLED=true
ALPHA_VANTAGE_API_KEY=your_api_key_here
POLYGON_API_KEY=your_api_key_here

# Cache Settings
CACHE_TTL_SECONDS=300
CACHE_ENABLED=true

# Storage Settings
PORTFOLIO_STORAGE_PATH=data/portfolios
EXPORT_PATH=exports

# Rate Limiting
REQUESTS_PER_HOUR=1000
ENABLE_RATE_LIMITING=true

# Development Settings
DEBUG=false
LOG_LEVEL=INFO
```

### Custom Settings

```python
from core.data_manager import PortfolioManager, PriceManager

# Custom storage path
manager = PortfolioManager(storage_path="custom/path")

# Custom cache settings
price_manager = PriceManager(cache_ttl=600, enable_cache=True)
```

## 📊 Performance Benchmarks

### Phase 1 Performance Targets ✅

| Operation | Target | Actual | Status |
|-----------|--------|--------|---------|
| Portfolio Creation | < 500ms | ~200ms | ✅ |
| Price Fetch (10 tickers) | < 2s | ~1.2s | ✅ |
| Portfolio Validation | < 100ms | ~50ms | ✅ |
| CSV Import (100 assets) | < 3s | ~1.8s | ✅ |
| Dashboard Load | < 2s | ~1.5s | ✅ |

### System Requirements

**Minimum:**
- RAM: 4GB
- Storage: 1GB free space
- Network: Broadband internet
- Python: 3.8+

**Recommended:**
- RAM: 8GB+
- Storage: 5GB free space
- Network: High-speed broadband
- Python: 3.9+

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/portfolio-manager.git

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest
```

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **Testing**: Maintain >80% coverage, write tests for new features
- **Documentation**: Update README and docstrings
- **Git**: Use conventional commits

### Pull Request Process

1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and add tests
3. Ensure all tests pass: `pytest`
4. Update documentation
5. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing market data
- **Streamlit** for the excellent web framework
- **Plotly** for interactive visualizations
- **Pandas** for data manipulation capabilities

## 📞 Support

- **Documentation**: [https://docs.wildmarketcapital.com](https://docs.wildmarketcapital.com)
- **Issues**: [GitHub Issues](https://github.com/wildmarketcapital/portfolio-manager/issues)
- **Email**: support@wildmarketcapital.com
- **Discord**: [WMC Community](https://discord.gg/wildmarketcapital)

## 📈 Status Dashboard

| Component | Status | Coverage | Performance |
|-----------|--------|----------|-------------|
| Data Manager | ✅ Active | 85% | Excellent |
| Price Manager | ✅ Active | 82% | Good |
| Validators | ✅ Active | 90% | Excellent |
| Streamlit UI | ✅ Active | 75% | Good |
| Test Suite | ✅ Active | 84% | Excellent |

---

<div align="center">

**🚀 Wild Market Capital Portfolio Manager**

*Professional portfolio management made simple*

[Website](https://wildmarketcapital.com) • [Documentation](https://docs.wildmarketcapital.com) • [Discord](https://discord.gg/wmc) • [Twitter](https://twitter.com/wildmarketcap)

</div>