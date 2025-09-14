# Portfolio Exports Directory

This directory contains exported portfolio data in various formats.

## Supported Export Formats

### CSV Export
- **Filename**: `{portfolio_name}_{timestamp}.csv`
- **Content**: Holdings data with current prices and performance
- **Use Case**: Excel analysis, external systems integration

### Excel Export  
- **Filename**: `{portfolio_name}_{timestamp}.xlsx`
- **Content**: Multi-sheet workbook with:
  - Holdings sheet
  - Portfolio info sheet
  - Sector allocation sheet
- **Use Case**: Comprehensive analysis, reporting, presentations

### JSON Export
- **Filename**: `{portfolio_name}_{timestamp}.json`
- **Content**: Complete portfolio data structure
- **Use Case**: Backup, data migration, API integration

## File Naming Convention

All exported files follow this pattern:
```
{portfolio_name}_{YYYYMMDD_HHMMSS}.{extension}
```

Examples:
- `Tech_Growth_Portfolio_20240115_143022.csv`
- `Balanced_Portfolio_20240115_143022.xlsx`
- `Conservative_Income_20240115_143022.json`

## CSV Format Details

| Column | Description | Example |
|--------|-------------|---------|
| ticker | Stock symbol | AAPL |
| name | Company name | Apple Inc. |
| weight | Portfolio weight | 0.30 |
| weight_percent | Weight as percentage | 30.00 |
| shares | Number of shares | 100.0 |
| current_price | Current market price | $150.00 |
| market_value | Total position value | $15,000.00 |
| purchase_price | Original purchase price | $140.00 |
| unrealized_pnl | Unrealized profit/loss | $1,000.00 |
| unrealized_pnl_percent | P&L as percentage | 7.14 |
| sector | Industry sector | Technology |
| asset_class | Asset classification | stock |
| currency | Trading currency | USD |
| exchange | Stock exchange | NASDAQ |

## Excel Format Details

**Sheet 1: Holdings**
- All asset holdings with detailed information
- Formatted for readability
- Conditional formatting for P&L

**Sheet 2: Info**
- Portfolio metadata
- Summary statistics
- Creation and modification dates

**Sheet 3: Sectors**
- Sector allocation breakdown
- Visual charts (if supported)

## Usage Examples

### Programmatic Export
```python
from core.data_manager import PortfolioManager

manager = PortfolioManager()

# Export to CSV
csv_path = manager.export_to_csv(portfolio_id, "exports/my_portfolio.csv")

# Export to Excel
excel_path = manager.export_to_excel(portfolio_id, "exports/my_portfolio.xlsx")

# Export to JSON
json_path = manager.export_to_json(portfolio_id, "exports/my_portfolio.json")
```

### Via Web Interface
1. Go to Portfolio Management page
2. Select portfolio
3. Choose export format
4. Click export button
5. File saved to this directory

## File Management

- Files accumulate over time
- Periodic cleanup recommended
- Large portfolios create larger files
- Consider compression for long-term storage

## Integration Notes

- CSV files work with Excel, Google Sheets, pandas
- JSON files work with any programming language
- Excel files provide richest formatting and multiple sheets
- All formats preserve data integrity

## Directory Contents

- `README.md` - This file
- `*.csv` - CSV exports (created on demand)
- `*.xlsx` - Excel exports (created on demand)
- `*.json` - JSON exports (created on demand)
- `.gitkeep` - Keeps directory in git (removed when files added)