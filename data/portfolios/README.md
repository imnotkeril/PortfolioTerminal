# Portfolio Data Directory

This directory contains portfolio data files stored in JSON format.

## File Structure

Each portfolio is stored as a separate JSON file named with the portfolio's UUID:
```
{portfolio_id}.json
```

## File Format

Portfolio files contain complete portfolio information including:

- Portfolio metadata (name, description, creation date, etc.)
- Asset holdings with weights and details
- Portfolio settings and constraints
- Trade history
- Custom metadata and tags

## Example Structure

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Tech Growth Portfolio",
  "description": "Technology-focused growth portfolio",
  "created_date": "2024-01-15T10:30:00.000000",
  "last_modified": "2024-01-15T10:30:00.000000",
  "initial_value": 100000.0,
  "portfolio_type": "growth",
  "tags": ["technology", "growth"],
  "metadata": {},
  "assets": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "weight": 0.3,
      "shares": 100.0,
      "current_price": 150.0,
      "purchase_price": 140.0,
      "sector": "Technology",
      "asset_class": "stock"
    }
  ],
  "settings": {
    "rebalancing_frequency": "quarterly",
    "auto_rebalance": false,
    "max_drawdown": 0.2
  },
  "trade_history": []
}
```

## Data Management

- Files are automatically created when portfolios are saved
- Portfolio Manager handles all file I/O operations
- Files are JSON formatted for human readability
- Backup recommended before major operations

## Security Notes

- This directory contains your portfolio data
- Consider encrypting this directory for sensitive portfolios
- Regular backups are recommended
- Do not commit portfolio files to version control

## Directory Contents

- `README.md` - This file
- `*.json` - Portfolio data files (created automatically)
- `.gitkeep` - Keeps directory in git (removed when files added)