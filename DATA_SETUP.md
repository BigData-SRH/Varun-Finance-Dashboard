# Data Setup Guide

## Overview

This dashboard uses **pre-downloaded CSV files** instead of making live API calls to Yahoo Finance. This approach:
- ✅ Eliminates rate limiting issues on Streamlit Cloud
- ✅ Loads data instantly (no API wait time)
- ✅ Works reliably in production
- ✅ Reduces dependency on external APIs

## Directory Structure

```
data/
├── indices/          # Indian index data (Nifty 50, Bank, IT)
│   ├── NSEI.csv
│   ├── NSEBANK.csv
│   └── CNXIT.csv
├── stocks/           # Individual stock data (50+ stocks)
│   ├── RELIANCE_NS.csv
│   ├── TCS_NS.csv
│   └── ...
├── global/           # Global indices (S&P 500, NASDAQ, etc.)
│   ├── GSPC.csv
│   ├── IXIC.csv
│   └── ...
├── stocks_metadata.json  # Stock info (name, sector, industry)
└── manifest.json     # Last update timestamp
```

## Initial Setup

### Step 1: Download Data

Run the download script to fetch all historical data:

```powershell
python download_data.py
```

This will:
- Download ~5-10 years of historical price data
- Save everything as CSV files in the `data/` folder
- Create a manifest file with timestamp
- Take ~5-15 minutes depending on your connection

**Note:** The script includes rate limiting and retries to avoid Yahoo Finance issues.

### Step 2: Commit to Git

After downloading, commit the data to your repository:

```powershell
git add data/
git commit -m "Add historical market data"
git push
```

### Step 3: Deploy to Streamlit Cloud

Once pushed to GitHub, Streamlit Cloud will:
1. Clone your repo (including the `data/` folder)
2. Load all data from CSV files
3. **Make ZERO API calls** during normal operation

## Updating Data

### When to Update

- **Daily**: For day traders (before market opens)
- **Weekly**: For swing traders
- **Monthly**: For long-term investors

### How to Update

Simply run the download script again:

```powershell
python download_data.py
git add data/
git commit -m "Update market data - $(Get-Date -Format 'yyyy-MM-dd')"
git push
```

Streamlit Cloud will automatically redeploy with fresh data.

## CSV File Format

Each CSV file contains standard OHLCV data:

```csv
Date,Open,High,Low,Close,Adj Close,Volume,Daily_Return,Cumulative_Return
2015-01-01,8274.75,8317.70,8257.45,8284.00,8284.00,117300000,0.0,0.0
2015-01-02,8302.70,8346.45,8297.15,8330.75,8330.75,129900000,0.0056,0.0056
...
```

## Fallback Mechanism

If a CSV file is missing, the app will:
1. Show a warning message
2. Attempt to fetch from Yahoo Finance API
3. Continue with available data

This ensures the app never crashes due to missing data.

## Data Storage Requirements

- **Indices**: ~3 MB (3 files)
- **Stocks**: ~150 MB (50+ files)
- **Global**: ~5 MB (6 files)
- **Total**: ~160 MB

GitHub repositories have a 1 GB limit, so this is well within bounds.

## Troubleshooting

### "CSV not found" warnings

**Solution:** Run `python download_data.py`

### Download script fails with 429 errors

**Solution:** Wait 5-10 minutes and try again. The script has built-in retries.

### Old data showing in dashboard

**Solution:** 
```powershell
python download_data.py  # Download fresh data
git push                  # Push to GitHub
```

Then restart your Streamlit Cloud app.

### Data folder too large for Git

**Solution:** Use Git LFS for large files:
```powershell
git lfs install
git lfs track "data/stocks/*.csv"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Live Data (Optional)

For real-time data in Stock Explorer, the app still fetches:
- Current price, PE ratio, market cap (only when stock is selected)
- This is a single API call per stock selection
- Cached for 24 hours in session state

## Benefits of This Approach

### Before (API-based):
- ❌ 72+ API calls on startup
- ❌ Rate limiting errors
- ❌ Slow load times
- ❌ Unreliable on Streamlit Cloud

### After (CSV-based):
- ✅ 0 API calls on startup
- ✅ No rate limiting
- ✅ Instant load times
- ✅ Reliable production deployment
- ✅ Works offline (for development)

## Automation (Advanced)

### GitHub Actions (Recommended)

Create `.github/workflows/update-data.yml`:

```yaml
name: Update Market Data
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python download_data.py
      - run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add data/
          git diff --quiet || git commit -m "Update market data"
          git push
```

This automatically updates your data daily!

## Questions?

- Check `download_data.py` for script configuration
- Check `utils/data_fetcher.py` for CSV loading logic
- Open an issue on GitHub for help
