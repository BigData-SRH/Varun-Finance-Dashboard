"""
Data Download Script
====================
This script downloads all required historical data and saves it to CSV files.
Run this script periodically (daily/weekly) to update the data.

Usage:
    python download_data.py

The CSV files will be committed to the GitHub repo so Streamlit Cloud
can access them without making API calls.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import random
from utils.config import INDICES, CONSTITUENTS, GLOBAL_INDICES, START_DATE, END_DATE

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Subdirectories
INDICES_DIR = DATA_DIR / "indices"
STOCKS_DIR = DATA_DIR / "stocks"
GLOBAL_DIR = DATA_DIR / "global"

INDICES_DIR.mkdir(exist_ok=True)
STOCKS_DIR.mkdir(exist_ok=True)
GLOBAL_DIR.mkdir(exist_ok=True)


def safe_download(ticker, start_date, end_date, retry_count=3):
    """Download with retry logic and clean the data."""
    for attempt in range(retry_count):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            if data is not None and not data.empty:
                # Flatten multi-index columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data
        except Exception as e:
            print(f"  ⚠️ Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < retry_count - 1:
                delay = 2 ** attempt + random.uniform(0, 1)
                time.sleep(delay)
    return None


def download_indices():
    """Download all Indian indices historical data."""
    print("\n📊 Downloading Indian Indices...")
    print("=" * 60)
    
    for name, ticker in INDICES.items():
        print(f"  Downloading {name} ({ticker})...")
        data = safe_download(ticker, START_DATE, END_DATE)
        
        if data is not None and not data.empty:
            filename = ticker.replace("^", "").replace(".", "_") + ".csv"
            filepath = INDICES_DIR / filename
            data.to_csv(filepath)
            print(f"    ✅ Saved to {filepath} ({len(data)} rows)")
        else:
            print(f"    ❌ Failed to download {name}")
        
        time.sleep(random.uniform(0.5, 1.5))  # Rate limiting


def download_global_indices():
    """Download global indices for comparison."""
    print("\n🌍 Downloading Global Indices...")
    print("=" * 60)
    
    for name, ticker in GLOBAL_INDICES.items():
        print(f"  Downloading {name} ({ticker})...")
        data = safe_download(ticker, START_DATE, END_DATE)
        
        if data is not None and not data.empty:
            filename = ticker.replace("^", "").replace(".", "_") + ".csv"
            filepath = GLOBAL_DIR / filename
            data.to_csv(filepath)
            print(f"    ✅ Saved to {filepath} ({len(data)} rows)")
        else:
            print(f"    ❌ Failed to download {name}")
        
        time.sleep(random.uniform(0.5, 1.5))


def download_stocks():
    """Download all stock constituents historical data."""
    print("\n🏢 Downloading Stock Data...")
    print("=" * 60)
    
    all_stocks = set()
    for stocks in CONSTITUENTS.values():
        all_stocks.update(stocks)
    
    all_stocks = sorted(list(all_stocks))
    total = len(all_stocks)
    
    print(f"  Total stocks to download: {total}")
    print()
    
    for i, ticker in enumerate(all_stocks, 1):
        print(f"  [{i}/{total}] Downloading {ticker}...")
        data = safe_download(ticker, START_DATE, END_DATE)
        
        if data is not None and not data.empty:
            filename = ticker.replace(".", "_") + ".csv"
            filepath = STOCKS_DIR / filename
            data.to_csv(filepath)
            print(f"    ✅ Saved ({len(data)} rows)")
        else:
            print(f"    ❌ Failed")
        
        # Rate limiting - longer delay every 10 stocks
        if i % 10 == 0:
            delay = random.uniform(3, 5)
            print(f"    ⏸️ Pausing {delay:.1f}s to avoid rate limiting...")
            time.sleep(delay)
        else:
            time.sleep(random.uniform(0.8, 1.5))


def download_stock_info():
    """
    Download stock fundamental info (name, sector, industry).
    This is saved to stocks_metadata.json (already exists).
    """
    print("\n📋 Stock Fundamental Info...")
    print("=" * 60)
    print("  ℹ️ Using existing data/stocks_metadata.json")
    print("  ℹ️ To update metadata, modify the JSON file directly")


def create_manifest():
    """Create a manifest file with download timestamp."""
    manifest = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "indices_count": len(list(INDICES_DIR.glob("*.csv"))),
        "stocks_count": len(list(STOCKS_DIR.glob("*.csv"))),
        "global_count": len(list(GLOBAL_DIR.glob("*.csv")))
    }
    
    manifest_file = DATA_DIR / "manifest.json"
    import json
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n✅ Created manifest.json")


def main():
    """Main download orchestrator."""
    print("=" * 60)
    print("📥 DATA DOWNLOAD SCRIPT")
    print("=" * 60)
    print(f"Start Date: {START_DATE}")
    print(f"End Date: {END_DATE}")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Download in order: indices -> global -> stocks (most important first)
    download_indices()
    download_global_indices()
    download_stocks()
    download_stock_info()
    
    # Create manifest
    create_manifest()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ DOWNLOAD COMPLETE!")
    print(f"⏱️ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 60)
    print("\n📌 Next Steps:")
    print("  1. Commit the data/ folder to your Git repository")
    print("  2. Push to GitHub")
    print("  3. Streamlit Cloud will use these CSV files")
    print("  4. Re-run this script daily/weekly to keep data fresh")


if __name__ == "__main__":
    main()
