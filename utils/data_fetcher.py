"""
Data fetching functions for Indian Market Analysis.
Uses pre-downloaded CSV files to minimize API calls.
Falls back to yfinance API only for missing/fresh data.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List, Any
from functools import wraps
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import warnings
import time
import random
import json
from datetime import datetime

from .config import INDICES, CONSTITUENTS, START_DATE, END_DATE

warnings.filterwarnings('ignore')

# =============================================================================
# CSV DATA CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
INDICES_DIR = DATA_DIR / "indices"
STOCKS_DIR = DATA_DIR / "stocks"
GLOBAL_DIR = DATA_DIR / "global"
METADATA_FILE = DATA_DIR / "stocks_metadata.json"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# Create directories if they don't exist
for directory in [DATA_DIR, INDICES_DIR, STOCKS_DIR, GLOBAL_DIR]:
    directory.mkdir(exist_ok=True)


def load_stock_metadata() -> Dict[str, Dict[str, str]]:
    """Load pre-populated stock metadata from JSON file."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_manifest() -> Dict[str, Any]:
    """Load manifest file with last update info."""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# =============================================================================
# CSV DATA LOADERS
# =============================================================================

def load_csv_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load historical data from CSV file."""
    if not csv_path.exists():
        return None
    
    try:
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Ensure required columns exist
        if 'Close' not in data.columns:
            return None
        
        # Calculate returns if not present
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        if 'Cumulative_Return' not in data.columns:
            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        return data
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None


def ticker_to_filename(ticker: str) -> str:
    """Convert ticker symbol to CSV filename."""
    return ticker.replace("^", "").replace(".", "_") + ".csv"


# =============================================================================
# FALLBACK API FUNCTIONS (only when CSV not available)
# =============================================================================

def retry_on_rate_limit(max_retries: int = 3, base_delay: float = 5.0):
    """Decorator to retry on rate limit errors with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in str(e) or "rate limit" in error_str or "too many requests" in error_str:
                        last_exception = e
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(delay)
                    else:
                        raise
            # Final attempt after all retries
            if last_exception:
                raise last_exception
            return func(*args, **kwargs)
        return wrapper
    return decorator


@retry_on_rate_limit(max_retries=2, base_delay=3.0)
def fetch_from_yfinance(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fallback function to fetch from yfinance API when CSV not available."""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        if data is None or len(data) == 0:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Calculate returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        return data
    except Exception as e:
        print(f"Error fetching {ticker} from API: {e}")
        return None


# =============================================================================
# MAIN DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="Loading index data...")
def load_all_index_data() -> Dict[str, pd.DataFrame]:
    """
    Load data for all Indian indices.
    PRIMARY: Uses CSV files from data/indices/
    FALLBACK: Uses yfinance API if CSV not available
    """
    index_data: Dict[str, pd.DataFrame] = {}
    
    for name, ticker in INDICES.items():
        # Try loading from CSV first
        csv_filename = ticker_to_filename(ticker)
        csv_path = INDICES_DIR / csv_filename
        
        data = load_csv_data(csv_path)
        
        # Fallback to API if CSV not available
        if data is None:
            st.warning(f"CSV not found for {name}, fetching from API...")
            data = fetch_from_yfinance(ticker, START_DATE, END_DATE)
        
        if data is not None and not data.empty:
            index_data[name] = data
    
    return index_data


@st.cache_data(ttl=14400, show_spinner="Loading global indices...")
def load_global_index_data() -> Dict[str, pd.DataFrame]:
    """
    Load data for global indices.
    PRIMARY: Uses CSV files from data/global/
    FALLBACK: Uses yfinance API if CSV not available
    """
    from .config import GLOBAL_INDICES
    
    global_data: Dict[str, pd.DataFrame] = {}
    
    for name, ticker in GLOBAL_INDICES.items():
        # Try loading from CSV first
        csv_filename = ticker_to_filename(ticker)
        csv_path = GLOBAL_DIR / csv_filename
        
        data = load_csv_data(csv_path)
        
        # Fallback to API if CSV not available (but don't warn - global is optional)
        if data is None:
            data = fetch_from_yfinance(ticker, START_DATE, END_DATE)
        
        if data is not None and not data.empty:
            global_data[name] = data
    
    return global_data


@st.cache_data(ttl=14400, show_spinner="Loading stock data...")
def load_all_stock_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """
    Load historical price data and metadata for all stocks.
    
    PRIMARY: Uses CSV files from data/stocks/ + metadata JSON
    FALLBACK: Uses yfinance API only for missing stocks
    
    Returns:
        Tuple of (stock_data dict, stock_info dict with metadata)
    """
    all_stocks = sorted(list(set(sum(CONSTITUENTS.values(), []))))
    
    stock_data: Dict[str, pd.DataFrame] = {}
    missing_count = 0
    
    # Load historical data from CSV
    for ticker in all_stocks:
        csv_filename = ticker_to_filename(ticker)
        csv_path = STOCKS_DIR / csv_filename
        
        data = load_csv_data(csv_path)
        
        if data is not None and not data.empty:
            stock_data[ticker] = data
        else:
            missing_count += 1
    
    # Show warning if many stocks are missing
    if missing_count > 0:
        st.warning(f"⚠️ {missing_count} stocks missing from CSV data. Run `python download_data.py` to update.")
    
    # Load metadata (NO API calls)
    metadata = load_stock_metadata()
    stock_info: Dict[str, Dict[str, Any]] = {}
    
    for ticker in all_stocks:
        if ticker in metadata:
            stock_info[ticker] = metadata[ticker]
        else:
            # Minimal fallback info
            stock_info[ticker] = {
                'name': ticker.replace('.NS', ''),
                'sector': 'N/A',
                'industry': 'N/A'
            }
    
    return stock_data, stock_info


# =============================================================================
# INDIVIDUAL FETCH FUNCTIONS (mostly for backwards compatibility)
# =============================================================================

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_index_data(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for an index.
    PRIMARY: Uses CSV files
    FALLBACK: Uses yfinance API
    """
    # Try CSV first
    csv_filename = ticker_to_filename(ticker)
    csv_path = INDICES_DIR / csv_filename
    data = load_csv_data(csv_path)
    
    # Fallback to API
    if data is None:
        data = fetch_from_yfinance(ticker, start_date, end_date)
    
    return data


@st.cache_data(ttl=14400, show_spinner=False)
def fetch_stock_data(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for a stock.
    PRIMARY: Uses CSV files
    FALLBACK: Uses yfinance API
    """
    # Try CSV first
    csv_filename = ticker_to_filename(ticker)
    csv_path = STOCKS_DIR / csv_filename
    data = load_csv_data(csv_path)
    
    # Fallback to API
    if data is None:
        data = fetch_from_yfinance(ticker, start_date, end_date)
    
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a stock.
    Uses ONLY metadata JSON (NO API calls unless explicitly needed).
    """
    # Load from metadata first
    metadata = load_stock_metadata()
    if ticker in metadata:
        return {
            'ticker': ticker,
            'name': metadata[ticker].get('name', ticker.replace('.NS', '')),
            'sector': metadata[ticker].get('sector', 'N/A'),
            'industry': metadata[ticker].get('industry', 'N/A'),
            # These would require API call - return NaN for now
            'market_cap': np.nan,
            'pe_ratio': np.nan,
            'forward_pe': np.nan,
            'pb_ratio': np.nan,
            'dividend_yield': np.nan,
            'roe': np.nan,
            'roa': np.nan,
            'debt_to_equity': np.nan,
            'current_price': np.nan,
            'fifty_two_week_high': np.nan,
            'fifty_two_week_low': np.nan,
            'avg_volume': np.nan,
            'beta': np.nan,
            'target_mean_price': np.nan,
            'recommendation': 'N/A',
            'num_analysts': np.nan,
        }
    
    # Minimal fallback
    return {
        'ticker': ticker,
        'name': ticker.replace('.NS', ''),
        'sector': 'N/A',
        'industry': 'N/A'
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_info_lazy(ticker: str) -> Dict[str, Any]:
    """
    Fetch stock info lazily when selected.
    For Stock Explorer page - combines metadata with live API data.
    """
    # Check session state first
    cache_key = f"stock_info_{ticker}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Start with metadata
    metadata = load_stock_metadata()
    info = {
        'ticker': ticker,
        'name': metadata.get(ticker, {}).get('name', ticker.replace('.NS', '')),
        'sector': metadata.get(ticker, {}).get('sector', 'N/A'),
        'industry': metadata.get(ticker, {}).get('industry', 'N/A'),
    }
    
    # Try to fetch live data (PE, PB, Market Cap, etc.) from API
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        info.update({
            'market_cap': stock_info.get('marketCap', np.nan),
            'pe_ratio': stock_info.get('trailingPE', np.nan),
            'forward_pe': stock_info.get('forwardPE', np.nan),
            'pb_ratio': stock_info.get('priceToBook', np.nan),
            'dividend_yield': stock_info.get('dividendYield', np.nan),
            'roe': stock_info.get('returnOnEquity', np.nan),
            'roa': stock_info.get('returnOnAssets', np.nan),
            'debt_to_equity': stock_info.get('debtToEquity', np.nan),
            'current_price': stock_info.get('currentPrice', stock_info.get('regularMarketPrice', np.nan)),
            'fifty_two_week_high': stock_info.get('fiftyTwoWeekHigh', np.nan),
            'fifty_two_week_low': stock_info.get('fiftyTwoWeekLow', np.nan),
            'avg_volume': stock_info.get('averageVolume', np.nan),
            'beta': stock_info.get('beta', np.nan),
            'target_mean_price': stock_info.get('targetMeanPrice', np.nan),
            'recommendation': stock_info.get('recommendationKey', 'N/A'),
            'num_analysts': stock_info.get('numberOfAnalystOpinions', np.nan),
        })
    except Exception as e:
        print(f"Could not fetch live data for {ticker}: {e}")
        # Fill with NaN
        info.update({
            'market_cap': np.nan,
            'pe_ratio': np.nan,
            'pb_ratio': np.nan,
            'dividend_yield': np.nan
        })
    
    # Cache in session state
    st.session_state[cache_key] = info
    return info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_unique_stocks() -> List[str]:
    """Get sorted list of all unique stock tickers."""
    return sorted(list(set(sum(CONSTITUENTS.values(), []))))


def get_stock_index(ticker: str) -> Optional[str]:
    """Get the parent index for a stock."""
    for idx_name, constituents in CONSTITUENTS.items():
        if ticker in constituents:
            return idx_name
    return None


def get_data_status() -> Dict[str, Any]:
    """Get status of CSV data availability."""
    manifest = load_manifest()
    
    status = {
        'manifest_exists': MANIFEST_FILE.exists(),
        'last_updated': manifest.get('last_updated', 'Unknown'),
        'indices_available': len(list(INDICES_DIR.glob("*.csv"))),
        'stocks_available': len(list(STOCKS_DIR.glob("*.csv"))),
        'global_available': len(list(GLOBAL_DIR.glob("*.csv"))),
        'indices_expected': len(INDICES),
        'stocks_expected': len(get_unique_stocks()),
        'metadata_exists': METADATA_FILE.exists()
    }
    
    return status
