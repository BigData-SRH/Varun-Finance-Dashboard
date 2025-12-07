"""
Data fetching functions for Indian Market Analysis.
Uses pre-downloaded CSV files to minimize API calls.
Falls back to yfinance API only for missing/fresh data.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List, Any
from functools import wraps
import os
import glob
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
# CSV DATA CONFIGURATION - Using os.path to avoid Path serialization issues
# =============================================================================

def _get_base_dir() -> str:
    """Get the base directory (project root) using os.path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _get_data_dir() -> str:
    """Get the data directory path."""
    return os.path.join(_get_base_dir(), "data")

def _ensure_directories():
    """Create data directories if they don't exist."""
    data_dir = _get_data_dir()
    for subdir in ["indices", "stocks", "global"]:
        path = os.path.join(data_dir, subdir)
        os.makedirs(path, exist_ok=True)

# Create directories at module load time
_ensure_directories()


def load_stock_metadata() -> Dict[str, Dict[str, Any]]:
    """Load pre-populated stock metadata from CSV file (more reliable on cloud)."""
    metadata_file = os.path.join(_get_data_dir(), "stocks_metadata.csv")
    if os.path.exists(metadata_file):
        try:
            df = pd.read_csv(metadata_file, index_col='ticker')
            return df.to_dict(orient='index')
        except Exception as e:
            print(f"Error loading metadata CSV: {e}")
    return {}


def load_manifest() -> Dict[str, Any]:
    """Load manifest file with last update info."""
    manifest_file = os.path.join(_get_data_dir(), "manifest.json")
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# =============================================================================
# CSV DATA LOADERS
# =============================================================================

def load_csv_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Load historical data from a CSV file.
    Handles malformed yfinance multi-index exports.
    csv_path: string path to CSV file
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        # Read CSV
        data = pd.read_csv(csv_path, index_col=0)
        
        # Check if first row contains ticker symbols (malformed yfinance export)
        if len(data) > 0:
            first_row = data.iloc[0]
            # If first row has string values like '^NSEI', skip it
            if any(isinstance(val, str) and (val.startswith('^') or '.NS' in str(val) or '.BO' in str(val)) for val in first_row.values):
                data = data.iloc[1:]  # Skip the ticker row
        
        # Convert index to datetime
        data.index = pd.to_datetime(data.index, errors='coerce')
        # Drop rows with invalid dates
        data = data[data.index.notna()]
        
        # Convert all columns to numeric (handles string values from malformed CSV)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Drop rows with NaN in Close column
        if 'Close' in data.columns:
            data = data.dropna(subset=['Close'])
        else:
            return None
        
        # Calculate returns if not present
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        if 'Cumulative_Return' not in data.columns:
            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        return data if len(data) > 0 else None
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
    
    # Use os.path to construct paths
    indices_dir = os.path.join(_get_data_dir(), "indices")
    
    for name, ticker in INDICES.items():
        # Try loading from CSV first
        csv_filename = ticker_to_filename(ticker)
        csv_path = os.path.join(indices_dir, csv_filename)
        
        # Debug: Check if file exists
        if not os.path.exists(csv_path):
            print(f"DEBUG: CSV not found at {csv_path}")
            st.warning(f"CSV not found for {name} at {csv_path}, fetching from API...")
        
        data = load_csv_data(csv_path)
        
        # Fallback to API if CSV not available
        if data is None:
            if os.path.exists(csv_path):
                st.warning(f"CSV exists but failed to load for {name}, trying API...")
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
    
    # Use os.path to construct paths
    global_dir = os.path.join(_get_data_dir(), "global")
    
    for name, ticker in GLOBAL_INDICES.items():
        # Try loading from CSV first
        csv_filename = ticker_to_filename(ticker)
        csv_path = os.path.join(global_dir, csv_filename)
        
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
    
    # Use os.path to construct paths
    stocks_dir = os.path.join(_get_data_dir(), "stocks")
    
    # Load historical data from CSV
    for ticker in all_stocks:
        csv_filename = ticker_to_filename(ticker)
        csv_path = os.path.join(stocks_dir, csv_filename)
        
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
    indices_dir = os.path.join(_get_data_dir(), "indices")
    csv_filename = ticker_to_filename(ticker)
    csv_path = os.path.join(indices_dir, csv_filename)
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
    stocks_dir = os.path.join(_get_data_dir(), "stocks")
    csv_filename = ticker_to_filename(ticker)
    csv_path = os.path.join(stocks_dir, csv_filename)
    data = load_csv_data(csv_path)
    
    # Fallback to API
    if data is None:
        data = fetch_from_yfinance(ticker, start_date, end_date)
    
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a stock from pre-downloaded metadata.
    Uses ONLY metadata JSON (NO API calls) for cloud compatibility.
    """
    metadata = load_stock_metadata()
    if ticker in metadata:
        m = metadata[ticker]
        return {
            'ticker': ticker,
            'name': m.get('name', ticker.replace('.NS', '')),
            'sector': m.get('sector', 'N/A'),
            'industry': m.get('industry', 'N/A'),
            'market_cap': m.get('market_cap', np.nan),
            'pe_ratio': m.get('pe_ratio', np.nan),
            'forward_pe': np.nan,  # Not in metadata
            'pb_ratio': np.nan,  # Not in metadata
            'dividend_yield': m.get('dividend_yield', np.nan),
            'current_price': m.get('current_price', np.nan),
            'fifty_two_week_high': m.get('52_week_high', np.nan),
            'fifty_two_week_low': m.get('52_week_low', np.nan),
            'book_value': m.get('book_value', np.nan),
            # These require live API, return NaN for metadata-only
            'roe': np.nan,
            'roa': np.nan,
            'debt_to_equity': np.nan,
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
    Uses pre-downloaded metadata first, then optionally tries API for extra data.
    Cloud-safe: works even if yfinance API fails.
    """
    # Check session state first
    cache_key = f"stock_info_{ticker}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Load all available data from metadata FIRST
    metadata = load_stock_metadata()
    m = metadata.get(ticker, {})

    info = {
        'ticker': ticker,
        'name': m.get('name', ticker.replace('.NS', '')),
        'sector': m.get('sector', 'N/A'),
        'industry': m.get('industry', 'N/A'),
        'market_cap': m.get('market_cap', np.nan),
        'pe_ratio': m.get('pe_ratio', np.nan),
        'forward_pe': np.nan,
        'pb_ratio': np.nan,
        'dividend_yield': m.get('dividend_yield', np.nan),
        'current_price': m.get('current_price', np.nan),
        'fifty_two_week_high': m.get('52_week_high', np.nan),
        'fifty_two_week_low': m.get('52_week_low', np.nan),
        'book_value': m.get('book_value', np.nan),
        # API-only fields default to NaN
        'roe': np.nan,
        'roa': np.nan,
        'debt_to_equity': np.nan,
        'avg_volume': np.nan,
        'beta': np.nan,
        'target_mean_price': np.nan,
        'recommendation': 'N/A',
        'num_analysts': np.nan,
    }

    # Optionally try API for additional data (may fail on cloud - that's OK)
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info

        if stock_info:
            # Only update fields that API provides (don't overwrite good metadata with None)
            info['forward_pe'] = stock_info.get('forwardPE', np.nan)
            info['pb_ratio'] = stock_info.get('priceToBook', np.nan)
            info['roe'] = stock_info.get('returnOnEquity', np.nan)
            info['roa'] = stock_info.get('returnOnAssets', np.nan)
            info['debt_to_equity'] = stock_info.get('debtToEquity', np.nan)
            info['avg_volume'] = stock_info.get('averageVolume', np.nan)
            info['beta'] = stock_info.get('beta', np.nan)
            info['target_mean_price'] = stock_info.get('targetMeanPrice', np.nan)
            info['recommendation'] = stock_info.get('recommendationKey', 'N/A')
            info['num_analysts'] = stock_info.get('numberOfAnalystOpinions', np.nan)
    except Exception as e:
        # API failed - that's OK, we still have metadata values
        print(f"API fetch skipped for {ticker}: {e}")

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
    data_dir = _get_data_dir()

    status = {
        'manifest_exists': os.path.exists(os.path.join(data_dir, "manifest.json")),
        'last_updated': manifest.get('last_updated', 'Unknown'),
        'indices_available': len(glob.glob(os.path.join(data_dir, "indices", "*.csv"))),
        'stocks_available': len(glob.glob(os.path.join(data_dir, "stocks", "*.csv"))),
        'global_available': len(glob.glob(os.path.join(data_dir, "global", "*.csv"))),
        'indices_expected': len(INDICES),
        'stocks_expected': len(get_unique_stocks()),
        'metadata_exists': os.path.exists(os.path.join(data_dir, "stocks_metadata.csv"))
    }
    
    return status
