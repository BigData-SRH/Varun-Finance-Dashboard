"""
Data fetching functions for Indian Market Analysis.
All functions use Streamlit caching + persistent disk caching for performance.
Includes rate limiting and retry logic to handle Yahoo Finance API limits.
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
import pickle
import hashlib

from .config import INDICES, CONSTITUENTS, START_DATE, END_DATE

warnings.filterwarnings('ignore')

# =============================================================================
# DISK CACHE CONFIGURATION
# =============================================================================

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache durations in hours
HISTORICAL_DATA_CACHE_HOURS = 4  # Historical price data
STOCK_INFO_CACHE_HOURS = 24      # Fundamental data (changes less frequently)


def get_cache_key(*args) -> str:
    """Generate a unique cache key from arguments."""
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def get_cached_data(cache_key: str, max_age_hours: int = 4) -> Optional[Any]:
    """Load data from disk cache if fresh."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            age_seconds = time.time() - cache_file.stat().st_mtime
            if age_seconds < max_age_hours * 3600:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass  # Cache read failed, will refetch
    return None


def save_to_cache(cache_key: str, data: Any) -> None:
    """Save data to disk cache."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass  # Cache write failed, continue without caching


# =============================================================================
# RATE LIMITING & RETRY LOGIC
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


def rate_limited_sleep(min_delay: float = 0.3, max_delay: float = 0.8) -> None:
    """Add a random delay to avoid rate limiting."""
    time.sleep(random.uniform(min_delay, max_delay))


# =============================================================================
# BATCH DOWNLOAD FUNCTIONS
# =============================================================================

@retry_on_rate_limit(max_retries=3, base_delay=5.0)
def batch_download_historical(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for multiple tickers in a single API call.

    This is MUCH more efficient than individual downloads.
    """
    if not tickers:
        return {}

    # Check disk cache first
    cache_key = get_cache_key("batch_hist", "_".join(sorted(tickers)), start_date, end_date)
    cached = get_cached_data(cache_key, HISTORICAL_DATA_CACHE_HOURS)
    if cached is not None:
        return cached

    try:
        # Single API call for all tickers
        raw_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker',
            threads=True,
            auto_adjust=False
        )

        if raw_data is None or len(raw_data) == 0:
            return {}

        result: Dict[str, pd.DataFrame] = {}

        # Handle single ticker case (no MultiIndex)
        if len(tickers) == 1:
            ticker = tickers[0]
            data = raw_data.copy()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data['Daily_Return'] = data['Close'].pct_change()
            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
            result[ticker] = data
        else:
            # Multiple tickers - data is grouped by ticker
            for ticker in tickers:
                try:
                    if ticker in raw_data.columns.get_level_values(0):
                        data = raw_data[ticker].copy()
                        data = data.dropna(how='all')
                        if len(data) > 0:
                            data['Daily_Return'] = data['Close'].pct_change()
                            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
                            result[ticker] = data
                except Exception:
                    continue

        # Save to disk cache
        if result:
            save_to_cache(cache_key, result)

        return result

    except Exception as e:
        error_str = str(e).lower()
        if "429" in str(e) or "rate limit" in error_str:
            raise  # Let retry decorator handle it
        print(f"Error in batch download: {e}")
        return {}


# =============================================================================
# INDIVIDUAL FETCH FUNCTIONS (with caching)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_data(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for an index.
    Uses disk cache + Streamlit cache for optimal performance.
    """
    # Check disk cache first
    cache_key = get_cache_key("index", ticker, start_date, end_date)
    cached = get_cached_data(cache_key, HISTORICAL_DATA_CACHE_HOURS)
    if cached is not None:
        return cached

    try:
        data = _fetch_single_ticker(ticker, start_date, end_date)
        if data is not None:
            save_to_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


@retry_on_rate_limit(max_retries=3, base_delay=5.0)
def _fetch_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """Internal function to fetch a single ticker with retry logic."""
    data: Optional[pd.DataFrame] = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        multi_level_index=False,
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> Optional[pd.DataFrame]:
    """Fetch historical data for a stock."""
    return fetch_index_data(ticker, start_date, end_date)


@st.cache_data(ttl=86400, show_spinner=False)  # 24h cache for fundamental data
def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a stock.
    Uses longer cache duration since fundamentals change less frequently.
    """
    # Check disk cache first (24h)
    cache_key = get_cache_key("info", ticker)
    cached = get_cached_data(cache_key, STOCK_INFO_CACHE_HOURS)
    if cached is not None:
        return cached

    try:
        result = _fetch_stock_info_internal(ticker)
        if result and 'name' in result:
            save_to_cache(cache_key, result)
        return result
    except Exception:
        return {'ticker': ticker, 'name': ticker.replace('.NS', '')}


@retry_on_rate_limit(max_retries=2, base_delay=3.0)
def _fetch_stock_info_internal(ticker: str) -> Dict[str, Any]:
    """Internal function to fetch stock info with retry logic."""
    stock = yf.Ticker(ticker)
    info: Dict[str, Any] = stock.info

    return {
        'ticker': ticker,
        'name': info.get('longName', info.get('shortName', ticker.replace('.NS', ''))),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', np.nan),
        'pe_ratio': info.get('trailingPE', np.nan),
        'forward_pe': info.get('forwardPE', np.nan),
        'pb_ratio': info.get('priceToBook', np.nan),
        'dividend_yield': info.get('dividendYield', np.nan),
        'roe': info.get('returnOnEquity', np.nan),
        'roa': info.get('returnOnAssets', np.nan),
        'debt_to_equity': info.get('debtToEquity', np.nan),
        'current_price': info.get('currentPrice', info.get('regularMarketPrice', np.nan)),
        'fifty_two_week_high': info.get('fiftyTwoWeekHigh', np.nan),
        'fifty_two_week_low': info.get('fiftyTwoWeekLow', np.nan),
        'avg_volume': info.get('averageVolume', np.nan),
        'beta': info.get('beta', np.nan),
        'target_mean_price': info.get('targetMeanPrice', np.nan),
        'recommendation': info.get('recommendationKey', 'N/A'),
        'num_analysts': info.get('numberOfAnalystOpinions', np.nan),
    }


# =============================================================================
# BATCH LOAD FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="Loading index data...")
def load_all_index_data() -> Dict[str, pd.DataFrame]:
    """Load data for all indices using batch download."""
    # Check disk cache first
    cache_key = get_cache_key("all_indices", START_DATE, END_DATE)
    cached = get_cached_data(cache_key, HISTORICAL_DATA_CACHE_HOURS)
    if cached is not None:
        return cached

    tickers = list(INDICES.values())
    ticker_to_name = {v: k for k, v in INDICES.items()}

    # Batch download all indices at once
    raw_data = batch_download_historical(tickers, START_DATE, END_DATE)

    # Map ticker symbols to index names
    index_data: Dict[str, pd.DataFrame] = {}
    for ticker, data in raw_data.items():
        name = ticker_to_name.get(ticker, ticker)
        index_data[name] = data

    if index_data:
        save_to_cache(cache_key, index_data)

    return index_data


@st.cache_data(ttl=3600, show_spinner="Loading global indices...")
def load_global_index_data() -> Dict[str, pd.DataFrame]:
    """Load data for global indices using batch download."""
    from .config import GLOBAL_INDICES

    # Check disk cache first
    cache_key = get_cache_key("global_indices", START_DATE, END_DATE)
    cached = get_cached_data(cache_key, HISTORICAL_DATA_CACHE_HOURS)
    if cached is not None:
        return cached

    tickers = list(GLOBAL_INDICES.values())
    ticker_to_name = {v: k for k, v in GLOBAL_INDICES.items()}

    # Batch download all global indices at once
    raw_data = batch_download_historical(tickers, START_DATE, END_DATE)

    # Map ticker symbols to index names
    global_data: Dict[str, pd.DataFrame] = {}
    for ticker, data in raw_data.items():
        name = ticker_to_name.get(ticker, ticker)
        global_data[name] = data

    if global_data:
        save_to_cache(cache_key, global_data)

    return global_data


@st.cache_data(ttl=3600, show_spinner="Loading stock data...")
def load_all_stock_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """
    Load data for all constituent stocks using batch download.

    Returns:
        Tuple of (stock_data dict, stock_info dict)
    """
    all_stocks = sorted(list(set(sum(CONSTITUENTS.values(), []))))

    # Check disk cache first for complete data
    cache_key = get_cache_key("all_stocks_complete", START_DATE, END_DATE)
    cached = get_cached_data(cache_key, HISTORICAL_DATA_CACHE_HOURS)
    if cached is not None:
        return cached

    # BATCH DOWNLOAD: All historical data in ONE API call
    progress_bar = st.progress(0, text="Downloading historical data...")
    stock_data = batch_download_historical(all_stocks, START_DATE, END_DATE)
    progress_bar.progress(0.5, text="Historical data loaded. Fetching stock info...")

    # RATE-LIMITED: Fetch stock info (cannot be batched, so add delays)
    stock_info: Dict[str, Dict[str, Any]] = {}
    total = len(all_stocks)

    for i, ticker in enumerate(all_stocks):
        progress_bar.progress(0.5 + (0.5 * (i + 1) / total), text=f"Fetching info for {ticker}...")

        try:
            info = fetch_stock_info(ticker)
            if info:
                stock_info[ticker] = info
        except Exception:
            stock_info[ticker] = {'ticker': ticker, 'name': ticker.replace('.NS', '')}

        # Rate limiting: pause every 10 stocks to avoid hitting limits
        if (i + 1) % 10 == 0 and i < total - 1:
            time.sleep(1.5)
        else:
            rate_limited_sleep(0.2, 0.5)

    progress_bar.empty()

    # Save complete data to disk cache
    result = (stock_data, stock_info)
    save_to_cache(cache_key, result)

    return result


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


def clear_disk_cache() -> None:
    """Clear all cached data from disk."""
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            cache_file.unlink()
        except Exception:
            pass
