"""
Data fetching functions for Indian Market Analysis.
All functions use Streamlit caching for performance.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List, Any
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import warnings

from .config import INDICES, CONSTITUENTS, START_DATE, END_DATE

warnings.filterwarnings('ignore')


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_data(
    ticker: str, 
    start_date: str = START_DATE, 
    end_date: str = END_DATE
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for an index.
    
    Parameters:
        ticker: Yahoo Finance ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data and calculated returns, or None if failed
    """
    try:
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
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(
    ticker: str, 
    start_date: str = START_DATE, 
    end_date: str = END_DATE
) -> Optional[pd.DataFrame]:
    """Fetch historical data for a stock."""
    return fetch_index_data(ticker, start_date, end_date)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a stock.
    
    Returns:
        Dictionary with stock information
    """
    try:
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
    except Exception:
        return {'ticker': ticker, 'name': ticker.replace('.NS', '')}


@st.cache_data(ttl=3600, show_spinner="Loading index data...")
def load_all_index_data() -> Dict[str, pd.DataFrame]:
    """Load data for all indices."""
    index_data: Dict[str, pd.DataFrame] = {}
    for name, ticker in INDICES.items():
        data = fetch_index_data(ticker, START_DATE, END_DATE)
        if data is not None:
            index_data[name] = data
    return index_data


@st.cache_data(ttl=3600, show_spinner="Loading global indices...")
def load_global_index_data() -> Dict[str, pd.DataFrame]:
    """Load data for global indices for comparison."""
    from .config import GLOBAL_INDICES
    global_data: Dict[str, pd.DataFrame] = {}
    for name, ticker in GLOBAL_INDICES.items():
        data = fetch_index_data(ticker, START_DATE, END_DATE)
        if data is not None:
            global_data[name] = data
    return global_data


@st.cache_data(ttl=3600, show_spinner="Loading stock data...")
def load_all_stock_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """
    Load data for all constituent stocks.
    
    Returns:
        Tuple of (stock_data dict, stock_info dict)
    """
    all_stocks = sorted(list(set(sum(CONSTITUENTS.values(), []))))
    
    stock_data: Dict[str, pd.DataFrame] = {}
    stock_info: Dict[str, Dict[str, Any]] = {}
    
    progress_bar = st.progress(0, text="Fetching stock data...")
    total = len(all_stocks)
    
    for i, ticker in enumerate(all_stocks):
        progress_bar.progress((i + 1) / total, text=f"Fetching {ticker}...")
        
        try:
            data = fetch_stock_data(ticker, START_DATE, END_DATE)
            if data is not None and len(data) > 0:
                stock_data[ticker] = data
        except Exception:
            pass
        
        try:
            info = fetch_stock_info(ticker)
            if info:
                stock_info[ticker] = info
        except Exception:
            pass
    
    progress_bar.empty()
    return stock_data, stock_info


def get_unique_stocks() -> List[str]:
    """Get sorted list of all unique stock tickers."""
    return sorted(list(set(sum(CONSTITUENTS.values(), []))))


def get_stock_index(ticker: str) -> Optional[str]:
    """Get the parent index for a stock."""
    for idx_name, constituents in CONSTITUENTS.items():
        if ticker in constituents:
            return idx_name
    return None
