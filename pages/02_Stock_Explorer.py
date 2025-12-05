"""
Stock Explorer Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import time
import random

from utils import (
    INDICES, CONSTITUENTS, COLORS, RISK_FREE_RATE,
    load_all_index_data, load_all_stock_data,
    calculate_all_metrics, calculate_beta, get_stock_index,
    format_large_number, format_ratio, format_percentage,
    plot_cumulative_returns, plot_drawdown, plot_annual_returns,
    plot_risk_return_scatter
)


def retry_on_rate_limit(func, *args, max_retries=3, base_delay=3.0, **kwargs):
    """Execute function with retry on rate limit errors."""
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
    if last_exception:
        raise last_exception
    return func(*args, **kwargs)

st.set_page_config(page_title="Stock Explorer", page_icon="🏢", layout="wide")

st.title("🏢 Stock Explorer")
st.markdown("Explore and analyze individual stocks across indices.")

# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def compute_stock_metrics(_index_data, _stock_data, _stock_info):
    """Compute metrics for all stocks."""
    all_metrics = []
    
    for ticker, data in _stock_data.items():
        info = _stock_info.get(ticker, {})
        metrics = calculate_all_metrics(data, RISK_FREE_RATE)
        
        parent_index = get_stock_index(ticker)
        
        beta = np.nan
        if parent_index and parent_index in _index_data:
            beta = calculate_beta(data['Daily_Return'], _index_data[parent_index]['Daily_Return'])
        
        all_metrics.append({
            'Ticker': ticker,
            'Name': info.get('name', ticker.replace('.NS', '')),
            'Index': parent_index,
            'Total Return (%)': metrics.get('Total Return (%)', np.nan),
            'CAGR (%)': metrics.get('CAGR (%)', np.nan),
            'Volatility (%)': metrics.get('Volatility (%)', np.nan),
            'Sharpe Ratio': metrics.get('Sharpe Ratio', np.nan),
            'Max Drawdown (%)': metrics.get('Max Drawdown (%)', np.nan),
            'Beta': beta,
            'PE Ratio': info.get('pe_ratio', np.nan),
            'PB Ratio': info.get('pb_ratio', np.nan),
            'Dividend Yield (%)': (info.get('dividend_yield', 0) or 0) * 100,
            'Market Cap': info.get('market_cap', np.nan),
            'Sector': info.get('sector', 'N/A')
        })
    
    return pd.DataFrame(all_metrics)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_corporate_events(ticker: str):
    """Fetch corporate events (dividends, splits, upcoming events) for a stock."""
    try:
        return _fetch_corporate_events_internal(ticker)
    except Exception as e:
        error_str = str(e).lower()
        if "429" in str(e) or "rate limit" in error_str:
            # Return empty data with rate limit flag
            return pd.DataFrame(), pd.DataFrame(), {"_rate_limited": True}
        return pd.DataFrame(), pd.DataFrame(), {}


def _fetch_corporate_events_internal(ticker: str):
    """Internal function to fetch corporate events with retry logic."""
    stock = yf.Ticker(ticker)

    # Add small delay to avoid rate limiting
    time.sleep(random.uniform(0.3, 0.6))

    # Get dividends
    dividends = stock.dividends
    if dividends is not None and len(dividends) > 0:
        div_df = dividends.reset_index()
        div_df.columns = ['Date', 'Amount (₹)']
        div_df['Date'] = pd.to_datetime(div_df['Date']).dt.tz_localize(None)
        div_df = div_df.sort_values('Date', ascending=False)
    else:
        div_df = pd.DataFrame(columns=['Date', 'Amount (₹)'])

    # Get stock splits
    splits = stock.splits
    if splits is not None and len(splits) > 0:
        split_df = splits.reset_index()
        split_df.columns = ['Date', 'Ratio']
        split_df['Date'] = pd.to_datetime(split_df['Date']).dt.tz_localize(None)
        split_df['Ratio'] = split_df['Ratio'].apply(lambda x: f"{int(x)}:1" if x >= 1 else f"1:{int(1/x)}")
        split_df = split_df.sort_values('Date', ascending=False)
    else:
        split_df = pd.DataFrame(columns=['Date', 'Ratio'])

    # Get calendar (upcoming events)
    calendar = stock.calendar
    upcoming = {}
    if calendar:
        if 'Earnings Date' in calendar:
            earnings_dates = calendar['Earnings Date']
            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                upcoming['Next Earnings'] = earnings_dates[0]
        if 'Ex-Dividend Date' in calendar:
            upcoming['Ex-Dividend Date'] = calendar['Ex-Dividend Date']
        if 'Earnings Average' in calendar:
            upcoming['Expected EPS'] = calendar['Earnings Average']
        if 'Revenue Average' in calendar:
            upcoming['Expected Revenue'] = calendar['Revenue Average']

    return div_df, split_df, upcoming


with st.spinner("Loading data..."):
    index_data = load_all_index_data()
    stock_data, stock_info = load_all_stock_data()
    metrics_df = compute_stock_metrics(index_data, stock_data, stock_info)

# =============================================================================
# FILTERS
# =============================================================================

st.sidebar.header("🔍 Filters")

# Index filter
filter_index = st.sidebar.selectbox(
    "Filter by Index",
    ["All"] + list(INDICES.keys())
)

if filter_index == "All":
    filtered_df = metrics_df.copy()
    filtered_stocks = list(stock_data.keys())
else:
    filtered_df = metrics_df[metrics_df['Index'] == filter_index].copy()
    filtered_stocks = CONSTITUENTS[filter_index]

# Sector filter
sectors = filtered_df['Sector'].dropna().unique().tolist()
if 'N/A' in sectors:
    sectors.remove('N/A')
    sectors = ['All', 'N/A'] + sorted(sectors)
else:
    sectors = ['All'] + sorted(sectors)

filter_sector = st.sidebar.selectbox("Filter by Sector", sectors)

if filter_sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == filter_sector]

st.sidebar.markdown(f"**Stocks shown:** {len(filtered_df)}")

# =============================================================================
# STOCK SELECTOR
# =============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    available_tickers = filtered_df['Ticker'].tolist()
    if available_tickers:
        selected_stock = st.selectbox(
            "Select Stock",
            available_tickers,
            format_func=lambda x: f"{x.replace('.NS', '')} - {filtered_df[filtered_df['Ticker']==x]['Name'].values[0]}"
        )
    else:
        st.warning("No stocks match the current filters.")
        st.stop()

# =============================================================================
# STOCK DETAILS
# =============================================================================

if selected_stock and selected_stock in stock_data:
    s_data = stock_data[selected_stock]
    s_info = stock_info.get(selected_stock, {})
    s_metrics = filtered_df[filtered_df['Ticker'] == selected_stock].iloc[0]
    
    st.markdown("---")
    st.header(f"{s_info.get('name', selected_stock)} ({selected_stock.replace('.NS', '')})")
    
    # Info row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector", s_info.get('sector', 'N/A'))
    col2.metric("Market Cap", format_large_number(s_info.get('market_cap')))
    col3.metric("PE Ratio", format_ratio(s_info.get('pe_ratio')))
    col4.metric("Dividend Yield", format_percentage(s_metrics.get('Dividend Yield (%)')))
    
    # Performance row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", f"{s_metrics['Total Return (%)']:.1f}%")
    col2.metric("CAGR", f"{s_metrics['CAGR (%)']:.2f}%")
    col3.metric("Volatility", f"{s_metrics['Volatility (%)']:.2f}%")
    col4.metric("Sharpe Ratio", f"{s_metrics['Sharpe Ratio']:.2f}")
    col5.metric("Beta", f"{s_metrics['Beta']:.2f}" if not pd.isna(s_metrics['Beta']) else "N/A")
    
    st.markdown("---")
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Annual Returns", "📉 Drawdown", "📅 Corporate Events"])
    
    with tab1:
        fig = plot_cumulative_returns(
            {selected_stock: s_data},
            f"{selected_stock.replace('.NS', '')} - Cumulative Returns"
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        fig = plot_annual_returns(s_data, selected_stock.replace('.NS', ''))
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        fig = plot_drawdown(s_data, selected_stock.replace('.NS', ''))
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        # Fetch corporate events
        with st.spinner("Loading corporate events..."):
            div_df, split_df, upcoming = fetch_corporate_events(selected_stock)

        # Check for rate limiting
        if upcoming.get("_rate_limited"):
            st.warning("Yahoo Finance rate limit reached. Corporate events data temporarily unavailable. Please try again in a few minutes.")
            upcoming = {}

        # Upcoming Events Section
        if upcoming:
            st.subheader("🔮 Upcoming Events")
            cols = st.columns(len(upcoming))
            for i, (key, value) in enumerate(upcoming.items()):
                with cols[i]:
                    if isinstance(value, (int, float)) and key == 'Expected Revenue':
                        st.metric(key, format_large_number(value))
                    elif isinstance(value, (int, float)):
                        st.metric(key, f"₹{value:.2f}")
                    else:
                        st.metric(key, str(value))
            st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Dividend History")
            if len(div_df) > 0:
                div_display = div_df.head(15).copy()
                div_display['Date'] = pd.to_datetime(div_display['Date']).dt.strftime('%Y-%m-%d')
                st.dataframe(
                    div_display.style.format({
                        'Amount (₹)': '{:.2f}'
                    }),
                    width='stretch',
                    hide_index=True,
                    height=400
                )
            else:
                st.info("No dividend history available.")
        
        with col2:
            st.subheader("✂️ Stock Splits")
            if len(split_df) > 0:
                split_display = split_df.copy()
                split_display['Date'] = pd.to_datetime(split_display['Date']).dt.strftime('%Y-%m-%d')
                st.dataframe(
                    split_display,
                    width='stretch',
                    hide_index=True,
                    height=400
                )
            else:
                st.info("No stock split history available.")

# =============================================================================
# RISK-RETURN SCATTER
# =============================================================================

st.markdown("---")
st.header("⚖️ Risk-Return Analysis")

fig = plot_risk_return_scatter(filtered_df)
st.plotly_chart(fig, width='stretch')

# =============================================================================
# STOCK TABLE
# =============================================================================

st.markdown("---")
st.header("📋 Stock Metrics Table")

# Sort options
sort_col = st.selectbox(
    "Sort by",
    ['CAGR (%)', 'Total Return (%)', 'Sharpe Ratio', 'Volatility (%)', 'PE Ratio', 'Dividend Yield (%)'],
    index=0
)

ascending = st.checkbox("Ascending", value=False)

display_df = filtered_df[[
    'Ticker', 'Name', 'Index', 'Total Return (%)', 'CAGR (%)', 'Volatility (%)',
    'Sharpe Ratio', 'PE Ratio', 'Dividend Yield (%)'
]].copy()

display_df['Ticker'] = display_df['Ticker'].str.replace('.NS', '')
display_df = display_df.sort_values(sort_col, ascending=ascending)

st.dataframe(
    display_df.style.format({
        'Total Return (%)': '{:.2f}',
        'CAGR (%)': '{:.2f}',
        'Volatility (%)': '{:.2f}',
        'Sharpe Ratio': '{:.2f}',
        'PE Ratio': '{:.2f}',
        'Dividend Yield (%)': '{:.2f}'
    }).background_gradient(subset=['Sharpe Ratio'], cmap='Greens'),
    width='stretch',
    hide_index=True,
    height=400
)
