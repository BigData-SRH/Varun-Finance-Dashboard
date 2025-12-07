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
    plot_risk_return_scatter,
    # New technical indicators
    calculate_52_week_high_low, calculate_moving_average_position,
    calculate_rsi, get_rsi_status, calculate_dividend_growth_rate
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

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Stock Card Styles */
    .stock-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        color: #1e293b;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        height: 100%;
    }
    .stock-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-color: #cbd5e1;
    }
    
    /* Stock Header Card */
    .stock-header-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 2rem;
        color: #1e293b;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .stock-header-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    .stock-header-name {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .stock-header-ticker {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .stock-header-price {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .stock-header-change {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Company Info Card */
    .company-info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    .company-info-card:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1.5rem;
    }
    .info-item {
        text-align: center;
    }
    .info-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .info-value {
        font-size: 1.3rem;
        color: #1e293b;
        font-weight: 700;
    }
    .info-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        border-color: #cbd5e1;
    }
    .metric-card-header {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .metric-item {
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* Technical Indicator Cards */
    .tech-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    .tech-card:hover {
        border-color: #94a3b8;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    .tech-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .tech-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .badge-positive {
        background: #dcfce7;
        color: #166534;
    }
    .badge-negative {
        background: #fee2e2;
        color: #991b1b;
    }
    .badge-neutral {
        background: #fef3c7;
        color: #854d0e;
    }
    .badge-overbought {
        background: #fee2e2;
        color: #991b1b;
    }
    .badge-oversold {
        background: #dcfce7;
        color: #166534;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Reduce padding */
    .main .block-container {
        padding-top: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Stock Explorer")
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


# SESSION STATE CACHING: Load data once per session
if 'stock_explorer_data' not in st.session_state or st.sidebar.button("🔄 Refresh", help="Reload data"):
    with st.spinner("Loading data..."):
        st.session_state.stock_explorer_data = {
            'index_data': load_all_index_data(),
            'stock_data': load_all_stock_data()[0],  # Only historical data
            'stock_info': load_all_stock_data()[1]   # Metadata only
        }

index_data = st.session_state.stock_explorer_data['index_data']
stock_data = st.session_state.stock_explorer_data['stock_data']
stock_info = st.session_state.stock_explorer_data['stock_info']
metrics_df = compute_stock_metrics(index_data, stock_data, stock_info)

# =============================================================================
# FILTERS
# =============================================================================

st.sidebar.header("Filters")

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
    
    # LAZY LOADING: Fetch detailed info only for selected stock
    from utils import fetch_stock_info_lazy
    s_info = fetch_stock_info_lazy(selected_stock)
    
    s_metrics = filtered_df[filtered_df['Ticker'] == selected_stock].iloc[0]
    
    st.markdown("---")
    
    # Calculate current price and change
    current_price = s_data['Close'].iloc[-1]
    prev_price = s_data['Close'].iloc[-2] if len(s_data) > 1 else current_price
    day_change = ((current_price / prev_price) - 1) * 100 if prev_price != 0 else 0
    change_color = "#10b981" if day_change >= 0 else "#ef4444"
    change_sign = "+" if day_change >= 0 else ""
    
    # Stock Header Card
    st.markdown(f"""
    <div class="stock-header-card">
        <div class="stock-header-name">{s_info.get('name', selected_stock)}</div>
        <div class="stock-header-ticker">{selected_stock.replace('.NS', '')} • {s_info.get('sector', 'N/A')}</div>
        <div style="display: flex; align-items: baseline; gap: 1rem;">
            <div class="stock-header-price">₹{current_price:,.2f}</div>
            <div class="stock-header-change" style="color: {change_color};">{change_sign}{day_change:.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Company Info Card
    st.markdown(f"""
    <div class="company-info-card">
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Sector</div>
                <div class="info-value">{s_info.get('sector', 'N/A')}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Market Cap</div>
                <div class="info-value">{format_large_number(s_info.get('market_cap'))}</div>
            </div>
            <div class="info-item">
                <div class="info-label">PE Ratio</div>
                <div class="info-value">{format_ratio(s_info.get('pe_ratio'))}</div>
            </div>
            <div class="info-item">
                <div class="info-label">PB Ratio</div>
                <div class="info-value">{format_ratio(s_info.get('pb_ratio'))}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Dividend Yield</div>
                <div class="info-value">{format_percentage(s_metrics.get('Dividend Yield (%)'))}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics Cards
    st.markdown("### Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">Returns</div>
            <div class="metric-item">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{s_metrics['Total Return (%)']:.1f}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">{s_metrics['CAGR (%)']:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        beta_display = f"{s_metrics['Beta']:.2f}" if not pd.isna(s_metrics['Beta']) else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">Risk</div>
            <div class="metric-item">
                <div class="metric-label">Volatility</div>
                <div class="metric-value">{s_metrics['Volatility (%)']:.2f}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{s_metrics['Max Drawdown (%)']:.2f}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Beta</div>
                <div class="metric-value">{beta_display}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">Quality</div>
            <div class="metric-item">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{s_metrics['Sharpe Ratio']:.2f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{calculate_all_metrics(s_data, RISK_FREE_RATE).get('Win Rate (%)', 0):.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Indicators Section
    st.markdown("### Technical Indicators")
    
    tech_52w = calculate_52_week_high_low(s_data)
    tech_ma = calculate_moving_average_position(s_data)
    tech_rsi = calculate_rsi(s_data)
    rsi_status = get_rsi_status(tech_rsi)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        distance_52w = tech_52w['Distance from 52W High (%)']
        badge_class = "badge-positive" if distance_52w > -5 else "badge-neutral" if distance_52w > -15 else "badge-negative"
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-label">52-Week High</div>
            <div class="tech-value">{distance_52w:.1f}%</div>
            <div class="tech-badge {badge_class}">
                {'Near High' if distance_52w > -5 else 'Moderate' if distance_52w > -15 else 'Far from High'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_badge_class = "badge-overbought" if rsi_status == "Overbought" else "badge-oversold" if rsi_status == "Oversold" else "badge-neutral"
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-label">RSI (14)</div>
            <div class="tech-value">{tech_rsi:.1f}</div>
            <div class="tech-badge {rsi_badge_class}">{rsi_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ma50_dist = tech_ma['Distance from MA 50 (%)']
        ma50_above = tech_ma['Above MA 50']
        ma50_badge = "badge-positive" if ma50_above else "badge-negative"
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-label">vs MA 50</div>
            <div class="tech-value">{ma50_dist:.1f}%</div>
            <div class="tech-badge {ma50_badge}">{'Above' if ma50_above else 'Below'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if not pd.isna(tech_ma['MA 200']):
            ma200_dist = tech_ma['Distance from MA 200 (%)']
            ma200_above = tech_ma['Above MA 200']
            ma200_badge = "badge-positive" if ma200_above else "badge-negative"
            st.markdown(f"""
            <div class="tech-card">
                <div class="tech-label">vs MA 200</div>
                <div class="tech-value">{ma200_dist:.1f}%</div>
                <div class="tech-badge {ma200_badge}">{'Above' if ma200_above else 'Below'}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tech-card">
                <div class="tech-label">vs MA 200</div>
                <div class="tech-value">N/A</div>
                <div class="tech-badge badge-neutral">Insufficient Data</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price History", "Performance", "Annual Returns", "Drawdown", "Corporate Events"])
    
    with tab1:
        # Raw price chart (not normalized)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s_data.index,
            y=s_data['Close'],
            mode='lines',
            name=selected_stock.replace('.NS', ''),
            line=dict(width=2, color='#3b82f6'),
            hovertemplate=f'{selected_stock.replace(".NS", "")}<br>Date: %{{x}}<br>Price: ₹%{{y:,.2f}}<extra></extra>'
        ))
        fig.update_layout(
            title=f"{selected_stock.replace('.NS', '')} - Price History",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        fig = plot_cumulative_returns(
            {selected_stock: s_data},
            f"{selected_stock.replace('.NS', '')} - Cumulative Returns"
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        fig = plot_annual_returns(s_data, selected_stock.replace('.NS', ''))
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        fig = plot_drawdown(s_data, selected_stock.replace('.NS', ''))
        st.plotly_chart(fig, width='stretch')
    
    with tab5:
        # Fetch corporate events
        with st.spinner("Loading corporate events..."):
            div_df, split_df, upcoming = fetch_corporate_events(selected_stock)

        # Check for rate limiting
        if upcoming.get("_rate_limited"):
            st.warning("Yahoo Finance rate limit reached. Corporate events data temporarily unavailable. Please try again in a few minutes.")
            upcoming = {}

        # Upcoming Events Section
        if upcoming:
            st.subheader("Upcoming Events")
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
            st.subheader("Dividend History")
            if len(div_df) > 0:
                # Calculate dividend growth metrics from div_df
                div_metrics = {}
                if len(div_df) >= 2:
                    try:
                        div_series = pd.Series(
                            div_df['Amount (₹)'].values,
                            index=pd.to_datetime(div_df['Date'])
                        )
                        div_metrics = calculate_dividend_growth_rate(div_series)
                    except Exception:
                        div_metrics = {}

                # Show dividend growth rate if available
                if not pd.isna(div_metrics.get('Dividend Growth Rate (%)', float('nan'))):
                    growth_rate = div_metrics['Dividend Growth Rate (%)']
                    st.metric(
                        "Dividend Growth Rate (CAGR)",
                        f"{growth_rate:.1f}%",
                        delta="Growing" if growth_rate > 0 else "Declining",
                        delta_color="normal" if growth_rate > 0 else "inverse"
                    )

                div_display = div_df.head(15).copy()
                div_display['Date'] = pd.to_datetime(div_display['Date']).dt.strftime('%Y-%m-%d')
                st.dataframe(
                    div_display.style.format({
                        'Amount (₹)': '{:.2f}'
                    }),
                    width='stretch',
                    hide_index=True,
                    height=350
                )
            else:
                st.info("No dividend history available.")
        
        with col2:
            st.subheader("Stock Splits")
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
st.header("Risk-Return Analysis")

fig = plot_risk_return_scatter(filtered_df)
st.plotly_chart(fig, width='stretch')

# =============================================================================
# STOCK TABLE
# =============================================================================

st.markdown("---")
st.header("Stock Metrics Table")

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
