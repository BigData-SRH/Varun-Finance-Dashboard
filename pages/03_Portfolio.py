"""
Portfolio & Insights Page
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    INDICES, COLORS, RISK_FREE_RATE,
    load_all_index_data, load_all_stock_data,
    calculate_all_metrics, calculate_beta, get_stock_index,
    plot_portfolio_allocation, plot_portfolio_risk_return,
    plot_top_performers_bar
)

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")

st.title("Portfolio & Insights")
st.markdown("Top performers, value opportunities, and portfolio suggestions.")

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
            'ROE (%)': (info.get('roe', 0) or 0) * 100,
            'Sector': info.get('sector', 'N/A')
        })
    
    return pd.DataFrame(all_metrics)


with st.spinner("Loading data..."):
    index_data = load_all_index_data()
    stock_data, stock_info = load_all_stock_data()
    metrics_df = compute_stock_metrics(index_data, stock_data, stock_info)

# =============================================================================
# TOP PERFORMERS SECTION
# =============================================================================

st.header("Top Performers")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Highest Returns")
    top_return = metrics_df.nlargest(10, 'Total Return (%)')[
        ['Ticker', 'Name', 'Index', 'Total Return (%)']
    ].copy()
    top_return['Ticker'] = top_return['Ticker'].str.replace('.NS', '')
    st.dataframe(
        top_return.style.format({'Total Return (%)': '{:.1f}'}),
        width='stretch',
        hide_index=True
    )

with col2:
    st.subheader("Best Risk-Adjusted (Sharpe)")
    top_sharpe = metrics_df[metrics_df['Sharpe Ratio'] > 0].nlargest(10, 'Sharpe Ratio')[
        ['Ticker', 'Name', 'Index', 'Sharpe Ratio']
    ].copy()
    top_sharpe['Ticker'] = top_sharpe['Ticker'].str.replace('.NS', '')
    st.dataframe(
        top_sharpe.style.format({'Sharpe Ratio': '{:.2f}'}),
        width='stretch',
        hide_index=True
    )

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Dividend Payers")
    top_div = metrics_df[metrics_df['Dividend Yield (%)'] > 0].nlargest(10, 'Dividend Yield (%)')[
        ['Ticker', 'Name', 'Index', 'Dividend Yield (%)']
    ].copy()
    top_div['Ticker'] = top_div['Ticker'].str.replace('.NS', '')
    st.dataframe(
        top_div.style.format({'Dividend Yield (%)': '{:.2f}'}),
        width='stretch',
        hide_index=True
    )

with col2:
    st.subheader("Value Opportunities (Low PE)")
    undervalued = metrics_df[
        (metrics_df['PE Ratio'] > 0) & 
        (metrics_df['PE Ratio'] < 15) &
        (metrics_df['CAGR (%)'] > 10)
    ].nsmallest(10, 'PE Ratio')[
        ['Ticker', 'Name', 'PE Ratio', 'CAGR (%)']
    ].copy()
    undervalued['Ticker'] = undervalued['Ticker'].str.replace('.NS', '')
    st.dataframe(
        undervalued.style.format({'PE Ratio': '{:.2f}', 'CAGR (%)': '{:.1f}'}),
        width='stretch',
        hide_index=True
    )
    st.caption("*Stocks with PE < 15 and CAGR > 10%*")

# =============================================================================
# PORTFOLIO SUGGESTION
# =============================================================================

st.markdown("---")
st.header("Suggested Portfolio")

st.markdown("""
The portfolio is selected based on:
- Sharpe Ratio > 0.3 (good risk-adjusted returns)
- CAGR > 5% (positive growth)
- Diversification across indices
- Composite score weighing Sharpe, CAGR, dividends, and volatility
""")

# Build portfolio
portfolio_candidates = metrics_df[
    (metrics_df['Sharpe Ratio'] > 0.3) &
    (metrics_df['CAGR (%)'] > 5)
].copy()

if len(portfolio_candidates) > 0:
    portfolio_candidates['Composite Score'] = (
        portfolio_candidates['Sharpe Ratio'] * 30 +
        portfolio_candidates['CAGR (%)'].clip(upper=50) * 1 +
        portfolio_candidates['Dividend Yield (%)'].fillna(0) * 5 -
        portfolio_candidates['Volatility (%)'].clip(upper=50) * 0.5
    )
    
    # Select top from each index
    portfolio = []
    for idx_name in INDICES.keys():
        idx_candidates = portfolio_candidates[portfolio_candidates['Index'] == idx_name]
        if len(idx_candidates) > 0:
            top_2 = idx_candidates.nlargest(2, 'Composite Score')
            portfolio.extend(top_2.to_dict('records'))
    
    portfolio_df = pd.DataFrame(portfolio).head(7)
    
    # Display portfolio
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg CAGR", f"{portfolio_df['CAGR (%)'].mean():.2f}%")
    col2.metric("Avg Sharpe", f"{portfolio_df['Sharpe Ratio'].mean():.2f}")
    col3.metric("Indices Covered", f"{portfolio_df['Index'].nunique()}/{len(INDICES)}")
    
    # Portfolio table
    display_cols = ['Ticker', 'Name', 'Index', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Dividend Yield (%)']
    portfolio_display = portfolio_df[display_cols].copy()
    portfolio_display['Ticker'] = portfolio_display['Ticker'].str.replace('.NS', '')
    
    st.dataframe(
        portfolio_display.style.format({
            'CAGR (%)': '{:.2f}',
            'Volatility (%)': '{:.2f}',
            'Sharpe Ratio': '{:.2f}',
            'Dividend Yield (%)': '{:.2f}'
        }),
        width='stretch',
        hide_index=True
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_portfolio_allocation(portfolio_df)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = plot_portfolio_risk_return(portfolio_df)
        st.plotly_chart(fig, width='stretch')

else:
    st.warning("Not enough stocks meet the portfolio criteria.")

# =============================================================================
# KEY INSIGHTS
# =============================================================================

st.markdown("---")
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    # Index performance
    st.subheader("Index Performance Summary")
    
    index_summary = []
    for name, data in index_data.items():
        m = calculate_all_metrics(data, RISK_FREE_RATE)
        index_summary.append({
            'Index': name,
            'CAGR (%)': m['CAGR (%)'],
            'Sharpe': m['Sharpe Ratio'],
            'Max DD (%)': m['Max Drawdown (%)']
        })
    
    idx_df = pd.DataFrame(index_summary)
    st.dataframe(
        idx_df.style.format({
            'CAGR (%)': '{:.2f}',
            'Sharpe': '{:.2f}',
            'Max DD (%)': '{:.2f}'
        }),
        width='stretch',
        hide_index=True
    )
    
    best_idx = idx_df.loc[idx_df['Sharpe'].idxmax(), 'Index']
    st.success(f"**Best Risk-Adjusted Index:** {best_idx}")

with col2:
    st.subheader("Stock Universe Summary")
    
    st.metric("Total Stocks Analyzed", len(metrics_df))
    
    positive_cagr = (metrics_df['CAGR (%)'] > 0).sum()
    st.metric("Positive CAGR", f"{positive_cagr} ({positive_cagr/len(metrics_df)*100:.0f}%)")
    
    high_sharpe = (metrics_df['Sharpe Ratio'] > 0.5).sum()
    st.metric("Sharpe > 0.5", f"{high_sharpe} stocks")
    
    dividend_payers = (metrics_df['Dividend Yield (%)'] > 1).sum()
    st.metric("Dividend Yield > 1%", f"{dividend_payers} stocks")

# =============================================================================
# DISCLAIMER
# =============================================================================

st.markdown("---")
st.warning("""
**Disclaimer:** This analysis is for educational purposes only. Past performance does not guarantee future returns.
Please consult a qualified financial advisor before making any investment decisions.
""")
