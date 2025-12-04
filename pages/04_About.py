"""
About & Documentation Page
"""

import streamlit as st

from utils import (
    START_DATE, END_DATE, RISK_FREE_RATE,
    INDICES, CONSTITUENTS
)

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Dashboard")

# =============================================================================
# OVERVIEW
# =============================================================================

st.header("📊 Overview")

st.markdown(f"""
This dashboard provides comprehensive analysis of Indian market indices and their constituent stocks.

**Analysis Period:** {START_DATE} to {END_DATE}

**Indices Covered:**
- **Nifty 50** - India's benchmark broad market index (50 large-cap stocks)
- **Nifty Bank** - Banking sector index (12 major banks)
- **Nifty IT** - Information Technology sector index (10 IT companies)
""")

# =============================================================================
# METHODOLOGY
# =============================================================================

st.header("📐 Methodology")

st.subheader("Data Source")
st.markdown("""
All data is fetched from **Yahoo Finance** using the `yfinance` library:
- Historical OHLCV (Open, High, Low, Close, Volume) data
- Fundamental data (PE ratio, market cap, dividend yield, etc.)
- Data is fetched fresh on each session (no local caching)
""")

st.subheader("Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Return Metrics:**
    - **Total Return**: Simple price return over the period
    - **CAGR**: Compound Annual Growth Rate
    - **Win Rate**: Percentage of positive daily returns
    
    **Risk Metrics:**
    - **Volatility**: Annualized standard deviation of returns
    - **Max Drawdown**: Largest peak-to-trough decline
    - **Beta**: Sensitivity to parent index movements
    """)

with col2:
    st.markdown(f"""
    **Risk-Adjusted Metrics:**
    - **Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility
    - **Sortino Ratio**: Uses only downside volatility
    - **Calmar Ratio**: CAGR / Max Drawdown
    
    **Parameters:**
    - Risk-Free Rate: {RISK_FREE_RATE * 100:.0f}% (annual)
    - Trading Days/Year: 252
    """)

st.subheader("Quality Scoring")

st.markdown("""
Stocks are scored (0-100) based on data quality:

| Component | Weight | Criteria |
|-----------|--------|----------|
| Price Completeness | 40 pts | % of expected trading days with data |
| Volume Data | 20 pts | % of days with valid volume |
| Fundamentals | 40 pts | Availability of PE, PB, Market Cap, ROE, Dividend Yield |

**Tiers:**
- **Tier 1** (80-100): Full analysis
- **Tier 2** (60-79): Limited analysis with warnings
- **Tier 3** (<60): Excluded or minimal mention
""")

# =============================================================================
# CONSTITUENTS
# =============================================================================

st.header("📋 Index Constituents")

tab1, tab2, tab3 = st.tabs(["Nifty 50", "Nifty Bank", "Nifty IT"])

with tab1:
    st.markdown(f"**{len(CONSTITUENTS['Nifty 50'])} stocks:**")
    cols = st.columns(5)
    for i, stock in enumerate(CONSTITUENTS['Nifty 50']):
        cols[i % 5].markdown(f"• {stock.replace('.NS', '')}")

with tab2:
    st.markdown(f"**{len(CONSTITUENTS['Nifty Bank'])} stocks:**")
    cols = st.columns(4)
    for i, stock in enumerate(CONSTITUENTS['Nifty Bank']):
        cols[i % 4].markdown(f"• {stock.replace('.NS', '')}")

with tab3:
    st.markdown(f"**{len(CONSTITUENTS['Nifty IT'])} stocks:**")
    cols = st.columns(3)
    for i, stock in enumerate(CONSTITUENTS['Nifty IT']):
        cols[i % 3].markdown(f"• {stock.replace('.NS', '')}")

# =============================================================================
# LIMITATIONS
# =============================================================================

st.header("⚠️ Limitations")

st.markdown("""
- **Data Quality**: Yahoo Finance data may have gaps or inaccuracies
- **Survivorship Bias**: Analysis only includes current index constituents
- **No Transaction Costs**: Returns don't account for brokerage, taxes, or slippage
- **Point-in-Time**: Constituent lists are current; historical membership not tracked
- **Fundamental Data**: May be delayed or incomplete for some stocks
""")

# =============================================================================
# DISCLAIMER
# =============================================================================

st.header("📜 Disclaimer")

st.error("""
**IMPORTANT DISCLAIMER**

This dashboard is provided for **educational and informational purposes only**.

- This is **NOT investment advice**
- Past performance does **NOT guarantee future results**
- Always consult a **qualified financial advisor** before investing
- The creators assume **NO liability** for investment decisions
- Data accuracy is **NOT guaranteed**

By using this dashboard, you acknowledge that you understand these limitations.
""")

# =============================================================================
# TECHNICAL INFO
# =============================================================================

st.header("🔧 Technical Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Built With:**
    - Python 3.11+
    - Streamlit
    - Plotly
    - Matplotlib
    - yfinance
    - pandas / numpy
    """)

with col2:
    st.markdown("""
    **Project Structure:**
    ```
    indian_market_analysis/
    ├── app.py
    ├── pages/
    │   ├── 01_Index_Analysis.py
    │   ├── 02_Stock_Explorer.py
    │   ├── 03_Portfolio.py
    │   └── 04_About.py
    └── utils/
        ├── config.py
        ├── data_fetcher.py
        ├── metrics.py
        ├── quality.py
        └── charts.py
    ```
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

# =============================================================================
# AUTHOR
# =============================================================================

st.header("👤 Author")

col1, col2 = st.columns([1, 3])

with col2:
    st.markdown("""
    **Varun H Shamaraju**
    
    💼 LinkedIn: [linkedin.com/in/varunhs306](https://www.linkedin.com/in/varunhs306/)
    
    🐙 GitHub: [github.com/varunhs306](https://github.com/varunhs306)
    """)

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>Indian Market Analysis Dashboard | Data Period: {START_DATE} - {END_DATE}</div>",
    unsafe_allow_html=True
)
