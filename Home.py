"""
Indian Market Analysis Dashboard - Home Page
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from utils import (
    INDICES, RISK_FREE_RATE, START_DATE,
    load_all_index_data, load_global_index_data, calculate_all_metrics,
    plot_cumulative_returns, plot_single_cumulative_returns, plot_drawdown,
    plot_annual_returns, plot_rolling_volatility, plot_correlation_matrix
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* KPI Card Styles - Lighter gradients */
    .kpi-card {
        background: linear-gradient(135deg, #a8c0ff 0%, #c4b5fd 100%);
        border-radius: 16px;
        padding: 1.25rem;
        color: #1e293b;
        text-align: center;
        box-shadow: 0 4px 15px rgba(168, 192, 255, 0.3);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
    }
    .kpi-card.green {
        background: linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%);
        box-shadow: 0 4px 15px rgba(167, 243, 208, 0.3);
    }
    .kpi-card.blue {
        background: linear-gradient(135deg, #bae6fd 0%, #7dd3fc 100%);
        box-shadow: 0 4px 15px rgba(186, 230, 253, 0.3);
    }
    .kpi-card.teal {
        background: linear-gradient(135deg, #99f6e4 0%, #5eead4 100%);
        box-shadow: 0 4px 15px rgba(153, 246, 228, 0.3);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .kpi-index {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.35rem;
    }
    .kpi-insight {
        font-size: 0.7rem;
        font-weight: 500;
        color: #64748b;
        font-style: italic;
    }

    /* Sidebar Index Card */
    .sidebar-index-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-index-name {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .sidebar-index-price {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
    }
    .sidebar-index-change {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sidebar-positive { color: #10b981; }
    .sidebar-negative { color: #ef4444; }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Reduce main container padding */
    .main .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Refresh time styling */
    .refresh-time {
        font-size: 0.75rem;
        color: #94a3b8;
        text-align: center;
        padding: 0.5rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Load data first
    with st.spinner("Loading market data..."):
        index_data = load_all_index_data()
        data_refresh_time = datetime.now()

    if not index_data:
        st.error("Failed to load index data. Please try again.")
        return

    # ==========================================================================
    # SIDEBAR: Market Snapshot & Index Cards
    # ==========================================================================
    
    with st.sidebar:
        st.markdown("### 📊 Market Snapshot")
        
        for name, data in index_data.items():
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            day_change = ((current_price / prev_price) - 1) * 100
            change_class = "sidebar-positive" if day_change >= 0 else "sidebar-negative"
            change_sign = "+" if day_change >= 0 else ""
            
            st.markdown(f"""
            <div class="sidebar-index-card">
                <div class="sidebar-index-name">{name}</div>
                <div class="sidebar-index-price">₹{current_price:,.2f}</div>
                <div class="sidebar-index-change {change_class}">{change_sign}{day_change:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="refresh-time">
            🔄 Last updated<br>{data_refresh_time.strftime('%d %b %Y, %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # HEADER
    # ==========================================================================
    
    st.title("📈 Finance Dashboard")
    st.subheader("Comprehensive market analysis dashboard")

    # ==========================================================================
    # KPI SUMMARY SECTION
    # ==========================================================================
    
    # Calculate metrics for all indices
    all_metrics = {}
    for name, data in index_data.items():
        all_metrics[name] = calculate_all_metrics(data, RISK_FREE_RATE)
    
    # Find best performers
    best_return_idx = max(all_metrics.keys(), key=lambda x: all_metrics[x]['Total Return (%)'])
    best_sharpe_idx = max(all_metrics.keys(), key=lambda x: all_metrics[x]['Sharpe Ratio'])
    lowest_vol_idx = min(all_metrics.keys(), key=lambda x: all_metrics[x]['Volatility (%)'])
    best_cagr_idx = max(all_metrics.keys(), key=lambda x: all_metrics[x]['CAGR (%)'])
    
    # Calculate analysis period for display
    start_year = int(START_DATE[:4])
    current_year = datetime.now().year
    years_analyzed = current_year - start_year
    
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📈 Best Total Return</div>
            <div class="kpi-value">{all_metrics[best_return_idx]['Total Return (%)']:.1f}%</div>
            <div class="kpi-index">{best_return_idx}</div>
            <div class="kpi-insight">Highest wealth growth</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-label">📊 Best CAGR</div>
            <div class="kpi-value">{all_metrics[best_cagr_idx]['CAGR (%)']:.2f}%</div>
            <div class="kpi-index">{best_cagr_idx}</div>
            <div class="kpi-insight">Consistent annual growth</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card blue">
            <div class="kpi-label">⚖️ Best Sharpe Ratio</div>
            <div class="kpi-value">{all_metrics[best_sharpe_idx]['Sharpe Ratio']:.2f}</div>
            <div class="kpi-index">{best_sharpe_idx}</div>
            <div class="kpi-insight">Best risk-adjusted return</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card teal">
            <div class="kpi-label">🛡️ Lowest Volatility</div>
            <div class="kpi-value">{all_metrics[lowest_vol_idx]['Volatility (%)']:.2f}%</div>
            <div class="kpi-index">{lowest_vol_idx}</div>
            <div class="kpi-insight">Most stable returns</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"📅 Analysis Period: Jan {start_year} to Present (~{years_analyzed} years)")
    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # KPI COMPARISON TABLE
    # ==========================================================================
    
    st.markdown("### 📋 Index KPI Comparison")
    
    metrics_list = []
    for name in index_data.keys():
        m = all_metrics[name].copy()
        m['Index'] = name
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[[
        'Index', 'Total Return (%)', 'CAGR (%)', 'Volatility (%)',
        'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Win Rate (%)'
    ]]

    st.dataframe(
        metrics_df.style.format({
            'Total Return (%)': '{:.2f}',
            'CAGR (%)': '{:.2f}',
            'Volatility (%)': '{:.2f}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown (%)': '{:.2f}',
            'Win Rate (%)': '{:.2f}'
        }).background_gradient(subset=['Sharpe Ratio'], cmap='Greens')
         .background_gradient(subset=['CAGR (%)'], cmap='Blues')
         .background_gradient(subset=['Max Drawdown (%)'], cmap='Reds_r'),
        width='stretch',
        hide_index=True
    )
    
    st.caption(f"📅 All metrics calculated from Jan {start_year} to Present (~{years_analyzed} years of data)")

    st.markdown("---")

    # ==========================================================================
    # PERFORMANCE CHART WITH TIME FILTERS
    # ==========================================================================
    
    st.markdown("### 📈 Performance Comparison")
    
    # Time period filter
    time_periods = {
        "1W": 7,
        "1M": 30,
        "6M": 180,
        "1Y": 365,
        "5Y": 1825,
        "MAX": None
    }

    # Initialize session state for period selection
    if "index_period" not in st.session_state:
        st.session_state.index_period = "MAX"

    # Filter data by selected period
    def filter_data_by_period(data_dict, period_key):
        """Filter index data dictionary by time period."""
        days = time_periods[period_key]
        if days is None:  # MAX
            return data_dict
        
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_dict = {}
        for name, data in data_dict.items():
            mask = data.index >= cutoff_date
            filtered_data = data.loc[mask].copy()
            if len(filtered_data) > 0:
                filtered_dict[name] = filtered_data
        return filtered_dict

    filtered_index_data = filter_data_by_period(index_data, st.session_state.index_period)

    fig = plot_cumulative_returns(
        filtered_index_data,
        f"All Indices: Normalized Performance (Base = 100)"
    )
    st.plotly_chart(fig, width='stretch')

    # Time period buttons below the chart
    selected_period = st.radio(
        "Select Time Period",
        options=list(time_periods.keys()),
        index=list(time_periods.keys()).index(st.session_state.index_period),
        horizontal=True,
        key="index_period_radio",
        label_visibility="collapsed"
    )
    
    # Update session state if changed
    if selected_period != st.session_state.index_period:
        st.session_state.index_period = selected_period
        st.rerun()

    st.markdown("---")

    # ==========================================================================
    # DETAILED INDEX ANALYSIS
    # ==========================================================================

    st.markdown("### 🔍 Detailed Index Analysis")

    # Index selector
    selected_index = st.selectbox(
        "Select Index",
        list(INDICES.keys()),
        index=0,
        label_visibility="collapsed"
    )

    data = index_data[selected_index]
    metrics = all_metrics[selected_index]

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("CAGR", f"{metrics['CAGR (%)']:.2f}%")
    col2.metric("Volatility", f"{metrics['Volatility (%)']:.2f}%")
    col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    col4.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    col5.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")

    # Interactive charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance", "📉 Drawdown", "📊 Annual Returns", "🌡️ Volatility"
    ])

    with tab1:
        fig = plot_single_cumulative_returns(
            data,
            selected_index,
            f"{selected_index} - Cumulative Returns"
        )
        st.plotly_chart(fig, width='stretch')

    with tab2:
        fig = plot_drawdown(data, selected_index)
        st.plotly_chart(fig, width='stretch')

    with tab3:
        fig = plot_annual_returns(data, selected_index)
        st.plotly_chart(fig, width='stretch')

    with tab4:
        window = st.slider("Rolling Window (days)", 30, 252, 90, key="vol_window")
        fig = plot_rolling_volatility(data, selected_index, window)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ==========================================================================
    # INDEX CORRELATIONS (Collapsible)
    # ==========================================================================

    with st.expander("🔗 Index Correlations", expanded=False):
        returns_df = pd.DataFrame({
            name: d['Daily_Return'] for name, d in index_data.items()
        }).dropna()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Correlation Heatmap")
            fig = plot_correlation_matrix(returns_df)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("#### Correlation Values")
            corr = returns_df.corr()
            st.dataframe(
                corr.style.format("{:.3f}").background_gradient(cmap='RdBu_r', vmin=-1, vmax=1),
                width='stretch'
            )

            st.markdown("""
            **Interpretation:**
            - Values close to **1.0** indicate strong positive correlation
            - Values close to **0** indicate low correlation (good for diversification)
            - Values close to **-1.0** indicate inverse correlation
            """)

    # ==========================================================================
    # GLOBAL MARKET CONTEXT (Collapsible)
    # ==========================================================================

    with st.expander("🌍 Global Market Context", expanded=False):
        st.markdown("#### How Indian Indices Compare to Global Markets")
        
        with st.spinner("Loading global market data..."):
            global_data = load_global_index_data()
        
        if global_data:
            # Calculate metrics for global indices
            global_metrics = {}
            for name, data in global_data.items():
                global_metrics[name] = calculate_all_metrics(data, RISK_FREE_RATE)
            
            # Combine Indian and Global metrics for comparison
            comparison_data = []
            
            # Add Indian indices
            for name in index_data.keys():
                m = all_metrics[name]
                comparison_data.append({
                    'Index': f"🇮🇳 {name}",
                    'Region': 'India',
                    '1Y Return (%)': m.get('Total Return (%)', 0) / (m.get('Years', 1) or 1),  # Approx
                    'CAGR (%)': m['CAGR (%)'],
                    'Volatility (%)': m['Volatility (%)'],
                    'Sharpe Ratio': m['Sharpe Ratio']
                })
            
            # Add Global indices
            region_flags = {
                'S&P 500': '🇺🇸', 'NASDAQ': '🇺🇸', 'FTSE 100': '🇬🇧',
                'DAX': '🇩🇪', 'Nikkei 225': '🇯🇵', 'Hang Seng': '🇭🇰'
            }
            for name, m in global_metrics.items():
                flag = region_flags.get(name, '🌐')
                comparison_data.append({
                    'Index': f"{flag} {name}",
                    'Region': 'Global',
                    '1Y Return (%)': m.get('Total Return (%)', 0) / (m.get('Years', 1) or 1),
                    'CAGR (%)': m['CAGR (%)'],
                    'Volatility (%)': m['Volatility (%)'],
                    'Sharpe Ratio': m['Sharpe Ratio']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns([1.2, 0.8])
            
            with col1:
                st.markdown("##### Performance Comparison")
                st.dataframe(
                    comparison_df.style.format({
                        '1Y Return (%)': '{:.2f}',
                        'CAGR (%)': '{:.2f}',
                        'Volatility (%)': '{:.2f}',
                        'Sharpe Ratio': '{:.2f}'
                    }).background_gradient(subset=['CAGR (%)'], cmap='RdYlGn')
                     .background_gradient(subset=['Sharpe Ratio'], cmap='Blues'),
                    width='stretch',
                    hide_index=True
                )
            
            with col2:
                st.markdown("##### Key Insights")
                
                # Find best performers
                best_global_cagr = max(global_metrics.items(), key=lambda x: x[1]['CAGR (%)'])
                best_indian_cagr = max(all_metrics.items(), key=lambda x: x[1]['CAGR (%)'])
                
                indian_avg_sharpe = sum(m['Sharpe Ratio'] for m in all_metrics.values()) / len(all_metrics)
                global_avg_sharpe = sum(m['Sharpe Ratio'] for m in global_metrics.values()) / len(global_metrics)
                
                st.markdown(f"""
                📊 **Best Global CAGR:** {best_global_cagr[0]} ({best_global_cagr[1]['CAGR (%)']:.2f}%)
                
                📈 **Best Indian CAGR:** {best_indian_cagr[0]} ({best_indian_cagr[1]['CAGR (%)']:.2f}%)
                
                ⚖️ **Avg Sharpe (India):** {indian_avg_sharpe:.2f}
                
                🌐 **Avg Sharpe (Global):** {global_avg_sharpe:.2f}
                """)
                
                if indian_avg_sharpe > global_avg_sharpe:
                    st.success("Indian indices show better risk-adjusted returns!")
                else:
                    st.info("Global indices showing stronger risk-adjusted performance.")
        else:
            st.warning("Could not load global market data. Please try again later.")

    # ==========================================================================
    # UNDERSTANDING THE METRICS (Educational Section)
    # ==========================================================================

    with st.expander("📚 Understanding the Metrics & How to Invest", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📖 What Do These Metrics Mean?")
            
            st.markdown("""
            **📈 Total Return**
            - The overall percentage gain/loss over the entire period
            - Higher is better for wealth accumulation
            
            **📊 CAGR (Compound Annual Growth Rate)**
            - Average yearly return, accounting for compounding
            - Best metric for comparing investments over different time periods
            - Example: 15% CAGR means ₹1 lakh becomes ~₹2 lakh in 5 years
            
            **🌡️ Volatility**
            - Measures price fluctuation (standard deviation of returns)
            - Lower volatility = smoother, more predictable returns
            - Higher volatility = bigger swings, more risk
            
            **⚖️ Sharpe Ratio**
            - Return earned per unit of risk taken
            - Above 1.0 = Good, Above 2.0 = Excellent
            - Helps compare investments with different risk levels
            
            **📉 Max Drawdown**
            - Largest peak-to-trough decline
            - Shows worst-case scenario during market crashes
            - Lower is better for risk management
            """)
        
        with col2:
            st.markdown("#### 💰 How to Invest in Indices?")
            
            st.markdown("""
            > **Note:** You cannot directly buy an index like Nifty 50. 
            > But you can invest through **Index Funds** and **ETFs**!
            
            **🏦 Nifty 50 Investment Options:**
            - **ETFs:** Nippon Nifty BeES, SBI Nifty 50 ETF, ICICI Nifty 50 ETF
            - **Index Funds:** UTI Nifty 50 Index Fund, HDFC Nifty 50 Index Fund
            
            **🏛️ Nifty Bank Investment Options:**
            - **ETFs:** Nippon Bank BeES, Kotak Nifty Bank ETF
            - **Index Funds:** ICICI Pru Nifty Bank Index Fund
            
            **💻 Nifty IT Investment Options:**
            - **ETFs:** ICICI Pru IT ETF, Nippon IT ETF
            - **Index Funds:** ICICI Pru Nifty IT Index Fund
            
            ---
            
            **📌 Why Track Indices?**
            - Benchmark for portfolio performance
            - Understand market trends and sectors
            - Low-cost diversification through index investing
            - "Beat the index" is the goal for active fund managers
            """)
        
        st.markdown("---")
        st.caption("""
        ⚠️ **Disclaimer:** This dashboard is for educational and informational purposes only. 
        It does not constitute financial advice. Always consult a qualified financial advisor 
        before making investment decisions. Past performance does not guarantee future results.
        """)


if __name__ == "__main__":
    main()