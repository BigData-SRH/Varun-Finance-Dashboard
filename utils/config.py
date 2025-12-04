"""
Configuration constants for Indian Market Analysis.
"""

from datetime import datetime

# =============================================================================
# DATE RANGE
# =============================================================================

START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# =============================================================================
# RISK-FREE RATE
# =============================================================================

RISK_FREE_RATE = 0.06  # 6% annual

# =============================================================================
# INDEX CONFIGURATION
# =============================================================================

INDICES = {
    'Nifty 50': '^NSEI',
    'Nifty Bank': '^NSEBANK',
    'Nifty IT': '^CNXIT'
}

# Global indices for comparison
GLOBAL_INDICES = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI'
}

# =============================================================================
# COLOR SCHEME
# =============================================================================

COLORS = {
    'Nifty 50': '#1f77b4',
    'Nifty Bank': '#d62728',
    'Nifty IT': '#2ca02c'
}

STOCK_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# =============================================================================
# HARDCODED CONSTITUENTS (as of late 2024/early 2025)
# =============================================================================

CONSTITUENTS = {
    'Nifty 50': [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'HCLTECH.NS', 'WIPRO.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'SUNPHARMA.NS',
        'ONGC.NS', 'NTPC.NS', 'TATAMOTORS.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
        'ADANIENT.NS', 'M&M.NS', 'BAJAJFINSV.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS',
        'TECHM.NS', 'ADANIPORTS.NS', 'INDUSINDBK.NS', 'COALINDIA.NS', 'DRREDDY.NS',
        'APOLLOHOSP.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'DIVISLAB.NS', 'HINDALCO.NS',
        'GRASIM.NS', 'BRITANNIA.NS', 'TATACONSUM.NS', 'BPCL.NS', 'HEROMOTOCO.NS',
        'BAJAJ-AUTO.NS', 'LTIM.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'TRENT.NS'
    ],
    'Nifty Bank': [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS',
        'BANDHANBNK.NS', 'AUBANK.NS'
    ],
    'Nifty IT': [
        'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
        'LTIM.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'MPHASIS.NS', 'LTTS.NS'
    ]
}

# =============================================================================
# EVENT PERIODS
# =============================================================================

EVENTS = {
    'COVID Crash': {'start': '2020-02-01', 'end': '2020-04-30'},
    'Post-COVID Recovery': {'start': '2020-05-01', 'end': '2021-12-31'},
    'Recent Period': {'start': '2023-01-01', 'end': END_DATE}
}
