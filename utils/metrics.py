"""
Metrics calculation functions for Indian Market Analysis.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from .config import RISK_FREE_RATE


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily and cumulative returns."""
    df = data.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    return df


def calculate_cagr(data: pd.DataFrame) -> float:
    """Calculate Compound Annual Growth Rate."""
    if len(data) < 2:
        return float('nan')
    
    start_value = float(data['Close'].iloc[0])
    end_value = float(data['Close'].iloc[-1])
    
    # Calculate years as float
    days = (data.index[-1] - data.index[0]).days
    years = float(days) / 365.25
    
    if years <= 0 or start_value <= 0:
        return float('nan')
    
    return float((end_value / start_value) ** (1.0 / years) - 1.0)


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility (standard deviation of returns)."""
    vol = float(returns.std())
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate Sharpe Ratio."""
    mean_return = float(returns.mean())
    excess_returns = mean_return * 252.0 - risk_free_rate
    volatility = calculate_volatility(returns, annualize=True)
    if volatility == 0:
        return float('nan')
    return float(excess_returns / volatility)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate Sortino Ratio (downside risk focus)."""
    mean_return = float(returns.mean())
    excess_returns = mean_return * 252.0 - risk_free_rate
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('nan')
    downside_vol = float(downside_returns.std()) * np.sqrt(252)
    if downside_vol == 0:
        return float('nan')
    return float(excess_returns / downside_vol)


def calculate_max_drawdown(data: pd.DataFrame) -> float:
    """Calculate maximum drawdown."""
    if 'Daily_Return' not in data.columns:
        return float('nan')
    
    cumulative = (1 + data['Daily_Return']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
    mean_return = float(returns.mean())
    cagr = (1.0 + mean_return) ** 252 - 1.0
    if max_drawdown == 0:
        return float('nan')
    return float(abs(cagr / max_drawdown))


def calculate_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    """Calculate beta relative to an index."""
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'index': index_returns
    }).dropna()
    
    if len(aligned) < 30:
        return float('nan')
    
    covariance = float(aligned['stock'].cov(aligned['index']))  # type: ignore[arg-type]
    variance = float(aligned['index'].var())  # type: ignore[arg-type]
    
    if variance == 0:
        return float('nan')
    
    return float(covariance / variance)


def calculate_all_metrics(data: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, Any]:
    """
    Calculate all performance metrics for a dataset.
    
    Returns:
        Dictionary of all metrics
    """
    if data is None or len(data) < 2:
        return {}
    
    returns = data['Daily_Return'].dropna()
    
    # Get scalar values with explicit float conversion
    close_first = float(data['Close'].iloc[0])
    close_last = float(data['Close'].iloc[-1])
    total_return = (close_last / close_first - 1.0) * 100.0
    
    return {
        'Total Return (%)': total_return,
        'CAGR (%)': calculate_cagr(data) * 100.0,
        'Volatility (%)': calculate_volatility(returns) * 100.0,
        'Sharpe Ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'Max Drawdown (%)': calculate_max_drawdown(data) * 100.0,
        'Calmar Ratio': calculate_calmar_ratio(returns, calculate_max_drawdown(data)),
        'Best Day (%)': float(returns.max()) * 100.0,
        'Worst Day (%)': float(returns.min()) * 100.0,
        'Win Rate (%)': float((returns > 0).sum()) / float(len(returns)) * 100.0,
        'Trading Days': len(data)
    }


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_large_number(num: Any) -> str:
    """Format large numbers to readable format (Cr, L)."""
    if num is None or pd.isna(num):
        return 'N/A'
    
    try:
        num = float(num)
    except (TypeError, ValueError):
        return 'N/A'
    
    if num >= 1e12:
        return f'₹{num/1e12:.2f}L Cr'
    elif num >= 1e9:
        return f'₹{num/1e9:.2f}K Cr'
    elif num >= 1e7:
        return f'₹{num/1e7:.2f} Cr'
    elif num >= 1e5:
        return f'₹{num/1e5:.2f} L'
    else:
        return f'₹{num:,.0f}'


def format_percentage(num: Any, decimals: int = 2) -> str:
    """Format number as percentage."""
    if num is None or pd.isna(num):
        return 'N/A'
    try:
        return f'{float(num):.{decimals}f}%'
    except (TypeError, ValueError):
        return 'N/A'


def format_ratio(num: Any, decimals: int = 2) -> str:
    """Format ratio number."""
    if num is None or pd.isna(num):
        return 'N/A'
    try:
        return f'{float(num):.{decimals}f}'
    except (TypeError, ValueError):
        return 'N/A'
