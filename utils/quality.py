"""
Data quality scoring and stock selection functions.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .config import CONSTITUENTS


def calculate_quality_score(
    ticker: str, 
    price_data: Optional[pd.DataFrame], 
    info_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate quality score (0-100) for a stock.
    
    Scoring:
    - Price completeness (40 pts)
    - Volume data presence (20 pts)
    - Fundamental data availability (40 pts)
    
    Returns:
        Dictionary with quality score breakdown
    """
    score = 0.0
    details: Dict[str, Any] = {}
    
    # Price completeness (40 pts)
    if price_data is not None and len(price_data) > 0:
        total_days = (price_data.index[-1] - price_data.index[0]).days
        expected_trading_days = total_days * (252 / 365)
        actual_days = len(price_data)
        
        completeness = min(100.0, (actual_days / expected_trading_days) * 100) if expected_trading_days > 0 else 0.0
        price_score = (completeness / 100.0) * 40.0
        score += price_score
        details['price_completeness'] = completeness
        details['price_score'] = price_score
    else:
        details['price_completeness'] = 0.0
        details['price_score'] = 0.0
    
    # Volume data (20 pts)
    if price_data is not None and 'Volume' in price_data.columns and len(price_data) > 0:
        valid_volume = int((price_data['Volume'] > 0).sum())
        volume_pct = (valid_volume / len(price_data)) * 100.0
        volume_score = (volume_pct / 100.0) * 20.0
        score += volume_score
        details['volume_pct'] = volume_pct
        details['volume_score'] = volume_score
    else:
        details['volume_pct'] = 0.0
        details['volume_score'] = 0.0
    
    # Fundamental data (40 pts)
    fundamental_fields = ['pe_ratio', 'pb_ratio', 'market_cap', 'roe', 'dividend_yield']
    available_fields = 0
    
    if info_data:
        for field in fundamental_fields:
            val = info_data.get(field)
            if val is not None and not pd.isna(val):
                available_fields += 1
    
    fundamental_pct = (available_fields / len(fundamental_fields)) * 100.0
    fundamental_score = (fundamental_pct / 100.0) * 40.0
    score += fundamental_score
    details['fundamental_pct'] = fundamental_pct
    details['fundamental_score'] = fundamental_score
    
    details['total_score'] = score
    details['tier'] = 'Tier 1' if score >= 80 else ('Tier 2' if score >= 60 else 'Tier 3')
    
    return details


def assess_all_quality(
    stock_data: Dict[str, pd.DataFrame], 
    stock_info: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Assess quality for all stocks.
    
    Returns:
        DataFrame with quality assessment
    """
    quality_list: List[Dict[str, Any]] = []
    
    for ticker in stock_data.keys():
        price_data = stock_data.get(ticker)
        info_data = stock_info.get(ticker, {})
        
        quality = calculate_quality_score(ticker, price_data, info_data)
        quality['ticker'] = ticker
        quality['name'] = info_data.get('name', ticker.replace('.NS', ''))
        
        quality_list.append(quality)
    
    df = pd.DataFrame(quality_list)
    df = df.sort_values('total_score', ascending=False).reset_index(drop=True)
    
    return df


def select_top_stocks(
    quality_df: pd.DataFrame, 
    stock_info: Dict[str, Dict[str, Any]], 
    n_per_index: int = 6
) -> Dict[str, List[str]]:
    """
    Select top stocks per index based on quality, market cap, and liquidity.
    
    Returns:
        Dictionary of selected stocks per index
    """
    selected: Dict[str, List[str]] = {}
    
    for index_name, tickers in CONSTITUENTS.items():
        # Filter to Tier 1 stocks in this index
        index_quality = quality_df[
            (quality_df['ticker'].isin(tickers)) & 
            (quality_df['tier'] == 'Tier 1')
        ].copy()
        
        # If not enough Tier 1, include Tier 2
        if len(index_quality) < n_per_index:
            tier2 = quality_df[
                (quality_df['ticker'].isin(tickers)) & 
                (quality_df['tier'] == 'Tier 2')
            ]
            index_quality = pd.concat([index_quality, tier2])
        
        # Add market cap for sorting
        def get_market_cap(ticker: str) -> float:
            info = stock_info.get(ticker, {})
            mc = info.get('market_cap', 0)
            return float(mc) if mc is not None and not pd.isna(mc) else 0.0
        
        index_quality['market_cap'] = index_quality['ticker'].apply(get_market_cap)
        
        # Sort by quality score and market cap
        index_quality = index_quality.sort_values(
            ['total_score', 'market_cap'], 
            ascending=[False, False]
        )
        
        # Select top N
        selected[index_name] = index_quality['ticker'].head(n_per_index).tolist()
    
    return selected
