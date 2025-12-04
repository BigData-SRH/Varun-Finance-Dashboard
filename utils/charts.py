"""
Plotly chart functions for Indian Market Analysis.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .config import COLORS, INDICES, RISK_FREE_RATE


def plot_cumulative_returns(data_dict: Dict[str, pd.DataFrame], title: str = "Cumulative Returns") -> go.Figure:
    """Create interactive cumulative returns chart (normalized to base 100 for comparison)."""
    fig = go.Figure()
    
    for name, data in data_dict.items():
        normalized = (data['Close'] / data['Close'].iloc[0]) * 100
        color = COLORS.get(name)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=normalized,
            name=name.replace('.NS', ''),
            mode='lines',
            line=dict(width=2, color=color) if color else dict(width=2),
            hovertemplate=f'{name.replace(".NS", "")}<br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base=100)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_single_cumulative_returns(data: pd.DataFrame, name: str, title: Optional[str] = None) -> go.Figure:
    """Create cumulative returns chart for a single index showing actual % returns."""
    fig = go.Figure()
    
    # Use actual cumulative return percentage
    cumulative_return = data['Cumulative_Return'] * 100  # Convert to percentage
    color = COLORS.get(name)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=cumulative_return,
        name=name.replace('.NS', ''),
        mode='lines',
        line=dict(width=2, color=color) if color else dict(width=2),
        fill='tozeroy',
        fillcolor='rgba(66, 133, 244, 0.1)',
        hovertemplate=f'{name.replace(".NS", "")}<br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title or f"{name} - Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_drawdown(data: pd.DataFrame, name: str) -> go.Figure:
    """Create drawdown chart."""
    cumulative = (1 + data['Daily_Return']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=drawdown,
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='red', width=1),
        name='Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{name} - Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_annual_returns(data: pd.DataFrame, name: str) -> go.Figure:
    """Create annual returns bar chart."""
    annual = data['Daily_Return'].resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Extract years from DatetimeIndex safely
    years: List[int] = [int(ts.year) for ts in pd.to_datetime(annual.index)]
    values: List[float] = [float(v) for v in annual.values]
    
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=years,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        hovertemplate='Year: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title=f"{name} - Annual Returns",
        xaxis_title="Year",
        yaxis_title="Return (%)",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_rolling_volatility(data: pd.DataFrame, name: str, window: int = 90) -> go.Figure:
    """Create rolling volatility chart."""
    rolling_vol = data['Daily_Return'].rolling(window=window).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rolling_vol,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color=COLORS.get(name, '#1f77b4'), width=1.5),
        hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
    ))
    
    avg_vol = float(rolling_vol.mean())
    fig.add_hline(y=avg_vol, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_vol:.1f}%")
    
    fig.update_layout(
        title=f"{name} - Rolling {window}-Day Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_risk_return_scatter(metrics_df: pd.DataFrame, highlight_tickers: Optional[List[str]] = None) -> go.Figure:
    """Create risk-return scatter plot."""
    fig = go.Figure()
    
    for idx_name in INDICES.keys():
        idx_df = metrics_df[metrics_df['Index'] == idx_name]
        
        fig.add_trace(go.Scatter(
            x=idx_df['Volatility (%)'],
            y=idx_df['CAGR (%)'],
            mode='markers+text',
            name=idx_name,
            marker=dict(size=12, color=COLORS[idx_name], opacity=0.7),
            text=idx_df['Ticker'].str.replace('.NS', ''),
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'CAGR: %{y:.2f}%<br>' +
                'Volatility: %{x:.2f}%<br>' +
                '<extra></extra>'
            )
        ))
    
    fig.add_hline(y=RISK_FREE_RATE * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"Risk-Free Rate ({RISK_FREE_RATE*100:.0f}%)")
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="CAGR (%)",
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_correlation_matrix(returns_df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    corr_matrix = returns_df.corr()
    
    # Create heatmap with annotations using go.Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=[[f'{val:.3f}' for val in row] for row in corr_matrix.values],
        texttemplate='%{text}',
        textfont=dict(size=12),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Matrix (Daily Returns)",
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_portfolio_allocation(portfolio_df: pd.DataFrame) -> go.Figure:
    """Create portfolio allocation pie chart."""
    idx_dist = portfolio_df['Index'].value_counts()
    
    fig = go.Figure(data=go.Pie(
        labels=idx_dist.index.tolist(),
        values=idx_dist.values.tolist(),
        marker=dict(colors=[COLORS.get(str(idx), '#333') for idx in idx_dist.index]),
        textinfo='label+percent',
        hovertemplate='%{label}<br>Count: %{value}<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio: Index Distribution",
        template='plotly_white'
    )
    
    return fig


def plot_portfolio_risk_return(portfolio_df: pd.DataFrame) -> go.Figure:
    """Create portfolio risk-return scatter."""
    fig = go.Figure()
    
    for _, row in portfolio_df.iterrows():
        ticker_name = str(row['Ticker']).replace('.NS', '')
        index_name = str(row['Index'])
        cagr_val = float(row['CAGR (%)'])
        vol_val = float(row['Volatility (%)'])
        
        fig.add_trace(go.Scatter(
            x=[vol_val],
            y=[cagr_val],
            mode='markers+text',
            marker=dict(size=20, color=COLORS.get(index_name, '#333')),
            text=[ticker_name],
            textposition='top center',
            name=ticker_name,
            showlegend=False,
            hovertemplate=(
                f"<b>{ticker_name}</b><br>" +
                f"CAGR: {cagr_val:.2f}%<br>" +
                f"Volatility: {vol_val:.2f}%<br>" +
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title="Portfolio: Risk-Return Profile",
        xaxis_title="Volatility (%)",
        yaxis_title="CAGR (%)",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_top_performers_bar(
    df: pd.DataFrame, 
    metric: str, 
    title: str, 
    top_n: int = 10
) -> go.Figure:
    """Create horizontal bar chart for top performers."""
    top_df = df.nlargest(top_n, metric)
    
    colors = [COLORS.get(str(idx), '#333333') for idx in top_df['Index']]
    labels = [f"{str(t).replace('.NS', '')} ({i})" for t, i in zip(top_df['Ticker'], top_df['Index'])]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_df[metric].values.tolist(),
        y=labels,
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}<br>Value: %{x:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=metric,
        yaxis=dict(autorange="reversed"),
        template='plotly_white',
        height=400
    )
    
    return fig
