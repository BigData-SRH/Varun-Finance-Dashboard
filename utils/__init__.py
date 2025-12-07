"""
Indian Market Analysis - Utilities Package
"""

from .config import (
    START_DATE,
    END_DATE,
    RISK_FREE_RATE,
    INDICES,
    GLOBAL_INDICES,
    COLORS,
    STOCK_COLORS,
    CONSTITUENTS,
    EVENTS
)

from .data_fetcher import (
    fetch_index_data,
    fetch_stock_data,
    fetch_stock_info,
    load_all_index_data,
    load_global_index_data,
    load_all_stock_data,
    get_unique_stocks,
    get_stock_index
)

from .metrics import (
    calculate_returns,
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_beta,
    calculate_all_metrics,
    format_large_number,
    format_percentage,
    format_ratio,
    # New technical indicators
    calculate_52_week_high_low,
    calculate_moving_average_position,
    calculate_rsi,
    get_rsi_status,
    calculate_dividend_growth_rate
)

from .quality import (
    calculate_quality_score,
    assess_all_quality,
    select_top_stocks
)

from .charts import (
    plot_cumulative_returns,
    plot_single_cumulative_returns,
    plot_drawdown,
    plot_annual_returns,
    plot_rolling_volatility,
    plot_risk_return_scatter,
    plot_correlation_matrix,
    plot_portfolio_allocation,
    plot_portfolio_risk_return,
    plot_top_performers_bar,
    # Sparklines
    create_sparkline_svg,
    create_sparkline_with_endpoint
)

__all__ = [
    # Config
    'START_DATE', 'END_DATE', 'RISK_FREE_RATE', 'INDICES', 'GLOBAL_INDICES', 'COLORS',
    'STOCK_COLORS', 'CONSTITUENTS', 'EVENTS',
    # Data
    'fetch_index_data', 'fetch_stock_data', 'fetch_stock_info',
    'load_all_index_data', 'load_global_index_data', 'load_all_stock_data', 'get_unique_stocks', 'get_stock_index',
    # Metrics
    'calculate_returns', 'calculate_cagr', 'calculate_volatility',
    'calculate_sharpe_ratio', 'calculate_sortino_ratio', 'calculate_max_drawdown',
    'calculate_calmar_ratio', 'calculate_beta', 'calculate_all_metrics',
    'format_large_number', 'format_percentage', 'format_ratio',
    # New technical indicators
    'calculate_52_week_high_low', 'calculate_moving_average_position',
    'calculate_rsi', 'get_rsi_status', 'calculate_dividend_growth_rate',
    # Quality
    'calculate_quality_score', 'assess_all_quality', 'select_top_stocks',
    # Charts
    'plot_cumulative_returns', 'plot_single_cumulative_returns', 'plot_drawdown', 'plot_annual_returns',
    'plot_rolling_volatility', 'plot_risk_return_scatter', 'plot_correlation_matrix',
    'plot_portfolio_allocation', 'plot_portfolio_risk_return', 'plot_top_performers_bar',
    # Sparklines
    'create_sparkline_svg', 'create_sparkline_with_endpoint'
]
