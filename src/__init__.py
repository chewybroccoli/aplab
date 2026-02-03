"""
aplab - Asset Pricing Laboratory
================================

Asset Pricing 연구를 위한 Python 라이브러리

주요 기능:
- Portfolio Sorting (Univariate, Bivariate, Triple)
- Fama-MacBeth Regression
- Panel Regression (Fixed Effects, Random Effects)
- Factor Model Regression (CAPM, FF3, Carhart, FF5)
- Factor Construction (SMB, HML, MOM 등)
- Summary Statistics & Output Formatting

Examples
--------
>>> import aplab
>>> 
>>> # Univariate Portfolio Sort
>>> results = aplab.univariate_sort(
...     df=data,
...     factors_df=factors,
...     sorting_variable='size',
...     n_quantiles=10,
...     weighting='vw'
... )
>>> 
>>> # Fama-MacBeth Regression
>>> fmb_results = aplab.fama_macbeth_regression(
...     df=data,
...     dependent_var='ret_excess_lead',
...     independent_vars=['size', 'bm', 'mom']
... )
>>> 
>>> # Panel Regression
>>> panel_results = aplab.panel_regression(
...     df=data,
...     dependent_var='ret_excess_lead',
...     independent_vars=['size', 'bm', 'mom'],
...     time_effects=True
... )
"""

__version__ = "0.1.0"
__author__ = "aplab contributors"

# Base types
from .base.types import (
    Weighting,
    SortingMethod,
    QuantileType,
    RegressionResult,
    PortfolioPerformance,
    ColumnNames
)

# Portfolio functions
from .portfolio.sorting import (
    PortfolioSorter,
    univariate_sort,
    bivariate_sort
)

from .portfolio.base import (
    assign_portfolio_quantile,
    assign_portfolio_breakpoints,
    calculate_portfolio_weights,
    calculate_portfolio_return,
    shift_portfolio_assignment,
    get_portfolio_summary
)

from .portfolio.performance import (
    PerformanceAnalyzer,
    factor_regression,
    calculate_alpha
)

# Regression functions
from .regression.fmr import (
    FamaMacBethRegression,
    fama_macbeth_regression,
    run_multiple_fmb_specifications
)

from .regression.panel import (
    PanelRegression,
    panel_regression,
    run_multiple_panel_specifications,
    simple_panel_ols
)

# Factor functions
from .factors.builder import (
    FactorBuilder,
    calculate_factor_returns,
    calculate_factor_premium,
    factor_correlation_matrix,
    factor_summary_statistics
)

# Utility functions
from .utils.stats import (
    newey_west_tstat,
    simple_tstat,
    winsorize,
    standardize,
    calculate_sharpe_ratio,
    get_summary_statistics,
    get_correlation_matrix,
    get_summary_and_correlation
)

from .utils.format import (
    add_significance_stars,
    format_coefficient,
    format_number,
    format_tstat,
    create_regression_table,
    create_portfolio_table,
    save_to_csv,
    save_to_excel,
    print_table,
    TableFormatter
)

from .utils.preprocessing import (
    # 거래시간
    get_trading_hours,
    get_market_open_time,
    get_market_close_time,
    # 필터링
    filter_financial_stocks,
    filter_by_price,
    filter_extreme_returns,
    filter_by_listing_period,
    # 수익률 계산
    calculate_returns,
    calculate_log_returns,
    calculate_excess_returns,
    # 래그/리드 변수
    lag_variable,
    lead_variable,
    # 거래 지표
    calculate_oib,
    calculate_abnormal_variable,
    # 날짜 관련
    add_year_month,
    add_fiscal_year,
    get_month_end_dates
)

# Public API
__all__ = [
    # Version
    "__version__",
    
    # Types
    "Weighting",
    "SortingMethod",
    "QuantileType",
    "RegressionResult",
    "PortfolioPerformance",
    "ColumnNames",
    
    # Portfolio
    "PortfolioSorter",
    "univariate_sort",
    "bivariate_sort",
    "assign_portfolio_quantile",
    "assign_portfolio_breakpoints",
    "calculate_portfolio_weights",
    "calculate_portfolio_return",
    "shift_portfolio_assignment",
    "get_portfolio_summary",
    
    # Performance
    "PerformanceAnalyzer",
    "factor_regression",
    "calculate_alpha",
    
    # Regression
    "FamaMacBethRegression",
    "fama_macbeth_regression",
    "run_multiple_fmb_specifications",
    "PanelRegression",
    "panel_regression",
    "run_multiple_panel_specifications",
    "simple_panel_ols",
    
    # Factors
    "FactorBuilder",
    "calculate_factor_returns",
    "calculate_factor_premium",
    "factor_correlation_matrix",
    "factor_summary_statistics",
    
    # Statistics
    "newey_west_tstat",
    "simple_tstat",
    "winsorize",
    "standardize",
    "calculate_sharpe_ratio",
    "get_summary_statistics",
    "get_correlation_matrix",
    "get_summary_and_correlation",
    
    # Formatting
    "add_significance_stars",
    "format_coefficient",
    "format_number",
    "format_tstat",
    "create_regression_table",
    "create_portfolio_table",
    "save_to_csv",
    "save_to_excel",
    "print_table",
    "TableFormatter",
    
    # Preprocessing
    "get_trading_hours",
    "get_market_open_time",
    "get_market_close_time",
    "filter_financial_stocks",
    "filter_by_price",
    "filter_extreme_returns",
    "filter_by_listing_period",
    "calculate_returns",
    "calculate_log_returns",
    "calculate_excess_returns",
    "lag_variable",
    "lead_variable",
    "calculate_oib",
    "calculate_abnormal_variable",
    "add_year_month",
    "add_fiscal_year",
    "get_month_end_dates",
]
