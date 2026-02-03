"""aplab.utils - 유틸리티 함수"""

from .stats import (
    newey_west_tstat,
    simple_tstat,
    winsorize,
    standardize,
    calculate_sharpe_ratio,
    get_summary_statistics,
    get_correlation_matrix,
    get_summary_and_correlation
)

from .format import (
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

from .preprocessing import (
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

__all__ = [
    # stats
    "newey_west_tstat",
    "simple_tstat",
    "winsorize",
    "standardize",
    "calculate_sharpe_ratio",
    "get_summary_statistics",
    "get_correlation_matrix",
    "get_summary_and_correlation",
    
    # format
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
    
    # preprocessing
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
    "get_month_end_dates"
]
