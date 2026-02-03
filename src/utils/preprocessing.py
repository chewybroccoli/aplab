"""
aplab.utils.preprocessing
=========================
데이터 전처리 유틸리티 함수
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Set, Union


# ============================================================================
# 한국 주식시장 거래시간
# ============================================================================

def get_trading_hours(date: Union[str, pd.Timestamp]) -> Tuple[str, str]:
    """
    한국 주식시장의 특정 날짜 거래시간 반환
    
    한국 주식시장의 정규 거래시간 및 예외일(신정, 수능일) 처리
    
    Parameters
    ----------
    date : str or pd.Timestamp
        날짜
    
    Returns
    -------
    Tuple[str, str]
        (장 시작 시간, 장 종료 시간)
    
    Examples
    --------
    >>> get_trading_hours('2023-01-02')
    ('10:00', '15:30')  # 신정 다음 영업일
    >>> get_trading_hours('2023-06-15')
    ('09:00', '15:30')  # 일반 거래일
    """
    date = pd.to_datetime(date).normalize()
    
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    
    # 신정 다음 영업일 예외 (10:00 개장)
    NEW_YEARS_EXCEPTION1 = pd.to_datetime([
        '2010-01-04', '2011-01-03', '2012-01-02', '2013-01-02',
        '2014-01-02', '2015-01-02', '2016-01-04'
    ]).normalize()
    
    NEW_YEARS_EXCEPTION2 = pd.to_datetime([
        '2017-01-02', '2018-01-02', '2019-01-02', '2020-01-02', '2021-01-04',
        '2022-01-03', '2023-01-02', '2024-01-02', '2025-01-02', '2026-01-02'
    ]).normalize()
    
    # 수능일 예외 (10:00 개장)
    CSAT_EXCEPTIONS1 = pd.to_datetime([
        '2010-11-18', '2011-11-10', '2012-11-08', '2013-11-07',
        '2014-11-13', '2015-11-12'
    ]).normalize()
    
    CSAT_EXCEPTIONS2 = pd.to_datetime([
        '2016-11-17', '2017-11-23', '2017-11-16',
        '2018-11-15', '2019-11-14', '2020-12-03', '2021-11-18',
        '2022-11-17', '2023-11-16', '2024-11-14', '2025-11-13'
    ]).normalize()
    
    if (NEW_YEARS_EXCEPTION1 == date).any():
        return "10:00", "15:00"
    elif (NEW_YEARS_EXCEPTION2 == date).any():
        return "10:00", "15:30"
    elif (CSAT_EXCEPTIONS1 == date).any():
        return "10:00", "16:00"
    elif (CSAT_EXCEPTIONS2 == date).any():
        return "10:00", "16:30"
    elif date < pd.Timestamp("2016-08-01"):
        return "09:00", "15:00"
    else:
        return "09:00", "15:30"


def get_market_open_time(date: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """장 시작 시간을 Timestamp로 반환"""
    date = pd.to_datetime(date)
    start_time, _ = get_trading_hours(date)
    return pd.Timestamp(date.strftime('%Y-%m-%d') + ' ' + start_time)


def get_market_close_time(date: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """장 종료 시간을 Timestamp로 반환"""
    date = pd.to_datetime(date)
    _, end_time = get_trading_hours(date)
    return pd.Timestamp(date.strftime('%Y-%m-%d') + ' ' + end_time)


# ============================================================================
# 종목 필터링
# ============================================================================

def filter_financial_stocks(
    df: pd.DataFrame,
    ticker_col: str = 'ticker',
    industry_col: Optional[str] = 'industry',
    financial_keywords: List[str] = ['은행', '증권', '보험', '금융', 'Bank', 'Securities', 'Insurance'],
    financial_tickers: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    금융주 제외
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    industry_col : str, optional
        업종 컬럼명
    financial_keywords : List[str]
        금융 업종 키워드
    financial_tickers : Set[str], optional
        직접 제공하는 금융주 종목코드 집합
    
    Returns
    -------
    pd.DataFrame
        금융주가 제외된 데이터
    """
    df = df.copy()
    
    if financial_tickers is not None:
        # 직접 제공된 금융주 목록 사용
        mask = ~df[ticker_col].isin(financial_tickers)
    elif industry_col and industry_col in df.columns:
        # 업종 컬럼으로 필터링
        pattern = '|'.join(financial_keywords)
        mask = ~df[industry_col].str.contains(pattern, na=False, case=False)
    else:
        # 필터링 불가
        return df
    
    filtered_count = (~mask).sum()
    if filtered_count > 0:
        print(f"Financial stocks filtered: {filtered_count} observations")
    
    return df[mask].copy()


def filter_by_price(
    df: pd.DataFrame,
    price_col: str = 'price',
    min_price: float = 0,
    max_price: Optional[float] = None
) -> pd.DataFrame:
    """
    가격 기준 필터링
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    price_col : str, default 'price'
        가격 컬럼명
    min_price : float, default 0
        최소 가격 (동전주 제외)
    max_price : float, optional
        최대 가격
    
    Returns
    -------
    pd.DataFrame
        필터링된 데이터
    """
    df = df.copy()
    mask = df[price_col] >= min_price
    
    if max_price is not None:
        mask = mask & (df[price_col] <= max_price)
    
    return df[mask].copy()


def filter_extreme_returns(
    df: pd.DataFrame,
    return_col: str = 'ret',
    lower_bound: float = -0.99,
    upper_bound: float = 10.0
) -> pd.DataFrame:
    """
    극단적인 수익률 필터링
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    return_col : str, default 'ret'
        수익률 컬럼명
    lower_bound : float, default -0.99
        하한 (-99%)
    upper_bound : float, default 10.0
        상한 (1000%)
    
    Returns
    -------
    pd.DataFrame
        필터링된 데이터
    """
    df = df.copy()
    mask = (df[return_col] >= lower_bound) & (df[return_col] <= upper_bound)
    return df[mask].copy()


def filter_by_listing_period(
    df: pd.DataFrame,
    date_col: str = 'datetime',
    ticker_col: str = 'ticker',
    min_months: int = 12
) -> pd.DataFrame:
    """
    최소 상장 기간 필터링
    
    IPO 직후 종목 제외 (최소 N개월 상장 데이터 필요)
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    date_col : str, default 'datetime'
        날짜 컬럼명
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    min_months : int, default 12
        최소 상장 기간 (월)
    
    Returns
    -------
    pd.DataFrame
        필터링된 데이터
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 각 종목의 첫 번째 관측일
    first_date = df.groupby(ticker_col)[date_col].transform('min')
    
    # 최소 기간 경과 여부
    min_date_threshold = first_date + pd.DateOffset(months=min_months)
    mask = df[date_col] >= min_date_threshold
    
    return df[mask].copy()


# ============================================================================
# 수익률 계산
# ============================================================================

def calculate_returns(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    ticker_col: str = 'ticker',
    date_col: str = 'datetime',
    return_col: str = 'ret'
) -> pd.DataFrame:
    """
    단순 수익률 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        가격 데이터
    price_col : str, default 'adj_close'
        가격 컬럼명
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    date_col : str, default 'datetime'
        날짜 컬럼명
    return_col : str, default 'ret'
        생성할 수익률 컬럼명
    
    Returns
    -------
    pd.DataFrame
        수익률이 추가된 데이터
    """
    df = df.copy()
    df = df.sort_values([ticker_col, date_col])
    df[return_col] = df.groupby(ticker_col)[price_col].pct_change()
    return df


def calculate_log_returns(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    ticker_col: str = 'ticker',
    date_col: str = 'datetime',
    return_col: str = 'log_ret'
) -> pd.DataFrame:
    """
    로그 수익률 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        가격 데이터
    price_col : str, default 'adj_close'
        가격 컬럼명
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    date_col : str, default 'datetime'
        날짜 컬럼명
    return_col : str, default 'log_ret'
        생성할 로그수익률 컬럼명
    
    Returns
    -------
    pd.DataFrame
        로그 수익률이 추가된 데이터
    """
    df = df.copy()
    df = df.sort_values([ticker_col, date_col])
    df[return_col] = df.groupby(ticker_col)[price_col].transform(lambda x: np.log(x / x.shift(1)))
    return df


def calculate_excess_returns(
    df: pd.DataFrame,
    return_col: str = 'ret',
    rf_col: str = 'rf',
    excess_return_col: str = 'ret_excess'
) -> pd.DataFrame:
    """
    초과수익률 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        수익률 데이터
    return_col : str, default 'ret'
        수익률 컬럼명
    rf_col : str, default 'rf'
        무위험 수익률 컬럼명
    excess_return_col : str, default 'ret_excess'
        초과수익률 컬럼명
    
    Returns
    -------
    pd.DataFrame
        초과수익률이 추가된 데이터
    """
    df = df.copy()
    df[excess_return_col] = df[return_col] - df[rf_col]
    return df


# ============================================================================
# 래그 변수 생성
# ============================================================================

def lag_variable(
    df: pd.DataFrame,
    variable: str,
    ticker_col: str = 'ticker',
    date_col: str = 'datetime',
    lag: int = 1,
    suffix: str = '_lag'
) -> pd.DataFrame:
    """
    래그 변수 생성
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    variable : str
        래그할 변수명
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    date_col : str, default 'datetime'
        날짜 컬럼명
    lag : int, default 1
        래그 기간
    suffix : str, default '_lag'
        새 컬럼 접미사
    
    Returns
    -------
    pd.DataFrame
        래그 변수가 추가된 데이터
    """
    df = df.copy()
    df = df.sort_values([ticker_col, date_col])
    new_col = f"{variable}{suffix}" if lag == 1 else f"{variable}_lag{lag}"
    df[new_col] = df.groupby(ticker_col)[variable].shift(lag)
    return df


def lead_variable(
    df: pd.DataFrame,
    variable: str,
    ticker_col: str = 'ticker',
    date_col: str = 'datetime',
    lead: int = 1,
    suffix: str = '_lead'
) -> pd.DataFrame:
    """
    리드 변수 생성 (다음 기간 변수)
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    variable : str
        리드할 변수명
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    date_col : str, default 'datetime'
        날짜 컬럼명
    lead : int, default 1
        리드 기간
    suffix : str, default '_lead'
        새 컬럼 접미사
    
    Returns
    -------
    pd.DataFrame
        리드 변수가 추가된 데이터
    """
    df = df.copy()
    df = df.sort_values([ticker_col, date_col])
    new_col = f"{variable}{suffix}" if lead == 1 else f"{variable}_lead{lead}"
    df[new_col] = df.groupby(ticker_col)[variable].shift(-lead)
    return df


# ============================================================================
# 거래 관련 지표
# ============================================================================

def calculate_oib(
    df: pd.DataFrame,
    buy_col: str = 'buy_volume',
    sell_col: str = 'sell_volume',
    oib_col: str = 'oib'
) -> pd.DataFrame:
    """
    Order Imbalance (OIB) 계산
    
    OIB = (Buy Volume - Sell Volume) / (Buy Volume + Sell Volume)
    
    Parameters
    ----------
    df : pd.DataFrame
        거래 데이터
    buy_col : str, default 'buy_volume'
        매수 거래량 컬럼명
    sell_col : str, default 'sell_volume'
        매도 거래량 컬럼명
    oib_col : str, default 'oib'
        생성할 OIB 컬럼명
    
    Returns
    -------
    pd.DataFrame
        OIB가 추가된 데이터
    """
    df = df.copy()
    total_volume = df[buy_col] + df[sell_col]
    df[oib_col] = (df[buy_col] - df[sell_col]) / total_volume
    df[oib_col] = df[oib_col].replace([np.inf, -np.inf], np.nan)
    return df


def calculate_abnormal_variable(
    df: pd.DataFrame,
    variable: str,
    ticker_col: str = 'ticker',
    date_col: str = 'datetime',
    window: int = 252,
    min_periods: int = 126,
    method: str = 'ratio'
) -> pd.DataFrame:
    """
    비정상(Abnormal) 변수 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    variable : str
        계산할 변수명
    ticker_col : str, default 'ticker'
        종목코드 컬럼명
    date_col : str, default 'datetime'
        날짜 컬럼명
    window : int, default 252
        롤링 윈도우 (거래일)
    min_periods : int, default 126
        최소 관측치 수
    method : str, default 'ratio'
        계산 방법 ('ratio' 또는 'diff')
    
    Returns
    -------
    pd.DataFrame
        비정상 변수가 추가된 데이터
    """
    df = df.copy()
    df = df.sort_values([ticker_col, date_col])
    
    new_col = f"ab_{variable}"
    
    def calc_abnormal(group):
        rolling_mean = group[variable].rolling(window=window, min_periods=min_periods).mean().shift(1)
        if method == 'ratio':
            return group[variable] / rolling_mean
        else:  # diff
            return group[variable] - rolling_mean
    
    df[new_col] = df.groupby(ticker_col, group_keys=False).apply(calc_abnormal)
    df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
    
    return df


# ============================================================================
# 날짜/기간 관련
# ============================================================================

def add_year_month(
    df: pd.DataFrame,
    date_col: str = 'datetime',
    year_month_col: str = 'year_month'
) -> pd.DataFrame:
    """
    연월 컬럼 추가
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    date_col : str, default 'datetime'
        날짜 컬럼명
    year_month_col : str, default 'year_month'
        생성할 연월 컬럼명
    
    Returns
    -------
    pd.DataFrame
        연월 컬럼이 추가된 데이터
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[year_month_col] = df[date_col].dt.to_period('M')
    return df


def add_fiscal_year(
    df: pd.DataFrame,
    date_col: str = 'datetime',
    fiscal_year_col: str = 'fiscal_year',
    fiscal_year_end_month: int = 6
) -> pd.DataFrame:
    """
    회계연도 컬럼 추가 (Fama-French 방식)
    
    6월 말 기준으로 회계연도 구분 (7월~다음해 6월)
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    date_col : str, default 'datetime'
        날짜 컬럼명
    fiscal_year_col : str, default 'fiscal_year'
        생성할 회계연도 컬럼명
    fiscal_year_end_month : int, default 6
        회계연도 종료 월
    
    Returns
    -------
    pd.DataFrame
        회계연도 컬럼이 추가된 데이터
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    year = df[date_col].dt.year
    month = df[date_col].dt.month
    
    # 회계연도: 7월 이전이면 전년도, 7월 이후면 당해연도
    df[fiscal_year_col] = np.where(month <= fiscal_year_end_month, year - 1, year)
    
    return df


def get_month_end_dates(
    start_date: str,
    end_date: str,
    freq: str = 'M'
) -> pd.DatetimeIndex:
    """
    기간 내 월말 날짜 리스트 반환
    
    Parameters
    ----------
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    freq : str, default 'M'
        빈도 ('M': 월말, 'Q': 분기말)
    
    Returns
    -------
    pd.DatetimeIndex
        월말 날짜 리스트
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)
