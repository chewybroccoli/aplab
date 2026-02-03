"""
aplab.portfolio.base
====================
포트폴리오 할당 기본 함수
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union, Literal
from ..base.types import Weighting, QuantileType


def assign_portfolio_quantile(
    data: pd.DataFrame,
    sorting_variable: str,
    n_quantiles: int = 10,
    labels: Optional[List] = None
) -> pd.Series:
    """
    단일 변수 기준으로 포트폴리오 분위수 할당
    
    Parameters
    ----------
    data : pd.DataFrame
        종목별 데이터
    sorting_variable : str
        정렬 기준 변수
    n_quantiles : int, default 10
        분위수 개수 (10=decile, 5=quintile, 3=tercile)
    labels : List, optional
        포트폴리오 라벨. None이면 1부터 n_quantiles까지 정수
    
    Returns
    -------
    pd.Series
        포트폴리오 번호
    
    Examples
    --------
    >>> df['portfolio'] = assign_portfolio_quantile(df, 'size', n_quantiles=10)
    """
    if labels is None:
        labels = list(range(1, n_quantiles + 1))
    
    # rank를 사용하여 동점자 처리
    ranks = data[sorting_variable].rank(method='first')
    
    try:
        portfolios = pd.qcut(
            ranks,
            q=n_quantiles,
            labels=labels,
            duplicates='raise'
        )
    except ValueError:
        # 중복값이 많아 분위수를 정확히 나눌 수 없는 경우
        portfolios = pd.qcut(
            ranks,
            q=n_quantiles,
            labels=False,
            duplicates='drop'
        ) + 1
    
    return portfolios


def assign_portfolio_breakpoints(
    data: pd.DataFrame,
    sorting_variable: str,
    breakpoints: Union[List[float], pd.Series],
    labels: Optional[List] = None,
    include_lowest: bool = True
) -> pd.Series:
    """
    명시적인 브레이크포인트를 사용하여 포트폴리오 할당
    
    Parameters
    ----------
    data : pd.DataFrame
        종목별 데이터
    sorting_variable : str
        정렬 기준 변수
    breakpoints : List[float] or pd.Series
        브레이크포인트 값들
    labels : List, optional
        포트폴리오 라벨
    include_lowest : bool, default True
        최솟값 포함 여부
    
    Returns
    -------
    pd.Series
        포트폴리오 번호
    
    Examples
    --------
    >>> # NYSE 브레이크포인트 사용
    >>> nyse_breaks = nyse_data['size'].quantile([0, 0.5, 1])
    >>> df['portfolio'] = assign_portfolio_breakpoints(df, 'size', nyse_breaks)
    """
    breakpoints = list(breakpoints)
    
    # 경계값을 무한대로 확장
    if breakpoints[0] != -np.inf:
        breakpoints[0] = -np.inf
    if breakpoints[-1] != np.inf:
        breakpoints[-1] = np.inf
    
    n_bins = len(breakpoints) - 1
    if labels is None:
        labels = list(range(1, n_bins + 1))
    
    portfolios = pd.cut(
        data[sorting_variable],
        bins=breakpoints,
        labels=labels,
        include_lowest=include_lowest
    )
    
    return portfolios


def assign_portfolio_by_exchange(
    data: pd.DataFrame,
    sorting_variable: str,
    exchange_col: str = 'exchange',
    exchange_filter: str = 'NYSE',
    percentiles: List[float] = [0, 0.5, 1]
) -> pd.Series:
    """
    특정 거래소(예: NYSE) 기준 브레이크포인트로 포트폴리오 할당
    (Fama-French 방법론)
    
    Parameters
    ----------
    data : pd.DataFrame
        종목별 데이터
    sorting_variable : str
        정렬 기준 변수
    exchange_col : str, default 'exchange'
        거래소 컬럼명
    exchange_filter : str, default 'NYSE'
        브레이크포인트 계산에 사용할 거래소
    percentiles : List[float], default [0, 0.5, 1]
        분위수 (예: [0, 0.5, 1]은 중앙값 기준 2그룹)
    
    Returns
    -------
    pd.Series
        포트폴리오 번호
    """
    # 특정 거래소의 브레이크포인트 계산
    breakpoints = (
        data.query(f"{exchange_col} == '{exchange_filter}'")
        .get(sorting_variable)
        .quantile(percentiles, interpolation='linear')
        .values
    )
    
    return assign_portfolio_breakpoints(data, sorting_variable, breakpoints)


def calculate_portfolio_weights(
    data: pd.DataFrame,
    weighting: Union[str, Weighting] = 'vw',
    weight_col: str = 'mktcap_lag'
) -> pd.Series:
    """
    포트폴리오 가중치 계산
    
    Parameters
    ----------
    data : pd.DataFrame
        포트폴리오 내 종목 데이터
    weighting : str or Weighting, default 'vw'
        가중 방식 ('ew' 또는 'vw')
    weight_col : str, default 'mktcap_lag'
        가중치 계산에 사용할 컬럼 (vw인 경우)
    
    Returns
    -------
    pd.Series
        종목별 가중치 (합계=1)
    """
    weighting = str(weighting)
    
    if weighting == 'ew':
        n = len(data)
        return pd.Series(1.0 / n, index=data.index)
    
    elif weighting == 'vw':
        total_weight = data[weight_col].sum()
        if total_weight > 0:
            return data[weight_col] / total_weight
        else:
            return pd.Series(0.0, index=data.index)
    
    else:
        raise ValueError(f"Invalid weighting method: {weighting}. Use 'ew' or 'vw'.")


def calculate_portfolio_return(
    data: pd.DataFrame,
    return_col: str = 'ret_excess',
    weighting: Union[str, Weighting] = 'vw',
    weight_col: str = 'mktcap_lag'
) -> float:
    """
    포트폴리오 수익률 계산
    
    Parameters
    ----------
    data : pd.DataFrame
        포트폴리오 내 종목 데이터 (단일 시점)
    return_col : str, default 'ret_excess'
        수익률 컬럼명
    weighting : str or Weighting, default 'vw'
        가중 방식 ('ew' 또는 'vw')
    weight_col : str, default 'mktcap_lag'
        가중치 계산에 사용할 컬럼
    
    Returns
    -------
    float
        포트폴리오 수익률
    """
    data = data.dropna(subset=[return_col])
    
    if len(data) == 0:
        return np.nan
    
    weighting = str(weighting)
    
    if weighting == 'ew':
        return data[return_col].mean()
    
    elif weighting == 'vw':
        data = data.dropna(subset=[weight_col])
        if data[weight_col].sum() > 0:
            return np.average(data[return_col], weights=data[weight_col])
        else:
            return np.nan
    
    else:
        raise ValueError(f"Invalid weighting method: {weighting}. Use 'ew' or 'vw'.")


def shift_portfolio_assignment(
    df: pd.DataFrame,
    date_col: str = 'datetime',
    ticker_col: str = 'ticker',
    portfolio_col: str = 'portfolio',
    periods: int = 1
) -> pd.DataFrame:
    """
    포트폴리오 할당을 다음 기간으로 이동 (Look-ahead bias 방지)
    
    t월의 정보로 할당된 포트폴리오를 t+1월에 적용
    
    Parameters
    ----------
    df : pd.DataFrame
        포트폴리오 할당 정보가 포함된 데이터
    date_col : str, default 'datetime'
        날짜 컬럼명
    ticker_col : str, default 'ticker'
        종목 컬럼명
    portfolio_col : str, default 'portfolio'
        포트폴리오 컬럼명
    periods : int, default 1
        이동할 기간 수 (월 단위)
    
    Returns
    -------
    pd.DataFrame
        날짜가 이동된 포트폴리오 정보
    
    Examples
    --------
    >>> # t월 정보 → t+1월에 적용
    >>> portfolio_info = shift_portfolio_assignment(df, periods=1)
    >>> # 원본 데이터와 병합하여 t+1월 수익률에 t월 포트폴리오 적용
    >>> df_analysis = df.merge(portfolio_info, on=['datetime', 'ticker'])
    """
    portfolio_info = df[[date_col, ticker_col, portfolio_col]].copy()
    portfolio_info[date_col] = portfolio_info[date_col] + pd.offsets.MonthEnd(periods)
    
    return portfolio_info


def get_portfolio_summary(
    df: pd.DataFrame,
    portfolio_col: str = 'portfolio',
    date_col: str = 'datetime',
    characteristics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    포트폴리오별 특성 요약 통계
    
    Parameters
    ----------
    df : pd.DataFrame
        분석 데이터
    portfolio_col : str, default 'portfolio'
        포트폴리오 컬럼명
    date_col : str, default 'datetime'
        날짜 컬럼명
    characteristics : List[str], optional
        요약할 특성 변수들
    
    Returns
    -------
    pd.DataFrame
        포트폴리오별 평균 특성
    """
    if characteristics is None:
        # 기본 특성 변수들
        characteristics = ['ret_excess', 'mktcap', 'size', 'bm', 'mom']
        characteristics = [c for c in characteristics if c in df.columns]
    
    # 시점별 포트폴리오별 평균 → 전체 기간 평균
    summary = (
        df.groupby([date_col, portfolio_col])[characteristics]
        .mean()
        .groupby(portfolio_col)
        .mean()
    )
    
    # 종목 수
    counts = df.groupby(portfolio_col).size().rename('N (avg)')
    summary = pd.concat([summary, counts], axis=1)
    
    return summary


def validate_portfolio_data(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    date_col: str = 'datetime',
    ticker_col: str = 'ticker',
    return_col: str = 'ret_excess'
) -> bool:
    """
    포트폴리오 분석을 위한 데이터 유효성 검사
    
    Parameters
    ----------
    df : pd.DataFrame
        검사할 데이터프레임
    required_cols : List[str], optional
        필수 컬럼 리스트
    date_col : str, default 'datetime'
        날짜 컬럼명
    ticker_col : str, default 'ticker'
        종목 컬럼명
    return_col : str, default 'ret_excess'
        수익률 컬럼명
    
    Returns
    -------
    bool
        유효성 검사 통과 여부
    
    Raises
    ------
    ValueError
        필수 컬럼이 없거나 데이터에 문제가 있는 경우
    """
    if required_cols is None:
        required_cols = [date_col, ticker_col, return_col]
    
    # 필수 컬럼 존재 확인
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 날짜 컬럼 타입 확인
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Column '{date_col}' must be datetime type")
    
    # 데이터 개수 확인
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # 수익률 이상치 경고
    ret_abs_max = df[return_col].abs().max()
    if ret_abs_max > 1:  # 100% 이상
        print(f"Warning: Maximum absolute return is {ret_abs_max:.2%}. Check for outliers.")
    
    return True
