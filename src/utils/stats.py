"""
aplab.utils.stats
=================
통계 관련 유틸리티 함수
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Tuple, List, Dict, Union
from scipy import stats


def newey_west_tstat(
    series: pd.Series, 
    maxlags: int = 12
) -> Tuple[float, float, float]:
    """
    시계열의 평균에 대한 Newey-West t-통계량 계산
    
    Parameters
    ----------
    series : pd.Series
        시계열 데이터
    maxlags : int, default 12
        HAC 추정에 사용할 최대 래그 수
    
    Returns
    -------
    Tuple[float, float, float]
        (평균, t-통계량, p-value)
    """
    series = series.dropna()
    if len(series) < 2:
        return np.nan, np.nan, np.nan
    
    y = series.values
    X = sm.add_constant(np.ones(len(y)))
    
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    
    mean_val = series.mean()
    t_stat = model.tvalues[0]
    p_value = model.pvalues[0]
    
    return mean_val, t_stat, p_value


def simple_tstat(series: pd.Series) -> Tuple[float, float, float]:
    """
    시계열의 평균에 대한 단순 t-통계량 계산
    
    Parameters
    ----------
    series : pd.Series
        시계열 데이터
    
    Returns
    -------
    Tuple[float, float, float]
        (평균, t-통계량, p-value)
    """
    series = series.dropna()
    if len(series) < 2:
        return np.nan, np.nan, np.nan
    
    mean_val = series.mean()
    std_val = series.std()
    n = len(series)
    
    if std_val == 0:
        return mean_val, np.nan, np.nan
    
    t_stat = mean_val / (std_val / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return mean_val, t_stat, p_value


def winsorize(
    df: pd.DataFrame, 
    columns: List[str], 
    limits: Tuple[float, float] = (0.01, 0.99),
    by_group: Optional[str] = None
) -> pd.DataFrame:
    """
    데이터프레임의 특정 컬럼들을 윈저라이즈
    
    Parameters
    ----------
    df : pd.DataFrame
        대상 데이터프레임
    columns : List[str]
        윈저라이즈할 컬럼 리스트
    limits : Tuple[float, float], default (0.01, 0.99)
        하위/상위 분위수 (예: 1%, 99%)
    by_group : str, optional
        그룹별로 윈저라이즈할 경우 그룹 컬럼명
    
    Returns
    -------
    pd.DataFrame
        윈저라이즈된 데이터프레임
    """
    df = df.copy()
    
    def _winsorize_series(s: pd.Series) -> pd.Series:
        lower = s.quantile(limits[0])
        upper = s.quantile(limits[1])
        return s.clip(lower=lower, upper=upper)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if by_group and by_group in df.columns:
            df[col] = df.groupby(by_group, group_keys=False)[col].transform(_winsorize_series)
        else:
            df[col] = _winsorize_series(df[col])
    
    return df


def standardize(
    df: pd.DataFrame,
    columns: List[str],
    by_group: Optional[str] = None
) -> pd.DataFrame:
    """
    데이터프레임의 특정 컬럼들을 표준화 (Z-score)
    
    Parameters
    ----------
    df : pd.DataFrame
        대상 데이터프레임
    columns : List[str]
        표준화할 컬럼 리스트
    by_group : str, optional
        그룹별로 표준화할 경우 그룹 컬럼명 (예: 월별)
    
    Returns
    -------
    pd.DataFrame
        표준화된 데이터프레임
    """
    df = df.copy()
    
    def _standardize_series(s: pd.Series) -> pd.Series:
        mean_val = s.mean()
        std_val = s.std()
        if std_val == 0:
            return s - mean_val
        return (s - mean_val) / std_val
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if by_group and by_group in df.columns:
            df[col] = df.groupby(by_group, group_keys=False)[col].transform(_standardize_series)
        else:
            df[col] = _standardize_series(df[col])
    
    return df


def calculate_sharpe_ratio(
    returns: pd.Series,
    rf: Optional[pd.Series] = None,
    annualize: bool = True,
    periods_per_year: int = 12
) -> float:
    """
    샤프 비율 계산
    
    Parameters
    ----------
    returns : pd.Series
        수익률 시계열
    rf : pd.Series, optional
        무위험 수익률. None이면 0으로 가정
    annualize : bool, default True
        연율화 여부
    periods_per_year : int, default 12
        연간 기간 수 (월별=12, 일별=252)
    
    Returns
    -------
    float
        샤프 비율
    """
    returns = returns.dropna()
    
    if rf is not None:
        excess_returns = returns - rf.reindex(returns.index)
    else:
        excess_returns = returns
    
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0:
        return np.nan
    
    sharpe = mean_return / std_return
    
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    
    return sharpe


def get_summary_statistics(
    df: pd.DataFrame,
    variables: List[str],
    percentiles: List[float] = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
    include_skew_kurt: bool = True
) -> pd.DataFrame:
    """
    변수들의 기술 통계량 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        데이터프레임
    variables : List[str]
        분석할 변수 리스트
    percentiles : List[float]
        계산할 분위수
    include_skew_kurt : bool, default True
        왜도/첨도 포함 여부
    
    Returns
    -------
    pd.DataFrame
        기술 통계량 테이블
    """
    df_analysis = df[variables].copy().dropna()
    
    desc_stats = df_analysis.describe(percentiles=percentiles).transpose()
    
    if include_skew_kurt:
        desc_stats['skew'] = df_analysis.skew()
        desc_stats['kurt'] = df_analysis.kurt()
    
    return desc_stats


def get_correlation_matrix(
    df: pd.DataFrame,
    variables: List[str],
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    변수들의 상관관계 행렬 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        데이터프레임
    variables : List[str]
        분석할 변수 리스트
    method : str, default 'pearson'
        상관계수 방법 ('pearson', 'spearman', 'kendall')
    
    Returns
    -------
    pd.DataFrame
        상관관계 행렬
    """
    df_analysis = df[variables].copy().dropna()
    return df_analysis.corr(method=method)


def get_summary_and_correlation(
    df: pd.DataFrame,
    variables: List[str],
    include_skew_kurt: bool = True,
    correlation_method: str = 'pearson',
    print_result: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    기술 통계량과 상관관계 행렬을 함께 계산
    
    Parameters
    ----------
    df : pd.DataFrame
        데이터프레임
    variables : List[str]
        분석할 변수 리스트
    include_skew_kurt : bool, default True
        왜도/첨도 포함 여부
    correlation_method : str, default 'pearson'
        상관계수 방법
    print_result : bool, default True
        결과 출력 여부
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (기술 통계량, 상관관계 행렬)
    """
    summary = get_summary_statistics(df, variables, include_skew_kurt=include_skew_kurt)
    correlation = get_correlation_matrix(df, variables, method=correlation_method)
    
    if print_result:
        print("### Summary Statistics ###")
        print("\n--- Panel A: Descriptive Statistics ---")
        print(summary.to_string(float_format="%.4f"))
        print("\n--- Panel B: Correlation Matrix ---")
        print(correlation.to_string(float_format="%.3f"))
    
    return summary, correlation
