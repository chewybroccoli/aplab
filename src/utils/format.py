"""
aplab.utils.format
==================
출력 포맷팅 유틸리티 함수
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
from pathlib import Path


def add_significance_stars(
    p_value: float,
    levels: tuple = (0.01, 0.05, 0.10)
) -> str:
    """
    p-value에 따른 유의성 별표 반환
    
    Parameters
    ----------
    p_value : float
        p-value
    levels : tuple, default (0.01, 0.05, 0.10)
        유의수준 (1%, 5%, 10%)
    
    Returns
    -------
    str
        유의성 별표 ('***', '**', '*', '')
    """
    if pd.isna(p_value):
        return ''
    
    if p_value < levels[0]:
        return '***'
    elif p_value < levels[1]:
        return '**'
    elif p_value < levels[2]:
        return '*'
    return ''


def format_coefficient(
    coef: float,
    t_stat: Optional[float] = None,
    p_value: Optional[float] = None,
    decimal: int = 2,
    include_stars: bool = True,
    include_tstat: bool = True
) -> str:
    """
    회귀 계수를 포맷팅하여 문자열로 반환
    
    Parameters
    ----------
    coef : float
        회귀 계수
    t_stat : float, optional
        t-통계량
    p_value : float, optional
        p-value
    decimal : int, default 2
        소수점 자릿수
    include_stars : bool, default True
        유의성 별표 포함 여부
    include_tstat : bool, default True
        t-통계량 포함 여부
    
    Returns
    -------
    str
        포맷팅된 문자열
    """
    if pd.isna(coef):
        return ''
    
    coef_str = f"{coef:.{decimal}f}"
    
    if include_stars and p_value is not None:
        coef_str += add_significance_stars(p_value)
    
    if include_tstat and t_stat is not None:
        coef_str += f"\n({t_stat:.{decimal}f})"
    
    return coef_str


def format_number(
    value: float,
    decimal: int = 2,
    percentage: bool = False,
    thousands_sep: bool = False
) -> str:
    """
    숫자를 포맷팅하여 문자열로 반환
    
    Parameters
    ----------
    value : float
        숫자 값
    decimal : int, default 2
        소수점 자릿수
    percentage : bool, default False
        퍼센트 형식으로 표시할지 여부
    thousands_sep : bool, default False
        천 단위 구분자 사용 여부
    
    Returns
    -------
    str
        포맷팅된 문자열
    """
    if pd.isna(value):
        return ''
    
    if thousands_sep:
        format_str = f"{{:,.{decimal}f}}"
    else:
        format_str = f"{{:.{decimal}f}}"
    
    if percentage:
        value = value * 100
        return format_str.format(value) + "%"
    
    return format_str.format(value)


def format_tstat(t_stat: float, decimal: int = 2) -> str:
    """
    t-통계량을 괄호로 감싸서 포맷팅
    
    Parameters
    ----------
    t_stat : float
        t-통계량
    decimal : int, default 2
        소수점 자릿수
    
    Returns
    -------
    str
        포맷팅된 문자열 (예: "(2.35)")
    """
    if pd.isna(t_stat):
        return ''
    return f"({t_stat:.{decimal}f})"


def create_regression_table(
    results: List[Dict[str, Any]],
    decimal: int = 2,
    include_stars: bool = True
) -> pd.DataFrame:
    """
    회귀분석 결과를 테이블 형태로 포맷팅
    
    Parameters
    ----------
    results : List[Dict]
        회귀분석 결과 딕셔너리 리스트
        각 딕셔너리는 'Variable', 'Coefficient', 't-stat', 'p-value' 키를 포함
    decimal : int, default 2
        소수점 자릿수
    include_stars : bool, default True
        유의성 별표 포함 여부
    
    Returns
    -------
    pd.DataFrame
        포맷팅된 테이블
    """
    formatted_results = []
    
    for res in results:
        variable = res.get('Variable', '')
        coef = res.get('Coefficient', np.nan)
        t_stat = res.get('t-stat', res.get('Newey-West t-stat', np.nan))
        p_value = res.get('p-value', np.nan)
        
        stars = add_significance_stars(p_value) if include_stars else ''
        
        formatted_results.append({
            'Variable': variable,
            'Coefficient': f"{coef:.{decimal}f}{stars}" if not pd.isna(coef) else '',
            't-stat': format_tstat(t_stat, decimal)
        })
    
    return pd.DataFrame(formatted_results)


def create_portfolio_table(
    results: pd.DataFrame,
    return_cols: List[str] = ['Mean Return (%)', 'Alpha (%)'],
    tstat_cols: List[str] = ['Mean Return t-stat', 'Alpha t-stat'],
    decimal: int = 2
) -> pd.DataFrame:
    """
    포트폴리오 분석 결과를 테이블 형태로 포맷팅
    
    Parameters
    ----------
    results : pd.DataFrame
        포트폴리오 분석 결과
    return_cols : List[str]
        수익률 관련 컬럼명
    tstat_cols : List[str]
        t-통계량 관련 컬럼명
    decimal : int, default 2
        소수점 자릿수
    
    Returns
    -------
    pd.DataFrame
        포맷팅된 테이블
    """
    df = results.copy()
    
    for col in return_cols + tstat_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{x:.{decimal}f}" if not pd.isna(x) else ''
            )
    
    return df


def save_to_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    decimal: int = 2,
    index: bool = False,
    encoding: str = 'utf-8-sig'
) -> None:
    """
    DataFrame을 CSV 파일로 저장 (한글 호환)
    
    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터프레임
    filepath : str or Path
        저장 경로
    decimal : int, default 2
        소수점 자릿수
    index : bool, default False
        인덱스 포함 여부
    encoding : str, default 'utf-8-sig'
        인코딩 (엑셀 한글 호환)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=index, encoding=encoding, float_format=f'%.{decimal}f')
    print(f"Results saved to: {filepath}")


def save_to_excel(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    sheet_name: str = 'Sheet1',
    index: bool = False
) -> None:
    """
    DataFrame을 Excel 파일로 저장
    
    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터프레임
    filepath : str or Path
        저장 경로
    sheet_name : str, default 'Sheet1'
        시트 이름
    index : bool, default False
        인덱스 포함 여부
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_excel(filepath, sheet_name=sheet_name, index=index)
    print(f"Results saved to: {filepath}")


def print_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    decimal: int = 4,
    max_rows: Optional[int] = None
) -> None:
    """
    DataFrame을 보기 좋게 출력
    
    Parameters
    ----------
    df : pd.DataFrame
        출력할 데이터프레임
    title : str, optional
        테이블 제목
    decimal : int, default 4
        소수점 자릿수
    max_rows : int, optional
        최대 출력 행 수
    """
    if title:
        print(f"\n{'='*60}")
        print(f"### {title} ###")
        print('='*60)
    
    if max_rows:
        print(df.head(max_rows).to_string(float_format=f"%.{decimal}f"))
        if len(df) > max_rows:
            print(f"... ({len(df) - max_rows} more rows)")
    else:
        print(df.to_string(float_format=f"%.{decimal}f"))
    
    print()


class TableFormatter:
    """
    테이블 포맷팅을 위한 클래스
    
    Examples
    --------
    >>> formatter = TableFormatter(decimal=2, include_stars=True)
    >>> formatted_df = formatter.format_regression_results(results_df)
    >>> formatter.save(formatted_df, 'output.csv')
    """
    
    def __init__(
        self,
        decimal: int = 2,
        include_stars: bool = True,
        encoding: str = 'utf-8-sig'
    ):
        """
        Parameters
        ----------
        decimal : int, default 2
            소수점 자릿수
        include_stars : bool, default True
            유의성 별표 포함 여부
        encoding : str, default 'utf-8-sig'
            파일 저장 시 인코딩
        """
        self.decimal = decimal
        self.include_stars = include_stars
        self.encoding = encoding
    
    def format_value(self, value: float) -> str:
        """숫자 값 포맷팅"""
        return format_number(value, self.decimal)
    
    def format_with_tstat(
        self, 
        coef: float, 
        t_stat: float, 
        p_value: Optional[float] = None
    ) -> str:
        """계수와 t-통계량 함께 포맷팅"""
        return format_coefficient(
            coef, t_stat, p_value, 
            self.decimal, self.include_stars, True
        )
    
    def save(
        self, 
        df: pd.DataFrame, 
        filepath: Union[str, Path],
        format: str = 'csv'
    ) -> None:
        """테이블 저장"""
        if format.lower() == 'csv':
            save_to_csv(df, filepath, self.decimal, encoding=self.encoding)
        elif format.lower() == 'excel':
            save_to_excel(df, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
