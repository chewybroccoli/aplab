"""
aplab.regression.fmr
====================
Fama-MacBeth Regression
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth 회귀분석 결과"""
    variable: str
    coefficient: float
    t_stat: float
    p_value: float
    std_error: float
    n_periods: int
    
    @property
    def significance(self) -> str:
        """유의성 별표"""
        if self.p_value < 0.01:
            return "***"
        elif self.p_value < 0.05:
            return "**"
        elif self.p_value < 0.10:
            return "*"
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Variable': self.variable,
            'Coefficient': self.coefficient,
            'Newey-West t-stat': self.t_stat,
            'p-value': self.p_value,
            'Std Error': self.std_error,
            'N (periods)': self.n_periods
        }


class FamaMacBethRegression:
    """
    Fama-MacBeth (1973) 2단계 회귀분석
    
    1단계: 매 기간 횡단면 회귀분석
    2단계: 시계열 평균 및 Newey-West t-통계량 계산
    
    Examples
    --------
    >>> fmb = FamaMacBethRegression(
    ...     df=data,
    ...     dependent_var='ret_excess_lead',
    ...     independent_vars=['size', 'bm', 'mom'],
    ...     date_col='datetime'
    ... )
    >>> results = fmb.fit()
    >>> print(results)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        date_col: str = 'datetime',
        add_constant: bool = True
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            패널 데이터
        dependent_var : str
            종속변수명 (일반적으로 다음 기간 초과수익률)
        independent_vars : List[str]
            독립변수 리스트
        date_col : str, default 'datetime'
            날짜 컬럼명
        add_constant : bool, default True
            상수항 포함 여부
        """
        self.df = df.copy()
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.date_col = date_col
        self.add_constant = add_constant
        
        # 결측치 제거
        all_vars = [dependent_var] + independent_vars
        self.df = self.df[[date_col] + all_vars].dropna(subset=all_vars)
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # 결과 저장
        self.monthly_coefs_ = None
        self.results_ = None
    
    def fit(
        self,
        newey_west_lags: int = 12,
        min_obs_per_period: int = 30
    ) -> pd.DataFrame:
        """
        Fama-MacBeth 회귀분석 실행
        
        Parameters
        ----------
        newey_west_lags : int, default 12
            Newey-West HAC 추정 래그 수
        min_obs_per_period : int, default 30
            기간별 최소 관측치 수
        
        Returns
        -------
        pd.DataFrame
            회귀분석 결과 테이블
        """
        # 1단계: 기간별 횡단면 회귀분석
        self.monthly_coefs_ = self._run_cross_sectional_regressions(min_obs_per_period)
        
        if self.monthly_coefs_.empty:
            raise ValueError("No valid cross-sectional regressions performed")
        
        # 2단계: 시계열 평균 및 t-통계량
        results = self._compute_time_series_statistics(newey_west_lags)
        
        self.results_ = results
        return results
    
    def fit_with_interactions(
        self,
        interaction_pairs: List[Tuple[str, str]],
        newey_west_lags: int = 12,
        min_obs_per_period: int = 30
    ) -> pd.DataFrame:
        """
        교차항을 포함한 Fama-MacBeth 회귀분석
        
        Parameters
        ----------
        interaction_pairs : List[Tuple[str, str]]
            교차항을 구성할 변수 쌍 리스트
            예: [('var1', 'var2'), ('var1', 'var3')]
        newey_west_lags : int, default 12
            Newey-West 래그 수
        min_obs_per_period : int, default 30
            기간별 최소 관측치 수
        
        Returns
        -------
        pd.DataFrame
            회귀분석 결과 테이블
        """
        df_copy = self.df.copy()
        additional_vars = []
        
        # 교차항 생성
        for var1, var2 in interaction_pairs:
            interaction_name = f"{var1}_x_{var2}"
            df_copy[interaction_name] = df_copy[var1] * df_copy[var2]
            additional_vars.append(interaction_name)
        
        # 독립변수 업데이트
        original_vars = self.independent_vars.copy()
        self.independent_vars = original_vars + additional_vars
        self.df = df_copy
        
        # 회귀분석 실행
        results = self.fit(newey_west_lags, min_obs_per_period)
        
        # 원래 독립변수 리스트 복원
        self.independent_vars = original_vars
        
        return results
    
    def _run_cross_sectional_regressions(
        self, 
        min_obs: int
    ) -> pd.DataFrame:
        """기간별 횡단면 회귀분석"""
        
        def fit_cross_section(group):
            if len(group) < min_obs:
                return None
            
            y = group[self.dependent_var]
            X = group[self.independent_vars]
            
            if self.add_constant:
                X = sm.add_constant(X)
            
            try:
                model = sm.OLS(y, X).fit()
                return model.params
            except Exception:
                return None
        
        # 기간별 회귀분석 수행
        results = []
        for date, group in self.df.groupby(self.date_col):
            coefs = fit_cross_section(group)
            if coefs is not None:
                coefs.name = date
                results.append(coefs)
        
        if not results:
            return pd.DataFrame()
        
        monthly_coefs = pd.DataFrame(results)
        monthly_coefs.index = pd.to_datetime(monthly_coefs.index)
        
        return monthly_coefs
    
    def _compute_time_series_statistics(
        self, 
        newey_west_lags: int
    ) -> pd.DataFrame:
        """시계열 평균 및 Newey-West t-통계량 계산"""
        results = []
        
        vars_to_analyze = ['const'] + self.independent_vars if self.add_constant else self.independent_vars
        
        for var in vars_to_analyze:
            if var not in self.monthly_coefs_.columns:
                continue
            
            coef_series = self.monthly_coefs_[var].dropna()
            n_periods = len(coef_series)
            
            if n_periods < 2:
                continue
            
            # Newey-West t-통계량
            y = coef_series.values
            X = sm.add_constant(np.ones(len(y)))
            
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})
            
            result = FamaMacBethResult(
                variable=var,
                coefficient=coef_series.mean(),
                t_stat=model.tvalues[0],
                p_value=model.pvalues[0],
                std_error=model.bse[0],
                n_periods=n_periods
            )
            
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def get_monthly_coefficients(self) -> pd.DataFrame:
        """기간별 계수 시계열 반환"""
        if self.monthly_coefs_ is None:
            raise ValueError("Call fit() first")
        return self.monthly_coefs_
    
    def summary(self) -> str:
        """결과 요약 문자열"""
        if self.results_ is None:
            return "Model not fitted yet. Call fit() first."
        
        summary_lines = [
            "=" * 70,
            "Fama-MacBeth Regression Results",
            "=" * 70,
            f"Dependent Variable: {self.dependent_var}",
            f"Independent Variables: {', '.join(self.independent_vars)}",
            f"Number of Periods: {self.monthly_coefs_.shape[0]}",
            "-" * 70,
            ""
        ]
        
        # 결과 테이블
        header = f"{'Variable':<20} {'Coef':>12} {'t-stat':>12} {'p-value':>12}"
        summary_lines.append(header)
        summary_lines.append("-" * 70)
        
        for _, row in self.results_.iterrows():
            var = row['Variable']
            coef = row['Coefficient']
            t_stat = row['Newey-West t-stat']
            p_val = row['p-value']
            
            # 유의성 별표
            if p_val < 0.01:
                stars = "***"
            elif p_val < 0.05:
                stars = "**"
            elif p_val < 0.10:
                stars = "*"
            else:
                stars = ""
            
            line = f"{var:<20} {coef:>12.4f}{stars:<3} {t_stat:>12.2f} {p_val:>12.4f}"
            summary_lines.append(line)
        
        summary_lines.append("=" * 70)
        summary_lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
        summary_lines.append("Standard errors: Newey-West HAC")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return self.summary()


def fama_macbeth_regression(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    date_col: str = 'datetime',
    newey_west_lags: int = 12,
    interaction_vars: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Fama-MacBeth 회귀분석 편의 함수
    
    Parameters
    ----------
    df : pd.DataFrame
        패널 데이터
    dependent_var : str
        종속변수명
    independent_vars : List[str]
        독립변수 리스트
    date_col : str, default 'datetime'
        날짜 컬럼명
    newey_west_lags : int, default 12
        Newey-West 래그 수
    interaction_vars : List[Tuple[str, str]], optional
        교차항 변수 쌍 리스트
    
    Returns
    -------
    pd.DataFrame
        회귀분석 결과 테이블
    
    Examples
    --------
    >>> # 기본 사용
    >>> results = fama_macbeth_regression(
    ...     df=data,
    ...     dependent_var='ret_excess_lead',
    ...     independent_vars=['size', 'bm', 'mom', 'illiq']
    ... )
    
    >>> # 교차항 포함
    >>> results = fama_macbeth_regression(
    ...     df=data,
    ...     dependent_var='ret_excess_lead',
    ...     independent_vars=['MAX', 'retail_ratio'],
    ...     interaction_vars=[('MAX', 'retail_ratio')]
    ... )
    """
    fmb = FamaMacBethRegression(
        df=df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        date_col=date_col
    )
    
    if interaction_vars:
        return fmb.fit_with_interactions(interaction_vars, newey_west_lags)
    else:
        return fmb.fit(newey_west_lags)


def run_multiple_fmb_specifications(
    df: pd.DataFrame,
    dependent_var: str,
    specifications: Dict[str, List[str]],
    date_col: str = 'datetime',
    newey_west_lags: int = 12
) -> pd.DataFrame:
    """
    여러 모델 사양으로 Fama-MacBeth 회귀분석 실행
    
    Parameters
    ----------
    df : pd.DataFrame
        패널 데이터
    dependent_var : str
        종속변수명
    specifications : Dict[str, List[str]]
        모델 사양 딕셔너리 {모델명: 독립변수 리스트}
    date_col : str, default 'datetime'
        날짜 컬럼명
    newey_west_lags : int, default 12
        Newey-West 래그 수
    
    Returns
    -------
    pd.DataFrame
        모든 모델의 결과를 열로 결합한 테이블
    
    Examples
    --------
    >>> specs = {
    ...     'Model 1': ['size', 'bm'],
    ...     'Model 2': ['size', 'bm', 'mom'],
    ...     'Model 3': ['size', 'bm', 'mom', 'illiq']
    ... }
    >>> results = run_multiple_fmb_specifications(data, 'ret_excess_lead', specs)
    """
    all_results = {}
    all_variables = set()
    
    for model_name, vars_list in specifications.items():
        try:
            results = fama_macbeth_regression(
                df=df,
                dependent_var=dependent_var,
                independent_vars=vars_list,
                date_col=date_col,
                newey_west_lags=newey_west_lags
            )
            
            # 결과를 딕셔너리로 변환
            result_dict = {}
            for _, row in results.iterrows():
                var = row['Variable']
                coef = row['Coefficient']
                t_stat = row['Newey-West t-stat']
                p_val = row['p-value']
                
                # 유의성 별표
                if p_val < 0.01:
                    stars = "***"
                elif p_val < 0.05:
                    stars = "**"
                elif p_val < 0.10:
                    stars = "*"
                else:
                    stars = ""
                
                result_dict[var] = f"{coef:.4f}{stars}"
                result_dict[f"{var}_tstat"] = f"({t_stat:.2f})"
                all_variables.add(var)
            
            all_results[model_name] = result_dict
            
        except Exception as e:
            print(f"Error in {model_name}: {e}")
            continue
    
    # 결과 테이블 생성
    rows = []
    for var in sorted(all_variables, key=lambda x: (x != 'const', x)):
        row = {'Variable': var}
        for model_name in specifications.keys():
            if model_name in all_results:
                row[model_name] = all_results[model_name].get(var, '')
        rows.append(row)
        
        # t-stat 행 추가
        tstat_row = {'Variable': ''}
        for model_name in specifications.keys():
            if model_name in all_results:
                tstat_row[model_name] = all_results[model_name].get(f"{var}_tstat", '')
        rows.append(tstat_row)
    
    return pd.DataFrame(rows)
