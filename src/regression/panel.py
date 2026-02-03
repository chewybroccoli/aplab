"""
aplab.regression.panel
======================
Panel Regression (Fixed Effects, Random Effects)
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, List, Tuple, Union, Dict, Any, Literal
from dataclasses import dataclass

# linearmodels는 선택적 의존성
try:
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False


@dataclass
class PanelRegressionResult:
    """패널 회귀분석 결과"""
    variable: str
    coefficient: float
    t_stat: float
    p_value: float
    std_error: float
    
    @property
    def significance(self) -> str:
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
            't-stat': self.t_stat,
            'p-value': self.p_value,
            'Std Error': self.std_error
        }


class PanelRegression:
    """
    패널 회귀분석 클래스
    
    고정효과(Fixed Effects), 랜덤효과(Random Effects), 
    이중 클러스터 표준오차(Double-Clustered SE) 지원
    
    Examples
    --------
    >>> panel = PanelRegression(
    ...     df=data,
    ...     dependent_var='ret_excess_lead',
    ...     independent_vars=['size', 'bm', 'mom'],
    ...     entity_col='ticker',
    ...     time_col='datetime'
    ... )
    >>> results = panel.fit(entity_effects=False, time_effects=True)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        entity_col: str = 'ticker',
        time_col: str = 'datetime'
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            패널 데이터
        dependent_var : str
            종속변수명
        independent_vars : List[str]
            독립변수 리스트
        entity_col : str, default 'ticker'
            개체(기업) 식별 컬럼
        time_col : str, default 'datetime'
            시간 식별 컬럼
        """
        if not HAS_LINEARMODELS:
            raise ImportError(
                "linearmodels is required for PanelRegression. "
                "Install it with: pip install linearmodels"
            )
        
        self.df = df.copy()
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.entity_col = entity_col
        self.time_col = time_col
        
        # 데이터 준비
        all_vars = [dependent_var] + independent_vars
        self.df = self.df[[entity_col, time_col] + all_vars].dropna(subset=all_vars)
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        
        # 패널 인덱스 설정
        self.df_panel = self.df.set_index([entity_col, time_col])
        
        # 결과 저장
        self.model_result_ = None
    
    def fit(
        self,
        entity_effects: bool = False,
        time_effects: bool = True,
        cluster_entity: bool = True,
        cluster_time: bool = True,
        add_constant: bool = True
    ) -> pd.DataFrame:
        """
        패널 고정효과 회귀분석
        
        Parameters
        ----------
        entity_effects : bool, default False
            개체 고정효과 포함 여부
        time_effects : bool, default True
            시간 고정효과 포함 여부
        cluster_entity : bool, default True
            개체 클러스터 표준오차
        cluster_time : bool, default True
            시간 클러스터 표준오차
        add_constant : bool, default True
            상수항 포함 여부 (time_effects=False일 때만 유효)
        
        Returns
        -------
        pd.DataFrame
            회귀분석 결과 테이블
        """
        y = self.df_panel[self.dependent_var]
        X = self.df_panel[self.independent_vars]
        
        if add_constant and not time_effects and not entity_effects:
            X = sm.add_constant(X)
        
        # PanelOLS 모델
        model = PanelOLS(
            y, X,
            entity_effects=entity_effects,
            time_effects=time_effects
        )
        
        # 클러스터링 옵션
        if cluster_entity and cluster_time:
            cov_type = 'clustered'
            results = model.fit(cov_type=cov_type, cluster_entity=True, cluster_time=True)
        elif cluster_entity:
            cov_type = 'clustered'
            results = model.fit(cov_type=cov_type, cluster_entity=True)
        elif cluster_time:
            cov_type = 'clustered'
            results = model.fit(cov_type=cov_type, cluster_time=True)
        else:
            results = model.fit()
        
        self.model_result_ = results
        
        return self._format_results(results)
    
    def fit_random_effects(self) -> pd.DataFrame:
        """
        랜덤효과 회귀분석
        
        Returns
        -------
        pd.DataFrame
            회귀분석 결과 테이블
        """
        y = self.df_panel[self.dependent_var]
        X = sm.add_constant(self.df_panel[self.independent_vars])
        
        model = RandomEffects(y, X)
        results = model.fit()
        
        self.model_result_ = results
        
        return self._format_results(results)
    
    def fit_pooled_ols(
        self,
        cluster_entity: bool = True
    ) -> pd.DataFrame:
        """
        풀링 OLS 회귀분석
        
        Parameters
        ----------
        cluster_entity : bool, default True
            개체 클러스터 표준오차
        
        Returns
        -------
        pd.DataFrame
            회귀분석 결과 테이블
        """
        y = self.df_panel[self.dependent_var]
        X = sm.add_constant(self.df_panel[self.independent_vars])
        
        model = PooledOLS(y, X)
        
        if cluster_entity:
            results = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            results = model.fit()
        
        self.model_result_ = results
        
        return self._format_results(results)
    
    def hausman_test(self) -> Dict[str, float]:
        """
        하우스만 검정 (Fixed vs Random Effects)
        
        Returns
        -------
        Dict[str, float]
            검정 통계량 및 p-value
        """
        y = self.df_panel[self.dependent_var]
        X = sm.add_constant(self.df_panel[self.independent_vars])
        
        # Fixed Effects
        fe_model = PanelOLS(y, X, entity_effects=True)
        fe_results = fe_model.fit()
        
        # Random Effects
        re_model = RandomEffects(y, X)
        re_results = re_model.fit()
        
        # 하우스만 통계량 계산
        b_fe = fe_results.params
        b_re = re_results.params
        
        common_params = b_fe.index.intersection(b_re.index)
        
        diff = b_fe[common_params] - b_re[common_params]
        var_diff = fe_results.cov[common_params].loc[common_params] - re_results.cov[common_params].loc[common_params]
        
        try:
            var_diff_inv = np.linalg.inv(var_diff)
            hausman_stat = float(diff.T @ var_diff_inv @ diff)
            
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(hausman_stat, df=len(common_params))
            
            return {
                'statistic': hausman_stat,
                'p_value': p_value,
                'df': len(common_params),
                'reject_null': p_value < 0.05
            }
        except Exception:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'df': len(common_params),
                'reject_null': None
            }
    
    def _format_results(self, results) -> pd.DataFrame:
        """결과 포맷팅"""
        output = []
        
        for var in results.params.index:
            result = PanelRegressionResult(
                variable=var,
                coefficient=results.params[var],
                t_stat=results.tstats[var],
                p_value=results.pvalues[var],
                std_error=results.std_errors[var]
            )
            output.append(result.to_dict())
        
        return pd.DataFrame(output)
    
    def summary(self) -> str:
        """결과 요약"""
        if self.model_result_ is None:
            return "Model not fitted yet. Call fit() first."
        return str(self.model_result_)


def panel_regression(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    entity_col: str = 'ticker',
    time_col: str = 'datetime',
    entity_effects: bool = False,
    time_effects: bool = True,
    cluster_entity: bool = True,
    cluster_time: bool = True
) -> pd.DataFrame:
    """
    패널 회귀분석 편의 함수
    
    Parameters
    ----------
    df : pd.DataFrame
        패널 데이터
    dependent_var : str
        종속변수명
    independent_vars : List[str]
        독립변수 리스트
    entity_col : str, default 'ticker'
        개체 컬럼명
    time_col : str, default 'datetime'
        시간 컬럼명
    entity_effects : bool, default False
        개체 고정효과
    time_effects : bool, default True
        시간 고정효과
    cluster_entity : bool, default True
        개체 클러스터 SE
    cluster_time : bool, default True
        시간 클러스터 SE
    
    Returns
    -------
    pd.DataFrame
        회귀분석 결과 테이블
    
    Examples
    --------
    >>> results = panel_regression(
    ...     df=data,
    ...     dependent_var='ret_excess_lead',
    ...     independent_vars=['size', 'bm', 'mom', 'illiq'],
    ...     time_effects=True
    ... )
    """
    panel = PanelRegression(
        df=df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        entity_col=entity_col,
        time_col=time_col
    )
    
    return panel.fit(
        entity_effects=entity_effects,
        time_effects=time_effects,
        cluster_entity=cluster_entity,
        cluster_time=cluster_time
    )


def run_multiple_panel_specifications(
    df: pd.DataFrame,
    dependent_var: str,
    specifications: Dict[str, List[str]],
    entity_col: str = 'ticker',
    time_col: str = 'datetime',
    entity_effects: bool = False,
    time_effects: bool = True
) -> pd.DataFrame:
    """
    여러 모델 사양으로 패널 회귀분석 실행
    
    Parameters
    ----------
    df : pd.DataFrame
        패널 데이터
    dependent_var : str
        종속변수명
    specifications : Dict[str, List[str]]
        모델 사양 {모델명: 독립변수 리스트}
    entity_col : str, default 'ticker'
        개체 컬럼명
    time_col : str, default 'datetime'
        시간 컬럼명
    entity_effects : bool, default False
        개체 고정효과
    time_effects : bool, default True
        시간 고정효과
    
    Returns
    -------
    pd.DataFrame
        모든 모델의 결과를 열로 결합한 테이블
    """
    all_results = {}
    all_variables = set()
    
    for model_name, vars_list in specifications.items():
        try:
            results = panel_regression(
                df=df,
                dependent_var=dependent_var,
                independent_vars=vars_list,
                entity_col=entity_col,
                time_col=time_col,
                entity_effects=entity_effects,
                time_effects=time_effects
            )
            
            result_dict = {}
            for _, row in results.iterrows():
                var = row['Variable']
                coef = row['Coefficient']
                t_stat = row['t-stat']
                p_val = row['p-value']
                
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
        
        tstat_row = {'Variable': ''}
        for model_name in specifications.keys():
            if model_name in all_results:
                tstat_row[model_name] = all_results[model_name].get(f"{var}_tstat", '')
        rows.append(tstat_row)
    
    return pd.DataFrame(rows)


# 간단한 OLS 회귀분석 (linearmodels 없이)
def simple_panel_ols(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    time_col: str = 'datetime',
    time_dummies: bool = True,
    cluster_col: Optional[str] = None
) -> pd.DataFrame:
    """
    간단한 패널 OLS (statsmodels만 사용)
    
    linearmodels 없이 시간 더미와 클러스터 SE 지원
    
    Parameters
    ----------
    df : pd.DataFrame
        패널 데이터
    dependent_var : str
        종속변수명
    independent_vars : List[str]
        독립변수 리스트
    time_col : str, default 'datetime'
        시간 컬럼명
    time_dummies : bool, default True
        시간 더미 포함 여부
    cluster_col : str, optional
        클러스터 변수
    
    Returns
    -------
    pd.DataFrame
        회귀분석 결과
    """
    df = df.copy()
    
    # 결측치 제거
    all_vars = [dependent_var] + independent_vars + [time_col]
    df = df[all_vars].dropna()
    
    # 시간 더미 생성
    if time_dummies:
        time_dummies_df = pd.get_dummies(df[time_col], prefix='time', drop_first=True)
        X = pd.concat([df[independent_vars], time_dummies_df], axis=1)
    else:
        X = df[independent_vars]
        X = sm.add_constant(X)
    
    y = df[dependent_var]
    
    # OLS 회귀분석
    if cluster_col and cluster_col in df.columns:
        # 클러스터 SE는 statsmodels의 get_robustcov_results 사용
        model = sm.OLS(y, X).fit()
        # 클러스터 SE 계산
        model = model.get_robustcov_results(
            cov_type='cluster',
            groups=df[cluster_col]
        )
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')  # 이분산성 강건 SE
    
    # 결과 정리 (시간 더미 제외)
    results = []
    for var in independent_vars:
        if var in model.params.index:
            results.append({
                'Variable': var,
                'Coefficient': model.params[var],
                't-stat': model.tvalues[var],
                'p-value': model.pvalues[var],
                'Std Error': model.bse[var]
            })
    
    if 'const' in model.params.index:
        results.insert(0, {
            'Variable': 'const',
            'Coefficient': model.params['const'],
            't-stat': model.tvalues['const'],
            'p-value': model.pvalues['const'],
            'Std Error': model.bse['const']
        })
    
    return pd.DataFrame(results)
