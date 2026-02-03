"""
aplab.portfolio.performance
===========================
포트폴리오 성과 분석 (Factor Regression, Risk-Adjusted Returns)
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, List, Tuple, Union, Dict, Any
from ..utils.stats import newey_west_tstat, calculate_sharpe_ratio


class PerformanceAnalyzer:
    """
    포트폴리오 성과 분석 클래스
    
    Examples
    --------
    >>> analyzer = PerformanceAnalyzer(factors_df)
    >>> results = analyzer.factor_regression(portfolio_returns, model='carhart')
    """
    
    def __init__(
        self,
        factors_df: Optional[pd.DataFrame] = None,
        date_col: str = 'datetime',
        rf_col: str = 'rf'
    ):
        """
        Parameters
        ----------
        factors_df : pd.DataFrame, optional
            팩터 데이터
        date_col : str, default 'datetime'
            날짜 컬럼명
        rf_col : str, default 'rf'
            무위험 수익률 컬럼명
        """
        self.factors_df = factors_df
        self.date_col = date_col
        self.rf_col = rf_col
        
        if factors_df is not None:
            if date_col in factors_df.columns:
                self.factors_df[date_col] = pd.to_datetime(factors_df[date_col])
            elif 'date' in factors_df.columns:
                self.factors_df['date'] = pd.to_datetime(factors_df['date'])
                self.date_col = 'date'
    
    def factor_regression(
        self,
        returns: pd.Series,
        model: str = 'carhart',
        custom_factors: Optional[List[str]] = None,
        newey_west_lags: int = 12,
        print_result: bool = False
    ) -> Dict[str, Any]:
        """
        팩터 모델 시계열 회귀분석
        
        Parameters
        ----------
        returns : pd.Series
            포트폴리오 수익률 시계열 (인덱스: 날짜)
        model : str, default 'carhart'
            팩터 모델 ('capm', 'ff3', 'carhart', 'ff5', 'custom')
        custom_factors : List[str], optional
            커스텀 팩터 리스트 (model='custom'인 경우)
        newey_west_lags : int, default 12
            Newey-West 래그 수
        print_result : bool, default False
            결과 출력 여부
        
        Returns
        -------
        Dict[str, Any]
            회귀분석 결과 (alpha, t-stats, loadings 등)
        """
        if self.factors_df is None:
            raise ValueError("factors_df is required for factor regression")
        
        # 팩터 선택
        factor_cols = self._get_factor_columns(model, custom_factors)
        
        # 데이터 병합
        merged = pd.merge(
            returns.rename('ret'),
            self.factors_df,
            left_index=True,
            right_on=self.date_col,
            how='inner'
        ).dropna(subset=['ret'] + factor_cols)
        
        if len(merged) < 20:
            return {'alpha': np.nan, 'alpha_tstat': np.nan, 'n_obs': len(merged)}
        
        # 회귀분석
        y = merged['ret']
        X = sm.add_constant(merged[factor_cols])
        
        model_fit = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})
        
        # 결과 정리
        results = {
            'alpha': model_fit.params['const'] * 100,  # 퍼센트로 변환
            'alpha_tstat': model_fit.tvalues['const'],
            'alpha_pvalue': model_fit.pvalues['const'],
            'r_squared': model_fit.rsquared,
            'r_squared_adj': model_fit.rsquared_adj,
            'n_obs': len(merged),
            'loadings': {},
            'loadings_tstat': {}
        }
        
        for factor in factor_cols:
            results['loadings'][factor] = model_fit.params[factor]
            results['loadings_tstat'][factor] = model_fit.tvalues[factor]
        
        if print_result:
            self._print_regression_result(results, model)
        
        return results
    
    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        rf: Optional[pd.Series] = None,
        annualize: bool = True,
        periods_per_year: int = 12
    ) -> Dict[str, float]:
        """
        포트폴리오 성과 지표 계산
        
        Parameters
        ----------
        returns : pd.Series
            포트폴리오 수익률 시계열
        benchmark_returns : pd.Series, optional
            벤치마크 수익률
        rf : pd.Series, optional
            무위험 수익률
        annualize : bool, default True
            연율화 여부
        periods_per_year : int, default 12
            연간 기간 수
        
        Returns
        -------
        Dict[str, float]
            성과 지표 딕셔너리
        """
        returns = returns.dropna()
        n = len(returns)
        
        if n < 2:
            return {}
        
        # 기본 통계
        mean_return = returns.mean()
        std_return = returns.std()
        
        if annualize:
            ann_return = mean_return * periods_per_year
            ann_std = std_return * np.sqrt(periods_per_year)
        else:
            ann_return = mean_return
            ann_std = std_return
        
        metrics = {
            'mean_return': ann_return * 100,
            'volatility': ann_std * 100,
            'skewness': returns.skew(),
            'kurtosis': returns.kurt(),
            'min_return': returns.min() * 100,
            'max_return': returns.max() * 100,
            'n_obs': n
        }
        
        # Sharpe Ratio
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(
            returns, rf, annualize, periods_per_year
        )
        
        # Sortino Ratio (하방 변동성)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if annualize:
                downside_std *= np.sqrt(periods_per_year)
            if downside_std > 0:
                metrics['sortino_ratio'] = ann_return / downside_std
            else:
                metrics['sortino_ratio'] = np.nan
        else:
            metrics['sortino_ratio'] = np.nan
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min() * 100
        
        # Information Ratio (벤치마크가 있는 경우)
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns.reindex(returns.index)
            active_returns = active_returns.dropna()
            if len(active_returns) > 0:
                tracking_error = active_returns.std()
                if annualize:
                    tracking_error *= np.sqrt(periods_per_year)
                if tracking_error > 0:
                    metrics['information_ratio'] = (
                        active_returns.mean() * periods_per_year / tracking_error
                    ) if annualize else (active_returns.mean() / tracking_error)
                    metrics['tracking_error'] = tracking_error * 100
        
        # t-statistic for mean return
        mean_val, t_stat, p_val = newey_west_tstat(returns)
        metrics['mean_return_tstat'] = t_stat
        metrics['mean_return_pvalue'] = p_val
        
        return metrics
    
    def compare_portfolios(
        self,
        portfolio_returns: Dict[str, pd.Series],
        model: str = 'carhart',
        newey_west_lags: int = 12
    ) -> pd.DataFrame:
        """
        여러 포트폴리오 성과 비교
        
        Parameters
        ----------
        portfolio_returns : Dict[str, pd.Series]
            포트폴리오별 수익률 시계열 딕셔너리
        model : str, default 'carhart'
            팩터 모델
        newey_west_lags : int, default 12
            Newey-West 래그 수
        
        Returns
        -------
        pd.DataFrame
            포트폴리오 비교 테이블
        """
        results = []
        
        for name, returns in portfolio_returns.items():
            # 기본 성과 지표
            metrics = self.calculate_performance_metrics(returns)
            
            # 팩터 회귀분석
            if self.factors_df is not None:
                factor_results = self.factor_regression(
                    returns, model=model, newey_west_lags=newey_west_lags
                )
                metrics.update({
                    'alpha': factor_results.get('alpha', np.nan),
                    'alpha_tstat': factor_results.get('alpha_tstat', np.nan)
                })
            
            metrics['portfolio'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index('portfolio')
        
        return df
    
    def spanning_test(
        self,
        returns: pd.Series,
        benchmark_factors: List[str],
        new_factors: List[str],
        newey_west_lags: int = 12
    ) -> Dict[str, Any]:
        """
        팩터 스패닝 테스트
        
        새로운 팩터가 기존 팩터로 설명되지 않는 추가적인 정보를 가지는지 검정
        
        Parameters
        ----------
        returns : pd.Series
            테스트할 팩터 수익률
        benchmark_factors : List[str]
            기존 팩터 리스트
        new_factors : List[str]
            추가 팩터 리스트
        newey_west_lags : int, default 12
            Newey-West 래그 수
        
        Returns
        -------
        Dict[str, Any]
            스패닝 테스트 결과
        """
        if self.factors_df is None:
            raise ValueError("factors_df is required for spanning test")
        
        # 전체 팩터로 회귀
        all_factors = benchmark_factors + new_factors
        merged = pd.merge(
            returns.rename('ret'),
            self.factors_df,
            left_index=True,
            right_on=self.date_col,
            how='inner'
        ).dropna(subset=['ret'] + all_factors)
        
        # 제한 모델 (기존 팩터만)
        y = merged['ret']
        X_restricted = sm.add_constant(merged[benchmark_factors])
        model_restricted = sm.OLS(y, X_restricted).fit()
        
        # 비제한 모델 (전체 팩터)
        X_full = sm.add_constant(merged[all_factors])
        model_full = sm.OLS(y, X_full).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})
        
        # F-test
        f_stat = ((model_restricted.ssr - model_full.ssr) / len(new_factors)) / (
            model_full.ssr / model_full.df_resid
        )
        
        from scipy import stats as scipy_stats
        f_pvalue = 1 - scipy_stats.f.cdf(f_stat, len(new_factors), model_full.df_resid)
        
        return {
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'restricted_r2': model_restricted.rsquared,
            'full_r2': model_full.rsquared,
            'r2_improvement': model_full.rsquared - model_restricted.rsquared,
            'alpha_full': model_full.params['const'] * 100,
            'alpha_full_tstat': model_full.tvalues['const']
        }
    
    def _get_factor_columns(
        self, 
        model: str, 
        custom_factors: Optional[List[str]] = None
    ) -> List[str]:
        """팩터 모델에 따른 컬럼 선택"""
        factor_models = {
            'capm': ['mkt'],
            'ff3': ['mkt', 'smb', 'hml'],
            'carhart': ['mkt', 'smb', 'hml', 'mom'],
            'ff5': ['mkt', 'smb', 'hml', 'rmw', 'cma'],
            'ff6': ['mkt', 'smb', 'hml', 'rmw', 'cma', 'mom'],
        }
        
        if model == 'custom':
            if custom_factors is None:
                raise ValueError("custom_factors required when model='custom'")
            return custom_factors
        
        if model not in factor_models:
            raise ValueError(f"Unknown model: {model}. Use one of {list(factor_models.keys())}")
        
        factors = factor_models[model]
        available = [f for f in factors if f in self.factors_df.columns]
        
        if not available:
            raise ValueError(f"No factor columns found in factors_df for model '{model}'")
        
        return available
    
    def _print_regression_result(self, results: Dict, model: str):
        """회귀분석 결과 출력"""
        print(f"\n{'='*50}")
        print(f"Factor Model Regression: {model.upper()}")
        print(f"{'='*50}")
        print(f"Alpha (%):    {results['alpha']:.4f}")
        print(f"t-stat:       {results['alpha_tstat']:.4f}")
        print(f"R-squared:    {results['r_squared']:.4f}")
        print(f"Observations: {results['n_obs']}")
        print(f"\nFactor Loadings:")
        for factor, loading in results['loadings'].items():
            tstat = results['loadings_tstat'][factor]
            print(f"  {factor:10s}: {loading:8.4f} (t={tstat:.2f})")


def factor_regression(
    returns: pd.Series,
    factors_df: pd.DataFrame,
    model: str = 'carhart',
    newey_west_lags: int = 12
) -> Dict[str, Any]:
    """
    팩터 회귀분석 편의 함수
    
    Parameters
    ----------
    returns : pd.Series
        포트폴리오 수익률
    factors_df : pd.DataFrame
        팩터 데이터
    model : str, default 'carhart'
        팩터 모델
    newey_west_lags : int, default 12
        Newey-West 래그 수
    
    Returns
    -------
    Dict[str, Any]
        회귀분석 결과
    """
    analyzer = PerformanceAnalyzer(factors_df)
    return analyzer.factor_regression(returns, model=model, newey_west_lags=newey_west_lags)


def calculate_alpha(
    returns: pd.Series,
    factors_df: pd.DataFrame,
    factor_cols: List[str] = ['mkt', 'smb', 'hml', 'mom'],
    newey_west_lags: int = 12
) -> Tuple[float, float]:
    """
    알파 계산 편의 함수
    
    Parameters
    ----------
    returns : pd.Series
        포트폴리오 수익률
    factors_df : pd.DataFrame
        팩터 데이터
    factor_cols : List[str]
        팩터 컬럼명
    newey_west_lags : int, default 12
        Newey-West 래그 수
    
    Returns
    -------
    Tuple[float, float]
        (알파 (%), t-통계량)
    """
    analyzer = PerformanceAnalyzer(factors_df)
    results = analyzer.factor_regression(
        returns, 
        model='custom', 
        custom_factors=factor_cols,
        newey_west_lags=newey_west_lags
    )
    return results.get('alpha', np.nan), results.get('alpha_tstat', np.nan)
