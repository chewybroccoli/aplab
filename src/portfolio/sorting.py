"""
aplab.portfolio.sorting
=======================
포트폴리오 정렬 (Univariate, Bivariate, Triple Sorting)
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, List, Tuple, Union, Dict, Any, Literal
from .base import (
    assign_portfolio_quantile,
    calculate_portfolio_return,
    shift_portfolio_assignment
)
from ..base.types import Weighting, SortingMethod


class PortfolioSorter:
    """
    포트폴리오 정렬 분석을 위한 클래스
    
    Examples
    --------
    >>> sorter = PortfolioSorter(
    ...     df=data,
    ...     factors_df=factors,
    ...     date_col='datetime',
    ...     ticker_col='ticker',
    ...     return_col='ret_excess',
    ...     weight_col='mktcap_lag'
    ... )
    >>> results = sorter.univariate_sort('size', n_quantiles=10, weighting='vw')
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        factors_df: Optional[pd.DataFrame] = None,
        date_col: str = 'datetime',
        ticker_col: str = 'ticker',
        return_col: str = 'ret_excess',
        weight_col: str = 'mktcap_lag'
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            분석 데이터
        factors_df : pd.DataFrame, optional
            팩터 데이터 (알파 계산에 필요)
        date_col : str, default 'datetime'
            날짜 컬럼명
        ticker_col : str, default 'ticker'
            종목 컬럼명
        return_col : str, default 'ret_excess'
            수익률 컬럼명
        weight_col : str, default 'mktcap_lag'
            가중치 컬럼명
        """
        self.df = df.copy()
        self.factors_df = factors_df
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.return_col = return_col
        self.weight_col = weight_col
        
        # 날짜 컬럼 datetime 변환
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        if factors_df is not None:
            if date_col in factors_df.columns:
                self.factors_df[date_col] = pd.to_datetime(factors_df[date_col])
            elif 'date' in factors_df.columns:
                self.factors_df['date'] = pd.to_datetime(factors_df['date'])
    
    def univariate_sort(
        self,
        sorting_variable: str,
        n_quantiles: int = 10,
        weighting: Union[str, Weighting] = 'vw',
        compute_alpha: bool = True,
        factor_cols: List[str] = ['mkt', 'smb', 'hml', 'mom'],
        newey_west_lags: int = 12,
        long_short: bool = True,
        inverse: bool = False
    ) -> pd.DataFrame:
        """
        단일 변수 기준 포트폴리오 정렬
        
        Parameters
        ----------
        sorting_variable : str
            정렬 기준 변수
        n_quantiles : int, default 10
            분위수 개수
        weighting : str or Weighting, default 'vw'
            가중 방식
        compute_alpha : bool, default True
            팩터 알파 계산 여부
        factor_cols : List[str]
            팩터 컬럼명 리스트
        newey_west_lags : int, default 12
            Newey-West 래그 수
        long_short : bool, default True
            롱숏 포트폴리오 포함 여부
        inverse : bool, default False
            롱숏 방향 반전 (True: Low-High)
        
        Returns
        -------
        pd.DataFrame
            포트폴리오별 성과 결과
        """
        weighting = str(weighting)
        
        # 필요한 컬럼만 선택하고 결측치 제거
        required_cols = [self.date_col, self.ticker_col, self.return_col, 
                        self.weight_col, sorting_variable]
        df_copy = self.df[required_cols].copy().dropna()
        
        # 1. 포트폴리오 할당
        df_copy['portfolio'] = df_copy.groupby(self.date_col, group_keys=False).apply(
            lambda x: assign_portfolio_quantile(x, sorting_variable, n_quantiles)
        )
        
        # 2. 포트폴리오 정보를 다음 달로 이동
        portfolio_info = df_copy[[self.date_col, self.ticker_col, 'portfolio']].copy()
        portfolio_info[self.date_col] = portfolio_info[self.date_col] + pd.offsets.MonthEnd(1)
        
        # 3. 원본 데이터와 병합
        df_analysis = pd.merge(
            self.df[[self.date_col, self.ticker_col, self.return_col, self.weight_col]],
            portfolio_info,
            on=[self.date_col, self.ticker_col],
            how='inner'
        ).dropna(subset=[self.return_col, self.weight_col])
        
        # 4. 포트폴리오별 수익률 계산
        if weighting == 'vw':
            portfolio_returns = df_analysis.groupby([self.date_col, 'portfolio']).apply(
                lambda x: np.average(x[self.return_col], weights=x[self.weight_col])
                if x[self.weight_col].sum() > 0 else np.nan
            )
        else:
            portfolio_returns = df_analysis.groupby([self.date_col, 'portfolio'])[self.return_col].mean()
        
        portfolio_returns = portfolio_returns.unstack(level='portfolio')
        
        # 5. 롱숏 포트폴리오 생성
        if long_short:
            high_port = portfolio_returns.columns.max()
            low_port = portfolio_returns.columns.min()
            
            if inverse:
                long_short_label = 'Low-High'
                portfolio_returns[long_short_label] = portfolio_returns[low_port] - portfolio_returns[high_port]
            else:
                long_short_label = 'High-Low'
                portfolio_returns[long_short_label] = portfolio_returns[high_port] - portfolio_returns[low_port]
        
        # 6. 성과 계산
        results = self._calculate_portfolio_performance(
            portfolio_returns,
            compute_alpha=compute_alpha,
            factor_cols=factor_cols,
            newey_west_lags=newey_west_lags
        )
        
        return results
    
    def bivariate_sort(
        self,
        sort_var1: str,
        sort_var2: str,
        n_quantiles1: int = 5,
        n_quantiles2: int = 5,
        method: Union[str, SortingMethod] = 'dependent',
        weighting: Union[str, Weighting] = 'vw',
        compute_alpha: bool = True,
        factor_cols: List[str] = ['mkt', 'smb', 'hml', 'mom'],
        newey_west_lags: int = 12,
        inverse: bool = False
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        2변량 포트폴리오 정렬 (Double Sort)
        
        Parameters
        ----------
        sort_var1 : str
            첫 번째 정렬 변수 (통제 변수)
        sort_var2 : str
            두 번째 정렬 변수 (관심 변수)
        n_quantiles1 : int, default 5
            첫 번째 변수의 분위수
        n_quantiles2 : int, default 5
            두 번째 변수의 분위수
        method : str, default 'dependent'
            정렬 방법 ('dependent' 또는 'independent')
        weighting : str or Weighting, default 'vw'
            가중 방식
        compute_alpha : bool, default True
            팩터 알파 계산 여부
        factor_cols : List[str]
            팩터 컬럼명
        newey_west_lags : int, default 12
            Newey-West 래그 수
        inverse : bool, default False
            롱숏 방향 반전
        
        Returns
        -------
        Tuple[pd.DataFrame, float, float]
            (알파 테이블, 평균 롱숏 알파, t-통계량)
        """
        weighting = str(weighting)
        method = str(method)
        
        # 필요한 컬럼 선택
        required_cols = [self.date_col, self.ticker_col, self.return_col,
                        self.weight_col, sort_var1, sort_var2]
        df_copy = self.df[required_cols].copy().dropna()
        
        # 1. 첫 번째 변수로 포트폴리오 할당
        df_copy['portfolio1'] = df_copy.groupby(self.date_col, group_keys=False).apply(
            lambda x: assign_portfolio_quantile(x, sort_var1, n_quantiles1)
        )
        
        # 2. 두 번째 변수로 포트폴리오 할당
        if method == 'dependent':
            # 종속 정렬: 첫 번째 변수 그룹 내에서 두 번째 변수로 정렬
            df_copy['portfolio2'] = df_copy.groupby(
                [self.date_col, 'portfolio1'], group_keys=False
            ).apply(
                lambda x: assign_portfolio_quantile(x, sort_var2, n_quantiles2)
            )
        else:
            # 독립 정렬: 두 번째 변수도 전체에서 독립적으로 정렬
            df_copy['portfolio2'] = df_copy.groupby(self.date_col, group_keys=False).apply(
                lambda x: assign_portfolio_quantile(x, sort_var2, n_quantiles2)
            )
        
        # 3. 포트폴리오 정보를 다음 달로 이동
        portfolio_info = df_copy[[self.date_col, self.ticker_col, 'portfolio1', 'portfolio2']].copy()
        portfolio_info[self.date_col] = portfolio_info[self.date_col] + pd.offsets.MonthEnd(1)
        
        # 4. 원본 데이터와 병합
        df_analysis = pd.merge(
            self.df[[self.date_col, self.ticker_col, self.return_col, self.weight_col]],
            portfolio_info,
            on=[self.date_col, self.ticker_col],
            how='inner'
        ).dropna(subset=[self.return_col, self.weight_col])
        
        # 5. 포트폴리오별 수익률 계산
        if weighting == 'vw':
            portfolio_returns = df_analysis.groupby(
                [self.date_col, 'portfolio1', 'portfolio2']
            ).apply(
                lambda x: np.average(x[self.return_col], weights=x[self.weight_col])
                if x[self.weight_col].sum() > 0 else np.nan
            )
        else:
            portfolio_returns = df_analysis.groupby(
                [self.date_col, 'portfolio1', 'portfolio2']
            )[self.return_col].mean()
        
        # 6. 알파 계산
        if compute_alpha and self.factors_df is not None:
            alpha_results = self._calculate_bivariate_alpha(
                portfolio_returns, 
                n_quantiles1, 
                n_quantiles2,
                factor_cols,
                newey_west_lags,
                inverse
            )
            return alpha_results
        else:
            # 알파 없이 평균 수익률 반환
            mean_returns = portfolio_returns.groupby(['portfolio1', 'portfolio2']).mean().unstack()
            return mean_returns, np.nan, np.nan
    
    def get_holdings(
        self,
        sorting_variable: str,
        n_quantiles: int = 10,
        weighting: Union[str, Weighting] = 'vw',
        return_weights: bool = True
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        포트폴리오 정렬 후 시점별 포트폴리오 홀딩 반환
        
        Parameters
        ----------
        sorting_variable : str
            정렬 기준 변수
        n_quantiles : int, default 10
            분위수 개수
        weighting : str or Weighting, default 'vw'
            가중 방식 ('ew' 또는 'vw')
        return_weights : bool, default True
            가중치 반환 여부 (False면 종목 리스트만)
        
        Returns
        -------
        Dict[str, Dict[int, Dict[str, float]]]
            {날짜: {포트폴리오번호: {종목코드: 가중치}}}
            
        Examples
        --------
        >>> sorter = PortfolioSorter(df, factors_df)
        >>> holdings = sorter.get_holdings('size', n_quantiles=10, weighting='vw')
        >>> # 2020-06 시점의 포트폴리오 1번 홀딩
        >>> holdings['2020-06'][1]
        {'A005930': 0.15, 'A000660': 0.08, ...}
        """
        weighting = str(weighting)
        
        required_cols = [self.date_col, self.ticker_col, self.weight_col, sorting_variable]
        df_copy = self.df[required_cols].copy().dropna()
        
        # 포트폴리오 할당
        df_copy['portfolio'] = df_copy.groupby(self.date_col, group_keys=False).apply(
            lambda x: assign_portfolio_quantile(x, sorting_variable, n_quantiles)
        )
        
        # 포트폴리오 정보를 다음 달로 이동 (look-ahead bias 방지)
        df_copy[self.date_col] = df_copy[self.date_col] + pd.offsets.MonthEnd(1)
        
        holdings = {}
        
        for date, date_group in df_copy.groupby(self.date_col):
            date_str = date.strftime('%Y-%m')
            holdings[date_str] = {}
            
            for port_num, port_group in date_group.groupby('portfolio'):
                port_num = int(port_num)
                
                if weighting == 'vw':
                    total_mktcap = port_group[self.weight_col].sum()
                    if total_mktcap > 0:
                        weights = (
                            port_group.set_index(self.ticker_col)[self.weight_col] / total_mktcap
                        ).to_dict()
                    else:
                        weights = {}
                else:  # ew
                    n_stocks = len(port_group)
                    if n_stocks > 0:
                        weights = {
                            ticker: 1.0 / n_stocks 
                            for ticker in port_group[self.ticker_col]
                        }
                    else:
                        weights = {}
                
                if return_weights:
                    holdings[date_str][port_num] = weights
                else:
                    holdings[date_str][port_num] = list(weights.keys())
        
        return holdings
    
    def get_bivariate_holdings(
        self,
        sort_var1: str,
        sort_var2: str,
        n_quantiles1: int = 5,
        n_quantiles2: int = 5,
        method: str = 'dependent',
        weighting: Union[str, Weighting] = 'vw'
    ) -> Dict[str, Dict[int, Dict[int, Dict[str, float]]]]:
        """
        2변량 포트폴리오 정렬 후 시점별 홀딩 반환
        
        Parameters
        ----------
        sort_var1 : str
            첫 번째 정렬 변수
        sort_var2 : str
            두 번째 정렬 변수
        n_quantiles1 : int, default 5
            첫 번째 분위수
        n_quantiles2 : int, default 5
            두 번째 분위수
        method : str, default 'dependent'
            정렬 방법
        weighting : str, default 'vw'
            가중 방식
        
        Returns
        -------
        Dict[str, Dict[int, Dict[int, Dict[str, float]]]]
            {날짜: {포트폴리오1: {포트폴리오2: {종목코드: 가중치}}}}
        """
        weighting = str(weighting)
        
        required_cols = [self.date_col, self.ticker_col, self.weight_col, sort_var1, sort_var2]
        df_copy = self.df[required_cols].copy().dropna()
        
        # 첫 번째 변수로 포트폴리오 할당
        df_copy['portfolio1'] = df_copy.groupby(self.date_col, group_keys=False).apply(
            lambda x: assign_portfolio_quantile(x, sort_var1, n_quantiles1)
        )
        
        # 두 번째 변수로 포트폴리오 할당
        if method == 'dependent':
            df_copy['portfolio2'] = df_copy.groupby(
                [self.date_col, 'portfolio1'], group_keys=False
            ).apply(
                lambda x: assign_portfolio_quantile(x, sort_var2, n_quantiles2)
            )
        else:
            df_copy['portfolio2'] = df_copy.groupby(self.date_col, group_keys=False).apply(
                lambda x: assign_portfolio_quantile(x, sort_var2, n_quantiles2)
            )
        
        # 다음 달로 이동
        df_copy[self.date_col] = df_copy[self.date_col] + pd.offsets.MonthEnd(1)
        
        holdings = {}
        
        for date, date_group in df_copy.groupby(self.date_col):
            date_str = date.strftime('%Y-%m')
            holdings[date_str] = {}
            
            for p1, p1_group in date_group.groupby('portfolio1'):
                p1 = int(p1)
                holdings[date_str][p1] = {}
                
                for p2, p2_group in p1_group.groupby('portfolio2'):
                    p2 = int(p2)
                    
                    if weighting == 'vw':
                        total_mktcap = p2_group[self.weight_col].sum()
                        if total_mktcap > 0:
                            weights = (
                                p2_group.set_index(self.ticker_col)[self.weight_col] / total_mktcap
                            ).to_dict()
                        else:
                            weights = {}
                    else:
                        n_stocks = len(p2_group)
                        if n_stocks > 0:
                            weights = {
                                ticker: 1.0 / n_stocks 
                                for ticker in p2_group[self.ticker_col]
                            }
                        else:
                            weights = {}
                    
                    holdings[date_str][p1][p2] = weights
        
        return holdings
    
    def triple_sort(
        self,
        sort_var1: str,
        sort_var2: str,
        sort_var3: str,
        n_q1: int = 3,
        n_q2: int = 3,
        n_q3: int = 3,
        method: str = 'dependent',
        weighting: Union[str, Weighting] = 'vw'
    ) -> pd.DataFrame:
        """
        3변량 포트폴리오 정렬 (Triple Sort)
        
        Parameters
        ----------
        sort_var1 : str
            첫 번째 정렬 변수
        sort_var2 : str
            두 번째 정렬 변수
        sort_var3 : str
            세 번째 정렬 변수
        n_q1, n_q2, n_q3 : int
            각 변수의 분위수
        method : str, default 'dependent'
            정렬 방법
        weighting : str, default 'vw'
            가중 방식
        
        Returns
        -------
        pd.DataFrame
            포트폴리오별 수익률 시계열
        """
        weighting = str(weighting)
        
        required_cols = [self.date_col, self.ticker_col, self.return_col,
                        self.weight_col, sort_var1, sort_var2, sort_var3]
        df_copy = self.df[required_cols].copy().dropna()
        
        # 종속 정렬
        df_copy['p1'] = df_copy.groupby(self.date_col, group_keys=False).apply(
            lambda x: assign_portfolio_quantile(x, sort_var1, n_q1)
        )
        
        if method == 'dependent':
            df_copy['p2'] = df_copy.groupby(
                [self.date_col, 'p1'], group_keys=False
            ).apply(
                lambda x: assign_portfolio_quantile(x, sort_var2, n_q2)
            )
            
            df_copy['p3'] = df_copy.groupby(
                [self.date_col, 'p1', 'p2'], group_keys=False
            ).apply(
                lambda x: assign_portfolio_quantile(x, sort_var3, n_q3)
            )
        else:
            df_copy['p2'] = df_copy.groupby(self.date_col, group_keys=False).apply(
                lambda x: assign_portfolio_quantile(x, sort_var2, n_q2)
            )
            df_copy['p3'] = df_copy.groupby(self.date_col, group_keys=False).apply(
                lambda x: assign_portfolio_quantile(x, sort_var3, n_q3)
            )
        
        # 다음 달로 이동
        portfolio_info = df_copy[[self.date_col, self.ticker_col, 'p1', 'p2', 'p3']].copy()
        portfolio_info[self.date_col] = portfolio_info[self.date_col] + pd.offsets.MonthEnd(1)
        
        df_analysis = pd.merge(
            self.df[[self.date_col, self.ticker_col, self.return_col, self.weight_col]],
            portfolio_info,
            on=[self.date_col, self.ticker_col],
            how='inner'
        ).dropna()
        
        if weighting == 'vw':
            portfolio_returns = df_analysis.groupby(
                [self.date_col, 'p1', 'p2', 'p3']
            ).apply(
                lambda x: np.average(x[self.return_col], weights=x[self.weight_col])
                if x[self.weight_col].sum() > 0 else np.nan
            )
        else:
            portfolio_returns = df_analysis.groupby(
                [self.date_col, 'p1', 'p2', 'p3']
            )[self.return_col].mean()
        
        return portfolio_returns
    
    def _calculate_portfolio_performance(
        self,
        portfolio_returns: pd.DataFrame,
        compute_alpha: bool = True,
        factor_cols: List[str] = ['mkt', 'smb', 'hml', 'mom'],
        newey_west_lags: int = 12
    ) -> pd.DataFrame:
        """포트폴리오별 성과 계산"""
        results = []
        
        for portfolio_name in portfolio_returns.columns:
            ret_series = portfolio_returns[portfolio_name].dropna()
            if len(ret_series) < 15:
                continue
            
            # 평균 수익률 및 t-stat
            mean_return = ret_series.mean() * 100
            std_return = ret_series.std() * 100
            n_obs = len(ret_series)
            mean_tstat = (mean_return / (std_return / np.sqrt(n_obs))) if std_return > 0 else 0
            
            res_dict = {
                'Portfolio': str(portfolio_name),
                'Mean Return (%)': mean_return,
                'Mean Return t-stat': mean_tstat,
                'N': n_obs
            }
            
            # 알파 계산
            if compute_alpha and self.factors_df is not None:
                alpha, alpha_tstat = self._compute_factor_alpha(
                    ret_series, factor_cols, newey_west_lags
                )
                res_dict['Alpha (%)'] = alpha
                res_dict['Alpha t-stat'] = alpha_tstat
            
            results.append(res_dict)
        
        return pd.DataFrame(results)
    
    def _compute_factor_alpha(
        self,
        returns: pd.Series,
        factor_cols: List[str],
        newey_west_lags: int = 12
    ) -> Tuple[float, float]:
        """팩터 알파 계산"""
        factors_date_col = self.date_col if self.date_col in self.factors_df.columns else 'date'
        
        merged = pd.merge(
            returns.rename('ret'),
            self.factors_df,
            left_index=True,
            right_on=factors_date_col
        ).dropna()
        
        if len(merged) < 20:
            return np.nan, np.nan
        
        available_factors = [f for f in factor_cols if f in merged.columns]
        if not available_factors:
            return np.nan, np.nan
        
        y = merged['ret']
        X = sm.add_constant(merged[available_factors])
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})
        alpha = model.params['const'] * 100
        alpha_tstat = model.tvalues['const']
        
        return alpha, alpha_tstat
    
    def _calculate_bivariate_alpha(
        self,
        portfolio_returns: pd.Series,
        n_q1: int,
        n_q2: int,
        factor_cols: List[str],
        newey_west_lags: int,
        inverse: bool
    ) -> Tuple[pd.DataFrame, float, float]:
        """2변량 정렬 알파 계산"""
        factors_date_col = self.date_col if self.date_col in self.factors_df.columns else 'date'
        
        # 각 포트폴리오 조합의 알파 계산
        def calc_alpha(returns):
            ret_series = returns.reset_index(level=[0, 1], drop=True)
            merged = pd.merge(
                ret_series.rename('ret'),
                self.factors_df,
                left_index=True,
                right_on=factors_date_col
            ).dropna()
            
            if len(merged) < 20:
                return np.nan
            
            available_factors = [f for f in factor_cols if f in merged.columns]
            if not available_factors:
                return np.nan
            
            y = merged['ret']
            X = sm.add_constant(merged[available_factors])
            model = sm.OLS(y, X).fit()
            return model.params['const'] * 100
        
        alpha_results = portfolio_returns.groupby(['portfolio1', 'portfolio2']).apply(calc_alpha)
        alpha_table = alpha_results.unstack(level='portfolio2')
        
        # 롱숏 포트폴리오
        portfolio_unstacked = portfolio_returns.unstack(level='portfolio2')
        high_p2 = portfolio_unstacked.columns.max()
        low_p2 = portfolio_unstacked.columns.min()
        
        if inverse:
            long_short_label = 'Low-High'
            alpha_table[long_short_label] = alpha_table[low_p2] - alpha_table[high_p2]
            hl_returns = portfolio_unstacked.groupby(level='portfolio1').apply(
                lambda x: x[low_p2] - x[high_p2]
            ).unstack(level='portfolio1')
        else:
            long_short_label = 'High-Low'
            alpha_table[long_short_label] = alpha_table[high_p2] - alpha_table[low_p2]
            hl_returns = portfolio_unstacked.groupby(level='portfolio1').apply(
                lambda x: x[high_p2] - x[low_p2]
            ).unstack(level='portfolio1')
        
        # 평균 롱숏 포트폴리오
        avg_hl_returns = hl_returns.mean(axis=1).dropna()
        
        # 평균 롱숏 알파 및 t-stat
        merged = pd.merge(
            avg_hl_returns.rename('ret'),
            self.factors_df,
            left_index=True,
            right_on=factors_date_col
        ).dropna()
        
        if len(merged) < 20:
            return alpha_table, np.nan, np.nan
        
        available_factors = [f for f in factor_cols if f in merged.columns]
        y = merged['ret']
        X = sm.add_constant(merged[available_factors])
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})
        
        final_alpha = model.params['const'] * 100
        hl_tstat = model.tvalues['const']
        
        alpha_table.loc[f'Avg. {long_short_label}'] = final_alpha
        
        return alpha_table, final_alpha, hl_tstat


# ============================================================================
# Convenience Functions
# ============================================================================

def univariate_sort(
    df: pd.DataFrame,
    factors_df: pd.DataFrame,
    sorting_variable: str,
    n_quantiles: int = 10,
    weighting: str = 'vw',
    inverse: bool = False,
    date_col: str = 'datetime'
) -> pd.DataFrame:
    """
    단일 변수 포트폴리오 정렬 (편의 함수)
    
    Parameters
    ----------
    df : pd.DataFrame
        분석 데이터
    factors_df : pd.DataFrame
        팩터 데이터
    sorting_variable : str
        정렬 변수
    n_quantiles : int, default 10
        분위수 개수
    weighting : str, default 'vw'
        가중 방식
    inverse : bool, default False
        롱숏 방향 반전
    date_col : str, default 'datetime'
        날짜 컬럼명
    
    Returns
    -------
    pd.DataFrame
        포트폴리오별 성과 테이블
    """
    sorter = PortfolioSorter(df, factors_df, date_col=date_col)
    return sorter.univariate_sort(
        sorting_variable=sorting_variable,
        n_quantiles=n_quantiles,
        weighting=weighting,
        inverse=inverse
    )


def bivariate_sort(
    df: pd.DataFrame,
    factors_df: pd.DataFrame,
    sort_var1: str,
    sort_var2: str,
    n_quantiles1: int = 5,
    n_quantiles2: int = 5,
    method: str = 'dependent',
    weighting: str = 'vw',
    inverse: bool = False,
    date_col: str = 'datetime'
) -> Tuple[pd.DataFrame, float, float]:
    """
    2변량 포트폴리오 정렬 (편의 함수)
    
    Parameters
    ----------
    df : pd.DataFrame
        분석 데이터
    factors_df : pd.DataFrame
        팩터 데이터
    sort_var1 : str
        첫 번째 정렬 변수
    sort_var2 : str
        두 번째 정렬 변수
    n_quantiles1 : int, default 5
        첫 번째 분위수
    n_quantiles2 : int, default 5
        두 번째 분위수
    method : str, default 'dependent'
        정렬 방법
    weighting : str, default 'vw'
        가중 방식
    inverse : bool, default False
        롱숏 방향 반전
    date_col : str, default 'datetime'
        날짜 컬럼명
    
    Returns
    -------
    Tuple[pd.DataFrame, float, float]
        (알파 테이블, 평균 롱숏 알파, t-통계량)
    """
    sorter = PortfolioSorter(df, factors_df, date_col=date_col)
    return sorter.bivariate_sort(
        sort_var1=sort_var1,
        sort_var2=sort_var2,
        n_quantiles1=n_quantiles1,
        n_quantiles2=n_quantiles2,
        method=method,
        weighting=weighting,
        inverse=inverse
    )
