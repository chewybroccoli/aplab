"""
aplab.factors.builder
=====================
팩터 생성 유틸리티
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union


class FactorBuilder:
    """
    자산가격결정 팩터 생성 클래스
    
    Fama-French 스타일의 팩터 포트폴리오를 구성하고 팩터 수익률을 계산
    
    Examples
    --------
    >>> builder = FactorBuilder(
    ...     price_data=monthly_data,
    ...     accounting_data=annual_data,
    ...     date_col='datetime',
    ...     ticker_col='ticker'
    ... )
    >>> factors_df = builder.build_ff3_factors()
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        accounting_data: Optional[pd.DataFrame] = None,
        date_col: str = 'datetime',
        ticker_col: str = 'ticker',
        return_col: str = 'ret',
        rf_col: str = 'rf',
        mktcap_col: str = 'mktcap',
        mktcap_lag_col: str = 'mktcap_lag',
        exchange_col: Optional[str] = 'exchange'
    ):
        """
        Parameters
        ----------
        price_data : pd.DataFrame
            가격/수익률 데이터 (월별)
        accounting_data : pd.DataFrame, optional
            회계 데이터 (연간) - BE, OP, INV 등
        date_col : str, default 'datetime'
            날짜 컬럼명
        ticker_col : str, default 'ticker'
            종목 컬럼명
        return_col : str, default 'ret'
            수익률 컬럼명
        rf_col : str, default 'rf'
            무위험 수익률 컬럼명
        mktcap_col : str, default 'mktcap'
            시가총액 컬럼명
        mktcap_lag_col : str, default 'mktcap_lag'
            전기 시가총액 컬럼명
        exchange_col : str, optional
            거래소 컬럼명 (브레이크포인트 계산용)
        """
        self.price_data = price_data.copy()
        self.accounting_data = accounting_data.copy() if accounting_data is not None else None
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.return_col = return_col
        self.rf_col = rf_col
        self.mktcap_col = mktcap_col
        self.mktcap_lag_col = mktcap_lag_col
        self.exchange_col = exchange_col
        
        # 날짜 변환
        self.price_data[date_col] = pd.to_datetime(self.price_data[date_col])
        
        # 초과수익률 계산
        if rf_col in self.price_data.columns:
            self.price_data['ret_excess'] = self.price_data[return_col] - self.price_data[rf_col]
    
    def build_mkt_factor(self) -> pd.DataFrame:
        """
        시장 팩터 (MKT) 계산
        
        시가총액 가중 시장 수익률 - 무위험 수익률
        
        Returns
        -------
        pd.DataFrame
            MKT 팩터 시계열
        """
        df = self.price_data.dropna(subset=[self.return_col, self.mktcap_lag_col])
        
        if self.rf_col in df.columns:
            df = df.dropna(subset=[self.rf_col])
        
        # 시가총액 가중 시장 수익률
        mkt_ret = df.groupby(self.date_col).apply(
            lambda x: np.average(x[self.return_col], weights=x[self.mktcap_lag_col])
            if x[self.mktcap_lag_col].sum() > 0 else np.nan
        )
        
        # 무위험 수익률
        if self.rf_col in df.columns:
            rf_ret = df.groupby(self.date_col)[self.rf_col].mean()
            mkt_factor = mkt_ret - rf_ret
        else:
            mkt_factor = mkt_ret
        
        result = pd.DataFrame({'mkt': mkt_factor})
        result.index.name = self.date_col
        return result.reset_index()
    
    def build_smb_hml_factors(
        self,
        size_col: str = 'size',
        bm_col: str = 'bm',
        rebalance_month: int = 6,
        size_percentiles: List[float] = [0, 0.5, 1],
        bm_percentiles: List[float] = [0, 0.3, 0.7, 1]
    ) -> pd.DataFrame:
        """
        SMB (Small Minus Big)와 HML (High Minus Low) 팩터 계산
        
        Fama-French (1993) 방법론 기반
        
        Parameters
        ----------
        size_col : str, default 'size'
            Size 변수 컬럼명
        bm_col : str, default 'bm'
            Book-to-Market 변수 컬럼명
        rebalance_month : int, default 6
            리밸런싱 월 (6월)
        size_percentiles : List[float]
            Size 분위수 (기본: 중앙값 기준 2그룹)
        bm_percentiles : List[float]
            BM 분위수 (기본: 30/70 기준 3그룹)
        
        Returns
        -------
        pd.DataFrame
            SMB, HML 팩터 시계열
        """
        df = self.price_data.copy()
        
        # 6월 데이터에서 Size와 BM으로 포트폴리오 구성
        june_data = df[df[self.date_col].dt.month == rebalance_month].copy()
        june_data = june_data.dropna(subset=[size_col, bm_col])
        
        # 포트폴리오 할당
        def assign_portfolios(group):
            group = group.copy()
            
            # Size: 2그룹
            size_breaks = group[size_col].quantile(size_percentiles).values
            size_breaks[0], size_breaks[-1] = -np.inf, np.inf
            group['size_port'] = pd.cut(
                group[size_col],
                bins=size_breaks,
                labels=range(1, len(size_percentiles)),
                include_lowest=True
            )
            
            # BM: 3그룹
            bm_breaks = group[bm_col].quantile(bm_percentiles).values
            bm_breaks[0], bm_breaks[-1] = -np.inf, np.inf
            group['bm_port'] = pd.cut(
                group[bm_col],
                bins=bm_breaks,
                labels=range(1, len(bm_percentiles)),
                include_lowest=True
            )
            
            return group[[self.ticker_col, 'size_port', 'bm_port']]
        
        # 매년 6월 포트폴리오 구성
        june_data['year'] = june_data[self.date_col].dt.year
        portfolios = june_data.groupby('year').apply(assign_portfolios).reset_index(drop=True)
        portfolios = pd.merge(
            june_data[[self.date_col, self.ticker_col, 'year']],
            portfolios,
            on=[self.ticker_col]
        )
        
        # 포트폴리오 정보를 다음 해 7월부터 적용
        portfolios['form_year'] = portfolios['year']
        
        # 전체 데이터에 포트폴리오 매칭
        df['year'] = df[self.date_col].dt.year
        df['month'] = df[self.date_col].dt.month
        df['form_year'] = np.where(df['month'] < 7, df['year'] - 1, df['year'])
        
        merged = pd.merge(
            df,
            portfolios[[self.ticker_col, 'form_year', 'size_port', 'bm_port']],
            on=[self.ticker_col, 'form_year'],
            how='inner'
        )
        merged = merged.dropna(subset=['size_port', 'bm_port', 'ret_excess', self.mktcap_lag_col])
        
        # 2x3 포트폴리오 수익률 계산
        def calc_vw_return(x):
            if x[self.mktcap_lag_col].sum() > 0:
                return np.average(x['ret_excess'], weights=x[self.mktcap_lag_col])
            return np.nan
        
        port_returns = merged.groupby([self.date_col, 'size_port', 'bm_port']).apply(calc_vw_return)
        port_returns = port_returns.unstack(level=['size_port', 'bm_port'])
        
        # SMB: (Small/Low + Small/Medium + Small/High)/3 - (Big/Low + Big/Medium + Big/High)/3
        small_cols = [col for col in port_returns.columns if col[0] == 1]
        big_cols = [col for col in port_returns.columns if col[0] == 2]
        
        smb = port_returns[small_cols].mean(axis=1) - port_returns[big_cols].mean(axis=1)
        
        # HML: (Small/High + Big/High)/2 - (Small/Low + Big/Low)/2
        high_bm_cols = [col for col in port_returns.columns if col[1] == 3]
        low_bm_cols = [col for col in port_returns.columns if col[1] == 1]
        
        hml = port_returns[high_bm_cols].mean(axis=1) - port_returns[low_bm_cols].mean(axis=1)
        
        result = pd.DataFrame({
            'smb': smb,
            'hml': hml
        })
        result.index.name = self.date_col
        return result.reset_index()
    
    def build_momentum_factor(
        self,
        formation_period: Tuple[int, int] = (12, 2),
        holding_period: int = 1,
        percentiles: List[float] = [0, 0.3, 0.7, 1]
    ) -> pd.DataFrame:
        """
        모멘텀 팩터 (MOM/UMD) 계산
        
        Parameters
        ----------
        formation_period : Tuple[int, int], default (12, 2)
            모멘텀 계산 기간 (t-12 ~ t-2)
        holding_period : int, default 1
            보유 기간 (월)
        percentiles : List[float]
            분위수 (기본: 30/70 기준 3그룹)
        
        Returns
        -------
        pd.DataFrame
            MOM 팩터 시계열
        """
        df = self.price_data.copy()
        df = df.sort_values([self.ticker_col, self.date_col])
        
        # 모멘텀 시그널: t-12 ~ t-2 누적수익률
        start_lag, end_lag = formation_period
        
        def calc_momentum(group):
            group = group.sort_values(self.date_col)
            # 누적수익률 계산
            cumret = (1 + group[self.return_col]).rolling(window=start_lag - end_lag + 1).apply(
                lambda x: x.prod(), raw=True
            ) - 1
            # end_lag 만큼 shift
            group['mom_signal'] = cumret.shift(end_lag)
            return group
        
        df = df.groupby(self.ticker_col).apply(calc_momentum).reset_index(drop=True)
        df = df.dropna(subset=['mom_signal', 'ret_excess', self.mktcap_lag_col])
        
        # 월별 모멘텀 포트폴리오 구성
        def assign_mom_portfolio(group):
            breaks = group['mom_signal'].quantile(percentiles).values
            breaks[0], breaks[-1] = -np.inf, np.inf
            return pd.cut(
                group['mom_signal'],
                bins=breaks,
                labels=range(1, len(percentiles)),
                include_lowest=True
            )
        
        df['mom_port'] = df.groupby(self.date_col, group_keys=False).apply(
            lambda x: assign_mom_portfolio(x)
        )
        
        # 포트폴리오 수익률 계산
        def calc_vw_return(x):
            if x[self.mktcap_lag_col].sum() > 0:
                return np.average(x['ret_excess'], weights=x[self.mktcap_lag_col])
            return np.nan
        
        port_returns = df.groupby([self.date_col, 'mom_port']).apply(calc_vw_return).unstack()
        
        # MOM: Winners - Losers
        high_port = port_returns.columns.max()
        low_port = port_returns.columns.min()
        mom = port_returns[high_port] - port_returns[low_port]
        
        result = pd.DataFrame({'mom': mom})
        result.index.name = self.date_col
        return result.reset_index()
    
    def build_ff3_factors(self) -> pd.DataFrame:
        """
        Fama-French 3요인 모델 팩터 생성
        
        Returns
        -------
        pd.DataFrame
            MKT, SMB, HML 팩터 시계열
        """
        mkt = self.build_mkt_factor()
        smb_hml = self.build_smb_hml_factors()
        
        factors = pd.merge(mkt, smb_hml, on=self.date_col, how='outer')
        return factors
    
    def build_carhart_factors(self) -> pd.DataFrame:
        """
        Carhart 4요인 모델 팩터 생성
        
        Returns
        -------
        pd.DataFrame
            MKT, SMB, HML, MOM 팩터 시계열
        """
        ff3 = self.build_ff3_factors()
        mom = self.build_momentum_factor()
        
        factors = pd.merge(ff3, mom, on=self.date_col, how='outer')
        return factors
    
    def build_rmw_cma_factors(
        self,
        op_col: str = 'op',
        inv_col: str = 'inv',
        size_col: str = 'size',
        rebalance_month: int = 6,
        size_percentiles: List[float] = [0, 0.5, 1],
        char_percentiles: List[float] = [0, 0.3, 0.7, 1]
    ) -> pd.DataFrame:
        """
        RMW (Robust Minus Weak)와 CMA (Conservative Minus Aggressive) 팩터 계산
        
        Fama-French (2015) 5요인 모델의 수익성/투자 팩터
        
        Parameters
        ----------
        op_col : str, default 'op'
            Operating Profitability 컬럼명 (수익성)
        inv_col : str, default 'inv'
            Investment 컬럼명 (총자산 성장률)
        size_col : str, default 'size'
            Size 변수 컬럼명
        rebalance_month : int, default 6
            리밸런싱 월
        size_percentiles : List[float]
            Size 분위수
        char_percentiles : List[float]
            특성 분위수
        
        Returns
        -------
        pd.DataFrame
            RMW, CMA 팩터 시계열
        """
        df = self.price_data.copy()
        
        june_data = df[df[self.date_col].dt.month == rebalance_month].copy()
        
        # RMW (수익성 팩터)
        rmw_result = None
        if op_col in june_data.columns:
            june_data_op = june_data.dropna(subset=[size_col, op_col])
            rmw_result = self._build_2x3_factor(
                june_data_op, size_col, op_col, 
                size_percentiles, char_percentiles, 
                rebalance_month, factor_name='rmw'
            )
        
        # CMA (투자 팩터) - 투자가 낮은 것(Conservative)이 높은 것보다 높은 수익
        cma_result = None
        if inv_col in june_data.columns:
            june_data_inv = june_data.dropna(subset=[size_col, inv_col])
            cma_result = self._build_2x3_factor(
                june_data_inv, size_col, inv_col,
                size_percentiles, char_percentiles,
                rebalance_month, factor_name='cma', inverse=True
            )
        
        # 결과 병합
        if rmw_result is not None and cma_result is not None:
            result = pd.merge(rmw_result, cma_result, on=self.date_col, how='outer')
        elif rmw_result is not None:
            result = rmw_result
        elif cma_result is not None:
            result = cma_result
        else:
            result = pd.DataFrame()
        
        return result
    
    def _build_2x3_factor(
        self,
        june_data: pd.DataFrame,
        size_col: str,
        char_col: str,
        size_percentiles: List[float],
        char_percentiles: List[float],
        rebalance_month: int,
        factor_name: str,
        inverse: bool = False
    ) -> pd.DataFrame:
        """2x3 포트폴리오 기반 팩터 생성 헬퍼"""
        df = self.price_data.copy()
        
        def assign_portfolios(group):
            group = group.copy()
            
            size_breaks = group[size_col].quantile(size_percentiles).values
            size_breaks[0], size_breaks[-1] = -np.inf, np.inf
            group['size_port'] = pd.cut(
                group[size_col], bins=size_breaks,
                labels=range(1, len(size_percentiles)), include_lowest=True
            )
            
            char_breaks = group[char_col].quantile(char_percentiles).values
            char_breaks[0], char_breaks[-1] = -np.inf, np.inf
            group['char_port'] = pd.cut(
                group[char_col], bins=char_breaks,
                labels=range(1, len(char_percentiles)), include_lowest=True
            )
            
            return group[[self.ticker_col, 'size_port', 'char_port']]
        
        june_data['year'] = june_data[self.date_col].dt.year
        portfolios = june_data.groupby('year').apply(assign_portfolios).reset_index(drop=True)
        portfolios = pd.merge(
            june_data[[self.date_col, self.ticker_col, 'year']],
            portfolios, on=[self.ticker_col]
        )
        portfolios['form_year'] = portfolios['year']
        
        df['year'] = df[self.date_col].dt.year
        df['month'] = df[self.date_col].dt.month
        df['form_year'] = np.where(df['month'] < 7, df['year'] - 1, df['year'])
        
        merged = pd.merge(
            df, portfolios[[self.ticker_col, 'form_year', 'size_port', 'char_port']],
            on=[self.ticker_col, 'form_year'], how='inner'
        )
        merged = merged.dropna(subset=['size_port', 'char_port', 'ret_excess', self.mktcap_lag_col])
        
        def calc_vw_return(x):
            if x[self.mktcap_lag_col].sum() > 0:
                return np.average(x['ret_excess'], weights=x[self.mktcap_lag_col])
            return np.nan
        
        port_returns = merged.groupby([self.date_col, 'size_port', 'char_port']).apply(calc_vw_return)
        port_returns = port_returns.unstack(level=['size_port', 'char_port'])
        
        # High - Low 또는 Low - High
        high_cols = [col for col in port_returns.columns if col[1] == 3]
        low_cols = [col for col in port_returns.columns if col[1] == 1]
        
        if inverse:
            # CMA: Conservative(Low) - Aggressive(High)
            factor = port_returns[low_cols].mean(axis=1) - port_returns[high_cols].mean(axis=1)
        else:
            # RMW: Robust(High) - Weak(Low)
            factor = port_returns[high_cols].mean(axis=1) - port_returns[low_cols].mean(axis=1)
        
        result = pd.DataFrame({factor_name: factor})
        result.index.name = self.date_col
        return result.reset_index()
    
    def build_ff5_factors(
        self,
        size_col: str = 'size',
        bm_col: str = 'bm',
        op_col: str = 'op',
        inv_col: str = 'inv'
    ) -> pd.DataFrame:
        """
        Fama-French 5요인 모델 팩터 생성
        
        Parameters
        ----------
        size_col : str
            Size 컬럼명
        bm_col : str
            Book-to-Market 컬럼명
        op_col : str
            Operating Profitability 컬럼명
        inv_col : str
            Investment 컬럼명
        
        Returns
        -------
        pd.DataFrame
            MKT, SMB, HML, RMW, CMA 팩터 시계열
        """
        ff3 = self.build_ff3_factors()
        rmw_cma = self.build_rmw_cma_factors(op_col=op_col, inv_col=inv_col, size_col=size_col)
        
        factors = pd.merge(ff3, rmw_cma, on=self.date_col, how='outer')
        return factors
    
    def build_ff6_factors(
        self,
        size_col: str = 'size',
        bm_col: str = 'bm',
        op_col: str = 'op',
        inv_col: str = 'inv'
    ) -> pd.DataFrame:
        """
        Fama-French 5요인 + Momentum (6요인) 모델 팩터 생성
        
        Returns
        -------
        pd.DataFrame
            MKT, SMB, HML, RMW, CMA, MOM 팩터 시계열
        """
        ff5 = self.build_ff5_factors(size_col, bm_col, op_col, inv_col)
        mom = self.build_momentum_factor()
        
        factors = pd.merge(ff5, mom, on=self.date_col, how='outer')
        return factors
    
    def build_q_factors(
        self,
        me_col: str = 'me',
        ia_col: str = 'ia',
        roe_col: str = 'roe',
        rebalance_month: int = 6,
        size_percentiles: List[float] = [0, 0.5, 1],
        char_percentiles: List[float] = [0, 0.3, 0.7, 1]
    ) -> pd.DataFrame:
        """
        Hou, Xue, Zhang (2015) q-factor 모델 팩터 생성
        
        q-factor model: MKT, ME, I/A, ROE
        
        Parameters
        ----------
        me_col : str, default 'me'
            Market Equity (Size) 컬럼명
        ia_col : str, default 'ia'
            Investment-to-Assets 컬럼명
        roe_col : str, default 'roe'
            Return on Equity 컬럼명
        rebalance_month : int, default 6
            리밸런싱 월
        size_percentiles : List[float]
            Size 분위수
        char_percentiles : List[float]
            특성 분위수
        
        Returns
        -------
        pd.DataFrame
            MKT, R_ME, R_IA, R_ROE 팩터 시계열
        
        Notes
        -----
        - R_ME: Size factor (Small - Big)
        - R_IA: Investment factor (Low I/A - High I/A)
        - R_ROE: Profitability factor (High ROE - Low ROE)
        """
        # MKT
        mkt = self.build_mkt_factor()
        
        df = self.price_data.copy()
        june_data = df[df[self.date_col].dt.month == rebalance_month].copy()
        
        results = [mkt]
        
        # R_ME (Size factor): Small - Big
        if me_col in june_data.columns:
            june_data_me = june_data.dropna(subset=[me_col])
            r_me = self._build_size_factor(june_data_me, me_col, rebalance_month)
            if r_me is not None:
                results.append(r_me)
        
        # R_IA (Investment factor): Low - High (conservative - aggressive)
        if ia_col in june_data.columns and me_col in june_data.columns:
            june_data_ia = june_data.dropna(subset=[me_col, ia_col])
            r_ia = self._build_2x3_factor(
                june_data_ia, me_col, ia_col,
                size_percentiles, char_percentiles,
                rebalance_month, factor_name='r_ia', inverse=True
            )
            if r_ia is not None and not r_ia.empty:
                results.append(r_ia)
        
        # R_ROE (Profitability factor): High - Low
        if roe_col in june_data.columns and me_col in june_data.columns:
            june_data_roe = june_data.dropna(subset=[me_col, roe_col])
            r_roe = self._build_2x3_factor(
                june_data_roe, me_col, roe_col,
                size_percentiles, char_percentiles,
                rebalance_month, factor_name='r_roe', inverse=False
            )
            if r_roe is not None and not r_roe.empty:
                results.append(r_roe)
        
        # 병합
        if len(results) > 1:
            factors = results[0]
            for r in results[1:]:
                factors = pd.merge(factors, r, on=self.date_col, how='outer')
            return factors
        elif len(results) == 1:
            return results[0]
        else:
            return pd.DataFrame()
    
    def _build_size_factor(
        self,
        june_data: pd.DataFrame,
        size_col: str,
        rebalance_month: int,
        percentiles: List[float] = [0, 0.5, 1]
    ) -> pd.DataFrame:
        """Size factor (Small - Big) 생성"""
        df = self.price_data.copy()
        
        def assign_size_portfolio(group):
            group = group.copy()
            breaks = group[size_col].quantile(percentiles).values
            breaks[0], breaks[-1] = -np.inf, np.inf
            group['size_port'] = pd.cut(
                group[size_col], bins=breaks,
                labels=range(1, len(percentiles)), include_lowest=True
            )
            return group[[self.ticker_col, 'size_port']]
        
        june_data['year'] = june_data[self.date_col].dt.year
        portfolios = june_data.groupby('year').apply(assign_size_portfolio).reset_index(drop=True)
        portfolios = pd.merge(
            june_data[[self.date_col, self.ticker_col, 'year']],
            portfolios, on=[self.ticker_col]
        )
        portfolios['form_year'] = portfolios['year']
        
        df['year'] = df[self.date_col].dt.year
        df['month'] = df[self.date_col].dt.month
        df['form_year'] = np.where(df['month'] < 7, df['year'] - 1, df['year'])
        
        merged = pd.merge(
            df, portfolios[[self.ticker_col, 'form_year', 'size_port']],
            on=[self.ticker_col, 'form_year'], how='inner'
        )
        merged = merged.dropna(subset=['size_port', 'ret_excess', self.mktcap_lag_col])
        
        def calc_vw_return(x):
            if x[self.mktcap_lag_col].sum() > 0:
                return np.average(x['ret_excess'], weights=x[self.mktcap_lag_col])
            return np.nan
        
        port_returns = merged.groupby([self.date_col, 'size_port']).apply(calc_vw_return).unstack()
        
        # Small - Big
        r_me = port_returns[1] - port_returns[2]
        
        result = pd.DataFrame({'r_me': r_me})
        result.index.name = self.date_col
        return result.reset_index()


def calculate_factor_returns(
    long_portfolio: pd.Series,
    short_portfolio: pd.Series
) -> pd.Series:
    """
    롱-숏 팩터 수익률 계산
    
    Parameters
    ----------
    long_portfolio : pd.Series
        롱 포트폴리오 수익률
    short_portfolio : pd.Series
        숏 포트폴리오 수익률
    
    Returns
    -------
    pd.Series
        팩터 수익률
    """
    return long_portfolio - short_portfolio


def calculate_factor_premium(
    returns: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 12
) -> Dict[str, float]:
    """
    팩터 프리미엄 계산
    
    Parameters
    ----------
    returns : pd.Series
        팩터 수익률 시계열
    annualize : bool, default True
        연율화 여부
    periods_per_year : int, default 12
        연간 기간 수
    
    Returns
    -------
    Dict[str, float]
        평균 수익률, 표준편차, t-통계량, 샤프비율
    """
    import statsmodels.api as sm
    
    returns = returns.dropna()
    n = len(returns)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # t-통계량 (Newey-West)
    y = returns.values
    X = sm.add_constant(np.ones(len(y)))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    t_stat = model.tvalues[0]
    
    if annualize:
        mean_ret *= periods_per_year
        std_ret *= np.sqrt(periods_per_year)
    
    sharpe = mean_ret / std_ret if std_ret > 0 else np.nan
    
    return {
        'mean_return': mean_ret * 100,
        'std': std_ret * 100,
        't_stat': t_stat,
        'sharpe_ratio': sharpe,
        'n_obs': n
    }


def factor_correlation_matrix(
    factors_df: pd.DataFrame,
    factor_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    팩터 간 상관관계 행렬
    
    Parameters
    ----------
    factors_df : pd.DataFrame
        팩터 데이터프레임
    factor_cols : List[str], optional
        분석할 팩터 컬럼 리스트
    
    Returns
    -------
    pd.DataFrame
        상관관계 행렬
    """
    if factor_cols is None:
        # 기본 팩터 컬럼 추측
        potential_cols = ['mkt', 'smb', 'hml', 'rmw', 'cma', 'mom']
        factor_cols = [c for c in potential_cols if c in factors_df.columns]
    
    return factors_df[factor_cols].corr()


def factor_summary_statistics(
    factors_df: pd.DataFrame,
    factor_cols: Optional[List[str]] = None,
    date_col: str = 'datetime'
) -> pd.DataFrame:
    """
    팩터 요약 통계량
    
    Parameters
    ----------
    factors_df : pd.DataFrame
        팩터 데이터프레임
    factor_cols : List[str], optional
        분석할 팩터 컬럼 리스트
    date_col : str, default 'datetime'
        날짜 컬럼명
    
    Returns
    -------
    pd.DataFrame
        팩터별 요약 통계량
    """
    if factor_cols is None:
        potential_cols = ['mkt', 'smb', 'hml', 'rmw', 'cma', 'mom']
        factor_cols = [c for c in potential_cols if c in factors_df.columns]
    
    results = []
    for col in factor_cols:
        stats = calculate_factor_premium(factors_df[col])
        stats['factor'] = col
        results.append(stats)
    
    df = pd.DataFrame(results)
    df = df.set_index('factor')
    
    return df
