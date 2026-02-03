"""
aplab.base.types
================
공통 타입 정의 및 상수
"""
from typing import Literal, List, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd


# ============================================================================
# 가중 방식 (Weighting Method)
# ============================================================================
class Weighting(str, Enum):
    """포트폴리오 가중 방식"""
    EQUAL = "ew"           # Equal-weighted
    VALUE = "vw"           # Value-weighted (시가총액 기준)
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# 정렬 방식 (Sorting Method)
# ============================================================================
class SortingMethod(str, Enum):
    """다변량 정렬 시 방법"""
    INDEPENDENT = "independent"   # 독립 정렬 (각 변수별로 독립적으로 분위수 할당)
    DEPENDENT = "dependent"       # 종속 정렬 (첫 번째 변수 내에서 두 번째 변수 정렬)
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# 분위수 타입 (Quantile Types)
# ============================================================================
class QuantileType(str, Enum):
    """포트폴리오 분위수 타입"""
    DECILE = "decile"       # 10분위
    QUINTILE = "quintile"   # 5분위
    TERCILE = "tercile"     # 3분위
    QUARTILE = "quartile"   # 4분위
    CUSTOM = "custom"       # 사용자 정의
    
    @property
    def n_quantiles(self) -> int:
        """분위수 개수 반환"""
        mapping = {
            "decile": 10,
            "quintile": 5,
            "tercile": 3,
            "quartile": 4,
            "custom": -1  # 사용자 정의
        }
        return mapping[self.value]
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# 회귀분석 결과 데이터클래스
# ============================================================================
@dataclass
class RegressionResult:
    """회귀분석 결과를 담는 데이터클래스"""
    variable: str
    coefficient: float
    t_stat: float
    p_value: float
    std_error: Optional[float] = None
    
    @property
    def significance(self) -> str:
        """유의성 별표 반환"""
        if self.p_value < 0.01:
            return "***"
        elif self.p_value < 0.05:
            return "**"
        elif self.p_value < 0.10:
            return "*"
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Variable": self.variable,
            "Coefficient": self.coefficient,
            "t-stat": self.t_stat,
            "p-value": self.p_value,
            "Std Error": self.std_error
        }


@dataclass
class PortfolioPerformance:
    """포트폴리오 성과 결과를 담는 데이터클래스"""
    portfolio_name: str
    mean_return: float
    mean_return_tstat: float
    alpha: Optional[float] = None
    alpha_tstat: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    n_obs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Portfolio": self.portfolio_name,
            "Mean Return (%)": self.mean_return,
            "t-stat (Mean)": self.mean_return_tstat,
            "Alpha (%)": self.alpha,
            "t-stat (Alpha)": self.alpha_tstat,
            "Sharpe Ratio": self.sharpe_ratio,
            "N": self.n_obs
        }


# ============================================================================
# 기본 컬럼명 상수
# ============================================================================
class ColumnNames:
    """데이터프레임 컬럼명 상수"""
    # 날짜 관련
    DATE = "date"
    DATETIME = "datetime"
    YEAR_MONTH = "year_month"
    
    # 종목 관련
    TICKER = "ticker"
    
    # 수익률 관련
    RETURN = "ret"
    RETURN_EXCESS = "ret_excess"
    RETURN_EXCESS_LEAD = "ret_excess_lead"
    RF = "rf"
    
    # 시가총액
    MKTCAP = "mktcap"
    MKTCAP_LAG = "mktcap_lag"
    
    # 팩터
    MKT = "mkt"
    SMB = "smb"
    HML = "hml"
    MOM = "mom"
    
    # 포트폴리오
    PORTFOLIO = "portfolio"


# ============================================================================
# 타입 힌트 별칭
# ============================================================================
WeightingType = Literal["ew", "vw"]
SortMethodType = Literal["independent", "dependent"]
QuantileTypeStr = Literal["decile", "quintile", "tercile", "quartile", "custom"]
