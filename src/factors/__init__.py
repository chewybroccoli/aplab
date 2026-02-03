"""aplab.factors - 팩터 생성 및 분석"""

from .builder import (
    FactorBuilder,
    calculate_factor_returns,
    calculate_factor_premium,
    factor_correlation_matrix,
    factor_summary_statistics
)

__all__ = [
    "FactorBuilder",
    "calculate_factor_returns",
    "calculate_factor_premium",
    "factor_correlation_matrix",
    "factor_summary_statistics"
]

# FactorBuilder 지원 모델 목록
SUPPORTED_FACTOR_MODELS = {
    'ff3': 'Fama-French 3-Factor (MKT, SMB, HML)',
    'carhart': 'Carhart 4-Factor (MKT, SMB, HML, MOM)',
    'ff5': 'Fama-French 5-Factor (MKT, SMB, HML, RMW, CMA)',
    'ff6': 'Fama-French 6-Factor (MKT, SMB, HML, RMW, CMA, MOM)',
    'q': 'Hou-Xue-Zhang q-Factor (MKT, R_ME, R_IA, R_ROE)'
}
