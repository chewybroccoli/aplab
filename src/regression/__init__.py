"""aplab.regression - 회귀분석 모듈"""

from .fmr import (
    FamaMacBethRegression,
    FamaMacBethResult,
    fama_macbeth_regression,
    run_multiple_fmb_specifications
)

from .panel import (
    PanelRegression,
    PanelRegressionResult,
    panel_regression,
    run_multiple_panel_specifications,
    simple_panel_ols
)

__all__ = [
    # Fama-MacBeth
    "FamaMacBethRegression",
    "FamaMacBethResult",
    "fama_macbeth_regression",
    "run_multiple_fmb_specifications",
    
    # Panel
    "PanelRegression",
    "PanelRegressionResult",
    "panel_regression",
    "run_multiple_panel_specifications",
    "simple_panel_ols"
]
