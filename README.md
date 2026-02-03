# aplab - Asset Pricing Laboratory

Asset Pricing 연구를 위한 Python 라이브러리

## 설치

```bash
pip install -e .
```

추가 기능이 필요한 경우:

```bash
# Panel Regression 지원
pip install -e ".[panel]"

# Excel 출력 지원
pip install -e ".[excel]"

# 전체 기능
pip install -e ".[full]"
```

## 주요 기능

### 1. Portfolio Sorting

```python
from src import univariate_sort, bivariate_sort, PortfolioSorter

# Univariate Sort (단일 변수 정렬)
results = univariate_sort(
    df=data,
    factors_df=factors,
    sorting_variable='size',
    n_quantiles=10,        # decile
    weighting='vw',        # value-weighted
    inverse=False          # High-Low
)

# Bivariate Sort (2변량 정렬)
alpha_table, avg_alpha, t_stat = bivariate_sort(
    df=data,
    factors_df=factors,
    sort_var1='size',      # 통제 변수
    sort_var2='bm',        # 관심 변수
    n_quantiles1=5,        # quintile
    n_quantiles2=5,
    method='dependent',    # 종속 정렬
    weighting='vw'
)

# 클래스 기반 사용
sorter = PortfolioSorter(df, factors_df)
results = sorter.univariate_sort('MAX', n_quantiles=10)
results = sorter.bivariate_sort('size', 'MAX', method='dependent')
results = sorter.triple_sort('size', 'bm', 'mom')
```

### 2. Fama-MacBeth Regression

```python
from src import fama_macbeth_regression, FamaMacBethRegression

# 간편 함수
results = fama_macbeth_regression(
    df=data,
    dependent_var='ret_excess_lead',
    independent_vars=['size', 'bm', 'mom', 'illiq']
)

# 교차항 포함
results = fama_macbeth_regression(
    df=data,
    dependent_var='ret_excess_lead',
    independent_vars=['MAX', 'retail_ratio'],
    interaction_vars=[('MAX', 'retail_ratio')]
)

# 클래스 기반 사용
fmb = FamaMacBethRegression(
    df=data,
    dependent_var='ret_excess_lead',
    independent_vars=['size', 'bm', 'mom']
)
results = fmb.fit(newey_west_lags=12)
print(fmb.summary())
```

### 3. Panel Regression

```python
from src import panel_regression, PanelRegression

# 간편 함수
results = panel_regression(
    df=data,
    dependent_var='ret_excess_lead',
    independent_vars=['size', 'bm', 'mom'],
    entity_col='ticker',
    time_col='datetime',
    entity_effects=False,   # 개체 고정효과
    time_effects=True,      # 시간 고정효과
    cluster_entity=True,    # 이중 클러스터
    cluster_time=True
)

# 클래스 기반 사용
panel = PanelRegression(
    df=data,
    dependent_var='ret_excess_lead',
    independent_vars=['size', 'bm', 'mom']
)
results = panel.fit(time_effects=True)
```

### 4. Factor Regression

```python
from src import factor_regression, PerformanceAnalyzer

# 간편 함수
results = factor_regression(
    returns=portfolio_returns,
    factors_df=factors,
    model='carhart',        # 'capm', 'ff3', 'carhart', 'ff5'
    newey_west_lags=12
)

# 클래스 기반 사용
analyzer = PerformanceAnalyzer(factors_df)
results = analyzer.factor_regression(returns, model='carhart')
metrics = analyzer.calculate_performance_metrics(returns)
```

### 5. Summary Statistics

```python
from src import get_summary_and_correlation, winsorize

# 기술통계 및 상관관계
summary, corr = get_summary_and_correlation(
    df=data,
    variables=['ret', 'size', 'bm', 'mom', 'MAX']
)

# 윈저라이징
df = winsorize(df, columns=['ret', 'MAX'], limits=(0.01, 0.99))
```

### 6. Output Formatting

```python
from src import save_to_csv, TableFormatter, add_significance_stars

# CSV 저장
save_to_csv(results, 'output/results.csv', decimal=2)

# 포매터 사용
formatter = TableFormatter(decimal=2, include_stars=True)
formatter.save(results, 'output/results.csv')

# 유의성 별표
stars = add_significance_stars(p_value)  # '***', '**', '*', ''
```

## 라이브러리 구조

```
aplab/
├── src/
│   ├── __init__.py           # 메인 API
│   ├── base/
│   │   ├── __init__.py
│   │   └── types.py          # 타입 정의
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── base.py           # 포트폴리오 할당 함수
│   │   ├── sorting.py        # 포트폴리오 정렬
│   │   └── performance.py    # 성과 분석
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── fmr.py            # Fama-MacBeth
│   │   └── panel.py          # Panel Regression
│   ├── factors/
│   │   ├── __init__.py
│   │   └── builder.py        # 팩터 생성
│   └── utils/
│       ├── __init__.py
│       ├── stats.py          # 통계 유틸리티
│       └── format.py         # 출력 포맷팅
├── requirements.txt
├── setup.py
└── README.md
```

## 요구사항

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.9.0
- statsmodels >= 0.13.0
- linearmodels >= 4.27 (Panel Regression 사용 시)

## 사용 예시

```python
import pandas as pd
from src import (
    univariate_sort,
    fama_macbeth_regression,
    get_summary_and_correlation,
    save_to_csv
)

# 데이터 로드
df = pd.read_csv('data/master_panel.csv')
factors = pd.read_csv('data/factors.csv')

# 1. Summary Statistics
summary, corr = get_summary_and_correlation(
    df, 
    variables=['ret_excess', 'size', 'bm', 'MAX']
)

# 2. Portfolio Sort
results = univariate_sort(
    df=df,
    factors_df=factors,
    sorting_variable='MAX',
    n_quantiles=10,
    weighting='vw'
)
print(results)

# 3. Fama-MacBeth Regression
fmb_results = fama_macbeth_regression(
    df=df,
    dependent_var='ret_excess_lead',
    independent_vars=['MAX', 'size', 'bm', 'mom', 'illiq']
)
print(fmb_results)

# 4. 결과 저장
save_to_csv(results, 'output/portfolio_sort.csv')
save_to_csv(fmb_results, 'output/fmb_regression.csv')
```

## License

MIT License
