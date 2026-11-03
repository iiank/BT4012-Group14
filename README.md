# BT4012-Group14

## Setup

```
conda env create -f environment.yml
conda activate bt4012
```

## Tabular features

### Data cleaning

```
python src/clean_tabular.py
```

Reads the nine yearly `ListedCompanyFeatures772_<year>.csv` files and writes:

- `data/processed/tabular_clean/<year>.parquet`: one row per company-year, with `nodeID`, `Year`, `Label` and 666 features.
- `data/processed/cleaning_log.csv`: every raw column with its description, whether it was kept or dropped, the reason, and its missing rate before and after.

Load all years at once:

```python
import pandas as pd
df = pd.read_parquet("data/processed/tabular_clean")
```



What the cleaning does:

- Checks that all years share the same columns, that `nodeID` + `Year` is unique and that labels are 0/1.
- Treats a blank financial statement line item as 0. The 2014-2015 files store unreported items as 0 while later files leave them blank; the share of non-zero values is the same in both, so they are made consistent.
- Fills 12 missing industry codes from the same company's nearest year.
- Drops 7 constant columns and 99 columns that are exact copies of another column.

Remaining missing values (about 2.5% of cells, mostly ratios that are undefined when the denominator is zero or negative) are left as NaN. Imputation, scaling and outlier handling must be fitted on training folds only, so they belong to the feature-engineering step.

### Feature engineering

```
python src/build_tabular_features.py        # all folds
python src/build_tabular_features.py 0      # a specific fold
```

Reads `data/processed/tabular_clean` and writes two kinds of output, the same split as the MD&A text pipeline (`src/build_text_features.py` vs. `src/build_text_tfidf.py`):

- **Fold-independent** (nothing fitted, safe to share across every fold) — `data/processed/tabular_features/<year>.parquet` and `data/processed/feature_engineering_log.csv`. Drops near-duplicate columns (`|corr| > 0.95` within a feature group; every drop logged with the real pairwise correlation and the column that actually survives), tames monetary skew with `sign(x) * log1p(|x|)`, adds a year-over-year delta for every remaining numeric column (except the two geographic columns), one-hot encodes `IndustryCodeC`, and adds a `_was_missing` flag per incomplete column plus one shared `is_first_year_on_record` flag for every company's first row.
- **Per fold** (fitted on that fold's training rows only, per `data/splits/temporal_splits.csv`) — `data/processed/tabular_features_folds/split_<s>/<year>.parquet`: median imputation then `RobustScaler`, mirroring how `build_text_tfidf.py` fits its vocabulary and SVD on train rows only.

Load either one with `pd.read_parquet("data/processed/tabular_features")` or `pd.read_parquet("data/processed/tabular_features_folds/split_<s>")` — both are split by year for the same reason `tabular_clean` is (GitHub's 100 MB file limit). Verified end-to-end in `notebook/check_features.ipynb`, including an explicit check that each fold's imputer/scaler never saw that fold's test rows.
