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
