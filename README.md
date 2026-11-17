# BT4012-Group14 · Financial fraud detection on FiGraph

## 1. Setup

```
conda env create -f environment.yml
conda activate bt4012
```

## 2. Run the pipeline

Open each notebook in `notebook/` and run all cells, in this order:

| Step | Notebook | Creates | Time |
|---|---|---|---|
| 1 | `01_data_preprocessing.ipynb` | `data/processed/tabular_clean/`, `mda_raw.parquet`, `mda_clean/` | ~20 min |
| 2 | `02_feature_engineering.ipynb` | `data/processed/tabular_features_folds/`, `text_features.parquet`, `text_embeddings.parquet`, `text_tfidf/` | ~1 h* |

\* On a fresh clone, step 2 takes ~15 h longer: the embedding model has to read every report (a GPU helps). The finished text files are already in the repo, so this is only needed to reproduce them from scratch.

`data/splits/temporal_splits.csv` is already in the repo, so step 2 can run straight after step 1.

### Quick start: only need the model inputs?

The text files and splits are already in the repo. Only the tabular files are missing (too large for GitHub), and they take under a minute to create:

1. Open `01_data_preprocessing.ipynb`.
2. Run the two setup cells at the top (imports and folder paths).
3. Run section 1 → creates `data/processed/tabular_clean/` (~5 s).
4. Open `02_feature_engineering.ipynb`.
5. Run the two setup cells at the top.
6. Run section 1 → creates `data/processed/tabular_features_folds/split_<s>/` (~30 s).

All model inputs are now ready. Go to section 4 to load them.

## 3. The model input files

All files are keyed on `nodeID` + `Year`, with at most one row per company-year. The shared files cover every company-year (33,664 rows); the per-split files cover only that split's train and test rows (e.g. split 0: 2014–2019, 19,494 rows).

| File (in `data/processed/`) | Columns | Which split uses it |
|---|---|---|
| `tabular_features_folds/split_<s>/` | 1,238 tabular features, imputed + scaled | **Only split `s`** |
| `text_features.parquet` | 8: `has_text`, `is_truncated`, 3 sentiment, 3 readability | All splits |
| `text_embeddings.parquet` | 64: `emb_0` … `emb_63` | All splits |
| `text_tfidf/split_<s>.parquet` | 100: `tfidf_0` … `tfidf_99` | **Only split `s`** |

`data/splits/temporal_splits.csv` says which rows belong to each split (`split` 0–3), whether each is `train` or `test`, and holds the `Label`.

Reference only (not model inputs): `tabular_features/` (before imputing and scaling), `cleaning_log.csv` and `feature_engineering_log.csv` (which columns were dropped and why).

## 4. Use the files for split `s`

```python
import pandas as pd

s = 0  # the split you're training on
rows = pd.read_csv("data/splits/temporal_splits.csv").query("split == @s")  # Label + train/test role

df = (rows
      .merge(pd.read_parquet(f"data/processed/tabular_features_folds/split_{s}"), on=["nodeID", "Year"])
      .merge(pd.read_parquet("data/processed/text_features.parquet"), on=["nodeID", "Year"])
      .merge(pd.read_parquet("data/processed/text_embeddings.parquet"), on=["nodeID", "Year"])
      .merge(pd.read_parquet(f"data/processed/text_tfidf/split_{s}.parquet"), on=["nodeID", "Year"]))

train, test = df[df.role == "train"], df[df.role == "test"]
X_train, y_train = train.drop(columns=["nodeID", "Year", "Label", "split", "role"]), train["Label"]
```

This gives 1,410 feature columns. For tabular-only models, keep only the first merge.

**Rules:**
1. **Match the split.** Split `s` uses `tabular_features_folds/split_<s>` and `text_tfidf/split_<s>` only. These are fitted on each split's training years, so mixing splits leaks test data.
2. **`Label` comes from `temporal_splits.csv`.** The feature files have none.
3. **Train on `tabular_features_folds/`, not `tabular_features/`.** The latter isn't imputed or scaled.
4. **Don't impute `has_text = 0` rows.** These 516 reports have no usable text, so their text columns are 0 on purpose.
5. **Check `is_truncated` with and without.** It mostly marks 2021–2022 reports rather than fraud.

## 5. How each feature is made

Each notebook explains its steps, with a chart after every section. The charts are also saved in `results/figures/` (`prep_*`, `fe_*`).
