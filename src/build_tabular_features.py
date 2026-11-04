"""Feature engineering for cleaned tabular data

Reads data/processed/tabular_clean (from src/clean_tabular.py) and writes
  data/processed/tabular_features/<year>.parquet              fold-independent features
  data/processed/feature_engineering_log.csv                   every dropped column and why
  data/processed/tabular_features_folds/split_<s>/<year>.parquet   per-fold, imputed + scaled

Both outputs are split by year. Load a whole table at once
with  pd.read_parquet("data/processed/tabular_features")  or
      pd.read_parquet("data/processed/tabular_features_folds/split_<s>")

Fold-independent:
  1. drop near-duplicate columns (|corr| > 0.95 within a feature group; keep whichever
     survivor has the fewest missing values). Compares features to each other only,
     never Label, so this is safe to compute globally the same way clean_tabular.py's
     exact-duplicate check already does.
  2. sign(x) * log1p(|x|) on the 4 statement groups, to tame the heavy right tails found
     in notebook/eda_tabular.ipynb (e.g. monetary values spanning 0 to 6e11).
  3. year-over-year deltas for every remaining numeric column except the two geographic
     columns (RegisterLongitude/RegisterLatitude are fixed facts about a company, not an
     evolving quantity) — this company's own prior reported year, so no fitting and no
     cross-company information.
  4. one-hot IndustryCodeC.
  5. a <col>_was_missing indicator for every original column that has any NaN, plus one
     shared is_first_year_on_record flag for the YoY columns' first-year gaps (every YoY
     column is NaN in a company's first row for the same single reason, so one column
     replaces hundreds of near-duplicate per-column indicators).

Per fold (fitted on that fold's train rows only, per data/splits/temporal_splits.csv):
  median imputation, then RobustScaler. Mirrors src/build_text_tfidf.py's per-split
  fit/transform pattern.

Run from the repo root:  python src/build_tabular_features.py        (all splits)
                         python src/build_tabular_features.py 0      (some splits)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "tabular_clean"
LOG_PATH = ROOT / "data" / "processed" / "cleaning_log.csv"
SPLITS_PATH = ROOT / "data" / "splits" / "temporal_splits.csv"
OUT_DIR = ROOT / "data" / "processed"
FEATURES_DIR = OUT_DIR / "tabular_features"
FOLD_OUT_DIR = OUT_DIR / "tabular_features_folds"

KEYS = ["nodeID", "Year"]
MONETARY_GROUPS = {"balance sheet", "income statement", "cash flow (direct)", "cash flow (indirect)"}
CORR_THRESHOLD = 0.95


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_parquet(CLEAN_DIR)
    log = pd.read_csv(LOG_PATH)
    return clean, log


def drop_correlated(df: pd.DataFrame, cols: list[str], missing: pd.Series, group: str) -> tuple[list[str], list[dict]]:
    """Within one feature group, drop a column when |corr| > 0.95 with some still-surviving column. 
    Pairs are resolved strongest-first, and a pair is only acted on while both members are still alive, 
    so every logged drop reflects a real correlation between the two named columns. 
    Never a transitive chain through some third column."""
    if len(cols) < 2:
        return [], []

    corr = df[cols].corr().abs().to_numpy()
    n = len(cols)
    pairs = [
        (corr[i, j], i, j)
        for i in range(n) for j in range(i + 1, n)
        if corr[i, j] > CORR_THRESHOLD
    ]
    pairs.sort(key=lambda p: p[0], reverse=True)

    alive = [True] * n
    dropped, log_rows = [], []
    for r, i, j in pairs:
        if not (alive[i] and alive[j]):
            continue
        a, b = cols[i], cols[j]
        keep, drop, drop_idx = (a, b, j) if (missing[a], a) <= (missing[b], b) else (b, a, i)
        alive[drop_idx] = False
        dropped.append(drop)
        log_rows.append({
            "dropped_column": drop,
            "matched_column": keep,
            "correlation": round(float(r), 4),
            "group": group,
        })
    return dropped, log_rows


def engineer_fold_independent(clean: pd.DataFrame, log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_log = log[log["action"] == "kept"]
    groups = kept_log.set_index("column")["group"]
    missing = kept_log.set_index("column")["missing_clean"]
    numeric_cols = [c for c in kept_log["column"] if c != "IndustryCodeC"]

    df = clean[KEYS + numeric_cols + ["IndustryCodeC"]].copy()

    all_dropped, all_log_rows = [], []
    for group in groups[numeric_cols].unique():
        cols_g = [c for c in numeric_cols if groups[c] == group]
        dropped, log_rows = drop_correlated(df, cols_g, missing, group)
        all_dropped += dropped
        all_log_rows += log_rows
    df = df.drop(columns=all_dropped)
    surviving_numeric = [c for c in numeric_cols if c not in all_dropped]

    # A dropped column's matched_column can itself go on to be dropped by something even more complete  
    # Resolve each drop to the column that actually survives in the output, without touching the logged correlation
    matched_of = {row["dropped_column"]: row["matched_column"] for row in all_log_rows}
    for row in all_log_rows:
        survivor = row["matched_column"]
        while survivor in matched_of:
            survivor = matched_of[survivor]
        row["survivor_column"] = survivor
    assert set(row["survivor_column"] for row in all_log_rows) <= set(surviving_numeric)

    monetary_cols = [c for c in surviving_numeric if groups[c] in MONETARY_GROUPS]
    df[monetary_cols] = np.sign(df[monetary_cols]) * np.log1p(df[monetary_cols].abs())

    # Sort before deriving anything else (YoY, missing-indicators) that depends on row order or gets concatenated back by position
    df = df.sort_values(KEYS, ignore_index=True)

    # missing-indicators for the original columns before YoY adds its own structural NaNs
    incomplete = [c for c in surviving_numeric if df[c].isna().any()]
    indicators = df[incomplete].isna().astype(np.int8).add_suffix("_was_missing")

    # RegisterLongitude/RegisterLatitude are fixed facts about a company, not an evolving financial quantity 
    yoy_cols = [c for c in surviving_numeric if c not in ("RegisterLongitude", "RegisterLatitude")]
    yoy = df.groupby("nodeID")[yoy_cols].diff()
    yoy.columns = [f"{c}_yoy" for c in yoy_cols]
    
    is_first_year = (df.groupby("nodeID").cumcount() == 0).astype(np.int8)
    df = pd.concat([df, yoy, indicators, is_first_year.rename("is_first_year_on_record")], axis=1)

    industry = pd.get_dummies(df["IndustryCodeC"], prefix="industry", dtype=np.int8)
    df = pd.concat([df.drop(columns="IndustryCodeC"), industry], axis=1)

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    feat_log = pd.DataFrame(
        all_log_rows, columns=["dropped_column", "matched_column", "correlation", "survivor_column", "group"]
    )
    return df, feat_log


def build_fold(s: int, splits: pd.DataFrame, features: pd.DataFrame) -> None:
    start = time.time()
    rows = splits.loc[splits["split"] == s, ["nodeID", "Year", "role"]]
    df = rows.merge(features, on=["nodeID", "Year"], how="left", validate="one_to_one")
    industry_cols = [c for c in df.columns if c.startswith("industry_")]
    assert (df[industry_cols].sum(axis=1) == 1).all(), "split row without engineered features"

    train = (df["role"] == "train").to_numpy()
    test_year = int(df.loc[~train, "Year"].min())
    assert df.loc[train, "Year"].max() < test_year, "a train row is not before the test year"

    feature_cols = [c for c in df.columns if c not in ("nodeID", "Year", "role")]
    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(df.loc[train, feature_cols])
    x_all_imp = imputer.transform(df[feature_cols])

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_all_scaled = scaler.transform(x_all_imp).astype(np.float32)

    out = pd.concat(
        [df[["nodeID", "Year"]], pd.DataFrame(x_all_scaled, columns=feature_cols)], axis=1
    )
    
    split_dir = FOLD_OUT_DIR / f"split_{s}"
    split_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for year, rows_y in out.groupby("Year"):
        path = split_dir / f"{year}.parquet"
        rows_y.to_parquet(path, index=False, compression="zstd")
        total_bytes += path.stat().st_size
    print(
        f"split {s}: train {int(train.sum()):,} rows (to {test_year - 1}), test {int((~train).sum()):,} rows ({test_year})  "
        f"features {len(feature_cols):,}  {total_bytes / 1e6:.1f} MB  {time.time() - start:.0f}s",
        flush=True,
    )


def main(split_ids: list[int]) -> None:
    clean, log = load_clean()
    features, feat_log = engineer_fold_independent(clean, log)

    kept_cols = [c for c in features.columns if c not in KEYS]
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for year, rows_y in features.groupby("Year"):
        rows_y.to_parquet(FEATURES_DIR / f"{year}.parquet", index=False, compression="zstd")
    feat_log.to_csv(OUT_DIR / "feature_engineering_log.csv", index=False)
    print(f"tabular_features: {features.shape[0]:,} rows, {len(kept_cols):,} columns")
    print(f"dropped {len(feat_log)} near-duplicate columns (|corr| > {CORR_THRESHOLD})")
    print(f"wrote {FEATURES_DIR.relative_to(ROOT)}/ and feature_engineering_log.csv")

    splits = pd.read_csv(SPLITS_PATH)
    for s in split_ids or sorted(splits["split"].unique()):
        build_fold(s, splits, features)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]])
