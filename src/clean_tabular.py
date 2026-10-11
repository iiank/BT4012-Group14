"""Clean the 772 listed-company financial features

Reads data/<year>/ListedCompanyFeatures772_<year>.csv for 2014-2022 and writes
  data/processed/tabular_clean/<year>.parquet   one row per company-year
  data/processed/cleaning_log.csv               what happened to every raw column

Only fold-independent cleaning happens here. Imputation, scaling, outlier
treatment and feature selection depend on the training window, so they are
left to the feature-engineering step and remaining gaps stay as NaN.

The output is split by year to stay under GitHub's 100 MB file limit. Load
all years at once with  pd.read_parquet("data/processed/tabular_clean").

Run from the repo root:  python src/clean_tabular.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data"
OUT_DIR = RAW_DIR / "processed"
YEARS = range(2014, 2023)

KEYS = ["nodeID", "Year", "Label"]
# Raw financial statement line items: A = balance sheet, B = income statement,
# C = cash flow statement (direct), D = cash flow statement (indirect).
STATEMENT_COL = re.compile(r"[A-D][0-9A-Za-z]{2}\d{7}")
STATEMENT_NAMES = {
    "A": "balance sheet",
    "B": "income statement",
    "C": "cash flow (direct)",
    "D": "cash flow (indirect)",
}


def load_raw() -> pd.DataFrame:
    frames = []
    columns = None
    for year in YEARS:
        frame = pd.read_csv(
            RAW_DIR / str(year) / f"ListedCompanyFeatures772_{year}.csv",
            low_memory=False,
        )
        if columns is None:
            columns = list(frame.columns)
        assert list(frame.columns) == columns, f"{year}: columns differ from 2014"
        assert (frame["Year"] == year).all(), f"{year}: Year column disagrees with file"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_keys(df: pd.DataFrame) -> None:
    assert df["nodeID"].str.fullmatch(r"L\d{6}").all(), "unexpected nodeID format"
    assert not df.duplicated(["nodeID", "Year"]).any(), "duplicate company-year rows"
    assert df["Label"].isin([0, 1]).all(), "Label must be 0/1"


def column_group(col: str) -> str:
    if STATEMENT_COL.fullmatch(col):
        return STATEMENT_NAMES[col[0]]
    if col == "IndustryCodeC":
        return "categorical"
    return "ratio / other"


def harmonise_statement_zeros(df: pd.DataFrame, statement_cols: list[str]) -> None:
    """Make "item not reported" mean 0 in every year.

    The 2014-2015 files (and ~8% of 2016) store an unreported statement line
    item as 0; later files leave it blank. The share of non-zero values is the
    same either way, so the blanks are zeros, not unknowns. A blank is only
    kept when the company has no values at all for that statement that year.
    """
    for prefix in STATEMENT_NAMES:
        cols = [c for c in statement_cols if c[0] == prefix]
        has_statement = df[cols].notna().any(axis=1)
        df.loc[has_statement, cols] = df.loc[has_statement, cols].fillna(0.0)


def fill_industry(df: pd.DataFrame) -> int:
    """Fill a missing industry code from the same company's nearest year."""
    missing = int(df["IndustryCodeC"].isna().sum())
    df["IndustryCodeC"] = df.groupby("nodeID")["IndustryCodeC"].transform(
        lambda s: s.ffill().bfill()
    )
    return missing


def find_redundant(df: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """Return {column: reason} for constant and exactly duplicated columns."""
    reasons = {}
    seen = {}
    for col in features:
        values = df[col]
        if values.nunique(dropna=True) <= 1:
            reasons[col] = "dropped: constant"
            continue
        if values.dtype.kind != "f":
            continue
        # + 0.0 turns -0.0 into 0.0 so the byte comparison treats them alike
        signature = (values.fillna(np.inf).to_numpy() + 0.0).tobytes()
        if signature in seen:
            reasons[col] = f"dropped: identical to {seen[signature]}"
        else:
            seen[signature] = col
    return reasons


def main() -> None:
    df = load_raw()
    check_keys(df)

    features = [c for c in df.columns if c not in KEYS]
    numeric = [c for c in features if c != "IndustryCodeC"]
    statement_cols = [c for c in numeric if STATEMENT_COL.fullmatch(c)]

    assert all(df[c].dtype.kind == "f" for c in numeric), "non-numeric feature column"
    assert not np.isinf(df[numeric].to_numpy()).any(), "infinite values present"

    missing_before = df[features].isna().mean()

    harmonise_statement_zeros(df, statement_cols)
    industry_filled = fill_industry(df)
    dropped = find_redundant(df, features)

    descriptions = pd.read_excel(RAW_DIR / "FeaturesDescription.xlsx").set_index(
        "Feature Name"
    )["Description.1"]
    log = pd.DataFrame(
        {
            "column": features,
            "group": [column_group(c) for c in features],
            "action": [dropped.get(c, "kept") for c in features],
            "missing_raw": missing_before.round(4).to_numpy(),
            "missing_clean": df[features].isna().mean().round(4).to_numpy(),
            "description": descriptions.reindex(features).to_numpy(),
        }
    )

    kept = [c for c in features if c not in dropped]
    clean = df[KEYS + kept].sort_values(["Year", "nodeID"], ignore_index=True)
    clean = clean.astype({"Year": "int16", "Label": "int8"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUT_DIR / "tabular_clean"
    table_dir.mkdir(exist_ok=True)
    for year, rows in clean.groupby("Year"):
        rows.to_parquet(table_dir / f"{year}.parquet", index=False, compression="zstd")
    log.to_csv(OUT_DIR / "cleaning_log.csv", index=False)

    n_constant = sum(r == "dropped: constant" for r in dropped.values())
    print(f"rows: {len(clean):,}  companies: {clean['nodeID'].nunique():,}")
    print(f"fraud rows: {int(clean['Label'].sum()):,} ({clean['Label'].mean():.2%})")
    print(f"raw features: {len(features)}  kept: {len(kept)}")
    print(f"  dropped constant: {n_constant}")
    print(f"  dropped exact duplicates: {len(dropped) - n_constant}")
    print(f"industry codes filled from another year: {industry_filled}")
    print(
        "missing cells: "
        f"{missing_before.mean():.1%} raw -> {clean[kept].isna().mean().mean():.1%} clean"
    )
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/tabular_clean/ and cleaning_log.csv")


if __name__ == "__main__":
    main()
