"""Load the MD&A text for every listed company-year into one table

Reads data/<year>/MDA_<year>*.xlsx for 2014-2022 (2021 and 2022 are split
into part1/part2) and writes
  data/processed/mda_raw.parquet   one row per company-year

No text cleaning happens here; that is a separate step.

Run from the repo root:  python src/load_mda.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data"
OUT_DIR = RAW_DIR / "processed"
TABULAR_DIR = OUT_DIR / "tabular_clean"
YEARS = range(2014, 2023)

KEYS = ["nodeID", "Year", "Label"]
TEXT = "ManaDiscAnal"
COLUMNS = KEYS + [TEXT]
# Excel caps a cell at 32,767 characters; texts at this length were likely cut off at the source.
EXCEL_CELL_LIMIT = 32_767


def load_raw() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        files = sorted((RAW_DIR / str(year)).glob(f"MDA_{year}*.xlsx"))
        assert files, f"{year}: no MD&A file found"
        for path in files:
            frame = pd.read_excel(path)
            assert list(frame.columns) == COLUMNS, f"{path.name}: unexpected columns"
            assert (frame["Year"] == year).all(), f"{path.name}: Year column disagrees with file"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_keys(df: pd.DataFrame) -> None:
    assert df["nodeID"].str.fullmatch(r"L\d{6}").all(), "unexpected nodeID format"
    assert not df.duplicated(["nodeID", "Year"]).any(), "duplicate company-year rows"
    assert df["Label"].isin([0, 1]).all(), "Label must be 0/1"


def check_against_tabular(df: pd.DataFrame) -> None:
    tabular = pd.read_parquet(TABULAR_DIR, columns=KEYS)
    merged = df[KEYS].merge(tabular, on=["nodeID", "Year"], how="outer",
                            suffixes=("_mda", "_tab"), indicator=True)
    only_mda = (merged["_merge"] == "left_only").sum()
    only_tab = (merged["_merge"] == "right_only").sum()
    both = merged[merged["_merge"] == "both"]
    label_mismatch = (both["Label_mda"] != both["Label_tab"]).sum()
    print(f"vs tabular_clean: only in MD&A {only_mda}, only in tabular {only_tab}, "
          f"label mismatches {label_mismatch}")


def main() -> None:
    df = load_raw()
    check_keys(df)

    text = df[TEXT].astype("string")
    empty = text.isna() | (text.str.strip() == "")
    length = text.str.len()
    print(f"rows: {len(df):,}  companies: {df['nodeID'].nunique():,}")
    print(f"fraud rows: {df['Label'].sum():,} ({df['Label'].mean():.2%})")
    print(f"empty/missing text: {empty.sum():,}")
    print(f"text length (chars): median {int(length.median()):,}, max {int(length.max()):,}")
    print(f"texts at Excel cell limit ({EXCEL_CELL_LIMIT:,} chars): {(length == EXCEL_CELL_LIMIT).sum():,}")

    if TABULAR_DIR.exists():
        check_against_tabular(df)
    else:
        print("tabular_clean not found; run src/clean_tabular.py to cross-check keys")

    df[TEXT] = text
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mda_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
