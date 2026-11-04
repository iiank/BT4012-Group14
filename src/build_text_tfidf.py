"""Build per-split MD&A TF-IDF features

For each walk-forward split in data/splits/temporal_splits.csv:
  1. fit TF-IDF on the `tokens` of that split's train rows only
  2. compress to N_COMPONENTS columns with TruncatedSVD, also fitted on train rows only
  3. transform the split's train and test rows
Writes data/processed/text_tfidf/split_<s>.parquet: nodeID, Year, tfidf_0 ... tfidf_99.

Values differ between splits (each split has its own vocabulary, IDF and SVD axes), so
these files are NOT part of the shared table: when training on split s, join split_<s>
only, and never compare tfidf_<k> across splits. Rows with has_text = 0 are all zeros.
Settings and reasons: docs/Findings&Issues.md #15.

Run from the repo root:  python src/build_text_tfidf.py        (all splits)
                         python src/build_text_tfidf.py 0      (some splits)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "mda_clean"
SPLITS = ROOT / "data" / "splits" / "temporal_splits.csv"
OUT_DIR = ROOT / "data" / "processed" / "text_tfidf"

MIN_DF = 5          # word must appear in at least 5 train reports (drops typos, one-off names)
MAX_DF = 0.5        # drop words in more than half the train reports (finance filler)
MAX_FEATURES = 20_000
N_COMPONENTS = 100
SEED = 0


def load_tokens() -> pd.DataFrame:
    years = range(2014, 2023)
    cols = ["nodeID", "Year", "has_text", "tokens"]
    return pd.concat([pd.read_parquet(CLEAN_DIR / f"{y}.parquet", columns=cols) for y in years], ignore_index=True)


def build_split(s: int, splits: pd.DataFrame, docs: pd.DataFrame) -> None:
    start = time.time()
    rows = splits.loc[splits["split"] == s, ["nodeID", "Year", "role"]]
    df = rows.merge(docs, on=["nodeID", "Year"], how="left", validate="one_to_one")
    assert df["tokens"].notna().all(), "split row without an MD&A row"
    train = (df["role"] == "train").to_numpy()
    test_year = int(df.loc[~train, "Year"].min())
    assert df.loc[train, "Year"].max() < test_year

    vec = TfidfVectorizer(
        analyzer=str.split, min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES,
        sublinear_tf=True, dtype=np.float32,
    )
    x_train = vec.fit_transform(df.loc[train, "tokens"])
    x_all = vec.transform(df["tokens"])
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED).fit(x_train)
    z = svd.transform(x_all).astype(np.float32)

    no_text = (df["has_text"] == 0).to_numpy()
    assert np.abs(z[no_text]).sum() == 0, "has_text = 0 rows should be all zeros"
    out = pd.concat(
        [df[["nodeID", "Year"]], pd.DataFrame(z, columns=[f"tfidf_{j}" for j in range(N_COMPONENTS)])], axis=1
    )
    path = OUT_DIR / f"split_{s}.parquet"
    out.to_parquet(path, index=False)
    print(
        f"split {s}: train {int(train.sum()):,} rows (to {test_year - 1}), test {int((~train).sum()):,} rows ({test_year})  "
        f"vocab {len(vec.vocabulary_):,}  SVD keeps {svd.explained_variance_ratio_.sum():.1%}  "
        f"{path.stat().st_size / 1e6:.1f} MB  {time.time() - start:.0f}s",
        flush=True,
    )


def main(split_ids: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS)
    docs = load_tokens()
    for s in split_ids or sorted(splits["split"].unique()):
        build_split(s, splits, docs)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]])
