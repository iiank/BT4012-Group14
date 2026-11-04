"""Build the fold-independent text features

Reads data/processed/mda_clean/<year>.parquet (from src/clean_mda.py) and writes
  data/processed/text_features.parquet   one row per company-year

Columns: nodeID, Year, has_text, is_truncated,
         sent_pos, sent_neg, sent_tone,                 (words; Yao et al. 2021 formal lists, negation-aware)
         read_sent_len, read_complex, read_advconj      (words / text_clean)
No Label (it comes from tabular_clean), and nothing is fitted, so one value per
company-year is valid for every fold. Rows with has_text = 0 get 0 in every score.
Rules and reasons: docs/Findings&Issues.md #12 and #13.

Run from the repo root:  python src/build_text_features.py            (all years)
                         python src/build_text_features.py 2014 2015  (partial run, writes text_features_partial.parquet)
"""

import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jieba.posseg as pseg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "mda_clean"
SENTIMENT_DIR = ROOT / "data" / "sentiment"
YEARS = range(2014, 2023)

NEGATIONS = {"不", "没有", "未", "无"}
NEGATION_WINDOW = 2
SENTENCE_END = {"。", "!", "?"}
COMPLEX_MIN_CHARS = 3
# Fewer sentence endings than this (e.g. punctuation lost in the source) means sentence
# length can't be measured. Such rows get the median of FILL_YEAR, which is in every
# walk-forward fold's training years, so the fill uses no validation or test data.
MIN_SENTENCE_ENDS = 5
FILL_YEAR = 2014
PUNCT_ONLY = re.compile(r"^[\W_]+$")
HAS_CJK = re.compile(r"[一-鿿]")

FEATURES = [
    "sent_pos", "sent_neg", "sent_tone",
    "read_sent_len", "read_complex", "read_advconj",
]


def load_words(name: str) -> set[str]:
    path = SENTIMENT_DIR / f"yao2021_formal_{name}.txt"
    return set(path.read_text(encoding="utf-8").split())


_POS, _NEG = load_words("pos"), load_words("neg")
# A word in both lists has no clear polarity, so it is ignored.
POSITIVE, NEGATIVE = _POS - _NEG, _NEG - _POS


def sentiment(tokens: list[str], total: int) -> tuple[float, float, float]:
    pos = neg = 0
    for i, word in enumerate(tokens):
        if word in POSITIVE:
            polarity = 1
        elif word in NEGATIVE:
            polarity = -1
        else:
            continue
        for j in range(i - 1, max(i - 1 - NEGATION_WINDOW, -1), -1):
            if PUNCT_ONLY.match(tokens[j]):
                break
            if tokens[j] in NEGATIONS:
                polarity = -polarity
                break
        pos += polarity == 1
        neg += polarity == -1
    tone = (pos - neg) / (pos + neg) if pos + neg else 0.0
    return pos / total, neg / total, tone


def sentence_lengths(tokens: list[str]) -> list[int]:
    lengths, count = [], 0
    for word in tokens:
        if word in SENTENCE_END:
            lengths.append(count)
            count = 0
        elif not PUNCT_ONLY.match(word):
            count += 1
    lengths.append(count)
    return [n for n in lengths if n > 0]


def adverb_conjunction_share(text: str) -> float:
    total = hits = 0
    for pair in pseg.lcut(text):
        if pair.flag.startswith("x") or not pair.word.strip():
            continue
        total += 1
        hits += pair.flag[0] in "cd"
    return hits / total if total else 0.0


def features(args: tuple[int, str, str]) -> tuple[float, ...]:
    has_text, words, text_clean = args
    if not has_text:
        return (0.0,) * len(FEATURES)
    tokens = words.split()
    content = [w for w in tokens if not PUNCT_ONLY.match(w)]
    if not content:
        return (0.0,) * len(FEATURES)
    pos, neg, tone = sentiment(tokens, len(content))
    cjk = [w for w in content if HAS_CJK.search(w)]
    complex_share = sum(len(w) >= COMPLEX_MIN_CHARS for w in cjk) / len(cjk) if cjk else 0.0
    ends = sum(w in SENTENCE_END for w in tokens)
    sent_len = float(np.median(sentence_lengths(tokens))) if ends >= MIN_SENTENCE_ENDS else np.nan
    return pos, neg, tone, sent_len, complex_share, adverb_conjunction_share(text_clean)


def main(years: list[int]) -> None:
    workers = max(1, (os.cpu_count() or 2) - 1)
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for year in years:
            start = time.time()
            df = pd.read_parquet(
                CLEAN_DIR / f"{year}.parquet",
                columns=["nodeID", "Year", "has_text", "is_truncated", "words", "text_clean"],
            )
            rows = zip(df["has_text"], df["words"].fillna(""), df["text_clean"].fillna(""))
            scores = pd.DataFrame(
                list(pool.map(features, rows, chunksize=16)), columns=FEATURES, index=df.index
            )
            frames.append(pd.concat([df[["nodeID", "Year", "has_text", "is_truncated"]], scores], axis=1))
            print(f"{year}: rows {len(df):,}  {time.time() - start:.0f}s")
    out = pd.concat(frames, ignore_index=True)
    unmeasurable = out["read_sent_len"].isna()
    print(f"read_sent_len unmeasurable (< {MIN_SENTENCE_ENDS} sentence endings): "
          f"{int(unmeasurable.sum())} of {int((out.has_text == 1).sum()):,} rows with text")
    print("  by year:", out[unmeasurable].groupby("Year").size().to_dict())
    base = out[(out.Year == FILL_YEAR) & (out.has_text == 1) & ~unmeasurable]
    if base.empty:
        print(f"  {FILL_YEAR} not in this run: left as NaN")
    else:
        fill = float(base["read_sent_len"].median())
        out.loc[unmeasurable, "read_sent_len"] = fill
        print(f"  filled with the {FILL_YEAR} median: {fill}")
    name = "text_features.parquet" if list(years) == list(YEARS) else "text_features_partial.parquet"
    path = ROOT / "data" / "processed" / name
    out.to_parquet(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}  {len(out):,} rows  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or list(YEARS))
