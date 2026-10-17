"""Clean and tokenise the MD&A text

Reads data/processed/mda_raw.parquet (from src/load_mda.py) and writes
  data/processed/mda_clean/<year>.parquet   one row per company-year

Columns: nodeID, Year, Label, has_text, is_truncated, text_clean, words, tokens
  text_clean  cleaned text, numbers and stopwords kept (sentiment, readability, embeddings)
  words       jieba output of text_clean, space-joined, nothing removed
  tokens      words minus stopwords, punctuation and numbers, space-joined (TF-IDF only)

Every step looks at one document at a time, so nothing here depends on the folds.
Rules and reasons: docs/Findings&Issues.md #9.

Run from the repo root:  python src/clean_mda.py            (all years)
                         python src/clean_mda.py 2014 2015  (some years)
"""

import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jieba
import pandas as pd

jieba.setLogLevel(60)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "processed" / "mda_raw.parquet"
OUT_DIR = ROOT / "data" / "processed" / "mda_clean"
STOPWORDS_FILE = ROOT / "data" / "stopwords" / "hit_stopwords.txt"
# Yao et al. (2021) formal financial sentiment words, added to jieba so they stay whole
SENTIMENT_FILES = sorted((ROOT / "data" / "sentiment").glob("yao2021_formal_*.txt"))
YEARS = range(2014, 2023)

EXCEL_CELL_LIMIT = 32_767
MIN_CHARS = 50

STOPWORDS = {
    unicodedata.normalize("NFKC", line).strip()
    for line in STOPWORDS_FILE.read_text(encoding="utf-8").splitlines()
} - {""}

for path in SENTIMENT_FILES:
    jieba.load_userdict(str(path))

CJK = r"一-鿿"
SPACE_NEXT_TO_CJK = re.compile(rf"(?<=[{CJK}])[ \t]+|[ \t]+(?=[{CJK}])")
DOT_RUN = re.compile(r"\.{4,}")
# B4: standard disclosure-exemption sentence. PDF line breaks can fall anywhere inside it.
EXEMPTION = re.compile(
    r"(?:[一二三四五六七八九十]+、\s*)?"
    + r"\s*".join("公司因不适用准则规定")
    + r"[\s\S]{0,80}?"
    + r"\s*".join("原因说明")
)
CHECKBOX = re.compile(r"[√□☑]")
TOP_HEADING = re.compile(r"^[一二三四五六七八九十]+、")
HEADING = re.compile(r"^(?:[一二三四五六七八九十]+、|\([一二三四五六七八九十\d]+\)|\d{1,2}[、.](?!\d))")
SENTENCE_PUNCT = re.compile(r"[。!?;]")
LINE_END = re.compile(r"[。!?;:][”\"’')\]]*$")
SENTENCE_END = re.compile(r"[。!?][”\"’')\]]*")
PUNCT_ONLY = re.compile(r"^[\W_]+$")
NUMBER = re.compile(r"^[\d.,%]+$")
MAX_HEADING_CHARS = 40


def is_visit_table_heading(line: str) -> bool:
    # B3: the heading itself, not narrative that happens to mention investor visits
    return "接待调研" in line and len(line) < 60 and not SENTENCE_PUNCT.search(line)


def is_heading(line: str) -> bool:
    # B2: short numbered line with no sentence punctuation
    return bool(HEADING.match(line)) and len(line) <= MAX_HEADING_CHARS and not SENTENCE_PUNCT.search(line)


def strip_boilerplate(lines: list[str]) -> list[str]:
    kept, skipping_visits = [], False
    for line in lines:
        if skipping_visits:
            if not TOP_HEADING.match(line):
                continue
            skipping_visits = False
        if is_visit_table_heading(line):
            skipping_visits = True
            continue
        if CHECKBOX.search(line) or line == "无" or is_heading(line):
            continue
        kept.append(line)
    return kept


def rejoin(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if not out or out.endswith("\n"):
            out += line
        elif out[-1].isascii() and out[-1].isalnum() and line[0].isascii() and line[0].isalnum():
            out += " " + line
        else:
            out += line
        if LINE_END.search(line):
            out += "\n"
    return out.strip()


def clean(text: str, truncated: bool) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = DOT_RUN.sub(" ", text)
    text = SPACE_NEXT_TO_CJK.sub("", text)
    text = EXEMPTION.sub("", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = strip_boilerplate([line for line in lines if line])
    text = rejoin(lines)
    if truncated:
        ends = list(SENTENCE_END.finditer(text))
        if ends:
            text = text[: ends[-1].end()]
    return text


def process(text: object) -> tuple[int, int, str, str, str]:
    if not isinstance(text, str):
        return 0, 0, "", "", ""
    truncated = int(len(text) == EXCEL_CELL_LIMIT)
    text_clean = clean(text, bool(truncated))
    if len(text_clean) < MIN_CHARS:
        return 0, truncated, "", "", ""
    words = [w for w in jieba.lcut(text_clean) if w.strip()]
    tokens = [
        w for w in words
        if w not in STOPWORDS and not PUNCT_ONLY.match(w) and not NUMBER.match(w)
    ]
    return 1, truncated, text_clean, " ".join(words), " ".join(tokens)


def main(years: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for year in years:
            start = time.time()
            df = pd.read_parquet(RAW, filters=[("Year", "==", year)])
            results = list(pool.map(process, df["ManaDiscAnal"], chunksize=16))
            cols = ["has_text", "is_truncated", "text_clean", "words", "tokens"]
            out = df[["nodeID", "Year", "Label"]].copy()
            for i, col in enumerate(cols):
                out[col] = [r[i] for r in results]
            out[["has_text", "is_truncated"]] = out[["has_text", "is_truncated"]].astype("int8")
            out[["text_clean", "words", "tokens"]] = out[["text_clean", "words", "tokens"]].astype("string")
            path = OUT_DIR / f"{year}.parquet"
            out.to_parquet(path, index=False)
            print(
                f"{year}: rows {len(out):,}  has_text=0 {int((out.has_text == 0).sum()):,}  "
                f"is_truncated {int(out.is_truncated.sum()):,}  "
                f"median chars raw->clean {int(df.ManaDiscAnal.str.len().median()):,}->"
                f"{int(out.text_clean.str.len()[out.has_text == 1].median()):,}  "
                f"{path.stat().st_size / 1e6:.0f} MB  {time.time() - start:.0f}s"
            )


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or list(YEARS))
