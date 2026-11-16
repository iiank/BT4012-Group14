# BT4012-Group14

## Setup

```
conda env create -f environment.yml
conda activate bt4012
```

## Pipeline

The whole pipeline is two notebooks in `notebook/`, run in order. Each notebook explains its steps and ends every section with a short chart of what it produced.

| Notebook | Sections | Time |
|---|---|---|
| `01_data_preprocessing.ipynb` | 1. Clean the tabular features · 2. Load the MD&A text · 3. Clean and tokenise the MD&A text | ~20 min |
| `02_feature_engineering.ipynb` | 1. Tabular features · 2. Sentiment + readability · 3. Embeddings · 4. TF-IDF per split | ~1 h, plus ~15 h for embeddings unless `data/processed/text_embeddings_raw/` already exists |

**For training you don't need to run either one for the text features:** those outputs are committed (see "MD&A text features" below). The tabular outputs are gitignored, so run notebook 01 section 1 and notebook 02 section 1 (under a minute together) to create them.

## Tabular features

### Data cleaning

Section 1 of `notebook/01_data_preprocessing.ipynb`. Reads the nine yearly `ListedCompanyFeatures772_<year>.csv` files and writes:

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

Section 1 of `notebook/02_feature_engineering.ipynb` (~30 s for all folds; needs `tabular_clean` from notebook 01 section 1).

Outputs are gitignored (too large for GitHub, ~105 MB and ~325 MB) — unlike the MD&A text files below, which are committed directly.

Reads `data/processed/tabular_clean` and writes two kinds of output, the same split as the MD&A text features (sentiment/readability vs. TF-IDF, sections 2 and 4 of the same notebook). Neither has a `Label` column — take `Label` from `tabular_clean`, same rule as the text files.

| Output | What's in it | Fitted? | Which split uses it |
|---|---|---|---|
| `data/processed/tabular_features/<year>.parquet` | 1,238 columns: deduped + log-transformed originals, YoY deltas, industry one-hot, missing-indicator flags | No — safe to share | **All splits** |
| `data/processed/tabular_features_folds/split_<s>/<year>.parquet` | Same 1,238 columns, imputed + scaled | Yes — median/`RobustScaler` fit on split `s`'s train rows only | **Only split `s`** |
| `data/processed/feature_engineering_log.csv` | Every dropped column, what it matched, the real correlation, and what actually survives | — | Reference only |

1. **Fold-independent** (nothing fitted, safe to share across every fold):
   - Drops near-duplicate columns with `|corr| > 0.95` within a feature group. Every drop is logged with the real pairwise correlation and the column that actually survives.
   - Tames monetary skew with `sign(x) * log1p(|x|)`.
   - Adds a year-over-year delta for every remaining numeric column (except the two geographic columns).
   - One-hot encodes `IndustryCodeC`.
   - Adds a `_was_missing` flag per incomplete column plus one shared `is_first_year_on_record` flag for every company's first row.

2. **Per fold** (fitted on that fold's training rows only, per `data/splits/temporal_splits.csv`):
   - Median imputation then `RobustScaler`, mirroring how the TF-IDF section fits its vocabulary and SVD on train rows only.

**To use it for split `s`**, merge on `nodeID` + `Year` (and pull in text the same way the MD&A section below shows):

```python
s = 0  # the split you're training on

tabular = pd.read_parquet(f"data/processed/tabular_features_folds/split_{s}")
splits = pd.read_csv("data/splits/temporal_splits.csv")

# splits already has Label and role (train/test) for this split — no need to fetch
# Label separately from tabular_clean
df = splits[splits["split"] == s].merge(tabular, on=["nodeID", "Year"])
```

**Rules:**
1. **Use the folder that matches the split**, same as the TF-IDF rule below: split 0 → `tabular_features_folds/split_0`, and so on. The imputation and scaling differ between folds because each is fit on that fold's own training years — mixing them leaks test data.
2. **There's no `Label` column in `tabular_features`/`tabular_features_folds` themselves.** Get it from `temporal_splits.csv` (as above) or `tabular_clean`.
3. **`tabular_features` (fold-independent) is for reference/EDA**, not for training directly — it isn't imputed or scaled. Train on the `tabular_features_folds/split_<s>` version.

Load a whole table at once with `pd.read_parquet("data/processed/tabular_features")` or `pd.read_parquet("data/processed/tabular_features_folds/split_<s>")` — both read a directory of year-files transparently. Verified end-to-end in `notebook/check_features.ipynb`, including an explicit check that each fold's imputer/scaler never saw that fold's test rows.

## MD&A text features

### Using the text features

**Nothing needs to be run.** The files are already in the repo; just pull `main`.

| File | What's in it | Rows | Which split uses it |
|---|---|---|---|
| `data/processed/text_features.parquet` | 8 columns: `has_text`, `is_truncated`, sentiment (`sent_pos`, `sent_neg`, `sent_tone`), readability (`read_sent_len`, `read_complex`, `read_advconj`) | Every company-year (33,664) | **All splits** |
| `data/processed/text_embeddings.parquet` | 64 columns: `emb_0` … `emb_63` (the meaning of each report as numbers) | Every company-year (33,664) | **All splits** |
| `data/processed/text_tfidf/split_<s>.parquet` | 100 columns: `tfidf_0` … `tfidf_99` (word-usage topics) | Only that split's train + test rows | **Only split `s`** |

**To add text to your data for split `s`**, merge on `nodeID` + `Year`:

```python
s = 0  # the split you're training on

text = (pd.read_parquet("data/processed/text_features.parquet")
        .merge(pd.read_parquet("data/processed/text_embeddings.parquet"), on=["nodeID", "Year"])
        .merge(pd.read_parquet(f"data/processed/text_tfidf/split_{s}.parquet"), on=["nodeID", "Year"]))

df = your_df.merge(text, on=["nodeID", "Year"], how="left")  # your_df = your rows for split s
```

That adds 172 text columns (8 + 64 + 100).

**Rules:**
1. **Use the TF-IDF file that matches the split:** split 0 → `split_0.parquet`, and so on. The numbers differ between files because each is fitted on its own split's training years. Mixing them leaks test data.
2. **There's no `Label` in these files.** Take `Label` from the tabular data.
3. **`has_text = 0` rows are all zeros on purpose.** These are the 516 company-years with no usable text, so don't impute them.
4. **`is_truncated = 1`** marks 3,409 reports cut off at Excel's 32,767-character limit in the source data. Treat it as an ordinary feature column.

### Rebuilding from the raw MD&A files

Run `notebook/01_data_preprocessing.ipynb`, then `notebook/02_feature_engineering.ipynb`, top to bottom. The text steps are:

| Notebook · section | Writes | Time |
|---|---|---|
| 01 · 2. Load the MD&A text | `mda_raw.parquet` | ~30 s |
| 01 · 3. Clean and tokenise | `mda_clean/<year>.parquet` | ~20 min |
| 02 · 2. Sentiment + readability | `text_features.parquet` | ~1 h |
| 02 · 3. Embeddings | `text_embeddings.parquet` | ~15 h on an M2 GPU (resumable), or seconds if `text_embeddings_raw/` exists |
| 02 · 4. TF-IDF per split | `text_tfidf/split_<s>.parquet` | ~6 min (needs `data/splits/temporal_splits.csv`) |

1. **Load** reads `data/<year>/MDA_<year>*.xlsx` (2021 and 2022 come in two parts) into one row per company-year. It checks the keys, duplicates and labels against `tabular_clean`.
2. **Clean and tokenise** writes `mda_clean/`:
   - Normalises characters and removes boilerplate: checkbox lines, "无" lines, the investor-visit table, numbered headings and the standard disclosure-exemption sentence.
   - Rejoins lines broken by the PDF conversion.
   - Flags `is_truncated`: 3,409 texts exactly 32,767 characters long, Excel's cell limit, so their endings were lost at the source.
   - Flags `has_text = 0`: texts under 50 characters, or with no Chinese characters.
   - Tokenises with jieba, keeping the sentiment words whole (the two lists in `data/sentiment/` must be present; the cell stops with an error otherwise). `words` is the full token list; `tokens` drops stopwords (`data/stopwords/hit_stopwords.txt`), punctuation and numbers.
3. **Sentiment + readability** writes `text_features.parquet`:
   - **Sentiment:** the share of positive and negative words from the Yao et al. (2021) financial word lists (`data/sentiment/`), with a negated word flipped (e.g. 不 + 增长 counts as negative). `sent_tone = (pos − neg) / (pos + neg)`.
   - **Readability:**
     - `read_sent_len`: median words per sentence. Reports with fewer than 5 sentence endings get the 2014 median.
     - `read_complex`: share of words with 3+ characters.
     - `read_advconj`: share of adverbs and conjunctions.
   - Ratios only, so cut-off reports aren't penalised for length.
4. **Embeddings** writes `text_embeddings.parquet`:
   - Embeds each whole report with `BAAI/bge-small-zh-v1.5`, in 510-token chunks averaged into one vector per report.
   - Compresses 512 → 64 columns with PCA fitted on 2014 only. 2014 is a training year in every split, so the result is the same for all splits with no leakage.
   - The uncompressed vectors are kept locally in `data/processed/text_embeddings_raw/` (gitignored). Years already embedded are skipped on a rerun.
5. **TF-IDF** writes `text_tfidf/split_<s>.parquet`:
   - For each split, fits TF-IDF on that split's training rows only: single words, `min_df=5`, `max_df=0.5`, 20,000 terms, sublinear term frequency.
   - Then fits TruncatedSVD to 100 columns, and applies both to the split's train and test rows.
   - Values differ between splits, so `tfidf_0` in split 0 is not the same topic as `tfidf_0` in split 3. Don't mix files or compare TF-IDF columns across splits.

`mda_raw.parquet` (832 MB), `mda_clean/` and `text_embeddings_raw/` are too large for GitHub and are gitignored; rebuild them with the notebooks.
