# BT4012 Group 14 · Financial fraud detection on FiGraph

Detecting financial statement fraud among Chinese listed companies (2014 to 2022) with tabular, text and graph features. Models: logistic regression, random forest, XGBoost, an MLP, an R-GCN and an MLP + R-GCN ensemble.

Dataset: [FiGraph](https://github.com/XiaoguangWang23/FiGraph).

## 1. Repository structure

Items marked *(not in repo)* are too large for GitHub. The pipeline creates them when you run it (Section 3).

```text
BT4012-Group14/
├── environment.yml                                 conda environment "bt4012"
├── data/
│   ├── 2014/ … 2022/                               raw FiGraph files, one folder per year
│   │   ├── ListedCompanyFeatures772_<year>.csv     financial features and fraud label
│   │   ├── MDA_<year>.xlsx                         MD&A report text (2021 and 2022 come in two parts)
│   │   └── edges<year>.csv                         raw links between companies and other entities
│   ├── FeaturesDescription.xlsx                    data dictionary for the financial features
│   ├── sentiment/                                  finance sentiment word lists (Yao et al., 2021)
│   ├── stopwords/                                  HIT Chinese stopword list
│   ├── splits/
│   │   └── temporal_splits.csv                     walk-forward splits, roles and labels
│   └── processed/
│       ├── tabular_clean/                          cleaned financial features (not in repo)
│       ├── mda_raw.parquet, mda_clean/             raw and cleaned MD&A text (not in repo)
│       ├── cleaning_log.csv                        which columns were dropped and why
│       ├── graph/
│       │   ├── company_edges.parquet               company-to-company links by relation and year
│       │   └── degree_features.csv                 link counts per company-year
│       ├── tabular_features/                       engineered financial features (not in repo)
│       ├── tabular_features_folds/                 the same, imputed and scaled per split (not in repo)
│       ├── feature_engineering_log.csv             feature engineering decisions
│       ├── text_features.parquet                   sentiment and readability features
│       ├── text_embeddings.parquet                 64-value text embeddings
│       └── text_tfidf/split_<s>.parquet            TF-IDF features per split
├── notebook/
│   ├── 01_data_preprocessing.ipynb                 clean the tables and the MD&A text
│   ├── graph_preprocessing.ipynb                   build the company graph and the splits
│   ├── 02_feature_engineering.ipynb                tabular, sentiment, embedding and TF-IDF features
│   ├── model.ipynb                                 MLP, R-GCN and ensemble
│   ├── model_classical.ipynb                       logistic regression, random forest, XGBoost
│   ├── model_comparison_&_misclassification.ipynb  compares all models, error analysis
│   └── eda_tabular.ipynb                           optional exploratory analysis
└── results/
    ├── figures/                                    charts from the preprocessing, feature and model notebooks
    ├── rgcn_results.csv                            test metrics per model and year
    ├── rgcn_predictions.csv                        test-year scores per company
    ├── confusion_matrices.csv                      top 5% confusion matrices
    ├── topk_precision_recall.csv                   precision and recall at the top 1%, 5% and 10%
    ├── feature_set_comparison.csv                  tabular only vs tabular + text
    ├── rgcn_embeddings/split_<s>.parquet           R-GCN hidden vectors per split (saved, not used later)
    ├── classical_results.csv                       classical model metrics
    └── classical_predictions.csv                   classical model scores
```

Sources for the two word lists in `data/`:

- `sentiment/`: [finance sentiment dictionary from cntext](https://github.com/hiDaDeng/cntext/blob/main/cntext/io/data/zh_common_FinacialFormalUnformal.yaml) (Yao et al., 2021)
- `stopwords/`: [HIT stopword list](https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt)

## 2. Setup

### 2.1 Clone the repo

```bash
git clone https://github.com/iiank/BT4012-Group14.git
cd BT4012-Group14-main
```

### 2.2 Create the environment

```bash
conda env create -f environment.yml
conda activate bt4012
```

**GPU note.** `environment.yml` installs PyTorch built for CUDA 13.2 (the `cu132` line). On a machine with a different CUDA version, or with no NVIDIA GPU, change that line before creating the environment, for example to `https://download.pytorch.org/whl/cpu`. All notebooks also run on a CPU. Our results were produced on an NVIDIA RTX 4060 Laptop GPU (8 GB).

### 2.3 Run in VS Code

Open the repo's root folder (BT4012-Group14) in VS Code and run the notebooks with the bt4012 kernel.

## 3. Run the pipeline

Run the notebooks in this order. For each one, use **Run All** unless the table says otherwise.

| Step | Notebook | Needs | Creates |
|---|---|---|---|
| 1 | `01_data_preprocessing.ipynb` | raw files in `data/<year>/` | `tabular_clean/`, `mda_raw.parquet`, `mda_clean/`, `cleaning_log.csv` |
| 2 | `graph_preprocessing.ipynb` | step 1, raw `edges<year>.csv` | `temporal_splits.csv`, `graph/company_edges.parquet`, `graph/degree_features.csv` |
| 3 | `02_feature_engineering.ipynb` | steps 1 and 2 | `tabular_features/`, `tabular_features_folds/`, `text_features.parquet`, `text_embeddings.parquet`, `text_tfidf/` |
| 4 | `model.ipynb` | steps 2 and 3 | `results/rgcn_*.csv`, model figures, R-GCN hidden vectors |
| 5 | `model_classical.ipynb` | steps 3 and 4 | `results/classical_*.csv` |
| 6 | `model_comparison_&_misclassification.ipynb` | steps 1 to 5 | comparison tables and error analysis |

Step 5 runs after step 4 because it checks that its results table has the same columns as `results/rgcn_results.csv`.

### Option A: reproduce the results

The graph files, splits and text features are already in the repo. Only the large tabular files are missing:

1. Open `01_data_preprocessing.ipynb`, run the two setup cells at the top, then run **section 1**. This creates `data/processed/tabular_clean/`.
2. Open `02_feature_engineering.ipynb`, run the two setup cells at the top, then run **section 1**. This creates `tabular_features/` and `tabular_features_folds/split_<s>/`.
3. Run steps 4 to 6 in order with **Run All**.

### Option B: rebuild everything from the raw data

1. Delete these three folders, so that nothing from an earlier run is reused:
   - `data/processed/`
   - `data/splits/`
   - `results/`

   Keep everything else in `data/`: the year folders, `FeaturesDescription.xlsx`, `sentiment/` and `stopwords/` are the raw inputs. The notebooks recreate the three deleted folders when they write their outputs.

2. Run steps 1 to 6 in order with **Run All**.

The text embeddings in section 3 of `02_feature_engineering.ipynb` are the slowest part of the pipeline, and a GPU speeds them up considerably.

### Optional

- `eda_tabular.ipynb` explores the cleaned financial features. Run it any time after step 2.
- In `model.ipynb`, the cells under "Choosing settings on a holdout year" (Steps B to F) are commented out. They record how the final settings were chosen. Uncomment one to rerun that experiment.

## 4. Model input files

All files are keyed on `nodeID` + `Year`, with one row per company-year. Shared files cover every company-year (33,664 rows). Per-split files cover only that split's rows (for example, split 0 covers 2014 to 2019, 19,494 rows).

| File (in `data/processed/`) | Columns | Used by |
|---|---|---|
| `tabular_features_folds/split_<s>/` | 1,238 financial features, imputed and scaled | **Only split `s`** |
| `text_features.parquet` | 8: `has_text`, `is_truncated`, 3 sentiment, 3 readability | All splits |
| `text_embeddings.parquet` | 64: `emb_0` to `emb_63` | All splits |
| `text_tfidf/split_<s>.parquet` | 100: `tfidf_0` to `tfidf_99` | **Only split `s`** |
| `graph/company_edges.parquet` | `source_id`, `target_id`, `relation`, `Year` | All splits |
| `graph/degree_features.csv` | 4 link counts | All splits |

`data/splits/temporal_splits.csv` lists each split's rows (`split` 0 to 3), their `role` (`train` or `test`) and the `Label`.

Loading the features for split `s`:

```python
import pandas as pd

s = 0
rows = pd.read_csv("data/splits/temporal_splits.csv").query("split == @s")

df = (rows
      .merge(pd.read_parquet(f"data/processed/tabular_features_folds/split_{s}"), on=["nodeID", "Year"])
      .merge(pd.read_parquet("data/processed/text_features.parquet"), on=["nodeID", "Year"])
      .merge(pd.read_parquet("data/processed/text_embeddings.parquet"), on=["nodeID", "Year"])
      .merge(pd.read_parquet(f"data/processed/text_tfidf/split_{s}.parquet"), on=["nodeID", "Year"]))

train, test = df[df.role == "train"], df[df.role == "test"]
```