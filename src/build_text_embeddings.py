"""Build fold-independent MD&A embeddings

Stage 1 (slow): embed every report's text_clean with BAAI/bge-small-zh-v1.5.
  Each report is split into 510-token chunks; each chunk's [CLS] vector is
  normalised, then the chunk vectors are averaged into one 512-number vector.
  Writes data/processed/text_embeddings_raw/<year>.parquet. Years already written
  are skipped, so an interrupted run resumes where it stopped.
Stage 2 (fast, once all years exist): compress 512 -> N_COMPONENTS with PCA fitted
  on PCA_FIT_YEAR only. That year is in every walk-forward fold's training years,
  so the result uses no validation/test data and stays fold-independent.
  Writes data/processed/text_embeddings.parquet: nodeID, Year, emb_0 ... emb_63.
Rows with has_text = 0 are all zeros. Rules and reasons: docs/Findings&Issues.md #14.

Run from the repo root:  python src/build_text_embeddings.py        (all years, resumes)
                         python src/build_text_embeddings.py 2014   (stage 1 for some years)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "mda_clean"
RAW_DIR = ROOT / "data" / "processed" / "text_embeddings_raw"
OUT = ROOT / "data" / "processed" / "text_embeddings.parquet"
YEARS = range(2014, 2023)

MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_TOKENS = 510  # 512 limit minus [CLS] and [SEP]
BATCH = 256
N_COMPONENTS = 64
PCA_FIT_YEAR = 2014


# iian
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
torch.set_num_threads(8)  # physical cores; only matters on the CPU fallback
# --

@torch.no_grad()
def embed(text: str, tok, model, device: str) -> np.ndarray:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = [ids[i:i + CHUNK_TOKENS] for i in range(0, len(ids), CHUNK_TOKENS)]
    vecs = []
    for i in range(0, len(pieces), BATCH):
        batch = [[tok.cls_token_id] + p + [tok.sep_token_id] for p in pieces[i:i + BATCH]]
        enc = tok.pad({"input_ids": batch}, return_tensors="pt").to(device)
        # cls = model(**enc).last_hidden_state[:, 0]
        # iian
        cls = model(**enc).last_hidden_state[:, 0].float()
        # --
        vecs.append(torch.nn.functional.normalize(cls, dim=-1).cpu())
    return torch.cat(vecs).mean(0).numpy()


def embed_year(year: int, tok, model, device: str) -> None:
    start = time.time()
    df = pd.read_parquet(CLEAN_DIR / f"{year}.parquet", columns=["nodeID", "Year", "has_text", "text_clean"])
    dim = model.config.hidden_size
    vecs = np.zeros((len(df), dim), dtype=np.float32)
    for i, (has_text, text) in enumerate(zip(df["has_text"], df["text_clean"])):
        if has_text:
            vecs[i] = embed(text, tok, model, device)
        if (i + 1) % 500 == 0:
            print(f"  {year}: {i + 1:,}/{len(df):,}  {time.time() - start:.0f}s", flush=True)
    out = df[["nodeID", "Year", "has_text"]].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(vecs, columns=[f"raw_{j}" for j in range(dim)])], axis=1)
    out.to_parquet(RAW_DIR / f"{year}.parquet", index=False)
    print(f"{year}: rows {len(df):,}  {time.time() - start:.0f}s", flush=True)


def compress() -> None:
    raw = pd.read_parquet(RAW_DIR)
    cols = [c for c in raw.columns if c.startswith("raw_")]
    fit_rows = (raw["Year"] == PCA_FIT_YEAR) & (raw["has_text"] == 1)
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full").fit(raw.loc[fit_rows, cols].to_numpy())
    z = pca.transform(raw[cols].to_numpy()).astype(np.float32)
    z[raw["has_text"].to_numpy() == 0] = 0
    out = pd.concat(
        [raw[["nodeID", "Year"]], pd.DataFrame(z, columns=[f"emb_{j}" for j in range(N_COMPONENTS)])], axis=1
    )
    out.to_parquet(OUT, index=False)
    print(
        f"PCA fitted on {int(fit_rows.sum()):,} reports from {PCA_FIT_YEAR}: {N_COMPONENTS} components "
        f"explain {pca.explained_variance_ratio_.sum():.1%} of the variance"
    )
    print(f"wrote {OUT.relative_to(ROOT)}  {len(out):,} rows  {OUT.stat().st_size / 1e6:.1f} MB")


def main(years: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [y for y in years if not (RAW_DIR / f"{y}.parquet").exists()]
    if todo:
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MODEL)
        tok.model_max_length = 10**9  # chunking is done here; silences the long-input warning

        # model = AutoModel.from_pretrained(MODEL).to(device).eval()
        # print(f"{MODEL} on {device}; years to embed: {todo}", flush=True)
        # for year in todo:
        #     embed_year(year, tok, model, device)

        # iian
        model = AutoModel.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
        print(f"{MODEL} on {DEVICE} ({DTYPE}); years to embed: {todo}", flush=True)
        for year in todo:
            embed_year(year, tok, model, DEVICE)
        # --
    if all((RAW_DIR / f"{y}.parquet").exists() for y in YEARS):
        compress()
    else:
        missing = [y for y in YEARS if not (RAW_DIR / f"{y}.parquet").exists()]
        print(f"stage 2 waits for all years; missing: {missing}")


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or list(YEARS))
