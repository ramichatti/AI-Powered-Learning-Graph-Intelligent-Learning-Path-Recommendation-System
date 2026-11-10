"""Embedding corpus sampling, caching and cosine-similarity search."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .preprocessing import TARGET


def sample_for_embedding(df: pd.DataFrame, per_class: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Class-balanced sample (at most `per_class` ads per job_title_short).

    Balancing keeps minority professions (ML Engineer, Cloud Engineer) visible to K-Means
    instead of letting the three large classes dominate every cluster."""
    parts = []
    for _, g in df.groupby(TARGET, sort=True):
        parts.append(g.sample(n=min(per_class, len(g)), random_state=seed))
    return pd.concat(parts).sort_values("job_id").reset_index(drop=True)


def ids_fingerprint(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def compute_or_load_embeddings(encoder_factory, texts: List[str], ids, cache_path: Path,
                               batch_size: int = 64, force: bool = False) -> np.ndarray:
    """Loads cached embeddings if they were computed for exactly these job ids, otherwise
    computes them (raw mean-pooled, L2-normalised) and caches them as .npy + .json meta."""
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")
    fp = ids_fingerprint(ids)
    if not force and cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("ids_fingerprint") == fp:
            print(f"Loaded cached embeddings {cache_path.name} {meta['shape']} "
                  f"(computed in {meta['seconds']:.0f}s)")
            return np.load(cache_path).astype(np.float32)
    enc = encoder_factory()
    t0 = time.time()
    emb = enc.encode(list(texts), batch_size=batch_size, verbose=True)
    secs = time.time() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # stored as float16 (half the disk size; cosine similarities change by < 1e-3)
    np.save(cache_path, emb.astype(np.float16))
    meta_path.write_text(json.dumps({"ids_fingerprint": fp, "shape": list(emb.shape),
                                     "seconds": secs, "n_truncated": int(enc.n_truncated),
                                     "pooling": "mean over attention-mask tokens, L2-normalised"}))
    print(f"Computed {emb.shape} embeddings in {secs:.0f}s ({len(texts)/secs:.1f} texts/s)")
    return emb


def center_normalize(emb: np.ndarray, mean: Optional[np.ndarray] = None):
    """Removes the common component shared by all DistilBERT sentence vectors (anisotropy)
    then re-normalises; returns (vectors, mean)."""
    if mean is None:
        mean = emb.mean(0)
    x = emb - mean
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
    return x.astype(np.float32), mean.astype(np.float32)


def top_k_cosine(query: np.ndarray, matrix: np.ndarray, k: int = 10, exclude: Optional[int] = None):
    sims = matrix @ query
    if exclude is not None:
        sims[exclude] = -np.inf
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]
