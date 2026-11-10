"""Locates and lazily loads the saved artifacts produced by the notebook.

Every loader is wrapped so that a missing / incompatible file produces a clear,
user-facing error message instead of crashing the whole application."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA_PROC = ROOT / "data" / "processed"

_lock = threading.RLock()  # re-entrant: a loader may load another artifact
_cache: Dict[str, Any] = {}
_errors: Dict[str, str] = {}


class ArtifactError(RuntimeError):
    pass


def bert_dir() -> Path:
    for d in (MODELS / "distilbert-base-uncased", ROOT):
        if (d / "model.safetensors").exists() and (d / "vocab.txt").exists() and (d / "config.json").exists():
            return d
    raise ArtifactError("DistilBERT files (config.json, vocab.txt, model.safetensors) not found in "
                        "models/distilbert-base-uncased/. Download them from "
                        "https://huggingface.co/distilbert-base-uncased")


def get(name: str, loader: Callable[[], Any]) -> Any:
    """Returns the cached artifact, loading it once (thread-safe)."""
    if name in _cache:
        return _cache[name]
    with _lock:
        if name in _cache:
            return _cache[name]
        try:
            _cache[name] = loader()
            _errors.pop(name, None)
            return _cache[name]
        except ArtifactError as e:
            _errors[name] = str(e)
            raise
        except FileNotFoundError as e:
            msg = (f"Missing artifact for '{name}': {Path(e.filename).name if e.filename else e}. "
                   "Run notebooks/job_offers_nlp_analysis.ipynb to generate the models.")
            _errors[name] = msg
            raise ArtifactError(msg) from e
        except Exception as e:  # incompatible pickle, corrupted file...
            msg = f"Could not load '{name}': {type(e).__name__}: {e}"
            _errors[name] = msg
            raise ArtifactError(msg) from e


def load_json(rel: str) -> dict:
    p = MODELS / rel
    if not p.exists():
        raise FileNotFoundError(2, "missing", str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def status() -> dict:
    files = {
        "classification_pipeline.joblib": MODELS / "classification_pipeline.joblib",
        "skills_only_pipeline.joblib": MODELS / "skills_only_pipeline.joblib",
        "kmeans_model.joblib": MODELS / "kmeans_model.joblib",
        "pca_2d.joblib": MODELS / "pca_2d.joblib",
        "job_embeddings.npy": MODELS / "job_embeddings.npy",
        "embedding_mean.npy": MODELS / "embedding_mean.npy",
        "skill_embeddings.npy": MODELS / "skill_embeddings.npy",
        "analytics.json": MODELS / "analytics.json",
        "clustering_config.json": MODELS / "clustering_config.json",
        "model_config.json": MODELS / "model_config.json",
        "kg/postings.npz": MODELS / "kg" / "postings.npz",
        "embedding_corpus.csv": DATA_PROC / "embedding_corpus.csv",
    }
    try:
        bd = str(bert_dir().relative_to(ROOT)) or "."
    except ArtifactError:
        bd = None
    return {"files": {k: v.exists() for k, v in files.items()}, "distilbert_dir": bd,
            "loaded": sorted(_cache), "errors": dict(_errors)}
