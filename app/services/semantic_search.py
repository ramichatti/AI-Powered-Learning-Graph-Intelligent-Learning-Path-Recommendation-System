"""Semantic job search: DistilBERT query embedding vs stored job embeddings (cosine)."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from jobnlp.distilbert import DistilBertEncoder
from jobnlp.embeddings import center_normalize, top_k_cosine

from . import artifacts as A
from .skill_extraction import get_extractor


def get_encoder() -> DistilBertEncoder:
    def load():
        cfg = A.load_json("model_config.json")["distilbert"]
        return DistilBertEncoder(A.bert_dir(), max_len=cfg["tokenizer"]["max_length"])
    return A.get("distilbert_encoder", load)


def get_index():
    def load():
        import pandas as pd
        raw = np.load(A.MODELS / "job_embeddings.npy").astype(np.float32)
        mean = np.load(A.MODELS / "embedding_mean.npy").astype(np.float32)
        emb, _ = center_normalize(raw, mean)
        corpus = pd.read_csv(A.DATA_PROC / "embedding_corpus.csv")
        if len(corpus) != len(emb):
            raise A.ArtifactError(f"embedding_corpus.csv has {len(corpus)} rows but job_embeddings.npy "
                                  f"has {len(emb)} - re-run the notebook.")
        corpus["skills_list"] = corpus["skills_str"].fillna("").str.split("|")
        return emb, mean, corpus
    return A.get("job_index", load)


def embed_query(text: str) -> np.ndarray:
    _, mean, _ = get_index()
    q = get_encoder().encode([text], normalize=True)[0] - mean
    return (q / np.linalg.norm(q)).astype(np.float32)


def search(query: str, k: int = 10, category: Optional[str] = None, cluster: Optional[int] = None) -> dict:
    from .classification import validate_text, InputError
    from .clustering import labels as cluster_labels, project_query
    query = validate_text(query, min_chars=3, field="query")
    k = int(k)
    if not 1 <= k <= 50:
        raise InputError("k must be between 1 and 50.")
    emb, _, corpus = get_index()
    t0 = time.time()
    q = embed_query(query)
    t_embed = time.time() - t0
    mask = np.ones(len(corpus), dtype=bool)
    if category:
        mask &= (corpus["job_title_short"] == category).to_numpy()
    if cluster is not None:
        mask &= (corpus["cluster"] == int(cluster)).to_numpy()
    if not mask.any():
        return {"results": [], "query_skills": [], "n_candidates": 0}
    sub = np.where(mask)[0]
    idx, sims = top_k_cosine(q, emb[sub], k=k)
    idx = sub[idx]
    qskills = set(get_extractor().extract_names(query))
    labs = cluster_labels()
    res = []
    for i, s in zip(idx, sims):
        r = corpus.iloc[i]
        sk = [x for x in r["skills_list"] if x]
        res.append({"job_id": int(r["job_id"]), "title": r["job_title"], "category": r["job_title_short"],
                    "company": r["company_name"], "location": r["job_location"], "country": r["job_country"],
                    "skills": sk, "matched_skills": sorted(qskills & set(sk)), "similarity": float(s),
                    "cluster": int(r["cluster"]), "cluster_label": labs.get(int(r["cluster"]), ""),
                    "salary": None if np.isnan(r["salary_year_avg"]) else float(r["salary_year_avg"])})
    proj = project_query(q)
    return {"results": res, "query_skills": sorted(qskills), "n_candidates": int(mask.sum()),
            "embed_ms": round(t_embed * 1000), "query_cluster": proj["cluster"],
            "query_cluster_label": labs.get(proj["cluster"], ""), "query_xy": proj["xy"]}


def similar_to(job_id: int, k: int = 10) -> dict:
    emb, _, corpus = get_index()
    hits = np.where(corpus["job_id"].to_numpy() == int(job_id))[0]
    if not len(hits):
        from .classification import InputError
        raise InputError(f"job_id {job_id} is not in the embedded corpus.")
    i = int(hits[0])
    idx, sims = top_k_cosine(emb[i], emb, k=k, exclude=i)
    r0 = corpus.iloc[i]
    return {"reference": {"title": r0["job_title"], "company": r0["company_name"],
                          "skills": [x for x in r0["skills_list"] if x]},
            "results": [{"job_id": int(corpus.iloc[j]["job_id"]), "title": corpus.iloc[j]["job_title"],
                         "category": corpus.iloc[j]["job_title_short"], "company": corpus.iloc[j]["company_name"],
                         "location": corpus.iloc[j]["job_location"],
                         "skills": [x for x in corpus.iloc[j]["skills_list"] if x], "similarity": float(s)}
                        for j, s in zip(idx, sims)]}


def random_ad(seed: Optional[int] = None) -> dict:
    _, _, corpus = get_index()
    rng = np.random.default_rng(seed)
    r = corpus.iloc[int(rng.integers(len(corpus)))]
    skills = [x for x in r["skills_list"] if x]
    return {"job_id": int(r["job_id"]), "title": r["job_title"], "category": r["job_title_short"],
            "company": r["company_name"], "skills": skills,
            "text": f"{r['job_title']}\nSkills: {', '.join(skills)}"}
