"""Skill extraction service: dictionary/rule extractor + dataset statistics + semantic neighbours."""
from __future__ import annotations

import json

import numpy as np

from jobnlp.skills import SkillExtractor

from . import artifacts as A


def get_extractor() -> SkillExtractor:
    return A.get("skill_extractor", SkillExtractor)


def _skill_embeddings():
    def load():
        emb = np.load(A.MODELS / "skill_embeddings.npy").astype(np.float32)
        names = json.loads((A.MODELS / "skill_embedding_names.json").read_text())
        return emb, names, {n: i for i, n in enumerate(names)}
    return A.get("skill_embeddings", load)


def related_skills(skill: str, k: int = 5):
    try:
        emb, names, idx = _skill_embeddings()
    except A.ArtifactError:
        return []
    if skill not in idx:
        return []
    sims = emb @ emb[idx[skill]]
    sims[idx[skill]] = -1
    top = np.argsort(-sims)[:k]
    return [{"skill": names[i], "similarity": float(sims[i])} for i in top]


def extract(text: str) -> dict:
    from .classification import validate_text
    from .knowledge_graph import get_kg
    text = validate_text(text)
    ex = get_extractor()
    found = ex.extract(text)
    kg = None
    try:
        kg = get_kg()
    except A.ArtifactError:
        pass
    rows = []
    for d in found:
        row = {"skill": d["skill"], "category": d["category"], "mentions": d["mentions"], "spans": d["spans"],
               "dataset_share": None, "top_job": None, "related_semantic": related_skills(d["skill"], 4)}
        if kg is not None and d["skill"] in kg.s2i:
            col = kg.X[:, kg.s2i[d["skill"]]].indices
            row["dataset_share"] = float(len(col) / kg.n)
            row["dataset_count"] = int(len(col))
            jc = np.bincount(kg.job[col], minlength=len(kg.jobs)) / np.maximum(kg.job_freq, 1)
            row["top_job"] = kg.jobs[int(np.argmax(jc))]
            row["top_job_share"] = float(jc.max())
        rows.append(row)
    by_cat = {}
    for d in found:
        by_cat.setdefault(d["category"], []).append(d["skill"])
    recs = []
    if kg is not None and found:
        recs = kg.recommend_skills([d["skill"] for d in found if d["skill"] in kg.s2i], k=8)
    return {"skills": rows, "by_category": by_cat, "n_skills": len(found),
            "n_categories": len(by_cat), "recommended": recs, "text": text}


def dictionary_info() -> dict:
    ex = get_extractor()
    return {"n_skills": len(ex.skill2cat), "n_aliases": len(ex.aliases),
            "categories": {c: len(v) for c, v in ex.categories.items()}}
