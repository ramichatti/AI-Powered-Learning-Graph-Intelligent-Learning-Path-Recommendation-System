"""K-Means clustering results: profiles, 2-D projection and per-cluster exploration."""
from __future__ import annotations

from collections import Counter

import joblib
import numpy as np

from . import artifacts as A


def config() -> dict:
    return A.get("clustering_config", lambda: A.load_json("clustering_config.json"))


def labels() -> dict:
    return {p["cluster"]: p["label"] for p in config()["profiles"]}


def _kmeans():
    return A.get("kmeans_model", lambda: joblib.load(A.MODELS / "kmeans_model.joblib"))


def _pca():
    return A.get("pca_2d", lambda: joblib.load(A.MODELS / "pca_2d.joblib"))


def _corpus():
    from .semantic_search import get_index
    return get_index()[2]


def project_query(q: np.ndarray) -> dict:
    try:
        km, pca = _kmeans(), _pca()
        c = int(km.predict(q[None].astype(km.cluster_centers_.dtype))[0])
        xy = pca.transform(q[None].astype(pca.components_.dtype))[0].tolist()
    except A.ArtifactError:
        return {"cluster": None, "xy": None}
    return {"cluster": c, "xy": xy}


def overview(max_points: int = 6000, seed: int = 42) -> dict:
    cfg = config()
    corpus = _corpus()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(corpus), min(max_points, len(corpus)), replace=False))
    pts = corpus.iloc[idx]
    points = [{"x": round(float(x), 4), "y": round(float(y), 4), "c": int(c), "t": t, "j": j}
              for x, y, c, t, j in zip(pts["pca_x"], pts["pca_y"], pts["cluster"], pts["job_title"],
                                      pts["job_title_short"])]
    return {"k": cfg["k_selected"], "k_elbow": cfg["k_elbow"], "k_range": cfg["k_range"],
            "inertia": cfg["inertia"], "silhouette": cfg["silhouette"], "final_silhouette": cfg["final_silhouette"],
            "ari": cfg["ari_vs_job_title_short"], "nmi": cfg["nmi_vs_job_title_short"],
            "selection_rule": cfg["selection_rule"], "explained_variance": cfg["pca_explained_variance"],
            "centroids": cfg["centroids_2d"], "profiles": cfg["profiles"], "points": points,
            "n_total": int(len(corpus)), "n_points": len(points)}


def cluster_job_ids(cluster: int):
    corpus = _corpus()
    return corpus.loc[corpus["cluster"] == int(cluster), "job_id"].to_numpy()


def detail(cluster: int, page: int = 1, per_page: int = 15, q: str = "") -> dict:
    from .classification import InputError
    cfg = config()
    if not 0 <= int(cluster) < cfg["k_selected"]:
        raise InputError(f"Cluster must be between 0 and {cfg['k_selected'] - 1}.")
    corpus = _corpus()
    sub = corpus[corpus["cluster"] == int(cluster)]
    if q:
        m = sub["job_title"].str.contains(q, case=False, regex=False) | sub["skills_str"].str.contains(q.lower(), regex=False)
        sub = sub[m.fillna(False)]
    prof = next(p for p in cfg["profiles"] if p["cluster"] == int(cluster))
    countries = Counter(sub["job_country"]).most_common(6)
    total = len(sub)
    page = max(1, int(page))
    rows = sub.iloc[(page - 1) * per_page: page * per_page]
    return {"profile": prof, "n_matching": total, "page": page, "pages": max(1, -(-total // per_page)),
            "countries": [{"name": k, "count": v} for k, v in countries],
            "ads": [{"job_id": int(r.job_id), "title": r.job_title, "category": r.job_title_short,
                     "company": r.company_name, "location": r.job_location,
                     "skills": [s for s in str(r.skills_str).split("|") if s]} for r in rows.itertuples()]}
