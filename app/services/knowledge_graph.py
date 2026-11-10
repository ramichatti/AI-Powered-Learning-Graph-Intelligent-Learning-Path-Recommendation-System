"""Knowledge-graph exploration and graph Q&A service."""
from __future__ import annotations

from typing import List, Optional

from jobnlp.graph import KnowledgeGraph

from . import artifacts as A
from .skill_extraction import get_extractor


def get_kg() -> KnowledgeGraph:
    return A.get("knowledge_graph", lambda: KnowledgeGraph(A.MODELS / "kg", get_extractor()))


def graph(jobs: Optional[List[str]] = None, n_companies: int = 25, n_skills: int = 35,
          min_share: float = 0.12, company_skill_k: int = 3, cluster: Optional[int] = None,
          focus: Optional[str] = None) -> dict:
    kg = get_kg()
    rows = None
    if cluster is not None:
        from .clustering import cluster_job_ids
        rows = cluster_job_ids(cluster)
    n_companies = max(0, min(int(n_companies), 80))
    n_skills = max(3, min(int(n_skills), 80))
    min_share = max(0.0, min(float(min_share), 0.9))
    jobs = [j for j in (jobs or []) if j in kg.j2i] or None
    G = kg.build_graph(n_companies=n_companies, n_skills=n_skills, jobs=jobs, min_share=min_share,
                       company_skill_k=company_skill_k, rows=rows,
                       focus_skills=[focus] if focus and focus in kg.s2i else None)
    out = kg.to_cytoscape(G)
    out["jobs"] = kg.jobs
    return out


def answer(question: str) -> dict:
    from .classification import validate_text
    question = validate_text(question, min_chars=4, field="question")
    if len(question) > 500:
        from .classification import InputError
        raise InputError("Please keep the question under 500 characters.")
    return get_kg().answer(question)


def search_nodes(q: str, limit: int = 12) -> list:
    kg = get_kg()
    ql = q.lower().strip()
    if len(ql) < 2:
        return []
    out = [{"type": "skill", "label": s} for s in kg.skills if ql in s][:limit]
    out += [{"type": "job", "label": j} for j in kg.jobs if ql in j.lower()]
    comp = [(c, kg.company_freq[i]) for i, c in enumerate(kg.companies) if ql in c.lower()]
    comp.sort(key=lambda t: -t[1])
    out += [{"type": "company", "label": c, "count": int(n)} for c, n in comp[:limit]]
    return out[: limit * 2]


def neighborhood(kind: str, label: str, k: int = 12) -> dict:
    """Ego graph of one node, computed from the full posting index (progressive exploration)."""
    import numpy as np
    import networkx as nx
    kg = get_kg()
    G = nx.Graph()
    nid = kg._nid(kind, label)
    if kind == "skill" and label in kg.s2i:
        rows = kg.X[:, kg.s2i[label]].indices
        G.add_node(nid, label=label, type="skill", count=len(rows), category=kg.extractor.skill2cat.get(label, ""))
        jc = np.bincount(kg.job[rows], minlength=len(kg.jobs))
        for j in np.argsort(-jc)[:6]:
            if jc[j]:
                G.add_node(kg._nid("job", kg.jobs[j]), label=kg.jobs[j], type="job", count=int(kg.job_freq[j]))
                G.add_edge(nid, kg._nid("job", kg.jobs[j]), type="requires", weight=float(jc[j] / kg.job_freq[j]),
                           label=f"{jc[j]/kg.job_freq[j]:.0%}")
        cc = np.bincount(kg.company[rows], minlength=len(kg.companies))
        for c in np.argsort(-cc)[:k]:
            if cc[c] and kg.companies[c] not in ("Confidential", "Unknown"):
                G.add_node(kg._nid("company", kg.companies[c]), label=kg.companies[c], type="company",
                           count=int(kg.company_freq[c]))
                G.add_edge(nid, kg._nid("company", kg.companies[c]), type="uses", weight=float(cc[c]), label=str(int(cc[c])))
        for r in kg.recommend_skills([label], k=6):
            G.add_node(kg._nid("skill", r["skill"]), label=r["skill"], type="skill",
                       count=int(kg.skill_freq[kg.s2i[r["skill"]]]), category=r["category"])
            G.add_edge(nid, kg._nid("skill", r["skill"]), type="co-occurs", weight=r["cooccurrence"],
                       label=f"{r['cooccurrence']:.0%}")
    elif kind in ("company", "job"):
        if kind == "company":
            if label not in kg.companies:
                return kg.to_cytoscape(G)
            rows = np.where(kg.company == kg.companies.index(label))[0]
        else:
            if label not in kg.j2i:
                return kg.to_cytoscape(G)
            rows = np.where(kg.job == kg.j2i[label])[0]
        G.add_node(nid, label=label, type=kind, count=len(rows))
        sc = np.asarray(kg.Xr[rows].sum(0)).ravel()
        for s in np.argsort(-sc)[:k]:
            if sc[s]:
                G.add_node(kg._nid("skill", kg.skills[s]), label=kg.skills[s], type="skill", count=int(kg.skill_freq[s]),
                           category=kg.extractor.skill2cat.get(kg.skills[s], ""))
                G.add_edge(nid, kg._nid("skill", kg.skills[s]), type="requires" if kind == "job" else "uses",
                           weight=float(sc[s] / len(rows)), label=f"{sc[s]/len(rows):.0%}")
        if kind == "company":
            jc = np.bincount(kg.job[rows], minlength=len(kg.jobs))
            for j in np.where(jc > 0)[0]:
                G.add_node(kg._nid("job", kg.jobs[j]), label=kg.jobs[j], type="job", count=int(kg.job_freq[j]))
                G.add_edge(nid, kg._nid("job", kg.jobs[j]), type="offers", weight=float(jc[j]), label=str(int(jc[j])))
        else:
            cc = np.bincount(kg.company[rows], minlength=len(kg.companies))
            for c in np.argsort(-cc)[:k]:
                if cc[c] and kg.companies[c] not in ("Confidential", "Unknown"):
                    G.add_node(kg._nid("company", kg.companies[c]), label=kg.companies[c], type="company",
                               count=int(kg.company_freq[c]))
                    G.add_edge(nid, kg._nid("company", kg.companies[c]), type="offers", weight=float(cc[c]),
                               label=str(int(cc[c])))
    return kg.to_cytoscape(G)
