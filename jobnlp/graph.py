"""Job - Company - Skill knowledge graph, community detection and a mini graph Q&A.

Storage
-------
The full cleaned dataset is compressed into a posting index (`models/kg/postings.npz`):
a sparse posting x skill matrix plus integer codes for company, job category and country.
Every graph and every answer is computed from that index on demand, so the Flask app
can explore all 560k postings without loading the CSV.

Graph schema (tripartite, weighted, undirected)
-----------------------------------------------
* Job node  (job_title_short)          - size = number of postings
* Company node                          - size = number of postings
* Skill node                            - size = number of postings mentioning it
* Company -[offers]-> Job               weight = postings of that job at that company
* Job     -[requires]-> Skill           weight = share of the job's postings listing the skill
* Company -[uses]-> Skill               weight = postings of the company listing the skill

Communities: Louvain modularity maximisation (networkx) on the weighted graph.
Recommendation: Personalised PageRank on the skill co-occurrence graph (PMI-weighted).

Q&A ("Graph-RAG lite"): rule-based intent detection + entity linking (skills via the
SkillExtractor, job categories and countries via dictionaries) -> structured retrieval
on the posting index -> templated natural-language answer + evidence table + subgraph.
No generative model is involved, so every number in an answer is a real count.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .skills import SkillExtractor

JOB_SYNONYMS = {
    "senior data engineer": "Senior Data Engineer", "senior data scientist": "Senior Data Scientist",
    "senior data analyst": "Senior Data Analyst", "machine learning engineer": "Machine Learning Engineer",
    "ml engineer": "Machine Learning Engineer", "mle": "Machine Learning Engineer",
    "data engineer": "Data Engineer", "data scientist": "Data Scientist", "data analyst": "Data Analyst",
    "business analyst": "Business Analyst", "software engineer": "Software Engineer",
    "cloud engineer": "Cloud Engineer", "ai engineer": "Machine Learning Engineer",
    "bi analyst": "Business Analyst", "business intelligence analyst": "Business Analyst",
    "software developer": "Software Engineer",
}
COUNTRY_ALIASES = {"usa": "United States", "us": "United States", "u.s.": "United States",
                   "america": "United States", "uk": "United Kingdom", "england": "United Kingdom",
                   "uae": "United Arab Emirates", "deutschland": "Germany"}


# ----------------------------------------------------------------------------
def build_posting_index(df: pd.DataFrame, out_dir: Path) -> dict:
    """Compresses the cleaned dataframe into the posting index used by the graph + Q&A."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    skills = sorted({s for lst in df["skills"] for s in lst})
    s2i = {s: i for i, s in enumerate(skills)}
    rows, cols = [], []
    for r, lst in enumerate(df["skills"]):
        for s in lst:
            rows.append(r); cols.append(s2i[s])
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                          shape=(len(df), len(skills)))
    comp_codes, companies = pd.factorize(df["company_name"])
    job_codes, jobs = pd.factorize(df["job_title_short"], sort=True)
    ctry_codes, countries = pd.factorize(df["job_country"])
    sal = df["salary_year_avg"].to_numpy(dtype=np.float32)
    sparse.save_npz(out_dir / "posting_skills.npz", X)
    np.savez_compressed(out_dir / "postings.npz", company=comp_codes.astype(np.int32),
                        job=job_codes.astype(np.int8), country=ctry_codes.astype(np.int16),
                        salary=sal, remote=df["job_work_from_home"].to_numpy(bool))
    vocab = {"skills": skills, "companies": list(companies), "jobs": list(jobs),
             "countries": list(countries)}
    (out_dir / "vocab.json").write_text(json.dumps(vocab))
    return vocab


class KnowledgeGraph:
    def __init__(self, kg_dir: str | Path, extractor: Optional[SkillExtractor] = None):
        kg_dir = Path(kg_dir)
        self.X = sparse.load_npz(kg_dir / "posting_skills.npz").tocsc().astype(np.float32)
        self.Xr = self.X.tocsr()
        z = np.load(kg_dir / "postings.npz")
        self.company, self.job, self.country = z["company"], z["job"], z["country"]
        self.salary, self.remote = z["salary"], z["remote"]
        v = json.loads((kg_dir / "vocab.json").read_text())
        self.skills, self.companies, self.jobs, self.countries = (
            v["skills"], v["companies"], v["jobs"], v["countries"])
        self.s2i = {s: i for i, s in enumerate(self.skills)}
        self.j2i = {j: i for i, j in enumerate(self.jobs)}
        self.n = len(self.job)
        self.extractor = extractor or SkillExtractor()
        self.skill_freq = np.asarray(self.X.sum(0)).ravel()
        self.company_freq = np.bincount(self.company, minlength=len(self.companies))
        self.job_freq = np.bincount(self.job, minlength=len(self.jobs))
        self._cooc = None
        self.communities: Dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _mask(self, jobs=None, skills_all=None, country=None, rows=None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if rows is not None:
            mm = np.zeros(self.n, dtype=bool); mm[np.asarray(rows, dtype=np.int64)] = True; m &= mm
        if jobs:
            m &= np.isin(self.job, [self.j2i[j] for j in jobs if j in self.j2i])
        if country:
            ci = [i for i, c in enumerate(self.countries) if c == country]
            m &= np.isin(self.country, ci)
        for s in skills_all or []:
            if s in self.s2i:
                col = self.X[:, self.s2i[s]]
                hit = np.zeros(self.n, dtype=bool); hit[col.indices] = True
                m &= hit
            else:
                m &= False
        return m

    @staticmethod
    def _nid(kind: str, name: str) -> str:
        return f"{kind}:{name}"

    # ------------------------------------------------------------------ graph
    def build_graph(self, n_companies: int = 25, n_skills: int = 35, jobs: Optional[List[str]] = None,
                    min_share: float = 0.12, company_skill_k: int = 3, rows=None,
                    focus_skills: Optional[List[str]] = None) -> nx.Graph:
        m = self._mask(jobs=jobs, rows=rows)
        idx = np.where(m)[0]
        G = nx.Graph()
        if len(idx) == 0:
            return G
        jcount = np.bincount(self.job[idx], minlength=len(self.jobs))
        scount = np.asarray(self.Xr[idx].sum(0)).ravel()
        ccount = np.bincount(self.company[idx], minlength=len(self.companies))
        # exclude placeholder employers
        for bad in ("Confidential", "Unknown", ""):
            if bad in self.companies:
                ccount[self.companies.index(bad)] = 0
        top_c = [c for c in np.argsort(-ccount)[:n_companies] if ccount[c] > 0]
        top_s = [s for s in np.argsort(-scount)[:n_skills] if scount[s] > 0]
        if focus_skills:
            top_s = list(dict.fromkeys([self.s2i[s] for s in focus_skills if s in self.s2i] + top_s))
        top_j = [j for j in range(len(self.jobs)) if jcount[j] > 0]
        for j in top_j:
            G.add_node(self._nid("job", self.jobs[j]), label=self.jobs[j], type="job",
                       count=int(jcount[j]))
        for c in top_c:
            G.add_node(self._nid("company", self.companies[c]), label=self.companies[c],
                       type="company", count=int(ccount[c]))
        for s in top_s:
            G.add_node(self._nid("skill", self.skills[s]), label=self.skills[s], type="skill",
                       count=int(scount[s]), category=self.extractor.skill2cat.get(self.skills[s], "Other"))
        # job -requires-> skill
        top_s_arr = np.array(top_s)
        for j in top_j:
            jr = idx[self.job[idx] == j]
            share = np.asarray(self.Xr[jr][:, top_s_arr].sum(0)).ravel() / max(len(jr), 1)
            for s, sh in zip(top_s, share):
                if sh >= min_share:
                    G.add_edge(self._nid("job", self.jobs[j]), self._nid("skill", self.skills[s]),
                               type="requires", weight=float(sh), label=f"{sh:.0%}")
        # company -offers-> job ; company -uses-> skill (top-k per company to limit density)
        for c in top_c:
            cr = idx[self.company[idx] == c]
            jc = np.bincount(self.job[cr], minlength=len(self.jobs))
            for j in np.where(jc > 0)[0]:
                if j in top_j and jc[j] / len(cr) >= 0.05:
                    G.add_edge(self._nid("company", self.companies[c]), self._nid("job", self.jobs[j]),
                               type="offers", weight=float(jc[j]), label=str(int(jc[j])))
            sc = np.asarray(self.Xr[cr][:, top_s_arr].sum(0)).ravel()
            for k in np.argsort(-sc)[:company_skill_k]:
                if sc[k] > 0:
                    G.add_edge(self._nid("company", self.companies[c]), self._nid("skill", self.skills[top_s[k]]),
                               type="uses", weight=float(sc[k]), label=str(int(sc[k])))
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
        return G

    def detect_communities(self, G: nx.Graph, seed: int = 42) -> Dict[str, int]:
        if G.number_of_nodes() == 0:
            return {}
        # normalise weights per edge type so counts do not dominate shares
        H = nx.Graph()
        maxw = {}
        for u, v, d in G.edges(data=True):
            maxw[d["type"]] = max(maxw.get(d["type"], 0), d["weight"])
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, weight=d["weight"] / maxw[d["type"]])
        comms = nx.community.louvain_communities(H, weight="weight", seed=seed, resolution=1.0)
        comms = sorted(comms, key=len, reverse=True)
        return {n: i for i, c in enumerate(comms) for n in c}

    def to_cytoscape(self, G: nx.Graph, communities: Optional[Dict[str, int]] = None) -> dict:
        communities = communities if communities is not None else self.detect_communities(G)
        nodes, edges = [], []
        for n, d in G.nodes(data=True):
            nodes.append({"data": {"id": n, "label": d["label"], "type": d["type"], "count": d["count"],
                                   "degree": G.degree(n), "community": communities.get(n, -1),
                                   "category": d.get("category", "")}})
        for u, v, d in G.edges(data=True):
            edges.append({"data": {"id": f"{u}->{v}", "source": u, "target": v, "type": d["type"],
                                   "weight": d["weight"], "label": d["label"]}})
        n_comm = len(set(communities.values())) if communities else 0
        mod = None
        if n_comm > 1:
            groups = {}
            for n, c in communities.items():
                groups.setdefault(c, set()).add(n)
            try:
                mod = float(nx.community.modularity(G, list(groups.values()), weight=None))
            except Exception:
                mod = None
        return {"elements": {"nodes": nodes, "edges": edges},
                "stats": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                          "communities": n_comm, "modularity": mod,
                          "by_type": {t: sum(1 for _, d in G.nodes(data=True) if d["type"] == t)
                                      for t in ("job", "company", "skill")}}}

    # ------------------------------------------------------------------ recommendation
    def cooccurrence(self):
        if self._cooc is None:
            C = (self.X.T @ self.X).toarray().astype(np.float64)
            f = np.diag(C).copy()
            np.fill_diagonal(C, 0)
            p_ij = C / self.n
            p_i = f / self.n
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.log(p_ij / np.outer(p_i, p_i))
            pmi[~np.isfinite(pmi)] = 0
            self._cooc = (C, np.clip(pmi, 0, None))
        return self._cooc

    def recommend_skills(self, seeds: List[str], job: Optional[str] = None, k: int = 10,
                         alpha: float = 0.85, min_support: int = 30) -> List[dict]:
        """Personalised PageRank on the skill co-occurrence graph.

        Graph: skill -> skill edges weighted by the conditional probability P(t | s), kept only
        for *positively associated* pairs (PMI > 0, i.e. lift > 1) with >= `min_support` shared
        postings.  The walk restarts on the seed skills (and, if a job is given, on the skills
        typical of that job).  Ranking = PPR score x log(lift) so that hub skills present in every
        job (python, sql) do not dominate.  Evidence reported per recommendation: the average
        co-occurrence rate P(t | seed) over the seeds and the lift P(t | seed) / P(t)."""
        C, pmi = self.cooccurrence()
        f = self.skill_freq.astype(np.float64)
        cond = C / np.clip(f[:, None], 1, None)                 # cond[s, t] = P(t | s)
        W = cond * ((pmi > 0) & (C >= min_support))
        deg = W.sum(1, keepdims=True); deg[deg == 0] = 1
        P = W / deg
        seed_idx = [self.s2i[s] for s in seeds if s in self.s2i]
        p0 = np.zeros(len(self.skills))
        p0[seed_idx] = 1.0
        if job and job in self.j2i:
            jm = self.job == self.j2i[job]
            share = np.asarray(self.Xr[jm].sum(0)).ravel() / max(jm.sum(), 1)
            p0 += share * (len(seed_idx) or 1) * 0.5
        if p0.sum() == 0:
            return []
        p0 /= p0.sum()
        r = p0.copy()
        for _ in range(60):
            r = (1 - alpha) * p0 + alpha * (P.T @ r)
        base = self.skill_freq / self.n
        if seed_idx:
            avg_cond = cond[seed_idx].mean(0)
        else:
            jm = self.job == self.j2i[job]
            avg_cond = np.asarray(self.Xr[jm].sum(0)).ravel() / max(jm.sum(), 1)
        exclude = set(seed_idx)
        if job and job in self.j2i and not seed_idx:
            exclude = set()
        lift = np.where(base > 0, avg_cond / np.clip(base, 1e-12, None), 0)
        # final relevance = PPR mass x positive association (log-lift): the walk alone drifts
        # toward hub skills (python, sql) that every job lists
        score = r * np.clip(np.log(np.clip(lift, 1e-12, None)), 0, None)
        out = []
        for i in np.argsort(-score):
            if score[i] <= 0:
                break
            if i in exclude or self.skill_freq[i] < min_support:
                continue
            out.append({"skill": self.skills[i], "category": self.extractor.skill2cat.get(self.skills[i], "Other"),
                        "score": float(score[i]), "ppr": float(r[i]), "cooccurrence": float(avg_cond[i]),
                        "lift": float(avg_cond[i] / base[i]) if base[i] else 0.0})
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ Q&A
    def parse_question(self, q: str) -> dict:
        ql = " " + re.sub(r"\s+", " ", q.lower()) + " "
        jobs = []
        work = ql
        for phrase in sorted(JOB_SYNONYMS, key=len, reverse=True):
            pat = r"(?<![a-z])" + re.escape(phrase) + r"s?(?![a-z])"
            if re.search(pat, work):
                jobs.append(JOB_SYNONYMS[phrase]); work = re.sub(pat, " ", work)
        country = None
        for c in sorted(self.countries, key=len, reverse=True):
            if c and c != "Unknown" and re.search(r"(?<![a-z])" + re.escape(c.lower()) + r"(?![a-z])", ql):
                country = c; break
        if not country:
            for a, c in COUNTRY_ALIASES.items():
                if re.search(r"(?<![a-z.])" + re.escape(a) + r"(?![a-z])", ql) and c in self.countries:
                    country = c; break
        # skills: run the extractor on the question with job titles removed
        text_wo_jobs = q
        for phrase in sorted(JOB_SYNONYMS, key=len, reverse=True):
            text_wo_jobs = re.sub(re.escape(phrase), " ", text_wo_jobs, flags=re.I)
        skills = [s for s in self.extractor.extract_names(text_wo_jobs) if s in self.s2i]
        if any(w in ql for w in (" salary", " salaries", " pay ", " paid", " earn")):
            intent = "salary"
        elif any(w in ql for w in ("related", "recommend", "learn next", "complement", "similar skill",
                                   "go with", "should i learn", "pair")):
            intent = "recommend"
        elif any(w in ql for w in ("compan", "employer", "who hires", "who is hiring", "hiring",
                                   "organisation", "organization", "firms")):
            intent = "companies"
        elif any(w in ql for w in (" countr", " where ")):
            intent = "countries"
        elif any(w in ql for w in ("which job", "what job", "which role", "what role", "position",
                                   "which profession", "jobs require", "jobs need", "roles")):
            intent = "jobs"
        elif "skill" in ql or "need" in ql or "require" in ql or "tool" in ql or "technolog" in ql:
            intent = "skills" if jobs else ("jobs" if skills else "unknown")
        else:
            intent = "companies" if skills else ("skills" if jobs else "unknown")
        return {"intent": intent, "jobs": jobs, "skills": skills, "country": country}

    def answer(self, question: str) -> dict:
        p = self.parse_question(question)
        intent, jobs, skills, country = p["intent"], p["jobs"], p["skills"], p["country"]
        where = []
        if jobs: where.append(" / ".join(jobs) + " postings")
        if skills: where.append("requiring " + " + ".join(skills))
        if country: where.append(f"in {country}")
        scope = " ".join(where) if where else "all postings"
        res = {"question": question, "parsed": p, "table": [], "columns": [], "graph": None}

        if intent == "unknown":
            res["answer"] = ("I could not recognise a skill, job category or country in the question. "
                             "Try e.g. 'Which companies offer jobs requiring Python and AWS?' or "
                             "'What skills does a Data Engineer need?'")
            return res

        m = self._mask(jobs=jobs or None, skills_all=skills or None, country=country)
        n = int(m.sum())
        res["n_postings"] = n
        if n == 0:
            res["answer"] = f"No posting in the dataset matches the query ({scope})."
            return res
        idx = np.where(m)[0]

        if intent == "companies":
            cc = np.bincount(self.company[idx], minlength=len(self.companies))
            for bad in ("Confidential", "Unknown"):
                if bad in self.companies: cc[self.companies.index(bad)] = 0
            top = [c for c in np.argsort(-cc)[:15] if cc[c] > 0]
            res["columns"] = ["Company", "Matching postings", "Share of matches"]
            res["table"] = [[self.companies[c], int(cc[c]), f"{cc[c]/n:.1%}"] for c in top]
            names = ", ".join(f"{self.companies[c]} ({cc[c]})" for c in top[:5])
            res["answer"] = (f"{n:,} postings match the query ({scope}), from {int((cc>0).sum()):,} companies. "
                             f"Top employers: {names}.")
            res["graph"] = self._answer_graph(idx, companies=top[:10], focus_skills=skills)
        elif intent == "skills":
            sc = np.asarray(self.Xr[idx].sum(0)).ravel()
            top = [s for s in np.argsort(-sc)[:15] if sc[s] > 0 and self.skills[s] not in skills]
            res["columns"] = ["Skill", "Category", "Postings", "Share"]
            res["table"] = [[self.skills[s], self.extractor.skill2cat.get(self.skills[s], ""),
                             int(sc[s]), f"{sc[s]/n:.1%}"] for s in top]
            names = ", ".join(f"{self.skills[s]} ({sc[s]/n:.0%})" for s in top[:6])
            res["answer"] = f"Across {n:,} postings ({scope}), the most requested skills are {names}."
            res["graph"] = self._answer_graph(idx, n_skills=12, companies=None)
        elif intent == "jobs":
            jc = np.bincount(self.job[idx], minlength=len(self.jobs))
            base = self.job_freq / self.n
            top = [j for j in np.argsort(-jc) if jc[j] > 0]
            res["columns"] = ["Job category", "Postings", "Share of matches", "Over-representation"]
            res["table"] = [[self.jobs[j], int(jc[j]), f"{jc[j]/n:.1%}", f"x{(jc[j]/n)/base[j]:.2f}"]
                            for j in top]
            names = ", ".join(f"{self.jobs[j]} ({jc[j]/n:.0%})" for j in top[:4])
            res["answer"] = f"{n:,} postings match the query ({scope}). They are mostly {names}."
            res["graph"] = self._answer_graph(idx, n_skills=10, companies=None, focus_skills=skills)
        elif intent == "countries":
            cc = np.bincount(self.country[idx], minlength=len(self.countries))
            top = [c for c in np.argsort(-cc)[:15] if cc[c] > 0]
            res["columns"] = ["Country", "Postings", "Share"]
            res["table"] = [[self.countries[c], int(cc[c]), f"{cc[c]/n:.1%}"] for c in top]
            names = ", ".join(f"{self.countries[c]} ({cc[c]:,})" for c in top[:5])
            res["answer"] = f"{n:,} postings match the query ({scope}). Leading countries: {names}."
        elif intent == "salary":
            sal = self.salary[idx]; sal = sal[np.isfinite(sal)]
            if len(sal) < 5:
                res["answer"] = (f"{n:,} postings match the query ({scope}), but only {len(sal)} report a yearly "
                                 "salary - too few for a reliable estimate.")
            else:
                q1, med, q3 = np.percentile(sal, [25, 50, 75])
                res["columns"] = ["Statistic", "Yearly salary (USD)"]
                res["table"] = [["Postings with salary", f"{len(sal):,}"], ["25th percentile", f"{q1:,.0f}"],
                                ["Median", f"{med:,.0f}"], ["75th percentile", f"{q3:,.0f}"]]
                res["answer"] = (f"Median advertised yearly salary for {scope}: ${med:,.0f} "
                                 f"(IQR ${q1:,.0f}-${q3:,.0f}, based on the {len(sal):,} postings that "
                                 f"publish a salary, {len(sal)/n:.1%} of matches).")
        elif intent == "recommend":
            recs = self.recommend_skills(skills, job=jobs[0] if jobs else None, k=12)
            res["columns"] = ["Recommended skill", "Category", "Relevance (PPR x log-lift)", "Avg. co-occurrence", "Lift"]
            res["table"] = [[r["skill"], r["category"], f"{r['score']:.4f}", f"{r['cooccurrence']:.1%}",
                             f"x{r['lift']:.2f}"] for r in recs]
            seed_txt = " + ".join(skills) if skills else (jobs[0] if jobs else "")
            res["answer"] = (f"Skills most connected to {seed_txt} in the graph (Personalised PageRank weighted by lift): "
                             + ", ".join(r["skill"] for r in recs[:6]) + ".")
            res["graph"] = self._answer_graph(idx, n_skills=12, companies=None,
                                              focus_skills=skills + [r["skill"] for r in recs[:8]])
        return res

    def _answer_graph(self, idx, companies=None, n_skills=10, focus_skills=None):
        n_comp = 10 if companies is not None else 0
        G = self.build_graph(n_companies=n_comp, n_skills=n_skills, rows=idx, min_share=0.15,
                             company_skill_k=2, focus_skills=focus_skills)
        return self.to_cytoscape(G)
