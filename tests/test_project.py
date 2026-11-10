"""End-to-end checks of the saved artifacts and of every Flask endpoint.

Run with:  python -m pytest -q tests        (or simply: python tests/test_project.py)
Requires the artifacts produced by the notebook."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobnlp.skills import SkillExtractor  # noqa: E402


# ----------------------------------------------------------------------------- skill extractor
def test_skill_extractor_rules():
    ex = SkillExtractor()
    found = ex.extract_names("Python, PySpark, Power-BI, sklearn, SQL Server, T-SQL, C++, C#, node.js, R and Go. "
                             "R&D team, C-level stakeholders, go to market.")
    for s in ["python", "pyspark", "power bi", "scikit-learn", "sql server", "t-sql", "c++", "c#", "node.js", "r", "go"]:
        assert s in found, s
    assert found.count("r") == 1 and "c" not in found          # R&D and C-level rejected


def test_skill_extractor_empty():
    assert SkillExtractor().extract("") == []


# ----------------------------------------------------------------------------- Flask API
def _client():
    from app.app import create_app
    app = create_app(); app.testing = True
    return app.test_client()


def test_pages_render():
    c = _client()
    for url in ["/", "/classification", "/skills", "/semantic-search", "/clustering", "/analytics", "/knowledge-graph"]:
        r = c.get(url)
        assert r.status_code == 200, url
        assert b"JobLens" in r.data
    assert c.get("/does-not-exist").status_code == 404


def test_health_all_files_present():
    h = _client().get("/api/health").get_json()
    assert all(h["files"].values()), h["files"]
    assert h["distilbert_dir"] is not None


def test_classify():
    c = _client()
    r = c.post("/api/classify", json={"text": "Senior Data Engineer: Spark, Kafka, Airflow, AWS, Python"}).get_json()
    assert r["prediction"] == "Senior Data Engineer"
    assert abs(sum(p["p"] for p in r["probabilities"]) - 1) < 1e-6
    assert c.post("/api/classify", json={"text": "   "}).status_code == 400
    assert c.post("/api/classify", json={"text": "12345 !!!"}).status_code == 400
    assert c.post("/api/classify", data="not json", content_type="text/plain").status_code == 400


def test_skills_endpoint():
    r = _client().post("/api/skills", json={"text": "Tableau and Power BI dashboards, SQL, Excel"}).get_json()
    names = {s["skill"] for s in r["skills"]}
    assert {"tableau", "power bi", "sql", "excel"} <= names
    assert r["recommended"]


def test_semantic_search():
    c = _client()
    r = c.post("/api/search", json={"query": "deep neural networks for image recognition", "k": 5}).get_json()
    assert len(r["results"]) == 5
    sims = [x["similarity"] for x in r["results"]]
    assert sims == sorted(sims, reverse=True)
    assert c.post("/api/search", json={"query": "python", "k": 500}).status_code == 400
    f = c.post("/api/search", json={"query": "sql dashboards", "k": 3, "category": "Data Analyst"}).get_json()
    assert all(x["category"] == "Data Analyst" for x in f["results"])


def test_clusters_and_graph():
    c = _client()
    d = c.get("/api/clusters").get_json()
    assert d["k"] == len(d["profiles"]) and d["points"]
    det = c.get("/api/clusters/0").get_json()
    assert det["ads"]
    assert c.get(f"/api/clusters/{d['k']}").status_code == 400
    g = c.get("/api/graph?companies=10&skills=15").get_json()
    assert g["stats"]["nodes"] > 0 and g["elements"]["edges"]
    nb = c.get("/api/graph/neighborhood?type=skill&label=python").get_json()
    assert nb["stats"]["nodes"] > 5
    qa = c.post("/api/qa", json={"question": "Which jobs require Spark and Kafka?"}).get_json()
    assert qa["parsed"]["intent"] == "jobs" and qa["table"]
    assert c.post("/api/qa", json={"question": ""}).status_code == 400


def test_summary_and_analytics():
    c = _client()
    s = c.get("/api/summary").get_json()
    assert s["analytics"]["totals"]["postings"] > 100_000
    a = c.get("/api/analytics").get_json()
    assert a["classification"]["accuracy"] > 0.5


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
