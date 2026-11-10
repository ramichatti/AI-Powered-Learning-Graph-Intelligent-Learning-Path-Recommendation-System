"""Flask application - Job Offers NLP Analytics.

Pages render HTML shells; all data is fetched from the JSON API below, which only
reads saved artifacts (models/, data/processed/, results/metrics/) - nothing is
retrained when the app starts."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import artifacts as A  # noqa: E402
from app.services import classification, clustering, knowledge_graph, semantic_search, skill_extraction  # noqa: E402
from app.services.classification import InputError  # noqa: E402

PAGES = [
    # endpoint, url, nav label, icon, nav group
    ("dashboard", "/", "Overview", "grid", "Start"),
    ("classification", "/classification", "Classify a job", "tag", "Tools"),
    ("skills", "/skills", "Extract skills", "spark", "Tools"),
    ("semantic_search", "/semantic-search", "Find similar jobs", "search", "Tools"),
    ("knowledge_graph", "/knowledge-graph", "Ask the graph", "graph", "Tools"),
    ("clustering", "/clustering", "Job families", "clusters", "Explore"),
    ("analytics", "/analytics", "Market insights", "chart", "Explore"),
]


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
    log = logging.getLogger("jobnlp.app")

    @app.context_processor
    def inject_nav():
        return {"pages": PAGES, "active": request.endpoint}

    # ------------------------------------------------------------------ pages
    def page(endpoint, template):
        def view():
            return render_template(template)
        view.__name__ = endpoint
        return view

    for ep, url, *_ in PAGES:
        app.add_url_rule(url, ep, page(ep, f"{ep}.html"))

    # ------------------------------------------------------------------ errors
    @app.errorhandler(InputError)
    def input_error(e):
        return jsonify({"error": str(e), "kind": "input"}), 400

    @app.errorhandler(A.ArtifactError)
    def artifact_error(e):
        return jsonify({"error": str(e), "kind": "artifact"}), 503

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Unknown API endpoint.", "kind": "not_found"}), 404
        return render_template("error.html", code=404, message="This page does not exist."), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "Request too large (max 1 MB).", "kind": "input"}), 413

    @app.errorhandler(Exception)
    def unexpected(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return jsonify({"error": e.description, "kind": "http"}), e.code
        log.exception("Unhandled error")
        return jsonify({"error": f"Internal error: {type(e).__name__}: {e}", "kind": "server"}), 500

    def body() -> dict:
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict() if request.form else None
        if not isinstance(data, dict):
            raise InputError("Expected a JSON body.")
        return data

    def opt_int(name, default=None):
        v = request.args.get(name)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except ValueError:
            raise InputError(f"'{name}' must be an integer.")

    # ------------------------------------------------------------------ API
    @app.get("/api/health")
    def health():
        return jsonify(A.status())

    @app.get("/api/summary")
    def api_summary():
        an = A.get("analytics", lambda: A.load_json("analytics.json"))
        out = {"analytics": an}
        try:
            out["classification"] = classification.model_info()
        except A.ArtifactError as e:
            out["classification_error"] = str(e)
        try:
            cfg = clustering.config()
            out["clusters"] = [{"cluster": p["cluster"], "label": p["label"], "size": p["size"],
                                "top_skills": p["top_skills"][:5], "top_categories": p["top_categories"][:2]}
                               for p in cfg["profiles"]]
            out["clustering_metrics"] = {"k": cfg["k_selected"], "silhouette": cfg["final_silhouette"],
                                         "nmi": cfg["nmi_vs_job_title_short"]}
        except A.ArtifactError as e:
            out["clusters_error"] = str(e)
        return jsonify(out)

    @app.get("/api/analytics")
    def api_analytics():
        an = A.get("analytics", lambda: A.load_json("analytics.json"))
        cfg = clustering.config()
        return jsonify({"analytics": an, "classification": classification.model_info(),
                        "clusters": cfg["profiles"], "clustering": {k: cfg[k] for k in (
                            "k_range", "inertia", "silhouette", "k_selected", "k_elbow", "final_silhouette",
                            "ari_vs_job_title_short", "nmi_vs_job_title_short", "skills_by_cluster")}})

    @app.post("/api/classify")
    def classify():
        return jsonify(classification.predict(body().get("text")))

    @app.get("/api/classification/info")
    def classification_info():
        return jsonify(classification.model_info())

    @app.post("/api/skills")
    def api_skills():
        return jsonify(skill_extraction.extract(body().get("text")))

    @app.get("/api/skills/dictionary")
    def skills_dictionary():
        return jsonify(skill_extraction.dictionary_info())

    @app.post("/api/search")
    def search():
        b = body()
        cluster = b.get("cluster")
        cluster = int(cluster) if cluster not in (None, "", "all") else None
        return jsonify(semantic_search.search(b.get("query"), k=b.get("k", 10),
                                              category=b.get("category") or None, cluster=cluster))

    @app.get("/api/similar/<int:job_id>")
    def similar(job_id):
        return jsonify(semantic_search.similar_to(job_id, k=opt_int("k", 8)))

    @app.get("/api/random-ad")
    def random_ad():
        return jsonify(semantic_search.random_ad(opt_int("seed")))

    @app.get("/api/clusters")
    def clusters():
        return jsonify(clustering.overview())

    @app.get("/api/clusters/<int:cid>")
    def cluster_detail(cid):
        return jsonify(clustering.detail(cid, page=opt_int("page", 1), q=request.args.get("q", "")[:60]))

    @app.get("/api/graph")
    def graph():
        jobs = [j for j in request.args.get("jobs", "").split("|") if j]
        return jsonify(knowledge_graph.graph(
            jobs=jobs, n_companies=opt_int("companies", 25), n_skills=opt_int("skills", 35),
            min_share=float(request.args.get("min_share", 0.12) or 0.12), cluster=opt_int("cluster"),
            company_skill_k=opt_int("company_skills", 3), focus=request.args.get("focus")))

    @app.get("/api/graph/neighborhood")
    def graph_neighborhood():
        kind, label = request.args.get("type", ""), request.args.get("label", "")
        if kind not in ("skill", "job", "company") or not label:
            raise InputError("Parameters 'type' (skill|job|company) and 'label' are required.")
        return jsonify(knowledge_graph.neighborhood(kind, label))

    @app.get("/api/graph/search")
    def graph_search():
        return jsonify(knowledge_graph.search_nodes(request.args.get("q", "")[:60]))

    @app.post("/api/qa")
    def qa():
        return jsonify(knowledge_graph.answer(body().get("question")))

    @app.get("/api/qa/examples")
    def qa_examples():
        try:
            return jsonify(A.load_json("kg/qa_examples.json"))
        except FileNotFoundError:
            return jsonify([])

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
