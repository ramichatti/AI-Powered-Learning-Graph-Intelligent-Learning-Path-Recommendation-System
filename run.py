"""Launch the JobLens Flask application.

    python run.py                 # http://127.0.0.1:5000
    python run.py --port 8000 --host 0.0.0.0
"""
import argparse
import faulthandler
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Start-up diagnostics: progress lines go to run_log.txt too, and if start-up hangs the
# Python stack of every thread is written there every 60 s (faulthandler).
_LOG = open(ROOT / "run_log.txt", "w", encoding="utf-8", buffering=1)
faulthandler.dump_traceback_later(60, repeat=True, file=_LOG)
_T0 = time.time()


def log(msg: str):
    line = f"[{time.time() - _T0:6.1f}s] {msg}"
    print(line, flush=True)
    _LOG.write(line + "\n")


def check_artifacts() -> list:
    m = ROOT / "models"
    required = ["classification_pipeline.joblib", "kmeans_model.joblib", "job_embeddings.npy", "analytics.json",
                "clustering_config.json", "model_config.json", "kg/postings.npz"]
    missing = [f for f in required if not (m / f).exists()]
    if not (ROOT / "data" / "processed" / "embedding_corpus.csv").exists():
        missing.append("data/processed/embedding_corpus.csv")
    return missing


def main():
    ap = argparse.ArgumentParser(description="JobLens - data-job NLP analytics (Flask)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-warmup", action="store_true", help="do not pre-load models at start-up")
    args = ap.parse_args()

    log(f"Python {sys.version.split()[0]} - checking artifacts")
    missing = check_artifacts()
    if missing:
        print("WARNING - missing artifacts:", ", ".join(missing))
        print("Run notebooks/job_offers_nlp_analysis.ipynb first. The app will start, and the affected pages will show an error.\n")

    log("importing numpy / pandas / scikit-learn (first start can take a while) ...")
    import numpy, pandas, sklearn  # noqa: F401
    log(f"numpy {numpy.__version__}, pandas {pandas.__version__}, scikit-learn {sklearn.__version__}")
    from app.app import app
    log("Flask app created")
    if not args.no_warmup and not missing:
        from app.services import artifacts, classification, knowledge_graph, semantic_search
        for name, fn in [("classifier", classification._main), ("knowledge graph", knowledge_graph.get_kg),
                         ("embedding index", semantic_search.get_index), ("DistilBERT", semantic_search.get_encoder)]:
            try:
                fn(); log(f"loaded {name}")
            except artifacts.ArtifactError as e:
                log(f"could not load {name}: {e}")
    faulthandler.cancel_dump_traceback_later()
    log(f"JobLens running on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
