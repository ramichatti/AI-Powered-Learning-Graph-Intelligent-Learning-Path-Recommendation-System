"""Job category prediction with the saved TF-IDF + Logistic Regression pipelines."""
from __future__ import annotations

import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from jobnlp.preprocessing import clean_for_tfidf

from . import artifacts as A
from .skill_extraction import get_extractor

MAX_CHARS = 20_000


class InputError(ValueError):
    pass


def validate_text(text, min_chars: int = 3, field: str = "text") -> str:
    if not isinstance(text, str):
        raise InputError(f"'{field}' must be a string.")
    text = text.strip()
    if not text:
        raise InputError("Please enter some text - the field is empty.")
    if len(text) < min_chars:
        raise InputError(f"The text is too short (minimum {min_chars} characters).")
    if len(text) > MAX_CHARS:
        raise InputError(f"The text is too long ({len(text):,} characters, maximum {MAX_CHARS:,}).")
    if not re.search(r"[A-Za-z]", text):
        raise InputError("The text must contain words (letters), not only numbers or symbols.")
    return text


def _main():
    return A.get("classification_pipeline", lambda: joblib.load(A.MODELS / "classification_pipeline.joblib"))


def _skills_only():
    return A.get("skills_only_pipeline", lambda: joblib.load(A.MODELS / "skills_only_pipeline.joblib"))


def _metrics():
    return A.get("classification_metrics", lambda: __import__("json").loads(
        (A.ROOT / "results" / "metrics" / "classification_metrics.json").read_text()))


def predict(text: str, top_k: int = 10) -> dict:
    text = validate_text(text)
    pipe = _main()
    skills = get_extractor().extract_names(text)
    doc = clean_for_tfidf(f"{text} | skills: {', '.join(skills)}")
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    X = vec.transform([doc])
    known_terms = int(X.nnz)
    proba = clf.predict_proba(X)[0]
    order = np.argsort(-proba)
    pred = clf.classes_[order[0]]
    # explanation: contribution of each present n-gram to the predicted class logit
    ci = order[0]
    fn = A.get("tfidf_feature_names", vec.get_feature_names_out)
    dense = X.toarray()[0]
    contrib = dense * clf.coef_[ci]
    nz = np.nonzero(dense)[0]
    # hide n-grams made only of stop-words ("and", "we is") - they carry weight in the model
    # (titles contain them) but are meaningless as an explanation
    def informative(term):
        return any(t not in ENGLISH_STOP_WORDS and len(t) > 1 for t in term.split())
    top_terms = sorted(((fn[i], float(contrib[i])) for i in nz if informative(fn[i])), key=lambda t: -t[1])[:8]
    result = {
        "prediction": pred, "confidence": float(proba[ci]),
        "probabilities": [{"label": clf.classes_[i], "p": float(proba[i])} for i in order[:top_k]],
        "top_terms": [{"term": t, "weight": w} for t, w in top_terms if w > 0],
        "known_terms": known_terms, "skills": skills, "warning": None,
    }
    if known_terms == 0:
        result["warning"] = ("None of the words of this text exist in the model's vocabulary - "
                             "the prediction is just the class prior and should not be trusted.")
    elif proba[ci] < 0.4:
        result["warning"] = "Low confidence: the text matches several job categories."
    if skills:
        sp = _skills_only()
        p2 = sp.predict_proba(["|".join(skills)])[0]
        o2 = np.argsort(-p2)
        result["skills_only"] = {"prediction": sp.classes_[o2[0]], "confidence": float(p2[o2[0]]),
                                 "probabilities": [{"label": sp.classes_[i], "p": float(p2[i])} for i in o2[:5]]}
    else:
        result["skills_only"] = None
    return result


def model_info() -> dict:
    m = _metrics()
    main = next(r for r in m["models"] if "MAIN" in r["model"])
    sk = next(r for r in m["models"] if "skills only" in r["model"])
    rep = m["classification_report"]
    return {"accuracy": main["accuracy"], "f1_macro": main["f1_macro"], "precision_macro": main["precision_macro"],
            "recall_macro": main["recall_macro"], "skills_only_accuracy": sk["accuracy"],
            "skills_only_f1_macro": sk["f1_macro"], "train": m["split"]["train"], "test": m["split"]["test"],
            "best_C": m["best_C"], "labels": m["labels"], "confusion_matrix": m["confusion_matrix"],
            "per_class": [{"label": l, "precision": rep[l]["precision"], "recall": rep[l]["recall"],
                           "f1": rep[l]["f1-score"], "support": rep[l]["support"],
                           "f1_skills_only": m["skills_only_report"][l]["f1-score"]} for l in m["labels"]],
            "models": m["models"], "top_confusions": m["top_confusions"]}
