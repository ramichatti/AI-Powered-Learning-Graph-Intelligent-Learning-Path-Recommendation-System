"""Dataset loading, cleaning and text construction for the data_jobs dataset."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .skills import SkillExtractor, parse_skill_list

TARGET = "job_title_short"
KEEP_COLUMNS = [
    "job_title_short", "job_title", "company_name", "job_location", "job_country", "job_via",
    "job_schedule_type", "job_work_from_home", "job_posted_date", "job_no_degree_mention",
    "job_health_insurance", "salary_year_avg", "salary_hour_avg", "job_skills",
]


def find_raw_dataset(project_root: str | Path) -> Path:
    root = Path(project_root)
    for cand in (root / "data" / "raw" / "data_jobs.csv", root / "data_jobs.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "data_jobs.csv not found. Put it in data/raw/ (download: "
        "https://huggingface.co/datasets/lukebarousse/data_jobs)")


def normalize_text(s: str) -> str:
    """Unicode NFKC, unify dashes/quotes, collapse whitespace (case preserved)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"[‘’“”]", "'", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_for_tfidf(s: str) -> str:
    """Lower-case, keep characters that matter in tech names (+ # . /), drop the rest."""
    s = normalize_text(s).lower()
    s = re.sub(r"[^a-z0-9+#./ ]", " ", s)
    s = re.sub(r"(?<![a-z0-9])[./]|[./](?![a-z0-9])", " ", s)   # stray dots/slashes
    return re.sub(r"\s+", " ", s).strip()


def skill_analyzer(text: str) -> List[str]:
    """TF-IDF analyzer for the skills-only model: one token per skill ("power bi" stays whole)."""
    return [t.strip() for t in text.split("|") if t.strip()]


def clean_dataset(df: pd.DataFrame, extractor: SkillExtractor) -> tuple[pd.DataFrame, dict]:
    """Returns the cleaned dataframe and a report of every cleaning step (row counts)."""
    report = {"raw_rows": int(len(df))}
    df = df[KEEP_COLUMNS].copy()

    n0 = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = int(n0 - len(df))

    for c in ["job_title", "company_name", "job_location", "job_country", "job_via",
              "job_schedule_type"]:
        df[c] = df[c].map(normalize_text)
    df["job_via"] = df["job_via"].str.replace(r"^via\s+", "", regex=True, flags=re.I)
    df["job_location"] = df["job_location"].replace("", "Unknown")
    df["job_schedule_type"] = df["job_schedule_type"].replace("", "Unknown")
    df["job_country"] = df["job_country"].replace("", "Unknown")

    # skills: stringified list -> canonical python list
    df["skills"] = df["job_skills"].map(parse_skill_list).map(
        lambda lst: sorted({s for s in (extractor.normalise_tag(x) for x in lst) if s}))
    df["n_skills"] = df["skills"].map(len)
    report["rows_without_skills"] = int((df["n_skills"] == 0).sum())
    df = df[df["n_skills"] > 0]
    report["rows_after_dropping_no_skills"] = int(len(df))

    df = df[df["job_title"].str.len() > 0]

    # near-duplicates: the same ad re-posted on several platforms (job_via) would leak
    # between train and test -> keep one copy per (title, company, location, skills)
    df["skills_str"] = df["skills"].map(lambda s: "|".join(s))
    n0 = len(df)
    df = df.drop_duplicates(subset=["job_title", "company_name", "job_location", "skills_str"])
    report["cross_platform_duplicates_removed"] = int(n0 - len(df))

    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], errors="coerce")
    df["salary_year_avg"] = pd.to_numeric(df["salary_year_avg"], errors="coerce")

    # text views used by the models
    df["text_title_skills"] = (df["job_title"] + " | skills: " + df["skills"].map(", ".join))
    df["text_tfidf"] = df["text_title_skills"].map(clean_for_tfidf)
    df["text_embed"] = df["job_title"] + ". Skills: " + df["skills"].map(", ".join) + "."
    df = df.drop(columns=["job_skills"]).reset_index(drop=True)
    df.insert(0, "job_id", np.arange(len(df)))
    report["final_rows"] = int(len(df))
    return df, report
