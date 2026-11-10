"""Skill extraction component.

Approach (documented in the notebook and README):

1. **Skill dictionary (gazetteer)** - the canonical vocabulary is the set of 250+
   technologies that appear in the dataset's own `job_type_skills` column, re-organised
   into project categories (programming languages, BI tools, ML frameworks, ...), plus a
   small set of data-domain concepts (machine learning, ETL, ...) that are frequent in
   real job ads but absent from the dataset vocabulary.
2. **Alias normalisation** - spelling variants are mapped to one canonical name
   (`powerbi`, `power-bi`, `Power BI` -> `power bi`; `sklearn` -> `scikit-learn`; ...).
3. **Rule-based phrase matching** - one compiled regular expression, longest
   alternatives first (so `sql server` wins over `sql`, `t-sql` over `sql`), with
   custom token boundaries that respect `c++`, `c#`, `node.js`, `.net`.
4. **Ambiguity rules** - single letters and common English words that are also tools
   (`R`, `C`, `Go`, `Word`, `Swift`, `Spring`, ...) are only matched in their
   case-sensitive technical spelling, and `R`/`C` additionally reject `R&D`, `C-level`.

Limitations: no context understanding beyond these rules (e.g. "Excel at teamwork"
will still match Excel), unseen technologies are not detected, and the category of a
skill is fixed (SAS is listed as an analysis tool even when used as a language).
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

# ----------------------------------------------------------------------------
# Project categories -> canonical skills.  Built from the dataset vocabulary.
# ----------------------------------------------------------------------------
SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Programming Languages": [
        "sql", "python", "r", "java", "scala", "go", "javascript", "c++", "c#", "c", "vba",
        "matlab", "bash", "shell", "t-sql", "html", "powershell", "css", "typescript", "perl",
        "ruby", "php", "julia", "kotlin", "visual basic", "rust", "assembly", "swift", "groovy",
        "vb.net", "solidity", "sass", "cobol", "elixir", "clojure", "fortran", "haskell", "dart",
        "apl", "lua", "objective-c", "f#", "erlang", "delphi", "lisp", "pascal", "ocaml", "crystal",
        "nosql",
    ],
    "BI & Visualization Tools": [
        "tableau", "power bi", "looker", "qlik", "ssrs", "cognos", "microstrategy", "dax",
        "alteryx", "plotly", "matplotlib", "seaborn", "ggplot2", "rshiny",
    ],
    "Data Analysis & Office Tools": [
        "excel", "sas", "spss", "word", "powerpoint", "sap", "sheets", "spreadsheet", "ms access",
        "outlook", "sharepoint", "visio", "splunk", "datarobot", "nuix", "esquisse", "jupyter",
        "pandas", "numpy", "tidyverse", "dplyr", "tidyr",
    ],
    "Machine Learning Frameworks": [
        "tensorflow", "pytorch", "scikit-learn", "keras", "hugging face", "nltk", "opencv",
        "mxnet", "theano", "mlr", "dlib", "mlpack", "chainer", "shogun", "watson",
    ],
    "Data Engineering & Big Data": [
        "spark", "pyspark", "hadoop", "kafka", "airflow", "databricks", "snowflake", "ssis",
        "flow", "gdpr",
    ],
    "Databases": [
        "sql server", "mysql", "postgresql", "mongodb", "cassandra", "elasticsearch", "dynamodb",
        "redis", "db2", "neo4j", "mariadb", "couchbase", "sqlite", "firestore", "couchdb",
        "oracle", "redshift", "bigquery", "aurora", "firebase",
    ],
    "Cloud Platforms": [
        "aws", "azure", "gcp", "ibm cloud", "vmware", "openstack", "digitalocean", "heroku",
        "ovh", "linode", "colocation",
    ],
    "DevOps & Engineering Tools": [
        "git", "docker", "kubernetes", "terraform", "github", "jenkins", "gitlab", "ansible",
        "bitbucket", "chef", "puppet", "yarn", "terminal", "svn", "pulumi", "npm", "codecommit",
        "homebrew", "linux", "windows", "unix", "ubuntu", "redhat", "macos", "centos", "debian",
        "arch", "suse", "wsl", "fedora", "kali",
    ],
    "Web & App Frameworks": [
        "react", "spring", "graphql", "selenium", "flutter", "electron", "qt", "ionic", "cordova",
        "xamarin", "capacitor", "express", "angular", "flask", "django", "node.js", "vue",
        "fastapi", "phoenix", "jquery", "asp.net", "ruby on rails", "laravel", "next.js", "drupal",
        "asp.net core", "symfony", "svelte", "blazor", "play framework", "gatsby", "fastify",
        "nuxt.js", "ember.js", "deno", "unity", "unreal",
    ],
    "Collaboration Tools": [
        "jira", "confluence", "atlassian", "notion", "planner", "smartsheet", "trello", "asana",
        "airtable", "monday.com", "clickup", "workfront", "wrike", "microsoft lists", "dingtalk",
        "wimi", "slack", "zoom", "unify", "microsoft teams", "wire", "symphony", "twilio", "webex",
        "ringcentral", "google chat", "mattermost", "rocketchat",
    ],
    # Concepts frequent in real job ads but absent from the dataset's tag vocabulary.
    "Data & AI Concepts": [
        "machine learning", "deep learning", "nlp", "computer vision", "statistics",
        "data visualization", "etl", "data warehouse", "data modeling", "a/b testing", "llm",
        "generative ai", "mlops", "data mining", "time series", "big data", "data governance",
        "dbt", "mlflow", "xgboost", "power query",
    ],
}

# alias (lower case) -> canonical
ALIASES: Dict[str, str] = {
    "powerbi": "power bi", "power-bi": "power bi", "microsoft power bi": "power bi",
    "msaccess": "ms access", "microsoft access": "ms access",
    "mongo": "mongodb", "golang": "go", "no-sql": "nosql", "huggingface": "hugging face",
    "sqlserver": "sql server", "ms sql server": "sql server", "mssql": "sql server",
    "visualbasic": "visual basic", "node": "node.js", "nodejs": "node.js", "react.js": "react",
    "reactjs": "react", "vue.js": "vue", "vuejs": "vue", "angular.js": "angular",
    "angularjs": "angular", "asp.netcore": "asp.net core", "rubyon rails": "ruby on rails",
    "rails": "ruby on rails", "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "postgres": "postgresql", "google cloud": "gcp", "google cloud platform": "gcp",
    "amazon web services": "aws", "microsoft azure": "azure", "ms excel": "excel",
    "microsoft excel": "excel", "google sheets": "sheets", "k8s": "kubernetes",
    "apache spark": "spark", "apache kafka": "kafka", "apache airflow": "airflow",
    "apache hadoop": "hadoop", "py spark": "pyspark", "tf.keras": "keras",
    "red hat": "redhat", "mac os": "macos", "ms teams": "microsoft teams",
    "objective c": "objective-c", "c sharp": "c#", "cpp": "c++", "js": "javascript",
    "ts": "typescript", "natural language processing": "nlp", "ml": "machine learning",
    "dl": "deep learning", "large language models": "llm", "large language model": "llm",
    "llms": "llm", "gen ai": "generative ai", "genai": "generative ai",
    "data visualisation": "data visualization", "data warehousing": "data warehouse",
    "data modelling": "data modeling", "ab testing": "a/b testing", "statistical": "statistics",
    "google bigquery": "bigquery", "amazon redshift": "redshift", "elastic search": "elasticsearch",
    "ssas": "ssis",
}

# Matched case-sensitively in their technical spelling only (common English words /
# single letters).  Value = list of accepted literal spellings.
CASE_SENSITIVE = {
    "r": ["R"], "c": ["C"], "go": ["Go", "GoLang", "Golang", "golang"], "word": ["Word", "MS Word"],
    "swift": ["Swift"], "spring": ["Spring", "Spring Boot"], "express": ["Express", "Express.js"],
    "flow": ["Flow"], "arch": ["Arch Linux"], "chef": ["Chef"], "yarn": ["Yarn", "YARN"],
    "unity": ["Unity", "Unity3D"], "notion": ["Notion"], "planner": ["Microsoft Planner", "MS Planner"],
    "wire": ["Wire"], "zoom": ["Zoom"], "sheets": ["Google Sheets", "Sheets"],
    "terminal": ["Terminal"], "symphony": ["Symphony"], "unify": ["Unify"],
    "shell": ["Shell", "shell scripting", "Shell scripting"], "assembly": ["Assembly"],
    "crystal": ["Crystal Reports", "Crystal"], "phoenix": ["Phoenix"], "electron": ["Electron"],
    "capacitor": ["Capacitor"], "oracle": ["Oracle", "ORACLE"], "rust": ["Rust"],
    "julia": ["Julia"], "dart": ["Dart"], "ruby": ["Ruby"], "colocation": ["colocation", "Colocation"],
    "apl": ["APL"], "lisp": ["Lisp", "LISP"], "sass": ["Sass", "SASS"], "qt": ["Qt"],
    "ovh": ["OVH"], "arch_": [], "outlook": ["Outlook"], "mlr": ["mlr"], "deno": ["Deno"],
    "ts": [], "js": [],
}
# aliases that are too ambiguous to use on free text
_BLOCKED_ALIASES = {"ml", "dl", "js", "ts", "node", "rails", "statistical"}


def _boundary_wrap(pat: str) -> str:
    return r"(?<![A-Za-z0-9_+#])" + pat + r"(?![A-Za-z0-9_+#])"


def _literal_pattern(term: str) -> str:
    # spaces / hyphens interchangeable inside multi-word terms
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", term) if p]
    return r"[\s\-]*".join(parts) if len(parts) > 1 else re.escape(term)


class SkillExtractor:
    def __init__(self, categories: Optional[Dict[str, List[str]]] = None,
                 aliases: Optional[Dict[str, str]] = None):
        self.categories = categories or SKILL_CATEGORIES
        self.aliases = dict(aliases or ALIASES)
        self.skill2cat: Dict[str, str] = {}
        for cat, skills in self.categories.items():
            for s in skills:
                self.skill2cat.setdefault(s, cat)
        self._build()

    # -------------------------------------------------------------------
    def _build(self):
        ci_terms: Dict[str, str] = {}   # case-insensitive surface -> canonical
        cs_terms: Dict[str, str] = {}   # case-sensitive surface -> canonical
        for s in self.skill2cat:
            if s in CASE_SENSITIVE:
                for lit in CASE_SENSITIVE[s]:
                    cs_terms[lit] = s
            else:
                ci_terms[s] = s
        for a, s in self.aliases.items():
            if a in _BLOCKED_ALIASES or s not in self.skill2cat:
                continue
            if a == "golang":
                continue
            ci_terms[a] = s
        self._ci_map = {k.lower(): v for k, v in ci_terms.items()}
        self._cs_map = cs_terms
        ci_alt = "|".join(_literal_pattern(t) for t in sorted(ci_terms, key=len, reverse=True))
        cs_alt = "|".join(_literal_pattern(t) for t in sorted(cs_terms, key=len, reverse=True) if t)
        self._ci_re = re.compile(_boundary_wrap(f"(?:{ci_alt})"), re.IGNORECASE)
        self._cs_re = re.compile(_boundary_wrap(f"(?:{cs_alt})"))
        # R / C false positives: R&D, C-level, C-suite, "Plan B / C"
        self._reject = re.compile(r"^(?:R\s*&|C\s*-\s*(?:level|suite|corp)|R\s*-\s*\d)", re.I)

    def _canonical(self, surface: str, case_sensitive: bool) -> Optional[str]:
        norm = re.sub(r"[\s\-]+", " ", surface.strip())
        if case_sensitive:
            for lit, s in self._cs_map.items():
                if re.sub(r"[\s\-]+", " ", lit) == norm:
                    return s
            return None
        low = norm.lower()
        if low in self._ci_map:
            return self._ci_map[low]
        # hyphen/space variants (e.g. "power-bi")
        for k, v in self._ci_map.items():
            if re.sub(r"[\s\-]+", " ", k) == low:
                return v
        return None

    def extract(self, text: str) -> List[dict]:
        """Returns one dict per distinct skill: {skill, category, mentions, spans}."""
        if not isinstance(text, str) or not text.strip():
            return []
        found: Dict[str, dict] = {}
        taken = []
        for regex, cs in ((self._ci_re, False), (self._cs_re, True)):
            for m in regex.finditer(text):
                a, b = m.span()
                if any(a < tb and ta < b for ta, tb in taken):
                    continue
                if cs and self._reject.match(text[a: a + 8]):
                    continue
                skill = self._canonical(m.group(0), cs)
                if not skill:
                    continue
                taken.append((a, b))
                d = found.setdefault(skill, {"skill": skill, "category": self.skill2cat[skill],
                                             "mentions": 0, "spans": []})
                d["mentions"] += 1
                d["spans"].append([a, b])
        return sorted(found.values(), key=lambda d: d["spans"][0][0])

    def extract_names(self, text: str) -> List[str]:
        return [d["skill"] for d in self.extract(text)]

    def by_category(self, text: str) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for d in self.extract(text):
            out.setdefault(d["category"], []).append(d["skill"])
        return out

    def normalise_tag(self, tag: str) -> Optional[str]:
        """Maps a dataset `job_skills` tag to its canonical skill."""
        t = tag.strip().lower()
        t = self.aliases.get(t, t)
        return t if t in self.skill2cat else None

    def to_config(self) -> dict:
        return {"categories": self.categories, "aliases": self.aliases,
                "case_sensitive": {k: v for k, v in CASE_SENSITIVE.items() if v},
                "method": "gazetteer + alias normalisation + regex phrase matching + ambiguity rules"}


def parse_skill_list(value) -> List[str]:
    """Parses the dataset's stringified python list, e.g. "['python', 'sql']"."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.startswith("["):
        return []
    try:
        return [str(x) for x in ast.literal_eval(value)]
    except (ValueError, SyntaxError):
        return []


def top_skills(skill_lists: Iterable[List[str]], n: int = 20) -> List[tuple]:
    c = Counter()
    for lst in skill_lists:
        c.update(set(lst))
    return c.most_common(n)
