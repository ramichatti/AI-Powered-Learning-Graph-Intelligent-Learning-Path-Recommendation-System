# JobLens — Job Offers Analysis in the Data Field using NLP and Machine Learning

**Advanced Learning project (S3 — *Analyse des offres d'emploi en Data*).**
An end-to-end system that analyses 785,741 real data-job advertisements:

* it **extracts skills**,
* it **classifies** ads into job categories with **TF-IDF + Logistic Regression**,
* it builds **semantic representations** with **DistilBERT** for similarity search,
* it **discovers professional families** with **K-Means**,
* it links jobs, companies and skills in a **knowledge graph** with communities, skill recommendation and a mini Q&A,
* it presents everything in an interactive **Flask** web application.

Every number shown in the notebook and in the app is computed from the real dataset and the trained models. Nothing is hard-coded.

---

## 1. Objectives

| Objective | Implementation |
|---|---|
| Analyse job advertisements | Schema analysis, cleaning, EDA (`notebooks/`, Analytics page) |
| Extract skills and entities | Gazetteer + alias normalisation + regex phrase matching + ambiguity rules; DistilBERT skill embeddings (`jobnlp/skills.py`) |
| Classify ads (`job_title_short`) | TF-IDF (1-2 grams) + multinomial Logistic Regression, saved as a scikit-learn `Pipeline` |
| Semantic representation and similarity | DistilBERT (`distilbert-base-uncased`), mean pooling over valid tokens, cosine similarity |
| Group similar ads | K-Means on the DistilBERT embeddings; K chosen with the elbow method and the silhouette score |
| Skills ↔ jobs ↔ companies relations | NetworkX tripartite graph, Louvain communities, Personalised PageRank recommendation, rule-based graph Q&A |
| Interactive exploration | Flask app with 7 pages and a JSON API |

## 2. Dataset

* **Source:** [Hugging Face — lukebarousse/data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs), a scrape of Google Jobs postings for data roles in 2023: 785,741 rows and 17 columns.
* **Columns used:** `job_title_short` (target, 10 classes), `job_title`, `company_name`, `job_location`, `job_country`, `job_via`, `job_schedule_type`, `job_work_from_home`, `job_posted_date`, `salary_year_avg`, and `job_skills` (a stringified list of skills extracted by the dataset author).
* **Important:** the dataset does **not** contain the full description text. The "document" for every model is therefore built as **job title + skill list**.

**Cleaning** (all steps and row counts are in `models/preprocessing_config.json`):

| Step | Rows |
|---|---|
| Raw | 785,741 |
| Exact duplicates removed | −101 |
| Postings without any skill removed (14.9 %) | −116,986 |
| Cross-platform re-posts removed (same title, company, location and skills). This is **leakage control**: copies of one ad must not end up in both train and test | −103,995 |
| **Final clean dataset** | **564,659** |

## 3. Project structure

```
ADVANCED LEARNING PROJECT/
├── notebooks/job_offers_nlp_analysis.ipynb   ← full executed pipeline (trains + saves everything)
├── jobnlp/                                   ← core library shared by the notebook AND the app
│   ├── preprocessing.py   cleaning, text views, leakage control
│   ├── skills.py          skill extractor (dictionary, aliases, rules)
│   ├── tokenizer.py       WordPiece tokenizer (pure Python)
│   ├── distilbert.py      DistilBERT forward pass in NumPy + mean pooling
│   ├── embeddings.py      corpus sampling, caching, cosine search
│   ├── graph.py           knowledge graph, communities, PPR recommendation, Q&A
│   └── viz.py             notebook figure style (validated colour palette)
├── data/
│   ├── raw/               data_jobs.csv (or keep it at the project root)
│   └── processed/         jobs_clean.csv.gz, embedding_corpus.csv
├── models/                ← every saved artifact (see §6)
├── app/
│   ├── app.py             Flask routes + JSON API
│   ├── services/          classification, skill_extraction, semantic_search, clustering, knowledge_graph, artifacts
│   ├── templates/         base + 7 pages
│   └── static/            css/style.css, js/app.js, js/charts.js, js/pages/*.js, vendor/ (D3, Cytoscape)
├── results/
│   ├── figures/           14 PNG figures exported by the notebook
│   └── metrics/           classification_metrics.json, clustering_metrics.json, results_summary.csv
├── tests/test_project.py  end-to-end tests (artifacts + every API endpoint)
├── requirements.txt
├── run.py                 launches the web app
└── README.md
```

## 4. Methodology

### 4.1 Skill extraction (`jobnlp/skills.py`)
1. **Gazetteer:** 258 canonical skills in 11 categories. They are the dataset's own `job_type_skills` vocabulary plus 21 data/AI concepts (machine learning, ETL, LLM, ...).
2. **66 aliases** normalised to one name (`PowerBI`, `power-bi` → `power bi`; `sklearn` → `scikit-learn`; `k8s` → `kubernetes`).
3. **One compiled regex** that tries the longest alternatives first (`sql server` before `sql`). Custom token boundaries keep `c++`, `c#`, `node.js` intact.
4. **Ambiguity rules:** `R`, `C`, `Go`, `Word`, `Swift`, `Spring`, ... only match their case-sensitive technical spelling. `R&D` and `C-level` are rejected.
5. **Semantic layer:** contextual DistilBERT skill embeddings (each skill = the centroid of the ads that require it), used for "related skills".

**Evaluation:** on real job titles, an extracted skill counts as correct if the posting's own tags contain it. Precision is **92.2 %** on 58,234 mentions. This is a lower bound, because the tags come from the unseen description.
**Limitations:** no deep context understanding ("Excel at teamwork" matches Excel), unseen technologies are missed, and each skill has a single category.

### 4.2 Classification — TF-IDF + Logistic Regression
* Input: cleaned lower-case `title | skills` text. TF-IDF on word 1-2 grams with `min_df=3`, `sublinear_tf`, and at most 150k features (52,014 kept).
* **Stratified 80/20 split** (`random_state=42`): 451,727 training ads and 112,932 test ads. `C` was chosen from {0.5, 2, 8} on a 10 % validation split of the training set (best C = 8). The vectoriser is inside the `Pipeline`, so it is fitted on the training data only.

| Model (test set) | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Majority-class baseline | 0.255 | 0.025 | 0.100 | 0.041 |
| TF-IDF + LogReg — skills only | 0.515 | 0.448 | 0.255 | 0.236 |
| **TF-IDF + LogReg — title + skills (main)** | **0.949** | **0.924** | **0.913** | **0.918** |

*Interpretation:* `job_title_short` was itself derived from the title, so the main model mainly learns that standardisation. The **skills-only** model shows that a skill profile identifies the profession *family* but not the *seniority*. The main remaining errors are Data Analyst ↔ Data Scientist, seniority (Senior X ↔ X), hybrid titles, and the small Cloud Engineer class. The confusion matrix, per-class F1, top n-grams and error examples are in the notebook and on the Classification page.

### 4.3 DistilBERT — semantic embeddings and similarity
* Checkpoint **`distilbert-base-uncased`** (66 M parameters; official `model.safetensors`).
* `jobnlp/distilbert.py` re-implements the forward pass in **NumPy**, and `jobnlp/tokenizer.py` re-implements the WordPiece tokenizer, so **PyTorch is not required**. Correctness is checked with the masked-LM head, which reproduces the model-card example exactly (*"Hello I'm a [MASK] model."* → role, fashion, business, model, modeling). If `torch` + `transformers` are installed, the notebook also compares the hidden states numerically with the official implementation.
* **Embedding:** `"<title>. Skills: <list>."`, at most 128 tokens (only 1 text was truncated), **mean pooling over attention-mask tokens**, L2-normalised. Batches of 64 are sorted by length with dynamic padding.
* **Corpus:** a class-balanced sample of **15,000 ads** (1,500 per category). The embeddings are computed once (≈ 21 min on 2 CPU cores) and cached in `models/job_embeddings.npy` (float16) with an id fingerprint.
* **Anisotropy:** raw mean-pooled vectors have an average pairwise cosine of 0.87. After subtracting the corpus mean it is 0.007, and centered vectors are used for similarity and clustering.
* **Limitation (stated explicitly):** DistilBERT is **not** a sentence-similarity model. It was pre-trained for masked-LM, not with a contrastive objective. Its cosine scores are uncalibrated and should be read as a *ranking*.
* **Keyword vs semantic:** for *"deep neural networks for image recognition"*, TF-IDF returns a *Security Detection* engineer and a *Networks Engineer* (literal word matches), while DistilBERT returns ML / computer-vision roles with PyTorch and TensorFlow. For *"real-time event streaming"*, TF-IDF wins because the exact words appear in titles.
* **Label agreement@10** (share of the 10 nearest neighbours with the same category, over 1,000 random ads): semantic **0.480**, TF-IDF 0.473, random 0.100.

### 4.4 K-Means — profession discovery
* Input: the centered, L2-normalised DistilBERT embeddings (on unit vectors, Euclidean K-Means is equivalent to clustering by cosine). `n_init=10`, `random_state=42`.
* K = 2…15 evaluated with the **elbow method** (inertia; Kneedle elbow at K = 6) and the **silhouette score** (cosine; it decreases with K, as is typical for language-model embeddings).
* **Rule fixed in advance:** the best silhouette for K in [4, 12] gives **K = 4** (silhouette 0.244). K = 2 or 3 would be too coarse to describe professional families.

| Cluster | Ads | Automatic label (dominant categories · most distinctive skills) |
|---|---|---|
| C0 | 5,414 | Senior Data Engineer / Data Engineer · kafka, docker, kubernetes |
| C1 | 3,995 | Business Analyst / Data Analyst · excel, sql |
| C2 | 2,838 | Senior Data Analyst / Data Analyst · tableau, power bi, r |
| C3 | 2,753 | Machine Learning Engineer / Software Engineer · python, aws, sql |

The clusters do **not** copy the official categories (ARI 0.061, NMI 0.111 against `job_title_short`). They regroup ads by *tool stack*, and seniority does not separate them. The PCA projection, cluster × category heatmap and skills-by-cluster heatmap are in the notebook and the app.

### 4.5 Knowledge graph, communities and Q&A
* The whole clean dataset is compressed into a posting index (sparse ad × skill matrix plus company, job and country codes, in `models/kg/`). Graphs and answers are computed on demand from these real counts.
* **Nodes:** jobs, companies, skills. **Edges:** company *offers* job, job *requires* skill (share of the job's ads), company *uses* skill.
* **Communities:** Louvain (default graph: 63 nodes, 286 edges, 4 communities, modularity 0.19).
* **Skill recommendation:** Personalised PageRank on the skill co-occurrence graph (only positively associated pairs), ranked by PPR × log(lift) so that hub skills do not dominate. Examples: *pytorch* → tensorflow, scikit-learn, keras; *Excel + SQL analyst* → PowerPoint, Tableau, Power BI, VBA.
* **Q&A ("Graph-RAG lite"):** intent detection plus entity linking (skills, job categories, countries), then a structured query on the graph, then a templated answer with an evidence table and a subgraph. No text is generated by a language model, so every number is an exact count.
  Supported intents: companies, skills, jobs, countries, salary, recommendation. Examples: *"Which companies offer jobs requiring Python and AWS?"*, *"What skills does a Data Engineer need?"*, *"Which skills should a data analyst who knows Excel and SQL learn next?"*.
* A trained GNN was **not** implemented (PyTorch / PyTorch-Geometric unavailable). PPR is the non-parametric graph-propagation baseline.

## 5. Installation

Requires **Python 3.11–3.13**. From the project folder (Windows PowerShell shown; on macOS/Linux use `source .venv/bin/activate`):

```powershell
cd "$HOME\Downloads\ADVANCED LEARNING PROJECT"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Data and model files** (already in your folder):
* `data_jobs.csv`: at the project root **or** in `data/raw/`.
* DistilBERT `model.safetensors`, `config.json`, `vocab.txt`: at the project root **or** in `models/distilbert-base-uncased/`.

## 6. Running

### Launch the web application (uses the saved models; no training)
```powershell
python run.py
```
Open **http://127.0.0.1:5000**. Options: `python run.py --port 8000 --host 0.0.0.0`.

### Re-run the full pipeline (re-trains and re-saves every artifact)
```powershell
jupyter notebook notebooks/job_offers_nlp_analysis.ipynb
```
Then use *Kernel → Restart & Run All*, or run it headless:
```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/job_offers_nlp_analysis.ipynb --ExecutePreprocessor.timeout=-1
```
Runtime is ≈ 5 min with the cached embeddings. Set `RECOMPUTE_EMBEDDINGS = True` in section 2 to recompute DistilBERT (≈ 20 min on CPU).

### Tests
```powershell
python -m pytest -q tests
```

### Saved artifacts (`models/`)
| File | Content |
|---|---|
| `classification_pipeline.joblib` | full TF-IDF + LogReg `Pipeline` (used by the app) |
| `tfidf_vectorizer.joblib`, `logistic_regression.joblib` | the two steps saved separately |
| `skills_only_pipeline.joblib` | skills-only classifier |
| `job_embeddings.npy` (+ `.meta.json`), `embedding_mean.npy` | 15,000 × 768 DistilBERT embeddings + centering vector |
| `skill_embeddings.npy`, `skill_embedding_names.json` | contextual skill embeddings |
| `kmeans_model.joblib`, `pca_2d.joblib`, `clustering_config.json` | K-Means, PCA projection, K selection curves, cluster profiles |
| `model_config.json` | DistilBERT checkpoint, tokenizer and pooling configuration, library versions |
| `preprocessing_config.json`, `skill_extraction_config.json` | cleaning report; skill dictionary and aliases |
| `analytics.json` | aggregated statistics for the dashboard |
| `kg/` | posting index, vocabulary, default graph, Q&A examples |

The notebook's last section reloads these files **in a separate Python process** to prove they are reusable.

## 7. The web application

| Page | What it does |
|---|---|
| **Dashboard** | KPIs (ads, categories, skills, clusters, companies, countries), most demanded skills, category distribution, cluster overview, model performance, monthly trend, top countries |
| **Job Classification** | paste a description, then see the predicted category, the confidence, the class probabilities, the most influential n-grams and the skills-only prediction; also the confusion matrix and per-class F1. "Load a real ad" fetches a real record together with its true label |
| **Skill Extraction** | detected skills highlighted in the text and grouped by category; frequency in the dataset; the job category that requests each skill most; nearest skills (DistilBERT); recommended complementary skills (graph) |
| **Semantic Search** | the query is embedded with DistilBERT, compared by cosine with 15,000 ads, filtered by category; results show title, company, location, skills shared with the query and score, plus the query's position on the cluster map |
| **Job Clustering** | zoomable PCA map (colour **and** marker shape per cluster, centroids, click to isolate a cluster), cluster cards, per-cluster skills, categories and searchable list of ads, elbow and silhouette charts, cluster × category heatmap |
| **Analytics & Insights** | skills overall, by category (filter), by job, by cluster; skill-category mix; salary ranges; category distribution; cluster sizes; model metrics; elbow and silhouette; monthly postings; schedule types |
| **Knowledge Graph & Q&A** | interactive force-directed graph (Cytoscape.js + fcose, with job categories pinned on a ring so nodes never overlap). Filters: job categories, number of companies and skills, minimum share, node types, Louvain community, K-Means cluster. Node search, hover tooltips with neighbour highlighting, click to expand a node's neighbourhood from the full dataset, colour by type or community, legend. Also includes the Q&A panel |

Invalid inputs (empty, too short, numbers only, more than 20,000 characters, wrong parameters) return clear messages. If an artifact is missing, the affected page explains which file is missing and how to regenerate it. All JS/CSS libraries are bundled in `app/static/vendor/`, so the app works offline.

## 8. Technologies
Python · pandas · NumPy · SciPy · scikit-learn · NetworkX · DistilBERT (NumPy implementation of the Hugging Face checkpoint) · Matplotlib · Seaborn · Flask · D3.js 7 (charts) · Cytoscape.js 3 + fcose (graph) · HTML5 / CSS3 (custom responsive design system with light and dark themes).

*Note on the front-end stack:* charts use **D3.js** rather than Chart.js. Chart.js was not available offline in the build environment, and D3 gives finer control over label placement, which avoids overlaps. A custom CSS design system replaces Bootstrap for the same reason.

## 9. Limitations and possible improvements
* **No description text** in the dataset, so all models work on title + skills; performance on long free-text descriptions is not measured.
* The main classifier depends mostly on title words (the target was derived from the title).
* DistilBERT is used without fine-tuning. A sentence-similarity model (Sentence-BERT) or contrastive fine-tuning on title/skill pairs would improve semantic search.
* K-Means silhouette values are modest, because embeddings form a continuum; the clusters should be read as regions, not hard boundaries.
* Recruiters and job boards (Dice, Harnham, Emprego) appear as "companies" in scraped data.
* UMAP and a trained GNN were not used (packages unavailable offline); PCA and Personalised PageRank were used instead.
