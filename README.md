# MLOps End-to-End Pipeline

Training a model in a notebook is the easy part. The hard part is everything around it:
knowing exactly which data produced which model, refusing to ship a model that does not
meet the bar, and serving it in a way you can reproduce six months later. This repo is a
complete, runnable implementation of that lifecycle for a text-classification task
(sentiment analysis) — the same machinery a production ML team would use, small enough
to read in an afternoon.

> **Scope:** this is a reference pipeline, not a deployed product. When no dataset is
> supplied it deterministically generates synthetic review text, so every step runs
> end-to-end with zero setup. No production accuracy claim is made — point it at a real,
> licensed dataset before reading anything into the metrics.

## The lifecycle

![Pipeline](docs/diagrams/pipeline.svg)

Each stage is a small CLI tool that can also be imported as a library:

```bash
pip install -r requirements.txt

python src/data_pipeline.py  --config configs/train_config.yaml     # validate + version data
python src/train.py          --config configs/train_config.yaml     # train + log to MLflow
python src/evaluate.py       --model_path models/latest             # metrics + quality gate
python src/model_registry.py --register --stage production          # promote the model
python src/serve.py          --model_path models/production         # FastAPI on :8000

# or the whole thing, containerized:
docker-compose up --build
```

## What each component actually does

| Component | Implementation | Why it matters |
|---|---|---|
| Data versioning | SHA-256 content hash + JSON manifest per version | every model traceable to the exact rows that trained it |
| Data validation | nulls, duplicates, class balance, drift (KS-test + PSI) | catch bad data before it silently becomes a bad model |
| Feature store | TF-IDF transformers fitted once, cached, reused at serve time | train/serve skew eliminated by construction |
| Experiment tracking | MLflow (params, metrics, model artifact per run) | compare runs, reproduce the best one |
| Quality gate | configurable min-accuracy threshold | a model that fails the gate is never promoted |
| Model registry | versioned local registry, stages none → staging → production | explicit, auditable promotion |
| Serving | FastAPI: `/predict`, `/predict/batch`, `/health` with latency + prediction logging | observable inference service |
| CI | GitHub Actions runs the pytest suite on every push and PR | the pipeline itself is tested like software |

## Repository layout

```
MLOps-End-to-End-Pipeline/
├── src/
│   ├── data_pipeline.py     # load, clean, validate, hash-version the data
│   ├── feature_store.py     # TF-IDF features, cached transformers
│   ├── train.py             # model factory (LR / RF / SVM), CV, MLflow logging
│   ├── evaluate.py          # metrics, ROC/PR curves, quality gates
│   ├── model_registry.py    # versioned registry with stage management
│   └── serve.py             # FastAPI inference service
├── configs/                 # train + deploy configuration (YAML)
├── tests/                   # data-pipeline, model, and API tests
├── notebooks/               # guided walkthrough of the whole flow
├── docs/diagrams/           # architecture diagrams
├── Dockerfile, docker-compose.yml
└── .github/workflows/       # CI: pytest on push/PR
```

## Verification

```bash
pip install -r requirements.txt
python -m pytest -q
```

No dataset, model, or evaluation report is versioned in the repo — running the data
pipeline without `data/reviews.csv` uses the deterministic synthetic demo data.

## Tech stack

scikit-learn · MLflow · FastAPI · Docker · pytest · PyYAML

## Author

**Mohd Zamin Quadri** — M.Sc. Mathematics in Science and Engineering, Technical University of Munich

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohdzaminquadri-blue)](https://www.linkedin.com/in/mohdzaminquadri/)
[![GitHub](https://img.shields.io/badge/GitHub-mzquadri-black)](https://github.com/mzquadri)
