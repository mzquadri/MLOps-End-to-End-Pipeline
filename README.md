# MLOps End-to-End Pipeline

A compact, runnable reference pipeline for text classification. It demonstrates
data validation, feature fitting, model training, evaluation gates, immutable
registry versions, and FastAPI serving without claiming production deployment or
real-world accuracy.

When `data/reviews.csv` is absent, the pipeline uses deterministic synthetic
reviews. No dataset, model bundle, MLflow state, or evaluation result is tracked.

## Lifecycle

![Atomic bundle lifecycle](docs/diagrams/pipeline.svg)

Run from the repository root with Python 3.11:

```bash
pip install -r requirements.txt

python -m src.train --config configs/train_config.yaml
python -m src.evaluate --bundle_path models/candidates/sentiment-classifier
python -m src.model_registry --register --model_name sentiment-classifier \
  --bundle_path models/candidates/sentiment-classifier --stage staging
python -m src.model_registry --model_name sentiment-classifier --promote v1
python -m src.serve \
  --bundle_path models/registry/sentiment-classifier/v1 \
  --host 127.0.0.1 --port 8000
```

Evaluation exits nonzero when any configured accuracy, weighted-F1, or latency
gate fails. Direct registration into production is rejected. A version must be a
checksum-valid, passing staging bundle before it can be promoted. Evaluation
reconstructs the exact stratified split and feature schema recorded by training,
not split settings supplied later at evaluation time.

`docker-compose up --build` starts the API and MLflow services only. It does not
train, evaluate, or promote a model; point `MODEL_BUNDLE_PATH` at an existing
passing registry bundle first.

## Atomic bundles

Training fits TF-IDF and numeric scaling on the training partition only. The
held-out partition and serving requests use transform-only methods. Training
atomically writes one candidate directory containing:

```text
model.joblib
feature_transformers.joblib
training_metrics.json
lineage.json
evaluation_report.json
manifest.json
```

The deterministic manifest declares format version `1.0`, the expected feature
dimension, every required filename, and SHA-256 checksums for every artifact.
Evaluation atomically replaces the pending report and manifest. Registry versions
copy the entire validated directory and are never partially assembled. Serving
loads the model and preprocessing state from that same bundle, checks dimensions,
confirms the registry marks that exact version as production, and has no
global-transformer or `latest` fallback.

See [Model bundle integrity and migration](docs/MODEL_BUNDLES.md) for the trust
boundary, manifest details, failure behavior, and migration from legacy split
artifacts.

## Components

| Component | Behavior |
|---|---|
| Data pipeline | Clean, validate, and SHA-256-version input rows |
| Feature store | Fit on training data; transform held-out and serving data without refitting |
| Training | Cross-validation, final fit, optional MLflow logging, atomic candidate bundle |
| Evaluation | Same-bundle transform, metrics and latency, embedded pass/fail report |
| Registry | Complete-bundle validation and explicit staging-to-production promotion |
| Serving | One checksum-valid production bundle, health/readiness, batch prediction |
| CI | Deterministic CPU-only pytest suite on Python 3.11 |

Prediction logs retain length, output, confidence, and latency, but never raw
request text. Lineage stores the data content hash, row counts, split parameters,
feature schema, experiment name, and model type; it excludes source paths,
credentials, environment variables, and raw records.

## Verification

```bash
python -m pytest -q -p no:cacheprovider
python -m compileall -q src tests
```

## Repository layout

```text
src/model_bundle.py       bundle contract, atomic writes, hashes, validation
src/feature_store.py      fit and transform-only feature paths
src/train.py              model training and candidate creation
src/evaluate.py           held-out evaluation and promotion gate
src/model_registry.py     immutable local versions and stage transitions
src/serve.py              FastAPI inference from one validated bundle
tests/                    deterministic data, bundle, registry, and API tests
configs/                  training and deployment examples
docs/                     lifecycle and integrity documentation
```

## Tech stack

scikit-learn, pandas, SciPy, MLflow, FastAPI, PyYAML, and pytest.

## Author

**Mohd Zamin Quadri**

[GitHub](https://github.com/mzquadri) · [LinkedIn](https://www.linkedin.com/in/mohd-zamin/)
