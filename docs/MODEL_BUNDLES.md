# Model bundle integrity and migration

## Why the bundle exists

The legacy workflow saved `models/latest/model.joblib` and
`models/feature_transformers.pkl` independently. Evaluation loaded the global
transformers and then called the training feature path on the held-out split,
which refitted TF-IDF and `StandardScaler`. Serving also searched for that global
file and silently fell back from `models/production` to `models/latest`. A model
could therefore be evaluated or served with preprocessing state it was not
trained against.

The bundle makes model, preprocessing, metrics, lineage, evaluation, and their
integrity metadata one promotion unit.

## Format 1.0

`manifest.json` is deterministic, sorted JSON with:

- `bundle_format_version`
- `expected_feature_dimension`
- `required_files`
- full SHA-256 checksums for the five non-manifest artifacts

`lineage.json` contains safe aggregate provenance only: data content hash, row
counts, split seed/fraction, feature column names, experiment, and model type. It
must not contain raw records, credentials, environment variables, or absolute
paths. `evaluation_report.json` records the evaluated data hash, metrics, latency,
individual checks, overall gate result, and `passed` or `failed` status.
Evaluation reconstructs the feature schema and exact stratified held-out split
from lineage. Promotion validation requires all three gate records and verifies
their values, thresholds, pass/fail decisions, and overall result for internal
consistency.

Training and evaluation build a complete sibling directory and replace the
candidate only after validation. Registry registration copies and revalidates the
complete bundle before publishing a new immutable version. Registry metadata uses
a path relative to the registry root.

## Promotion rules

1. A pending or failed candidate cannot enter staging.
2. Direct production registration is rejected.
3. Production promotion requires the current stage to be staging.
4. Staging and production both recheck required files, checksums, evaluation/data
   lineage agreement, passing gates, and model/transformer dimensions.
5. Serving verifies that the registry index marks the exact bundle version as
   production, then fails startup/readiness when any artifact is missing, corrupt,
   pending, failed, or dimensionally inconsistent.

## Trust boundary

Scikit-learn/joblib files are pickle-capable and may execute code while loading.
Only load bundles created locally or received through a trusted, authenticated
artifact channel. Manifest checksums detect accidental corruption and incomplete
copies; because the manifest is not cryptographically signed, checksums do not
make an untrusted bundle safe. Validation checks every byte checksum before any
joblib artifact is loaded.

## Legacy migration

Legacy split artifacts are intentionally not accepted for staging or serving.
Retrain once with `python -m src.train`, evaluate the generated candidate, and
register that complete bundle. Do not copy a legacy global transformer next to a
model: its fit partition and feature dimension cannot be proven by location.

The optional MLflow logging path remains independent. Its compatibility
registration helper is staging-only and cannot directly transition a run to
production. The local promotion gate and FastAPI service use the file-system
bundle as their source of truth.
