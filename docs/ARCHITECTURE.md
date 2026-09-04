# Architecture

![Lifecycle](diagrams/pipeline.svg)

## Modules and what each one owns

| Module | Owns | Deliberately does not |
| --- | --- | --- |
| `src/datasets.py` | Fetching, checksum verification, parsing, provenance records | Decide whether data is *good* |
| `src/data_pipeline.py` | Cleaning, validation and its failure policy, content hashing | Fetch anything, or write files as a side effect |
| `src/feature_store.py` | Fitting transformers on train, transform-only everywhere else | Know about models, splits or gates |
| `src/train.py` | Splitting, cross-validation, fitting, validation metrics | Touch the test split |
| `src/evaluate.py` | Test metrics, baseline comparison, latency, the gate | Choose thresholds, or re-roll a split |
| `src/model_bundle.py` | Atomic writes, checksums, lineage and gate-consistency validation | Trust anything it did not verify |
| `src/model_registry.py` | Immutable versions, stage transitions | Evaluate quality |
| `src/serve.py` | Loading one production bundle, HTTP, readiness | Load a model that has not passed |
| `src/pipeline.py` | Sequencing the above into the reference run | Contain logic that belongs in a stage |

Each module is a CLI and an importable library. That is not decoration: the tests import
the same functions the CLI calls, so what is tested is what runs.

## Data flow

```
dataset (checksum-verified)  ->  validate  ->  content hash
                                     |
                    train / validation / test  (stratified, one seed)
                                     |
                fit TF-IDF + scaler on TRAIN only
                                     |
                     train  ->  CV + validation metrics
                                     |
                     atomic candidate bundle (manifest + lineage)
                                     |
                     evaluate on TEST (once)  ->  promotion gate
                                     |
                     pass -> registry: staging -> production
                                     |
                     FastAPI loads one production bundle
```

## Design decisions

Each record answers the same seven questions. They are the decisions that would be
asked about in an interview, and the ones most likely to be got wrong.

### 1. The model bundle is the unit of promotion

1. **Problem** A model is useless without the exact preprocessing it was fitted with.
   Storing `model.joblib` and `feature_transformers.pkl` separately lets them drift, and
   the failure is silent: predictions are wrong, not absent.
2. **Why it exists** So that model, transformers, metrics, lineage and evaluation move
   together or not at all.
3. **Alternatives** A pickled sklearn `Pipeline`; MLflow's model registry; two files and
   a naming convention.
4. **Why this one** A `Pipeline` couples the contract to one library version and carries
   no lineage or gate state. MLflow would work but makes a tracking server load-bearing
   for correctness. A directory with a manifest is inspectable with `cat` and has no
   runtime dependency.
5. **What could fail** A partially written directory. This is why writes go to a temp
   directory and are moved into place atomically, with backup-based crash recovery.
6. **How it is tested** `tests/test_model_bundle.py` tampers with checksums, interrupts
   replacements, and forges gate reports. `tests/test_integration.py` verifies published
   checksums against the files on disk.
7. **In production** The bundle would live in object storage with a content-addressed
   key, and checksums would be accompanied by a signature so integrity becomes
   authenticity. See `docs/MODEL_BUNDLES.md` for that boundary.

### 2. Only one function is allowed to fit

1. **Problem** Preprocessing leakage. If TF-IDF sees test text while fitting, the
   vocabulary encodes the test set and every downstream number is optimistic.
2. **Why it exists** `FeatureStore.fit_transform_train` is the sole fitting entry point;
   `transform` cannot fit.
3. **Alternatives** A comment saying "don't fit on test"; an sklearn `Pipeline` inside
   cross-validation; trusting review.
4. **Why this one** A narrow API surface makes the property checkable. A test can assert
   that a token appearing only outside training never enters the vocabulary, which a
   comment cannot.
5. **What could fail** Someone adds a second fitting path. The bundle's
   `expected_feature_dimension` and the evaluation dimension check would catch a shape
   change, but not a same-shape refit, which is why the test above exists.
6. **How it is tested** `test_vocabulary_is_fitted_on_training_rows_only`, plus a bundle
   test that monkeypatches both fitting methods to raise during evaluation.
7. **In production** Unchanged in principle. At scale the fitted state would be
   versioned in a feature store keyed by training-set hash, which this repository
   approximates with the bundle's lineage.

### 3. Validation can refuse

1. **Problem** The original pipeline detected 99% duplicate rows, logged a warning, and
   went on to report 100% accuracy. Detection without consequence is theatre.
2. **Why it exists** `validation.data.on_failure: error` stops the run.
3. **Alternatives** Always fail; always warn; a separate manual data-review step.
4. **Why this one** Exploratory work legitimately needs to look at bad data, so the
   policy is configurable, but the *reference* configuration is strict, and the loose
   setting has to be chosen deliberately and is visible in version control.
5. **What could fail** A threshold set so loose it never fires. The CI config relaxes
   the duplicate ceiling for the template fixture and says so in a comment; the
   reference config does not.
6. **How it is tested** `TestValidationPolicy` covers both policies and the no-op path.
7. **In production** The same checks would run as a scheduled job against incoming data
   with results emitted as metrics, and the failure policy would differ per environment.

### 4. The gate requires a margin over a baseline

1. **Problem** An accuracy floor is meaningless on its own. On a 95/5 imbalanced task,
   predicting the majority class scores 0.95 and clears most thresholds.
2. **Why it exists** `accuracy_over_baseline` compares against a majority-class
   predictor measured on the same rows.
3. **Alternatives** Accuracy only; balanced accuracy; MCC; a champion/challenger
   comparison against the current production model.
4. **Why this one** It is the smallest change that makes the gate mean the same thing
   across datasets, and it is easy to explain. Champion/challenger is the better answer
   once a production model exists, and is noted as future work rather than pretended at.
5. **What could fail** On a heavily imbalanced dataset a 0.20 absolute margin becomes
   very hard to reach; the threshold is a per-project decision, not a universal constant.
6. **How it is tested** `test_fails_when_accuracy_is_high_but_no_better_than_baseline`
   constructs exactly the case a bare floor would let through.
7. **In production** The gate would also compare against the incumbent model on the same
   evaluation set, and would include slice-level checks rather than one aggregate.

### 5. The service starts unready instead of crashing

1. **Problem** A missing bundle raised during startup, so the process died. The operator
   got a crash loop and no endpoint to ask what was wrong, and the `degraded` branch of
   `/health` was unreachable.
2. **Why it exists** Startup records the failure and serves; `/ready` returns 503.
3. **Alternatives** Fail fast; retry in the background; serve a fallback model.
4. **Why this one** Readiness probes are the standard mechanism for keeping traffic away
   from an unusable instance, and a live process can explain itself. A fallback model is
   worse than an error, because it answers wrongly and silently. Fail-fast remains
   available through `REQUIRE_MODEL_AT_STARTUP=1` for deployments that prefer it.
5. **What could fail** An operator watches `/health` instead of `/ready` and routes
   traffic to an instance with no model. The endpoints are named and documented to make
   that mistake harder.
6. **How it is tested** `TestUnreadyService` covers degraded liveness, 503 readiness,
   503 predictions and the fail-fast opt-in.
7. **In production** The same two endpoints map directly onto Kubernetes liveness and
   readiness probes. No orchestration is included here because none is needed to
   demonstrate the semantics.

### 6. The GitHub-facing reference run uses real, licensed data

1. **Problem** A synthetic fixture built from ten templates is separable by inspection.
   Every metric it produces is a property of the fixture.
2. **Why it exists** The reference run downloads a small, openly licensed dataset,
   verifies its checksum, and records its provenance in the bundle.
3. **Alternatives** Commit a small dataset; keep synthetic data; use a large benchmark.
4. **Why this one** Committing data means redistributing it, which is a licensing
   decision the repository should not make silently. A deterministic download with a
   pinned checksum gives reproducibility without redistribution. See `docs/DATA.md`.
5. **What could fail** The upstream host goes away, or re-cuts the archive. The checksum
   turns the second case into a loud failure rather than a silent change; the first
   fails CI, which is the correct signal.
6. **How it is tested** `tests/test_datasets.py` exercises verification, corrupt-cache
   recovery and offline behaviour without touching the network.
7. **In production** The dataset would be mirrored into internal storage with retention
   and access control, and the pinned checksum would be the link between the mirror and
   the original.
