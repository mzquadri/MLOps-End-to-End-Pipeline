# Testing

```bash
pip install -r requirements-dev.txt

pytest -m "not docker and not network"  # 153 of 164: no network, no daemon
pytest -m network                       # 5 tests, downloads the licensed dataset
pytest tests/test_docker.py -m docker   # 6 tests, needs a running Docker daemon
ruff check .
```

**164 tests: 153 offline, 5 network, 6 container.** Counted with
`pytest --collect-only -q`, which is the only number worth quoting here because
parametrised cases make a count of `def test_` wrong.

This document said "99 tests: 93 offline, 6 container" until 2026-10-23, which was right
once and had been left behind by three whole test files. The README was correct throughout;
this page was the stale one.

## Layers

| File | Tests | What it protects |
| --- | --- | --- |
| `test_datasets.py` | 15 | Checksum verification, corrupt-cache recovery, parsing, provenance |
| `test_data_pipeline.py` | 22 | Cleaning, hashing, validation checks and the failure policy |
| `test_model.py` | 16 | Model construction, feature store, training metrics, gate arithmetic, registry |
| `test_model_bundle.py` | 12 | Atomicity, tampering, rollback, crash recovery, forged gates |
| `test_bundle_provenance.py` | 31 | What a bundle records about where it came from, and refusing one that cannot prove it |
| `test_training_serving_parity.py` | 26 | That the served path and the training path agree, feature by feature |
| `test_integration.py` | 12 | The whole lifecycle: split discipline, leakage, gate refusal, reproducibility |
| `test_api.py` | 19 | Readiness semantics, real bundle loading, prediction, privacy |
| `test_split_contamination.py` | 5 | **`network`.** Duplicate texts spanning two splits, measured against the licensed data |
| `test_docker.py` | 6 | **`docker`.** Image contents, non-root, healthcheck, HTTP prediction from a container |

That sums to 164. The three files this table used to omit are the two largest after
`test_data_pipeline.py` and the whole of the `network` marker.

## The tests that exist because something was wrong

**`test_writes_nothing_to_the_working_tree`**, loading data used to drop
`data/manifest_<hash>.json` into the repository as a side effect. A stale one was still
sitting in the working tree when this sprint started.

**`test_vocabulary_is_fitted_on_training_rows_only`**, adds a token that appears only in
the test split and asserts it never enters the fitted vocabulary. This is the concrete
form of "no leakage"; a comment claiming it cannot be checked.

**`test_fails_when_accuracy_is_high_but_no_better_than_baseline`**, 0.95 accuracy with a
0.01 margin over baseline. Passes an accuracy floor, fails the gate. This is the case a
single threshold cannot catch.

**`test_prediction_log_is_bounded`**, the log used to be an unbounded list that was only
ever read as its last hundred entries, which is a slow memory leak in a long-lived
process.

**`test_fail_fast_mode_refuses_to_start` and `TestUnreadyService`**, the service used to
raise during startup when a bundle was missing, so the process died and the `degraded`
branch of `/health` was unreachable. Both behaviours are now explicit and covered.

**`test_model_artifact_is_byte_identical_across_runs`**, same seed, same data, same
pinned dependencies should serialise to the same bytes. Timestamps and latency live in
the report precisely so the model artifact can be compared this way.

## What the API tests used to prove, and did not

The previous suite built its client as `TestClient(app)` rather than
`with TestClient(app)`. Starlette only runs the application lifespan inside the context
manager, so startup never executed and the model was never loaded. The tests then set
`state.model` by hand and asserted against a stub.

Two consequences: the bundle → registry → serve path had no coverage at all, and the
suite quietly depended on startup *not* running, had it run, it would have raised and
every test would have errored.

Every client is now a context manager, and `ready_client` runs the real pipeline, gets a
bundle promoted to production, and points the service at it. The service under test is
the service that ships.

## Test data policy

The 153 offline tests use the deterministic synthetic fixture and never touch the network.
That is what keeps `pytest -m "not docker and not network"` fast and runnable anywhere.

**Both markers have to be excluded, not just `docker`.** This page used to give
`-m "not docker"`, which selects the five `network` tests and makes the "never touches the
network" claim false for the command printed directly above it.

The download path is still tested, `test_datasets.py` builds an archive in a temp
directory and points a `DatasetSpec` at it via a `file://` URL, so checksum verification,
corrupt-cache handling and offline behaviour are all covered without depending on UCI
being reachable.

The one place real data is exercised is the `reference-run` CI job, which is separate on
purpose: if it fails, the message is "the documented result stopped reproducing", not
"the tests broke".

## Markers

```
docker     builds and runs the container image
network    downloads the licensed dataset
```

Deselect both in a local run: `-m "not docker and not network"`.

This page previously said the `network` marker "is currently unused, since no test needs
the network". That is no longer true. `tests/test_split_contamination.py` sets
`pytestmark = pytest.mark.network` at module level and holds **five** tests, and CI runs
them in the `reference-run` job under the step "Split contamination stays within the
measured bound" — which is the one job that has the licensed dataset, so it is the only
place the split can be measured against it.
