# What production would add

This repository is a reference implementation. It is honest about the lifecycle and
dishonest about nothing, but a number of things that a production system needs are
deliberately absent. This page lists them, because a clear account of what is missing is
worth more than a diagram of infrastructure that does not exist.

## Monitoring

**What exists.** `/metrics` returns in-process counters over the last 100 predictions:
total count, mean and p95 latency, and the distribution of predicted labels. It resets
when the process restarts, and it is JSON rather than a Prometheus exposition format.

That is enough to notice a service is answering and roughly how fast. It is not
monitoring.

**What a real setup would watch**

| Signal | Why | How it would be detected |
| --- | --- | --- |
| Input drift | The vocabulary the model was fitted on stops matching what arrives | PSI or KS between a reference sample and a rolling window, per feature. `DataValidator.check_drift` already implements the statistics; nothing schedules it. |
| Out-of-vocabulary rate | A cheap early proxy for drift on a TF-IDF model: rising OOV means the fitted vocabulary is going stale | Fraction of tokens per request absent from `tfidf.vocabulary_` |
| Prediction distribution | A model that suddenly answers "positive" 95% of the time has changed behaviour even if latency is fine | Compare the rolling label distribution against the evaluation-set distribution |
| Confidence distribution | Mass shifting toward the decision boundary indicates inputs the model has no basis for | Histogram of `max(predict_proba)` |
| Latency | Serving cost regression | p50/p95/p99 at the service boundary, under real concurrency, not in-process |
| Error and 503 rate | Readiness flapping, bad inputs, dependency failure | HTTP status counters |
| Model and data freshness | The quiet failure: everything is green and the model is a year old | Age of the production bundle and of the dataset hash behind it |

**Ground truth is the hard part.** None of the above measures accuracy, because accuracy
needs labels and labels arrive later, if at all. A real deployment would need a labelling
path — delayed feedback, sampled human review, or a downstream signal — before it could
claim to monitor model quality rather than model behaviour.

**Why none of it is implemented here.** Emitting metrics nobody scrapes, into a
dashboard nobody watches, with no alert routing and no on-call, is the appearance of
monitoring. The statistics are implemented and tested; wiring them to infrastructure is
the part that needs a real deployment to be meaningful.

## Serving

| Gap | What production would do |
| --- | --- |
| Single process, single worker | Multiple replicas behind a load balancer; `/ready` already provides the right probe |
| No authentication | mTLS or a gateway with API keys; the service currently assumes a trusted network |
| No rate limiting | Per-client quotas, since batch endpoints are an easy way to exhaust CPU |
| No request tracing | Correlation IDs propagated through logs, so one slow prediction can be found |
| Model updates need a restart | A reload endpoint or a sidecar that watches the registry, with a readiness dip during swap |
| No graceful degradation | Currently the choice is "serve" or "503". A cached or simpler fallback model is possible, but only if the caller can tell which answered |

## Data and training

- **Scheduling.** Training is run by hand. Production would run it on a schedule or on a
  data-arrival trigger, with the run itself versioned.
- **Champion/challenger.** The gate compares against a majority-class baseline, not
  against the model currently in production. Once an incumbent exists, the more useful
  question is "is this better than what we are already serving", evaluated on the same
  held-out data.
- **Slice-based evaluation.** One aggregate number hides per-segment failure. The
  reference dataset pools three sources; per-source metrics would very likely differ and
  are not reported.
- **Rollback.** The registry archives the previous production version and keeps its
  bundle, so a rollback is a stage transition. That path is not automated or drilled.
- **Larger-than-memory data.** Everything here fits in a pandas frame. Beyond that, the
  feature-fitting and evaluation stages need a different execution model.

## Security

- Checksums are integrity, not authenticity. They detect a truncated or corrupted
  bundle; they cannot make an untrusted one safe, because joblib artifacts are
  pickle-capable and loading one executes code. `load_trusted_bundle` is named to force
  that thought. Production would sign bundles and verify signatures before loading.
- The container runs as a non-root user with no write access to its own code, and the
  bundle is mounted read-only.
- Running as a non-root user makes file ownership a deployment concern rather than an
  afterthought: the bundle has to be readable by the runtime user. This is easy to miss
  because Docker Desktop for Windows does not enforce Unix ownership across a bind
  mount, so a permission problem can pass locally and fail on a Linux host. A real
  deployment would set ownership when the artifact is placed, not when it is consumed.
- No secrets exist in the repository, and none are needed: the pipeline reads a public
  dataset and writes to local paths. A production version with object storage or a
  tracking server would need real secret management, not environment variables baked
  into an image.

## Cost

Not measured. Training takes about eleven milliseconds on the reference dataset and the
image is a slim Python base with four numeric dependencies. At this scale cost is not
an interesting question, which is precisely why no cost dashboard is pretended at.
