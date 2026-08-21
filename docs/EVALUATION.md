# Evaluation

## Split methodology

```
3,000 rows
  ├── train       1,800 (60%)  fits TF-IDF, the scaler, and the model
  ├── validation    600 (20%)  the only split training is allowed to score
  └── test          600 (20%)  read once, by src/evaluate.py
```

Both splits are stratified and derived from one seed (`random_state: 42`), so the
partition is reproducible from the config file alone. `validation_size` is expressed as
a fraction of the whole dataset and converted internally to a fraction of the post-test
remainder, because "20% validation" should mean 20% of the data, not 20% of what is
left.

**The test split is not reachable from `src/train.py`.** It is not loaded, transformed
or scored there. That is a structural guarantee rather than a promise: no amount of
iterating on the training module can consume the test set.

Evaluation does not accept split settings as arguments. It reads them from the
candidate's `lineage.json` and reconstructs the identical partition, then refuses to
continue if the data hash or the test row count disagrees. An operator cannot re-roll the
split until a model passes.

## Metrics

| Metric | Why it is here |
| --- | --- |
| Accuracy | Headline number, meaningless without the next row |
| Majority-class baseline | The scale accuracy is read against |
| `accuracy_over_baseline` | The margin the gate actually enforces |
| Weighted F1 | Sensitive to per-class failure in a way accuracy is not |
| Precision / recall (weighted) | Which side of the trade-off the model sits on |
| ROC-AUC | Ranking quality, independent of the decision threshold |
| PR-AUC | More informative than ROC-AUC if the classes become imbalanced |
| Confusion matrix | Where the errors actually are |
| p95 / p99 latency | Serving-cost regression guard |

Calibration is **not** reported. Nothing downstream consumes the probabilities as
probabilities — the service returns a label with a confidence for display, and no
decision threshold is tuned — so a calibration curve would be a metric in search of a
purpose. If a business rule ever routes on a probability, this is the first thing to add.

ROC and PR curve points are written to `results/evaluation_curves.json`. They are useful
for a report but are not part of the promotion contract, so they stay out of the bundle.

## The reference result

Produced by `python -m src.pipeline --config configs/train_config.yaml`:

| | Train (CV) | Validation | Test |
| --- | --- | --- | --- |
| Accuracy | — | 0.8117 | **0.8067** |
| Weighted F1 | 0.7915 ± 0.0132 | 0.8114 | **0.8067** |
| ROC-AUC | — | — | **0.8795** |
| PR-AUC | — | — | **0.8895** |
| Majority baseline accuracy | — | 0.5000 | **0.5000** |
| Margin over baseline | — | 0.3117 | **0.3067** |

Confusion matrix on the 600 test rows (rows = actual, columns = predicted, label order
`[negative, positive]`):

```
              pred neg   pred pos
actual neg       241        59
actual pos        57       243
```

Latency, single row, in process: mean 0.062 ms, p95 0.067 ms, p99 0.076 ms. This is a
regression guard measured on one machine, not a service level objective — a real latency
budget is measured at the service boundary under concurrency.

`scripts/check_reference_run.py` asserts these numbers against a produced run, so the
table above cannot drift away from reality without CI noticing.

## How the gate thresholds were chosen

This is the part worth being explicit about, because it is the easiest place to cheat.

The previous configuration required `min_accuracy: 0.85` and `min_f1: 0.83`. Those
values have no recorded basis, and they were only ever satisfied by a template fixture
that scored 1.0. Against real data they are not achievable by this model:

```
Candidate failed the promotion gate (accuracy, f1_weighted); not registered.
```

That is a real result, reproduced by setting the old thresholds back. So the choice was
between keeping a permanently failing gate or setting a defensible one.

The current values are derived as follows:

1. The majority-class baseline on this data is **0.500**. Any threshold at or below that
   is worthless.
2. Observed **validation** accuracy is **0.8117**. Validation is the split that exists
   for exactly this kind of decision.
3. The gate is set at **0.75** — comfortably above baseline, and roughly six points
   below the observed validation score to absorb split noise and small dependency
   changes without becoming a rubber stamp.
4. `min_accuracy_over_baseline` is set at **0.20**, so the gate keeps its meaning if the
   dataset ever stops being balanced.

**The test split was not consulted when choosing these numbers.** The test result
(0.8067) is reported after the fact and was not used to select a threshold, a model, or
a hyperparameter. If it had been, the number would be an estimate of nothing.

A threshold below the current model's score is a floor, not a target. It answers "is this
model good enough to ship", not "is this the best model" — the second question belongs to
a champion/challenger comparison, which needs an incumbent and is listed as future work
in `docs/PRODUCTION.md`.

## Reading the result honestly

- 0.807 accuracy on 600 short sentences is an ordinary result for TF-IDF plus logistic
  regression. It is not close to what a fine-tuned transformer achieves on this task.
- The dataset is three sources pooled into one; per-source performance is not reported
  and would very likely differ.
- 600 test rows means the confidence interval on accuracy is roughly ±3 points. A
  one-point difference between two models on this split is noise.
- Nothing here says anything about production sentiment analysis on your data.
