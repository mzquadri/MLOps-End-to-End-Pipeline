# Data

## The reference dataset

| | |
| --- | --- |
| Name | Sentiment Labelled Sentences |
| Source | UCI Machine Learning Repository, dataset 331 |
| Homepage | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences |
| Archive URL | https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip |
| Size | 84,188 bytes |
| SHA-256 | `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343` |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Rows used | 3,000 labelled sentences (1,000 each from Amazon, IMDb, Yelp) |
| Class balance | 1,500 positive / 1,500 negative |
| Duplicate rows | 0.57% |

### Required attribution

The dataset is CC BY 4.0, which obliges attribution. Anything derived from it — a model,
a chart, a blog post — carries that obligation:

> Kotzias, D. (2015). Sentiment Labelled Sentences [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C57604
> Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The original research is Kotzias et al., *"From Group to Individual Labels using Deep
Features"*, KDD 2015.

**The dataset is not MIT licensed.** The MIT license in this repository covers the code
only. See the scope note at the bottom of `LICENSE`.

## Why it is not committed

Committing the archive would mean redistributing it, and redistribution is a licensing
decision that should be made explicitly rather than by `git add`. A pinned checksum plus
a download script gives the same reproducibility with none of that:

- a fresh clone can reproduce the run
- nothing is redistributed
- a changed upstream archive fails loudly instead of silently changing results

`data/` is git-ignored except for `.gitkeep`.

## How acquisition works

```
src/datasets.py
  ensure_archive()  download (or reuse cache) -> verify SHA-256 -> write
  parse_archive()   three labelled files -> one tidy frame
  provenance_record()  the record embedded in the model bundle
```

Three behaviours worth knowing:

- **A cached file whose checksum does not match is deleted, not reused.** A poisoned
  cache would otherwise silently affect every later run.
- **A mismatched download is never written to disk.** There is nothing partial left for
  a later run to trust.
- **Unparseable lines are skipped and counted, not guessed at.** The IMDb file contains
  two malformed lines; the loader logs `Skipped 2 unparseable lines` and yields 1,000
  valid rows from that file. Silent loss is worse than visible loss.

Offline, `allow_download: false` raises with the homepage URL so the archive can be
placed in `data/cache/` by hand.

## Provenance travels with the model

Every bundle's `lineage.json` carries the dataset record, and `model_bundle.py` refuses
to validate a lineage without `kind`, `key` and `license`. A model that cannot say where
its data came from, or under what terms, cannot be published by this pipeline.

```json
"dataset": {
  "kind": "licensed-download",
  "key": "uci-sentiment-labelled-sentences",
  "license": "CC BY 4.0",
  "citation": "Kotzias, D. (2015). ...",
  "sha256": "afc26626..."
}
```

The record contains no local paths — for user-supplied CSVs, only the file name and row
count are recorded, never the directory it came from.

## The synthetic fixture

`datasets.synthetic_fixture()` generates deterministic template text. It exists so tests
and CI can exercise the full lifecycle in about a second with no network.

**It is not evidence.** It is built from ten templates, so a linear model reaches near
perfect accuracy on it — a property of the fixture, not of the model. Its provenance
record says so explicitly, and `configs/ci_config.yaml` repeats the warning.

This distinction is the whole reason the reference run changed. The previous version of
this repository used the fixture as its only data path and reported accuracy 1.0000,
F1 1.0000 and CV 1.0 ± 0.0 on rows that were 99% duplicates.

## Features

| Feature | Source | Leakage risk |
| --- | --- | --- |
| TF-IDF, 1–2 grams, `min_df=2`, max 10,000 | cleaned text, fitted on train only | Would leak if fitted on the full frame; see `docs/ARCHITECTURE.md` decision 2 |
| `review_length` | `len(text)` of that row | None — computed per row from its own text |
| `word_count` | token count of that row | None — same |

The reference run produces 3,463 features: 3,461 TF-IDF terms plus the two length
features.

## Using your own data

```yaml
data:
  source: "data/my_reviews.csv"   # must contain the text and label columns
```

A local CSV takes priority over the reference dataset. Its provenance is recorded as
`local-csv` with `license: "Unknown - supplied by the operator"`, because the pipeline
cannot know what you are allowed to do with it.
