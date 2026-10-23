# Review

A read of this repository against its own claims, done on 2026-10-23 on Windows 11 with
CPython 3.11 in a fresh virtual environment. Every number below came from a run recorded
here.

## What it is

A sentiment classifier is the excuse; the pipeline is the subject. Acquire a licensed
dataset by checksum, validate it, split it reproducibly, fit features without leaking the
test split, train, write an immutable bundle, evaluate against a promotion gate, register,
and serve the bundle over HTTP, with CI running all of it.

The two design decisions worth naming are both about refusing rather than producing. The
bundle, not the model file, is the unit of promotion, so a model can never be served
beside transformers it was not fitted with. And the gate requires a **margin over a
baseline** rather than an accuracy floor, which is the difference between a check that
means something and a number that looks reassuring.

## What was run, and what passed

| Step | Command | Result |
|---|---|---|
| Install | `pip install -r requirements-dev.txt` | Clean |
| Lint | `ruff check .` | **All checks passed** |
| Offline tests | `pytest -m "not docker and not network"` | **153 passed, 11 deselected**, 19.7 s |

The 11 deselected are the `docker` and `network` markers. CI runs them in their own jobs:
`container` builds the image, mounts a bundle and makes a real HTTP prediction, and
`reference-run` downloads the licensed dataset and runs the whole pipeline. Splitting them
by marker is why a red build here says *which* thing broke.

### The CI design is the strongest thing here

Three jobs with deliberately different dependency sets, a dataset cache keyed by the
**pinned checksum** so a hit is provably the right bytes, and
`scripts/check_reference_run.py` turning the documented result into an assertion CI
evaluates. That last one is the pattern every repository in this portfolio should have:
the README's numbers are not prose, they are a test.

## Two things I got wrong first, recorded so the next reader does not repeat them

**`scripts/check_reference_run.py` appears to fail locally.** Run on a clean checkout it
reports `bundle format: expected 1.2, got 1.1`, which looks exactly like a stale committed
artefact that CI is somehow not catching. It is not. `.gitignore` excludes
`results/*.json`, so `results/reference_run.json` is **untracked local output** — in this
working tree, from a run on 2026-08-21. CI regenerates it by running the pipeline and then
checks it, which is the correct order. There is no stale artefact and nothing to fix.

**The promotion gate has four checks, not three.** `accuracy`, `f1_weighted`,
`accuracy_over_baseline` and `latency_p95`, with `all()` over them. A grep for the three
threshold names in `configs/train_config.yaml` misses `latency_p95` because its threshold
is `max_latency_ms`. The website states four and is right.

## What was wrong

### 1. No `permissions` block

Three jobs, no declared token scope. In effect already narrow, because that is the
repository default, but a change to the repository or organisation default would widen it
silently. `permissions: contents: read` is now set at workflow level. That is the whole
requirement even with `actions/upload-artifact`, which needs no repository write.

### 2. Two actions on deprecated Node

`actions/checkout@v4` and `actions/setup-python@v5` target Node 20, which GitHub has
deprecated and is force-running on Node 24. Both moved to `@v7`. `actions/cache@v4` and
`actions/upload-artifact@v4` are already current and were left alone.

### 3. A stale branch on the remote as well as locally

`feature/atomic-model-bundles` had **zero** commits not already on `main`, its tip is a
clean ancestor of `main`, and its pull request was merged on 2026-08-15. `main` was 26
commits ahead. Deleted locally and on the remote.

## What was changed

| Change | Why |
|---|---|
| `permissions: contents: read` | Declared least privilege |
| `checkout@v7`, `setup-python@v7` | Clears the Node 20 deprecation |
| One Mermaid diagram, and the gate as a table | See below |
| Stale branch deleted both sides | Zero unique commits, PR merged in August |
| This file | The brief asks for it |

### The diagram

This repository already keeps its SVGs where the convention wants them, in
`docs/diagrams/`, with light and dark variants served through `<picture>`. Those stay.

The Mermaid diagram added is **not** a redrawing of the pipeline SVG. That SVG draws the
lifecycle as a sequence, which is the right picture for "what are the stages". It cannot
show the thing that actually matters about this pipeline, which is that there are two
points where the run **stops**: the input data checks, and the promotion gate. Those are
decisions, so they are drawn as decisions, with the failure edges in red going to "run
stops" and "not promoted".

Every threshold in it is read from `configs/train_config.yaml`
(`max_null_pct: 0.05`, `max_duplicate_pct: 0.10`, `min_class_ratio: 0.10`,
`min_accuracy: 0.75`, `min_f1: 0.75`, `min_accuracy_over_baseline: 0.20`,
`max_latency_ms: 100`) and every observed value from `EXPECTED` in
`scripts/check_reference_run.py`, so neither is taken from the prose. Strokes use the
blue, green and red already in the figure palette; fills are left to Mermaid's theme so it
stays legible in GitHub's light and dark modes. The link indices were counted and checked,
because an out-of-range `linkStyle` makes Mermaid refuse to render the whole diagram.

## The website agrees with this repository, which is worth saying

Two other repositories in this pass had hand-copied mirrors on the portfolio site that
had drifted from the commit they cite. **This one has not.** `ZQ/src/content/mlops-world.ts`
pins `ada5465` and states `bundleFormat: "1.1"`; that commit's
`check_reference_run.py` carries `"bundle_format_version": "1.1"`. Every metric, every
gate threshold, every comparison direction and the whole split match the code exactly,
including the four-check gate and the three input-data checks.

The one thing to know: **the pin is 12 commits behind `main`**, and the bundle format has
since moved to 1.2 while every metric stayed identical. That is the pin doing its job
rather than a defect, and the site is truthful at the commit it cites. Re-pinning is a
content decision for the website repository, not a fix here.

## What was not changed, and why

- **The `httpx` deprecation.** The suite emits one warning:
  `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated;
  install httpx2 instead`. `requirements-dev.txt` pins `httpx==0.28.1`, which
  `fastapi.testclient` needs today. This will become a real break when starlette drops
  the old client, so it is a scheduled upgrade rather than a defect, and it belongs in its
  own change where the test suite can be run against `httpx2` deliberately. Noted rather
  than done.
- **The reference run numbers.** 0.8067 accuracy is modest and the README says so in those
  words: well above the 0.5 baseline, nowhere near a fine-tuned transformer, on a 600-row
  test split whose confidence interval is roughly plus or minus 3 points. That is the
  correct way to publish a modest result.
- **`results/reference_run.json` in this working tree.** Untracked local output from
  August. It is not mine to tidy and it affects nothing.

## Guidance applied

- **[Security hardening for GitHub Actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions)**:
  the explicit `permissions` block and the action version bumps.
- **[Google Engineering Practices](https://google.github.io/eng-practices/)**: the change
  is CI hygiene and one diagram. The `httpx2` migration was explicitly declined as a
  separate concern rather than folded in.
- **[Conventional Commits](https://www.conventionalcommits.org/)**: `ci:`, since no
  application behaviour moves.
- **PEP 8 via ruff**, which the repository enforces over the whole tree here and which
  passes unchanged.
- **[Diátaxis](https://diataxis.fr/)**: README as explanation and reference, `docs/` for
  the how-to and the per-topic detail. Already right; not restructured.
