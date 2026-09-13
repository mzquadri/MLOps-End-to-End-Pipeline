"""Identical texts can appear in more than one split, because the source data repeats.

The split itself is clean: `three_way_split` partitions by index and
test_three_splits_are_disjoint_and_sized_as_configured already proves no row is in two
places. That is not the same as no *text* being in two places. The UCI corpus contains 18
exact duplicate rows in 3,000, and stratified sampling has no reason to keep copies
together, so a sentence can be trained on and then scored.

Measured on the reference configuration rather than argued about:

  test        2 of 600 rows (0.33%) repeat a training text, accuracy 0.8067 -> 0.8060
  validation  7 of 600 rows (1.17%) repeat a training text, accuracy 0.8117 -> 0.8094

Every repeated row is predicted correctly, which is what makes them mildly flattering.
Validation is the more contaminated of the two, and it is the split the promotion gate
thresholds were derived from, so it is the one worth naming: the gate sits at 0.75 against
an observed 0.8117 that is inflated by 0.0022. Both effects are an order of magnitude
inside the +/-3 point interval of a 600-row split, and neither comes close to explaining
the 0.31 margin over the majority-class baseline.

These tests exist so that it stays that small. Deduplicating instead would change the
published number for a rounding-level effect, which is a worse trade than measuring it.

Marked `network`: they need the licensed dataset. The reference-run CI job has it.
"""

from __future__ import annotations

import pytest
import yaml

from src.data_pipeline import load_and_preprocess
from src.train import three_way_split

#: Above this, the contamination stops being a footnote and starts being a finding.
#: Measured at 0.0033 on test and 0.0117 on validation, so the bound is set at twice the
#: larger of the two: loose enough not to break on a dataset revision, tight enough that
#: a real change in the source data fails here instead of moving a published number.
MAX_CONTAMINATED_FRACTION = 0.025

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def reference_split():
    with open("configs/train_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    frame, _, _, _ = load_and_preprocess(config)
    labels = frame[config["data"]["label_column"]].to_numpy()
    train, validation, test, *_ = three_way_split(
        frame, labels,
        config["data"]["test_size"], config["data"]["validation_size"],
        config["data"]["random_state"],
    )
    return config, frame, train, validation, test


def test_the_split_sizes_are_the_documented_ones(reference_split):
    _, frame, train, validation, test = reference_split
    assert (len(train), len(validation), len(test)) == (1800, 600, 600)
    assert len(frame) == 3000


def test_cross_split_text_duplication_stays_negligible(reference_split):
    config, _, train, _, test = reference_split
    column = config["data"]["text_column"]
    shared = set(train[column]) & set(test[column])
    contaminated = int(test[column].isin(shared).sum())
    fraction = contaminated / len(test)
    assert fraction <= MAX_CONTAMINATED_FRACTION, (
        f"{contaminated} of {len(test)} test rows ({fraction:.2%}) repeat a training text; "
        "the measured baseline was 2 rows (0.33%)"
    )


def test_validation_is_checked_too_since_the_gate_is_derived_from_it(reference_split):
    """Validation carries more of this than test, and the gate is set from validation."""
    config, _, train, validation, _ = reference_split
    column = config["data"]["text_column"]
    shared = set(train[column]) & set(validation[column])
    contaminated = int(validation[column].isin(shared).sum())
    fraction = contaminated / len(validation)
    assert fraction <= MAX_CONTAMINATED_FRACTION, (
        f"{contaminated} of {len(validation)} validation rows ({fraction:.2%}) repeat a "
        "training text; the measured baseline was 7 rows (1.17%)"
    )


def test_duplicated_texts_never_carry_conflicting_labels(reference_split):
    """A repeated sentence with two different labels would be a labelling fault.

    That is a different and much worse problem than duplication, so it is separated:
    duplication costs a little optimism, contradictory labels cost trust in the target.
    """
    config, frame, *_ = reference_split
    text, label = config["data"]["text_column"], config["data"]["label_column"]
    conflicting = (
        frame.groupby(text)[label].nunique().loc[lambda counts: counts > 1]
    )
    assert conflicting.empty, f"{len(conflicting)} texts carry more than one label"


def test_the_whole_dataset_duplicate_rate_matches_what_the_lineage_records(reference_split):
    """The validator records this number into every bundle; it should be true."""
    _, frame, *_ = reference_split
    assert float(frame.duplicated().mean()) == pytest.approx(0.0057, abs=0.0005)
