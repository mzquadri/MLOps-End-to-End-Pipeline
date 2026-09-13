"""Training and serving must produce the same features for the same text.

This is the skew that quietly ruins a deployed model: the pipeline computes a feature one
way and the service computes it another, the numbers stay plausible, and nothing fails.
Two things are checked here, because the repository does them differently.

The transformers are genuinely shared - `FeatureStore.transform_single` is the one entry
point both paths use, and only `fit_transform_train` may fit - so TF-IDF and scaling cannot
diverge. The two numeric columns can. Training derives them in pandas
(`.str.len()`, `.str.split().str.len()`) and serving derives them in plain Python
(`len(text)`, `len(text.split())`). Those agree today, but nothing said so, and nothing
would notice if a pandas release changed how `str.split()` treats an exotic space.

The end of it is the parity test: one input, offline prediction and HTTP prediction, equal.
"""

from __future__ import annotations

import random
import string

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src.data_pipeline import add_derived_columns

#: Whitespace that is easy to get wrong: non-breaking, line separator, ideographic,
#: zero-width. A tokeniser that treats any of these differently from the other side
#: changes review_length or word_count without changing anything visible.
AWKWARD = "  　​"


def serving_derivation(text: str) -> dict[str, int]:
    """Exactly what src.serve.AppState.predict_single computes for the numeric columns."""
    return {"review_length": len(text), "word_count": len(text.split())}


def training_derivation(text: str) -> dict[str, int]:
    """Exactly what src.data_pipeline.add_derived_columns computes, through pandas."""
    frame = add_derived_columns(
        pd.DataFrame({"review_text": [text], "sentiment": ["positive"]}),
        "review_text", "sentiment",
    )
    return {
        "review_length": int(frame["review_length"].iloc[0]),
        "word_count": int(frame["word_count"].iloc[0]),
    }


class TestNumericFeatureDerivation:
    @pytest.mark.parametrize(
        "text",
        [
            "normal sentence here", "  leading and trailing  ", "", "   ", "one",
            "tabs\tand\nnewlines", "double  spaces  between", "trailing space ",
            "non breaking space", "​zero width", "　ideographic",
            "a line separator", "hyphen-word count", "emoji \U0001F600 here",
        ],
    )
    def test_the_two_paths_agree(self, text):
        assert training_derivation(text) == serving_derivation(text)

    def test_they_agree_on_random_strings(self):
        """Hand-picked cases only cover what was thought of."""
        random.seed(7)
        alphabet = string.printable + AWKWARD
        for _ in range(2000):
            text = "".join(
                random.choice(alphabet) for _ in range(random.randint(0, 40))
            )
            assert training_derivation(text) == serving_derivation(text), repr(text)

    def test_word_count_counts_words_not_separators(self):
        assert serving_derivation("a  b   c")["word_count"] == 3
        assert serving_derivation("   ")["word_count"] == 0

    def test_review_length_is_the_raw_length_not_the_cleaned_one(self):
        """Cleaning strips punctuation; the feature was fitted on the raw length."""
        text = "Wow!!! Great, really."
        assert training_derivation(text)["review_length"] == len(text)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """One real pipeline run, promoted, shared by the parity tests."""
    from src.pipeline import run

    workspace = tmp_path_factory.mktemp("parity")
    with open("configs/ci_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    summary = run(
        config,
        candidate_dir=str(workspace / "candidates"),
        registry_dir=str(workspace / "registry"),
        results_dir=str(workspace / "results"),
    )
    return workspace / "registry" / "sentiment-classifier" / summary["registry"]["version"]


class TestOfflineAndServingAgree:
    """One input, two code paths, one answer."""

    @pytest.fixture
    def client(self, monkeypatch, bundle):
        from src.serve import app

        monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle))
        with TestClient(app) as test_client:
            yield test_client

    def offline_prediction(self, bundle, text: str):
        """Predict the way training would, and let the service derive its own features.

        The numeric values here come from `training_derivation`, the pandas path used to
        build the training frame. The request below sends only the text, so the service
        derives its own. That is the whole point: if the two derivations ever disagree the
        two predictions disagree, and this test is the thing that notices. Passing
        serving's own numbers in here instead would compare the service against a copy of
        itself and pass no matter what the service did.
        """
        from src.feature_store import FeatureStore
        from src.model_bundle import LINEAGE_FILE, load_trusted_bundle, read_bundle_json

        model, transformers, _ = load_trusted_bundle(str(bundle))
        schema = read_bundle_json(str(bundle), LINEAGE_FILE)["feature_schema"]
        store = FeatureStore(
            {"data": {"text_column": schema["text_column"],
                      "label_column": schema["label_column"]}}
        )
        store.import_transformers(transformers)
        features = store.transform_single(text, training_derivation(text))
        prediction = str(model.predict(features)[0])
        confidence = float(max(model.predict_proba(features)[0]))
        return prediction, confidence

    @pytest.mark.parametrize(
        "text",
        [
            "This product is absolutely wonderful and I love it",
            "Terrible quality, broke after one day, very disappointed",
            "It arrived on time.",
            "Not bad, not great.",
            "a",
            # Chosen to separate the two derivations rather than to read nicely:
            # runs of spaces, leading and trailing space, and a tab.
            "  double  spaces   and  padding  ",
            "tabs	between	words",
        ],
    )
    def test_the_same_text_gets_the_same_answer(self, client, bundle, text):
        offline_label, offline_confidence = self.offline_prediction(bundle, text)
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        served = response.json()
        assert served["prediction"] == offline_label
        # The API rounds to four places; compare at its precision, not beyond it.
        assert served["confidence"] == pytest.approx(round(offline_confidence, 4), abs=1e-9)

    @pytest.mark.parametrize(
        "text", ["This product is absolutely wonderful and I love it",
                 "  double  spaces   and  padding  "]
    )
    def test_the_service_derivation_equals_the_training_derivation(self, client, text):
        """Send the training-derived numbers explicitly; the answer must not move.

        The service fills these in itself when they are absent. If what it computes
        differs from what training computed, these two responses differ.
        """
        implicit = client.post("/predict", json={"text": text}).json()
        explicit = client.post(
            "/predict", json={"text": text, **training_derivation(text)}
        ).json()
        assert implicit["prediction"] == explicit["prediction"]
        assert implicit["confidence"] == explicit["confidence"]
