"""A bundle must be able to say what produced it.

Format 1.1 made a bundle say where its *data* came from. It still could not say what
*code* built it. A deployed model.joblib recorded the dataset, the split and the metrics,
and nothing at all about the commit or the scikit-learn version, so the failure this
record exists to make diagnosable - an artifact pickled by one version and loaded by
another - had no evidence to work from.

Format 1.2 adds that record. It carries no timestamp on purpose: lineage.json is
byte-identical across two runs of the same commit on the same machine, which is a property
worth keeping, and the creation time is already in the registry index and the run summary.
"""

from __future__ import annotations

import json

import pytest

from src.model_bundle import (
    PROVENANCE_PACKAGES,
    UNKNOWN_COMMIT,
    BundleIntegrityError,
    _validate_lineage,
    runtime_provenance,
    source_commit,
    validate_bundle,
)
from tests.conftest import create_passing_bundle, make_lineage


class TestRuntimeProvenance:
    def test_it_records_the_interpreter(self):
        import platform

        assert runtime_provenance()["python"] == platform.python_version()

    @pytest.mark.parametrize("package", PROVENANCE_PACKAGES)
    def test_every_declared_package_version_is_captured(self, package):
        from importlib.metadata import version

        recorded = runtime_provenance()["packages"][package]
        assert recorded == version(package)

    def test_scikit_learn_is_recorded_because_it_decides_whether_the_pickle_loads(self):
        assert "scikit-learn" in PROVENANCE_PACKAGES
        assert runtime_provenance()["packages"]["scikit-learn"]

    def test_it_carries_no_timestamp(self):
        """Two calls must be equal, or lineage.json stops being reproducible."""
        assert runtime_provenance() == runtime_provenance()
        flat = json.dumps(runtime_provenance()).lower()
        for word in ("timestamp", "created_at", "generated_at"):
            assert word not in flat

    def test_it_contains_no_absolute_path(self):
        """_has_absolute_path rejects the whole lineage if it does."""
        flat = json.dumps(runtime_provenance())
        assert ":\\" not in flat and '"/' not in flat


class TestSourceCommit:
    def test_an_environment_variable_takes_priority(self, monkeypatch):
        """The image excludes .git, so a build states its commit this way."""
        monkeypatch.setenv("SOURCE_COMMIT", "abc123")
        assert source_commit() == "abc123"

    @pytest.mark.parametrize("variable", ["SOURCE_COMMIT", "GIT_COMMIT", "GITHUB_SHA"])
    def test_each_supported_variable_is_read(self, monkeypatch, variable):
        for name in ("SOURCE_COMMIT", "GIT_COMMIT", "GITHUB_SHA"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(variable, "deadbeef")
        assert source_commit() == "deadbeef"

    def test_blank_variables_are_ignored_rather_than_recorded(self, monkeypatch):
        monkeypatch.setenv("SOURCE_COMMIT", "   ")
        for name in ("GIT_COMMIT", "GITHUB_SHA"):
            monkeypatch.delenv(name, raising=False)
        assert source_commit() != "   "

    def test_it_admits_when_it_cannot_tell(self, monkeypatch):
        """A tarball build genuinely has no commit. Saying so beats inventing one."""
        for name in ("SOURCE_COMMIT", "GIT_COMMIT", "GITHUB_SHA"):
            monkeypatch.delenv(name, raising=False)

        def no_git(*args, **kwargs):
            raise OSError("git not found")

        monkeypatch.setattr("src.model_bundle.subprocess.run", no_git)
        assert source_commit() == UNKNOWN_COMMIT


class TestLineageRequiresProvenance:
    """The contract has to refuse, or it is a suggestion.

    These drive _validate_lineage directly. It is the function that decides whether a
    bundle may exist, and reaching it through a full bundle build would test the builder
    rather than the rule.
    """

    def test_a_lineage_without_runtime_is_rejected(self):
        lineage = make_lineage()
        lineage.pop("runtime")
        with pytest.raises(BundleIntegrityError, match="runtime provenance"):
            _validate_lineage(lineage)

    @pytest.mark.parametrize("field", ["python", "source_commit"])
    def test_a_missing_runtime_field_is_rejected(self, field):
        lineage = make_lineage()
        lineage["runtime"].pop(field)
        with pytest.raises(BundleIntegrityError, match=field.replace("_", "[_ ]")):
            _validate_lineage(lineage)

    @pytest.mark.parametrize("value", ["", None, 42, []])
    def test_a_non_string_runtime_field_is_rejected(self, value):
        lineage = make_lineage()
        lineage["runtime"]["source_commit"] = value
        with pytest.raises(BundleIntegrityError):
            _validate_lineage(lineage)

    def test_missing_package_versions_are_rejected(self):
        lineage = make_lineage()
        lineage["runtime"]["packages"] = {}
        with pytest.raises(BundleIntegrityError, match="package versions"):
            _validate_lineage(lineage)

    @pytest.mark.parametrize("package", PROVENANCE_PACKAGES)
    def test_dropping_any_declared_package_is_rejected(self, package):
        lineage = make_lineage()
        lineage["runtime"]["packages"].pop(package)
        with pytest.raises(BundleIntegrityError, match=package):
            _validate_lineage(lineage)

    def test_a_complete_lineage_passes(self):
        _validate_lineage(make_lineage())


class TestRealBundleCarriesProvenance:
    def test_a_built_bundle_validates_at_format_1_2(self, tmp_path):
        create_passing_bundle(tmp_path / "bundle")
        manifest = validate_bundle(tmp_path / "bundle")
        assert manifest["bundle_format_version"] == "1.2"

    def test_the_built_bundle_records_the_versions_that_made_it(self, tmp_path):
        from importlib.metadata import version

        create_passing_bundle(tmp_path / "bundle")
        lineage = json.loads((tmp_path / "bundle" / "lineage.json").read_text(encoding="utf-8"))
        assert lineage["runtime"]["packages"]["scikit-learn"] == version("scikit-learn")
        assert lineage["runtime"]["source_commit"]
