"""Tests for dataset acquisition, integrity verification and provenance.

None of these touch the network. The download path is exercised by pointing the module
at a locally built archive, so the checksum logic is tested without depending on UCI
being reachable from a CI runner.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from src import datasets


def build_archive(rows_per_file: int = 4) -> bytes:
    """Build an in-memory archive with the same member layout as the real one."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for member in datasets.UCI_SENTIMENT.members:
            lines = []
            for index in range(rows_per_file):
                label = index % 2
                lines.append(f"sample sentence {index}\t{label}")
            archive.writestr(member, "\n".join(lines) + "\n")
    return buffer.getvalue()


@pytest.fixture
def local_spec(tmp_path):
    """A spec whose checksum matches a locally generated archive."""
    payload = build_archive()
    path = tmp_path / "source.zip"
    path.write_bytes(payload)
    spec = datasets.DatasetSpec(
        key="test-dataset",
        name="Test Dataset",
        url=path.as_uri(),
        sha256=datasets.sha256_bytes(payload),
        size_bytes=len(payload),
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Test citation",
        homepage="https://example.invalid/dataset",
        members=datasets.UCI_SENTIMENT.members,
    )
    return spec, payload


class TestSpecRegistry:
    def test_reference_spec_is_registered(self):
        spec = datasets.get_spec(datasets.UCI_SENTIMENT.key)
        assert spec.license_name == "CC BY 4.0"
        assert spec.citation.startswith("Kotzias")
        assert len(spec.sha256) == 64

    def test_unknown_key_is_rejected(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            datasets.get_spec("not-a-dataset")


class TestIntegrity:
    def test_download_verifies_and_caches(self, local_spec, tmp_path):
        spec, payload = local_spec
        cache = tmp_path / "cache"
        path = datasets.ensure_archive(spec, str(cache))
        assert path.read_bytes() == payload

    def test_checksum_mismatch_refuses_to_write(self, local_spec, tmp_path):
        spec, _ = local_spec
        bad = datasets.DatasetSpec(**{**spec.__dict__, "sha256": "0" * 64})
        cache = tmp_path / "cache"
        with pytest.raises(datasets.DatasetIntegrityError, match="Checksum mismatch"):
            datasets.ensure_archive(bad, str(cache))
        # Nothing partial is left behind for a later run to trust.
        assert not (cache / bad.archive_name).exists()

    def test_corrupt_cache_is_discarded_not_reused(self, local_spec, tmp_path):
        spec, payload = local_spec
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / spec.archive_name).write_bytes(b"corrupted")
        path = datasets.ensure_archive(spec, str(cache))
        assert path.read_bytes() == payload

    def test_offline_mode_reports_where_to_get_it(self, local_spec, tmp_path):
        spec, _ = local_spec
        with pytest.raises(datasets.DatasetIntegrityError, match="not cached"):
            datasets.ensure_archive(spec, str(tmp_path / "cache"), allow_download=False)


class TestParsing:
    def test_parses_every_member_into_labelled_rows(self, local_spec, tmp_path):
        spec, _ = local_spec
        archive = datasets.ensure_archive(spec, str(tmp_path / "cache"))
        frame = datasets.parse_archive(spec, archive)
        assert len(frame) == 4 * len(spec.members)
        assert set(frame["sentiment"]) == {"positive", "negative"}
        assert set(frame.columns) == {"review_text", "sentiment", "source"}

    def test_unlabelled_lines_are_skipped_not_guessed(self, tmp_path):
        buffer = io.BytesIO()
        member = datasets.UCI_SENTIMENT.members[0]
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(member, "good\t1\nno label here\nbad\t0\n")
        path = tmp_path / "partial.zip"
        path.write_bytes(buffer.getvalue())
        spec = datasets.DatasetSpec(
            **{**datasets.UCI_SENTIMENT.__dict__, "members": (member,)}
        )
        frame = datasets.parse_archive(spec, path)
        assert len(frame) == 2

    def test_archive_with_no_labels_is_an_error(self, tmp_path):
        buffer = io.BytesIO()
        member = datasets.UCI_SENTIMENT.members[0]
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(member, "nothing parseable\n")
        path = tmp_path / "empty.zip"
        path.write_bytes(buffer.getvalue())
        spec = datasets.DatasetSpec(
            **{**datasets.UCI_SENTIMENT.__dict__, "members": (member,)}
        )
        with pytest.raises(datasets.DatasetIntegrityError, match="No labelled rows"):
            datasets.parse_archive(spec, path)


class TestProvenance:
    def test_record_carries_licence_and_citation(self):
        record = datasets.provenance_record(datasets.UCI_SENTIMENT)
        assert record["license"] == "CC BY 4.0"
        assert "doi.org" in record["citation"]
        assert record["sha256"] == datasets.UCI_SENTIMENT.sha256

    def test_record_contains_no_local_paths(self):
        """Provenance travels inside published bundles, so it must stay path-free."""
        record = datasets.provenance_record(datasets.UCI_SENTIMENT)
        blob = " ".join(record.values())
        assert "C:\\" not in blob
        assert "/home/" not in blob
        assert "/Users/" not in blob

    def test_local_csv_provenance_records_shape_not_path(self, tmp_path):
        import pandas as pd

        frame = pd.DataFrame({"review_text": ["a"], "sentiment": ["positive"]})
        record = datasets.local_provenance(str(tmp_path / "secret" / "data.csv"), frame)
        assert record["name"] == "data.csv"
        assert str(tmp_path) not in " ".join(record.values())
        assert record["license"].startswith("Unknown")


class TestSyntheticFixture:
    def test_is_deterministic(self):
        first = datasets.synthetic_fixture(50, seed=7)
        second = datasets.synthetic_fixture(50, seed=7)
        assert first.equals(second)

    def test_is_balanced_and_labelled(self):
        frame = datasets.synthetic_fixture(100, seed=1)
        assert len(frame) == 100
        assert frame["sentiment"].value_counts().min() == 50

    def test_provenance_marks_it_unusable_as_evidence(self):
        record = datasets.synthetic_provenance(100, 1)
        assert record["kind"] == "synthetic-fixture"
        assert "never be quoted" in record["note"]
