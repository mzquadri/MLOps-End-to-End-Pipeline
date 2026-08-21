"""Dataset acquisition with recorded provenance.

The reference run uses a real, openly licensed dataset rather than generated text.
Two rules shape this module:

1. The dataset is never committed. It is downloaded on demand into a local cache and
   verified against a pinned SHA-256 before anything reads it.
2. Provenance travels with the model. The source URL, license, citation and checksum are
   returned alongside the data so training can record them in the bundle lineage.

The checksum answers "did I get the bytes I expected", not "is this source trustworthy".
It detects a truncated download, a silently re-cut archive, or a corrupted cache. It is
not a signature, and it cannot make an untrusted host safe. This is the same integrity
boundary the model bundle draws, and it is worth being explicit about both times.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 120
LABEL_NAMES = {0: "negative", 1: "positive"}


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to fetch a dataset and to credit it correctly."""

    key: str
    name: str
    url: str
    sha256: str
    size_bytes: int
    license_name: str
    license_url: str
    citation: str
    homepage: str
    members: tuple[str, ...] = field(default_factory=tuple)

    @property
    def archive_name(self) -> str:
        return f"{self.key}.zip"


UCI_SENTIMENT = DatasetSpec(
    key="uci-sentiment-labelled-sentences",
    name="Sentiment Labelled Sentences",
    url="https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
    sha256="afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343",
    size_bytes=84_188,
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    citation=(
        "Kotzias, D. (2015). Sentiment Labelled Sentences [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C57604"
    ),
    homepage="https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
    members=(
        "sentiment labelled sentences/amazon_cells_labelled.txt",
        "sentiment labelled sentences/imdb_labelled.txt",
        "sentiment labelled sentences/yelp_labelled.txt",
    ),
)

DATASETS: dict[str, DatasetSpec] = {UCI_SENTIMENT.key: UCI_SENTIMENT}


class DatasetIntegrityError(RuntimeError):
    """Raised when downloaded bytes do not match the pinned checksum."""


def get_spec(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"Unknown dataset '{key}'. Known: {sorted(DATASETS)}") from None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_archive(
    spec: DatasetSpec,
    cache_dir: str = "data/cache",
    *,
    allow_download: bool = True,
) -> Path:
    """Return a checksum-verified local archive path, downloading only if needed.

    A cached file whose checksum does not match is treated as corrupt and removed,
    so a bad cache cannot poison every later run.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / spec.archive_name

    if archive.is_file():
        digest = sha256_bytes(archive.read_bytes())
        if digest == spec.sha256:
            logger.info("Using cached dataset archive %s", archive)
            return archive
        logger.warning("Cached archive checksum mismatch; discarding %s", archive)
        archive.unlink()

    if not allow_download:
        raise DatasetIntegrityError(
            f"Dataset '{spec.key}' is not cached and downloading is disabled. "
            f"Fetch it manually from {spec.homepage} into {archive}."
        )

    logger.info("Downloading %s (%s)", spec.name, spec.url)
    with urlopen(spec.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = response.read()

    digest = sha256_bytes(payload)
    if digest != spec.sha256:
        raise DatasetIntegrityError(
            f"Checksum mismatch for {spec.key}: expected {spec.sha256}, got {digest}. "
            "The archive was not written."
        )

    archive.write_bytes(payload)
    logger.info("Verified and cached %s (%d bytes)", archive, len(payload))
    return archive


def parse_archive(spec: DatasetSpec, archive_path: Path) -> pd.DataFrame:
    """Parse the three labelled source files into one tidy frame.

    Each line is `<sentence>\\t<0 or 1>`. Lines that do not carry a valid label are
    skipped rather than guessed at; the count is logged so silent loss is visible.
    """
    rows: list[tuple[str, str, str]] = []
    skipped = 0

    with zipfile.ZipFile(archive_path) as bundle:
        for member in spec.members:
            raw = bundle.read(member).decode("utf-8", errors="replace")
            source = Path(member).stem.replace("_labelled", "").replace("_cells", "")
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                text, separator, label = line.rpartition("\t")
                if not separator or label.strip() not in {"0", "1"}:
                    skipped += 1
                    continue
                rows.append((text.strip(), LABEL_NAMES[int(label)], source))

    if skipped:
        logger.warning("Skipped %d unparseable lines while reading %s", skipped, spec.key)
    if not rows:
        raise DatasetIntegrityError(f"No labelled rows parsed from {archive_path}")

    frame = pd.DataFrame(rows, columns=["review_text", "sentiment", "source"])
    logger.info("Parsed %d labelled sentences from %s", len(frame), spec.name)
    return frame


def load_reference_dataset(
    key: str = UCI_SENTIMENT.key,
    cache_dir: str = "data/cache",
    *,
    allow_download: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the licensed reference dataset and the provenance record describing it."""
    spec = get_spec(key)
    archive = ensure_archive(spec, cache_dir, allow_download=allow_download)
    frame = parse_archive(spec, archive)
    return frame, provenance_record(spec)


def provenance_record(spec: DatasetSpec) -> dict[str, str]:
    """Provenance safe to embed in a bundle: no local paths, no credentials."""
    return {
        "kind": "licensed-download",
        "key": spec.key,
        "name": spec.name,
        "url": spec.url,
        "sha256": spec.sha256,
        "license": spec.license_name,
        "license_url": spec.license_url,
        "citation": spec.citation,
        "homepage": spec.homepage,
    }


def synthetic_provenance(n_rows: int, seed: int) -> dict[str, str]:
    return {
        "kind": "synthetic-fixture",
        "key": "synthetic-reviews",
        "name": "Deterministic synthetic review fixture",
        "license": "Not applicable (generated in-process)",
        "citation": "Not applicable",
        "note": (
            "Template-generated text for tests and CI only. It is close to linearly "
            "separable and must never be quoted as a model-quality result."
        ),
        "n_rows": str(n_rows),
        "seed": str(seed),
    }


def synthetic_fixture(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Deterministic stand-in used by tests and CI, never as published evidence.

    This exists so the pipeline can run with no network in under a second. It is a
    small template vocabulary, so a linear model reaches near-perfect accuracy on it.
    That is a property of the fixture, not of the model, which is exactly why the
    reference run uses the licensed dataset instead.
    """
    rng = np.random.default_rng(seed)
    positive = [
        "this product is excellent and works great",
        "i love this item it exceeded my expectations",
        "very happy with the quality and fast delivery",
        "best purchase i have made this year",
        "outstanding product highly recommended",
    ]
    negative = [
        "terrible quality broke after one day",
        "very disappointed with this purchase",
        "do not buy this product waste of money",
        "horrible experience would not recommend",
        "poor quality and even worse customer service",
    ]

    texts: list[str] = []
    labels: list[str] = []
    for index in range(n):
        pool, label = (positive, "positive") if index % 2 == 0 else (negative, "negative")
        texts.append(f"{rng.choice(pool)} number {index}")
        labels.append(label)

    return pd.DataFrame(
        {"review_text": texts, "sentiment": labels, "source": "synthetic"}
    )


def read_local_csv(path: str, text_column: str, label_column: str) -> pd.DataFrame:
    """Read a user-supplied CSV, failing loudly if the declared columns are absent."""
    frame = pd.read_csv(path)
    missing = [c for c in (text_column, label_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def local_provenance(path: str, frame: pd.DataFrame) -> dict[str, str]:
    """Provenance for user-supplied data: record the shape, never the path."""
    return {
        "kind": "local-csv",
        "key": "user-supplied",
        "name": Path(path).name,
        "license": "Unknown - supplied by the operator",
        "citation": "Not applicable",
        "note": "Licensing and redistribution rights are the operator's responsibility.",
        "n_rows": str(len(frame)),
    }


def describe_archive(spec: DatasetSpec, archive_path: Path | None = None) -> str:
    location = archive_path or Path("data/cache") / spec.archive_name
    return (
        f"{spec.name} ({spec.license_name})\n"
        f"  source   {spec.url}\n"
        f"  sha256   {spec.sha256}\n"
        f"  cached   {location}\n"
        f"  cite     {spec.citation}"
    )


def _read_archive_bytes(payload: bytes) -> zipfile.ZipFile:
    """Helper used by tests to build in-memory archives."""
    return zipfile.ZipFile(io.BytesIO(payload))
