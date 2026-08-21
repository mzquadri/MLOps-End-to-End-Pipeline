"""Container integration test.

A container that builds proves very little. This builds the image, mounts a bundle that
was produced by the real pipeline, starts the service, waits for readiness, and makes an
actual prediction over HTTP. It also checks the two properties that are easy to regress:
the process is not root, and no model or dataset is baked into the image.

Marked `docker` and skipped automatically when no daemon is reachable, so the default
suite stays runnable on a laptop without Docker.

    python -m pytest tests/test_docker.py -m docker
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.docker

IMAGE_TAG = "mlops-pipeline-test:latest"
CONTAINER_PORT = 8000
READY_TIMEOUT_SECONDS = 90


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker daemon is not reachable"
)


def run_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # Decode explicitly: docker emits UTF-8, while the default console encoding on a
    # Windows runner is cp1252 and raises on build output containing box characters.
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def wait_for_ready(url: str, timeout: int = READY_TIMEOUT_SECONDS) -> dict:
    """Poll readiness until the service reports it can actually serve."""
    deadline = time.monotonic() + timeout
    last_error = "no attempt made"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return json.loads(response.read())
                last_error = f"status {response.status}"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1.5)
    raise AssertionError(f"Service never became ready: {last_error}")


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


@pytest.fixture(scope="module")
def image() -> str:
    """Build the image from a clean context."""
    result = run_docker("build", "-t", IMAGE_TAG, ".", check=False)
    assert result.returncode == 0, f"docker build failed:\n{result.stderr[-4000:]}"
    return IMAGE_TAG


def make_world_readable(root: Path) -> None:
    """Let the container's unprivileged user read a bundle produced on the host.

    pytest creates tmp directories as 0700 owned by the invoking user. Bind-mounting
    one into a container that runs as uid 10001 then fails with EACCES on Linux, while
    passing on Docker Desktop for Windows, which does not enforce Unix ownership across
    the share. This is not a test-only quirk: any non-root deployment has to make the
    bundle readable by the runtime user, so doing it explicitly here mirrors what a real
    deploy must arrange.
    """
    for path in [root, *root.rglob("*")]:
        # Windows has no POSIX mode bits to set; the mount is readable regardless.
        with contextlib.suppress(OSError, NotImplementedError):
            path.chmod(0o755 if path.is_dir() else 0o644)


@pytest.fixture(scope="module")
def promoted_bundle(tmp_path_factory) -> Path:
    """Produce a real production bundle on the host, to be mounted into the container."""
    from src.pipeline import run

    workspace = tmp_path_factory.mktemp("docker-bundle")
    with open("configs/ci_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run(
        config,
        candidate_dir=str(workspace / "models" / "candidates"),
        registry_dir=str(workspace / "models" / "registry"),
        results_dir=str(workspace / "results"),
    )
    # tmp_path_factory roots are 0700; the container user must be able to traverse them.
    make_world_readable(workspace)
    return workspace / "models"


@pytest.fixture(scope="module")
def running_container(image, promoted_bundle):
    """Start the service with the bundle mounted read-only, then tear it down."""
    name = f"mlops-test-{uuid.uuid4().hex[:8]}"
    result = run_docker(
        "run",
        "--rm",
        "--detach",
        "--name",
        name,
        "--publish",
        f"0:{CONTAINER_PORT}",
        "--volume",
        f"{promoted_bundle.as_posix()}:/app/models:ro",
        "--env",
        "MODEL_BUNDLE_PATH=models/registry/sentiment-classifier/v1",
        image,
        check=False,
    )
    assert result.returncode == 0, f"docker run failed:\n{result.stderr[-4000:]}"

    try:
        port_result = run_docker("port", name, str(CONTAINER_PORT))
        host_port = port_result.stdout.strip().splitlines()[0].rsplit(":", 1)[-1]
        yield name, f"http://127.0.0.1:{host_port}"
    finally:
        logs = run_docker("logs", name, check=False)
        print(f"--- container logs ---\n{logs.stdout}\n{logs.stderr}")
        run_docker("stop", "--time", "5", name, check=False)


@requires_docker
class TestContainerImage:
    def test_no_model_or_data_is_baked_into_the_image(self, image):
        """The image must be model-independent; a promotion should not need a rebuild."""
        result = run_docker(
            "run", "--rm", "--entrypoint", "python", image,
            "-c",
            "import os,json;"
            "print(json.dumps(sorted(os.listdir('/app'))))",
        )
        contents = json.loads(result.stdout.strip())
        assert "models" not in contents
        assert "data" not in contents
        assert "src" in contents and "configs" in contents

    def test_runs_as_a_non_root_user(self, image):
        result = run_docker("run", "--rm", "--entrypoint", "python", image,
                            "-c", "import os;print(os.getuid())")
        assert result.stdout.strip() == "10001"

    def test_healthcheck_does_not_depend_on_curl(self, image):
        """The previous healthcheck shelled out to curl, which this base image lacks."""
        inspected = run_docker("inspect", "--format", "{{json .Config.Healthcheck}}", image)
        healthcheck = json.loads(inspected.stdout.strip())
        assert healthcheck is not None
        assert "curl" not in " ".join(healthcheck["Test"])


@requires_docker
class TestContainerService:
    def test_becomes_ready_and_serves_a_real_prediction(self, running_container):
        _, base_url = running_container

        ready = wait_for_ready(f"{base_url}/ready")
        assert ready["ready"] is True
        assert ready["model_version"] == "v1"

        health = json.loads(urllib.request.urlopen(f"{base_url}/health").read())
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True

        prediction = post_json(
            f"{base_url}/predict",
            {"text": "this product is excellent and works great"},
        )
        assert prediction["prediction"] in {"positive", "negative"}
        assert 0.0 < prediction["confidence"] <= 1.0
        assert prediction["model_version"] == "v1"

    def test_batch_prediction_over_http(self, running_container):
        _, base_url = running_container
        wait_for_ready(f"{base_url}/ready")
        body = post_json(
            f"{base_url}/predict/batch",
            {"texts": ["excellent quality", "terrible waste of money"]},
        )
        assert len(body["predictions"]) == 2

    def test_metrics_reflect_the_requests_made(self, running_container):
        _, base_url = running_container
        wait_for_ready(f"{base_url}/ready")
        post_json(f"{base_url}/predict", {"text": "a review"})
        metrics = json.loads(urllib.request.urlopen(f"{base_url}/metrics").read())
        assert metrics["ready"] is True
        assert metrics["total_predictions"] >= 1
