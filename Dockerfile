# Inference image. Serving only - it does not train, evaluate, or promote.
#
# The model is not baked in. A bundle is mounted or copied at deploy time, so the image
# is independent of any particular model version and a promotion does not require a
# rebuild. The previous version ran `COPY models/ models/`, which copied an empty
# directory on a clean clone and implied a model was present when none was.

FROM python:3.11-slim

# Fail fast on unbuffered output and stale bytecode in a container context.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt actually changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

# Run as a non-root user. The application needs no write access to its own code.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The healthcheck uses the interpreter that is already in the image. The previous
# version shelled out to curl, which python:3.11-slim does not ship, so the container
# reported unhealthy forever. Readiness is checked rather than liveness: a process that
# is up with no usable model should not be receiving traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).status==200 else 1)"]

CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
