# One image, two roles. The API and the RQ worker run the same code and differ
# only in the command, so building them separately would only let them drift.
#
#   docker build -t frontier .
#   docker run -p 8000:8000 frontier                  # API
#   docker run frontier frontier-worker               # worker
#
# `docker compose up` wires both to Redis; see compose.yaml.
#
# The keras/TensorFlow backend is deliberately not installed -- 1.3 GB on disk
# for the one model the measured table shows is worse than a random walk. It
# drops out of /health cleanly. Add it with --build-arg EXTRAS="--extra keras".

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.13

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- build: resolve and install into a self-contained /srv/.venv ---------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /srv

ARG EXTRAS=""

# Dependencies first, without the project: this layer is keyed on pyproject.toml
# and uv.lock alone, so editing source does not re-download torch. --locked
# fails the build if the lockfile is stale rather than resolving something else.
#
# Deliberately plain COPY/RUN rather than BuildKit cache and bind mounts: those
# are faster on rebuilds but make the image unbuildable with the classic
# builder, which is what `docker build` still uses wherever buildx is not
# installed.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project ${EXTRAS}

COPY . /srv

# --no-editable copies the code into site-packages, so the runtime stage needs
# the venv and nothing else.
RUN uv sync --locked --no-dev --no-editable ${EXTRAS}

# --- runtime -------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# LightGBM links against system libgomp; the torch wheel ships its own copy.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root, and a fixed uid so a bind-mounted data/ has predictable ownership.
RUN useradd --create-home --uid 10001 frontier

WORKDIR /srv
COPY --from=builder --chown=frontier:frontier /srv/.venv /srv/.venv

# The Parquet store and the MLflow database are written relative to the working
# directory; mount a volume here to keep them across container restarts.
RUN install -d -o frontier -g frontier /srv/data

ENV PATH="/srv/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER frontier
EXPOSE 8000

# urllib rather than curl, so the image needs no extra package. /health reports
# the job backend, so this fails if Redis was required and is unreachable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
