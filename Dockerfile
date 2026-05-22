# Multi-stage build using the official uv image so we get a vendored uv binary
# and a known-good Python toolchain.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install OS deps required by Playwright/Chromium at build time so the wheels
# (and Chromium binary) can run inside the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy lock + manifest first so layer caching reuses the venv on code-only changes.
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
RUN uv sync --frozen --no-dev

# Fetch Chromium with all browser deps.
RUN uv run playwright install --with-deps chromium


FROM python:3.13-slim-bookworm

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Runtime libs Chromium needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright
COPY app ./app
COPY chainlit.md ./chainlit.md
COPY .chainlit ./.chainlit

EXPOSE 8000
CMD ["chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
