FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=0

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Python deps + Playwright Chromium and OS deps into the final runtime image.
RUN python -m pip install -U pip \
    && python -m pip install -r /app/requirements.txt \
    && python -m playwright install --with-deps chromium

COPY src /app/src
COPY pyproject.toml /app/pyproject.toml

ENV PYTHONPATH=/app/src

CMD ["sh", "-lc", "uvicorn gradenza_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
