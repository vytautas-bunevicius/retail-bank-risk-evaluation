FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml .
COPY ./src /app/src
COPY ./app /app/app
COPY ./models /app/models

RUN uv pip install --no-cache .

ENV PYTHONPATH=/app:/app/src

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}