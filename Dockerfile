FROM python:3.10-slim

# Install build essentials and curl for uv installation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy all necessary files
COPY pyproject.toml .
COPY ./src /app/src
COPY ./app /app/app
COPY ./models /app/models

# Install dependencies using uv
RUN uv pip install --no-cache .

ENV PYTHONPATH=/app:/app/src

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}