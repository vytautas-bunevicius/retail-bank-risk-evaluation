FROM python:3.10-slim

# Install build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all necessary files
COPY requirements.txt .
COPY setup.py .
COPY ./src /app/src
COPY ./app /app/app
COPY ./models /app/models

# Create a minimal README.md if it doesn't exist
RUN echo "# Retail Bank Risk Evaluation" > README.md

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install -e .

ENV PYTHONPATH=/app:/app/src

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
