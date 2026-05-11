FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
    && uv pip install --system -r pyproject.toml

COPY app.py ./
COPY src ./src
COPY Site_Internet ./Site_Internet

# data/, models/, results/ sont fournis par des volumes au runtime
# (cf. docker-compose.yml) — ne pas les embarquer dans l'image.

EXPOSE 5000
CMD ["python", "app.py"]
