# Stage 1: run the data-preparation step so the API can retrieve from all 1,000 abstracts.
FROM python:3.11-slim AS data
WORKDIR /build
RUN pip install --no-cache-dir datasets pandas scikit-learn==1.6.1
COPY src/ src/
COPY data/sample.csv data/sample.csv
# If the download fails (no network), the API falls back to the committed 50-row sample.
RUN python src/prepare_data.py || echo "PubMedQA download failed, serving with data/sample.csv"

# Stage 2: the API image
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY --from=data /build/data/ data/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
