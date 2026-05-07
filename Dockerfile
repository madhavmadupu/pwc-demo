     1|# Stage 1: Builder
     2|FROM python:3.11-slim AS builder
     3|
     4|WORKDIR /build
     5|COPY requirements.txt .
     6|RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
     7|
     8|# Stage 2: Production
     9|FROM python:3.11-slim AS production
    10|
    11|LABEL maintainer="Citta AI <team@citta.ai>"
    12|LABEL description="PwC Agentic Document Processing System"
    13|
    14|# Security: non-root user
    15|RUN groupadd -r appuser && useradd -r -g appuser appuser
    16|
    17|WORKDIR /app
    18|COPY --from=builder /install /usr/local
    19|COPY src/ ./src/
    20|COPY pyproject.toml .
    21|
    22|# Environment
    23|ENV PYTHONUNBUFFERED=1
    24|ENV PYTHONDONTWRITEBYTECODE=1
    25|
    26|EXPOSE 8080
    27|
    28|# Health check
    29|HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    30|    CMD curl -f http://localhost:8080/_stcore/health || exit 1
    31|
    32|USER appuser
    33|
    34|CMD ["streamlit", "run", "src/app.py", \
    35|     "--server.port=8080", \
    36|     "--server.address=0.0.0.0", \
    37|     "--server.headless=true", \
    38|     "--browser.gatherUsageStats=false"]
    39|