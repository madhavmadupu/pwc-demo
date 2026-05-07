# Deployment Guide — PwC Agentic Document Processing

## Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- Docker (for container deployment)

## Local Development

### 1. Clone and setup

```bash
git clone <repository-url>
cd pwc-demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Google Cloud settings
```

### 3. Run locally

```bash
streamlit run src/app.py
```

### 4. Run with Docker Compose (hot-reload)

```bash
docker compose up
```

## Production Deployment

### Build the image

```bash
docker build --target production -t pwc-agentic-doc-processor:latest .
```

### Run the container

```bash
docker run -d \
  -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_CLOUD_LOCATION=us-central1 \
  -e VECTOR_SEARCH_ENDPOINT=your-endpoint \
  -e DEPLOYED_INDEX_ID=your-index \
  pwc-agentic-doc-processor:latest
```

## Google Cloud Run Deployment

```bash
# Build and push to Artifact Registry
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/REPO/pwc-agentic-doc:latest

# Deploy to Cloud Run
gcloud run deploy pwc-agentic-doc \
  --image us-central1-docker.pkg.dev/PROJECT_ID/REPO/pwc-agentic-doc:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=PROJECT_ID
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag.py -v
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | `pwc-agentic-demo` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Yes | `us-central1` | GCP region |
| `VECTOR_SEARCH_ENDPOINT` | No | — | Vertex AI Vector Search endpoint |
| `DEPLOYED_INDEX_ID` | No | — | Deployed vector index ID |

## Health Check

The application exposes a health endpoint at `/_stcore/health` (Streamlit default).

```bash
curl http://localhost:8080/_stcore/health
```
