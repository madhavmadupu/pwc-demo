#!/bin/bash
# PwC Agentic Document Processor — Cloud Run Deployment
# Usage: ./deploy.sh

set -e

# Configuration
PROJECT_ID="pwc-agentic-demo"
REGION="us-central1"
SERVICE_NAME="pwc-agentic-demo"

echo "🚀 Deploying PwC Agentic Document Processor to Cloud Run..."
echo ""

# Step 1: Enable required APIs
echo "📋 Step 1: Enabling required APIs..."
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    --project=$PROJECT_ID

# Step 2: Set up IAM permissions
echo "📋 Step 2: Setting up IAM permissions..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# Grant Vertex AI access to the default compute service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/aiplatform.user" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectViewer" \
    --quiet

# Step 3: Deploy to Cloud Run
echo "📋 Step 3: Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION" \
    --project=$PROJECT_ID

# Step 4: Get the URL
echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app is running at:"
echo "   $(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')"
echo ""
echo "📊 To view logs:"
echo "   gcloud run services logs read $SERVICE_NAME --region $REGION --limit 50"
echo ""
