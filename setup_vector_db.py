"""
Ingest company rules and SOPs into Vertex AI Vector Search.

This script:
1. Generates embeddings for all company rules using Vertex AI Embeddings
2. Creates a GCS bucket for index data
3. Creates a Vertex AI Vector Search index
4. Deploys the index to an endpoint
5. Outputs the endpoint ID for use in app.py

NOTE: Vector Search deployment requires MatchingEngineDeployedIndexNodes quota.
If quota is exceeded, the app uses Vertex AI Embeddings + semantic search directly
as the RAG retrieval mechanism (implemented in app.py).

Usage: python setup_vector_db.py
"""

import json
import os
import time
from google.cloud import aiplatform
from google.cloud import storage
from google import genai
from google.cloud.aiplatform.matching_engine import MatchingEngineIndex, MatchingEngineIndexEndpoint

# ============================================================
# CONFIG
# ============================================================
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "pwc-agentic-demo")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET_NAME = f"{PROJECT_ID}-vector-db"
INDEX_DISPLAY_NAME = "pwc-company-rules-index"
ENDPOINT_DISPLAY_NAME = "pwc-rules-endpoint"
DIMENSIONS = 3072  # gemini-embedding-001 output dimension

# ============================================================
# COMPANY RULES & SOPs
# ============================================================
COMPANY_RULES = {
    "invoice": [
        "All invoices must have a valid invoice number",
        "Invoice date must not be in the future",
        "Tax calculation must match: subtotal x tax rate = tax amount",
        "Line items must sum to subtotal",
        "Payment terms must be specified (Net 15-90 days)",
        "Bank details are required for payments above 50,000 INR",
        "All amounts must be in positive currency",
        "Vendor name and address are mandatory"
    ],
    "contract": [
        "Both parties must be clearly identified",
        "Effective date and term duration must be specified",
        "Compensation terms must be clearly stated",
        "A termination clause is mandatory",
        "Confidentiality clause is required",
        "IP rights must be addressed",
        "Dispute resolution mechanism must be specified",
        "Force majeure clause is recommended"
    ],
    "report": [
        "Report must have a clear title and period",
        "Executive summary must be present (more than 50 chars)",
        "At least 3 key metrics with values required",
        "Trend indicators (up/down/stable) for each metric",
        "Challenges must be documented",
        "Actionable recommendations required",
        "Data sources should be cited",
        "Charts/graphs recommended for visual reports"
    ],
    "email": [
        "Must have valid sender and recipient",
        "Subject line must be present and descriptive",
        "Date must be parseable",
        "Body summary must be coherent (more than 20 chars)",
        "Action items should be clearly identified",
        "Sentiment analysis required",
        "Urgency level must be classified",
        "Attachments should be noted if present"
    ]
}

def build_rules_list():
    """Build a flat list of rules with metadata."""
    rules = []
    for doc_type, rule_list in COMPANY_RULES.items():
        for i, rule in enumerate(rule_list):
            rules.append({
                "id": f"{doc_type}_{i}",
                "doc_type": doc_type,
                "rule": rule,
                "text": f"[{doc_type}] {rule}"
            })
    return rules

def generate_embeddings(rules, client):
    """Generate embeddings for all rules using Vertex AI."""
    print(f"Generating embeddings for {len(rules)} rules...")
    
    texts = [rule["text"] for rule in rules]
    
    # Embed in batches (max 250 per request for embedding-001)
    embeddings = []
    batch_size = 200
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch
        )
        for emb in response.embeddings:
            embeddings.append(emb.values)
        print(f"  Embedded batch {i//batch_size + 1} ({len(batch)} rules)")
    
    return embeddings

def create_gcs_bucket():
    """Create GCS bucket for vector index data."""
    from google.api_core.exceptions import Conflict
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    try:
        bucket.create(location=LOCATION)
        print(f"Created GCS bucket: gs://{BUCKET_NAME}")
    except Conflict:
        print(f"GCS bucket already exists: gs://{BUCKET_NAME}")
    
    return f"gs://{BUCKET_NAME}"

def upload_to_gcs(rules, embeddings, gcs_uri):
    """Upload embeddings as JSON to GCS."""
    print("Uploading embeddings to GCS...")
    
    # Create JSONL format for Vertex AI Vector Search
    jsonl_lines = []
    for rule, emb in zip(rules, embeddings):
        record = {
            "id": rule["id"],
            "embedding": emb,
            "doc_type": rule["doc_type"],
            "rule": rule["rule"],
            "text": rule["text"]
        }
        jsonl_lines.append(json.dumps(record))
    
    # Write to GCS
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("embeddings/rules.json")
    blob.upload_from_string("\n".join(jsonl_lines), content_type="application/json")
    
    gcs_uri = f"gs://{BUCKET_NAME}/embeddings"
    print(f"Uploaded to {gcs_uri}")
    return gcs_uri

def create_vector_index(gcs_uri):
    """Create a Vertex AI Vector Search index."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    print(f"Creating Vector Search index: {INDEX_DISPLAY_NAME}...")
    
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_uri,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=10,
        leaf_node_embedding_count=10,
        leaf_nodes_to_search_percent=10.0,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        description="PwC Company Rules and SOPs Vector Index"
    )
    
    print(f"Index created: {index.resource_name}")
    return index

def deploy_index(index):
    """Deploy the index to an endpoint."""
    print(f"Deploying index to endpoint: {ENDPOINT_DISPLAY_NAME}...")
    
    # Create or get endpoint
    try:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=ENDPOINT_DISPLAY_NAME,
            public_endpoint_enabled=True,
            description="Endpoint for PwC company rules retrieval"
        )
        print(f"Created endpoint: {endpoint.resource_name}")
    except Exception as e:
        # Try to find existing endpoint
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f"display_name={ENDPOINT_DISPLAY_NAME}"
        )
        if endpoints:
            endpoint = endpoints[0]
            print(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            raise
    
    # Deploy index to endpoint
    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id=f"pwc_rules_{int(time.time())}",
        min_replica_count=1,
        max_replica_count=2
    )
    
    print(f"Index deployed to endpoint: {endpoint.resource_name}")
    return endpoint

def main():
    print("=" * 60)
    print("PwC Vector DB Setup - Vertex AI Vector Search")
    print("=" * 60)
    
    # Initialize clients
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    # Step 1: Build rules
    print("\n[Step 1] Building company rules list...")
    rules = build_rules_list()
    print(f"  Total rules: {len(rules)}")
    for doc_type in COMPANY_RULES:
        count = len(COMPANY_RULES[doc_type])
        print(f"    {doc_type}: {count} rules")
    
    # Step 2: Generate embeddings
    print("\n[Step 2] Generating embeddings...")
    embeddings = generate_embeddings(rules, client)
    print(f"  Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")
    
    # Step 3: Create GCS bucket and upload
    print("\n[Step 3] Setting up GCS storage...")
    gcs_uri = create_gcs_bucket()
    gcs_uri = upload_to_gcs(rules, embeddings, gcs_uri)
    
    # Step 4: Create vector index
    print("\n[Step 4] Creating Vertex AI Vector Search index...")
    index = create_vector_index(gcs_uri)
    
    # Step 5: Deploy index
    print("\n[Step 5] Deploying index to endpoint...")
    endpoint = deploy_index(index)
    
    # Output configuration
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"\nAdd these to your .env file:")
    print(f"VECTOR_SEARCH_ENDPOINT={endpoint.resource_name}")
    print(f"DEPLOYED_INDEX_ID={endpoint.deployed_indexes[0].id if endpoint.deployed_indexes else 'N/A'}")
    print(f"\nIndex: {index.resource_name}")
    print(f"Endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    main()
