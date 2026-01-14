"""
Embedding Model Setup for Databricks
=====================================

This script deploys a sentence transformer embedding model (all-MiniLM-L6-v2)
in Databricks for generating 384-dimensional embeddings from text data.

TUTORIAL: MLflow Model Deployment Pipeline
------------------------------------------

This script automates the complete model deployment workflow:

1. **Define Model Wrapper**: Create a PyFunc wrapper for sentence-transformers
2. **Log to MLflow**: Register the model with dependencies in an experiment
3. **Local Validation**: Test the model before deployment
4. **Unity Catalog Registration**: Register for enterprise governance
5. **Endpoint Testing**: Verify the serving endpoint works

Why Use MLflow PyFunc?
----------------------

MLflow's PyFunc interface provides a standardized way to wrap any Python model:

- **Portability**: Same interface regardless of underlying framework
- **Dependency Management**: Captures pip requirements automatically
- **Versioning**: Full model lineage and reproducibility
- **Deployment**: One-click deployment to Model Serving

The all-MiniLM-L6-v2 model produces 384-dimensional embeddings optimized
for semantic similarity tasks. Embeddings are normalized for cosine similarity.

Pipeline Flow:
--------------

    ┌─────────────────────────┐
    │ MiniLMPyfunc Class      │  Custom wrapper for sentence-transformers
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ MLflow Experiment       │  Log model with dependencies
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Local Validation        │  Test embeddings, similarity, edge cases
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Unity Catalog           │  Register for governance
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Serving Endpoint        │  Test deployed endpoint
    └─────────────────────────┘

Usage:
    # Run with defaults (full setup and validation)
    main()

    # Skip endpoint testing (if endpoint not deployed yet)
    main(test_endpoint=False)

    # Skip model registration (for local testing only)
    main(register_model=False, test_endpoint=False)

Cluster Requirements:
    - Databricks Runtime 13.x+ with ML
    - sentence-transformers library available
    - Unity Catalog enabled workspace

Manual Step Required:
    After running this script, create the serving endpoint via Databricks UI:
    1. Navigate to Serving → Create serving endpoint
    2. Select the registered model: airline_test.ml.minilm_l6_v2_embedder
    3. Choose workload size: Small (sufficient for this model)
    4. Enable scale-to-zero for cost optimization

References:
    - MLflow PyFunc: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    - Model Serving: https://docs.databricks.com/machine-learning/model-serving/
    - sentence-transformers: https://www.sbert.net/
"""

from __future__ import annotations

import time
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import requests
from mlflow.pyfunc import PythonModel


# =============================================================================
# IMPORTS FROM MODULAR COMPONENTS
# =============================================================================
# Reuse utilities for consistent output formatting across all scripts.

from load_utils import (
    print_section_header,
    format_duration,
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# These constants define the specific settings for this model setup.
# Modify these to adapt to your environment.

# MLflow experiment path for tracking model runs
MLFLOW_EXPERIMENT = "/Shared/airline-replacement-events-embedding-loadtest"

# Unity Catalog location for model registration
UC_CATALOG = "airline_test"
UC_SCHEMA = "ml"
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.minilm_l6_v2_embedder"

# Model serving endpoint name (must match what you create in UI)
ENDPOINT_NAME = "minilm-embedder"

# Embedding configuration
EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2 output dimensions
BATCH_SIZE = 64  # Encoding batch size for efficiency


# =============================================================================
# CUSTOM PYFUNC MODEL
# =============================================================================

class MiniLMPyfunc(PythonModel):
    """Custom MLflow PyFunc wrapper for all-MiniLM-L6-v2 embeddings.

    Tutorial: Creating a PyFunc Model Wrapper
    -----------------------------------------

    MLflow PyFunc provides a standard interface for any Python model:

    1. **load_context**: Called once when model is loaded
       - Initialize your model here
       - Load weights, config, tokenizers, etc.

    2. **predict**: Called for each inference request
       - Receives model_input (DataFrame or dict)
       - Returns predictions

    This wrapper adds:
    - Input validation (requires "text" column)
    - Batch processing for efficiency
    - Normalized embeddings for cosine similarity

    Example:
        >>> model = MiniLMPyfunc()
        >>> # In MLflow context, load_context is called automatically
        >>> embeddings = model.predict(None, pd.DataFrame({"text": ["Hello world"]}))
        >>> len(embeddings[0])  # 384 dimensions
        384
    """

    def load_context(self, context) -> None:
        """Load the sentence-transformer model.

        This is called once when the model is loaded (not per-request).
        The model is cached in self.model for reuse.

        Args:
            context: MLflow context (contains artifacts path, etc.)
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def predict(self, context, model_input) -> list:
        """Generate embeddings for input texts.

        Args:
            context: MLflow context (unused but required by interface)
            model_input: DataFrame with "text" column, or dict/list convertible to DataFrame

        Returns:
            List of embedding vectors (each is a list of 384 floats)

        Raises:
            ValueError: If input doesn't have a "text" column
        """
        # Handle various input formats
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        if "text" not in model_input.columns:
            raise ValueError("Expected a 'text' column in input DataFrame.")

        texts = model_input["text"].astype(str).tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,  # Good for cosine similarity
            show_progress_bar=False,
        )

        return [e.tolist() for e in embeddings]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors.

    Tutorial: Why Cosine Similarity?
    --------------------------------

    Cosine similarity measures the angle between vectors, not their magnitude.
    This is ideal for embeddings because:

    - Two texts with similar meaning have similar directions
    - Vector length variations (from text length) don't affect similarity
    - Range is [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite

    For normalized vectors (like our embeddings), cosine similarity
    equals dot product, making it very efficient.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def query_endpoint(
    texts: list,
    host: str,
    token: str,
    endpoint_name: str = ENDPOINT_NAME,
) -> list:
    """Query the model serving endpoint.

    Tutorial: Calling Model Serving Endpoints
    -----------------------------------------

    Databricks Model Serving exposes models via REST API:

    - **URL**: https://{workspace}/serving-endpoints/{endpoint}/invocations
    - **Auth**: Bearer token (from dbutils or service principal)
    - **Payload**: JSON with dataframe_records or dataframe_split format

    The endpoint handles:
    - Auto-scaling based on traffic
    - Scale-to-zero when idle (if enabled)
    - Load balancing across replicas

    Args:
        texts: List of text strings to embed
        host: Databricks workspace hostname
        token: API token for authentication
        endpoint_name: Name of the serving endpoint

    Returns:
        List of embedding vectors from the endpoint

    Raises:
        requests.HTTPError: If endpoint returns an error
    """
    url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"dataframe_records": [{"text": t} for t in texts]}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["predictions"]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_local_model(model) -> bool:
    """Validate the logged model locally before deployment.

    Tutorial: Pre-Deployment Validation
    ------------------------------------

    Before deploying a model, verify it works correctly:

    1. **Basic functionality**: Does it return embeddings?
    2. **Correct dimensions**: Are embeddings the right size?
    3. **Semantic validity**: Do similar texts have higher similarity?
    4. **Edge cases**: Does it handle empty strings, long texts, etc.?

    Catching issues here saves debugging time in production.

    Args:
        model: Loaded MLflow PyFunc model

    Returns:
        True if all validations pass, False otherwise
    """
    print_section_header("VALIDATING MODEL LOCALLY")

    # Test 1: Basic functionality
    print("  Testing basic functionality...")
    test_df = pd.DataFrame({"text": [
        "Aircraft engine replacement due to oil leak",
        "Landing gear inspection completed",
        "Hydraulic pump failure requiring immediate replacement",
    ]})

    embeddings = model.predict(test_df)
    if len(embeddings) != 3:
        print(f"  ERROR: Expected 3 embeddings, got {len(embeddings)}")
        return False
    if len(embeddings[0]) != EMBEDDING_DIMENSIONS:
        print(f"  ERROR: Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(embeddings[0])}")
        return False
    print(f"  [OK] Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")

    # Test 2: Semantic similarity
    print("  Testing semantic similarity...")
    similar_texts = pd.DataFrame({"text": [
        "Engine replacement due to failure",
        "Engine swap because of malfunction",  # Similar meaning
        "Cabin seat cushion replacement",  # Different topic
    ]})

    emb = model.predict(similar_texts)
    sim_similar = cosine_similarity(emb[0], emb[1])  # Similar texts
    sim_different = cosine_similarity(emb[0], emb[2])  # Different texts

    print(f"    Similarity (engine texts): {sim_similar:.4f}")
    print(f"    Similarity (engine vs cabin): {sim_different:.4f}")

    if sim_similar <= sim_different:
        print("  ERROR: Similar texts should have higher similarity than different texts")
        return False
    print("  [OK] Semantic similarity test passed")

    # Test 3: Edge cases
    print("  Testing edge cases...")
    edge_cases = pd.DataFrame({"text": [
        "",  # Empty string
        "A",  # Single character
        "x " * 500,  # Long text
        "12345",  # Numbers only
    ]})

    edge_embeddings = model.predict(edge_cases)
    if len(edge_embeddings) != 4:
        print(f"  ERROR: Expected 4 edge case embeddings, got {len(edge_embeddings)}")
        return False
    if not all(len(e) == EMBEDDING_DIMENSIONS for e in edge_embeddings):
        print("  ERROR: Not all edge case embeddings have correct dimensions")
        return False
    print("  [OK] Edge case tests passed")

    print("\n  Local validation PASSED")
    return True


def validate_endpoint(host: str, token: str, local_model) -> bool:
    """Validate the deployed serving endpoint.

    Tutorial: Endpoint Validation
    -----------------------------

    After deploying a model endpoint, verify:

    1. **Connectivity**: Can we reach the endpoint?
    2. **Correct output**: Does it return embeddings?
    3. **Consistency**: Do local and endpoint produce same results?

    The consistency check is critical - it catches:
    - Wrong model version deployed
    - Missing dependencies in serving environment
    - Serialization issues

    Args:
        host: Databricks workspace hostname
        token: API token for authentication
        local_model: Loaded local model for comparison

    Returns:
        True if all validations pass, False otherwise
    """
    print_section_header("VALIDATING SERVING ENDPOINT")
    print(f"  Endpoint: {ENDPOINT_NAME}")

    # Test 1: Basic endpoint functionality
    print("  Testing endpoint connectivity...")
    test_texts = [
        "Replacement of faulty fuel pump",
        "Routine maintenance inspection",
    ]

    try:
        endpoint_embeddings = query_endpoint(test_texts, host, token)
        print(f"  [OK] Endpoint returned {len(endpoint_embeddings)} embeddings")
        print(f"  [OK] Embedding dimension: {len(endpoint_embeddings[0])}")

        if len(endpoint_embeddings[0]) != EMBEDDING_DIMENSIONS:
            print(f"  ERROR: Expected {EMBEDDING_DIMENSIONS} dimensions")
            return False

    except requests.exceptions.HTTPError as e:
        print(f"  ERROR: Endpoint request failed: {e}")
        print("  Make sure the endpoint is deployed and running")
        return False

    # Test 2: Consistency with local model
    print("  Testing local vs endpoint consistency...")
    comparison_text = ["Test text for comparison"]

    local_emb = local_model.predict(pd.DataFrame({"text": comparison_text}))[0]
    endpoint_emb = query_endpoint(comparison_text, host, token)[0]

    diff = np.abs(np.array(local_emb) - np.array(endpoint_emb)).max()
    print(f"    Max difference: {diff:.8f}")

    if diff >= 1e-5:
        print("  ERROR: Local and endpoint results differ significantly")
        return False
    print("  [OK] Local vs endpoint consistency test passed")

    print("\n  Endpoint validation PASSED")
    return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(
    register_model: bool = True,
    test_endpoint: bool = True,
) -> dict:
    """Run the complete model setup and validation pipeline.

    Tutorial: Running Model Setup
    -----------------------------

    This function orchestrates the complete model deployment:

    1. **Log Model**: Create MLflow experiment and log PyFunc model
    2. **Validate Locally**: Test model before deployment
    3. **Register**: Add to Unity Catalog (optional)
    4. **Test Endpoint**: Verify serving endpoint (optional)

    Common Use Cases:
    -----------------

    **Full Setup** (default):
        >>> main()
        # Logs, validates, registers, and tests endpoint

    **Initial Setup** (no endpoint yet):
        >>> main(test_endpoint=False)
        # Logs, validates, and registers; skip endpoint test

    **Local Development**:
        >>> main(register_model=False, test_endpoint=False)
        # Just log and validate locally

    Args:
        register_model: Whether to register model to Unity Catalog
        test_endpoint: Whether to test the serving endpoint

    Returns:
        Dictionary with setup results:
        - run_id: MLflow run ID
        - model_uri: URI for the logged model
        - registered_version: Unity Catalog version (if registered)
        - local_validation: True if local tests passed
        - endpoint_validation: True if endpoint tests passed (or None if skipped)
    """
    pipeline_start = time.time()
    results = {}

    # =========================================================================
    # STEP 1: Print header and set up MLflow experiment
    # =========================================================================
    print_section_header("EMBEDDING MODEL SETUP")
    print("Deploying all-MiniLM-L6-v2 embedding model to Databricks")
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT}")
    print(f"Embedding Dimensions: {EMBEDDING_DIMENSIONS}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # =========================================================================
    # STEP 2: Log the model to MLflow
    # =========================================================================
    print_section_header("LOGGING MODEL TO MLFLOW")

    with mlflow.start_run() as run:
        print(f"  Run ID: {run.info.run_id}")

        mlflow.pyfunc.log_model(
            artifact_path="minilm_embedder",
            python_model=MiniLMPyfunc(),
            pip_requirements=[
                "mlflow",
                "sentence-transformers",
                "torch",
                "transformers",
                "pandas",
                "numpy",
            ],
            input_example=pd.DataFrame({"text": ["Example replacement event text"]}),
        )
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/minilm_embedder"
    print(f"  Model URI: {model_uri}")

    results["run_id"] = run_id
    results["model_uri"] = model_uri

    # =========================================================================
    # STEP 3: Load and validate the model locally
    # =========================================================================
    print_section_header("LOADING MODEL FOR VALIDATION")
    print(f"  Loading from: {model_uri}")

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("  [OK] Model loaded successfully")

    local_valid = validate_local_model(loaded_model)
    results["local_validation"] = local_valid

    if not local_valid:
        print("\nAborting: Local validation failed")
        return results

    # =========================================================================
    # STEP 4: Register model to Unity Catalog (optional)
    # =========================================================================
    if register_model:
        print_section_header("REGISTERING MODEL TO UNITY CATALOG")
        print(f"  Target: {UC_MODEL_NAME}")

        mlflow.set_registry_uri("databricks-uc")

        registered = mlflow.register_model(model_uri=model_uri, name=UC_MODEL_NAME)
        print(f"  [OK] Registered version: {registered.version}")

        results["registered_version"] = registered.version
    else:
        print_section_header("SKIPPING UNITY CATALOG REGISTRATION")
        results["registered_version"] = None

    # =========================================================================
    # STEP 5: Test serving endpoint (optional)
    # =========================================================================
    if test_endpoint:
        # Get Databricks host and token from dbutils (available in Databricks)
        host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

        print(f"\n  Databricks Host: {host}")

        endpoint_valid = validate_endpoint(host, token, loaded_model)
        results["endpoint_validation"] = endpoint_valid
    else:
        print_section_header("SKIPPING ENDPOINT VALIDATION")
        print("  To test the endpoint later, run: main(test_endpoint=True)")
        results["endpoint_validation"] = None

    # =========================================================================
    # STEP 6: Print summary
    # =========================================================================
    print_section_header("SETUP SUMMARY")
    total_time = time.time() - pipeline_start

    print(f"  MLflow Run ID: {results['run_id']}")
    print(f"  Model URI: {results['model_uri']}")
    print(f"  Local Validation: {'PASSED' if results['local_validation'] else 'FAILED'}")

    if results.get("registered_version"):
        print(f"  Unity Catalog Version: {results['registered_version']}")
        print(f"  Full Model Name: {UC_MODEL_NAME}")

    if results.get("endpoint_validation") is not None:
        print(f"  Endpoint Validation: {'PASSED' if results['endpoint_validation'] else 'FAILED'}")
    else:
        print("  Endpoint Validation: SKIPPED")

    print(f"\n  Total setup time: {format_duration(total_time)}")

    if not results.get("endpoint_validation"):
        print("\n  NEXT STEP: Create serving endpoint in Databricks UI")
        print(f"    1. Navigate to Serving -> Create serving endpoint")
        print(f"    2. Select model: {UC_MODEL_NAME}")
        print(f"    3. Endpoint name: {ENDPOINT_NAME}")
        print(f"    4. Workload size: Small")
        print(f"    5. Enable scale-to-zero")

    print("\nDone!")

    return results


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
