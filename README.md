# Embedding Load Test - Quick Start Guide

This guide walks you through setting up Databricks notebooks for generating embeddings and loading them into Neo4j.

## Prerequisites

- Databricks workspace with access to create compute clusters
- Neo4j database (Aura or self-hosted)
- Databricks CLI installed and configured

## 1. Create Databricks Compute Cluster

Create a new compute cluster using **Machine Learning** runtime:

1. Go to **Compute** in your Databricks workspace
2. Click **Create Compute**
3. Under **Databricks Runtime Version**, select a **ML** runtime (e.g., `15.4 LTS ML`)
   - Machine Learning runtimes include preinstalled libraries: PyTorch, TensorFlow, XGBoost, MLflow, and more
4. Configure node type and auto-scaling as needed
5. Click **Create Compute**

### Additional Python Dependencies

After creating the cluster, install the following additional packages:

```
sentence-transformers
faker
neo4j
requests
tenacity
```

**To add dependencies:**
1. Navigate to your cluster's **Libraries** tab
2. Click **Install New** → **PyPI**
3. Enter each package name and install

Alternatively, run in the first notebook cell:
```python
%pip install sentence-transformers faker neo4j requests tenacity
```

## 2. Configure Neo4j Secrets

### Step 2a: Create environment file

```bash
cp .env.sample .env
```

Edit `.env` with your Neo4j credentials.

### Step 2b: Push secrets to Databricks

```bash
# Ensure Databricks CLI is configured
databricks auth login

# Run the setup script (uses default scope: airline-neo4j-secrets)
./setup_secrets.sh

# Or specify a custom scope name
./setup_secrets.sh my-custom-scope
```

Verify secrets were created:
```bash
databricks secrets list-secrets airline-neo4j-secrets
```

## 3. Upload Files to Databricks

Upload the following files to your Databricks workspace:

| File | Description |
|------|-------------|
| `load_utils.py` | Shared utilities for configuration and Neo4j connections |
| `model_setup.py` | Register embedding model in Unity Catalog |
| `neo4j_connection.py` | Reusable Neo4j connection class |
| `260109 Airline Embedding Load Test.ipynb` | Test Neo4j connection and query embeddings |
| `sample_code_parallel_ingestion-review-redacted (4).ipynb` | Parallel data ingestion example |

## 4. Workflow

### Step 4a: Create Unity Catalog Schema (One-time Setup)

Before registering the embedding model, create the required Unity Catalog schema. Run this SQL in a Databricks notebook or SQL editor:

```sql
CREATE CATALOG IF NOT EXISTS airline_test;
CREATE SCHEMA IF NOT EXISTS airline_test.ml;
```

This creates the `airline_test.ml` schema where the embedding model will be registered.

### Step 4b: Set Up Embedding Model

Run **`model_setup.py`** to register the MiniLM embedding model:

```python
# In a Databricks notebook or as a script
%run ./model_setup

# Or import and run with options
from model_setup import main

# Full setup: log model, validate, and register to Unity Catalog
main()

# Skip endpoint testing (if endpoint not deployed yet)
main(test_endpoint=False)

# Local testing only (skip Unity Catalog registration)
main(register_model=False, test_endpoint=False)
```

The script will:
1. Log the model to MLflow experiment `/Shared/airline-replacement-events-embedding-loadtest`
2. Validate embeddings locally (functionality, semantic similarity, edge cases)
3. Register to Unity Catalog as `airline_test.ml.minilm_l6_v2_embedder`

After running the script, create a Model Serving endpoint:
1. Go to **Serving** → **Create endpoint**
2. Select **Custom model**
3. Choose `airline_test.ml.minilm_l6_v2_embedder` and the registered version
4. Set endpoint name to `minilm-embedder`
5. Choose workload size: **Small**
6. Enable **Scale to zero** for cost optimization
7. Create the endpoint

### Step 4c: Test Neo4j Connection

Run **`260109 Airline Embedding Load Test.ipynb`** to:
1. Verify Neo4j connection using secrets
2. Test the embedding endpoint

### Step 4d: Parallel Data Ingestion (Optional)

Run **`sample_code_parallel_ingestion-review-redacted (4).ipynb`** for:
- Parallel batch ingestion of data into Neo4j
- Example Cypher queries for merging nodes and relationships

## 5. Using Secrets in Notebooks

### Recommended: Use load_utils Module

The `load_utils` module provides a standardized way to load Neo4j configuration from Databricks Secrets:

```python
from load_utils import load_config, test_neo4j_connection, neo4j_driver

SCOPE_NAME = "airline-neo4j-secrets"

# Load configuration from secrets
config = load_config(dbutils, SCOPE_NAME)

# Test connection before running jobs
if not test_neo4j_connection(config):
    raise Exception("Cannot connect to Neo4j!")

# Use the driver context manager
with neo4j_driver(config) as driver:
    with driver.session(database=config.database) as session:
        result = session.run("MATCH (n) RETURN count(n) AS count")
        print(result.single()["count"])
```

### Alternative: Direct Secret Access

For simple scripts, access credentials directly:

```python
SCOPE_NAME = "airline-neo4j-secrets"

NEO4J_HOST = dbutils.secrets.get(scope=SCOPE_NAME, key="host")
NEO4J_USER = dbutils.secrets.get(scope=SCOPE_NAME, key="username")
NEO4J_PASSWORD = dbutils.secrets.get(scope=SCOPE_NAME, key="password")
NEO4J_DB = dbutils.secrets.get(scope=SCOPE_NAME, key="database")
NEO4J_PROTOCOL = dbutils.secrets.get(scope=SCOPE_NAME, key="protocol")

NEO4J_URL = f"{NEO4J_PROTOCOL}://{NEO4J_HOST}"
```

## 6. Using the Neo4j Connection Class

```python
from neo4j_connection import Neo4jConnection

# Create connection
conn = Neo4jConnection(uri=NEO4J_URL, user=NEO4J_USER, pwd=NEO4J_PASSWORD)

# Execute read query
result = conn.read("MATCH (n) RETURN count(n) AS count", db=NEO4J_DB)
print(result)

# Execute write query
conn.write("CREATE (n:Test {name: $name})", parameters={"name": "example"}, db=NEO4J_DB)

# Close connection
conn.close()
```

## 7. Embedding Generation: ai_query vs UDFs

This project uses Databricks' native `ai_query` SQL function for embedding generation rather than Python UDFs. This section explains why.

### Why ai_query is Preferred

The `ai_query` function is a built-in Databricks SQL function that runs AI inference directly within Spark's execution engine. It is the recommended approach for embedding generation because it avoids the overhead associated with User Defined Functions.

### How UDFs Work (and Their Limitations)

When you use a Python UDF (including Pandas UDFs) in Spark:

1. The UDF code is serialized (pickled) and shipped to each worker node
2. Each worker spawns a separate Python process
3. Data must be transferred between the JVM and Python processes
4. Even with Apache Arrow optimization, there is still inter-process communication overhead
5. The UDF runs outside of Spark's query optimizer

This architecture means that no matter how well you optimize a UDF, it will always have overhead compared to native Spark functions.

### How ai_query is Different

The `ai_query` function:

1. Runs entirely within Spark's execution engine (no separate Python process)
2. Has no serialization or data transfer overhead between JVM and Python
3. Handles batching internally and automatically
4. Is optimized by Databricks for their Model Serving infrastructure
5. Scales automatically with Spark's distributed processing

### Performance Comparison

| Aspect | Python UDF | Pandas UDF (Arrow) | ai_query |
|--------|------------|-------------------|----------|
| Serialization | Pickle (slow) | Arrow (fast) | None |
| Data transfer | Row-by-row | Batched | Native |
| Process overhead | High | Medium | None |
| Batching | Manual | Manual | Automatic |
| Query optimization | Excluded | Excluded | Included |

Pandas UDFs with Arrow are 10-100x faster than regular Python UDFs, but native Spark functions like `ai_query` eliminate overhead entirely.

### Implementation Reference

The embedding generation implementation using `ai_query` can be found in:

- **`removals_spark_loader.py`** - The `process_batch()` function generates embeddings for each micro-batch using ai_query

The previous Pandas UDF approach (for reference) was replaced with the simpler ai_query implementation, reducing approximately 100 lines of UDF code to a single expression.

### Requirements

- Databricks Runtime 15.4 LTS or higher (recommended for optimal ai_query performance)
- Access to a Databricks Model Serving endpoint

### Documentation References

- [ai_query Function - Databricks SQL](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query) - Complete function reference
- [AI Functions Overview - Databricks](https://docs.databricks.com/aws/en/large-language-models/ai-functions) - Built-in AI functions for batch inference
- [Pandas User-Defined Functions - Databricks](https://docs.databricks.com/aws/en/udf/pandas) - Understanding Pandas UDFs (for comparison)
- [Apache Arrow in PySpark](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html) - How Arrow optimizes UDF data transfer

### When to Use Pandas UDFs Instead

Pandas UDFs may still be appropriate when:

- The ai_query function does not support your specific model endpoint
- You need custom preprocessing or postprocessing logic within the UDF
- You are using a Databricks Runtime version older than 15.4

In these cases, use the Iterator-style Pandas UDF pattern which initializes resources once per worker rather than per row.

---

## 8. Structured Streaming for Large Dataset Processing

The `removals_spark_loader.py` script uses Databricks Structured Streaming to process large Delta tables in controlled batches. This is the Databricks-recommended approach for processing large datasets without loading everything into memory at once.

### Why Structured Streaming

Traditional batch processing loads all data into memory before processing. For large datasets, this causes memory issues and provides no recovery mechanism if the job fails partway through.

Structured Streaming solves these problems by reading the Delta table as a stream and processing it in micro-batches. Each batch is independently processed (embeddings generated and written to Neo4j) before moving to the next batch.

### How It Works

The pipeline uses the `foreachBatch` pattern with an `availableNow` trigger. This processes all existing data in the Delta table in controlled micro-batches, then automatically stops when complete. It is not a continuous streaming job.

The `maxBytesPerTrigger` option controls batch size by limiting how much data Spark reads per micro-batch. This keeps memory usage predictable regardless of total dataset size.

### Checkpointing and Resume

Structured Streaming automatically tracks progress using checkpoints. If a job fails, restarting it will resume from the last successful batch rather than starting over. Checkpoints are stored at the location specified by `CHECKPOINT_LOCATION`.

To start fresh and reprocess all data, delete the checkpoint directory before running.

### Configuration

Batch size and checkpoint location are controlled by constants at the top of `removals_spark_loader.py`:

- `BATCH_SIZE` controls rows per micro-batch and Neo4j transaction size
- `BYTES_PER_ROW` is used to calculate `maxBytesPerTrigger`
- `WRITE_PARTITIONS` controls parallel writers to Neo4j
- `CHECKPOINT_LOCATION` specifies where streaming progress is stored

### Documentation References

- [Delta Table Streaming Reads and Writes](https://docs.databricks.com/aws/en/structured-streaming/delta-lake) - How to read Delta tables as streams
- [Using foreachBatch](https://docs.databricks.com/aws/en/structured-streaming/foreach) - Processing micro-batches with custom logic
- [Databricks Performance Best Practices](https://docs.databricks.com/aws/en/lakehouse-architecture/performance-efficiency/best-practices) - General optimization guidance

---

## Troubleshooting

### "Secret scope not found"
Run `setup_secrets.sh` to create the scope and secrets.

### "Driver not initialized"
Check that your Neo4j credentials are correct and the host is reachable.

### "Resource does not support direct Data Plane access"
Use the REST API fallback for querying Model Serving endpoints (see notebook examples).

### Missing dependencies
Run at the start of your notebook:
```python
%pip install mlflow sentence-transformers torch transformers pandas numpy faker neo4j requests tenacity
dbutils.library.restartPython()
```
