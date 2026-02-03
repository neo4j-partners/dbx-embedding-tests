# Embedding Load Test - Quick Start Guide

This project generates vector embeddings in Databricks and loads them into Neo4j.

## Project Overview

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `neo4j_load_test_nb.ipynb` | Test Neo4j connectivity and write performance using random embeddings |
| `custom_embeddings_load_test_nb.ipynb` | Test with your custom MiniLM model deployed to Model Serving |
| `dbx_embeddings_load_test_nb.ipynb` | Test with Databricks hosted foundation models (BGE, GTE) |

### Python Scripts

**Load Tests** - Standalone versions of the notebooks:
| Script | Purpose |
|--------|---------|
| `neo4j_load_test.py` | Baseline Neo4j write performance with random 384-dim embeddings |
| `custom_embeddings_load_test.py` | End-to-end test with custom model endpoint |
| `dbx_embeddings_load_test.py` | End-to-end test with Databricks hosted models |

**Utility Modules** - Shared functionality:
| Module | Purpose |
|--------|---------|
| `embedding_providers.py` | Abstraction for embedding generation (random, custom model, hosted model) |
| `streaming_pipeline.py` | Structured Streaming pipeline with `foreachBatch` for Neo4j writes |
| `neo4j_connection.py` | Neo4j driver wrapper with session/transaction management |
| `neo4j_schema.py` | Schema setup: uniqueness constraints and vector indexes |
| `load_utils.py` | Configuration, secrets loading, connection testing utilities |

**Setup**:
| Script | Purpose |
|--------|---------|
| `model_setup.py` | Register MiniLM embedding model in Unity Catalog |

## Quick Start

1. **Configure Neo4j secrets** - Copy `.env.sample` to `.env`, add your Neo4j credentials, then run:
   ```bash
   databricks auth login
   ./setup_secrets.sh
   ```

2. **Upload files to Databricks** - Upload all `.py` and `.ipynb` files to your Databricks workspace.

3. **Run `model_setup.py`** - Execute this script first in Databricks to register the MiniLM embedding model in Unity Catalog. Then create a Model Serving endpoint named `minilm-embedder` via the Databricks UI.

4. **Run the notebooks** - Start with `neo4j_load_test_nb.ipynb` to verify Neo4j connectivity, then proceed to the embedding notebooks.

---

## How Embeddings Work in This Pipeline

The pipeline uses Databricks' `ai_query()` function to generate embeddings directly in Spark SQL. This calls your Model Serving endpoint and returns an array of floats that stores directly in Neo4j:

```python
df.withColumn("embedding", expr("ai_query('my-endpoint', text_column)"))
```

The embedding column is an `ARRAY<FLOAT>` which the Neo4j Spark Connector writes as a `List<Float>` property—the exact format Neo4j vector indexes require. Invalid embeddings (null or wrong dimensions) are filtered before writing:

```python
df.filter(col("embedding").isNotNull() & (size(col("embedding")) == 384))
```

Then simply write to Neo4j using the Spark Connector—no additional transformation needed:

```python
df.write.format("org.neo4j.spark.DataSource").option("labels", ":MyNode").save()
```

## Prerequisites

- Databricks workspace with Unity Catalog
- Neo4j database (Aura or self-hosted)
- Databricks CLI installed and configured

## 1. Create Databricks Compute Cluster

1. Go to **Compute** → **Create Compute**
2. Select a **ML** runtime (e.g., `15.4 LTS ML`)
3. Install additional packages: `sentence-transformers`, `neo4j`, `requests`, `tenacity`

## 2. Configure Neo4j Secrets

```bash
cp .env.sample .env
# Edit .env with your Neo4j credentials

databricks auth login
./setup_secrets.sh
```

## 3. Upload Files to Databricks

Upload all Python files (`*.py`) and notebooks (`*.ipynb`) from this project to your Databricks workspace.

## 4. Create Unity Catalog Schema

Before registering the embedding model, create the required schema:

```sql
CREATE CATALOG IF NOT EXISTS airline_test;
CREATE SCHEMA IF NOT EXISTS airline_test.ml;
```

## 5. Set Up Embedding Model

Run `model_setup.py` to register the MiniLM embedding model in Unity Catalog, then create a Model Serving endpoint named `minilm-embedder` via the Databricks UI.

## 6. Test Neo4j Connectivity (Random Embeddings)

Before testing with real embeddings, verify that your Neo4j connection works and establish a baseline for write performance. This test generates random float arrays instead of calling an embedding model, which lets you measure pure Neo4j throughput without API latency.

Run `neo4j_load_test.py` or `neo4j_load_test_nb.ipynb`. The script reads from your Delta table, generates fake 384-dimensional embeddings, and writes nodes to Neo4j. If this fails, check your Neo4j credentials and network connectivity before proceeding.

```python
from neo4j_load_test import main
main()  # Uses random embeddings, no model calls
```

**Typical results:** 2,000-5,000 rows/second. This is your ceiling—real embeddings will be slower due to model inference time.

## 7. Test Custom Model Embeddings

Now test with your deployed MiniLM embedding model. This validates that your Model Serving endpoint is working correctly and measures end-to-end throughput including embedding generation.

Run `custom_embeddings_load_test.py` or `custom_embeddings_load_test_nb.ipynb`. The script calls your custom endpoint (e.g., `minilm-embedder`) to generate 384-dimensional embeddings for each row, then writes the results to Neo4j with a `:RemovalEvent` label. The difference between this throughput and Step 6 shows how much time embedding generation adds.

```python
from custom_embeddings_load_test import main
main()  # Calls your custom model endpoint
```

**Typical results:** 200-500 rows/second. The gap from Step 6 is your embedding model overhead.

## 8. Test Databricks Hosted Embeddings

Alternatively, use Databricks Foundation Model APIs instead of deploying your own model. These pre-hosted models (`databricks-bge-large-en`, `databricks-gte-large-en`) require no deployment and produce 1024-dimensional embeddings.

Run `dbx_embeddings_load_test.py` or `dbx_embeddings_load_test_nb.ipynb`. This creates nodes with a `:RemovalEventDBX` label to keep results separate from the custom model test. Hosted models are convenient but produce larger embeddings (1024 vs 384 dimensions), which affects storage and search performance.

```python
from dbx_embeddings_load_test import main
main()  # Uses Databricks hosted BGE model
```

---

## Embedding Generation with ai_query (Reference)

This project uses Databricks' native `ai_query` SQL function for embedding generation rather than Python UDFs.

### Why ai_query is Preferred

The `ai_query` function is a built-in Spark SQL function that runs AI inference directly within Spark's execution engine. It avoids the serialization overhead of Python UDFs:

| Aspect | Python UDF | Pandas UDF (Arrow) | ai_query |
|--------|------------|-------------------|----------|
| Serialization | Pickle (slow) | Arrow (fast) | None |
| Data transfer | Row-by-row | Batched | Native |
| Process overhead | High | Medium | None |
| Batching | Manual | Manual | Automatic |

### How It Works

1. `ai_query()` calls the Model Serving endpoint for each row
2. Databricks batches requests automatically for efficiency
3. Returns `ARRAY<FLOAT>` directly—no parsing needed
4. Failed requests return NULL (filter these before writing)

### Implementation

The embedding providers in `embedding_providers.py` wrap ai_query:

```python
ai_query_expr = f"ai_query('{endpoint_name}', {text_column})"
df.withColumn("embedding", expr(ai_query_expr))
```

### API Format Differences

**Custom models** (your deployed MiniLM):
```
Input:  {"dataframe_records": [{"text": "..."}]}
Output: {"predictions": [[0.1, 0.2, ...]]}
```

**Databricks hosted models** (BGE, GTE):
```
Input:  {"input": ["..."]}
Output: {"data": [{"embedding": [0.1, ...]}]}
```

The `ai_query()` function handles both formats transparently.

### Documentation

- [ai_query Function](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query)
- [AI Functions Overview](https://docs.databricks.com/aws/en/large-language-models/ai-functions)

---

## Structured Streaming for Large Datasets

The pipeline uses Structured Streaming with `foreachBatch` to process large Delta tables in controlled micro-batches:

1. **Memory control**: Only one batch in memory at a time
2. **Checkpointing**: Resume from failure without reprocessing
3. **Backpressure**: Automatic rate limiting

### Configuration

Key settings in each load test script:

- `BATCH_SIZE`: Rows per Neo4j transaction (default 5000)
- `WRITE_PARTITIONS`: Parallel writers (1 = serial, 2-4 for more throughput)
- `MAX_ROWS`: Limit rows for testing (-1 for all data)
- `CHECKPOINT_LOCATION`: Path for streaming progress

To reprocess all data, delete the checkpoint directory or set `clear_checkpoint=True`.

### Documentation

- [Delta Table Streaming](https://docs.databricks.com/aws/en/structured-streaming/delta-lake)
- [Using foreachBatch](https://docs.databricks.com/aws/en/structured-streaming/foreach)

---

## Troubleshooting

### "Secret scope not found"
Run `./setup_secrets.sh` to create the scope and secrets.

### "Schema does not exist"
Create the Unity Catalog schema:
```sql
CREATE CATALOG IF NOT EXISTS airline_test;
CREATE SCHEMA IF NOT EXISTS airline_test.ml;
```

### Embedding dimensions mismatch
Ensure `EMBEDDING_DIMENSIONS` in your script matches your model:
- MiniLM: 384
- BGE/GTE: 1024

### Endpoint not responding
Wait 3-5 minutes after creating a new serving endpoint. Check status in Serving UI.

### Missing dependencies
```python
%pip install mlflow sentence-transformers torch pandas numpy neo4j requests tenacity
dbutils.library.restartPython()
```
