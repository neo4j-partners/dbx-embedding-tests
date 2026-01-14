"""
Embedding Pipeline with Custom Model Endpoints
===============================================

This script loads data from a Delta table into Neo4j with vector embeddings
using a custom model deployed to Databricks Model Serving.

TUTORIAL: Custom Embedding Models in Databricks
------------------------------------------------

You can deploy your own embedding models (e.g., sentence-transformers) to
Databricks Model Serving. This gives you control over:
- Model architecture and size
- Embedding dimensions
- Fine-tuning on domain-specific data

**Deployment Steps:**
1. Train or select a sentence-transformer model
2. Log the model with MLflow
3. Register in Unity Catalog Model Registry
4. Create a Model Serving endpoint

**API Format (Custom Models):**
Custom models use the "dataframe_records" input format:

    Input:  {"dataframe_records": [{"text": "hello"}, {"text": "world"}]}
    Output: {"predictions": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}

This differs from hosted models which use OpenAI-compatible format.

**Common Model Dimensions:**
- all-MiniLM-L6-v2: 384 dimensions (fast, good quality)
- all-mpnet-base-v2: 768 dimensions (higher quality)
- e5-large-v2: 1024 dimensions (highest quality)

Schema Separation:
------------------
This script uses a SEPARATE Neo4j schema from dbx_embeddings_load_test.py:
- Node label: :RemovalEvent (not :RemovalEventDBX)
- Vector index: removal_reason_embeddings (384 dimensions)
- Constraint: removal_event_removal_id_unique

This allows both scripts to run in parallel without conflicts.

Pipeline Flow:
--------------

    Delta Table
         │
         ▼ (readStream, maxFilesPerTrigger=1)
    ┌─────────────────┐
    │ Micro-batch     │
    └────────┬────────┘
             │
             ▼ (chunk into BATCH_SIZE pieces)
    ┌─────────────────┐
    │ ai_query()      │  Custom MiniLM endpoint
    │ (custom model)  │  384 dimensions
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Neo4j Write     │  :RemovalEvent nodes
    └─────────────────┘

Usage:
    # Process all data
    main()

    # Skip schema setup
    main(setup_schema=False)

    # Resume from checkpoint
    main(clear_checkpoint=False)

Cluster Requirements:
    - Databricks Runtime 15.4+ (for ai_query performance)
    - Neo4j Spark Connector installed
    - Neo4j credentials in Databricks Secrets
    - Access to your Model Serving endpoint

References:
    - Model Serving: https://docs.databricks.com/machine-learning/model-serving/
    - ai_query: https://docs.databricks.com/sql/language-manual/functions/ai_query
    - Neo4j Spark Connector: https://neo4j.com/docs/spark/current/
"""

from __future__ import annotations

import time
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

# =============================================================================
# IMPORTS FROM MODULAR COMPONENTS
# =============================================================================

from load_utils import (
    Config,
    load_config,
    neo4j_driver,
    print_config,
    print_section_header,
    test_neo4j_connection,
    format_duration,
)

from neo4j_schema import (
    SchemaConfig,
    setup_neo4j_schema,
    wait_for_vector_index,
)

from embedding_providers import (
    EmbeddingConfig,
    CustomModelEmbeddingProvider,
    generate_query_embedding,
)

from streaming_pipeline import (
    PipelineConfig,
    run_pipeline,
    print_pipeline_summary,
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Databricks secret scope containing Neo4j credentials
SCOPE_NAME = "rk-airline-neo4j-secrets"

# Default values
DEFAULT_DATABASE = "neo4j"
DEFAULT_PROTOCOL = "neo4j+s"  # TLS for Aura

# Custom embedding model endpoint
# This should be your deployed sentence-transformer model
DEFAULT_EMBEDDING_ENDPOINT = "rk_serving_airline_embedding"

# Source table in Unity Catalog
SOURCE_TABLE = "airline_test.airline_test_lakehouse.nodes_removals_large"
TEXT_COLUMN = "RMV_REA_TX"
ID_COLUMN = ":ID(RemovalEvent)"

# Embedding configuration (MiniLM uses 384 dimensions)
EMBEDDING_DIMENSIONS = 384

# Neo4j schema (separate from DBX hosted model to avoid conflicts)
NODE_LABEL = "RemovalEvent"
CONSTRAINT_NAME = "removal_event_removal_id_unique"
VECTOR_INDEX_NAME = "removal_reason_embeddings"

# Batch processing configuration
BATCH_SIZE = 5000
WRITE_PARTITIONS = 4  # Higher parallelism for custom models (often faster)
CHECKPOINT_LOCATION = "/tmp/removal_embeddings_checkpoint"

# Row limit: Set to a positive number to limit rows, or -1 for all rows
# Use positive values for testing (faster), -1 for production (all data)
MAX_ROWS = 500  # Set to 500 for quick testing, -1 for all rows


# =============================================================================
# COLUMN SELECTOR
# =============================================================================

def select_columns(df: DataFrame) -> DataFrame:
    """Select and rename columns from the source table.

    Tutorial: Column Mapping for Embedding Pipeline
    -----------------------------------------------

    This function maps source columns to the schema expected by:
    1. The embedding provider (needs text_column)
    2. Neo4j (needs id column and property names)

    Source Column          → Target Column
    :ID(RemovalEvent)     → removal_id (unique key for MERGE)
    RMV_REA_TX            → removal_reason (text for embedding)
    ...                   → ... (additional properties)

    Args:
        df: Source DataFrame from Delta table

    Returns:
        DataFrame ready for embedding and Neo4j write
    """
    return df.select(
        col(f"`{ID_COLUMN}`").alias("removal_id"),
        col(TEXT_COLUMN).alias("removal_reason"),
        col("RMV_TRK_NO").alias("rmv_trk_no"),
        col("component_id"),
        col("aircraft_id"),
        col("removal_date"),
    )


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_embeddings(spark: SparkSession, config: Config) -> bool:
    """Verify embeddings were stored correctly using Spark Connector.

    Tutorial: Post-Load Verification
    ---------------------------------

    Always verify after loading embeddings:

    1. **Count check**: All expected nodes exist
    2. **Embedding check**: All nodes have embeddings
    3. **Dimension check**: Embeddings have correct size
    4. **Sample review**: Spot-check a few embeddings

    Why use Spark Connector for reading?
    - Consistent with how data was written
    - Distributed processing for large datasets
    - Full DataFrame operations available

    Args:
        spark: SparkSession
        config: Neo4j configuration

    Returns:
        True if all verifications pass
    """
    print_section_header("VERIFYING EMBEDDINGS")

    # Read nodes from Neo4j
    df = (
        spark.read.format("org.neo4j.spark.DataSource")
        .option("url", config.uri)
        .option("authentication.basic.username", config.username)
        .option("authentication.basic.password", config.password)
        .option("database", config.database)
        .option("labels", NODE_LABEL)
        .load()
    )

    total_count = df.count()
    print(f"Total {NODE_LABEL} nodes: {total_count:,}")

    if "embedding" not in df.columns:
        print("Warning: 'embedding' column not found!")
        return False

    # Check embeddings
    with_embeddings = df.filter(col("embedding").isNotNull())
    embedding_count = with_embeddings.count()
    print(f"Nodes with embeddings: {embedding_count:,}")

    # Sample verification
    print("\nSample nodes:")
    sample = with_embeddings.limit(5).collect()

    all_valid = True
    for i, row in enumerate(sample):
        emb = row["embedding"]
        if emb is None:
            print(f"  [{i+1}] {row['removal_id']}: No embedding")
            all_valid = False
        elif len(emb) != EMBEDDING_DIMENSIONS:
            print(f"  [{i+1}] {row['removal_id']}: Wrong dimensions ({len(emb)})")
            all_valid = False
        else:
            preview = [f"{v:.4f}" for v in emb[:3]]
            try:
                reason = (row["removal_reason"] or "")[:40]
            except Exception:
                reason = ""
            print(f"  [{i+1}] {row['removal_id']}: {len(emb)} dims")
            print(f"       Embedding: [{', '.join(preview)}, ...]")
            print(f"       Reason: {reason}...")

    return all_valid and embedding_count == total_count


def test_vector_search(config: Config, test_text: str = "hydraulic pump failure") -> None:
    """Test vector similarity search with the loaded embeddings.

    Tutorial: Vector Search with Custom Embeddings
    -----------------------------------------------

    To search for similar items:

    1. **Generate query embedding**: Use the SAME custom model
    2. **Query the vector index**: Neo4j's db.index.vector.queryNodes()
    3. **Review results**: Higher score = more similar

    Important: The query embedding MUST be from the same model as the
    stored embeddings. Mixing models produces meaningless results.

    Cypher for vector search:

        CALL db.index.vector.queryNodes(
            'removal_reason_embeddings',  // Index name
            5,                            // Top-k results
            $embedding                    // Query vector
        ) YIELD node, score
        RETURN node.removal_id, score

    Args:
        config: Neo4j configuration
        test_text: Query text for similarity search
    """
    print_section_header("TESTING VECTOR SEARCH")

    print(f"Query: '{test_text}'")

    # Generate embedding using custom model
    print("\nGenerating query embedding...")
    test_embedding = generate_query_embedding(
        DEFAULT_EMBEDDING_ENDPOINT,
        test_text,
        api_format="custom",  # Use custom model format
    )
    print(f"  Dimensions: {len(test_embedding)}")

    # Query Neo4j
    query = f"""
        CALL db.index.vector.queryNodes(
            '{VECTOR_INDEX_NAME}',
            5,
            $embedding
        ) YIELD node, score
        RETURN node.removal_id AS removal_id,
               node.removal_reason AS reason,
               score
    """

    print("\nTop 5 similar items:")
    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            result = session.run(query, embedding=test_embedding)
            records = list(result)

            if not records:
                print("  No results found. Is the vector index populated?")
            else:
                for i, record in enumerate(records):
                    reason = record["reason"] or ""
                    print(f"  [{i+1}] Score: {record['score']:.4f}")
                    print(f"      ID: {record['removal_id']}")
                    print(f"      Reason: {reason}")
                    print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(
    setup_schema: bool = True,
    test_search: bool = True,
    checkpoint_location: Optional[str] = None,
    clear_checkpoint: bool = True,
) -> dict:
    """Run the embedding pipeline with custom model endpoints.

    Tutorial: Complete Custom Model Pipeline
    ----------------------------------------

    This function runs the complete embedding pipeline using your custom
    sentence-transformer model:

    1. **Configuration**: Load Neo4j credentials from Databricks Secrets
    2. **Connection Test**: Verify Neo4j is accessible
    3. **Endpoint Validation**: Test your custom model returns correct format
    4. **Schema Setup**: Create constraint and vector index (384 dims)
    5. **Pipeline Run**: Stream data, generate embeddings, write to Neo4j
    6. **Verification**: Confirm embeddings were stored correctly
    7. **Search Test**: Verify vector similarity search works

    Custom Model Considerations:
    ----------------------------

    1. **Endpoint Name**: Update DEFAULT_EMBEDDING_ENDPOINT to match your
       deployed model's endpoint name.

    2. **Dimensions**: Update EMBEDDING_DIMENSIONS to match your model:
       - MiniLM: 384
       - MPNet: 768
       - E5-Large: 1024

    3. **Write Parallelism**: Custom endpoints often handle concurrent
       requests well. WRITE_PARTITIONS=4 is a good starting point.

    Args:
        setup_schema: Create Neo4j schema elements
        test_search: Run similarity search test after loading
        checkpoint_location: Custom checkpoint path
        clear_checkpoint: Delete checkpoint for fresh start

    Returns:
        Dictionary with pipeline statistics
    """
    if checkpoint_location is None:
        checkpoint_location = CHECKPOINT_LOCATION

    pipeline_start = time.time()

    # =========================================================================
    # STEP 1: Print header and configuration
    # =========================================================================
    print_section_header("EMBEDDING PIPELINE (CUSTOM MODEL)")
    print(f"Embedding Model: {DEFAULT_EMBEDDING_ENDPOINT}")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Neo4j Label: :{NODE_LABEL}")
    print(f"Vector Index: {VECTOR_INDEX_NAME}")

    # Load configuration
    config = load_config(
        dbutils,
        SCOPE_NAME,
        DEFAULT_DATABASE,
        DEFAULT_PROTOCOL,
        DEFAULT_EMBEDDING_ENDPOINT,
    )
    config.embedding_endpoint = DEFAULT_EMBEDDING_ENDPOINT
    print_config(config, SCOPE_NAME, EMBEDDING_DIMENSIONS, BATCH_SIZE)

    # =========================================================================
    # STEP 2: Test Neo4j connection
    # =========================================================================
    if not test_neo4j_connection(config):
        print("\nAborting: Neo4j connection failed.")
        return {}

    # =========================================================================
    # STEP 3: Validate embedding endpoint
    # =========================================================================
    embedding_config = EmbeddingConfig(
        endpoint_name=DEFAULT_EMBEDDING_ENDPOINT,
        dimensions=EMBEDDING_DIMENSIONS,
        text_column="removal_reason",
        output_column="embedding",
    )
    embedding_provider = CustomModelEmbeddingProvider(embedding_config)

    print_section_header("VALIDATING EMBEDDING ENDPOINT")
    if not embedding_provider.validate_endpoint():
        print("\nAborting: Embedding endpoint validation failed.")
        print("Check that your Model Serving endpoint is running and")
        print(f"EMBEDDING_DIMENSIONS ({EMBEDDING_DIMENSIONS}) matches your model.")
        return {}

    # =========================================================================
    # STEP 4: Setup Neo4j schema
    # =========================================================================
    if setup_schema:
        schema_config = SchemaConfig(
            node_label=NODE_LABEL,
            id_property="removal_id",
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            constraint_name=CONSTRAINT_NAME,
            vector_index_name=VECTOR_INDEX_NAME,
        )
        setup_neo4j_schema(config, schema_config)
        wait_for_vector_index(config, VECTOR_INDEX_NAME)

    # =========================================================================
    # STEP 5: Run pipeline (streaming with optional row limit)
    # =========================================================================
    pipeline_config = PipelineConfig(
        source_table=SOURCE_TABLE,
        node_label=NODE_LABEL,
        id_column="removal_id",
        batch_size=BATCH_SIZE,
        write_partitions=WRITE_PARTITIONS,
        checkpoint_location=checkpoint_location,
        max_files_per_trigger=1,
        max_rows=MAX_ROWS,
    )

    # Validate max_rows was set correctly
    print(f"PipelineConfig created: max_rows={pipeline_config.max_rows}")
    if MAX_ROWS > 0 and pipeline_config.max_rows != MAX_ROWS:
        raise ValueError(f"max_rows mismatch: expected {MAX_ROWS}, got {pipeline_config.max_rows}")

    stats = run_pipeline(
        spark=spark,
        neo4j_config=config,
        pipeline_config=pipeline_config,
        embedding_provider=embedding_provider,
        column_selector=select_columns,
        clear_checkpoint=clear_checkpoint,
        dbutils=dbutils,
    )

    # =========================================================================
    # STEP 6: Verify embeddings
    # =========================================================================
    verification_passed = verify_embeddings(spark, config)

    # =========================================================================
    # STEP 7: Test similarity search
    # =========================================================================
    if test_search:
        test_vector_search(config)

    # =========================================================================
    # STEP 8: Print summary
    # =========================================================================
    print_pipeline_summary(stats, pipeline_config, embedding_provider)

    total_time = time.time() - pipeline_start
    print(f"\nVerification: {'PASSED' if verification_passed else 'FAILED'}")
    print(f"Total time: {format_duration(total_time)}")
    print("\nDone!")

    return stats


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
