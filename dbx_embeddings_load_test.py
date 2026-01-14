"""
Embedding Pipeline with Databricks Hosted Models (Foundation Model APIs)
=========================================================================

This script loads data from a Delta table into Neo4j with vector embeddings
using Databricks Foundation Model APIs (hosted models like BGE and GTE).

TUTORIAL: Databricks Foundation Model APIs
------------------------------------------

Databricks provides pre-deployed embedding models as part of the Foundation
Model APIs. These are ready to use without deployment:

**Available Models:**
- `databricks-bge-large-en`: 1024 dimensions, 512 token context
- `databricks-gte-large-en`: 1024 dimensions, 8192 token context

**Advantages of Hosted Models:**
- No deployment needed (ready to use immediately)
- Managed scaling and availability
- Pay-per-token pricing
- OpenAI-compatible API format

**API Format (OpenAI-compatible):**
    Input:  {"input": ["text1", "text2"]}
    Output: {"data": [{"embedding": [0.1, ...]}, {"embedding": [0.3, ...]}]}

**Using ai_query() with Hosted Models:**
The ai_query() SQL function works seamlessly:

    SELECT ai_query('databricks-bge-large-en', text_column) AS embedding
    FROM my_table

Databricks handles the API format conversion internally.

Schema Separation:
------------------
This script uses a SEPARATE Neo4j schema from custom_embeddings_load_test.py:
- Node label: :RemovalEventDBX
- Vector index: removal_reason_embeddings_dbx (1024 dimensions)
- Constraint: removal_event_dbx_removal_id_unique

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
    │ ai_query()      │  databricks-bge-large-en
    │ (hosted model)  │  1024 dimensions
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Neo4j Write     │  :RemovalEventDBX nodes
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
    - Access to Databricks Foundation Model APIs

References:
    - Foundation Model APIs: https://docs.databricks.com/machine-learning/foundation-model-apis/
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
    DatabricksHostedEmbeddingProvider,
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
SCOPE_NAME = "airline-neo4j-secrets"

# Default values
DEFAULT_DATABASE = "neo4j"
DEFAULT_PROTOCOL = "neo4j+s"  # TLS for Aura

# Databricks hosted embedding model
# Options:
#   databricks-bge-large-en: 1024 dims, 512 token context, normalized
#   databricks-gte-large-en: 1024 dims, 8192 token context
DEFAULT_EMBEDDING_ENDPOINT = "databricks-bge-large-en"

# Source table in Unity Catalog
SOURCE_TABLE = "airline_test.airline_test_lakehouse.nodes_removals_large"
TEXT_COLUMN = "RMV_REA_TX"
ID_COLUMN = ":ID(RemovalEvent)"

# Embedding configuration (BGE/GTE use 1024 dimensions)
EMBEDDING_DIMENSIONS = 1024

# Neo4j schema (separate from custom model to avoid conflicts)
NODE_LABEL = "RemovalEventDBX"
CONSTRAINT_NAME = "removal_event_dbx_removal_id_unique"
VECTOR_INDEX_NAME = "removal_reason_embeddings_dbx"

# Batch processing configuration
BATCH_SIZE = 5000
WRITE_PARTITIONS = 1
CHECKPOINT_LOCATION = "/tmp/removal_embeddings_hosted_checkpoint"

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

    This function prepares the source data for embedding:

    1. **removal_id**: Unique identifier for MERGE operations
    2. **removal_reason**: Text to embed (mapped to embedding provider's text_column)
    3. **Other columns**: Additional properties to store in Neo4j

    The text column name (removal_reason) must match what you configure
    in the EmbeddingConfig.text_column setting.

    Args:
        df: Source DataFrame

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

    Tutorial: Verification Best Practices
    -------------------------------------

    After writing embeddings, verify:

    1. **Node count**: Expected number of nodes created
    2. **Embedding presence**: All nodes have embedding property
    3. **Dimension check**: Embeddings have correct dimensions
    4. **Sample inspection**: A few embeddings look reasonable

    Using Spark Connector for verification:
    - Faster than Neo4j driver for large reads
    - Consistent with how data was written
    - Can leverage Spark's distributed processing

    Args:
        spark: SparkSession
        config: Neo4j configuration

    Returns:
        True if verification passes
    """
    print_section_header("VERIFYING EMBEDDINGS")

    # Read nodes back from Neo4j
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

    # Check embedding presence
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

    Tutorial: Vector Similarity Search in Neo4j
    -------------------------------------------

    After loading embeddings, test that similarity search works:

    1. **Generate query embedding**: Use the SAME model as stored embeddings
    2. **Query the vector index**: Use db.index.vector.queryNodes()
    3. **Verify results**: Check that similar items are returned

    The query uses:

        CALL db.index.vector.queryNodes(
            'index_name',  -- Name of the vector index
            5,             -- Number of results (top-k)
            $embedding     -- Query vector (same dimensions!)
        ) YIELD node, score

    The score is the similarity (higher = more similar for cosine).

    Args:
        config: Neo4j configuration
        test_text: Text to search for similar items
    """
    print_section_header("TESTING VECTOR SEARCH")

    print(f"Query: '{test_text}'")

    # Generate embedding for test text using hosted model
    print("\nGenerating query embedding...")
    test_embedding = generate_query_embedding(
        DEFAULT_EMBEDDING_ENDPOINT,
        test_text,
        api_format="hosted",
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
    """Run the embedding pipeline with Databricks hosted models.

    Tutorial: Complete Pipeline Execution
    -------------------------------------

    This function runs the complete embedding pipeline:

    1. **Configuration**: Load Neo4j credentials from Databricks Secrets
    2. **Connection Test**: Verify Neo4j is accessible
    3. **Endpoint Validation**: Test the embedding model returns correct format
    4. **Schema Setup**: Create constraint and vector index
    5. **Pipeline Run**: Stream data, generate embeddings, write to Neo4j
    6. **Verification**: Confirm embeddings were stored correctly
    7. **Search Test**: Verify vector similarity search works

    Pipeline Parameters:
    --------------------

    **setup_schema** (default: True)
        Create Neo4j constraint and vector index. Set to False if already created.

    **test_search** (default: True)
        Run a sample similarity search after loading. Helps verify everything works.

    **clear_checkpoint** (default: True)
        Delete checkpoint to process all data from scratch.
        Set to False to resume from last successful batch.

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
    print_section_header("EMBEDDING PIPELINE (DATABRICKS HOSTED MODEL)")
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
    embedding_provider = DatabricksHostedEmbeddingProvider(embedding_config)

    print_section_header("VALIDATING EMBEDDING ENDPOINT")
    if not embedding_provider.validate_endpoint():
        print("\nAborting: Embedding endpoint validation failed.")
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
