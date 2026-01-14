"""
Neo4j Write Performance Test with Random Embeddings
====================================================

This script tests Neo4j write performance in isolation by using random embeddings
instead of calling an AI model. This establishes a baseline for Neo4j throughput
before adding embedding generation overhead.

TUTORIAL: Why Test with Random Embeddings?
------------------------------------------

When building an embedding pipeline, you have two performance bottlenecks:

1. **Embedding Generation**: Calling AI models (ai_query, MLflow, etc.)
2. **Neo4j Writes**: Writing nodes with embeddings to the graph

To understand and optimize each bottleneck independently, this script:
- Generates random float arrays instead of real embeddings
- Tests only the Neo4j write path
- Measures throughput without API call latency

This helps you:
- Set realistic expectations for write performance
- Tune batch size and parallelism settings
- Identify if Neo4j or embedding generation is the bottleneck

Typical Results:
- Random embeddings: 2,000-5,000 rows/second
- With ai_query: 100-500 rows/second (depending on model)

The difference shows how much the embedding model affects throughput.

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
    │ Random floats   │  384 random values per row
    │ as "embeddings" │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Neo4j Write     │  :RemovalEventTest nodes
    └─────────────────┘

Usage:
    # Run with defaults (fresh start)
    main()

    # Custom batch size
    main(batch_size=1000)

    # Resume previous run
    main(clear_checkpoint=False, cleanup_nodes=False)

Cluster Requirements:
    - Databricks Runtime 13.x+
    - Neo4j Spark Connector installed
    - Neo4j credentials in Databricks Secrets

References:
    - Neo4j Spark Connector: https://neo4j.com/docs/spark/current/
    - Structured Streaming: https://docs.databricks.com/structured-streaming/
"""

from __future__ import annotations

import time
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

# =============================================================================
# IMPORTS FROM MODULAR COMPONENTS
# =============================================================================
# These modules contain the reusable components that power this pipeline.
# See each module's docstring for detailed tutorials.

from load_utils import (
    Config,
    load_config,
    neo4j_driver,
    print_config,
    print_section_header,
    test_neo4j_connection,
    format_duration,
    format_rate,
)

from neo4j_schema import (
    SchemaConfig,
    setup_neo4j_schema,
    delete_nodes_by_label,
)

from embedding_providers import (
    EmbeddingConfig,
    RandomEmbeddingProvider,
)

from streaming_pipeline import (
    PipelineConfig,
    run_pipeline,
    print_pipeline_summary,
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# These constants define the specific settings for this test.
# Modify these to adapt the test to your environment.

# Databricks secret scope containing Neo4j credentials
# Tutorial: Create this scope with: databricks secrets create-scope my-scope
SCOPE_NAME = "airline-neo4j-secrets"

# Default values when secrets are not found
DEFAULT_DATABASE = "neo4j"
DEFAULT_PROTOCOL = "neo4j+s"  # TLS-encrypted for Aura
DEFAULT_EMBEDDING_ENDPOINT = "unused"  # Not used for random embeddings

# Source table in Unity Catalog
# Tutorial: This should be a Delta table with your text data
SOURCE_TABLE = "airline_test.airline_test_lakehouse.nodes_removals_large"
TEXT_COLUMN = "RMV_REA_TX"  # Column containing text to "embed"
ID_COLUMN = ":ID(RemovalEvent)"  # Column with unique identifier

# Embedding configuration
# Tutorial: Even random embeddings need dimensions to match what production would use
EMBEDDING_DIMENSIONS = 384  # MiniLM dimensions (change if using different model)

# Batch processing configuration
# Tutorial: Tune these for your Neo4j cluster capacity
BATCH_SIZE = 5000  # Rows per Neo4j transaction
WRITE_PARTITIONS = 1  # Parallel writers (1 = serial, increase for more throughput)
CHECKPOINT_LOCATION = "/tmp/neo4j_write_test_checkpoint"

# Row limit: Set to a positive number to limit rows, or -1 for all rows
# Use positive values for testing (faster), -1 for production (all data)
MAX_ROWS = 500  # Set to 500 for quick testing, -1 for all rows

# Test label (different from production to avoid conflicts)
# Tutorial: Using a separate label lets you run tests without affecting production data
TEST_LABEL = "RemovalEventTest"


# =============================================================================
# COLUMN SELECTOR
# =============================================================================

def select_columns(df: DataFrame) -> DataFrame:
    """Select and rename columns from the source table.

    Tutorial: Column Mapping
    ------------------------

    Your source Delta table may have different column names than what
    Neo4j expects. This function handles the mapping:

        Source Column          → Target Column (for Neo4j)
        :ID(RemovalEvent)     → removal_id
        RMV_REA_TX            → removal_reason (text for embeddings)
        RMV_TRK_NO            → rmv_trk_no
        ...

    The target column names should:
    - Match what your Neo4j schema expects
    - Be valid Neo4j property names (no special chars)
    - Include the text column for embeddings

    Args:
        df: Source DataFrame from Delta table

    Returns:
        DataFrame with renamed columns ready for processing
    """
    return df.select(
        col(f"`{ID_COLUMN}`").alias("removal_id"),  # Backticks for special chars
        col(TEXT_COLUMN).alias("removal_reason"),
        col("RMV_TRK_NO").alias("rmv_trk_no"),
        col("component_id"),
        col("aircraft_id"),
        col("removal_date"),
    )


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_test_nodes(config: Config) -> int:
    """Verify test nodes were created correctly in Neo4j.

    Tutorial: Post-Write Verification
    ---------------------------------

    After a write job, always verify the data landed correctly:

    1. **Count nodes**: Do we have the expected number?
    2. **Check embeddings**: Are embedding properties present?
    3. **Sample data**: Do a few records look correct?

    This catches issues like:
    - Partial writes (some batches failed)
    - Missing embeddings (filtered out as invalid)
    - Wrong dimensions (model mismatch)

    Args:
        config: Neo4j configuration

    Returns:
        Total number of test nodes in Neo4j

    Example:
        >>> node_count = verify_test_nodes(config)
        >>> if node_count < expected:
        ...     print("Warning: Some rows may have failed to write")
    """
    print_section_header("VERIFYING TEST NODES")

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            # Count total nodes
            result = session.run(f"""
                MATCH (n:{TEST_LABEL})
                RETURN count(n) AS count
            """)
            total_count = result.single()["count"]
            print(f"  Total {TEST_LABEL} nodes: {total_count:,}")

            # Count nodes with embeddings
            result = session.run(f"""
                MATCH (n:{TEST_LABEL})
                WHERE n.embedding IS NOT NULL
                RETURN count(n) AS count
            """)
            embedding_count = result.single()["count"]
            print(f"  Nodes with embeddings: {embedding_count:,}")

            # Sample verification - check embedding dimensions
            result = session.run(f"""
                MATCH (n:{TEST_LABEL})
                WHERE n.embedding IS NOT NULL
                RETURN n.removal_id AS id, size(n.embedding) AS dims
                LIMIT 3
            """)
            records = list(result)
            if records:
                print("\n  Sample nodes:")
                for r in records:
                    print(f"    ID: {r['id']}, embedding dims: {r['dims']}")

    return total_count


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(
    batch_size: int = BATCH_SIZE,
    checkpoint_location: Optional[str] = None,
    clear_checkpoint: bool = True,
    cleanup_nodes: bool = True,
    setup_schema: bool = True,
) -> dict:
    """Run the Neo4j write performance test with random embeddings.

    Tutorial: Running the Test
    --------------------------

    This function orchestrates the complete test:

    1. **Load Configuration**: Read Neo4j credentials from Databricks Secrets
    2. **Test Connection**: Verify Neo4j is accessible before starting
    3. **Setup Schema**: Create uniqueness constraint (no vector index needed for testing)
    4. **Cleanup**: Delete any existing test nodes (optional)
    5. **Run Pipeline**: Stream data, generate random embeddings, write to Neo4j
    6. **Verify**: Check that nodes were created correctly
    7. **Report**: Print summary statistics

    Common Use Cases:
    -----------------

    **Fresh Test Run** (default):
        >>> main()
        # Clears checkpoint and test nodes, processes all data

    **Resume After Failure**:
        >>> main(clear_checkpoint=False, cleanup_nodes=False)
        # Continues from last checkpoint, keeps existing nodes

    **Tune Batch Size**:
        >>> main(batch_size=1000)  # Smaller batches, less memory
        >>> main(batch_size=10000)  # Larger batches, potentially faster

    **Skip Schema Setup** (if already created):
        >>> main(setup_schema=False)

    Args:
        batch_size: Rows per processing chunk (default 5000)
        checkpoint_location: Custom checkpoint path (optional)
        clear_checkpoint: Delete checkpoint for fresh start (default True)
        cleanup_nodes: Delete existing test nodes (default True)
        setup_schema: Create Neo4j constraint (default True)

    Returns:
        Dictionary with pipeline statistics:
        - batches_processed: Number of micro-batches
        - rows_written: Total rows written to Neo4j
        - total_time: Elapsed time in seconds
        - rows_per_second: Throughput metric
    """
    if checkpoint_location is None:
        checkpoint_location = CHECKPOINT_LOCATION

    pipeline_start = time.time()

    # =========================================================================
    # STEP 1: Print header and load configuration
    # =========================================================================
    print_section_header("NEO4J WRITE PERFORMANCE TEST")
    print("Testing Neo4j write performance with random embeddings")
    print(f"Target label: :{TEST_LABEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS} (random)")

    # Load Neo4j configuration from Databricks Secrets
    config = load_config(
        dbutils,
        SCOPE_NAME,
        DEFAULT_DATABASE,
        DEFAULT_PROTOCOL,
        DEFAULT_EMBEDDING_ENDPOINT,
    )
    print_config(config, SCOPE_NAME, EMBEDDING_DIMENSIONS, batch_size)

    # =========================================================================
    # STEP 2: Test Neo4j connection
    # =========================================================================
    if not test_neo4j_connection(config):
        print("\nAborting: Neo4j connection test failed.")
        return {}

    # =========================================================================
    # STEP 3: Setup Neo4j schema (constraint only, no vector index needed)
    # =========================================================================
    if setup_schema:
        schema_config = SchemaConfig(
            node_label=TEST_LABEL,
            id_property="removal_id",
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            constraint_name=f"{TEST_LABEL.lower()}_removal_id_unique",
            vector_index_name=f"{TEST_LABEL.lower()}_embeddings",  # Won't be created
        )
        # Note: For testing, we only need the constraint, not the vector index
        # The setup function will create both, but the index isn't used in testing
        setup_neo4j_schema(config, schema_config)

    # =========================================================================
    # STEP 4: Cleanup existing test nodes (optional)
    # =========================================================================
    if cleanup_nodes:
        delete_nodes_by_label(config, TEST_LABEL)

    # =========================================================================
    # STEP 5: Configure and run the streaming pipeline
    # =========================================================================

    # Configure the embedding provider (random for testing)
    embedding_config = EmbeddingConfig(
        endpoint_name="random",  # Not used for random provider
        dimensions=EMBEDDING_DIMENSIONS,
        text_column="removal_reason",
        output_column="embedding",
    )
    embedding_provider = RandomEmbeddingProvider(embedding_config)

    # Validate (always passes for random, but good practice)
    print_section_header("VALIDATING EMBEDDING PROVIDER")
    embedding_provider.validate_endpoint()

    # Configure the pipeline
    pipeline_config = PipelineConfig(
        source_table=SOURCE_TABLE,
        node_label=TEST_LABEL,
        id_column="removal_id",
        batch_size=batch_size,
        write_partitions=WRITE_PARTITIONS,
        checkpoint_location=checkpoint_location,
        max_files_per_trigger=1,
        max_rows=MAX_ROWS,
    )

    # Validate max_rows was set correctly
    print(f"PipelineConfig created: max_rows={pipeline_config.max_rows}")
    if MAX_ROWS > 0 and pipeline_config.max_rows != MAX_ROWS:
        raise ValueError(f"max_rows mismatch: expected {MAX_ROWS}, got {pipeline_config.max_rows}")

    # Run the pipeline (streaming with optional row limit)
    stats = run_pipeline(
        spark=spark,  # Databricks provides this globally
        neo4j_config=config,
        pipeline_config=pipeline_config,
        embedding_provider=embedding_provider,
        column_selector=select_columns,
        clear_checkpoint=clear_checkpoint,
        dbutils=dbutils,  # Databricks provides this globally
    )

    # =========================================================================
    # STEP 6: Verify results
    # =========================================================================
    verify_test_nodes(config)

    # =========================================================================
    # STEP 7: Print summary
    # =========================================================================
    print_pipeline_summary(stats, pipeline_config, embedding_provider)

    total_time = time.time() - pipeline_start
    print(f"\nTotal test time: {format_duration(total_time)}")
    print("\nDone!")

    return stats


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
