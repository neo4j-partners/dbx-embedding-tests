"""
Structured Streaming Pipeline for Neo4j Embedding Writes
========================================================

This module provides a complete Structured Streaming pipeline for reading
Delta tables, generating embeddings, and writing to Neo4j.

TUTORIAL: Structured Streaming for ETL
--------------------------------------

Databricks Structured Streaming is ideal for processing large datasets
in controlled batches. Key benefits:

1. **Memory Control**: Only one micro-batch in memory at a time
2. **Checkpointing**: Resume from failure without reprocessing
3. **Backpressure**: Automatic rate limiting
4. **Delta Integration**: Native support for Delta Lake tables

Pipeline Architecture:
----------------------

    Delta Table (Source)
         │
         ▼
    ┌─────────────────┐
    │ readStream      │  maxFilesPerTrigger=1
    │ (micro-batches) │  (one Delta file per batch)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ foreachBatch    │  Custom batch processing
    │ (embed + write) │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌───────┐       ┌───────────┐
│Chunk 1│  ...  │ Chunk N   │  (BATCH_SIZE rows each)
└───┬───┘       └─────┬─────┘
    │                 │
    ▼                 ▼
┌──────────────────────────┐
│ Embedding Generation     │  ai_query() or random
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Neo4j Spark Connector    │  MERGE on node.keys
└──────────────────────────┘

Why foreachBatch?
-----------------

The foreachBatch sink gives you complete control over batch processing:

    def process_batch(df: DataFrame, batch_id: int):
        # Your custom logic here
        # - Generate embeddings
        # - Transform data
        # - Write to external systems

    stream.writeStream.foreachBatch(process_batch).start()

This is more flexible than built-in sinks because you can:
- Call external APIs (like embedding models)
- Write to multiple destinations
- Apply custom error handling
- Log detailed metrics

References:
    - https://docs.databricks.com/aws/en/structured-streaming/foreach
    - https://docs.databricks.com/aws/en/structured-streaming/delta-lake
    - https://neo4j.com/docs/spark/current/write/
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, floor as spark_floor, lit, row_number
from pyspark.sql.window import Window

from embedding_providers import EmbeddingProvider
from load_utils import Config, print_section_header


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the streaming pipeline.

    Tutorial: Pipeline Tuning Parameters
    ------------------------------------

    **batch_size (default: 5000)**
        Number of rows to process in each chunk. This affects:
        - ai_query batch size (larger = more efficient, but more memory)
        - Neo4j transaction size (larger = faster, but more lock contention)

        Recommended: 1000-10000 depending on embedding model throughput

    **write_partitions (default: 1)**
        Number of Spark partitions for Neo4j writes. This controls
        parallel writes to Neo4j.

        - 1: Serial writes (safest, no lock contention)
        - 2-4: Moderate parallelism (good for high-performance clusters)
        - 4+: High parallelism (may cause lock contention)

        Start with 1 and increase if you need more throughput.

    **checkpoint_location**
        Path for Spark streaming checkpoints. This enables:
        - Resume from failure
        - Exactly-once processing semantics
        - Progress tracking

        Delete the checkpoint to reprocess all data from scratch.

    **max_files_per_trigger (default: 1)**
        Number of Delta files to process per micro-batch.
        - 1: Predictable batch sizes (recommended)
        - >1: Larger batches, potentially faster

    Attributes:
        source_table: Unity Catalog table path
        node_label: Neo4j node label for writes
        id_column: Column to use as node key
        batch_size: Rows per processing chunk
        write_partitions: Parallel Neo4j writers
        checkpoint_location: Path for streaming checkpoints
        max_files_per_trigger: Delta files per micro-batch
    """

    source_table: str
    node_label: str
    id_column: str
    batch_size: int = 5000
    write_partitions: int = 1
    checkpoint_location: str = "/tmp/embedding_pipeline_checkpoint"
    max_files_per_trigger: int = 1
    max_rows: int = -1  # -1 means no limit, otherwise limit to this many rows

    # Column mappings (source -> destination)
    column_mappings: dict = field(default_factory=dict)


# =============================================================================
# STATISTICS TRACKING
# =============================================================================

@dataclass
class PipelineStats:
    """Accumulates statistics across streaming batches.

    Tutorial: Metrics Collection Pattern
    ------------------------------------

    This class collects metrics during pipeline execution. It's passed to
    the batch processor as a mutable container that persists across batches.

    Why a dataclass with mutable fields?
    - Spark's foreachBatch closure captures variables by reference
    - A mutable container allows updating state across batches
    - Dataclass provides clean structure and defaults

    Metrics collected:
    - batches_processed: Number of micro-batches completed
    - rows_read: Total input rows (may include filtered nulls)
    - rows_written: Rows successfully written to Neo4j
    - batch_times: List of batch durations for analysis
    - failed_batches: Count of failed batches (for error tracking)

    Example:
        >>> stats = PipelineStats()
        >>> process_batch(df, 0, config, stats)
        >>> print(f"Wrote {stats.rows_written:,} rows")
    """

    batches_processed: int = 0
    rows_read: int = 0
    rows_written: int = 0
    failed_batches: int = 0
    batch_times: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_batch(
        self,
        input_count: int,
        written_count: int,
        batch_time: float,
        failed: bool = False,
    ) -> None:
        """Record metrics for a completed batch.

        Args:
            input_count: Rows in the input batch
            written_count: Rows successfully written
            batch_time: Total batch processing time in seconds
            failed: Whether the batch failed
        """
        self.batches_processed += 1
        self.rows_read += input_count
        self.rows_written += written_count
        self.batch_times.append(batch_time)
        if failed:
            self.failed_batches += 1

    def to_dict(self) -> dict:
        """Convert stats to dictionary for reporting.

        Returns:
            Dictionary with all metrics including derived values
        """
        total_time = time.time() - self.start_time
        avg_batch_time = (
            sum(self.batch_times) / len(self.batch_times)
            if self.batch_times else 0
        )

        return {
            "batches_processed": self.batches_processed,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "failed_batches": self.failed_batches,
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "min_batch_time": min(self.batch_times) if self.batch_times else 0,
            "max_batch_time": max(self.batch_times) if self.batch_times else 0,
            "rows_per_second": self.rows_written / total_time if total_time > 0 else 0,
        }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def chunk_dataframe(
    df: DataFrame,
    chunk_size: int,
) -> DataFrame:
    """Add chunk IDs to a DataFrame for processing in smaller pieces.

    Tutorial: Chunking Large Batches
    --------------------------------

    Delta files can be large (50k+ rows), but ai_query and Neo4j writes
    work better with smaller batches. This function adds a _chunk_id column
    to enable splitting:

        df with 50,000 rows, chunk_size=5,000
        → 10 chunks with _chunk_id 0-9

    Implementation uses row_number() window function:

        1. Assign sequential row numbers (1, 2, 3, ...)
        2. Calculate chunk ID: (row_num - 1) // chunk_size
        3. Filter by chunk_id in a loop to process each chunk

    Why not repartition()?
    - repartition() shuffles data across the cluster
    - We want logical grouping, not physical redistribution
    - This approach is deterministic and reproducible

    Args:
        df: Input DataFrame
        chunk_size: Rows per chunk

    Returns:
        DataFrame with _row_num and _chunk_id columns added
    """
    # Window for assigning row numbers
    # orderBy(lit(1)) provides deterministic ordering
    window = Window.orderBy(lit(1))

    return (
        df
        .withColumn("_row_num", row_number().over(window))
        .withColumn("_chunk_id", spark_floor((col("_row_num") - 1) / lit(chunk_size)))
    )


def write_to_neo4j(
    df: DataFrame,
    neo4j_config: Config,
    node_label: str,
    node_key: str,
    batch_size: int = 5000,
) -> None:
    """Write a DataFrame to Neo4j using the Spark Connector.

    Tutorial: Neo4j Spark Connector Write Modes
    -------------------------------------------

    The Neo4j Spark Connector supports several write patterns:

    **Mode "Overwrite" with node.keys (MERGE pattern)**
        - If node exists (matching key): UPDATE properties
        - If node doesn't exist: CREATE node
        - This is what we use for embedding updates

        Generated Cypher:
            UNWIND $rows AS row
            MERGE (n:Label {key: row.key})
            SET n += row

    **Mode "Append" (CREATE pattern)**
        - Always creates new nodes
        - Fast but may create duplicates
        - Use when you're sure nodes don't exist

    **Custom query mode**
        - Full control with your own Cypher
        - More flexible but potentially slower

    Connector Options:
    ------------------
    - batch.size: Rows per UNWIND transaction (default 5000)
    - transaction.retries: Retry count for transient failures
    - transaction.retry.timeout: Delay between retries (ms)

    Args:
        df: DataFrame to write (must include node_key column and embedding)
        neo4j_config: Neo4j connection configuration
        node_label: Label for nodes (e.g., "RemovalEvent")
        node_key: Column to use as unique key for MERGE
        batch_size: Rows per transaction

    Example:
        >>> write_to_neo4j(
        ...     df_with_embeddings,
        ...     config,
        ...     node_label="RemovalEvent",
        ...     node_key="removal_id",
        ... )
    """
    (
        df.write.format("org.neo4j.spark.DataSource")
        .mode("Overwrite")
        .option("url", neo4j_config.uri)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", neo4j_config.username)
        .option("authentication.basic.password", neo4j_config.password)
        .option("database", neo4j_config.database)
        .option("labels", f":{node_label}")
        .option("node.keys", node_key)
        .option("batch.size", str(batch_size))
        .option("transaction.retries", "3")
        .option("transaction.retry.timeout", "30000")
        .save()
    )


def process_batch(
    batch_df: DataFrame,
    batch_id: int,
    neo4j_config: Config,
    pipeline_config: PipelineConfig,
    embedding_provider: EmbeddingProvider,
    stats: PipelineStats,
    debug: bool = True,
) -> None:
    """Process a single micro-batch: chunk, embed, filter, and write to Neo4j.

    Tutorial: Batch Processing Flow
    -------------------------------

    This is the core batch processor called by foreachBatch. It handles:

    1. **Count input rows** - Know what we're working with
    2. **Chunk the batch** - Split large batches into manageable pieces
    3. **For each chunk:**
       a. Generate embeddings (via provider)
       b. Filter invalid embeddings (null or wrong dimensions)
       c. Write valid rows to Neo4j
    4. **Collect statistics** - Track progress and performance

    Error Handling:
    ---------------
    - Individual chunk failures are logged but don't stop the pipeline
    - Stats track failed batches for later investigation
    - Neo4j connection errors are raised to fail the batch

    Caching Strategy:
    -----------------
    We cache intermediate DataFrames to avoid recomputation:
    - batch_numbered: Cached for chunk filtering
    - chunk_with_embeddings: Cached for count + write

    Always unpersist after use to free memory!

    Args:
        batch_df: The micro-batch DataFrame from Structured Streaming
        batch_id: Unique identifier for this batch (from Spark)
        neo4j_config: Neo4j connection settings
        pipeline_config: Pipeline configuration (batch size, partitions, etc.)
        embedding_provider: Provider for generating embeddings
        stats: Mutable stats container (updated in-place)
        debug: If True, print detailed debug information
    """
    batch_start = time.time()
    max_rows = pipeline_config.max_rows

    # Check if we've already hit the row limit
    if max_rows > 0 and stats.rows_written >= max_rows:
        print(f"  Batch {batch_id}: Skipping (already processed {stats.rows_written:,} of {max_rows:,} max rows)", flush=True)
        return

    # Step 1: Count input rows
    input_count = batch_df.count()
    print(f"  Batch {batch_id}: {input_count:,} input rows ({time.time() - batch_start:.2f}s)", flush=True)

    if input_count == 0:
        print(f"  Batch {batch_id}: Empty, skipping", flush=True)
        return

    # Limit rows if we're approaching max_rows
    if max_rows > 0:
        remaining = max_rows - stats.rows_written
        if input_count > remaining:
            print(f"  Batch {batch_id}: Limiting to {remaining:,} rows (max_rows={max_rows:,})", flush=True)
            batch_df = batch_df.limit(remaining)
            input_count = remaining

    # Step 2: Chunk the batch for processing
    chunk_size = pipeline_config.batch_size
    num_chunks = (input_count + chunk_size - 1) // chunk_size
    print(f"  Batch {batch_id}: Processing in {num_chunks} chunk(s) of ~{chunk_size:,} rows", flush=True)

    # Add chunking columns and cache
    batch_numbered = chunk_dataframe(batch_df, chunk_size).cache()

    total_written = 0

    # Step 3: Process each chunk
    for chunk_idx in range(num_chunks):
        chunk_start = time.time()

        # Extract rows for this chunk
        chunk_df = (
            batch_numbered
            .filter(col("_chunk_id") == chunk_idx)
            .drop("_row_num", "_chunk_id")
        )

        chunk_count = chunk_df.count()
        if chunk_count == 0:
            continue

        print(f"    Chunk {chunk_idx + 1}/{num_chunks}: {chunk_count:,} rows - generating embeddings...", flush=True)

        # Step 3a: Generate embeddings
        embed_start = time.time()
        chunk_with_embeddings = embedding_provider.add_embeddings(chunk_df).cache()
        chunk_with_embeddings.count()  # Force evaluation
        print(f"    Chunk {chunk_idx + 1}/{num_chunks}: embeddings done ({time.time() - embed_start:.2f}s)", flush=True)

        # Debug: Check embedding format
        if debug:
            _debug_embeddings(chunk_with_embeddings, embedding_provider.config.dimensions)

        # Step 3b: Filter valid embeddings
        chunk_valid = (
            embedding_provider.filter_valid_embeddings(chunk_with_embeddings)
            .coalesce(pipeline_config.write_partitions)
        )

        valid_count = chunk_valid.count()
        chunk_with_embeddings.unpersist()

        if valid_count == 0:
            print(f"    Chunk {chunk_idx + 1}/{num_chunks}: no valid rows, skipping", flush=True)
            if debug:
                print(f"    DEBUG: All {chunk_count} rows filtered - check embedding format", flush=True)
            continue

        # Step 3c: Write to Neo4j
        print(f"    Chunk {chunk_idx + 1}/{num_chunks}: writing {valid_count:,} to Neo4j...", flush=True)

        try:
            write_to_neo4j(
                chunk_valid,
                neo4j_config,
                pipeline_config.node_label,
                pipeline_config.id_column,
                pipeline_config.batch_size,
            )

            total_written += valid_count
            chunk_elapsed = time.time() - chunk_start
            rate = valid_count / chunk_elapsed if chunk_elapsed > 0 else 0
            print(f"    Chunk {chunk_idx + 1}/{num_chunks}: DONE ({chunk_elapsed:.2f}s, {rate:.1f} rows/s)", flush=True)

        except Exception as e:
            print(f"    ERROR in chunk {chunk_idx + 1}: {e}", flush=True)
            stats.failed_batches += 1
            raise

    # Cleanup cached DataFrame
    batch_numbered.unpersist()

    # Record statistics
    batch_time = time.time() - batch_start
    stats.record_batch(input_count, total_written, batch_time)

    rate = total_written / batch_time if batch_time > 0 else 0
    print(f"  Batch {batch_id}: COMPLETE - {total_written:,} rows in {batch_time:.2f}s ({rate:.1f} rows/s)", flush=True)


def _debug_embeddings(df: DataFrame, expected_dims: int) -> None:
    """Print debug information about embeddings in a DataFrame.

    Args:
        df: DataFrame with embedding column
        expected_dims: Expected embedding dimensions
    """
    sample = df.select("embedding").limit(1).collect()
    if sample:
        emb = sample[0]["embedding"]
        if emb is None:
            print("    DEBUG: embedding is NULL", flush=True)
        else:
            size_str = str(len(emb)) if hasattr(emb, '__len__') else 'N/A'
            print(f"    DEBUG: embedding type={type(emb).__name__}, size={size_str}", flush=True)
            if hasattr(emb, '__len__') and len(emb) > 0:
                preview = [f"{v:.4f}" for v in emb[:3]]
                print(f"    DEBUG: first 3 values=[{', '.join(preview)}]", flush=True)
            if hasattr(emb, '__len__') and len(emb) != expected_dims:
                print(f"    DEBUG: DIMENSION MISMATCH! Expected {expected_dims}, got {len(emb)}", flush=True)


# =============================================================================
# STREAMING PIPELINE
# =============================================================================

def run_streaming_pipeline(
    spark: SparkSession,
    neo4j_config: Config,
    pipeline_config: PipelineConfig,
    embedding_provider: EmbeddingProvider,
    column_selector: Callable[[DataFrame], DataFrame],
    clear_checkpoint: bool = True,
    dbutils=None,
) -> dict:
    """Run the complete streaming pipeline.

    Tutorial: Structured Streaming with availableNow
    ------------------------------------------------

    This function sets up and runs a Structured Streaming pipeline using
    the "availableNow" trigger:

        stream.writeStream
            .trigger(availableNow=True)
            .start()

    The availableNow trigger:
    - Processes all available data in micro-batches
    - Stops automatically when done
    - Respects checkpoints for incremental processing

    This is perfect for batch-like processing with streaming benefits:
    - Checkpoint-based resume on failure
    - Controlled batch sizes
    - Memory-efficient processing

    Pipeline Steps:
    1. Clear checkpoint if requested (for fresh start)
    2. Set up stream reader with maxFilesPerTrigger
    3. Apply column selection/transformation
    4. Filter null text values
    5. Start foreachBatch with custom processor
    6. Wait for completion
    7. Return statistics

    Args:
        spark: SparkSession instance
        neo4j_config: Neo4j connection configuration
        pipeline_config: Pipeline settings (source table, batch size, etc.)
        embedding_provider: Provider for generating embeddings
        column_selector: Function to select/rename columns from source
        clear_checkpoint: If True, delete checkpoint for fresh start
        dbutils: Databricks dbutils for checkpoint clearing (optional)

    Returns:
        Dictionary with pipeline statistics

    Example:
        >>> def select_columns(df):
        ...     return df.select(
        ...         col("id").alias("removal_id"),
        ...         col("text").alias("removal_reason"),
        ...     )
        >>> stats = run_streaming_pipeline(
        ...     spark, neo4j_config, pipeline_config,
        ...     embedding_provider, select_columns
        ... )
        >>> print(f"Wrote {stats['rows_written']:,} rows")
    """
    print_section_header("STRUCTURED STREAMING PIPELINE")

    print(f"Source table: {pipeline_config.source_table}")
    print(f"Target label: :{pipeline_config.node_label}")
    print(f"Batch size: {pipeline_config.batch_size:,}")
    print(f"Max files per trigger: {pipeline_config.max_files_per_trigger}")
    print(f"Checkpoint: {pipeline_config.checkpoint_location}")
    if pipeline_config.max_rows > 0:
        print(f"Max rows: {pipeline_config.max_rows:,}")
    else:
        print(f"Max rows: unlimited")

    # Clear checkpoint if requested
    if clear_checkpoint:
        _clear_checkpoint(pipeline_config.checkpoint_location, dbutils)

    # Initialize statistics
    stats = PipelineStats()

    # Build streaming query
    stream_df = (
        spark.readStream
        .option("maxFilesPerTrigger", str(pipeline_config.max_files_per_trigger))
        .table(pipeline_config.source_table)
    )

    # Apply column selection
    stream_df = column_selector(stream_df)

    # Filter nulls in text column (embedding would fail anyway)
    text_column = embedding_provider.config.text_column
    stream_df = stream_df.filter(col(text_column).isNotNull())

    # Create batch processor closure
    def batch_processor(batch_df: DataFrame, batch_id: int) -> None:
        """Closure that processes each micro-batch by generating embeddings and writing to Neo4j."""
        process_batch(
            batch_df,
            batch_id,
            neo4j_config,
            pipeline_config,
            embedding_provider,
            stats,
        )

    # Start streaming query
    print("\nStarting streaming pipeline...", flush=True)
    print(f"  (Delete checkpoint to reprocess all data)", flush=True)

    query = (
        stream_df.writeStream
        .foreachBatch(batch_processor)
        .option("checkpointLocation", pipeline_config.checkpoint_location)
        .trigger(availableNow=True)
        .start()
    )

    print(f"  Query ID: {query.id}", flush=True)

    # Wait for completion
    while query.isActive:
        if query.awaitTermination(timeout=30):
            break

    # Check for errors
    if query.exception():
        print(f"  ERROR: {query.exception()}", flush=True)
        raise query.exception()

    # Return statistics
    return stats.to_dict()


def _clear_checkpoint(checkpoint_location: str, dbutils=None) -> None:
    """Clear the checkpoint directory.

    Args:
        checkpoint_location: Path to checkpoint directory
        dbutils: Databricks dbutils (optional)
    """
    print(f"\nClearing checkpoint at {checkpoint_location}...", flush=True)

    if dbutils is not None:
        try:
            dbutils.fs.rm(checkpoint_location, recurse=True)
            print("  Checkpoint cleared", flush=True)
        except Exception:
            print("  No existing checkpoint to clear", flush=True)
    else:
        # Local filesystem fallback
        import shutil
        import os
        try:
            if os.path.exists(checkpoint_location):
                shutil.rmtree(checkpoint_location)
                print("  Checkpoint cleared", flush=True)
            else:
                print("  No existing checkpoint to clear", flush=True)
        except Exception as e:
            print(f"  Could not clear checkpoint: {e}", flush=True)


# =============================================================================
# RESULT PRINTING
# =============================================================================

def print_pipeline_summary(
    stats: dict,
    pipeline_config: PipelineConfig,
    embedding_provider: EmbeddingProvider,
) -> None:
    """Print a formatted summary of the pipeline run.

    Args:
        stats: Pipeline statistics dictionary
        pipeline_config: Pipeline configuration
        embedding_provider: Embedding provider used
    """
    print_section_header("PIPELINE SUMMARY")

    print("\n  Settings:")
    print(f"    Source table: {pipeline_config.source_table}")
    print(f"    Neo4j label: :{pipeline_config.node_label}")
    print(f"    Embedding endpoint: {embedding_provider.config.endpoint_name}")
    print(f"    Embedding dimensions: {embedding_provider.config.dimensions}")
    print(f"    Batch size: {pipeline_config.batch_size:,}")
    print(f"    Checkpoint: {pipeline_config.checkpoint_location}")

    print("\n  Results:")
    print(f"    Batches processed: {stats['batches_processed']}")
    print(f"    Rows read: {stats['rows_read']:,}")
    print(f"    Rows written: {stats['rows_written']:,}")
    if stats['failed_batches'] > 0:
        print(f"    Failed batches: {stats['failed_batches']}")

    print("\n  Timing:")
    total_time = stats['total_time']
    print(f"    Total time: {total_time:.2f}s", end="")
    if total_time > 60:
        print(f" ({total_time / 60:.1f} minutes)")
    else:
        print()
    print(f"    Avg batch time: {stats['avg_batch_time']:.2f}s")
    print(f"    Throughput: {stats['rows_per_second']:.1f} rows/second")
    if pipeline_config.max_rows > 0:
        print(f"    Max rows limit: {pipeline_config.max_rows:,}")
    print()


# =============================================================================
# BATCH PIPELINE (FOR TESTING WITH ROW LIMITS)
# =============================================================================

def run_batch_pipeline(
    spark: SparkSession,
    neo4j_config: Config,
    pipeline_config: PipelineConfig,
    embedding_provider: EmbeddingProvider,
    column_selector: Callable[[DataFrame], DataFrame],
) -> dict:
    """Run the pipeline in batch mode with optional row limit.

    Tutorial: Batch Mode for Testing
    ---------------------------------

    When testing with a limited number of rows (MAX_ROWS), it's more efficient
    to use batch mode instead of Structured Streaming:

    - **Streaming**: Designed for processing all data with checkpointing
    - **Batch**: Better for testing with row limits, no checkpoint overhead

    This function:
    1. Reads data as a batch (not streaming)
    2. Applies the row limit if max_rows > 0
    3. Processes in chunks with embedding generation
    4. Writes to Neo4j

    Use this for:
    - Quick tests with limited rows (e.g., MAX_ROWS=500)
    - Debugging embedding or write issues
    - Validating pipeline configuration

    Args:
        spark: SparkSession instance
        neo4j_config: Neo4j connection configuration
        pipeline_config: Pipeline settings including max_rows
        embedding_provider: Provider for generating embeddings
        column_selector: Function to select/rename columns from source

    Returns:
        Dictionary with pipeline statistics

    Example:
        >>> config = PipelineConfig(..., max_rows=500)
        >>> stats = run_batch_pipeline(spark, neo4j_config, config, provider, selector)
    """
    print_section_header("BATCH PIPELINE (LIMITED ROWS)")

    max_rows = pipeline_config.max_rows
    print(f"Source table: {pipeline_config.source_table}")
    print(f"Target label: :{pipeline_config.node_label}")
    print(f"Batch size: {pipeline_config.batch_size:,}")
    print(f"Max rows: {max_rows:,}" if max_rows > 0 else "Max rows: unlimited")

    # Initialize statistics
    stats = PipelineStats()

    # Read data as batch
    print("\nReading source data...", flush=True)
    df = spark.read.table(pipeline_config.source_table)

    # Apply column selection
    df = column_selector(df)

    # Filter nulls
    text_column = embedding_provider.config.text_column
    df = df.filter(col(text_column).isNotNull())

    # Apply row limit if specified
    if max_rows > 0:
        df = df.limit(max_rows)
        print(f"  Limited to {max_rows:,} rows", flush=True)

    # Count rows
    total_rows = df.count()
    print(f"  Total rows to process: {total_rows:,}", flush=True)

    if total_rows == 0:
        print("  No rows to process!", flush=True)
        return stats.to_dict()

    # Process as a single "batch"
    process_batch(
        df,
        batch_id=0,
        neo4j_config=neo4j_config,
        pipeline_config=pipeline_config,
        embedding_provider=embedding_provider,
        stats=stats,
        debug=True,
    )

    return stats.to_dict()


def run_pipeline(
    spark: SparkSession,
    neo4j_config: Config,
    pipeline_config: PipelineConfig,
    embedding_provider: EmbeddingProvider,
    column_selector: Callable[[DataFrame], DataFrame],
    clear_checkpoint: bool = True,
    dbutils=None,
) -> dict:
    """Run the streaming pipeline with optional row limit.

    Tutorial: Unified Streaming Pipeline
    ------------------------------------

    This function always uses Structured Streaming, which provides:
    - Checkpoint-based resume on failure
    - Memory-efficient micro-batch processing
    - Consistent behavior for both limited and full dataset runs

    The max_rows setting is enforced within process_batch():
    - max_rows > 0: Stop after processing max_rows rows
    - max_rows <= 0: Process all available data

    Args:
        spark: SparkSession instance
        neo4j_config: Neo4j connection configuration
        pipeline_config: Pipeline settings including max_rows
        embedding_provider: Provider for generating embeddings
        column_selector: Function to select/rename columns from source
        clear_checkpoint: If True, delete checkpoint for fresh start
        dbutils: Databricks dbutils for checkpoint clearing

    Returns:
        Dictionary with pipeline statistics
    """
    # Log configuration
    if pipeline_config.max_rows > 0:
        print(f"\nPipeline: max_rows={pipeline_config.max_rows:,} (will stop after limit)")
    else:
        print(f"\nPipeline: max_rows={pipeline_config.max_rows} (processing all rows)")

    return run_streaming_pipeline(
        spark=spark,
        neo4j_config=neo4j_config,
        pipeline_config=pipeline_config,
        embedding_provider=embedding_provider,
        column_selector=column_selector,
        clear_checkpoint=clear_checkpoint,
        dbutils=dbutils,
    )
