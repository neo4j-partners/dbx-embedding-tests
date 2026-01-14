# Writing Embeddings from Spark to Neo4j

This document compares approaches for writing embeddings from Spark DataFrames (Databricks) into Neo4j and explains the patterns demonstrated in this repository.

## Two Approaches

### 1. Cypher Query with `db.create.setNodeVectorProperty()` (Slow)

```python
query = """
MATCH (n:ScheduleInterruption {AC_EV_ID: event.AC_EV_ID})
CALL db.create.setNodeVectorProperty(n, "embedding", event.embedding)
"""

df.coalesce(1).write.format("org.neo4j.spark.DataSource").mode("overwrite")\
    .option("query", query)\
    .option("batch.size", "1000").save()
```

**Why this is slow (even with an index on the match property):**

While an index makes each individual `MATCH` fast, the fundamental bottleneck is the execution model:

1. **Per-row procedure overhead**: The `db.create.setNodeVectorProperty()` procedure is called separately for each row. Procedure invocation has inherent overhead that compounds across millions of rows.

2. **No true batching**: When using the `query` option with `event.` syntax, the Spark Connector executes the query once per row. The `batch.size` option only controls how many rows are sent per transaction, but each row still triggers a separate query execution within that transaction.

3. **Sequential processing**: The query executes as: "for each row, find node, call procedure" rather than "find all nodes at once, update all at once."

In contrast, the direct Spark Connector approach generates a single `UNWIND $rows AS row MERGE ... SET ...` statement that processes the entire batch in one query execution, dramatically reducing overhead.

### 2. Direct Spark Connector Write (Fast)

```python
df.write.format("org.neo4j.spark.DataSource")
    .mode("Overwrite")
    .option("url", neo4j_url)
    .option("authentication.type", "basic")
    .option("authentication.basic.username", username)
    .option("authentication.basic.password", password)
    .option("labels", ":NodeLabel")
    .option("node.keys", "node_id")
    .option("batch.size", "5000")
    .option("transaction.retries", "3")
    .option("transaction.retry.timeout", "30000")
    .save()
```

This approach uses optimized batch writes with UNWIND + MERGE under the hood.

---

## Comparison

| Aspect | Cypher Query Approach | Direct Spark Connector |
|--------|----------------------|------------------------|
| **Mechanism** | `MATCH` + `db.create.setNodeVectorProperty()` | Native DataFrame write with UNWIND + MERGE |
| **Speed** | Slow - sequential node lookups per row | Fast - optimized batch writes |
| **Parallelism** | Limited by single Cypher execution | Configurable via DataFrame partitions |
| **Batching** | Manual via `batch.size` option | Built-in batch optimization |
| **Type Handling** | Explicit vector property procedure | Relies on Spark Connector type mapping |
| **Memory Usage** | Lower (processes row by row) | Higher (batch buffering) |
| **Error Handling** | Per-row error visibility | Batch-level errors |

### Performance Expectations

For the direct Spark Connector approach with proper configuration:
- **Batch size**: 5,000 rows per Neo4j transaction
- **Parallel writers**: 4 partitions writing concurrently
- **Expected throughput**: 500-2,000 rows/second (depending on network latency and Neo4j cluster size)

---

## Common Error: `ArrayType(DoubleType,true)`

When using the direct Spark Connector approach, you may encounter:

```
java.util.NoSuchElementException: key not found: ArrayType(DoubleType,true)
```

This indicates a **Spark-to-Neo4j type mapping issue**:

1. **Type Mismatch**: Neo4j vector indexes expect `LIST<FLOAT>`, but Spark may produce `ArrayType(DoubleType)` (64-bit doubles instead of 32-bit floats).

2. **Connector Version**: Older versions of the Neo4j Spark Connector may not handle array types correctly.

---

## Patterns Demonstrated in This Repository

The `custom_embeddings_load_test.py` script demonstrates a working pattern by addressing these type mapping issues:

### 1. Uses Compatible Embedding Generation

The script uses Databricks `ai_query()` function which returns embeddings as `ArrayType(FloatType)` by default. This means no explicit cast is needed—the embeddings are already in the correct format for the Neo4j Spark Connector.

If you're using a different embedding method that produces `ArrayType(DoubleType)` (e.g., a custom UDF or external API), you would need to cast explicitly:

```python
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, FloatType

df = df.withColumn("embedding", col("embedding").cast(ArrayType(FloatType())))
```

But with `ai_query()`, this step is unnecessary.

### 2. Configures Vector Index for LIST<FLOAT>

The vector index is created to work with `LIST<FLOAT>` properties:

```python
CREATE VECTOR INDEX removal_reason_embeddings IF NOT EXISTS
FOR (r:RemovalEvent) ON (r.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
}}
```

---

## Recommended Fixes

### Fix 1: Cast Embeddings to Float Array (if not using ai_query)

If your embeddings are `ArrayType(DoubleType)`, cast them to `ArrayType(FloatType)` before writing to Neo4j. See the cast example in "Patterns Demonstrated" above.

Note: If you're using Databricks `ai_query()`, this is not needed—it already returns floats.

### Fix 2: Upgrade Spark Connector

Ensure you're using Neo4j Spark Connector 5.3.0 or later:

```
org.neo4j:neo4j-connector-apache-spark_2.12:5.3.0_for_spark_3
```

### Fix 3: Add Transaction Retry Options

Add retry options to handle transient failures:

```python
df.write.format("org.neo4j.spark.DataSource")\
    .mode("Overwrite")\
    .option("labels", ":ComponentRemovalNK")\
    .option("node.keys", "COMP_RMV_H_NATURAL_KEY_ID")\
    .option("batch.size", "5000")\
    .option("transaction.retries", "3")\
    .option("transaction.retry.timeout", "30000")\
    .save()
```

---

## Updating Existing Nodes with Embeddings

If nodes already exist in Neo4j without embeddings and you want to add embeddings to them:

### Approach 1: Mode "Overwrite" with Node Keys (Recommended)

The Spark Connector's `Overwrite` mode with `node.keys` performs a MERGE operation:

```python
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, FloatType

df_embeddings = df.select(
    col("COMP_RMV_H_NATURAL_KEY_ID"),
    col("embedding").cast(ArrayType(FloatType()))
)

df_embeddings.write.format("org.neo4j.spark.DataSource")\
    .mode("Overwrite")\
    .option("url", neo4j_url)\
    .option("authentication.basic.username", username)\
    .option("authentication.basic.password", password)\
    .option("labels", ":ComponentRemovalNK")\
    .option("node.keys", "COMP_RMV_H_NATURAL_KEY_ID")\
    .option("batch.size", "5000")\
    .save()
```

**How it works:**
- The Spark Connector generates: `UNWIND $rows AS row MERGE (n:ComponentRemovalNK {COMP_RMV_H_NATURAL_KEY_ID: row.COMP_RMV_H_NATURAL_KEY_ID}) SET n.embedding = row.embedding`
- Existing nodes are matched by the key and updated with the embedding
- If a node doesn't exist, it will be created

### Approach 2: Custom Cypher Query for Updates Only

If you only want to update existing nodes (not create new ones):

```python
update_query = """
MATCH (n:ComponentRemovalNK {COMP_RMV_H_NATURAL_KEY_ID: event.COMP_RMV_H_NATURAL_KEY_ID})
SET n.embedding = event.embedding
"""

df_embeddings.write.format("org.neo4j.spark.DataSource")\
    .mode("Overwrite")\
    .option("url", neo4j_url)\
    .option("authentication.basic.username", username)\
    .option("authentication.basic.password", password)\
    .option("query", update_query)\
    .option("batch.size", "5000")\
    .save()
```

### Performance Comparison for Updates

| Approach | Speed | Creates New Nodes? | Notes |
|----------|-------|-------------------|-------|
| Mode "Overwrite" + node.keys | Fastest | Yes (via MERGE) | Recommended if creating nodes is acceptable |
| Custom MATCH query | Medium | No | Use when you must not create nodes |

### Best Practices for Updating Existing Nodes

1. **Ensure an index exists on the match key:**
   ```cypher
   CREATE INDEX comp_rmv_key IF NOT EXISTS
   FOR (n:ComponentRemovalNK) ON (n.COMP_RMV_H_NATURAL_KEY_ID)
   ```

2. **Partition your DataFrame appropriately:**
   ```python
   df_embeddings = df_embeddings.repartition(4)  # Match to Neo4j write parallelism
   ```

3. **Monitor Neo4j memory during large updates:**
   - Large batch sizes consume more heap memory
   - Reduce `batch.size` if you see memory pressure

---

## References

- [Neo4j Spark Connector Documentation](https://neo4j.com/docs/spark/current/)
- [Neo4j Spark Connector on Databricks](https://neo4j.com/docs/spark/current/databricks/)
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Databricks ai_query Function](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query)
