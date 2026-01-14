"""
Neo4j Schema Management for Vector Embeddings
==============================================

This module provides functions for setting up and managing Neo4j schema elements
required for vector similarity search. It's designed to work with embeddings
written from Spark DataFrames.

TUTORIAL: Neo4j Schema for Vector Search
----------------------------------------

When storing embeddings in Neo4j for vector similarity search, you need:

1. **Uniqueness Constraint**: Ensures each node has a unique identifier.
   This enables MERGE operations (upsert) from Spark and prevents duplicates.

   Example:
       CREATE CONSTRAINT removal_event_id_unique IF NOT EXISTS
       FOR (r:RemovalEvent) REQUIRE r.removal_id IS UNIQUE

2. **Vector Index**: Enables fast approximate nearest neighbor (ANN) search
   on embedding properties. Neo4j uses HNSW algorithm internally.

   Example:
       CREATE VECTOR INDEX removal_embeddings IF NOT EXISTS
       FOR (r:RemovalEvent) ON (r.embedding)
       OPTIONS {indexConfig: {
           `vector.dimensions`: 384,
           `vector.similarity_function`: 'cosine'
       }}

IMPORTANT: LIST<FLOAT> vs VECTOR Type
-------------------------------------

Neo4j stores embeddings as LIST<FLOAT> when written from Spark Connector.
Vector indexes work natively with LIST<FLOAT> - no type conversion needed.

However, if you previously created a VECTOR type constraint on the embedding
property, Spark writes will fail with type errors. This module automatically
detects and removes such constraints.

The error you'd see without this fix:
    java.util.NoSuchElementException: key not found: ArrayType(DoubleType,true)

References:
    - https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
    - https://neo4j.com/docs/spark/current/write/labels/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from load_utils import Config, neo4j_driver, print_section_header


# =============================================================================
# SCHEMA CONFIGURATION
# =============================================================================

@dataclass
class SchemaConfig:
    """Configuration for Neo4j schema elements.

    This dataclass holds all the configuration needed to set up Neo4j schema
    for a specific node type with vector embeddings.

    Tutorial: Why Use a Dataclass?
    ------------------------------
    A dataclass provides:
    - Type hints for IDE support and documentation
    - Automatic __init__, __repr__, and __eq__ methods
    - Immutable-like semantics (though not enforced)
    - Clear documentation of required configuration

    Attributes:
        node_label: The Neo4j node label (e.g., "RemovalEvent")
        id_property: The property used as unique identifier (e.g., "removal_id")
        embedding_property: The property storing the embedding vector (usually "embedding")
        embedding_dimensions: Size of the embedding vector (e.g., 384 for MiniLM, 1024 for BGE)
        constraint_name: Name for the uniqueness constraint
        vector_index_name: Name for the vector similarity index
        similarity_function: Vector similarity metric ("cosine", "euclidean", or "dot_product")

    Example:
        >>> config = SchemaConfig(
        ...     node_label="RemovalEvent",
        ...     id_property="removal_id",
        ...     embedding_dimensions=384,
        ...     constraint_name="removal_event_id_unique",
        ...     vector_index_name="removal_embeddings",
        ... )
    """

    node_label: str
    id_property: str
    embedding_dimensions: int
    constraint_name: str
    vector_index_name: str
    embedding_property: str = "embedding"
    similarity_function: str = "cosine"


# =============================================================================
# TYPE CONSTRAINT MANAGEMENT
# =============================================================================

def find_type_constraints(
    config: Config,
    node_label: str,
    property_name: str,
) -> List[str]:
    """Find any TYPE constraints on a property that would block LIST<FLOAT> writes.

    Tutorial: Why Type Constraints Cause Problems
    ---------------------------------------------

    Neo4j allows you to create type constraints on properties:

        CREATE CONSTRAINT embedding_type FOR (n:MyNode)
        REQUIRE n.embedding IS :: VECTOR(384)

    This constrains the property to be of type VECTOR. However, when the
    Neo4j Spark Connector writes embeddings, it sends them as LIST<FLOAT>,
    not VECTOR. The constraint rejects the write, causing errors.

    This function finds such constraints so they can be dropped before
    writing embeddings from Spark.

    Args:
        config: Neo4j connection configuration
        node_label: The label to check for constraints
        property_name: The property to check (usually "embedding")

    Returns:
        List of constraint names that need to be dropped

    Example:
        >>> constraints = find_type_constraints(neo4j_config, "RemovalEvent", "embedding")
        >>> print(f"Found {len(constraints)} type constraints to drop")
    """
    query = """
        SHOW CONSTRAINTS
        YIELD name, type, labelsOrTypes, properties
        WHERE $node_label IN labelsOrTypes
          AND $property_name IN properties
          AND type CONTAINS 'TYPE'
        RETURN name
    """

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            result = session.run(
                query,
                node_label=node_label,
                property_name=property_name,
            )
            return [record["name"] for record in result]


def drop_type_constraints(
    config: Config,
    constraint_names: List[str],
    verbose: bool = True,
) -> int:
    """Drop specified constraints from Neo4j.

    Tutorial: Safe Constraint Dropping
    ----------------------------------

    Constraints are dropped by name. We use backticks to escape the name
    in case it contains special characters:

        DROP CONSTRAINT `my-constraint-name`

    This function handles errors gracefully - if a constraint doesn't exist
    (perhaps already dropped), it continues with the next one.

    Args:
        config: Neo4j connection configuration
        constraint_names: List of constraint names to drop
        verbose: If True, print progress messages

    Returns:
        Number of constraints successfully dropped

    Example:
        >>> dropped = drop_type_constraints(config, ["old_constraint"])
        >>> print(f"Dropped {dropped} constraints")
    """
    dropped_count = 0

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            for name in constraint_names:
                if verbose:
                    print(f"    Dropping type constraint: {name}...")
                try:
                    # Use backticks to escape constraint name safely
                    session.run(f"DROP CONSTRAINT `{name}`")
                    dropped_count += 1
                    if verbose:
                        print(f"      Done")
                except Exception as e:
                    if verbose:
                        print(f"      Note: {e}")

    return dropped_count


# =============================================================================
# SCHEMA CREATION
# =============================================================================

def build_schema_queries(schema_config: SchemaConfig) -> List[Tuple[str, str]]:
    """Build the Cypher queries needed to create schema elements.

    Tutorial: Schema Query Patterns
    -------------------------------

    This function generates two types of schema elements:

    1. **Uniqueness Constraint**: Enables efficient MERGE operations

       CREATE CONSTRAINT name IF NOT EXISTS
       FOR (n:Label) REQUIRE n.property IS UNIQUE

       The "IF NOT EXISTS" clause makes the query idempotent - safe to run
       multiple times without errors.

    2. **Vector Index**: Enables fast similarity search

       CREATE VECTOR INDEX name IF NOT EXISTS
       FOR (n:Label) ON (n.embedding)
       OPTIONS {indexConfig: {
           `vector.dimensions`: 384,
           `vector.similarity_function`: 'cosine'
       }}

       Configuration options:
       - vector.dimensions: Must match your embedding model's output size
       - vector.similarity_function: 'cosine' (normalized), 'euclidean', or 'dot_product'

    Args:
        schema_config: Configuration for the schema elements

    Returns:
        List of (description, query) tuples

    Example:
        >>> queries = build_schema_queries(my_schema_config)
        >>> for desc, query in queries:
        ...     print(f"Creating: {desc}")
        ...     session.run(query)
    """
    return [
        # Uniqueness constraint for MERGE operations
        (
            f"Uniqueness constraint on {schema_config.node_label}.{schema_config.id_property}",
            f"""
            CREATE CONSTRAINT {schema_config.constraint_name} IF NOT EXISTS
            FOR (r:{schema_config.node_label}) REQUIRE r.{schema_config.id_property} IS UNIQUE
            """,
        ),
        # Vector index for similarity search
        # Note: Works natively with LIST<FLOAT> from Spark Connector
        (
            f"Vector index for {schema_config.similarity_function} similarity",
            f"""
            CREATE VECTOR INDEX {schema_config.vector_index_name} IF NOT EXISTS
            FOR (r:{schema_config.node_label}) ON (r.{schema_config.embedding_property})
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {schema_config.embedding_dimensions},
                `vector.similarity_function`: '{schema_config.similarity_function}'
            }}}}
            """,
        ),
    ]


def setup_neo4j_schema(
    config: Config,
    schema_config: SchemaConfig,
    drop_existing_type_constraints: bool = True,
    verbose: bool = True,
) -> None:
    """Create Neo4j constraints and vector index for embedding storage.

    Tutorial: Complete Schema Setup Flow
    ------------------------------------

    This function performs the complete schema setup:

    1. **Check for conflicting type constraints**
       If there's a VECTOR type constraint on the embedding property,
       Spark writes will fail. We detect and remove these first.

    2. **Create uniqueness constraint**
       Enables efficient MERGE (upsert) operations when writing nodes.
       The Spark Connector uses MERGE internally when node.keys is specified.

    3. **Create vector index**
       Enables fast approximate nearest neighbor search on embeddings.
       The index is created asynchronously - use wait_for_vector_index()
       to wait for it to become ONLINE.

    Why "IF NOT EXISTS"?
    --------------------
    All CREATE statements use "IF NOT EXISTS" to make the function idempotent.
    You can safely call it multiple times without errors - it will skip
    creating elements that already exist.

    Args:
        config: Neo4j connection configuration
        schema_config: Configuration for schema elements
        drop_existing_type_constraints: If True, drop TYPE constraints on embedding property
        verbose: If True, print progress messages

    Example:
        >>> schema = SchemaConfig(
        ...     node_label="RemovalEvent",
        ...     id_property="removal_id",
        ...     embedding_dimensions=384,
        ...     constraint_name="removal_id_unique",
        ...     vector_index_name="removal_embeddings",
        ... )
        >>> setup_neo4j_schema(neo4j_config, schema)
        >>> wait_for_vector_index(neo4j_config, schema.vector_index_name)
    """
    if verbose:
        print_section_header("SETTING UP NEO4J SCHEMA")

    # Step 1: Handle type constraints that would block LIST<FLOAT> writes
    if drop_existing_type_constraints:
        if verbose:
            print("  Checking for type constraints on embedding property...")

        constraints = find_type_constraints(
            config,
            schema_config.node_label,
            schema_config.embedding_property,
        )

        if constraints:
            drop_type_constraints(config, constraints, verbose=verbose)
        elif verbose:
            print("    No type constraints found")

    # Step 2: Create schema elements
    schema_queries = build_schema_queries(schema_config)

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            for description, query in schema_queries:
                if verbose:
                    print(f"  Creating: {description}...")
                try:
                    session.run(query)
                    if verbose:
                        print("    Done")
                except Exception as e:
                    if verbose:
                        print(f"    Note: {e}")

    if verbose:
        print("\nSchema setup complete.")


# =============================================================================
# INDEX STATUS MONITORING
# =============================================================================

def get_index_status(config: Config, index_name: str) -> Optional[str]:
    """Get the current status of a Neo4j index.

    Tutorial: Index States
    ----------------------

    Neo4j indexes go through several states:

    - POPULATING: Index is being built (reading existing data)
    - ONLINE: Index is ready for queries
    - FAILED: Index creation failed (check logs)

    Vector indexes, especially on large datasets, can take minutes to
    populate. This function lets you check the current state.

    Args:
        config: Neo4j connection configuration
        index_name: Name of the index to check

    Returns:
        Index state string, or None if index not found

    Example:
        >>> status = get_index_status(config, "removal_embeddings")
        >>> if status == "ONLINE":
        ...     print("Index is ready for queries!")
    """
    query = """
        SHOW INDEXES YIELD name, state
        WHERE name = $index_name
        RETURN state
    """

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            result = session.run(query, index_name=index_name)
            record = result.single(strict=False)
            return record["state"] if record else None


def wait_for_vector_index(
    config: Config,
    index_name: str,
    timeout_seconds: int = 300,
    poll_interval: int = 5,
    verbose: bool = True,
) -> bool:
    """Wait for a vector index to reach ONLINE status.

    Tutorial: Why Wait for Index?
    -----------------------------

    After creating a vector index, Neo4j needs to:
    1. Scan all existing nodes with the indexed property
    2. Build the HNSW graph structure for approximate nearest neighbor search

    During this time, vector similarity queries will either fail or return
    incomplete results. This function polls the index status until it's ONLINE.

    Timeout Considerations:
    - Small datasets (<10k nodes): Usually < 30 seconds
    - Medium datasets (10k-100k): 1-5 minutes
    - Large datasets (100k+): May need longer timeout

    Args:
        config: Neo4j connection configuration
        index_name: Name of the vector index
        timeout_seconds: Maximum time to wait (default 5 minutes)
        poll_interval: Seconds between status checks (default 5)
        verbose: If True, print status updates

    Returns:
        True if index is ONLINE, False if timeout

    Example:
        >>> setup_neo4j_schema(config, schema_config)
        >>> if wait_for_vector_index(config, "my_embeddings", timeout_seconds=600):
        ...     print("Ready for similarity search!")
        ... else:
        ...     print("Index still building - try again later")
    """
    if verbose:
        print(f"\nWaiting for vector index '{index_name}' to come online...")

    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        status = get_index_status(config, index_name)

        if status is None:
            if verbose:
                print("  Index not found yet, waiting...")
        elif status == "ONLINE":
            elapsed = time.time() - start_time
            if verbose:
                print(f"  Index is ONLINE (took {elapsed:.1f}s)")
            return True
        else:
            if verbose:
                print(f"  Index state: {status}, waiting...")

        time.sleep(poll_interval)

    if verbose:
        print(f"  Timeout waiting for index after {timeout_seconds}s")
    return False


# =============================================================================
# CLEANUP UTILITIES
# =============================================================================

def delete_nodes_by_label(
    config: Config,
    node_label: str,
    batch_size: int = 10000,
    verbose: bool = True,
) -> int:
    """Delete all nodes with a specific label in batches.

    Tutorial: Safe Bulk Deletion
    ----------------------------

    Deleting millions of nodes in a single transaction can:
    - Cause memory issues (all changes held in memory until commit)
    - Lock the database for extended periods
    - Time out

    This function deletes in batches to avoid these issues:

        MATCH (n:MyLabel)
        WITH n LIMIT 10000
        DELETE n
        RETURN count(*) AS deleted

    The loop continues until a batch returns 0 deleted nodes.

    Args:
        config: Neo4j connection configuration
        node_label: Label of nodes to delete
        batch_size: Number of nodes per deletion batch
        verbose: If True, print progress

    Returns:
        Total number of nodes deleted

    Example:
        >>> # Clean up test data before a fresh run
        >>> deleted = delete_nodes_by_label(config, "RemovalEventTest")
        >>> print(f"Cleaned up {deleted:,} test nodes")
    """
    if verbose:
        print(f"\nDeleting {node_label} nodes in batches of {batch_size:,}...")

    total_deleted = 0

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            while True:
                result = session.run(f"""
                    MATCH (n:{node_label})
                    WITH n LIMIT {batch_size}
                    DELETE n
                    RETURN count(*) AS deleted
                """)
                deleted = result.single()["deleted"]
                total_deleted += deleted

                if deleted == 0:
                    break

                if verbose:
                    print(f"  Deleted {total_deleted:,} nodes...", flush=True)

    if verbose:
        print(f"  Cleanup complete: {total_deleted:,} nodes deleted")

    return total_deleted
