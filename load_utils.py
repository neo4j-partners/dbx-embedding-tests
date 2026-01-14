"""
Shared Utilities for Neo4j Data Loading
=======================================

This module provides common utilities shared across all embedding and load test scripts:
- Configuration management with Databricks Secrets
- Neo4j connection handling with context managers
- Connection testing and validation
- Output formatting utilities

TUTORIAL: Connecting Databricks to Neo4j
----------------------------------------

**Step 1: Store Credentials in Databricks Secrets**

    # Create a secret scope (one-time setup)
    databricks secrets create-scope my-neo4j-secrets

    # Add your Neo4j credentials
    databricks secrets put-secret my-neo4j-secrets host
    databricks secrets put-secret my-neo4j-secrets username
    databricks secrets put-secret my-neo4j-secrets password

**Step 2: Load Configuration in Your Script**

    from load_utils import load_config, test_neo4j_connection

    config = load_config(dbutils, "my-neo4j-secrets")
    if not test_neo4j_connection(config):
        raise Exception("Cannot connect to Neo4j!")

**Step 3: Use the Configuration**

    from load_utils import neo4j_driver

    with neo4j_driver(config) as driver:
        with driver.session(database=config.database) as session:
            result = session.run("MATCH (n) RETURN count(n)")
            print(result.single()[0])

Neo4j Aura Connection Tips:
---------------------------
- Protocol: Use "neo4j+s" for TLS-encrypted connections (required for Aura)
- Database: Use "neo4j" for single-database instances
- Paused Instances: Aura instances pause after inactivity; wait 1-2 minutes after resume
- Allowlisting: Ensure Databricks IP ranges are allowlisted in Aura

Usage:
    from load_utils import (
        Config,
        get_secret_with_default,
        load_config,
        print_config,
        neo4j_driver,
        test_neo4j_connection,
        print_section_header,
    )

References:
    - https://neo4j.com/docs/python-manual/current/
    - https://docs.databricks.com/en/security/secrets/
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

from neo4j import Driver, GraphDatabase


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for Neo4j connection and embedding endpoint.

    Tutorial: Configuration as a Dataclass
    ---------------------------------------

    Using a dataclass for configuration provides:
    - **Type Safety**: IDE autocomplete and type checking
    - **Immutability**: Reduces accidental modifications
    - **Documentation**: Self-documenting field names
    - **Validation**: Can add __post_init__ for validation

    The `uri` property combines protocol and host into the connection string
    that Neo4j drivers expect.

    Attributes:
        host: Neo4j server hostname (e.g., "xxx.databases.neo4j.io")
        username: Neo4j username (usually "neo4j" for Aura)
        password: Neo4j password
        database: Database name (usually "neo4j" for single-database)
        protocol: Connection protocol ("neo4j+s" for TLS, "neo4j" for plain)
        embedding_endpoint: Databricks Model Serving endpoint for embeddings

    Example:
        >>> config = Config(
        ...     host="abc123.databases.neo4j.io",
        ...     username="neo4j",
        ...     password="secret",
        ...     database="neo4j",
        ...     protocol="neo4j+s",
        ...     embedding_endpoint="my-embedding-model",
        ... )
        >>> print(config.uri)
        neo4j+s://abc123.databases.neo4j.io
    """

    host: str
    username: str
    password: str
    database: str
    protocol: str
    embedding_endpoint: str

    @property
    def uri(self) -> str:
        """Construct Neo4j connection URI from protocol and host.

        The URI format is: {protocol}://{host}

        For Neo4j Aura, use "neo4j+s" protocol for TLS encryption.
        For local/self-hosted, use "neo4j" (plain) or "neo4j+s" (TLS).

        Returns:
            Complete Neo4j connection URI
        """
        return f"{self.protocol}://{self.host}"


def get_secret_with_default(
    dbutils,
    scope: str,
    key: str,
    default: str,
) -> str:
    """Retrieve a secret from Databricks, falling back to default if not found.

    Tutorial: Safe Secret Access
    ----------------------------

    Databricks Secrets raise an exception if a key doesn't exist. This wrapper
    provides a safe way to access optional secrets with defaults:

        # Required secret (will raise if missing)
        password = dbutils.secrets.get(scope="my-scope", key="password")

        # Optional secret with default (won't raise)
        database = get_secret_with_default(dbutils, "my-scope", "database", "neo4j")

    Args:
        dbutils: Databricks dbutils object (available in notebooks)
        scope: Databricks secret scope name
        key: Secret key within the scope
        default: Value to return if secret doesn't exist

    Returns:
        Secret value or default

    Example:
        >>> # In a Databricks notebook:
        >>> database = get_secret_with_default(dbutils, "neo4j-secrets", "database", "neo4j")
    """
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return default


def load_config(
    dbutils,
    scope_name: str,
    default_database: str = "neo4j",
    default_protocol: str = "neo4j+s",
    default_embedding_endpoint: str = "airline_embedding_test",
) -> Config:
    """Load configuration from Databricks Secrets.

    Tutorial: Loading Neo4j Configuration
    -------------------------------------

    This function loads Neo4j connection details from Databricks Secrets.
    Secrets are the recommended way to store credentials in Databricks
    because they are encrypted and access-controlled.

    **Required Secrets** (must exist in the scope):
        - host: Neo4j hostname (e.g., "xxx.databases.neo4j.io")
        - username: Neo4j username
        - password: Neo4j password

    **Optional Secrets** (have defaults):
        - database: Database name (default: "neo4j")
        - protocol: Connection protocol (default: "neo4j+s")

    The embedding_endpoint is passed as a parameter, not loaded from secrets,
    because each script may use a different endpoint.

    Args:
        dbutils: Databricks dbutils object
        scope_name: Name of the Databricks secret scope
        default_database: Default database if not in secrets
        default_protocol: Default protocol if not in secrets
        default_embedding_endpoint: Default embedding endpoint

    Returns:
        Populated Config object

    Raises:
        Exception: If required secrets (host, username, password) are missing

    Example:
        >>> config = load_config(dbutils, "my-neo4j-secrets")
        >>> print(f"Connecting to {config.uri}")
    """
    return Config(
        host=dbutils.secrets.get(scope=scope_name, key="host"),
        username=dbutils.secrets.get(scope=scope_name, key="username"),
        password=dbutils.secrets.get(scope=scope_name, key="password"),
        database=get_secret_with_default(dbutils, scope_name, "database", default_database),
        protocol=get_secret_with_default(dbutils, scope_name, "protocol", default_protocol),
        embedding_endpoint=default_embedding_endpoint,
    )


def print_config(
    config: Config,
    scope_name: str,
    embedding_dimensions: int = 384,
    batch_size: int = 5000,
) -> None:
    """Display configuration for verification (excludes sensitive values).

    Tutorial: Safe Logging of Configuration
    ---------------------------------------

    When debugging, it's helpful to print configuration. But NEVER print
    passwords or sensitive values!

    This function prints:
    - Secret scope name (safe)
    - Neo4j URI (safe - no credentials)
    - Database name (safe)
    - Username (safe - just the user, not the password)
    - Embedding settings (safe)

    It does NOT print:
    - Password
    - Full connection strings with credentials

    Args:
        config: Configuration object to display
        scope_name: Secret scope name for display
        embedding_dimensions: Embedding vector size for display
        batch_size: Batch size for display
    """
    print("\nConfiguration:")
    print(f"  Secret Scope: {scope_name}")
    print(f"  Neo4j URI: {config.uri}")
    print(f"  Database: {config.database}")
    print(f"  Username: {config.username}")
    print(f"  Embedding Endpoint: {config.embedding_endpoint}")
    print(f"  Embedding Dimensions: {embedding_dimensions}")
    print(f"  Batch Size: ~{batch_size:,} rows")


# =============================================================================
# NEO4J CONNECTION MANAGEMENT
# =============================================================================

@contextmanager
def neo4j_driver(config: Config) -> Generator[Driver, None, None]:
    """Context manager for Neo4j driver lifecycle.

    Tutorial: Managing Neo4j Connections
    ------------------------------------

    The Neo4j Python driver maintains a connection pool. It's important to:
    1. Create one driver per application (not per query)
    2. Close the driver when done (releases connections)

    This context manager handles proper cleanup:

        with neo4j_driver(config) as driver:
            with driver.session() as session:
                result = session.run("RETURN 1")

        # Driver is automatically closed here, even if an error occurred

    **Driver Options:**
    - max_connection_lifetime: How long connections stay in the pool (seconds)
    - keep_alive: Send periodic pings to prevent connection drops
    - max_connection_pool_size: Maximum concurrent connections (default 100)

    Args:
        config: Configuration with Neo4j connection details

    Yields:
        Neo4j driver instance

    Example:
        >>> with neo4j_driver(config) as driver:
        ...     with driver.session(database="neo4j") as session:
        ...         result = session.run("MATCH (n) RETURN count(n) AS count")
        ...         print(result.single()["count"])
    """
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.username, config.password),
        max_connection_lifetime=200,  # Seconds before connection is recycled
        keep_alive=True,  # Prevent idle connection drops
    )
    try:
        yield driver
    finally:
        driver.close()


# =============================================================================
# CONNECTION TESTING
# =============================================================================

def test_neo4j_connection(config: Config) -> bool:
    """Test Neo4j connection before attempting operations.

    Tutorial: Pre-Flight Connection Checks
    --------------------------------------

    Before running a long ETL job, it's crucial to verify Neo4j connectivity.
    A failed connection halfway through wastes time and leaves partial data.

    This function performs three tests:

    1. **Basic Connectivity**: Can we connect and run a query?
       - Tests network connectivity
       - Verifies credentials are correct
       - Confirms the database exists

    2. **Version Check**: What Neo4j version is running?
       - Useful for debugging
       - Ensures compatibility (need 4.x+ for vector search)

    3. **Write Permission**: Can we create and delete nodes?
       - Some accounts may have read-only access
       - We create and immediately delete a test node

    **Common Failure Causes:**
    - Neo4j Aura instance is paused (resume and wait 1-2 minutes)
    - Wrong credentials in Databricks Secrets
    - Databricks IP not allowlisted in Neo4j Aura
    - Network connectivity issues

    Args:
        config: Neo4j configuration

    Returns:
        True if all tests pass, False otherwise

    Example:
        >>> if not test_neo4j_connection(config):
        ...     print("Cannot connect to Neo4j! Aborting.")
        ...     return
    """
    print_section_header("TESTING NEO4J CONNECTION")
    print(f"  URI: {config.uri}")
    print(f"  Database: {config.database}")

    try:
        with neo4j_driver(config) as driver:
            # Test 1: Basic connectivity with simple query
            with driver.session(database=config.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record["test"] != 1:
                    print("  ERROR: Basic query returned unexpected result")
                    return False
                print("  [OK] Basic connectivity")

            # Test 2: Get Neo4j version info
            with driver.session(database=config.database) as session:
                result = session.run(
                    "CALL dbms.components() YIELD name, versions RETURN name, versions"
                )
                record = result.single()
                print(f"  [OK] Connected to {record['name']} {record['versions'][0]}")

            # Test 3: Verify write permissions
            with driver.session(database=config.database) as session:
                # Create and immediately delete a test node
                session.run(
                    "CREATE (t:_ConnectionTest {ts: $ts}) "
                    "WITH t DELETE t",
                    ts=time.time(),
                )
                print("  [OK] Write operations permitted")

        print("\n  Connection test PASSED")
        return True

    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        print("\n  Troubleshooting tips:")
        print("    - Check if Neo4j Aura instance is running (not paused)")
        print("    - Verify credentials in Databricks secrets")
        print("    - Ensure network access from Databricks to Neo4j")
        print("    - Wait 1-2 minutes if instance was just resumed")
        return False


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_section_header(title: str, width: int = 70) -> None:
    """Print a formatted section header for console output.

    Tutorial: Clear Console Output
    ------------------------------

    When running long jobs, clear section headers help track progress
    in the logs. This creates headers like:

        ======================================================================
        SETTING UP NEO4J SCHEMA
        ======================================================================

    Args:
        title: Header text to display
        width: Width of the separator line (default 70)

    Example:
        >>> print_section_header("PROCESSING DATA")
        ======================================================================
        PROCESSING DATA
        ======================================================================
    """
    print(f"\n{'=' * width}")
    print(title)
    print("=" * width)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1.5s" or "2.5 minutes"

    Example:
        >>> format_duration(90)
        '90.0s (1.5 minutes)'
        >>> format_duration(30)
        '30.0s'
    """
    if seconds >= 60:
        return f"{seconds:.1f}s ({seconds / 60:.1f} minutes)"
    return f"{seconds:.1f}s"


def format_rate(count: int, seconds: float) -> str:
    """Format a throughput rate.

    Args:
        count: Number of items processed
        seconds: Time taken in seconds

    Returns:
        Formatted string like "1,234.5 rows/s"

    Example:
        >>> format_rate(10000, 5)
        '2,000.0 rows/s'
    """
    if seconds > 0:
        rate = count / seconds
        return f"{rate:,.1f} rows/s"
    return "N/A"
