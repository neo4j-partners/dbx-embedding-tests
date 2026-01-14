"""
Embedding Providers for Databricks Spark
========================================

This module provides abstraction for different embedding generation methods
in Databricks Spark. It supports three approaches:

1. **Random Embeddings**: For testing Neo4j write performance in isolation
2. **Custom Model Endpoints**: Your own models deployed to Databricks Model Serving
3. **Databricks Hosted Models**: Pre-built foundation models (BGE, GTE)

TUTORIAL: Embedding Generation in Databricks
--------------------------------------------

Databricks provides several ways to generate embeddings:

**Option 1: ai_query() SQL Function (Recommended)**
    The `ai_query()` function is a native Spark SQL function that calls
    AI endpoints directly from DataFrame operations:

        df.withColumn("embedding", expr("ai_query('my-endpoint', text_col)"))

    Advantages:
    - Runs in Spark's execution engine (no UDF serialization overhead)
    - Automatic batching handled by the function
    - Works with both custom and hosted models

    Disadvantage:
    - Limited control over batch size per API call

**Option 2: pandas_udf with MLflow Client**
    Use a pandas UDF to call the MLflow deployment client:

        @pandas_udf(ArrayType(FloatType()))
        def embed(texts: pd.Series) -> pd.Series:
            client = mlflow.deployments.get_deploy_client("databricks")
            ...

    Advantage: Full control over batching
    Disadvantage: UDF serialization overhead

**Option 3: Random Embeddings (Testing)**
    For testing Neo4j write performance without embedding overhead:

        df.withColumn("embedding", array(*[rand() for _ in range(384)]))

    Use this to establish a baseline for Neo4j write throughput.

API Format Differences
----------------------

**Custom Model Endpoints** (e.g., MiniLM deployed to Model Serving):
    Input:  {"dataframe_records": [{"text": "..."}]}
    Output: {"predictions": [[0.1, 0.2, ...]]}

**Databricks Hosted Models** (BGE, GTE - OpenAI-compatible):
    Input:  {"input": ["..."]}
    Output: {"data": [{"embedding": [0.1, 0.2, ...]}]}

This module handles both formats transparently.

Deep Dive: How ai_query() Works
-------------------------------

The `ai_query()` function is a built-in Spark SQL function that calls AI model
endpoints directly from SQL expressions. It's the recommended way to generate
embeddings in Databricks because it integrates natively with Spark's execution
engine.

**Basic Syntax:**

    ai_query(endpoint_name, request [, returnType])

    - endpoint_name: String name of the Model Serving endpoint
    - request: The input data (column reference or literal)
    - returnType: Optional schema for the response (auto-inferred if omitted)

**How It Works Internally:**

1. **Request Batching**: ai_query automatically batches multiple rows into
   single API calls for efficiency. You don't control the batch size directly.

2. **Execution**: The function runs within Spark's execution engine, avoiding
   the serialization overhead of Python UDFs.

3. **Response Parsing**: For embedding models, ai_query returns the embedding
   array directly. Databricks handles format differences between custom and
   hosted models internally.

4. **Error Handling**: Failed requests return NULL. Always filter for valid
   embeddings after calling ai_query.

**Example Usage in DataFrames:**

    from pyspark.sql.functions import expr

    # Add embedding column using ai_query
    df_with_embeddings = df.withColumn(
        "embedding",
        expr("ai_query('databricks-bge-large-en', text_column)")
    )

    # With explicit return type (optional)
    df_with_embeddings = df.withColumn(
        "embedding",
        expr("ai_query('my-endpoint', text_column, 'ARRAY<FLOAT>')")
    )

**How This File Uses ai_query:**

This module wraps ai_query in embedding provider classes:

    # In CustomModelEmbeddingProvider.add_embeddings():
    ai_query_expr = f"ai_query('{self.config.endpoint_name}', {self.config.text_column})"
    return df.withColumn(self.config.output_column, expr(ai_query_expr))

The provider classes add:
- Configuration management (endpoint name, dimensions)
- Validation (test endpoint before batch processing)
- Filtering (remove rows with NULL or wrong-dimension embeddings)

**Performance Considerations:**

- ai_query is faster than pandas_udf for most workloads
- The function handles retries for transient failures
- Large batches may hit rate limits; use streaming with small micro-batches
- Monitor Model Serving endpoint metrics for throughput bottlenecks

**Comparison: ai_query vs MLflow Client:**

| Aspect           | ai_query()              | MLflow Client (in UDF)  |
|------------------|-------------------------|-------------------------|
| Batching         | Automatic               | Manual control          |
| Overhead         | Low (native Spark)      | Higher (serialization)  |
| Error handling   | Returns NULL            | Raises exceptions       |
| Flexibility      | Limited                 | Full control            |
| Best for         | Batch processing        | Custom retry logic      |

This module uses ai_query for DataFrame operations and the MLflow client
directly for single-query embedding generation (see `generate_query_embedding`).

References:
    - https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query
    - https://docs.databricks.com/aws/en/machine-learning/model-serving/query-embedding-models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import array, col, expr, rand, size
from pyspark.sql.types import ArrayType, FloatType


# =============================================================================
# EMBEDDING PROVIDER CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for an embedding provider.

    Tutorial: Embedding Model Parameters
    ------------------------------------

    When working with embeddings, you need to configure:

    - **endpoint_name**: The Databricks Model Serving endpoint
      - Custom models: Your deployed endpoint name (e.g., "my-minilm-endpoint")
      - Hosted models: The foundation model name (e.g., "databricks-bge-large-en")

    - **dimensions**: Number of floats in each embedding vector
      - MiniLM: 384 dimensions
      - BGE-Large: 1024 dimensions
      - GTE-Large: 1024 dimensions
      - OpenAI ada-002: 1536 dimensions

    - **text_column**: The DataFrame column containing text to embed

    - **output_column**: Where to store the embedding (usually "embedding")

    Dimension Matching:
    ------------------
    The embedding dimensions MUST match between:
    1. Your model's output
    2. Neo4j vector index configuration
    3. Filtering logic in your pipeline

    Mismatched dimensions will cause silent failures or errors.

    Attributes:
        endpoint_name: Databricks Model Serving endpoint
        dimensions: Size of embedding vectors
        text_column: Source column with text to embed
        output_column: Target column for embeddings
    """

    endpoint_name: str
    dimensions: int
    text_column: str = "text"
    output_column: str = "embedding"


# =============================================================================
# BASE EMBEDDING PROVIDER
# =============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding generation.

    Tutorial: Strategy Pattern for Embeddings
    -----------------------------------------

    This uses the Strategy pattern to allow different embedding implementations
    to be swapped without changing the pipeline code:

        # Production: Use Databricks hosted model
        provider = DatabricksHostedEmbeddingProvider(config)

        # Testing: Use random embeddings
        provider = RandomEmbeddingProvider(config)

        # Both work the same way:
        df_with_embeddings = provider.add_embeddings(df)

    Each provider implements:
    - add_embeddings(): Add embedding column to DataFrame
    - validate_endpoint(): Test that the endpoint works
    - filter_valid_embeddings(): Remove rows with invalid embeddings
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding provider.

        Args:
            config: Embedding configuration
        """
        self.config = config

    @abstractmethod
    def add_embeddings(self, df: DataFrame) -> DataFrame:
        """Add embeddings to a DataFrame.

        This method should add a column named `config.output_column` containing
        the embedding vector for each row.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with embedding column added
        """
        pass

    @abstractmethod
    def validate_endpoint(self) -> bool:
        """Validate that the embedding endpoint is working correctly.

        Returns:
            True if endpoint returns valid embeddings, False otherwise
        """
        pass

    def filter_valid_embeddings(self, df: DataFrame) -> DataFrame:
        """Filter DataFrame to only rows with valid embeddings.

        Tutorial: Why Filter Embeddings?
        ---------------------------------

        Embedding generation can fail for individual rows due to:
        - Empty or null text
        - Text exceeding model's context length
        - API rate limiting or transient errors

        This method removes rows where:
        - Embedding is NULL
        - Embedding has wrong dimensions

        Args:
            df: DataFrame with embedding column

        Returns:
            DataFrame with only valid embeddings
        """
        return df.filter(
            col(self.config.output_column).isNotNull()
            & (size(col(self.config.output_column)) == self.config.dimensions)
        )


# =============================================================================
# RANDOM EMBEDDING PROVIDER (FOR TESTING)
# =============================================================================

class RandomEmbeddingProvider(EmbeddingProvider):
    """Generate random embeddings for testing Neo4j write performance.

    Tutorial: Performance Testing Without Embedding Overhead
    --------------------------------------------------------

    When testing Neo4j write performance, you often want to isolate the
    database performance from embedding generation time.

    This provider generates random float arrays as "fake" embeddings:

        df.withColumn("embedding", array(rand(), rand(), rand(), ...))

    This is MUCH faster than calling an embedding model, allowing you to:
    1. Establish baseline Neo4j write throughput
    2. Test batch size and parallelism settings
    3. Verify schema and index configuration

    Random embeddings won't produce meaningful similarity search results,
    but they're perfect for performance testing.

    Example:
        >>> config = EmbeddingConfig(
        ...     endpoint_name="unused",  # Not used for random embeddings
        ...     dimensions=384,
        ...     text_column="removal_reason",
        ... )
        >>> provider = RandomEmbeddingProvider(config)
        >>> df_with_embeddings = provider.add_embeddings(df)
    """

    def add_embeddings(self, df: DataFrame) -> DataFrame:
        """Add random embedding vectors to the DataFrame.

        Each row gets a random vector of `config.dimensions` floats.
        Values are uniformly distributed between 0 and 1.

        Note: Random vectors are NOT normalized. For production use with
        cosine similarity, you'd want unit-length vectors. But for write
        testing, this doesn't matter.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with random embedding column
        """
        # Generate array of random floats
        # Each rand() call produces a different random value per row
        random_values = [rand() for _ in range(self.config.dimensions)]

        return df.withColumn(
            self.config.output_column,
            array(*random_values)
        )

    def validate_endpoint(self) -> bool:
        """Random provider is always valid (no external dependency).

        Returns:
            Always True
        """
        print(f"  Endpoint: Random embeddings (no external service)")
        print(f"  Dimensions: {self.config.dimensions}")
        print(f"  Validation PASSED (random generator always works)")
        return True


# =============================================================================
# CUSTOM MODEL EMBEDDING PROVIDER
# =============================================================================

class CustomModelEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings using a custom model deployed to Databricks Model Serving.

    Tutorial: Custom Embedding Models in Databricks
    -----------------------------------------------

    You can deploy your own embedding models (e.g., sentence-transformers)
    to Databricks Model Serving. These models use the MLflow deployment format.

    **Deployment Steps:**
    1. Log your model with MLflow
    2. Register in Unity Catalog
    3. Create a Model Serving endpoint

    **API Format (Custom Models):**
    Custom models typically use the "dataframe_records" input format:

        Input:  {"dataframe_records": [{"text": "hello"}, {"text": "world"}]}
        Output: {"predictions": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}

    **Using ai_query():**
    The ai_query() function handles the API call:

        ai_query('my-endpoint', text_column)

    The function automatically:
    - Batches rows for efficiency
    - Handles retries on transient failures
    - Returns the embedding as an array

    Common Models and Dimensions:
    - all-MiniLM-L6-v2: 384 dimensions
    - all-mpnet-base-v2: 768 dimensions
    - e5-large-v2: 1024 dimensions

    Example:
        >>> config = EmbeddingConfig(
        ...     endpoint_name="my-minilm-endpoint",
        ...     dimensions=384,
        ...     text_column="removal_reason",
        ... )
        >>> provider = CustomModelEmbeddingProvider(config)
        >>> if provider.validate_endpoint():
        ...     df_with_embeddings = provider.add_embeddings(df)
    """

    def add_embeddings(self, df: DataFrame) -> DataFrame:
        """Add embeddings using ai_query() with custom model format.

        The ai_query() function is called for each row. Databricks handles
        batching internally for efficiency.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with embedding column
        """
        # Build ai_query expression
        # ai_query returns the raw model output, which for custom models
        # is typically the embedding array directly
        ai_query_expr = f"ai_query('{self.config.endpoint_name}', {self.config.text_column})"

        return df.withColumn(
            self.config.output_column,
            expr(ai_query_expr)
        )

    def validate_endpoint(self) -> bool:
        """Validate the custom model endpoint returns correct embeddings.

        Sends a test request to verify:
        1. Endpoint is reachable
        2. Response format is correct
        3. Embedding dimensions match configuration

        Returns:
            True if validation passes
        """
        import mlflow.deployments

        print(f"  Endpoint: {self.config.endpoint_name}", flush=True)
        print(f"  Expected dimensions: {self.config.dimensions}", flush=True)
        print(f"  API format: Custom (dataframe_records)", flush=True)

        test_text = "test embedding validation"

        try:
            print("  Calling endpoint...", flush=True)
            client = mlflow.deployments.get_deploy_client("databricks")

            # Custom model format
            response = client.predict(
                endpoint=self.config.endpoint_name,
                inputs={"dataframe_records": [{"text": test_text}]},
            )

            print(f"  Response received, type: {type(response).__name__}", flush=True)

            # Handle different response formats
            # Some models return {"predictions": [[...]]} others return {"predictions": [...]}
            if "predictions" in response:
                predictions = response["predictions"]
                print(f"  Predictions type: {type(predictions).__name__}, len: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}", flush=True)

                # Get the first prediction
                if len(predictions) > 0:
                    embedding = predictions[0]
                    # Handle nested list case
                    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                        print(f"  Note: Detected nested list, extracting inner embedding", flush=True)
                        embedding = embedding[0]
                else:
                    print(f"  ERROR: Empty predictions list", flush=True)
                    return False
            else:
                print(f"  ERROR: No 'predictions' key in response. Keys: {list(response.keys())}", flush=True)
                return False

            if not hasattr(embedding, '__len__'):
                print(f"  ERROR: Embedding is not a list, type: {type(embedding).__name__}", flush=True)
                return False

            if len(embedding) != self.config.dimensions:
                print(f"  ERROR: Dimension mismatch! Expected {self.config.dimensions}, got {len(embedding)}", flush=True)
                return False

            print(f"  Dimensions: {len(embedding)}", flush=True)
            print(f"  Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]", flush=True)
            print(f"  Validation PASSED", flush=True)
            return True

        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            print(f"  Traceback: {traceback.format_exc()}", flush=True)
            return False


# =============================================================================
# DATABRICKS HOSTED MODEL EMBEDDING PROVIDER
# =============================================================================

class DatabricksHostedEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings using Databricks Foundation Model APIs.

    Tutorial: Databricks Hosted Embedding Models
    --------------------------------------------

    Databricks provides pre-deployed embedding models as part of the
    Foundation Model APIs. These are ready to use without deployment.

    **Available Models (as of 2024):**
    - databricks-bge-large-en: 1024 dims, 512 token context
    - databricks-gte-large-en: 1024 dims, 8192 token context

    **API Format (OpenAI-Compatible):**
    Hosted models use OpenAI-compatible format:

        Input:  {"input": ["hello", "world"]}
        Output: {"data": [{"embedding": [0.1, ...]}, {"embedding": [0.3, ...]}]}

    **Key Advantages:**
    - No deployment needed
    - Managed scaling
    - Consistent availability
    - Pay-per-token pricing

    **Context Length:**
    - BGE: 512 tokens (shorter but faster)
    - GTE: 8192 tokens (longer documents, slower)

    Text exceeding context length will be truncated. Consider chunking
    long documents before embedding.

    Example:
        >>> config = EmbeddingConfig(
        ...     endpoint_name="databricks-bge-large-en",
        ...     dimensions=1024,
        ...     text_column="removal_reason",
        ... )
        >>> provider = DatabricksHostedEmbeddingProvider(config)
        >>> df_with_embeddings = provider.add_embeddings(df)
    """

    def add_embeddings(self, df: DataFrame) -> DataFrame:
        """Add embeddings using ai_query() with hosted model.

        Databricks hosted models use the same ai_query() function,
        but the internal API format is OpenAI-compatible. The ai_query()
        function handles this automatically.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with embedding column
        """
        # ai_query works the same way for hosted models
        # Databricks handles the OpenAI format internally
        ai_query_expr = f"ai_query('{self.config.endpoint_name}', {self.config.text_column})"

        return df.withColumn(
            self.config.output_column,
            expr(ai_query_expr)
        )

    def validate_endpoint(self) -> bool:
        """Validate the Databricks hosted model endpoint.

        Sends a test request using OpenAI-compatible format.

        Returns:
            True if validation passes
        """
        import mlflow.deployments

        print(f"  Endpoint: {self.config.endpoint_name}", flush=True)
        print(f"  Expected dimensions: {self.config.dimensions}", flush=True)
        print(f"  API format: OpenAI-compatible", flush=True)

        test_text = "test embedding validation"

        try:
            print("  Calling endpoint...", flush=True)
            client = mlflow.deployments.get_deploy_client("databricks")

            # OpenAI-compatible format for hosted models
            response = client.predict(
                endpoint=self.config.endpoint_name,
                inputs={"input": [test_text]},
            )

            print(f"  Response received, type: {type(response).__name__}", flush=True)

            # Extract embedding from OpenAI format response
            if "data" not in response:
                print(f"  ERROR: No 'data' key in response. Keys: {list(response.keys())}", flush=True)
                return False

            embedding = response["data"][0]["embedding"]

            if len(embedding) != self.config.dimensions:
                print(f"  ERROR: Dimension mismatch! Expected {self.config.dimensions}, got {len(embedding)}", flush=True)
                return False

            print(f"  Dimensions: {len(embedding)}", flush=True)
            print(f"  Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]", flush=True)
            print(f"  Validation PASSED", flush=True)
            return True

        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            print(f"  Traceback: {traceback.format_exc()}", flush=True)
            return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_embedding_provider(
    provider_type: str,
    config: EmbeddingConfig,
) -> EmbeddingProvider:
    """Factory function to create the appropriate embedding provider.

    Tutorial: Factory Pattern for Flexibility
    -----------------------------------------

    This factory makes it easy to switch between embedding providers
    based on configuration or runtime decisions:

        # Development/testing
        provider = create_embedding_provider("random", config)

        # Production with custom model
        provider = create_embedding_provider("custom", config)

        # Production with hosted model
        provider = create_embedding_provider("hosted", config)

    Args:
        provider_type: One of "random", "custom", or "hosted"
        config: Embedding configuration

    Returns:
        Configured embedding provider

    Raises:
        ValueError: If provider_type is not recognized

    Example:
        >>> config = EmbeddingConfig("my-endpoint", 384)
        >>> provider = create_embedding_provider("custom", config)
    """
    providers = {
        "random": RandomEmbeddingProvider,
        "custom": CustomModelEmbeddingProvider,
        "hosted": DatabricksHostedEmbeddingProvider,
    }

    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Must be one of: {list(providers.keys())}"
        )

    return providers[provider_type](config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_query_embedding(
    endpoint_name: str,
    text: str,
    api_format: str = "hosted",
) -> List[float]:
    """Generate a single embedding for a query string.

    Tutorial: Query-Time Embeddings
    -------------------------------

    When performing similarity search, you need to embed the query text
    using the SAME model that generated the stored embeddings.

    This function is for single queries (not batch processing).
    It uses the MLflow deployments client directly.

    Args:
        endpoint_name: Databricks Model Serving endpoint
        text: Query text to embed
        api_format: "hosted" (OpenAI format) or "custom" (dataframe_records)

    Returns:
        Embedding vector as a list of floats

    Example:
        >>> embedding = generate_query_embedding(
        ...     "databricks-bge-large-en",
        ...     "hydraulic pump failure",
        ...     api_format="hosted"
        ... )
        >>> # Use embedding in Neo4j vector search
        >>> session.run(
        ...     "CALL db.index.vector.queryNodes('my_index', 5, $embedding)",
        ...     embedding=embedding
        ... )
    """
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")

    if api_format == "hosted":
        # OpenAI-compatible format
        response = client.predict(
            endpoint=endpoint_name,
            inputs={"input": [text]},
        )
        return response["data"][0]["embedding"]
    else:
        # Custom model format
        response = client.predict(
            endpoint=endpoint_name,
            inputs={"dataframe_records": [{"text": text}]},
        )
        return response["predictions"][0]
