"""Neo4j database connection class for Databricks notebooks.

This module provides a reusable Neo4j connection class that handles
session management and transaction execution.

Usage:
    neo4j_conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="password")
    result = neo4j_conn.read("MATCH (n) RETURN count(n) AS count", db="neo4j")
    neo4j_conn.close()
"""

from typing import Any, Optional
from neo4j import GraphDatabase


class Neo4jConnection:
    """A wrapper class for Neo4j database connections.

    Provides read and write transaction methods with automatic session management.

    Attributes:
        uri: Neo4j connection URI (e.g., "neo4j+s://xxx.databases.neo4j.io")
        user: Neo4j username
        pwd: Neo4j password
    """

    def __init__(self, uri: str, user: str, pwd: str) -> None:
        """Initialize the Neo4j connection.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            pwd: Neo4j password
        """
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(
                self.__uri,
                auth=(self.__user, self.__pwd),
                max_connection_lifetime=200,
                keep_alive=True
            )
        except Exception as e:
            print(f"Failed to create the driver: {e}")

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.__driver is not None:
            self.__driver.close()

    def write(
        self,
        query: str,
        parameters: Optional[dict] = None,
        db: Optional[str] = None
    ) -> Optional[tuple[list[dict], Any]]:
        """Execute a write transaction.

        Args:
            query: Cypher query to execute
            parameters: Optional query parameters
            db: Optional database name (defaults to default database)

        Returns:
            Tuple of (results list, query summary) or None on error
        """
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = session.execute_write(self._run_transaction, query, parameters)
        except Exception as e:
            print(f"Query failed: {e}")
        finally:
            if session is not None:
                session.close()
        return response

    def read(
        self,
        query: str,
        parameters: Optional[dict] = None,
        db: Optional[str] = None
    ) -> Optional[tuple[list[dict], Any]]:
        """Execute a read transaction.

        Args:
            query: Cypher query to execute
            parameters: Optional query parameters
            db: Optional database name (defaults to default database)

        Returns:
            Tuple of (results list, query summary) or None on error
        """
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = session.execute_read(self._run_transaction, query, parameters)
        except Exception as e:
            print(f"Query failed: {e}")
        finally:
            if session is not None:
                session.close()
        return response

    @staticmethod
    def _run_transaction(tx, query: str, parameters: Optional[dict]) -> tuple[list[dict], Any]:
        """Execute a query within a transaction.

        Args:
            tx: Neo4j transaction object
            query: Cypher query to execute
            parameters: Optional query parameters

        Returns:
            Tuple of (results as list of dicts, query summary)
        """
        result = tx.run(query, parameters)
        values = [dict(record) for record in result]
        summary = result.consume()
        return values, summary
