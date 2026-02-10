"""DataFrame context for managing a registry of Polars DataFrames with SQL support."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Self

import polars as pl
from pydantic import validate_call

from dfkit.identifier import DataFrameId


class DataFrameContext:
    """A registry of Polars DataFrames with SQL query support.

    Manages DataFrames by dataframe_id and provides access via a Polars SQLContext
    for SQL queries. When DataFrames are registered or unregistered, both the
    internal mapping and the underlying SQLContext are updated.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        Initialize context and register a DataFrame
        >>> ctx = DataFrameContext()
        >>> ctx.register("df_00000001", df)
        DataFrameContext(dataframes=['df_00000001'])

        List registered DataFrames
        >>> ctx.dataframe_ids
        ('df_00000001',)

        Number of registered DataFrames
        >>> len(ctx)
        1

        Register another DataFrame
        >>> df2 = pl.DataFrame({"a": [1, 1, 2], "c": ["apple", "banana", "cherry"]})
        >>> ctx.register("df_00000002", df2)
        DataFrameContext(dataframes=['df_00000001', 'df_00000002'])

        Execute a SQL query against the registered DataFrame
        >>> result = ctx.execute_sql(
        ...     "SELECT df_00000001.a"
        ...     " FROM df_00000001 JOIN df_00000002 ON df_00000001.a = df_00000002.a"
        ...     " WHERE c = 'banana'"
        ... )

        Get a registered DataFrame by its identifier
        >>> retrieved_df = ctx.get_dataframe("df_00000001")

        Unregister a DataFrame
        >>> ctx.unregister("df_00000001")
        DataFrameContext(dataframes=['df_00000002'])
        >>> len(ctx)
        1

        Clear all registered DataFrames
        >>> ctx.clear()
        DataFrameContext(dataframes=[])
        >>> len(ctx)
        0
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(self, dataframes: Mapping[DataFrameId, pl.DataFrame | pl.LazyFrame] | None = None) -> None:
        """Initialize the DataFrameContext.

        Args:
            dataframes (Mapping[DataFrameId, pl.DataFrame | pl.LazyFrame] | None): Optional mapping of identifiers to
                DataFrames to register on initialization. Defaults to None.
        """
        # Initialize internal mapping of registered DataFrames
        self._dataframes: dict[DataFrameId, pl.DataFrame | pl.LazyFrame] = {}
        self._sql_context = pl.SQLContext()

        self.register_many(dataframes or {})

    def __len__(self) -> int:
        """Return the number of registered DataFrames.

        Returns:
            int: The count of registered DataFrames and LazyFrames.
        """
        return len(self._dataframes)

    def __contains__(self, dataframe_id: DataFrameId) -> bool:
        """Check if a DataFrame is registered.

        Args:
            dataframe_id (DataFrameId): The identifier to check.

        Returns:
            bool: True if the DataFrame is registered, False otherwise.
        """
        return dataframe_id in self._dataframes

    def __repr__(self) -> str:
        """Return repr(self).

        Returns:
            str: String representation of the DataFrameContext.
        """
        dataframe_ids = ", ".join(f"'{dataframe_id}'" for dataframe_id in self._dataframes)
        return f"DataFrameContext(dataframes=[{dataframe_ids}])"

    @property
    def dataframe_ids(self) -> tuple[DataFrameId, ...]:
        """tuple[DataFrameId, ...]: Identifiers of all registered DataFrames."""
        return tuple(self._dataframes.keys())

    def execute_sql(self, query: str, *, eager: bool | None = None) -> pl.DataFrame | pl.LazyFrame:
        """Execute a SQL query against the registered DataFrames.

        Args:
            query (str): The SQL query to execute.
            eager (bool | None): Whether to return an eager DataFrame (True) or LazyFrame (False).
                If None, defaults to eager if any registered DataFrames are eager. Defaults to None.

        Returns:
            pl.DataFrame | pl.LazyFrame: The result of the SQL query as a Polars DataFrame or LazyFrame.

        Raises:
            ValueError: If the query is empty or contains only whitespace or if no DataFrames are registered.
        """
        # Validate non-empty query
        if not query or not query.strip():
            msg = "SQL query cannot be empty or whitespace-only"
            raise ValueError(msg)

        # Validate non-empty registry
        if not self._dataframes:
            msg = "Cannot execute SQL query: no DataFrames are registered in the context"
            raise ValueError(msg)

        return self._sql_context.execute(query, eager=eager)

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_dataframe(self, dataframe_id: DataFrameId) -> pl.DataFrame | pl.LazyFrame:
        """Get a registered DataFrame by its identifier.

        Args:
            dataframe_id (DataFrameId): The identifier of the registered DataFrame.

        Returns:
            pl.DataFrame | pl.LazyFrame: The registered DataFrame or LazyFrame.

        Raises:
            KeyError: If the dataframe_id is not registered.
        """
        if dataframe_id not in self._dataframes:
            msg = f"Frame '{dataframe_id}' is not registered"
            raise KeyError(msg)

        return self._dataframes[dataframe_id]

    @validate_call(config={"arbitrary_types_allowed": True})
    def register(self, dataframe_id: DataFrameId, dataframe: pl.DataFrame | pl.LazyFrame) -> Self:
        """Register a DataFrame or LazyFrame with the given dataframe_id.

        Args:
            dataframe_id (DataFrameId): The identifier to register the DataFrame under.
            dataframe (pl.DataFrame | pl.LazyFrame): The DataFrame or LazyFrame to register.

        Returns:
            Self: Self for method chaining.

        Raises:
            TypeError: If the DataFrame is not a Polars DataFrame or LazyFrame.
            ValueError: If the identifier is already registered.
        """
        # Validate dataframe type
        if not isinstance(dataframe, (pl.DataFrame, pl.LazyFrame)):
            msg = f"DataFrame must be a Polars DataFrame or LazyFrame, got {type(dataframe).__name__}"
            raise TypeError(msg)

        # Check for existing registration
        if dataframe_id in self._dataframes:
            msg = f"DataFrame '{dataframe_id}' is already registered"
            raise ValueError(msg)

        # Register in internal mapping and SQL context
        self._dataframes[dataframe_id] = dataframe
        self._sql_context.register(dataframe_id, dataframe)

        return self

    @validate_call(config={"arbitrary_types_allowed": True})
    def register_many(self, dataframes: Mapping[DataFrameId, pl.DataFrame | pl.LazyFrame]) -> Self:
        """Register multiple DataFrames or LazyFrames.

        Args:
            dataframes (Mapping[DataFrameId, pl.DataFrame | pl.LazyFrame]): Mapping of dataframe_ids to DataFrames to
                register with the context.

        Returns:
            Self: Self for method chaining.

        Raises:
            ValueError: If any dataframe_id is already registered.
        """
        # Pre-validate all dataframe_ids do not already exist
        # Do this before registering any to maintain atomicity
        for dataframe_id in dataframes:
            if dataframe_id in self._dataframes:
                msg = f"Frame '{dataframe_id}' is already registered"
                raise ValueError(msg)

        # Register each frame
        for dataframe_id, frame in dataframes.items():
            self.register(dataframe_id, frame)

        return self

    @validate_call(config={"arbitrary_types_allowed": True})
    def unregister(self, dataframe_ids: str | Collection[str]) -> Self:
        """Unregister a DataFrames by dataframe_id.

        Args:
            dataframe_ids (str | Collection[str]): The dataframe_id or dataframe_ids of the DataFrames to unregister.

        Returns:
            Self: Self for method chaining.

        Raises:
            KeyError: If the dataframe_id is not registered.
        """
        # Convert single dataframe_id to list
        if isinstance(dataframe_ids, str):
            dataframe_ids = [dataframe_ids]

        # Unregister each specified frame
        for dataframe_id in dataframe_ids:
            # Verify registration exists
            if dataframe_id not in self._dataframes:
                msg = f"Frame '{dataframe_id}' is not registered"
                raise KeyError(msg)

            # Unregister from SQL context
            self._sql_context.unregister(dataframe_id)

            # Remove from internal mapping
            self._dataframes.pop(dataframe_id)

        return self

    def clear(self) -> Self:
        """Unregister all DataFrames.

        Returns:
            Self: Self for method chaining.
        """
        # Unregister from SQL context first
        for dataframe_id in self._dataframes:
            self._sql_context.unregister(dataframe_id)

        # Clear internal mapping
        self._dataframes.clear()

        return self
