"""Context API for tool modules to access DataFrame registry state.

This module provides ToolModuleContext, a narrow API that modules use to interact
with the DataFrame registry without exposing internal implementation details.

The context provides four core capabilities:
1. Viewing all registered DataFrame references
2. Getting DataFrames or DataFrameReferences by identifier (name or ID)
3. Registering new DataFrames with validation

Module authors receive a ToolModuleContext instance but never construct it directly.
The toolkit creates the context and passes it to modules at initialization time.

Examples:
    A tool module using ToolModuleContext to access DataFrames:

    >>> import polars as pl
    >>> from dfkit.toolkit import DataFrameToolkit
    >>> from dfkit.tool_module_context import ToolModuleContext
    >>> from dfkit.models import ToolCallError
    >>> from langchain_core.tools import BaseTool, tool
    >>> class MyModule:
    ...     def __init__(self, context: ToolModuleContext) -> None:
    ...         self._context = context
    ...
    ...     @property
    ...     def system_prompt(self) -> str:
    ...         return "My custom module for DataFrame operations."
    ...
    ...     def get_tools(self) -> list[BaseTool]:
    ...         @tool
    ...         def my_tool(dataframe_name: str) -> str:
    ...             '''Get row count for a DataFrame.'''
    ...             df_or_error = self._context.get_dataframe(dataframe_name)
    ...             if isinstance(df_or_error, ToolCallError):
    ...                 return df_or_error.message
    ...             return f"DataFrame has {len(df_or_error)} rows"
    ...         return [my_tool]
    >>> toolkit = DataFrameToolkit()
    >>> df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
    >>> ref = toolkit.register_dataframe("sales", df)
    >>> tools = toolkit.get_tools(MyModule)
    >>> len(tools) >= 6  # 5 core tools + 1 custom tool
    True
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
from loguru import logger

from dfkit.identifier import DataFrameId
from dfkit.models import DataFrameReference, ToolCallError
from dfkit.registry import DataFrameRegistry


class ToolModuleContext:
    """Context API for tool modules to access DataFrame registry state.

    Provides a narrow interface for modules to:
    - View all registered DataFrame references
    - Get DataFrames or DataFrameReferences by identifier (name or ID)
    - Register new DataFrames with validation
    """

    def __init__(
        self,
        registry: DataFrameRegistry,
        get_dataframe_fn: Callable[[str], pl.DataFrame | ToolCallError],
        get_dataframe_reference_fn: Callable[[str], DataFrameReference | ToolCallError],
        validate_dataframe_name_fn: Callable[[str], ToolCallError | None],
    ) -> None:
        """Initialize the ToolModuleContext.

        Args:
            registry (DataFrameRegistry): The registry managing DataFrame state.
            get_dataframe_fn (Callable[[str], pl.DataFrame | ToolCallError]):
                Callable that returns a DataFrame for a given identifier, or an error if not found.
            get_dataframe_reference_fn (Callable[[str], DataFrameReference | ToolCallError]):
                Callable that returns a DataFrameReference for a given identifier, or an error if not found.
            validate_dataframe_name_fn (Callable[[str], ToolCallError | None]):
                Callable that validates a result name, returning error if invalid.
        """
        self._registry = registry
        self._get_dataframe_fn = get_dataframe_fn
        self._get_dataframe_reference_fn = get_dataframe_reference_fn
        self._validate_dataframe_name_fn = validate_dataframe_name_fn

    @property
    def references(self) -> tuple[DataFrameReference, ...]:
        """All registered DataFrame references.

        Returns:
            tuple[DataFrameReference, ...]: Tuple of all registered references.
        """
        return tuple(self._registry.references.values())

    def get_dataframe(self, identifier: str) -> pl.DataFrame | ToolCallError:
        """Get a DataFrame by its identifier (name or ID).

        Accepts either a DataFrame ID (e.g., "df_abc123") or a name (e.g., "sales_data").
        LazyFrames are automatically collected to DataFrames.

        Args:
            identifier (str): The DataFrame ID or name to look up.

        Returns:
            pl.DataFrame | ToolCallError: The matching DataFrame, or an error if not found.

        Examples:
            >>> import polars as pl
            >>> from dfkit.toolkit import DataFrameToolkit
            >>> from dfkit.models import ToolCallError

            Get by name:
            >>> toolkit = DataFrameToolkit()
            >>> df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            >>> ref = toolkit.register_dataframe("sales", df)
            >>> context = toolkit._tool_module_context
            >>> result = context.get_dataframe("sales")
            >>> len(result)
            3

            Handle not found error:
            >>> result = context.get_dataframe("nonexistent")
            >>> isinstance(result, ToolCallError)
            True
        """
        return self._get_dataframe_fn(identifier)

    def get_dataframe_reference(self, identifier: str) -> DataFrameReference | ToolCallError:
        """Get a DataFrameReference by its identifier (name or ID).

        Accepts either a DataFrame ID (e.g., "df_abc123") or a name (e.g., "sales_data").

        Args:
            identifier (str): The DataFrame ID or name to look up.

        Returns:
            DataFrameReference | ToolCallError: The matching reference, or an error if not found.

        Examples:
            >>> import polars as pl
            >>> from dfkit.toolkit import DataFrameToolkit
            >>> from dfkit.models import ToolCallError

            Get by name:
            >>> toolkit = DataFrameToolkit()
            >>> df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            >>> ref = toolkit.register_dataframe("sales", df)
            >>> context = toolkit._tool_module_context
            >>> result = context.get_dataframe_reference("sales")
            >>> result.name
            'sales'

            Handle not found error:
            >>> result = context.get_dataframe_reference("nonexistent")
            >>> isinstance(result, ToolCallError)
            True
        """
        return self._get_dataframe_reference_fn(identifier)

    def register_dataframe(
        self,
        name: str,
        dataframe: pl.DataFrame,
        *,
        description: str | None = None,
        parent_ids: list[DataFrameId] | None = None,
        source_query: str | None = None,
        column_descriptions: dict[str, str] | None = None,
    ) -> DataFrameReference | ToolCallError:
        """Register a new DataFrame with validation.

        Validates the name is not a duplicate or ID-pattern, creates a DataFrameReference,
        and registers it in the registry.

        Args:
            name (str): The name for the DataFrame.
            dataframe (pl.DataFrame): The DataFrame to register.
            description (str | None): Optional description of the DataFrame. Defaults to None.
            parent_ids (list[DataFrameId] | None): Parent DataFrame IDs for lineage tracking.
                Required for derivative DataFrames. Defaults to None.
            source_query (str | None): SQL query that generated this DataFrame.
                Required for derivative DataFrames. Defaults to None.
            column_descriptions (dict[str, str] | None): Optional column descriptions.
                Defaults to None.

        Returns:
            DataFrameReference | ToolCallError: The created reference, or an error if validation fails.

        Examples:
            >>> import polars as pl
            >>> from dfkit.toolkit import DataFrameToolkit
            >>> from dfkit.models import ToolCallError

            Register a new DataFrame:
            >>> toolkit = DataFrameToolkit()
            >>> context = toolkit._tool_module_context
            >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> result = context.register_dataframe("my_data", df, description="Sample data")
            >>> isinstance(result, ToolCallError)
            False
            >>> result.name
            'my_data'

            Handle duplicate name error:
            >>> duplicate = context.register_dataframe("my_data", df)
            >>> isinstance(duplicate, ToolCallError)
            True
        """
        error = self._validate_dataframe_name_fn(name)
        if error is not None:
            logger.warning("Module DataFrame registration failed", name=name, error_type=error.error_type)
            return error

        reference = DataFrameReference.from_dataframe(
            name=name,
            dataframe=dataframe,
            description=description,
            parent_ids=parent_ids,
            source_query=source_query,
            column_descriptions=column_descriptions,
        )

        self._registry.register(reference, dataframe)

        logger.info("DataFrame registered via module", name=name, dataframe_id=reference.id, shape=dataframe.shape)

        return reference
