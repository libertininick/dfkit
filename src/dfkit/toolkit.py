"""DataFrame toolkit for managing DataFrames with LangChain tool integration.

This toolkit provides a high-level interface for working with DataFrames in
LLM agent contexts. It maintains a registry of DataFrames with descriptive
metadata that helps agents understand and query the data.

The toolkit uses composition to manage an internal DataFrameContext for SQL
query execution while exposing a user-friendly API based on DataFrame names
rather than internal identifiers.

Design Note:
    DataFrames are keyed internally by generated ID (e.g., "df_00000001") for
    SQL safety and consistency. User-friendly names are stored in references and
    looked up via O(n) scan, which is acceptable for typical usage (2-20 DataFrames).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import polars as pl
from langchain_core.tools import BaseTool, tool
from loguru import logger
from sqlglot import exp

from dfkit.exceptions import (
    ColumnsNotFoundError,
    DuplicateColumnsError,
    SQLBlacklistedCommandError,
    SQLColumnError,
    SQLSyntaxError,
    SQLTableError,
)
from dfkit.identifier import (
    DATAFRAME_ID_PATTERN,
    DataFrameId,
)
from dfkit.logging import TOOL_CALL_LEVEL
from dfkit.models import (
    DataFrameReference,
    DataFrameToolkitState,
    ToolCallError,
)
from dfkit.persistence import REL_TOL_DEFAULT, restore_registry_from_state
from dfkit.polars_utils import ensure_dataframe, to_markdown_table
from dfkit.registry import DataFrameRegistry
from dfkit.sql_utils import DESTRUCTIVE_COMMANDS, extract_table_names, validate_sql
from dfkit.tool_module import ToolModule
from dfkit.tool_module_context import ToolModuleContext


class DataFrameToolkit:
    """A toolkit for registering and managing Polars DataFrames for LLM tool access.

    This toolkit provides a high-level interface for working with DataFrames in
    LLM agent contexts. It maintains a registry of DataFrames with descriptive
    metadata that helps agents understand and query the data.

    Attributes:
        CORE_SYSTEM_PROMPT (str): System prompt describing core toolkit tools and workflow.

    Examples:
        >>> import polars as pl

        Register a DataFrame with metadata:
        >>> toolkit = DataFrameToolkit()
        >>> df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        >>> ref = toolkit.register_dataframe(
        ...     "sales",
        ...     df,
        ...     description="Daily sales data",
        ... )
        >>> ref.name
        'sales'

        Register multiple DataFrames at once:
        >>> toolkit = DataFrameToolkit()
        >>> df1 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        >>> df2 = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        >>> refs = toolkit.register_dataframes({"sales": df1, "products": df2})
        >>> len(refs)
        2

        Unregister a DataFrame:
        >>> toolkit.unregister_dataframe("products")

        Restore from saved state:
        >>> state = toolkit.export_state()
        >>> new_toolkit = DataFrameToolkit.from_state(state, {"sales": df1})
    """

    CORE_SYSTEM_PROMPT: str = (
        "You have access to a DataFrame toolkit with the following core "
        "tools:\n\n"
        "- **list_dataframes**: List all available DataFrames with their "
        "names, IDs, and schema information.\n"
        "- **get_dataframe_id**: Get the unique ID for a DataFrame by its "
        "human-readable name. Use IDs in SQL queries.\n"
        "- **get_dataframe_reference**: Get detailed schema information "
        "about a DataFrame by name or ID.\n"
        "- **execute_sql**: Execute a SQL SELECT query against registered "
        "DataFrames. Use DataFrame IDs as table names.\n"
        "- **view_as_markdown_table**: View a DataFrame as a formatted "
        "markdown table for data inspection.\n\n"
        "Workflow: First use list_dataframes to discover available data, "
        "then get_dataframe_id to get IDs, then execute_sql to query using "
        "those IDs."
    )

    def __init__(self, registry: DataFrameRegistry | None = None) -> None:
        """Initialize the toolkit with an optional DataFrame registry.

        Args:
            registry (DataFrameRegistry | None): An existing registry to use.
                If None, a new empty registry is created. Defaults to None.
        """
        self._registry = registry if registry is not None else DataFrameRegistry()

        self._core_tools = (
            tool(self.get_dataframe_id),
            tool(self.get_dataframe_reference),
            tool(self.list_dataframes),
            tool(self.execute_sql),
            tool(self.view_as_markdown_table),
        )

        self._tool_module_context = ToolModuleContext(
            registry=self._registry,
            get_dataframe_fn=self._get_dataframe,
            get_dataframe_reference_fn=self._get_dataframe_reference,
            validate_dataframe_name_fn=self._validate_dataframe_name,
        )
        self._module_cache: dict[type[ToolModule], ToolModule] = {}

    # -------------------------------------------------------------------------
    # Tool Access (Main API)
    # -------------------------------------------------------------------------

    def get_tools(
        self,
        *module_classes: type[ToolModule],
        exclude: set[str] | None = None,
    ) -> list[BaseTool]:
        """Return LangChain tools for this toolkit.

        Composes core tools with optional module tools, with per-call exclusion.
        This enables per-agent customization where different agents receive
        different subsets of tools based on their needs.

        Args:
            *module_classes (type[ToolModule]): Optional tool module classes to include.
                Modules are instantiated on first access and cached.
            exclude (set[str] | None): Optional set of tool names to exclude from the
                result. Applies to both core tools and module tools. Defaults to None.

        Returns:
            list[BaseTool]: Combined list of core tools and module tools, minus any
                excluded tools.

        Examples:
            Get core tools only (backward compatible):

            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> _ = toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))
            >>> tools = toolkit.get_tools()
            >>> len(tools) >= 1
            True

            Exclude specific tools:

            >>> toolkit = DataFrameToolkit()
            >>> tools = toolkit.get_tools(exclude={"view_as_markdown_table"})
            >>> tool_names = {t.name for t in tools}
            >>> "view_as_markdown_table" in tool_names
            False

            Get core tools plus a custom module::

                toolkit.get_tools(StatsModule)
                toolkit.get_tools(StatsModule, PlottingModule)
                toolkit.get_tools(StatsModule, exclude={"execute_sql"})
        """
        tools = list(self._core_tools)

        # dict.fromkeys preserves insertion order while deduplicating
        unique_module_classes = list(dict.fromkeys(module_classes))

        for module_class in unique_module_classes:
            module = self._get_or_create_module(module_class)
            tools.extend(module.get_tools())

        if exclude is not None:
            tools = [t for t in tools if t.name not in exclude]

        return tools

    def get_core_tools(self) -> list[BaseTool]:
        """Return core DataFrame management tools.

        Core tools provide essential DataFrame operations like ID lookup,
        schema inspection, and SQL querying.

        Returns:
            list[BaseTool]: Core tools for DataFrame management.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> _ = toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))
            >>> core_tools = toolkit.get_core_tools()
            >>> len(core_tools) >= 1
            True
        """
        return list(self._core_tools)

    def get_system_prompt(
        self,
        *module_classes: type[ToolModule],
        exclude: set[str] | None = None,
    ) -> str:
        """Generate a system prompt for the toolkit with optional module prompts.

        Combines the core system prompt with prompts from specified modules,
        respecting exclusions to accurately describe available tools. This enables
        per-agent customization where different agents receive different guidance
        based on their tool subsets.

        Note:
            Modules with zero tools (after exclusions) have their system prompt
            silently omitted from the combined prompt. Factory-generated classes
            with the same `__name__` will produce duplicate section headers in
            the system prompt.

        Args:
            *module_classes (type[ToolModule]): Optional tool module classes to include.
                Modules are instantiated on first access and cached.
            exclude (set[str] | None): Optional set of tool names to exclude. If provided,
                the system prompt will note which tools are unavailable. Defaults to None.

        Returns:
            str: Combined system prompt describing available tools and their usage.

        Examples:
            Get core prompt only:

            >>> toolkit = DataFrameToolkit()
            >>> prompt = toolkit.get_system_prompt()
            >>> "list_dataframes" in prompt
            True

            Exclude tools and document unavailability:

            >>> toolkit = DataFrameToolkit()
            >>> prompt = toolkit.get_system_prompt(exclude={"view_as_markdown_table"})
            >>> "not available" in prompt
            True

            Get core prompt plus module prompt::

                toolkit.get_system_prompt(StatsModule)
                toolkit.get_system_prompt(StatsModule, PlottingModule)
                toolkit.get_system_prompt(StatsModule, exclude={"execute_sql"})
        """
        prompt_parts = [self.CORE_SYSTEM_PROMPT]

        if exclude is not None:
            excluded_core_note = self._build_excluded_core_note(exclude)
            if excluded_core_note:
                prompt_parts.append(excluded_core_note)

        module_prompts = self._build_module_prompts(module_classes, exclude)
        prompt_parts.extend(module_prompts)

        return "".join(prompt_parts)

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    @property
    def references(self) -> tuple[DataFrameReference, ...]:
        """tuple[DataFrameReference, ...]: All registered DataFrame references."""
        return tuple(self._registry.references.values())

    def register_dataframe(
        self,
        name: str,
        dataframe: pl.DataFrame,
        *,
        description: str | None = None,
        column_descriptions: dict[str, str] | None = None,
    ) -> DataFrameReference:
        """Register a DataFrame with the toolkit.

        Creates a DataFrameReference containing metadata about the DataFrame and
        registers it for SQL query access. A generated unique ID is assigned for SQL queries.
        The user-provided name is stored in the DataFrameReference for reference.

        Args:
            name (str): A unique name to identify this DataFrame.
            dataframe (pl.DataFrame): The Polars DataFrame to register.
            description (str | None): Optional description of the DataFrame's contents
                and purpose. Helps LLM agents understand the data. Defaults to None.
            column_descriptions (dict[str, str] | None): Optional mapping of column
                names to descriptions. Helps LLM agents understand column semantics.
                Defaults to None.

        Returns:
            DataFrameReference: A reference containing metadata about the registered
                DataFrame, including its unique ID for SQL queries.

        Raises:
            ValueError: If a DataFrame with the given name is already registered.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> df = pl.DataFrame({"user_id": [1, 2], "score": [85, 92]})
            >>> ref = toolkit.register_dataframe(
            ...     "user_scores",
            ...     df,
            ...     description="User performance scores",
            ...     column_descriptions={"score": "Performance score from 0-100"},
            ... )
            >>> ref.name
            'user_scores'
        """
        if DATAFRAME_ID_PATTERN.match(name):
            msg = f"DataFrame name '{name}' cannot match ID pattern 'df_<8 hex chars>'"
            logger.warning("DataFrame registration failed", name=name, reason=msg)
            raise ValueError(msg)

        if self._dataframe_name_exists(name):
            msg = f"DataFrame '{name}' is already registered"
            logger.warning("DataFrame registration failed", name=name, reason=msg)
            raise ValueError(msg)

        reference = DataFrameReference.from_dataframe(
            name,
            dataframe,
            description=description,
            column_descriptions=column_descriptions,
        )

        self._registry.register(reference, dataframe)

        logger.info(
            "DataFrame registered",
            name=name,
            dataframe_id=reference.id,
            shape=dataframe.shape,
            column_count=len(dataframe.columns),
        )

        return reference

    def register_dataframes(
        self,
        dataframes: Mapping[str, pl.DataFrame],
        *,
        descriptions: Mapping[str, str] | None = None,
        column_descriptions: Mapping[str, dict[str, str]] | None = None,
    ) -> list[DataFrameReference]:
        """Register multiple DataFrames with the toolkit.

        Validates all inputs before modifying state and commits to the SQL
        context before updating references, so the two stores stay in sync.

        Args:
            dataframes (Mapping[str, pl.DataFrame]): Mapping of names to DataFrames.
            descriptions (Mapping[str, str] | None): Optional mapping of names to
                descriptions. Defaults to None.
            column_descriptions (Mapping[str, dict[str, str]] | None): Optional mapping of names to
                column descriptions. Defaults to None.

        Returns:
            list[DataFrameReference]: List of references for all registered DataFrames,
                in the same order as the input mapping.

        Raises:
            ValueError: If any name is already registered in the toolkit.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> dfs = {
            ...     "users": pl.DataFrame({"id": [1, 2]}),
            ...     "orders": pl.DataFrame({"id": [1], "user_id": [1]}),
            ... }
            >>> refs = toolkit.register_dataframes(
            ...     dfs,
            ...     descriptions={"users": "User accounts", "orders": "User orders"},
            ... )
            >>> len(refs)
            2
        """
        descriptions = descriptions or {}
        column_descriptions = column_descriptions or {}

        # Validate all names before modifying state
        existing_names = {ref.name for ref in self._registry.references.values()}
        for name in dataframes:
            if DATAFRAME_ID_PATTERN.match(name):
                msg = f"DataFrame name '{name}' cannot match ID pattern 'df_<8 hex chars>'"
                raise ValueError(msg)
            if name in existing_names:
                msg = f"DataFrame '{name}' is already registered"
                raise ValueError(msg)

        # Build all references first without modifying state (can fail without side effects)
        references = [
            DataFrameReference.from_dataframe(
                name,
                dataframe,
                description=descriptions.get(name),
                column_descriptions=column_descriptions.get(name),
            )
            for name, dataframe in dataframes.items()
        ]

        # Register each dataframe with its reference atomically
        for ref, df in zip(references, dataframes.values(), strict=True):
            self._registry.register(ref, df)
            logger.info(
                "DataFrame registered",
                name=ref.name,
                dataframe_id=ref.id,
                shape=df.shape,
                column_count=len(df.columns),
            )

        logger.info("DataFrames registered", count=len(references), names=[r.name for r in references])

        return references

    def unregister_dataframe(self, name: str) -> None:
        """Unregister a DataFrame from the toolkit.

        Removes the DataFrame from both the internal registry and the SQL context.

        Args:
            name (str): The name of the DataFrame to unregister.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> df = pl.DataFrame({"a": [1, 2, 3]})
            >>> _ = toolkit.register_dataframe("test", df)
            >>> toolkit.unregister_dataframe("test")
        """
        reference = self._get_dataframe_reference_by_name(name)

        self._registry.unregister(reference.id)

        logger.info("DataFrame unregistered", name=name, dataframe_id=reference.id)

    def get_dataframe_id(self, name: str) -> DataFrameId | ToolCallError:
        """Get the DataFrameId for a DataFrame by its name.

        Use this tool when you need the unique identifier for a DataFrame
        to use in SQL queries. DataFrame names are human-readable labels
        while IDs are the actual table names in SQL.

        Args:
            name (str): The human-readable name of the DataFrame.

        Returns:
            DataFrameId | ToolCallError: DataFrameId if found, or ToolCallError if
                name not registered or if an ID was passed instead of a name.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> _ = toolkit.register_dataframe("sales", pl.DataFrame({"a": [1]}))
            >>> toolkit.get_dataframe_id("sales")  # doctest: +ELLIPSIS
            'df_...'
            >>> toolkit.get_dataframe_id("nonexistent")  # doctest: +ELLIPSIS
            ToolCallError(error_type='DataFrameNotFound', ...)
            >>> toolkit.get_dataframe_id("df_1a2b3c4d")  # doctest: +ELLIPSIS
            ToolCallError(error_type='InvalidArgument', ...)
        """
        tool_name = _current_tool_name()
        logger.log(TOOL_CALL_LEVEL, _TOOL_CALL_MSG.format(tool_name=tool_name), name=name)

        # Guard: detect if an ID was passed instead of a name
        if DATAFRAME_ID_PATTERN.match(name):
            result = ToolCallError(
                error_type="InvalidArgument",
                message=(
                    f"'{name}' is already an ID, not a name. "
                    "This tool converts names to IDs. "
                    "Use the name to look up the ID, or use get_dataframe_reference "
                    "if you need schema details from an ID."
                ),
                details={"available_names": [ref.name for ref in self._registry.references.values()]},
            )
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name),
                name=name,
                error_type=result.error_type,
                message=result.message,
            )
            return result

        try:
            reference = self._get_dataframe_reference_by_name(name)
            logger.debug(_TOOL_CALL_RESULT_MSG.format(tool_name=tool_name), name=name, dataframe_id=str(reference.id))
            return reference.id
        except KeyError:
            result = ToolCallError(
                error_type="DataFrameNotFound",
                message=f"DataFrame '{name}' is not registered",
                details={"available_names": [ref.name for ref in self._registry.references.values()]},
            )
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name),
                name=name,
                error_type=result.error_type,
                message=result.message,
            )
            return result

    def list_dataframes(self) -> list[DataFrameReference]:
        """List all available DataFrames in the toolkit.

        Use this tool to discover what DataFrames are available for querying.
        Returns a list of DataFrameReferences with names, IDs, and schema
        information for each registered DataFrame.

        Returns:
            list[DataFrameReference]: List of DataFrameReference objects (may be empty).

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> toolkit.list_dataframes()
            []
            >>> _ = toolkit.register_dataframe("sales", pl.DataFrame({"a": [1]}))
            >>> len(toolkit.list_dataframes())
            1
        """
        tool_name = _current_tool_name()
        logger.log(TOOL_CALL_LEVEL, _TOOL_CALL_MSG.format(tool_name=tool_name))
        result = list(self._registry.references.values())
        logger.debug(
            _TOOL_CALL_RESULT_MSG.format(tool_name=tool_name), count=len(result), names=[r.name for r in result]
        )
        return result

    def get_dataframe_reference(self, identifier: str) -> DataFrameReference | ToolCallError:
        """Get detailed information about a DataFrame by name or ID.

        Use this tool to understand a DataFrame schema, column statistics,
        and metadata before writing SQL queries. Returns comprehensive
        information including column names, data types, and summaries.

        Args:
            identifier (str): Either the DataFrame name or its ID (df_xxxxxxxx).

        Returns:
            DataFrameReference | ToolCallError: DataFrameReference with full schema
                info, or ToolCallError if not found.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> ref = toolkit.register_dataframe("sales", pl.DataFrame({"a": [1]}))
            >>> toolkit.get_dataframe_reference("sales")  # doctest: +ELLIPSIS
            DataFrameReference(...)
            >>> toolkit.get_dataframe_reference(ref.id)  # doctest: +ELLIPSIS
            DataFrameReference(...)
        """
        tool_name = _current_tool_name()
        logger.log(TOOL_CALL_LEVEL, _TOOL_CALL_MSG.format(tool_name=tool_name), identifier=identifier)
        result = self._get_dataframe_reference(identifier)
        if isinstance(result, ToolCallError):
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name), identifier=identifier, error_type=result.error_type
            )
        else:
            logger.debug(
                _TOOL_CALL_RESULT_MSG.format(tool_name=tool_name),
                identifier=identifier,
                name=result.name,
                dataframe_id=result.id,
            )
        return result

    def execute_sql(
        self,
        *,
        query: str,
        result_name: str,
        result_description: str | None = None,
    ) -> DataFrameReference | ToolCallError:
        """Execute a SQL query and store the result as a new DataFrame.

        Use this tool to query registered DataFrames using SQL. The query
        is validated before execution to catch errors early. Use DataFrame
        IDs (df_xxxxxxxx) as table names in your SQL queries.

        IMPORTANT: Only SELECT queries are allowed. Destructive commands
        (DELETE, DROP, INSERT, UPDATE, etc.) are blocked.

        Args:
            query (str): SQL SELECT query using DataFrame IDs as table names.
            result_name (str): Human-readable name for the result DataFrame.
            result_description (str | None): Optional description of what this
                result represents. Defaults to None.

        Returns:
            DataFrameReference | ToolCallError: DataFrameReference for the query
                result, or ToolCallError on failure.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> ref = toolkit.register_dataframe("sales", pl.DataFrame({"id": [1], "amount": [100]}))
            >>> result = toolkit.execute_sql(
            ...     query=f"SELECT * FROM {ref.id} WHERE amount > 50",
            ...     result_name="high_sales",
            ...     result_description="Sales over $50",
            ... )
            >>> isinstance(result, DataFrameReference)
            True
        """
        tool_name = _current_tool_name()
        logger.log(TOOL_CALL_LEVEL, _TOOL_CALL_MSG.format(tool_name=tool_name), result_name=result_name, query=query)

        name_error = self._validate_dataframe_name(result_name)
        if name_error is not None:
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name), error_type=name_error.error_type, query=query
            )
            return name_error

        validated_expression = self._validate_query(query)
        if isinstance(validated_expression, ToolCallError):
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name),
                error_type=validated_expression.error_type,
                query=query,
            )
            return validated_expression

        try:
            result_df = self._registry.context.execute_sql(query, eager=True)
        except pl.exceptions.PolarsError as e:
            result = ToolCallError(
                error_type="SQLExecutionError",
                message=f"Query execution failed: {e}",
                details={"query": query},
            )
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name), error_type="SQLExecutionError", query=query
            )
            return result

        # Polars SQLContext.execute_sql(eager=True) should always return a DataFrame,
        # but guard against LazyFrame in case of future API changes.
        result_df = ensure_dataframe(result_df)

        referenced_tables = set(extract_table_names(validated_expression))
        parent_ids = [ref.id for ref in self._registry.references.values() if ref.id.lower() in referenced_tables]

        reference = DataFrameReference.from_dataframe(
            result_name,
            result_df,
            description=result_description,
            parent_ids=parent_ids,
            source_query=query,
        )

        self._registry.register(reference, result_df)

        logger.debug(
            _TOOL_CALL_RESULT_MSG.format(tool_name=tool_name), result_name=result_name, dataframe_id=reference.id
        )
        logger.info(
            "DataFrame created via SQL", name=result_name, dataframe_id=reference.id, shape=result_df.shape, query=query
        )

        return reference

    def view_as_markdown_table(
        self,
        identifier: str,
        columns: Sequence[str] | None = None,
        num_rows: int = 10,
        *,
        sample: bool = False,
        seed: int | None = None,
    ) -> str | ToolCallError:
        """View a DataFrame as a markdown-formatted table.

        Use this tool to preview DataFrame contents in a human-readable markdown
        format. This is useful for inspecting data before writing SQL queries or
        for presenting results to users. You can optionally select specific
        columns and control the number of rows displayed.

        Note:
            This method temporarily modifies global `pl.Config` state to render
            the table. It is not thread-safe: concurrent calls from different
            threads may observe each other's configuration.

        Args:
            identifier (str): Either the DataFrame name or its ID (df_xxxxxxxx).
            columns (Sequence[str] | None): Optional sequence of column names to
                display. If None, all columns are shown. Defaults to None.
            num_rows (int): Maximum number of rows to display. Must be at least 1.
                Defaults to 10.
            sample (bool): If True, randomly sample rows instead of taking the
                first rows. Useful for large DataFrames. Defaults to False.
            seed (int | None): Random seed for sampling when sample=True. Only
                valid when sample=True. Defaults to None.

        Returns:
            str | ToolCallError: Markdown-formatted table string, or ToolCallError
                if the DataFrame is not found or columns are invalid.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
            >>> ref = toolkit.register_dataframe("sales", df)
            >>> result = toolkit.view_as_markdown_table("sales", num_rows=2)
            >>> isinstance(result, str)
            True
            >>> result = toolkit.view_as_markdown_table(ref.id, columns=["id"])
            >>> isinstance(result, str)
            True
        """
        tool_name = _current_tool_name()
        logger.log(
            TOOL_CALL_LEVEL,
            _TOOL_CALL_MSG.format(tool_name=tool_name),
            identifier=identifier,
            num_rows=num_rows,
            sample=sample,
        )

        df = self._get_dataframe(identifier)
        if isinstance(df, ToolCallError):
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name), identifier=identifier, error_type=df.error_type
            )
            return df

        try:
            result = to_markdown_table(
                df,
                columns=columns,
                num_rows=num_rows,
                sample=sample,
                seed=seed,
            )
            logger.debug(_TOOL_CALL_RESULT_MSG.format(tool_name=tool_name), identifier=identifier)
            return result
        except (ColumnsNotFoundError, DuplicateColumnsError) as e:
            error = _handle_column_validation_error(e, columns, df.columns)
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name), identifier=identifier, error_type=error.error_type
            )
            return error
        # Catches remaining ValueError cases: num_rows < 1, seed without sample=True
        except ValueError as e:
            result = ToolCallError(
                error_type="InvalidArgument",
                message=str(e),
                details={
                    "columns": list(columns) if columns is not None else None,
                    "num_rows": num_rows,
                    "sample": sample,
                    "seed": seed,
                },
            )
            logger.warning(
                _TOOL_CALL_ERROR_MSG.format(tool_name=tool_name), identifier=identifier, error_type=result.error_type
            )
            return result

    def export_state(self) -> DataFrameToolkitState:
        """Export the current toolkit state for serialization.

        Returns a DataFrameToolkitState containing all registered references.
        The actual DataFrame data is NOT included - only metadata and provenance.

        Returns:
            DataFrameToolkitState: Serializable state containing all references.

        Examples:
            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> _ = toolkit.register_dataframe("sales", pl.DataFrame({"a": [1, 2, 3]}))
            >>> state = toolkit.export_state()
            >>> len(state.references)
            1
        """
        return DataFrameToolkitState(references=list(self._registry.references.values()))

    # -------------------------------------------------------------------------
    # Public Class Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_state(
        cls,
        state: DataFrameToolkitState,
        base_dataframes: Mapping[str, pl.DataFrame],
        *,
        rel_tol: float = REL_TOL_DEFAULT,
    ) -> DataFrameToolkit:
        """Create a toolkit from saved state and base dataframes.

        This is the recommended way to restore a toolkit from serialized state.
        Matches base dataframes to their state references by name or ID,
        preserving original IDs for proper derivative reconstruction.

        Provided base dataframes are validated against the expected schema
        and column statistics from the saved state to ensure data consistency.

        Validation checks: shape, column names/order, dtype, count, null_count,
        unique_count, min, max, mean, std, p25, p50, p75.

        Args:
            state (DataFrameToolkitState): Serialized state from export_state().
            base_dataframes (Mapping[str, pl.DataFrame]): Mapping of identifier to
                DataFrame for all base tables. Keys can be either names or IDs
                (df_xxxxxxxx format). DataFrames must match the schema and
                statistics from when the state was exported.
            rel_tol (float): Relative tolerance for floating point comparisons
                during validation. Defaults to 1e-9.

        Returns:
            DataFrameToolkit: Fully reconstructed toolkit with all base and
                derivative dataframes.

        Examples:
            Restore by name (most common):

            >>> import polars as pl
            >>> toolkit = DataFrameToolkit()
            >>> df = pl.DataFrame({"a": [1, 2, 3]})
            >>> _ = toolkit.register_dataframe("sales", df)
            >>> state = toolkit.export_state()
            >>> new_toolkit = DataFrameToolkit.from_state(state, {"sales": df})
            >>> len(new_toolkit.references)
            1

            Restore by ID:

            >>> ref = toolkit.references[0]
            >>> new_toolkit = DataFrameToolkit.from_state(state, {ref.id: df})
            >>> len(new_toolkit.references)
            1
        """
        registry = restore_registry_from_state(
            state=state,
            base_dataframes=base_dataframes,
            rel_tol=rel_tol,
        )
        return cls(registry=registry)

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _get_or_create_module(self, module_class: type[ToolModule]) -> ToolModule:
        """Get a cached module instance, creating it on first access.

        Args:
            module_class (type[ToolModule]): The module class to instantiate.

        Returns:
            ToolModule: The cached or newly created module instance.

        Raises:
            TypeError: If the module class cannot be instantiated with a
                ToolModuleContext argument.
        """
        if module_class not in self._module_cache:
            try:
                self._module_cache[module_class] = module_class(self._tool_module_context)
            except TypeError as e:
                msg = (
                    f"Failed to instantiate {module_class.__name__}. "
                    f"Tool modules must accept a single ToolModuleContext argument: "
                    f"__init__(self, context: ToolModuleContext). Got: {e}"
                )
                raise TypeError(msg) from e
        return self._module_cache[module_class]

    def _build_excluded_core_note(self, exclude: set[str]) -> str:
        """Build a note about excluded core tools.

        Args:
            exclude (set[str]): Set of tool names to exclude.

        Returns:
            str: Note about excluded core tools, or empty string if none excluded.
        """
        core_tool_names = {t.name for t in self._core_tools}
        excluded_core = core_tool_names & exclude
        if not excluded_core:
            return ""

        excluded_list = ", ".join(sorted(excluded_core))
        return f"\nNote: The following tools are not available in this context: {excluded_list}"

    def _build_module_prompts(
        self,
        module_classes: tuple[type[ToolModule], ...],
        exclude: set[str] | None,
    ) -> list[str]:
        """Build system prompt sections for modules.

        Args:
            module_classes (tuple[type[ToolModule], ...]): Module classes to include.
            exclude (set[str] | None): Optional set of tool names to exclude.

        Returns:
            list[str]: List of prompt sections for each module.
        """
        # dict.fromkeys preserves insertion order while deduplicating
        unique_module_classes = list(dict.fromkeys(module_classes))
        prompts = []

        for module_class in unique_module_classes:
            module_prompt = self._build_module_prompt(module_class, exclude)
            if module_prompt:
                prompts.append(module_prompt)

        return prompts

    def _build_module_prompt(
        self,
        module_class: type[ToolModule],
        exclude: set[str] | None,
    ) -> str:
        """Build system prompt section for a single module.

        Args:
            module_class (type[ToolModule]): The module class.
            exclude (set[str] | None): Optional set of tool names to exclude.

        Returns:
            str: Prompt section for the module, or empty string if all tools excluded.
        """
        module = self._get_or_create_module(module_class)
        module_tool_names = {t.name for t in module.get_tools()}

        excluded_module_tool_names = module_tool_names & exclude if exclude is not None else set()

        # If all module tools are excluded (or module has no tools), omit its prompt entirely
        if excluded_module_tool_names == module_tool_names:
            return ""

        parts = [f"\n## {module_class.__name__}\n", module.system_prompt]

        if excluded_module_tool_names:
            excluded_list = ", ".join(sorted(excluded_module_tool_names))
            parts.append(f"\n\nNote: The following tools are not available in this context: {excluded_list}")

        return "".join(parts)

    def _get_dataframe_reference_by_name(self, name: str) -> DataFrameReference:
        """Get a DataFrameReference by its user-friendly name.

        Performs O(n) scan over registered references. This is acceptable because
        typical usage involves few DataFrames (2-20), and name lookup only
        occurs during registration/unregistration, not during queries.

        Args:
            name (str): The user-friendly name to look up.

        Returns:
            DataFrameReference: The reference with the matching name.

        Raises:
            KeyError: If no DataFrame with the given name is registered.
        """
        for ref in self._registry.references.values():
            if ref.name == name:
                return ref
        msg = f"DataFrame '{name}' is not registered"
        raise KeyError(msg)

    def _get_dataframe(self, identifier: str) -> pl.DataFrame | ToolCallError:
        """Get a DataFrame by its identifier (name or ID).

        Looks up the identifier to find a reference, then fetches the actual
        DataFrame from the SQL context. Returns ToolCallError if the
        identifier cannot be found.

        Args:
            identifier (str): Either the DataFrame name or its ID (df_xxxxxxxx).

        Returns:
            pl.DataFrame | ToolCallError: The matching DataFrame, or
                ToolCallError if not found.

        Raises:
            RuntimeError: If the reference exists in the registry but the
                DataFrame is missing from the SQL context (internal invariant
                violation).
        """
        ref = self._get_dataframe_reference(identifier)
        if isinstance(ref, ToolCallError):
            return ref

        try:
            df = self._registry.context.get_dataframe(ref.id)
        except KeyError as e:
            msg = f"DataFrame '{ref.id}' found in registry but missing from SQL context (identifier={identifier!r})"
            raise RuntimeError(msg) from e

        df = ensure_dataframe(df)
        return df

    def _get_dataframe_reference(self, identifier: str) -> DataFrameReference | ToolCallError:
        """Get a DataFrameReference by its identifier (name or ID).

        Tries lookup by ID first (O(1)), then falls back to name lookup (O(n)).
        Returns a ToolCallError if no match is found.

        Args:
            identifier (str): Either the DataFrame name or its ID (df_xxxxxxxx).

        Returns:
            DataFrameReference | ToolCallError: The matching reference, or
                ToolCallError if no DataFrame matches the identifier.
        """
        # Try lookup by ID first (O(1) since registry.references is keyed by ID)
        if identifier in self._registry.references:
            return self._registry.references[identifier]

        # Try lookup by name (O(n) scan)
        try:
            return self._get_dataframe_reference_by_name(identifier)
        except KeyError:
            pass

        return ToolCallError(
            error_type="DataFrameNotFound",
            message=f"DataFrame '{identifier}' not found by name or ID",
            details={
                "available_names": [ref.name for ref in self._registry.references.values()],
                "available_ids": list(self._registry.references.keys()),
            },
        )

    def _validate_dataframe_name(self, dataframe_name: str) -> ToolCallError | None:
        """Validate that a result name is acceptable for registration.

        Args:
            dataframe_name (str): The proposed name for the result DataFrame.

        Returns:
            ToolCallError | None: Error if the name is invalid, None if valid.
        """
        if DATAFRAME_ID_PATTERN.match(dataframe_name):
            return ToolCallError(
                error_type="InvalidArgument",
                message=f"Result name '{dataframe_name}' cannot match ID pattern 'df_<8 hex chars>'",
                details={"suggestion": "Choose a descriptive name that doesn't look like a DataFrame ID"},
            )

        if self._dataframe_name_exists(dataframe_name):
            return ToolCallError(
                error_type="DuplicateName",
                message=f"DataFrame name '{dataframe_name}' is already registered",
                details={"suggestion": "Choose a different name for the result"},
            )

        return None

    def _validate_query(self, query: str) -> exp.Expression | ToolCallError:
        """Validate a SQL query against registered DataFrames.

        Args:
            query (str): The SQL query to validate.

        Returns:
            exp.Expression | ToolCallError: The parsed AST on success,
                or ToolCallError on failure.
        """
        table_columns = self._build_table_columns_schema()
        try:
            return validate_sql(query, table_columns, blacklist=DESTRUCTIVE_COMMANDS)
        except SQLSyntaxError as e:
            return ToolCallError(
                error_type="SQLSyntaxError",
                message=str(e),
                details={"errors": e.errors, "query": e.query},
            )
        except SQLTableError as e:
            return ToolCallError(
                error_type="SQLTableError",
                message=str(e),
                details={
                    "invalid_tables": e.invalid_tables,
                    "available_tables": list(table_columns.keys()),
                    "query": e.query,
                },
            )
        except SQLColumnError as e:
            return ToolCallError(
                error_type="SQLColumnError",
                message=e.format_details(),
                details={
                    "invalid_columns": e.invalid_columns,
                    "ambiguous_columns": e.ambiguous_columns,
                    "not_found_columns": e.not_found_columns,
                    "table_columns": {k: list(v) for k, v in e.table_columns.items()},
                    "query": e.query,
                },
            )
        except SQLBlacklistedCommandError as e:
            return ToolCallError(
                error_type="SQLBlacklistedCommand",
                message=f"Command '{e.command_type}' is not allowed. Only SELECT queries are permitted.",
                details={"blocked_command": e.command_type, "query": e.query},
            )

    def _build_table_columns_schema(self) -> dict[str, set[str]]:
        """Build table_columns schema from registered references.

        Returns:
            dict[str, set[str]]: Mapping of DataFrame IDs to their column name sets.
        """
        return {ref.id: set(ref.column_names) for ref in self._registry.references.values()}

    def _dataframe_name_exists(self, name: str) -> bool:
        """Check if a name is already registered.

        Args:
            name (str): The name to check.

        Returns:
            bool: True if a DataFrame with this name exists, False otherwise.
        """
        return any(ref.name == name for ref in self._registry.references.values())


# -----------------------------------------------------------------------------
# Private Helpers
# -----------------------------------------------------------------------------

# Tool logging message templates
_TOOL_CALL_MSG = "Tool call: {tool_name}"
_TOOL_CALL_ERROR_MSG = "Tool call error: {tool_name}"
_TOOL_CALL_RESULT_MSG = "Tool call result: {tool_name}"


def _current_tool_name() -> str:
    """Get the name of the calling function via inspect.

    Returns:
        str: The name of the calling function, or "unknown" if unable to determine.
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame is not None else None
    name = caller_frame.f_code.co_name if caller_frame is not None else "unknown"
    return name


def _handle_column_validation_error(
    error: ColumnsNotFoundError | DuplicateColumnsError,
    requested_columns: Sequence[str] | None,
    available_columns: Sequence[str],
) -> ToolCallError:
    """Convert a column validation error to a ToolCallError.

    Args:
        error (ColumnsNotFoundError | DuplicateColumnsError): The column
            validation error raised by to_markdown_table.
        requested_columns (Sequence[str] | None): The columns that were
            requested.
        available_columns (Sequence[str]): The columns available in the
            DataFrame.

    Returns:
        ToolCallError: Formatted error with details appropriate to the error type.
    """
    if isinstance(error, DuplicateColumnsError):
        return ToolCallError(
            error_type="DuplicateColumns",
            message=str(error),
            details={
                "available_columns": list(available_columns),
                "requested_columns": list(requested_columns) if requested_columns is not None else None,
                "duplicate_columns": error.duplicate_columns,
            },
        )

    return ToolCallError(
        error_type="InvalidColumns",
        message=f"Invalid columns specified: {error}",
        details={
            "available_columns": list(available_columns),
            "requested_columns": list(requested_columns) if requested_columns is not None else None,
            "invalid_columns": error.missing_columns,
        },
    )
