"""Models for DataFrame manipulation and analysis using LangChain tools."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, Field, JsonValue, model_validator

from dfkit.identifier import DataFrameId, generate_dataframe_id
from dfkit.polars_utils import get_series_description


class ToolCallError(BaseModel):
    """Structured error response for LLM tool calls.

    Provides detailed error information that enables LLM agents to
    understand what went wrong and potentially self-correct.

    Attributes:
        error_type (str): Category of error (e.g., "DataFrameNotFound", "SQLValidationError").
            Must be at least 1 character to ensure meaningful classification.
        message (str): Human-readable error description. Must be at least 1 character
            to ensure errors always have meaningful content for LLM interpretation.
        details (dict[str, JsonValue]): Additional context-specific information (JSON-serializable).

    Examples:
        Create a ToolCallError for a missing column in a SQL query:
        >>> error = ToolCallError(
        ...     error_type="SQLColumnError",
        ...     message="Invalid columns in query",
        ...     details={
        ...         "invalid_columns": ["unknown_col"],
        ...         "table_columns": {"df_abc123": ["id", "name", "value"]},
        ...     },
        ... )
    """

    error_type: str = Field(description="Category of the error.", min_length=1)
    message: str = Field(description="Human-readable error description.", min_length=1)
    details: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional error context and suggestions (JSON-serializable).",
    )


class ColumnSummary(BaseModel):
    """A summary of a single column in a DataFrame.

    Attributes:
        description (str): A textual description of the column for analysis context.
        dtype (str): The data type of the column.
        count (int): The number of non-null entries in the column.
        null_count (int): The number of null entries in the column.
        unique_count (int): The number of unique entries in the column.
        min (float | str | None): The minimum value in the column, or None for empty/all-null columns.
        max (float | str | None): The maximum value in the column, or None for empty/all-null columns.
        mean (float | str | None): The mean value in the column.
        std (float | str | None): The standard deviation of the values in the column.
        p25 (float | str | None): The 25th percentile of the values in the column.
        p50 (float | str | None): The 50th percentile (median) of the values in the column.
        p75 (float | str | None): The 75th percentile of the values in the column.
    """

    description: str = Field(description="A textual description of the column for analysis context.")
    dtype: str = Field(description="The data type of the column.")
    count: int = Field(description="The number of non-null entries in the column.")
    null_count: int = Field(description="The number of null entries in the column.")
    unique_count: int = Field(description="The number of unique entries in the column.")
    min: float | str | None = Field(description="The minimum value in the column, or None for empty/all-null columns.")
    max: float | str | None = Field(description="The maximum value in the column, or None for empty/all-null columns.")
    mean: float | str | None = Field(description="The mean value in the column.")
    std: float | str | None = Field(description="The standard deviation of the values in the column.")
    p25: float | str | None = Field(description="The 25th percentile of the values in the column.")
    p50: float | str | None = Field(description="The 50th percentile (median) of the values in the column.")
    p75: float | str | None = Field(description="The 75th percentile of the values in the column.")

    @classmethod
    def from_series(cls, series: pl.Series, description: str | None = None) -> ColumnSummary:
        """Create a ColumnSummary from a Polars Series.

        Args:
            series (pl.Series): The Polars Series to summarize.
            description (str | None, optional): An optional textual description of the column. Defaults to None.

        Returns:
            ColumnSummary: The generated ColumnSummary.

        Note:
            For empty series or all-null columns, min and max will be None since
            these statistics are not defined for such data.
        """
        # Get descriptive statistics for the series
        des_dict = get_series_description(series)

        return cls(
            description=description or "",
            dtype=str(series.dtype),
            count=int(des_dict["count"]),
            null_count=int(des_dict["null_count"]),
            unique_count=series.n_unique(),
            min=des_dict.get("min"),
            max=des_dict.get("max"),
            mean=des_dict.get("mean"),
            std=des_dict.get("std"),
            p25=des_dict.get("25%"),
            p50=des_dict.get("50%"),
            p75=des_dict.get("75%"),
        )


class DataFrameReference(BaseModel):
    """A reference to a Polars DataFrame in a dataframe registry.

    References are classified as either **base** or **derivative**:

    - **Base**: User-provided DataFrames with no lineage. ``parent_ids`` is empty
      and ``source_query`` is None.
    - **Derivative**: DataFrames produced by SQL queries. ``parent_ids`` is non-empty
      and ``source_query`` is set.

    Invariant:
        ``parent_ids`` and ``source_query`` must be consistent: both empty/None (base)
        or both populated (derivative). A model validator enforces this at construction
        time, so ``ref.is_base`` is a reliable test for base references.

    Attributes:
        id (DataFrameId): Unique identifier to reference the DataFrame in the registry and SQL queries.
        name (str): The name of the DataFrame.
        description (str): A textual description of the DataFrame for analysis context.
        num_rows (int): The number of rows in the DataFrame.
        num_columns (int): The number of columns in the DataFrame.
        column_names (list[str]): The names of the columns in the DataFrame.
        column_summaries (dict[str, ColumnSummary]): A summary of each column in the DataFrame.
        parent_ids (list[DataFrameId]): The identifiers of the immediate parent DataFrames.
            Empty for base DataFrames, non-empty for derivatives.
        source_query (str | None): The SQL query that generated this DataFrame.
            None for base DataFrames, set for derivatives.
    """

    id: DataFrameId = Field(
        description="Unique identifier to reference the DataFrame in the dataframe registry and SQL queries.",
        default_factory=generate_dataframe_id,
    )
    name: str = Field(description="The name of the DataFrame.")
    description: str = Field(description="A textual description of the DataFrame for analysis context.")
    num_rows: int = Field(description="The number of rows in the DataFrame.")
    num_columns: int = Field(description="The number of columns in the DataFrame.")
    column_names: list[str] = Field(description="The names of the columns in the DataFrame.")
    column_summaries: dict[str, ColumnSummary] = Field(description="A summary of each column in the DataFrame.")
    parent_ids: list[DataFrameId] = Field(
        description=(
            "The identifiers of the immediate parent DataFrames. "
            "Empty for base DataFrames, non-empty for derivatives. "
            "Must be non-empty when source_query is set."
        ),
        default_factory=list,
    )
    source_query: str | None = Field(
        default=None,
        description=(
            "The SQL query that generated this DataFrame. "
            "None for user-provided base DataFrames, set for derivatives. "
            "Must be set when parent_ids is non-empty."
        ),
        min_length=1,
    )

    @property
    def is_base(self) -> bool:
        """Whether this is a base (user-provided) reference with no lineage."""
        return not self.parent_ids

    @model_validator(mode="after")
    def _validate_base_derivative_consistency(self) -> DataFrameReference:
        """Enforce that parent_ids and source_query are consistent.

        Returns:
            DataFrameReference: The validated instance.

        Raises:
            ValueError: If source_query is set without parent_ids, or parent_ids
                is non-empty without source_query.
        """
        has_parents = bool(self.parent_ids)
        has_query = self.source_query is not None

        if has_query and not has_parents:
            msg = (
                f"Derivative DataFrameReference '{self.name}' has source_query but empty parent_ids. "
                f"Derivatives must specify their parent DataFrames."
            )
            raise ValueError(msg)

        if has_parents and not has_query:
            msg = (
                f"Derivative DataFrameReference '{self.name}' has parent_ids but no source_query. "
                f"Derivatives must include the SQL query that produced them."
            )
            raise ValueError(msg)

        return self

    @classmethod
    def from_dataframe(
        cls,
        name: str,
        dataframe: pl.DataFrame,
        *,
        description: str | None = None,
        column_descriptions: dict[str, str] | None = None,
        parent_ids: list[DataFrameId] | None = None,
        source_query: str | None = None,
    ) -> DataFrameReference:
        """Create a DataFrameReference from a Polars DataFrame.

        Args:
            name (str): The name of the DataFrame.
            dataframe (pl.DataFrame): Polars DataFrame.
            description (str | None): An optional textual description of the DataFrame. Defaults to None.
            column_descriptions (dict[str, str] | None): Optional textual descriptions of the columns. Defaults to None.
            parent_ids (list[DataFrameId] | None): Identifiers of the parent DataFrames.
                Required for derivatives, must be None/empty for base DataFrames. Defaults to None.
            source_query (str | None): The SQL query that generated this DataFrame.
                Required for derivatives, must be None for base DataFrames. Defaults to None.
                NOTE: empty queries are not allowed, use None for no query.

        Returns:
            DataFrameReference: The generated DataFrameReference.
        """
        if column_descriptions is None:
            column_descriptions = {}

        return cls(
            name=name,
            num_rows=dataframe.height,
            num_columns=dataframe.width,
            column_names=dataframe.columns,
            column_summaries={
                col: ColumnSummary.from_series(dataframe[col], description=column_descriptions.get(col))
                for col in dataframe.columns
            },
            description=description or "",
            parent_ids=parent_ids or [],
            source_query=source_query,
        )


class DataFrameToolkitState(BaseModel):
    """Serializable state of a DataFrameToolkit for reconstruction.

    Contains only the references (metadata), not the actual DataFrame data.
    Base dataframes must be re-registered before reconstruction.

    Attributes:
        references (list[DataFrameReference]): All registered DataFrame references with provenance.

    Examples:
        Export and serialize toolkit state:

        >>> state = DataFrameToolkitState(references=[])
        >>> state.model_dump_json()
        '{"references":[]}'
    """

    references: list[DataFrameReference] = Field(
        default_factory=list,
        description="All registered DataFrame references with provenance.",
    )
