"""Models for tool module error responses."""

from pydantic import BaseModel, Field, JsonValue


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
