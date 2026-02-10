"""Tests for the ToolCallError model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest_check import check

from dfkit.models import ToolCallError


class TestToolCallError:
    """Tests for the ToolCallError model."""

    def test_tool_call_error_all_fields(self) -> None:
        """Given all fields, When instantiated, Then model is valid."""
        error = ToolCallError(
            error_type="DataFrameNotFound",
            message="DataFrame 'sales' is not registered",
            details={"available_names": ["orders", "customers"]},
        )

        with check:
            assert error.error_type == "DataFrameNotFound"
        with check:
            assert error.message == "DataFrame 'sales' is not registered"
        with check:
            assert error.details == {"available_names": ["orders", "customers"]}

    def test_tool_call_error_minimal(self) -> None:
        """Given only required fields, When instantiated, Then details is empty dict."""
        error = ToolCallError(
            error_type="SQLSyntaxError",
            message="Invalid SQL syntax near 'SELEC'",
        )

        with check:
            assert error.error_type == "SQLSyntaxError"
        with check:
            assert error.message == "Invalid SQL syntax near 'SELEC'"
        with check:
            assert error.details == {}

    def test_tool_call_error_serialization(self) -> None:
        """Given model instance, When converted to dict, Then all fields present."""
        error = ToolCallError(
            error_type="SQLTableError",
            message="Table 'unknown_table' does not exist",
            details={"invalid_tables": ["unknown_table"], "available_tables": ["df_abc123"]},
        )

        error_dict = error.model_dump()

        with check:
            assert "error_type" in error_dict
        with check:
            assert "message" in error_dict
        with check:
            assert "details" in error_dict
        with check:
            assert error_dict["error_type"] == "SQLTableError"
        with check:
            assert error_dict["message"] == "Table 'unknown_table' does not exist"
        with check:
            assert error_dict["details"]["invalid_tables"] == ["unknown_table"]

    def test_tool_call_error_serialization_minimal(self) -> None:
        """Given minimal model instance, When converted to dict, Then details is empty dict."""
        error = ToolCallError(
            error_type="ExecutionError",
            message="Query execution failed",
        )

        error_dict = error.model_dump()

        with check:
            assert error_dict["details"] == {}

    def test_tool_call_error_required_fields(self) -> None:
        """Given missing required fields, When instantiated, Then raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolCallError(error_type="SomeError")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            ToolCallError(message="Some message")  # type: ignore[call-arg]

    def test_tool_call_error_has_field_descriptions(self) -> None:
        """Given ToolCallError model, When schema inspected, Then fields have descriptions."""
        schema = ToolCallError.model_json_schema()
        properties = schema["properties"]

        with check:
            assert "description" in properties["error_type"]
        with check:
            assert "description" in properties["message"]
        with check:
            assert "description" in properties["details"]

    def test_tool_call_error_json_serialization(self) -> None:
        """Given ToolCallError with nested details, When serialized to JSON, Then valid JSON string."""
        error = ToolCallError(
            error_type="SQLColumnError",
            message="Invalid columns in query",
            details={
                "invalid_columns": ["unknown_col"],
                "table_columns": {"df_abc123": ["id", "name", "value"]},
                "count": 1,
                "is_recoverable": True,
            },
        )

        json_str = error.model_dump_json()

        with check:
            assert isinstance(json_str, str)
        with check:
            assert "SQLColumnError" in json_str
        with check:
            assert "invalid_columns" in json_str
        with check:
            assert "unknown_col" in json_str
