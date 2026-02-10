"""Tests for the DataFrameToolkitState model."""

from __future__ import annotations

import json

import polars as pl
import pytest
from pydantic import ValidationError
from pytest_check import check

from dfkit.models import (
    DataFrameReference,
    DataFrameToolkitState,
)


class TestDataFrameToolkitState:
    """Tests for the DataFrameToolkitState model."""

    def test_toolkit_state_empty(self) -> None:
        """Given empty references, When instantiated, Then state has empty references list."""
        # Act
        state = DataFrameToolkitState(references=[])

        # Assert
        with check:
            assert state.references == []

    def test_toolkit_state_with_references(self) -> None:
        """Given references list, When instantiated, Then state contains all references."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe("test", df)

        # Act
        state = DataFrameToolkitState(references=[ref])

        # Assert
        with check:
            assert len(state.references) == 1
        with check:
            assert state.references[0].name == "test"

    def test_toolkit_state_json_round_trip(self) -> None:
        """Given state with references, When serialized and deserialized, Then data preserved."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe(
            "test", df, source_query="SELECT * FROM base", parent_ids=["df_00000001"]
        )
        state = DataFrameToolkitState(references=[ref])

        # Act
        json_str = state.model_dump_json()
        restored = DataFrameToolkitState.model_validate_json(json_str)

        # Assert
        with check:
            assert len(restored.references) == 1
        with check:
            assert restored.references[0].name == "test"
        with check:
            assert restored.references[0].source_query == "SELECT * FROM base"

    def test_toolkit_state_to_json_produces_valid_json(self) -> None:
        """Given state with references, When model_dump_json called, Then produces valid JSON string."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe("test", df)
        state = DataFrameToolkitState(references=[ref])

        # Act
        json_str = state.model_dump_json()

        # Assert
        with check:
            assert isinstance(json_str, str)
        # Should be parseable JSON
        parsed = json.loads(json_str)
        with check:
            assert "references" in parsed
        with check:
            assert len(parsed["references"]) == 1

    def test_toolkit_state_to_json_with_indent(self) -> None:
        """Given state, When model_dump_json called with indent, Then output is formatted."""
        # Arrange
        state = DataFrameToolkitState(references=[])

        # Act
        compact = state.model_dump_json()
        formatted = state.model_dump_json(indent=2)

        # Assert
        with check:
            assert "\n" not in compact
        with check:
            assert "\n" in formatted

    def test_toolkit_state_from_json_invalid_raises_error(self) -> None:
        """Given invalid JSON, When model_validate_json called, Then raises ValidationError."""
        # Act/Assert
        with pytest.raises(ValidationError):
            DataFrameToolkitState.model_validate_json("invalid json")

    def test_toolkit_state_serialization_preserves_all_fields(self) -> None:
        """Given reference with all fields, When round-tripped, Then all fields preserved."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ref = DataFrameReference.from_dataframe(
            "test_df",
            df,
            description="A test DataFrame",
            source_query="SELECT * FROM parent WHERE x > 0",
            parent_ids=["df_00000001", "df_00000002"],
        )
        state = DataFrameToolkitState(references=[ref])

        # Act
        json_str = state.model_dump_json()
        restored = DataFrameToolkitState.model_validate_json(json_str)

        # Assert
        restored_ref = restored.references[0]
        with check:
            assert restored_ref.name == "test_df"
        with check:
            assert restored_ref.description == "A test DataFrame"
        with check:
            assert restored_ref.source_query == "SELECT * FROM parent WHERE x > 0"
        with check:
            assert restored_ref.parent_ids == ["df_00000001", "df_00000002"]
        with check:
            assert restored_ref.num_rows == 3
        with check:
            assert restored_ref.num_columns == 2
        with check:
            assert restored_ref.column_names == ["a", "b"]
