"""Tests for DataFrameToolkit registration, lookup, and tool operations."""

from __future__ import annotations

import polars as pl
import pytest
from pytest_check import check

from dfkit.models import (
    DataFrameReference,
    ToolCallError,
)
from dfkit.toolkit import DataFrameToolkit


class TestDataFrameToolkitInit:
    """Tests for DataFrameToolkit initialization."""

    def test_init_empty_references(self) -> None:
        """Given new toolkit, When checked, Then references is empty."""
        # Arrange/Act
        toolkit = DataFrameToolkit()

        # Assert - use public API (references property)
        with check:
            assert len(toolkit.references) == 0


class TestRegisterDataFrame:
    """Tests for DataFrameToolkit.register_dataframe method."""

    def test_register_success(self) -> None:
        """Given valid DataFrame, When registered, Then DataFrameReference returned with metadata."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Act
        reference = toolkit.register_dataframe(
            "my_dataframe",
            df,
            description="Test DataFrame for unit tests",
            column_descriptions={"a": "First column", "b": "Second column"},
        )

        # Assert
        with check:
            assert isinstance(reference, DataFrameReference)
        with check:
            assert reference.name == "my_dataframe"
        with check:
            assert reference.description == "Test DataFrame for unit tests"
        with check:
            assert reference.num_rows == 3
        with check:
            assert reference.num_columns == 2
        with check:
            assert reference.column_names == ["a", "b"]

    def test_register_stores_in_context(self) -> None:
        """Given DataFrame, When registered, Then context has it."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Act
        reference = toolkit.register_dataframe("stored_df", df)

        # Assert - use public API (references property)
        with check:
            assert reference in toolkit.references
        with check:
            assert any(ref.name == "stored_df" for ref in toolkit.references)

    def test_register_duplicate_name_error(self) -> None:
        """Given existing name, When registered, Then ValueError raised."""
        # Arrange
        toolkit = DataFrameToolkit()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"x": [10, 20, 30]})
        toolkit.register_dataframe("duplicate_name", df1)

        # Act/Assert
        with pytest.raises(ValueError, match="DataFrame 'duplicate_name' is already registered"):
            toolkit.register_dataframe("duplicate_name", df2)

    def test_register_name_matching_id_pattern_rejected(self) -> None:
        """Given name matching ID pattern, When registered, Then ValueError raised."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act/Assert
        with pytest.raises(ValueError, match="cannot match ID pattern"):
            toolkit.register_dataframe("df_1a2b3c4d", df)

    def test_register_name_similar_to_id_but_not_matching_allowed(self) -> None:
        """Given name similar to but not matching ID pattern, When registered, Then succeeds."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act - these should NOT match the pattern df_[0-9a-f]{8}
        ref1 = toolkit.register_dataframe("df_sales", df)  # Not 8 hex chars
        ref2 = toolkit.register_dataframe("df_12345678901", pl.DataFrame({"b": [1]}))  # Too long
        ref3 = toolkit.register_dataframe("dataframe_1a2b3c4d", pl.DataFrame({"c": [1]}))  # Wrong prefix

        # Assert
        with check:
            assert ref1.name == "df_sales"
        with check:
            assert ref2.name == "df_12345678901"
        with check:
            assert ref3.name == "dataframe_1a2b3c4d"


class TestRegisterDataFrames:
    """Tests for DataFrameToolkit.register_dataframes method."""

    def test_register_dataframes_success(self) -> None:
        """Given multiple DataFrames, When registered, Then all references created."""
        # Arrange
        toolkit = DataFrameToolkit()
        dfs = {
            "df1": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "df2": pl.DataFrame({"x": ["foo", "bar"], "y": [1.0, 2.0]}),
        }

        # Act
        references = toolkit.register_dataframes(dfs)

        # Assert - use public API (references property) instead of _references
        registered_names = {ref.name for ref in toolkit.references}
        with check:
            assert len(references) == 2
        with check:
            assert "df1" in registered_names
        with check:
            assert "df2" in registered_names

    def test_register_dataframes_returns_all_refs(self) -> None:
        """Given multiple DataFrames, When registered, Then returns list of references."""
        # Arrange
        toolkit = DataFrameToolkit()
        dfs = {
            "users": pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
            "orders": pl.DataFrame({"order_id": [100], "user_id": [1]}),
        }
        descriptions = {"users": "User accounts", "orders": "User orders"}

        # Act
        references = toolkit.register_dataframes(dfs, descriptions=descriptions)

        # Assert
        with check:
            assert isinstance(references, list)
        with check:
            assert all(isinstance(ref, DataFrameReference) for ref in references)
        with check:
            assert len(references) == 2

        # Verify each reference has correct metadata
        ref_by_name = {ref.name: ref for ref in references}
        with check:
            assert ref_by_name["users"].description == "User accounts"
        with check:
            assert ref_by_name["orders"].description == "User orders"

    def test_register_dataframes_existing_name_error(self) -> None:
        """Given name already in toolkit, When registered, Then ValueError before any registration."""
        # Arrange
        toolkit = DataFrameToolkit()
        existing_df = pl.DataFrame({"id": [1, 2, 3]})
        toolkit.register_dataframe("existing", existing_df)

        # Attempt to register batch with conflicting name
        new_dfs = {
            "existing": pl.DataFrame({"x": [10, 20]}),
            "brand_new": pl.DataFrame({"y": [30, 40]}),
        }

        # Act/Assert - should fail before registering any
        with pytest.raises(ValueError, match="DataFrame 'existing' is already registered"):
            toolkit.register_dataframes(new_dfs)

        # Verify atomicity: neither DataFrame should be registered
        registered_names = {ref.name for ref in toolkit.references}
        with check:
            assert "brand_new" not in registered_names
        with check:
            assert len(toolkit.references) == 1  # Only the original

    def test_register_dataframes_name_matching_id_pattern_rejected(self) -> None:
        """Given name matching ID pattern in batch, When registered, Then ValueError before any registration."""
        # Arrange
        toolkit = DataFrameToolkit()
        dfs = {
            "valid_name": pl.DataFrame({"a": [1, 2]}),
            "df_abcd1234": pl.DataFrame({"b": [3, 4]}),  # Matches ID pattern
        }

        # Act/Assert
        with pytest.raises(ValueError, match="cannot match ID pattern"):
            toolkit.register_dataframes(dfs)

        # Verify atomicity: no DataFrames registered
        with check:
            assert len(toolkit.references) == 0


class TestUnregisterDataFrame:
    """Tests for DataFrameToolkit.unregister_dataframe method."""

    def test_unregister_success(self) -> None:
        """Given registered name, When unregistered, Then removed."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        toolkit.register_dataframe("to_remove", df)

        # Act
        toolkit.unregister_dataframe("to_remove")

        # Assert - use public API (references property)
        registered_names = {ref.name for ref in toolkit.references}
        with check:
            assert "to_remove" not in registered_names
        with check:
            assert len(toolkit.references) == 0

    def test_unregister_not_found_error(self) -> None:
        """Given unknown name, When unregistered, Then KeyError raised."""
        # Arrange
        toolkit = DataFrameToolkit()

        # Act/Assert
        with pytest.raises(KeyError, match="DataFrame 'nonexistent' is not registered"):
            toolkit.unregister_dataframe("nonexistent")


class TestGetDataFrameId:
    """Tests for DataFrameToolkit.get_dataframe_id method."""

    def test_get_id_success(self) -> None:
        """Given registered name, When called, Then returns DataFrameId."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        reference = toolkit.register_dataframe("my_data", df)

        # Act
        result = toolkit.get_dataframe_id("my_data")

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert result.startswith("df_")
        with check:
            assert result == reference.id

    def test_get_id_not_found(self) -> None:
        """Given unknown name, When called, Then returns ToolCallError."""
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        result = toolkit.get_dataframe_id("nonexistent")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "DataFrameNotFound"
        with check:
            assert "nonexistent" in result.message

    def test_get_id_error_has_available_names(self) -> None:
        """Given unknown name with other DataFrames registered, When called, Then error has available names."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("users", pl.DataFrame({"id": [1, 2]}))
        toolkit.register_dataframe("orders", pl.DataFrame({"id": [10, 20]}))

        # Act
        result = toolkit.get_dataframe_id("unknown_table")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert "available_names" in result.details
        available_names = result.details["available_names"]
        assert isinstance(available_names, list)
        with check:
            assert set(available_names) == {"users", "orders"}

    def test_get_id_with_id_input_returns_invalid_argument_error(self) -> None:
        """Given ID instead of name, When called, Then returns InvalidArgument error with guidance."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("sales", pl.DataFrame({"amount": [100, 200]}))

        # Act
        result = toolkit.get_dataframe_id("df_1a2b3c4d")  # This is an ID, not a name

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "already an ID" in result.message
        with check:
            assert "Use the name" in result.message
        with check:
            assert "available_names" in result.details

    def test_get_id_with_actual_registered_id_returns_invalid_argument_error(self) -> None:
        """Given actual registered ID, When called, Then returns InvalidArgument error (not the ID)."""
        # Arrange
        toolkit = DataFrameToolkit()
        ref = toolkit.register_dataframe("sales", pl.DataFrame({"amount": [100, 200]}))
        actual_id = ref.id  # e.g., "df_a1b2c3d4"

        # Act
        result = toolkit.get_dataframe_id(actual_id)

        # Assert - should return error, not the ID itself
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "already an ID" in result.message


class TestGetDataFrameReference:
    """Tests for DataFrameToolkit.get_dataframe_reference method."""

    def test_get_reference_by_name_returns_dataframe_reference(self) -> None:
        """Given registered name, When called, Then returns DataFrameReference."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected_reference = toolkit.register_dataframe(
            "my_data",
            df,
            description="Test data",
        )

        # Act
        result = toolkit.get_dataframe_reference("my_data")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result == expected_reference
        with check:
            assert result.name == "my_data"
        with check:
            assert result.description == "Test data"

    def test_get_reference_by_id_returns_dataframe_reference(self) -> None:
        """Given registered ID, When called, Then returns DataFrameReference."""
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"x": [10, 20], "y": ["foo", "bar"]})
        expected_reference = toolkit.register_dataframe("lookup_by_id", df)
        dataframe_id = expected_reference.id

        # Act
        result = toolkit.get_dataframe_reference(dataframe_id)

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result == expected_reference
        with check:
            assert result.id == dataframe_id
        with check:
            assert result.name == "lookup_by_id"

    def test_get_reference_not_found_returns_tool_call_error(self) -> None:
        """Given unknown identifier, When called, Then returns ToolCallError."""
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        result = toolkit.get_dataframe_reference("nonexistent")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "DataFrameNotFound"
        with check:
            assert "nonexistent" in result.message
        with check:
            assert "not found by name or ID" in result.message

    def test_get_reference_error_has_both_names_and_ids(self) -> None:
        """Given unknown identifier with registered DataFrames, When called, Then error has available names AND IDs."""
        # Arrange
        toolkit = DataFrameToolkit()
        ref_users = toolkit.register_dataframe("users", pl.DataFrame({"id": [1, 2]}))
        ref_orders = toolkit.register_dataframe("orders", pl.DataFrame({"id": [10, 20]}))

        # Act
        result = toolkit.get_dataframe_reference("unknown_table")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert "available_names" in result.details
        with check:
            assert "available_ids" in result.details

        # Verify available_names contains expected values
        available_names = result.details["available_names"]
        assert isinstance(available_names, list)
        with check:
            assert set(available_names) == {"users", "orders"}

        # Verify available_ids contains expected values
        available_ids = result.details["available_ids"]
        assert isinstance(available_ids, list)
        with check:
            assert set(available_ids) == {ref_users.id, ref_orders.id}
