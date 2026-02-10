"""Tests for DataFrameToolkit state export, reconstruction, and conversation resumption."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from pytest_check import check

from dfkit.models import (
    DataFrameReference,
    DataFrameToolkitState,
)
from dfkit.toolkit import DataFrameToolkit


class TestExportState:
    """Tests for DataFrameToolkit.export_state method."""

    def test_export_state_empty_toolkit(self) -> None:
        """Given empty toolkit, When exported, Then state has empty references."""
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        state = toolkit.export_state()

        # Assert
        with check:
            assert len(state.references) == 0

    def test_export_state_with_registered_dataframes(self) -> None:
        """Given toolkit with registered dataframes, When exported, Then all references included."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("users", pl.DataFrame({"id": [1, 2]}))
        toolkit.register_dataframe("orders", pl.DataFrame({"order_id": [10, 20]}))

        # Act
        state = toolkit.export_state()

        # Assert
        with check:
            assert len(state.references) == 2
        names = {ref.name for ref in state.references}
        with check:
            assert names == {"users", "orders"}


class TestFromState:
    """Tests for DataFrameToolkit.from_state classmethod."""

    def test_from_state_by_name(self) -> None:
        """Given state and base dataframes by name, When from_state called, Then toolkit reconstructed."""
        # Arrange - create original toolkit and export state
        original = DataFrameToolkit()
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        base_ref = original.register_dataframe("base", base_df)
        state = original.export_state()

        # Act - restore from state by name
        new_toolkit = DataFrameToolkit.from_state(state, {"base": base_df})

        # Assert
        with check:
            assert len(new_toolkit.references) == 1
        with check:
            assert new_toolkit.references[0].name == "base"
        with check:
            assert new_toolkit.references[0].id == base_ref.id

    def test_from_state_by_id(self) -> None:
        """Given state and base dataframes by ID, When from_state called, Then toolkit reconstructed."""
        # Arrange
        original = DataFrameToolkit()
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        base_ref = original.register_dataframe("base", base_df)
        state = original.export_state()

        # Act - restore from state by ID
        new_toolkit = DataFrameToolkit.from_state(state, {base_ref.id: base_df})

        # Assert
        with check:
            assert len(new_toolkit.references) == 1
        with check:
            assert new_toolkit.references[0].id == base_ref.id

    def test_from_state_mixed_name_and_id(self) -> None:
        """Given state and base dataframes by mixed name/ID, When from_state called, Then toolkit reconstructed."""
        # Arrange
        original = DataFrameToolkit()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        ref1 = original.register_dataframe("first", df1)
        ref2 = original.register_dataframe("second", df2)
        state = original.export_state()

        # Act - restore with mixed keys
        new_toolkit = DataFrameToolkit.from_state(state, {"first": df1, ref2.id: df2})

        # Assert
        with check:
            assert len(new_toolkit.references) == 2
        ref_ids = {ref.id for ref in new_toolkit.references}
        with check:
            assert ref_ids == {ref1.id, ref2.id}

    def test_from_state_with_derivatives(self) -> None:
        """Given state with derivative, When from_state called, Then derivative reconstructed."""
        # Arrange - original toolkit with derivative
        original = DataFrameToolkit()
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        base_ref = original.register_dataframe("base", base_df)

        # Create derivative reference with source_query
        derived_ref = DataFrameReference.from_dataframe(
            "derived",
            pl.DataFrame({"a": [1, 2]}),
            source_query=f"SELECT * FROM {base_ref.id} WHERE a < 3",  # noqa: S608
            parent_ids=[base_ref.id],
        )
        state = DataFrameToolkitState(references=[base_ref, derived_ref])

        # Act
        new_toolkit = DataFrameToolkit.from_state(state, {"base": base_df})

        # Assert
        with check:
            assert len(new_toolkit.references) == 2
        names = {ref.name for ref in new_toolkit.references}
        with check:
            assert names == {"base", "derived"}

    def test_from_state_missing_base_raises_error(self) -> None:
        """Given state requiring base not provided, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        original.register_dataframe("first", df1)
        original.register_dataframe("second", df2)
        state = original.export_state()

        # Act/Assert - only provide one of two required bases
        with pytest.raises(ValueError, match="Missing base dataframes"):
            DataFrameToolkit.from_state(state, {"first": df1})

    def test_from_state_unknown_name_raises_error(self) -> None:
        """Given name not in state, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        original.register_dataframe("base", base_df)
        state = original.export_state()

        # Act/Assert
        with pytest.raises(ValueError, match="Name 'wrong_name' not in state's base references"):
            DataFrameToolkit.from_state(state, {"wrong_name": base_df})

    def test_from_state_unknown_id_raises_error(self) -> None:
        """Given ID not in state, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        original.register_dataframe("base", base_df)
        state = original.export_state()

        # Act/Assert
        with pytest.raises(ValueError, match="ID 'df_00000000' not in state's base references"):
            DataFrameToolkit.from_state(state, {"df_00000000": base_df})

    def test_from_state_preserves_metadata(self) -> None:
        """Given state with metadata, When from_state called, Then metadata preserved."""
        # Arrange
        original = DataFrameToolkit()
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        original.register_dataframe(
            "base",
            base_df,
            description="Test description",
            column_descriptions={"a": "Column A"},
        )
        state = original.export_state()

        # Act
        new_toolkit = DataFrameToolkit.from_state(state, {"base": base_df})

        # Assert
        restored_ref = new_toolkit.references[0]
        with check:
            assert restored_ref.description == "Test description"
        with check:
            assert restored_ref.column_summaries["a"].description == "Column A"


class TestFromStateErrorHandling:
    """Tests for error handling in DataFrameToolkit.from_state."""

    def test_from_state_column_name_mismatch_raises_error(self) -> None:
        """Given DataFrame with different columns, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Different column names
        different_df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Act/Assert
        with pytest.raises(ValueError, match=r"column mismatch.*Expected.*a.*b.*got.*x.*y"):
            DataFrameToolkit.from_state(state, {"data": different_df})

    def test_from_state_row_count_mismatch_raises_error(self) -> None:
        """Given DataFrame with different row count, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, 2, 3]})
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Different row count
        different_df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

        # Act/Assert
        with pytest.raises(ValueError, match=r"shape mismatch.*Expected.*3.*got.*5"):
            DataFrameToolkit.from_state(state, {"data": different_df})

    def test_from_state_column_count_mismatch_raises_error(self) -> None:
        """Given DataFrame with different column count, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Different column count (fewer columns)
        different_df = pl.DataFrame({"a": [1, 2, 3]})

        # Act/Assert - shape is checked first, so fails on shape mismatch
        with pytest.raises(ValueError, match="shape mismatch"):
            DataFrameToolkit.from_state(state, {"data": different_df})

    def test_from_state_sql_error_clear_message(self) -> None:
        """Given invalid SQL in source_query, When from_state called, Then clear error message."""
        # Arrange - create state with a derivative that has invalid SQL
        base_df = pl.DataFrame({"a": [1, 2, 3]})
        base_ref = DataFrameReference.from_dataframe("base", base_df)

        derived_ref = DataFrameReference.from_dataframe(
            "derived",
            pl.DataFrame({"a": [1]}),
            source_query="SELECT * FROM nonexistent_table",  # Invalid SQL
            parent_ids=[base_ref.id],
        )

        state = DataFrameToolkitState(references=[base_ref, derived_ref])

        # Act/Assert
        with pytest.raises(ValueError, match=r"SQL execution failed.*derived"):
            DataFrameToolkit.from_state(state, {"base": base_df})

    def test_from_state_data_values_changed_raises_error(self) -> None:
        """Given DataFrame with different data values, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, 2, 3]})
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Same shape and columns, but different values
        different_df = pl.DataFrame({"a": [100, 200, 300]})

        # Act/Assert - should fail on column summary mismatch (min, max, mean differ)
        with pytest.raises(ValueError, match=r"statistics mismatch.*Differences"):
            DataFrameToolkit.from_state(state, {"data": different_df})

    def test_from_state_dtype_changed_raises_error(self) -> None:
        """Given DataFrame with different dtype, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, 2, 3]})  # Int64
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Same values but different dtype
        different_df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})  # Float64

        # Act/Assert
        with pytest.raises(ValueError, match=r"statistics mismatch.*dtype"):
            DataFrameToolkit.from_state(state, {"data": different_df})

    def test_from_state_null_count_changed_raises_error(self) -> None:
        """Given DataFrame with different null count, When from_state called, Then raises ValueError."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, None, 3]})
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Same shape but different null count
        different_df = pl.DataFrame({"a": [1, 2, 3]})  # No nulls

        # Act/Assert
        with pytest.raises(ValueError, match=r"statistics mismatch.*null_count"):
            DataFrameToolkit.from_state(state, {"data": different_df})

    def test_from_state_identical_data_passes(self) -> None:
        """Given identical DataFrame data, When from_state called, Then validation passes."""
        # Arrange
        original = DataFrameToolkit()
        original_df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        original.register_dataframe("data", original_df)
        state = original.export_state()

        # Exact same data
        same_df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        # Act
        new_toolkit = DataFrameToolkit.from_state(state, {"data": same_df})

        # Assert - should succeed without error
        with check:
            assert len(new_toolkit.references) == 1


class TestConversationResumptionScenarios:
    """End-to-end tests for conversation resumption workflow using from_state."""

    def test_conversation_resumption_scenario(self) -> None:
        """Full workflow: create toolkit, execute SQL, export, reconstruct with from_state."""
        # === Session 1: Original conversation ===
        original_toolkit = DataFrameToolkit()

        # Register base dataframe
        base_df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [85, 92, 78, 95, 88],
        })
        base_ref = original_toolkit.register_dataframe("students", base_df)

        # Execute SQL to create derivative (using public API)
        derived_ref = original_toolkit.execute_sql(
            query=f"SELECT id, name, score FROM {base_ref.id} WHERE score >= 85",  # noqa: S608
            result_name="high_scorers",
            result_description="Students with score >= 85",
        )
        assert isinstance(derived_ref, DataFrameReference)

        # Export state (would be persisted to conversation thread)
        state = original_toolkit.export_state()
        state_json = state.model_dump_json()

        # === Session 2: Resume conversation using from_state ===
        restored_state = DataFrameToolkitState.model_validate_json(state_json)
        new_toolkit = DataFrameToolkit.from_state(restored_state, {"students": base_df})

        # === Verify reconstruction ===
        with check:
            assert len(new_toolkit.references) == 2

        ref_names = {ref.name for ref in new_toolkit.references}
        with check:
            assert ref_names == {"students", "high_scorers"}

        # Verify reconstructed derivative is accessible and has correct shape
        restored_ref = new_toolkit.get_dataframe_reference(derived_ref.id)
        assert isinstance(restored_ref, DataFrameReference)
        with check:
            assert restored_ref.num_rows == 4  # 4 students with score >= 85
        with check:
            assert restored_ref.num_columns == 3

    def test_multi_level_derivation_reconstruction(self) -> None:
        """Test A -> B -> C chain reconstruction using from_state."""
        # Setup base data
        original_toolkit = DataFrameToolkit()
        a_df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })
        a_ref = original_toolkit.register_dataframe("A", a_df)

        # Create B from A (using public API)
        b_ref = original_toolkit.execute_sql(
            query=f"SELECT x, y FROM {a_ref.id} WHERE x <= 5",  # noqa: S608
            result_name="B",
        )
        assert isinstance(b_ref, DataFrameReference)

        # Create C from B (using public API)
        c_ref = original_toolkit.execute_sql(
            query=f"SELECT x, y FROM {b_ref.id} WHERE x <= 2",  # noqa: S608
            result_name="C",
        )
        assert isinstance(c_ref, DataFrameReference)

        # Export and restore using from_state
        state = original_toolkit.export_state()
        new_toolkit = DataFrameToolkit.from_state(state, {"A": a_df})

        # Verify
        with check:
            assert len(new_toolkit.references) == 3

        ref_names = {ref.name for ref in new_toolkit.references}
        with check:
            assert ref_names == {"A", "B", "C"}

        # Verify reconstructed C data
        restored_c = new_toolkit.get_dataframe_reference(c_ref.id)
        assert isinstance(restored_c, DataFrameReference)
        with check:
            assert restored_c.num_rows == 2
        with check:
            assert restored_c.num_columns == 2

    def test_join_reconstruction(self) -> None:
        """Test reconstruction of derivatives from JOINs using from_state."""
        # Setup two base tables
        original_toolkit = DataFrameToolkit()

        users_df = pl.DataFrame({
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        users_ref = original_toolkit.register_dataframe("users", users_df)

        orders_df = pl.DataFrame({
            "order_id": [101, 102, 103, 104],
            "user_id": [1, 1, 2, 3],
            "amount": [50, 75, 100, 25],
        })
        orders_ref = original_toolkit.register_dataframe("orders", orders_df)

        # Create derived table from JOIN (using public API)
        joined_ref = original_toolkit.execute_sql(
            query=f"""
            SELECT u.name, o.order_id, o.amount
            FROM {users_ref.id} u
            JOIN {orders_ref.id} o ON u.user_id = o.user_id
            """,  # noqa: S608
            result_name="user_orders",
        )
        assert isinstance(joined_ref, DataFrameReference)

        # Export and restore using from_state
        state = original_toolkit.export_state()
        new_toolkit = DataFrameToolkit.from_state(
            state,
            {"users": users_df, "orders": orders_df},
        )

        # Verify
        with check:
            assert len(new_toolkit.references) == 3

        ref_names = {ref.name for ref in new_toolkit.references}
        with check:
            assert ref_names == {"users", "orders", "user_orders"}

        # Verify reconstructed data
        restored_join = new_toolkit.get_dataframe_reference(joined_ref.id)
        assert isinstance(restored_join, DataFrameReference)
        with check:
            assert restored_join.num_rows == 4
        with check:
            assert restored_join.num_columns == 3

    def test_null_heavy_dataframe_reconstruction(self) -> None:
        """Test reconstruction with null-heavy DataFrames and date columns."""
        original_toolkit = DataFrameToolkit()

        # Base dataframe with nulls across multiple columns and date data
        events_df = pl.DataFrame({
            "event_id": [1, 2, 3, 4, 5, 6],
            "event_date": [
                date(2025, 1, 15),
                date(2025, 3, 22),
                None,
                date(2025, 6, 1),
                None,
                date(2025, 12, 31),
            ],
            "category": ["sales", None, "support", None, None, "sales"],
            "revenue": [100.5, None, None, 250.0, None, 75.25],
        })
        events_ref = original_toolkit.register_dataframe("events", events_df)

        # Derive: filter to rows with non-null revenue (using public API)
        derived_ref = original_toolkit.execute_sql(
            query=f"SELECT event_id, event_date, category, revenue FROM {events_ref.id} WHERE revenue IS NOT NULL",  # noqa: S608
            result_name="revenue_events",
        )
        assert isinstance(derived_ref, DataFrameReference)

        # Export and restore
        state = original_toolkit.export_state()
        new_toolkit = DataFrameToolkit.from_state(state, {"events": events_df})

        # Verify
        with check:
            assert len(new_toolkit.references) == 2

        restored_ref = new_toolkit.get_dataframe_reference(derived_ref.id)
        assert isinstance(restored_ref, DataFrameReference)
        with check:
            assert restored_ref.num_rows == 3  # 3 rows with non-null revenue
        with check:
            assert restored_ref.num_columns == 4
