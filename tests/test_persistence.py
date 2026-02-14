"""Tests for the persistence module functions."""

# ruff: noqa: S608

from __future__ import annotations

import polars as pl
import pytest
from pytest_check import check

from dfkit.models import (
    ColumnSummary,
    DataFrameReference,
    DataFrameToolkitState,
)
from dfkit.persistence import (
    _compare_column_summaries,
    _reconstruct_derivatives,
    _resolve_dataframe_keys_to_ids,
    _sort_references_by_dependency_order,
    _validate_dataframe_matches_reference,
    _values_nearly_equal,
    restore_registry_from_state,
)
from dfkit.registry import DataFrameRegistry


class TestValuesNearlyEqual:
    """Tests for _values_nearly_equal helper function."""

    def test_values_nearly_equal_both_none_returns_true(self) -> None:
        """Given both values are None, When called, Then returns True."""
        # Arrange
        actual = None
        expected = None

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (None, 1.0),
            (1.0, None),
            (None, "test"),
            ("test", None),
        ],
    )
    def test_values_nearly_equal_one_none_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given one value is None and other is not, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    def test_values_nearly_equal_equal_floats_returns_true(self) -> None:
        """Given two equal float values, When called, Then returns True."""
        # Arrange
        actual = 42.0
        expected = 42.0

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    def test_values_nearly_equal_floats_within_tolerance_returns_true(
        self,
    ) -> None:
        """Given two floats within relative tolerance, When called, Then returns True."""
        # Arrange - values that differ by a tiny relative amount
        actual = 1.0
        expected = 1.0 + 1e-10  # Within default rel_tol of 1e-9

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected, rel_tol=1e-9)

        # Assert
        with check:
            assert result is True

    def test_values_nearly_equal_different_floats_returns_false(self) -> None:
        """Given two different float values outside tolerance, When called, Then returns False."""
        # Arrange
        actual = 1.0
        expected = 2.0

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is False

    def test_values_nearly_equal_equal_strings_returns_true(self) -> None:
        """Given two equal string values, When called, Then returns True."""
        # Arrange
        actual = "hello"
        expected = "hello"

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    def test_values_nearly_equal_different_strings_returns_false(self) -> None:
        """Given two different string values, When called, Then returns False."""
        # Arrange
        actual = "hello"
        expected = "world"

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is False

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            ("1.0", 1.0),
            (1.0, "1.0"),
        ],
    )
    def test_values_nearly_equal_mixed_types_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given one string and one float, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    def test_values_nearly_equal_both_nan_returns_true(self) -> None:
        """Given both values are NaN, When called, Then returns True."""
        # Arrange
        actual = float("nan")
        expected = float("nan")

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (float("nan"), 1.0),
            (1.0, float("nan")),
        ],
    )
    def test_values_nearly_equal_one_nan_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given one value is NaN and other is not, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    def test_values_nearly_equal_both_true_returns_true(self) -> None:
        """Given both values are True, When called, Then returns True."""
        # Act
        result = _values_nearly_equal(actual=True, expected=True)

        # Assert
        with check:
            assert result is True

    def test_values_nearly_equal_both_false_returns_true(self) -> None:
        """Given both values are False, When called, Then returns True."""
        # Act
        result = _values_nearly_equal(actual=False, expected=False)

        # Assert
        with check:
            assert result is True

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (True, False),
            (False, True),
        ],
    )
    def test_values_nearly_equal_different_bools_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given one True and one False, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (True, 1.0),
            (1.0, True),
            (False, 0.0),
            (0.0, False),
            (True, "True"),
            ("True", True),
        ],
    )
    def test_values_nearly_equal_bool_and_non_bool_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given one bool and one non-bool, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    def test_values_nearly_equal_very_large_floats_returns_true(self) -> None:
        """Given two very large floats within tolerance, When called, Then returns True."""
        # Arrange
        actual = 1e308
        expected = 1e308

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    def test_values_nearly_equal_very_large_floats_different_returns_false(self) -> None:
        """Given two very large floats that differ, When called, Then returns False."""
        # Arrange
        actual = 1e308
        expected = 1e307

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is False

    def test_values_nearly_equal_positive_infinity_returns_true(self) -> None:
        """Given both values are positive infinity, When called, Then returns True."""
        # Arrange
        actual = float("inf")
        expected = float("inf")

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    def test_values_nearly_equal_negative_infinity_returns_true(self) -> None:
        """Given both values are negative infinity, When called, Then returns True."""
        # Arrange
        actual = float("-inf")
        expected = float("-inf")

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (float("inf"), float("-inf")),
            (float("-inf"), float("inf")),
        ],
    )
    def test_values_nearly_equal_opposite_infinities_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given positive and negative infinity, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (float("inf"), 1e308),
            (1e308, float("inf")),
        ],
    )
    def test_values_nearly_equal_infinity_vs_finite_returns_false(
        self, actual: float | str | None, expected: float | str | None
    ) -> None:
        """Given infinity and a finite number, When called, Then returns False.

        Args:
            actual (float | str | None): The actual value to compare.
            expected (float | str | None): The expected value to compare.
        """
        assert _values_nearly_equal(actual=actual, expected=expected) is False

    def test_values_nearly_equal_negative_zero_vs_zero_returns_true(self) -> None:
        """Given -0.0 and 0.0, When called, Then returns True."""
        # Arrange
        actual = -0.0
        expected = 0.0

        # Act
        result = _values_nearly_equal(actual=actual, expected=expected)

        # Assert
        with check:
            assert result is True


class TestCompareColumnSummaries:
    """Tests for _compare_column_summaries helper function."""

    def test_compare_column_summaries_identical_returns_empty_dict(self) -> None:
        """Given two identical column summaries, When compared, Then returns empty dict."""
        # Arrange
        series = pl.Series("test_col", [1, 2, 3, 4, 5])
        summary1 = ColumnSummary.from_series(series)
        summary2 = ColumnSummary.from_series(series)

        # Act
        result = _compare_column_summaries(summary1, summary2)

        # Assert
        with check:
            assert result == {}

    def test_compare_column_summaries_dtype_mismatch_returns_dtype_key(self) -> None:
        """Given summaries with different dtypes, When compared, Then returns dict with dtype key."""
        # Arrange
        series_int = pl.Series("col", [1, 2, 3])
        series_float = pl.Series("col", [1.0, 2.0, 3.0])
        summary_int = ColumnSummary.from_series(series_int)
        summary_float = ColumnSummary.from_series(series_float)

        # Act
        result = _compare_column_summaries(summary_int, summary_float)

        # Assert
        with check:
            assert "dtype" in result
        with check:
            assert result["dtype"][0] == "Int64"
        with check:
            assert result["dtype"][1] == "Float64"

    def test_compare_column_summaries_count_mismatch_returns_count_key(self) -> None:
        """Given summaries with different counts, When compared, Then returns dict with count key."""
        # Arrange
        series_short = pl.Series("col", [1, 2, 3])
        series_long = pl.Series("col", [1, 2, 3, 4, 5])
        summary_short = ColumnSummary.from_series(series_short)
        summary_long = ColumnSummary.from_series(series_long)

        # Act
        result = _compare_column_summaries(summary_short, summary_long)

        # Assert
        with check:
            assert "count" in result
        with check:
            assert result["count"] == (3, 5)

    def test_compare_column_summaries_statistical_mismatch_returns_appropriate_keys(
        self,
    ) -> None:
        """Given summaries with different statistics, When compared, Then returns dict with statistical keys."""
        # Arrange - same count but different values (different min/max/mean)
        series1 = pl.Series("col", [1, 2, 3])
        series2 = pl.Series("col", [10, 20, 30])
        summary1 = ColumnSummary.from_series(series1)
        summary2 = ColumnSummary.from_series(series2)

        # Act
        result = _compare_column_summaries(summary1, summary2)

        # Assert - should have min, max, mean differences
        with check:
            assert "min" in result
        with check:
            assert "max" in result
        with check:
            assert "mean" in result

    def test_compare_column_summaries_null_count_mismatch_returns_null_count_key(self) -> None:
        """Given summaries with different null counts, When compared, Then returns dict with null_count key."""
        # Arrange
        series_no_nulls = pl.Series("col", [1, 2, 3])
        series_with_nulls = pl.Series("col", [1, None, 3])
        summary_no_nulls = ColumnSummary.from_series(series_no_nulls)
        summary_with_nulls = ColumnSummary.from_series(series_with_nulls)

        # Act
        result = _compare_column_summaries(summary_no_nulls, summary_with_nulls)

        # Assert
        with check:
            assert "null_count" in result
        with check:
            assert result["null_count"] == (0, 1)

    def test_compare_column_summaries_unique_count_mismatch_returns_unique_count_key(self) -> None:
        """Given summaries with different unique counts, When compared, Then returns dict with unique_count key."""
        # Arrange
        series_all_unique = pl.Series("col", [1, 2, 3])
        series_with_dupes = pl.Series("col", [1, 1, 1])
        summary_all_unique = ColumnSummary.from_series(series_all_unique)
        summary_with_dupes = ColumnSummary.from_series(series_with_dupes)

        # Act
        result = _compare_column_summaries(summary_all_unique, summary_with_dupes)

        # Assert
        with check:
            assert "unique_count" in result
        with check:
            assert result["unique_count"] == (3, 1)


class TestValidateDataframeMatchesReference:
    """Tests for _validate_dataframe_matches_reference function."""

    def test_validate_dataframe_matches_reference_valid_no_exception(self) -> None:
        """Given DataFrame matching reference, When validated, Then no exception raised."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        reference = DataFrameReference.from_dataframe("test", df)

        # Act/Assert - should not raise
        _validate_dataframe_matches_reference(df, reference)

    def test_validate_dataframe_matches_reference_column_mismatch_raises(self) -> None:
        """Given DataFrame with different columns, When validated, Then raises ValueError."""
        # Arrange
        df_original = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        reference = DataFrameReference.from_dataframe("test", df_original)
        df_different = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Act/Assert
        with pytest.raises(ValueError, match="column mismatch"):
            _validate_dataframe_matches_reference(df_different, reference)

    def test_validate_dataframe_matches_reference_shape_mismatch_raises(self) -> None:
        """Given DataFrame with different shape, When validated, Then raises ValueError."""
        # Arrange
        df_original = pl.DataFrame({"a": [1, 2, 3]})
        reference = DataFrameReference.from_dataframe("test", df_original)
        df_different = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

        # Act/Assert
        with pytest.raises(ValueError, match="shape mismatch"):
            _validate_dataframe_matches_reference(df_different, reference)

    def test_validate_dataframe_matches_reference_statistics_mismatch_raises(
        self,
    ) -> None:
        """Given DataFrame with different statistics, When validated, Then raises ValueError."""
        # Arrange
        df_original = pl.DataFrame({"a": [1, 2, 3]})
        reference = DataFrameReference.from_dataframe("test", df_original)
        # Same shape and columns but different values
        df_different = pl.DataFrame({"a": [100, 200, 300]})

        # Act/Assert
        with pytest.raises(ValueError, match="statistics mismatch"):
            _validate_dataframe_matches_reference(df_different, reference)


class TestResolveDataframeKeysToIds:
    """Tests for _resolve_dataframe_keys_to_ids function."""

    def test_resolve_dataframe_keys_to_ids_by_name(self) -> None:
        """Given dataframes keyed by name, When normalized, Then returns ID-keyed mapping."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        names_to_ids = {"users": "df_00000001", "orders": "df_00000002"}
        dataframes = {"users": df}

        # Act
        result = _resolve_dataframe_keys_to_ids(
            dataframes=dataframes,
            names_to_ids=names_to_ids,
        )

        # Assert
        with check:
            assert "df_00000001" in result
        with check:
            assert result["df_00000001"] is df

    def test_resolve_dataframe_keys_to_ids_by_id(self) -> None:
        """Given dataframes keyed by ID, When normalized, Then returns ID-keyed mapping."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        names_to_ids = {"users": "df_00000001", "orders": "df_00000002"}
        dataframes = {"df_00000001": df}

        # Act
        result = _resolve_dataframe_keys_to_ids(
            dataframes=dataframes,
            names_to_ids=names_to_ids,
        )

        # Assert
        with check:
            assert "df_00000001" in result
        with check:
            assert result["df_00000001"] is df

    def test_resolve_dataframe_keys_to_ids_unknown_name_raises(self) -> None:
        """Given unknown name key, When normalized, Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        names_to_ids = {"users": "df_00000001", "orders": "df_00000002"}
        dataframes = {"unknown_name": df}

        # Act/Assert
        with pytest.raises(ValueError, match="Name 'unknown_name' not in state's base references"):
            _resolve_dataframe_keys_to_ids(
                dataframes=dataframes,
                names_to_ids=names_to_ids,
            )

    def test_resolve_dataframe_keys_to_ids_duplicate_name_and_id_raises(self) -> None:
        """Given same base provided by both name and ID, When normalized, Then raises ValueError."""
        # Arrange - "users" resolves to "df_00000001", so both keys target the same ID
        names_to_ids = {"users": "df_00000001", "orders": "df_00000002"}
        df_a = pl.DataFrame({"a": [1, 2, 3]})
        df_b = pl.DataFrame({"a": [4, 5, 6]})
        dataframes = {"users": df_a, "df_00000001": df_b}

        # Act/Assert
        with pytest.raises(ValueError, match="Duplicate"):
            _resolve_dataframe_keys_to_ids(
                dataframes=dataframes,
                names_to_ids=names_to_ids,
            )

    def test_resolve_dataframe_keys_to_ids_unknown_id_raises(self) -> None:
        """Given unknown ID key, When normalized, Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        names_to_ids = {"users": "df_00000001", "orders": "df_00000002"}
        dataframes = {"df_99999999": df}

        # Act/Assert
        with pytest.raises(ValueError, match="ID 'df_99999999' not in state's base references"):
            _resolve_dataframe_keys_to_ids(
                dataframes=dataframes,
                names_to_ids=names_to_ids,
            )

    def test_resolve_dataframe_keys_to_ids_empty_inputs_returns_empty(self) -> None:
        """Given empty dataframes and names_to_ids, When normalized, Then returns empty dict."""
        # Arrange/Act
        result = _resolve_dataframe_keys_to_ids(
            dataframes={},
            names_to_ids={},
        )

        # Assert
        with check:
            assert result == {}


class TestSortReferencesByDependencyOrder:
    """Tests for _sort_references_by_dependency_order function."""

    def test_sort_references_base_only(self) -> None:
        """Given only base references (no parents), When sorted, Then all refs returned."""
        # Arrange
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        ref1 = DataFrameReference.from_dataframe("ref1", df1)
        ref2 = DataFrameReference.from_dataframe("ref2", df2)
        references = [ref1, ref2]

        # Act
        result = _sort_references_by_dependency_order(references)

        # Assert
        with check:
            assert len(result) == 2
        result_ids = {ref.id for ref in result}
        with check:
            assert result_ids == {ref1.id, ref2.id}

    def test_sort_references_chain_dependency(self) -> None:
        """Given chain A -> B -> C, When sorted, Then parents come before children."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Create A (base)
        ref_a = DataFrameReference.from_dataframe("A", df)

        # Create B (depends on A)
        ref_b = _build_derivative_reference(
            ref_id="df_bbbbbbbb",
            name="B",
            parent_ids=[ref_a.id],
            source_query="SELECT * FROM A",
            df=df,
        )

        # Create C (depends on B)
        ref_c = _build_derivative_reference(
            ref_id="df_cccccccc",
            name="C",
            parent_ids=[ref_b.id],
            source_query="SELECT * FROM B",
            df=df,
        )

        # Arrange in reverse order
        references = [ref_c, ref_b, ref_a]

        # Act
        result = _sort_references_by_dependency_order(references)

        # Assert - A should come before B, B before C
        result_ids = [ref.id for ref in result]
        a_index = result_ids.index(ref_a.id)
        b_index = result_ids.index(ref_b.id)
        c_index = result_ids.index(ref_c.id)

        with check:
            assert a_index < b_index, "A should come before B"
        with check:
            assert b_index < c_index, "B should come before C"

    def test_sort_references_diamond_dependency(self) -> None:
        """Given diamond A -> B, A -> C, B -> D, C -> D, When sorted, Then correct topological order."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        col_summary = {"a": ColumnSummary.from_series(df["a"])}

        # Create A (base)
        ref_a = DataFrameReference(
            id="df_aaaaaaaa",
            name="A",
            description="",
            num_rows=3,
            num_columns=1,
            column_names=["a"],
            column_summaries=col_summary,
            parent_ids=[],
        )

        # Create B (depends on A)
        ref_b = _build_derivative_reference(
            ref_id="df_bbbbbbbb",
            name="B",
            parent_ids=[ref_a.id],
            source_query="SELECT * FROM A",
            df=df,
        )

        # Create C (depends on A)
        ref_c = _build_derivative_reference(
            ref_id="df_cccccccc",
            name="C",
            parent_ids=[ref_a.id],
            source_query="SELECT * FROM A",
            df=df,
        )

        # Create D (depends on B and C)
        ref_d = _build_derivative_reference(
            ref_id="df_dddddddd",
            name="D",
            parent_ids=[ref_b.id, ref_c.id],
            source_query="SELECT * FROM B JOIN C",
            df=df,
        )

        # Arrange in any order
        references = [ref_d, ref_b, ref_c, ref_a]

        # Act
        result = _sort_references_by_dependency_order(references)

        # Assert - A before B and C, B and C before D
        result_ids = [ref.id for ref in result]
        a_index = result_ids.index(ref_a.id)
        b_index = result_ids.index(ref_b.id)
        c_index = result_ids.index(ref_c.id)
        d_index = result_ids.index(ref_d.id)

        with check:
            assert a_index < b_index, "A should come before B"
        with check:
            assert a_index < c_index, "A should come before C"
        with check:
            assert b_index < d_index, "B should come before D"
        with check:
            assert c_index < d_index, "C should come before D"

    def test_sort_references_cyclic_dependency_raises_error(self) -> None:
        """Given cyclic dependency A -> B -> A, When sorted, Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Create A (depends on B) - circular
        ref_a = _build_derivative_reference(
            ref_id="df_aaaaaaaa",
            name="A",
            parent_ids=["df_bbbbbbbb"],
            source_query="SELECT * FROM B",
            df=df,
        )

        # Create B (depends on A) - circular
        ref_b = _build_derivative_reference(
            ref_id="df_bbbbbbbb",
            name="B",
            parent_ids=["df_aaaaaaaa"],
            source_query="SELECT * FROM A",
            df=df,
        )

        references = [ref_a, ref_b]

        # Act/Assert
        with pytest.raises(ValueError, match="Cyclic dependency detected"):
            _sort_references_by_dependency_order(references)

    def test_sort_references_unknown_parent_id_raises_error(self) -> None:
        """Given reference with unknown parent_id, When sorted, Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Create A (base)
        ref_a = DataFrameReference(
            id="df_aaaaaaaa",
            name="A",
            description="",
            num_rows=3,
            num_columns=1,
            column_names=["a"],
            column_summaries={"a": ColumnSummary.from_series(df["a"])},
            parent_ids=[],
        )

        # Create B (depends on non-existent reference)
        ref_b = _build_derivative_reference(
            ref_id="df_bbbbbbbb",
            name="B",
            parent_ids=["df_aaaaaaaa", "df_cccccccc"],  # 'df_cccccccc' does not exist
            source_query="SELECT * FROM A JOIN missing",
            df=df,
        )

        references = [ref_a, ref_b]

        # Act/Assert
        with pytest.raises(ValueError, match=r"unknown parent_ids.*df_cccccccc"):
            _sort_references_by_dependency_order(references)


class TestRestoreRegistryFromState:
    """Tests for restore_registry_from_state function."""

    def test_restore_registry_from_state_single_base(self) -> None:
        """Given state with single base DataFrame, When restored, Then context has DataFrame."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe("test", df)
        state = DataFrameToolkitState(references=[ref])

        # Act
        registry = restore_registry_from_state(state=state, base_dataframes={"test": df})

        # Assert
        with check:
            assert len(registry.references) == 1
        with check:
            assert ref.id in registry.references
        with check:
            assert ref.id in registry.context

    def test_restore_registry_from_state_multiple_bases(self) -> None:
        """Given state with multiple base DataFrames, When restored, Then all in context."""
        # Arrange
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        ref1 = DataFrameReference.from_dataframe("first", df1)
        ref2 = DataFrameReference.from_dataframe("second", df2)
        state = DataFrameToolkitState(references=[ref1, ref2])

        # Act
        registry = restore_registry_from_state(state=state, base_dataframes={"first": df1, "second": df2})

        # Assert
        with check:
            assert len(registry.references) == 2
        with check:
            assert ref1.id in registry.references
        with check:
            assert ref2.id in registry.references
        with check:
            assert ref1.id in registry.context
        with check:
            assert ref2.id in registry.context

    def test_restore_registry_from_state_with_derivative(self) -> None:
        """Given state with derivative, When restored, Then derivative reconstructed."""
        # Arrange
        base_df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        base_ref = DataFrameReference.from_dataframe("base", base_df)

        # Create derivative reference that filters to a < 3
        derived_df = pl.DataFrame({"a": [1, 2]})
        derived_ref = _build_derivative_reference(
            ref_id="df_de11ed11",
            name="derived",
            description="Filtered data",
            parent_ids=[base_ref.id],
            source_query=f"SELECT * FROM {base_ref.id} WHERE a < 3",
            df=derived_df,
        )

        state = DataFrameToolkitState(references=[base_ref, derived_ref])

        # Act
        registry = restore_registry_from_state(state=state, base_dataframes={"base": base_df})

        # Assert
        with check:
            assert len(registry.references) == 2
        with check:
            assert base_ref.id in registry.references
        with check:
            assert derived_ref.id in registry.references
        with check:
            assert derived_ref.id in registry.context

        # Verify reconstructed data
        reconstructed = registry.context.get_dataframe(derived_ref.id)
        with check:
            assert reconstructed.shape == (2, 1)
        with check:
            assert set(reconstructed["a"].to_list()) == {1, 2}

    def test_restore_registry_from_state_missing_base_raises(self) -> None:
        """Given state requiring base not provided, When restored, Then raises ValueError."""
        # Arrange
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        ref1 = DataFrameReference.from_dataframe("first", df1)
        ref2 = DataFrameReference.from_dataframe("second", df2)
        state = DataFrameToolkitState(references=[ref1, ref2])

        # Act/Assert - only provide one of two required bases
        with pytest.raises(ValueError, match="Missing base dataframes"):
            restore_registry_from_state(state=state, base_dataframes={"first": df1})

    def test_restore_registry_from_state_empty_state(self) -> None:
        """Given empty state with no references, When restored, Then registry is empty."""
        # Arrange
        state = DataFrameToolkitState(references=[])

        # Act
        registry = restore_registry_from_state(state=state, base_dataframes={})

        # Assert
        with check:
            assert len(registry.references) == 0

    def test_restore_registry_from_state_extra_base_raises(self) -> None:
        """Given extra base dataframe not in state, When restored, Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = DataFrameReference.from_dataframe("users", df)
        state = DataFrameToolkitState(references=[ref])

        extra_df = pl.DataFrame({"b": [4, 5, 6]})

        # Act/Assert - provide required base plus an extra one not in state
        with pytest.raises(ValueError, match="not in state's base references"):
            restore_registry_from_state(
                state=state,
                base_dataframes={"users": df, "unknown_extra": extra_df},
            )


class TestReconstructDerivatives:
    """Tests for _reconstruct_derivatives function."""

    def test_reconstruct_derivatives_validation_failure_propagates_value_error(self) -> None:
        """Given derivative SQL produces data mismatching saved statistics, When reconstructed, Then raises ValueError.

        Simulates a non-deterministic query scenario where replay produces different
        data than the original execution (e.g., ORDER BY without LIMIT affecting
        percentile statistics).
        """
        # Arrange - register base dataframe
        base_df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        base_ref = DataFrameReference.from_dataframe("base", base_df)
        registry = DataFrameRegistry()
        registry.register(base_ref, base_df)

        # Create derivative reference with statistics from [1, 2] (original result),
        # but source_query that produces [4, 5] (non-deterministic replay).
        # Same shape, different values -> statistics mismatch on min/max/mean.
        original_result = pl.DataFrame({"a": [1, 2]})
        derived_ref = _build_derivative_reference(
            ref_id="df_de11ed11",
            name="derived",
            description="Non-deterministic derivative",
            parent_ids=[base_ref.id],
            source_query=f"SELECT * FROM {base_ref.id} WHERE a > 3",
            df=original_result,
        )

        state = DataFrameToolkitState(references=[base_ref, derived_ref])

        # Act/Assert - validation catches the statistics mismatch
        with pytest.raises(ValueError, match="statistics mismatch"):
            _reconstruct_derivatives(state, registry)


def _build_derivative_reference(
    *,
    ref_id: str,
    name: str,
    parent_ids: list[str],
    source_query: str,
    df: pl.DataFrame | None = None,
    description: str = "",
) -> DataFrameReference:
    """Build a derivative DataFrameReference.

    Args:
        ref_id (str): The unique identifier for this reference.
        name (str): Human-readable name for the reference.
        parent_ids (list[str]): IDs of parent references this derives from.
        source_query (str): SQL query used to derive this reference.
        df (pl.DataFrame | None): DataFrame to derive schema/statistics from.
            Defaults to a simple 3-row frame.
        description (str): Optional description text.

    Returns:
        DataFrameReference: A reference configured as a derivative.
    """
    if df is None:
        df = pl.DataFrame({"a": [1, 2, 3]})
    return DataFrameReference(
        id=ref_id,
        name=name,
        description=description,
        num_rows=df.shape[0],
        num_columns=df.shape[1],
        column_names=df.columns,
        column_summaries={col: ColumnSummary.from_series(df[col]) for col in df.columns},
        parent_ids=parent_ids,
        source_query=source_query,
    )
