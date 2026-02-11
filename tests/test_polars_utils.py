"""Tests for polars_utils module."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest
from pytest_check import check

from dfkit.exceptions import ColumnsNotFoundError, DuplicateColumnsError
from dfkit.polars_utils import get_series_description, to_markdown_table


class TestGetSeriesDescription:
    """Test suite for get_series_description function."""

    def test_numeric_integer_series_returns_expected_keys(self) -> None:
        """Given integer series, When called with defaults, Then returns expected statistical keys."""
        # Arrange
        series = pl.Series("values", [10, 20, 30, 40, 50])

        # Act
        result = get_series_description(series)

        # Assert - verify expected keys exist
        expected_keys = {"count", "null_count", "mean", "std", "min", "max", "25%", "50%", "75%"}
        with check:
            assert set(result.keys()) == expected_keys

    def test_numeric_integer_series_returns_correct_values(self) -> None:
        """Given integer series [5, 15, 25, 35, 45, 55], When called, Then returns correct statistical values."""
        # Arrange
        series = pl.Series("values", [5, 15, 25, 35, 45, 55])

        # Act
        result = get_series_description(series)

        # Assert - verify actual values
        with check:
            assert result["count"] == 6.0
        with check:
            assert result["null_count"] == 0.0
        with check:
            assert result["mean"] == 30.0
        with check:
            assert result["min"] == 5.0
        with check:
            assert result["max"] == 55.0
        with check:
            assert result["50%"] == 35.0

    def test_numeric_float_series_returns_expected_keys_and_values(self) -> None:
        """Given float series, When called, Then returns dict with correct keys and float values."""
        # Arrange
        series = pl.Series("scores", [1.5, 2.5, 3.5, 4.5, 5.5])

        # Act
        result = get_series_description(series)

        # Assert - verify return type and expected keys
        with check:
            assert isinstance(result, dict)
        expected_keys = {"count", "null_count", "mean", "std", "min", "max", "25%", "50%", "75%"}
        with check:
            assert set(result.keys()) == expected_keys
        # Assert - verify sample values
        with check:
            assert result["count"] == 5.0
        with check:
            assert result["mean"] == 3.5
        with check:
            assert result["min"] == 1.5
        with check:
            assert result["max"] == 5.5

    def test_string_series_returns_appropriate_stats(self) -> None:
        """Given string series, When called, Then returns dict with count, null_count, min, and max."""
        # Arrange
        series = pl.Series("names", ["Alice", "Bob", "Charlie", "Diana"])

        # Act
        result = get_series_description(series)

        # Assert - string series should have count, null_count, min, max as strings
        with check:
            assert isinstance(result, dict)
        with check:
            assert "count" in result
        with check:
            assert "null_count" in result
        with check:
            assert result["count"] == "4"
        with check:
            assert result["null_count"] == "0"
        with check:
            assert result["min"] == "Alice"
        with check:
            assert result["max"] == "Diana"

    def test_boolean_series_returns_stats(self) -> None:
        """Given boolean series, When called, Then returns dict with count, null_count, and sum."""
        # Arrange
        series = pl.Series("flags", [True, False, True, True, False])

        # Act
        result = get_series_description(series)

        # Assert - boolean series should have count, null_count
        with check:
            assert isinstance(result, dict)
        with check:
            assert "count" in result
        with check:
            assert "null_count" in result
        with check:
            assert result["count"] == 5.0
        with check:
            assert result["null_count"] == 0.0

    def test_series_with_null_values_correctly_reports_null_count(self) -> None:
        """Given numeric series with nulls, When called, Then correctly reports null and non-null counts."""
        # Arrange
        series = pl.Series("values", [10, None, 30, None, 50])

        # Act
        result = get_series_description(series)

        # Assert - count is number of non-null values, null_count is number of nulls
        with check:
            assert result["null_count"] == 2.0
        with check:
            assert result["count"] == 3.0

    def test_custom_percentiles_parameter_works(self) -> None:
        """Given integer series, When called with custom percentiles, Then result contains those keys."""
        # Arrange
        series = pl.Series("values", list(range(1, 101)))

        # Act
        result = get_series_description(series, percentiles=(0.1, 0.9))

        # Assert - verify custom percentile keys exist
        with check:
            assert "10%" in result
        with check:
            assert "90%" in result
        # Assert - default percentiles should not exist
        with check:
            assert "25%" not in result
        with check:
            assert "50%" not in result
        with check:
            assert "75%" not in result

    def test_empty_numeric_series_works_without_error(self) -> None:
        """Given empty numeric series, When called, Then returns dict without error."""
        # Arrange
        series = pl.Series("empty", [], dtype=pl.Int64)

        # Act
        result = get_series_description(series)

        # Assert - should return a dict without error
        with check:
            assert isinstance(result, dict)
        with check:
            assert result["count"] == 0.0

    def test_series_with_all_nulls(self) -> None:
        """Given series with all nulls, When called, Then returns count=0 and correct null_count."""
        # Arrange
        series = pl.Series("all_nulls", [None, None, None, None], dtype=pl.Float64)

        # Act
        result = get_series_description(series)

        # Assert - count is 0 (no non-null values), null_count is 4
        with check:
            assert result["count"] == 0.0
        with check:
            assert result["null_count"] == 4.0
        # Assert - at least count and null_count present
        with check:
            assert {"count", "null_count"} <= set(result.keys())

    def test_single_element_series(self) -> None:
        """Given single-element series, When called, Then returns dict with correct values."""
        # Arrange
        series = pl.Series("single", [42])

        # Act
        result = get_series_description(series)

        # Assert - single element should have consistent stats
        with check:
            assert result["count"] == 1.0
        with check:
            assert result["null_count"] == 0.0
        with check:
            assert result["mean"] == 42.0
        with check:
            assert result["min"] == 42.0
        with check:
            assert result["max"] == 42.0
        with check:
            assert result["50%"] == 42.0

    def test_series_with_negative_numbers(self) -> None:
        """Given series with negative numbers, When called, Then correctly computes statistics."""
        # Arrange
        series = pl.Series("negatives", [-50, -20, 0, 20, 50])

        # Act
        result = get_series_description(series)

        # Assert
        with check:
            assert result["mean"] == 0.0
        with check:
            assert result["min"] == -50.0
        with check:
            assert result["max"] == 50.0
        with check:
            assert result["50%"] == 0.0

    def test_series_with_large_numbers(self) -> None:
        """Given series with large numbers, When called, Then handles large values correctly."""
        # Arrange
        large_values = [1_000_000, 2_000_000, 3_000_000]
        series = pl.Series("large", large_values)

        # Act
        result = get_series_description(series)

        # Assert
        with check:
            assert result["min"] == 1_000_000.0
        with check:
            assert result["max"] == 3_000_000.0
        with check:
            assert result["mean"] == 2_000_000.0


class TestToMarkdownTableValidation:
    """Tests for to_markdown_table input validation and error handling."""

    def test_invalid_columns_raises_columns_not_found_error(self) -> None:
        """Given DataFrame with columns [a, b], When columns=["a", "nonexistent"], Then raises ColumnsNotFoundError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act/Assert
        with pytest.raises(ColumnsNotFoundError, match="Columns not found"):
            to_markdown_table(df, columns=["a", "nonexistent"])

    def test_duplicate_columns_raises_duplicate_columns_error(self) -> None:
        """Given DataFrame, When columns contains duplicates, Then raises DuplicateColumnsError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act/Assert
        with pytest.raises(DuplicateColumnsError, match="Duplicate column names"):
            to_markdown_table(df, columns=["a", "a", "b"])

    def test_invalid_columns_error_message_contains_sorted_extra_columns(self) -> None:
        """Given DataFrame, When invalid columns passed, Then error message contains sorted list of extra columns."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act/Assert
        with pytest.raises(ColumnsNotFoundError) as exc_info:
            to_markdown_table(df, columns=["a", "nonexistent2", "nonexistent1"])

        error_message = str(exc_info.value)
        with check:
            assert "['nonexistent1', 'nonexistent2']" in error_message

    @pytest.mark.parametrize("num_rows", [0, -1, -10])
    def test_num_rows_below_one_raises_value_error(self, num_rows: int) -> None:
        """Given DataFrame, When num_rows < 1, Then raises ValueError.

        Args:
            num_rows (int): The invalid row count to test.
        """
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act/Assert
        with pytest.raises(ValueError, match="num_rows must be at least 1"):
            to_markdown_table(df, num_rows=num_rows)

    def test_seed_without_sample_raises_value_error(self) -> None:
        """Given DataFrame, When seed is provided without sample=True, Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Act/Assert
        with pytest.raises(ValueError, match="seed is only used when sample=True"):
            to_markdown_table(df, seed=42)

    def test_empty_columns_list_raises_value_error(self) -> None:
        """Given DataFrame, When columns=[], Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act / Assert
        with pytest.raises(ValueError, match="columns list must not be empty"):
            to_markdown_table(df, columns=[])


class TestToMarkdownTableColumns:
    """Tests for to_markdown_table column selection behavior."""

    def test_default_includes_all_columns(self) -> None:
        """Given multi-column DataFrame, When called with defaults, Then all columns and data values appear."""
        # Arrange
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.0, 92.3],
        })

        # Act
        result = to_markdown_table(df)

        # Assert - all columns present in header
        with check:
            assert "id" in result
        with check:
            assert "name" in result
        with check:
            assert "score" in result
        # Assert - actual data values appear in output
        with check:
            assert "Alice" in result
        with check:
            assert "Charlie" in result
        with check:
            assert "95.5" in result

    def test_columns_filter_includes_only_specified(self) -> None:
        """Given DataFrame with columns [a, b, c], When columns=["a", "c"], Then output contains only a and c."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        # Act
        result = to_markdown_table(df, columns=["a", "c"])

        # Assert - only specified columns in output
        lines = result.splitlines()
        header_line = lines[0]
        with check:
            assert "a" in header_line
        with check:
            assert "c" in header_line
        with check:
            assert "b" not in header_line

    def test_columns_preserves_order(self) -> None:
        """Given DataFrame, When columns=["c", "a"], Then columns appear in specified order."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        # Act
        result = to_markdown_table(df, columns=["c", "a"])

        # Assert - columns in specified order by parsing pipe-delimited cells
        header_line = result.splitlines()[0]
        column_names = [c.strip() for c in header_line.split("|") if c.strip()]
        with check:
            assert column_names == ["c", "a"]

    def test_columns_with_sample_true(self) -> None:
        """Given DataFrame, When columns and sample=True, Then sampled rows contain only specified columns."""
        # Arrange
        df = pl.DataFrame({
            "id": list(range(100)),
            "name": [f"item_{i}" for i in range(100)],
            "value": list(range(100, 200)),
        })

        # Act
        result = to_markdown_table(df, columns=["id", "value"], num_rows=5, sample=True, seed=42)

        # Assert - only specified columns appear
        lines = result.splitlines()
        header_line = lines[0]
        with check:
            assert "id" in header_line
        with check:
            assert "value" in header_line
        with check:
            assert "name" not in header_line
        # Assert - correct number of data rows
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 5


class TestToMarkdownTableSampling:
    """Tests for to_markdown_table row limiting and sampling behavior."""

    def test_num_rows_limits_output(self) -> None:
        """Given DataFrame with 20 rows, When num_rows=5, Then output is truncated with ellipsis."""
        # Arrange
        df = pl.DataFrame({"value": list(range(20))})

        # Act
        result = to_markdown_table(df, num_rows=5)

        # Assert - output should be truncated (less than full 20 rows) and include ellipsis
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count < 20, "Output should be truncated"
        with check:
            assert "…" in result, "Truncated output should contain ellipsis marker"

    def test_num_rows_larger_than_dataframe(self) -> None:
        """Given DataFrame with 3 rows, When num_rows=10, Then output contains all 3 rows without error."""
        # Arrange
        df = pl.DataFrame({"x": [100, 200, 300]})

        # Act
        result = to_markdown_table(df, num_rows=10)

        # Assert - all rows present
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 3

    def test_sample_true_returns_random_rows(self) -> None:
        """Given DataFrame with 100 rows, When sample=True and num_rows=5, Then output contains 5 data rows."""
        # Arrange
        df = pl.DataFrame({"idx": list(range(100))})

        # Act
        result = to_markdown_table(df, num_rows=5, sample=True)

        # Assert
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 5

    def test_sample_with_seed_is_reproducible(self) -> None:
        """Given same DataFrame, When called twice with sample=True and same seed, Then outputs are identical."""
        # Arrange
        df = pl.DataFrame({"value": list(range(50))})

        # Act
        result1 = to_markdown_table(df, num_rows=5, sample=True, seed=123)
        result2 = to_markdown_table(df, num_rows=5, sample=True, seed=123)

        # Assert
        with check:
            assert result1 == result2

    def test_sample_with_different_seeds_differ(self) -> None:
        """Given same DataFrame with enough rows, When called with different seeds, Then outputs differ."""
        # Arrange
        df = pl.DataFrame({"value": list(range(100))})

        # Act
        result1 = to_markdown_table(df, num_rows=10, sample=True, seed=1)
        result2 = to_markdown_table(df, num_rows=10, sample=True, seed=999)

        # Assert
        with check:
            assert result1 != result2

    def test_sample_false_is_deterministic(self) -> None:
        """Given same DataFrame, When called twice with sample=False, Then outputs are identical."""
        # Arrange
        df = pl.DataFrame({"value": list(range(20))})

        # Act
        result1 = to_markdown_table(df, num_rows=5, sample=False)
        result2 = to_markdown_table(df, num_rows=5, sample=False)

        # Assert
        with check:
            assert result1 == result2

    def test_sample_num_rows_exceeds_dataframe_height(self) -> None:
        """Given DataFrame with 3 rows, When sample=True and num_rows=10, Then returns all 3 rows without error."""
        # Arrange
        df = pl.DataFrame({"value": [10, 20, 30]})

        # Act
        result = to_markdown_table(df, num_rows=10, sample=True)

        # Assert - should contain all 3 rows
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 3


class TestToMarkdownTableRendering:
    """Tests for to_markdown_table output format and data type rendering."""

    def test_default_returns_markdown_string(self) -> None:
        """Given DataFrame, When called with defaults, Then returns string with markdown table markers and data."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Act
        result = to_markdown_table(df)

        # Assert - should be a string with markdown table markers and actual data values
        with check:
            assert isinstance(result, str)
        with check:
            assert "|" in result
        with check:
            assert "---" in result
        with check:
            assert "1" in result
        with check:
            assert "6" in result

    def test_hides_shape_and_dtypes(self) -> None:
        """Given DataFrame, When called, Then output does not contain shape info or dtype annotations."""
        # Arrange
        df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        # Act
        result = to_markdown_table(df)

        # Assert - shape info should not appear
        with check:
            assert "shape:" not in result.lower()
        # Assert - dtype annotations should not appear as cell contents
        # Check exact cell values instead of bare substrings to avoid false matches
        # (e.g., bare "str" could match data like "frustrated")
        known_dtype_labels = {"i64", "str", "f64"}
        for line in result.splitlines():
            cells = {c.strip() for c in line.split("|") if c.strip()}
            with check:
                assert not cells & known_dtype_labels, f"Dtype annotation visible in: {line}"

    def test_single_column_dataframe(self) -> None:
        """Given DataFrame with one column, When called, Then returns valid markdown."""
        # Arrange
        df = pl.DataFrame({"only_col": [10, 20, 30]})

        # Act
        result = to_markdown_table(df)

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert "only_col" in result
        with check:
            assert "|" in result
        with check:
            assert "---" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 3

    def test_single_row_dataframe(self) -> None:
        """Given DataFrame with one row, When called, Then returns valid markdown with that row."""
        # Arrange
        df = pl.DataFrame({"a": [999], "b": [888], "c": [777]})

        # Act
        result = to_markdown_table(df)

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert "a" in result
        with check:
            assert "b" in result
        with check:
            assert "c" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 1
        # The single data row should contain the values
        with check:
            assert "999" in result
        with check:
            assert "888" in result
        with check:
            assert "777" in result

    def test_empty_dataframe(self) -> None:
        """Given empty DataFrame (0 rows), When called, Then returns valid markdown with headers but no data rows."""
        # Arrange
        df = pl.DataFrame({"col1": [], "col2": []}, schema={"col1": pl.Int64, "col2": pl.Int64})

        # Act
        result = to_markdown_table(df)

        # Assert - header present but no data rows
        with check:
            assert "col1" in result
        with check:
            assert "col2" in result
        with check:
            assert "|" in result
        with check:
            assert "---" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 0

    def test_empty_dataframe_with_sample(self) -> None:
        """Given empty DataFrame, When sample=True, Then returns valid markdown with headers but no data rows."""
        # Arrange
        df = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})

        # Act
        result = to_markdown_table(df, sample=True)

        # Assert - should not error, returns empty table with headers
        with check:
            assert "x" in result
        with check:
            assert "y" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 0

    def test_bool_column_type(self) -> None:
        """Given DataFrame with bool column, When called, Then bool values render correctly."""
        # Arrange
        df = pl.DataFrame({"flag": [True, False, True]})

        # Act
        result = to_markdown_table(df)

        # Assert
        with check:
            assert "flag" in result
        with check:
            assert "true" in result
        with check:
            assert "false" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 3

    def test_date_column_type(self) -> None:
        """Given DataFrame with date column, When called, Then date values render correctly."""
        # Arrange
        df = pl.DataFrame({"event_date": [date(2024, 1, 15), date(2024, 6, 30)]})

        # Act
        result = to_markdown_table(df)

        # Assert
        with check:
            assert "event_date" in result
        with check:
            assert "2024-01-15" in result
        with check:
            assert "2024-06-30" in result

    def test_datetime_column_type(self) -> None:
        """Given DataFrame with datetime column, When called, Then datetime values render correctly."""
        # Arrange
        df = pl.DataFrame({"ts": [datetime(2024, 1, 15, 10, 30), datetime(2024, 6, 30, 23, 59)]})

        # Act
        result = to_markdown_table(df)

        # Assert
        with check:
            assert "ts" in result
        with check:
            assert "2024-01-15" in result
        with check:
            assert "2024-06-30" in result

    def test_float_with_nan_and_none(self) -> None:
        """Given DataFrame with float NaN and None values, When called, Then renders without error."""
        # Arrange
        df = pl.DataFrame({"score": [1.5, float("nan"), None, 4.0]})

        # Act
        result = to_markdown_table(df)

        # Assert
        with check:
            assert "score" in result
        with check:
            assert "1.5" in result
        with check:
            assert "4.0" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 4

    def test_null_values_across_types(self) -> None:
        """Given DataFrame with null values in multiple typed columns, When called, Then renders without error."""
        # Arrange
        df = pl.DataFrame({
            "int_col": [1, None, 3],
            "str_col": ["a", None, "c"],
            "float_col": [1.1, 2.2, None],
        })

        # Act
        result = to_markdown_table(df)

        # Assert - non-null values present
        with check:
            assert "int_col" in result
        with check:
            assert "str_col" in result
        with check:
            assert "float_col" in result
        with check:
            assert "1" in result
        with check:
            assert "c" in result
        with check:
            assert "2.2" in result
        data_row_count = _count_markdown_table_data_rows(result)
        with check:
            assert data_row_count == 3

    def test_diverse_column_types_together(self) -> None:
        """Given DataFrame with bool, date, datetime, int, float, and str columns, When called, Then all render."""
        # Arrange
        df = pl.DataFrame({
            "id": [1, 2],
            "active": [True, False],
            "score": [99.5, 87.3],
            "label": ["alpha", "beta"],
            "created": [date(2024, 3, 1), date(2024, 4, 15)],
            "updated": [datetime(2024, 3, 1, 12, 0), datetime(2024, 4, 15, 8, 30)],
        })

        # Act
        result = to_markdown_table(df)

        # Assert - all columns present
        for col in ("id", "active", "score", "label", "created", "updated"):
            with check:
                assert col in result
        # Assert - sample data values present
        with check:
            assert "alpha" in result
        with check:
            assert "true" in result
        with check:
            assert "99.5" in result
        with check:
            assert "2024-03-01" in result


def _count_markdown_table_data_rows(markdown: str) -> int:
    """Count data rows in markdown output.

    Args:
        markdown (str): Markdown table string.

    Returns:
        int: Number of data rows (excluding header and separator).
    """
    lines = [line.strip() for line in markdown.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    # Subtract 2: header row (always first) + separator row (always second)
    # Using position instead of "---" content match avoids incorrectly filtering
    # data rows that contain literal "---" as cell content
    return max(0, len(table_lines) - 2)
