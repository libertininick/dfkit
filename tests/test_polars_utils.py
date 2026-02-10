"""Tests for polars_utils module: to_markdown_table function."""

from __future__ import annotations

import polars as pl
import pytest
from pytest_check import check

from dfkit.polars_utils import to_markdown_table


class TestToMarkdownTable:
    """Test suite for to_markdown_table function."""

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

        # Assert - columns in specified order
        lines = result.splitlines()
        header_line = lines[0]
        # "c" should appear before "a" in the header
        c_pos = header_line.index("c")
        a_pos = header_line.index("a")
        with check:
            assert c_pos < a_pos

    def test_invalid_columns_raises_value_error(self) -> None:
        """Given DataFrame with columns [a, b], When columns=["a", "nonexistent"], Then raises ValueError."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act/Assert
        with pytest.raises(ValueError, match="Columns not found"):
            to_markdown_table(df, columns=["a", "nonexistent"])

    def test_invalid_columns_error_message_contains_sorted_extra_columns(self) -> None:
        """Given DataFrame, When invalid columns passed, Then error message contains sorted list of extra columns."""
        # Arrange
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act/Assert
        with pytest.raises(ValueError) as exc_info:
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

    def test_num_rows_limits_output(self) -> None:
        """Given DataFrame with 20 rows, When num_rows=5, Then output is truncated with ellipsis."""
        # Arrange
        df = pl.DataFrame({"value": list(range(20))})

        # Act
        result = to_markdown_table(df, num_rows=5)

        # Assert - output should be truncated (less than full 20 rows) and include ellipsis
        data_row_count = _count_data_rows(result)
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
        data_row_count = _count_data_rows(result)
        with check:
            assert data_row_count == 3

    def test_sample_true_returns_random_rows(self) -> None:
        """Given DataFrame with 100 rows, When sample=True and num_rows=5, Then output contains 5 data rows."""
        # Arrange
        df = pl.DataFrame({"idx": list(range(100))})

        # Act
        result = to_markdown_table(df, num_rows=5, sample=True)

        # Assert
        data_row_count = _count_data_rows(result)
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
        data_row_count = _count_data_rows(result)
        with check:
            assert data_row_count == 3

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
        data_row_count = _count_data_rows(result)
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
        data_row_count = _count_data_rows(result)
        with check:
            assert data_row_count == 0

    def test_hides_shape_and_dtypes(self) -> None:
        """Given DataFrame, When called, Then output does not contain shape info or dtype annotations."""
        # Arrange
        df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        # Act
        result = to_markdown_table(df)

        # Assert - shape and dtype info should not appear
        # Shape info typically looks like "shape: (2, 2)"
        # Dtype info typically looks like "<i64>" or "i64"
        with check:
            assert "shape:" not in result.lower()
        with check:
            assert "i64" not in result
        with check:
            assert "str" not in result

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
        data_row_count = _count_data_rows(result)
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
        data_row_count = _count_data_rows(result)
        with check:
            assert data_row_count == 1
        # The single data row should contain the values
        with check:
            assert "999" in result
        with check:
            assert "888" in result
        with check:
            assert "777" in result


def _count_data_rows(markdown: str) -> int:
    """Count data rows in markdown output.

    Args:
        markdown (str): Markdown table string.

    Returns:
        int: Number of data rows (excluding header and separator).
    """
    lines = [line.strip() for line in markdown.strip().splitlines() if line.strip()]
    # Lines with | that are not the separator (contains ---)
    table_lines = [line for line in lines if line.startswith("|") and "---" not in line]
    # Subtract header (1)
    return max(0, len(table_lines) - 1)
