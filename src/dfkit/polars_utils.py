"""Utility functions for working with Polars DataFrames."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from dfkit.exceptions import ColumnsNotFoundError, DuplicateColumnsError


def get_series_description(
    series: pl.Series, percentiles: Sequence[float] = (0.25, 0.5, 0.75)
) -> dict[str, float | str]:
    """Get descriptive statistics for a Polars Series as a dictionary.

    Args:
        series (pl.Series): The Polars Series to describe.
        percentiles (Sequence[float], optional): Percentiles to compute. Defaults to (0.25, 0.5, 0.75).

    Returns:
        dict[str, float | str]: A dictionary containing descriptive statistics.
    """
    series_description = series.describe(percentiles=percentiles)
    return dict(zip(series_description["statistic"], series_description["value"], strict=True))


def to_markdown_table(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    num_rows: int = 10,
    *,
    sample: bool = False,
    seed: int | None = None,
) -> str:
    """Convert a Polars DataFrame to a markdown table string.

    Use this to display DataFrames in a human-readable markdown format,
    optionally sampling rows or selecting specific columns. This function
    temporarily modifies global ``pl.Config`` state to render the table.
    It is not thread-safe: concurrent calls from different threads may
    observe each other's configuration. Use only from a single thread
    (or protect calls with a lock).

    Args:
        df (pl.DataFrame): The DataFrame to convert.
        columns (Sequence[str] | None): Optional list of column names to include.
            If None, all columns are included.
        num_rows (int): Maximum number of rows to display. Defaults to 10.
        sample (bool): If True, randomly sample rows instead of taking the first rows.
            Defaults to False.
        seed (int | None): Random seed for sampling when sample=True. Defaults to None.

    Returns:
        str: Markdown-formatted table string.

    Examples:
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> print(to_markdown_table(df, num_rows=3))
        | a | b |
        |---|---|
        | 1 | 4 |
        | 2 | 5 |
        | 3 | 6 |
    """
    _validate_markdown_table_inputs(df, columns, num_rows, sample=sample, seed=seed)

    if columns is not None:
        df = df.select(columns)

    if sample:
        df = df.sample(n=min(num_rows, df.height), seed=seed)

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_column_names=False,
        tbl_hide_dataframe_shape=True,
        tbl_rows=num_rows,
        tbl_cols=df.width,
    ):
        return str(df)


def _validate_markdown_table_inputs(
    df: pl.DataFrame,
    columns: Sequence[str] | None,
    num_rows: int,
    *,
    sample: bool,
    seed: int | None,
) -> None:
    """Validate inputs for to_markdown_table.

    Args:
        df (pl.DataFrame): The DataFrame to validate against.
        columns (Sequence[str] | None): Column names to validate.
        num_rows (int): Row count to validate.
        sample (bool): Whether sampling is enabled.
        seed (int | None): Random seed to validate.

    Raises:
        ValueError: If num_rows is less than 1, seed is provided without
            sample=True, columns contain duplicates, or any columns do not
            exist in the DataFrame.
    """
    if num_rows < 1:
        raise ValueError(f"num_rows must be at least 1, got {num_rows}")
    if seed is not None and not sample:
        raise ValueError("seed is only used when sample=True")
    if columns is not None:
        _validate_columns(columns, df.columns)


def _validate_columns(columns: Sequence[str], df_columns: Sequence[str]) -> None:
    """Validate that columns exist in the DataFrame and contain no duplicates.

    Args:
        columns (Sequence[str]): Column names to validate.
        df_columns (Sequence[str]): Column names present in the DataFrame.

    Raises:
        ValueError: If columns list is empty.
        DuplicateColumnsError: If columns contain duplicates.
        ColumnsNotFoundError: If any columns do not exist in the DataFrame.
    """
    if len(columns) == 0:
        msg = "columns list must not be empty; pass None to include all columns"
        raise ValueError(msg)
    if len(columns) != len(set(columns)):
        raise DuplicateColumnsError(columns=list(columns))
    extra_columns = set(columns) - set(df_columns)
    if extra_columns:
        raise ColumnsNotFoundError(
            missing_columns=sorted(extra_columns),
            available_columns=list(df_columns),
        )
