"""Utility functions for working with Polars DataFrames."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


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
    des = series.describe(percentiles=percentiles)
    return dict(zip(des["statistic"], des["value"], strict=True))


def to_markdown_table(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    num_rows: int = 10,
    *,
    sample: bool = False,
    seed: int | None = None,
) -> str:
    """Convert a Polars DataFrame to a markdown table string.

    Use this to display DataFrames in a human-readable markdown format,
    optionally sampling rows or selecting specific columns.

    Args:
        df (pl.DataFrame): The DataFrame to convert.
        columns (list[str] | None): Optional list of column names to include.
            If None, all columns are included.
        num_rows (int): Maximum number of rows to display. Defaults to 10.
        sample (bool): If True, randomly sample rows instead of taking the first rows.
            Defaults to False.
        seed (int | None): Random seed for sampling when sample=True. Defaults to None.

    Returns:
        str: Markdown-formatted table string.

    Raises:
        ValueError: If any columns in the columns list do not exist in the DataFrame.

    Note:
        This function temporarily modifies global ``pl.Config`` state to render
        the table.  It is **not thread-safe**: concurrent calls from different
        threads may observe each other's configuration.  Use only from a single
        thread (or protect calls with a lock).

    Examples:
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> print(to_markdown_table(df, num_rows=3))
        | a | b |
        |---|---|
        | 1 | 4 |
        | 2 | 5 |
        | 3 | 6 |
    """
    if columns is not None:
        extra_columns = set(columns) - set(df.columns)
        if extra_columns:
            raise ValueError(f"Columns not found in DataFrame: {extra_columns}")

    if sample:
        df = df.sample(n=min(num_rows, df.height), seed=seed)

    if columns is not None:
        df = df.select(columns)

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_column_names=False,
        tbl_hide_dataframe_shape=True,
        tbl_rows=num_rows,
        tbl_cols=df.width if columns is None else len(columns),
    ):
        return str(df)
