"""Utility functions for working with Polars DataFrames."""

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
