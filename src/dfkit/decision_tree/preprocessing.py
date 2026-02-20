"""Preprocessing pipeline: column classification, feature filtering, task detection, and encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from dfkit.decision_tree.models import ColumnType, DecisionTreeTask

# ---------------------------------------------------------------------------
# Private helpers -- Column type classification
# ---------------------------------------------------------------------------

_DTYPE_TO_COLUMN_TYPE: dict[type[pl.DataType] | pl.DataType, ColumnType] = {
    pl.Int8: "numeric",
    pl.Int16: "numeric",
    pl.Int32: "numeric",
    pl.Int64: "numeric",
    pl.UInt8: "numeric",
    pl.UInt16: "numeric",
    pl.UInt32: "numeric",
    pl.UInt64: "numeric",
    pl.Float32: "numeric",
    pl.Float64: "numeric",
    pl.Boolean: "boolean",
    pl.String: "categorical",
    pl.Categorical: "categorical",
    pl.Enum: "categorical",
    pl.Date: "datetime",
    pl.Datetime: "datetime",
    pl.Duration: "duration",
}


def _classify_column(dtype: pl.DataType) -> ColumnType:
    """Classify a Polars column dtype into a broad feature category.

    The lookup map uses bare class references as keys (e.g. `pl.Datetime`),
    which works for singleton dtypes but not for parameterized instances such
    as `Datetime("us")` whose hash differs from the bare class.  An
    `isinstance` fallback handles those cases.

    Args:
        dtype (pl.DataType): The Polars data type of the column to classify.

    Returns:
        ColumnType: One of `"numeric"`, `"boolean"`, `"categorical"`,
            `"datetime"`, `"duration"`, or `"excluded"`.
    """
    result = _DTYPE_TO_COLUMN_TYPE.get(dtype)
    if result is not None:
        return result
    if isinstance(dtype, (pl.Datetime, pl.Date)):
        return "datetime"
    if isinstance(dtype, pl.Duration):
        return "duration"
    if isinstance(dtype, pl.Enum):
        return "categorical"
    return "excluded"


# ---------------------------------------------------------------------------
# Private helpers -- Feature filtering
# ---------------------------------------------------------------------------

_HIGH_CARDINALITY_RATIO: float = 0.9
_THRESHOLD_DECIMAL_PLACES: int = 4  # Decimal places for rounding numeric split thresholds in predicates.


class _ExcludedFeature(NamedTuple):
    """A feature column that was excluded from preprocessing, with the reason.

    Attributes:
        name (str): The column name that was excluded.
        reason (str): Human-readable explanation for the exclusion.
    """

    name: str
    reason: str


def _filter_features(
    df: pl.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[_ExcludedFeature]]:
    """Partition feature columns into kept and excluded sets.

    Columns are excluded when they have an unsupported dtype, contain only
    null values, carry a single unique value (zero variance), or are
    high-cardinality categorical columns that are likely unique identifiers.

    Args:
        df (pl.DataFrame): The input DataFrame to inspect.
        feature_columns (list[str]): Column names to evaluate.

    Returns:
        tuple[list[str], list[_ExcludedFeature]]: A 2-tuple of
            `(kept_names, excluded_features)` where *kept_names* is the
            ordered list of columns that passed all filters and
            *excluded_features* holds the dropped columns together with their
            exclusion reasons.
    """
    row_count = len(df)
    kept: list[str] = []
    excluded: list[_ExcludedFeature] = []

    for col_name in feature_columns:
        exclusion_reason = _get_exclusion_reason(df[col_name], row_count)
        if exclusion_reason is not None:
            excluded.append(_ExcludedFeature(name=col_name, reason=exclusion_reason))
        else:
            kept.append(col_name)

    return kept, excluded


def _get_exclusion_reason(series: pl.Series, row_count: int) -> str | None:
    """Return the exclusion reason for a column, or `None` if it should be kept.

    Args:
        series (pl.Series): The column to inspect.
        row_count (int): Total number of rows in the containing DataFrame.

    Returns:
        str | None: A human-readable reason string if the column should be
            excluded, or `None` if the column passes all filters.
    """
    column_type = _classify_column(series.dtype)

    if column_type == "excluded":
        return "unsupported dtype"
    if series.is_null().all():
        return "all values are null"

    unique_count = series.n_unique()
    if unique_count <= 1:
        return "single unique value"
    if column_type == "categorical" and row_count > 0 and unique_count / row_count > _HIGH_CARDINALITY_RATIO:
        return "high cardinality: likely unique identifier"

    return None


# ---------------------------------------------------------------------------
# Private helpers -- Category mapping
# ---------------------------------------------------------------------------


def _make_category_mapping(labels: Any) -> dict[int, str]:
    """Build a `{code: label}` mapping from an ordered sequence of category labels.

    Args:
        labels (Any): An iterable of category labels (e.g. numpy array or list)
            ordered by their ordinal code, starting at 0.

    Returns:
        dict[int, str]: Mapping of integer ordinal code to string label.
    """
    return dict(enumerate(str(label) for label in labels))


# ---------------------------------------------------------------------------
# Private helpers -- Task type detection
# ---------------------------------------------------------------------------

_INTEGER_DTYPES: frozenset[type[pl.DataType]] = frozenset({
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
})
_MAX_CLASSIFICATION_UNIQUE_COUNT: int = 20
_MAX_CLASSIFICATION_UNIQUE_RATIO: float = 0.05


def _detect_task_type(
    series: pl.Series,
    task_type_override: str | None,
) -> DecisionTreeTask:
    """Infer whether the target column represents a classification or regression task.

    When *task_type_override* is `"classification"` or `"regression"`, that
    value is returned directly.  Otherwise the dtype and cardinality of *series*
    are used to make an automatic determination.

    Args:
        series (pl.Series): The target column to inspect.
        task_type_override (str | None): Explicit override; `"auto"` or
            `None` triggers automatic detection.

    Returns:
        DecisionTreeTask: The detected task type.
    """
    if task_type_override in {"classification", "regression"}:
        return task_type_override  # type: ignore[return-value]

    column_type = _DTYPE_TO_COLUMN_TYPE.get(series.dtype)
    if column_type in {"boolean", "categorical"}:
        return "classification"

    if series.dtype in _INTEGER_DTYPES and _integer_series_is_classification(series):
        return "classification"

    return "regression"


def _integer_series_is_classification(series: pl.Series) -> bool:
    """Return `True` if an integer series has low enough cardinality for classification.

    Args:
        series (pl.Series): An integer-dtype Polars Series.

    Returns:
        bool: `True` when the series has at most
            `_MAX_CLASSIFICATION_UNIQUE_COUNT` unique values or when the
            ratio of unique values to non-null values is below
            `_MAX_CLASSIFICATION_UNIQUE_RATIO`.
    """
    non_null_series = series.drop_nulls()
    unique_count = non_null_series.n_unique()
    non_null_count = non_null_series.len()

    if unique_count <= _MAX_CLASSIFICATION_UNIQUE_COUNT:
        return True
    return non_null_count > 0 and unique_count / non_null_count < _MAX_CLASSIFICATION_UNIQUE_RATIO


# ---------------------------------------------------------------------------
# Private helpers -- Feature and target encoding
# ---------------------------------------------------------------------------


@dataclass
class _FeatureEncoder:
    """Metadata describing how one feature column was encoded to a numpy array.

    Attributes:
        column_name (str): The source column name in the original DataFrame.
        column_type (ColumnType): The broad type category assigned to the column.
        category_mapping (dict[int, str] | None): Maps integer codes back to
            the original category labels.  `None` for non-categorical columns.
    """

    column_name: str
    column_type: ColumnType
    category_mapping: dict[int, str] | None = field(default=None)


def _encode_features(
    df: pl.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[_FeatureEncoder]]:
    """Encode DataFrame feature columns into a 2-D float64 numpy array.

    Each column is converted according to its :func:`_classify_column` type:

    - `"numeric"` → cast to `float64` (Polars nulls become `NaN`).
    - `"boolean"` → cast via `Int8` to `float64`.
    - `"categorical"` → ordinal-encoded with unknown/missing mapped to `NaN`.
    - `"datetime"` → epoch microseconds as `float64`.
    - `"duration"` → total microseconds as `float64`.

    Args:
        df (pl.DataFrame): The source DataFrame.
        feature_columns (list[str]): Ordered list of column names to encode.

    Returns:
        tuple[np.ndarray, list[_FeatureEncoder]]: A 2-tuple of
            `(feature_matrix, encoders)` where *feature_matrix* has shape
            `(n_rows, n_features)` and *encoders* is a parallel list of
            `_FeatureEncoder` objects (one per column).
    """
    column_arrays: list[np.ndarray] = []
    encoders: list[_FeatureEncoder] = []

    for col_name in feature_columns:
        series = df[col_name]
        column_type = _classify_column(series.dtype)
        column_array, category_mapping = _encode_single_feature(series, column_type)
        column_arrays.append(column_array)
        encoders.append(
            _FeatureEncoder(column_name=col_name, column_type=column_type, category_mapping=category_mapping)
        )

    # Return a zero-width matrix when no features survived filtering.
    feature_matrix = np.column_stack(column_arrays) if column_arrays else np.empty((len(df), 0), dtype=np.float64)
    return feature_matrix, encoders


def _encode_target(
    series: pl.Series,
    task_type: DecisionTreeTask,
) -> tuple[np.ndarray, dict[int, str] | None]:
    """Encode a target column into a 1-D numpy array.

    For regression the raw numeric values are returned as `float64`.  For
    classification a `sklearn.preprocessing.LabelEncoder` is used so
    that string or mixed-type targets are mapped to integer codes, and a
    reverse mapping is returned.

    Args:
        series (pl.Series): The target column to encode.
        task_type (DecisionTreeTask): Determines the
            encoding strategy.

    Returns:
        tuple[np.ndarray, dict[int, str] | None]: A 2-tuple of
            `(encoded_array, category_mapping)` where *category_mapping* maps
            integer codes back to original label strings, or `None` for
            regression tasks.

    Raises:
        ValueError: If `series` contains any null values.
    """
    if series.null_count() > 0:
        raise ValueError(f"Target column '{series.name}' contains null values. Remove or impute nulls before training.")

    if task_type == "regression":
        return series.to_numpy(allow_copy=True).astype(np.float64), None

    label_encoder = LabelEncoder()
    raw_array = series.to_numpy(allow_copy=True)
    # fit_transform returns int64; cast to float64 for sklearn estimator compatibility.
    encoded_array = label_encoder.fit_transform(raw_array).astype(np.float64)
    category_mapping = _make_category_mapping(label_encoder.classes_)
    return encoded_array, category_mapping


def _encode_single_feature(
    series: pl.Series,
    column_type: ColumnType,
) -> tuple[np.ndarray, dict[int, str] | None]:
    """Convert one Polars Series to a 1-D float64 numpy array.

    Args:
        series (pl.Series): The column to encode.
        column_type (ColumnType): The pre-classified type of the column.

    Returns:
        tuple[np.ndarray, dict[int, str] | None]: A 1-D float64 array and an
            optional `{code: label}` mapping (non-`None` only for
            `"categorical"` columns).

    Raises:
        ValueError: If `column_type` is not a supported encodable type
            (e.g. `"excluded"` or any unexpected value).
    """
    if column_type == "numeric":
        return series.to_numpy(allow_copy=True).astype(np.float64), None

    if column_type == "boolean":
        return series.cast(pl.Int8).to_numpy(allow_copy=True).astype(np.float64), None

    if column_type == "categorical":
        return _encode_categorical_series(series)

    if column_type in {"datetime", "duration"}:
        return _encode_temporal_series(series, column_type)

    raise ValueError(f"Cannot encode column_type={column_type!r}")


def _encode_temporal_series(
    series: pl.Series,
    column_type: ColumnType,
) -> tuple[np.ndarray, None]:
    """Convert a datetime or duration Polars Series to a float64 array.

    Datetime columns are cast to epoch microseconds; duration columns are cast
    to total microseconds.

    Args:
        series (pl.Series): A Polars Datetime or Duration Series.
        column_type (ColumnType): Determines the encoding path; must be
            `"datetime"` or `"duration"`.

    Returns:
        tuple[np.ndarray, None]: A 1-D float64 array and `None` (no category
            mapping for temporal columns).
    """
    if column_type == "datetime":
        return series.cast(pl.Datetime("us")).dt.epoch("us").to_numpy(allow_copy=True).astype(np.float64), None
    return series.cast(pl.Duration("us")).dt.total_microseconds().to_numpy(allow_copy=True).astype(np.float64), None


def _encode_categorical_series(series: pl.Series) -> tuple[np.ndarray, dict[int, str]]:
    """Ordinal-encode a categorical Polars Series with NaN for unknowns and nulls.

    Args:
        series (pl.Series): A string, Categorical, or Enum Polars Series.

    Returns:
        tuple[np.ndarray, dict[int, str]]: A 1-D float64 array of ordinal
            codes (unknowns and missing values encoded as `NaN`) and a
            `{code: label}` mapping for the fitted categories.
    """
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    raw_column = series.to_numpy(allow_copy=True).reshape(-1, 1)
    encoded_column = ordinal_encoder.fit_transform(raw_column).astype(np.float64).ravel()
    category_mapping = _make_category_mapping(ordinal_encoder.categories_[0])
    return encoded_column, category_mapping
