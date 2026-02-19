"""Pydantic result models and preprocessing pipeline for the decision tree module.

Provides structured output types for decision tree analysis, designed for
LLM readability.  These models capture the rules, metrics, and feature
importance produced by the build_decision_tree tool.

The preprocessing pipeline handles column type classification, feature
filtering, task type detection, and Polars-to-numpy encoding required
before fitting a sklearn decision tree.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

type PredicateOp = Literal[">", ">=", "!=", "==", "<", "<=", "in", "not in"]

type ColumnType = Literal["numeric", "boolean", "categorical", "datetime", "duration", "excluded"]

type DecisionTreeTask = Literal["classification", "regression"]

# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class Predicate(BaseModel):
    """A single boolean condition on one feature variable.

    Represents a comparison such as `tenure_months > 6` or
    `plan_type in {basic, standard}`. Predicates are the building blocks
    of decision-tree rules; each rule contains an ordered list of predicates
    describing the path from the tree root to a leaf node.

    Attributes:
        variable (str): Feature or column name the condition applies to,
            e.g. `"tenure_months"` or `"plan_type"`.
        operator (PredicateOp):
            Comparison operator. Use `"in"` or `"not in"` for membership
            tests against a set of values.
        value (float | str | set[float] | set[str]): Threshold for scalar
            comparisons, or a set of candidate values for `"in"` /
            `"not in"` membership tests.

    Examples:
        >>> p = Predicate(variable="tenure_months", operator=">", value=6.0)
        >>> str(p)
        'tenure_months > 6.0'
        >>> p.eval(12.0)
        True
        >>> p2 = Predicate(variable="plan_type", operator="in", value={"basic", "standard"})
        >>> p2.eval("basic")
        True
    """

    variable: str = Field(
        description="Feature or column name the condition applies to, e.g. 'tenure_months'.",
    )
    operator: PredicateOp = Field(
        description=(
            "Comparison operator. Use 'in' or 'not in' for membership tests "
            "against a set of values; use the scalar operators for threshold comparisons."
        ),
    )
    value: float | str | set[float] | set[str] = Field(
        description=(
            "Threshold for scalar comparisons (float or str), or a set of candidate "
            "values for 'in' / 'not in' membership tests."
        ),
    )

    @model_validator(mode="after")
    def _validate_operator_value_compatibility(self) -> Predicate:
        """Validate that the operator and value type are compatible.

        Returns:
            Predicate: The validated model instance.

        Raises:
            ValueError: If a membership operator is used with a non-set value,
                or if a scalar operator is used with a set value.
        """
        try:
            _validate_operator_threshold_types(self.operator, self.value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def __str__(self) -> str:
        """Return a human-readable representation of this predicate.

        Returns:
            str: The predicate as `"<variable> <operator> <value>"`,
                e.g. `"plan_type in {basic, standard}"` or `"age >= 30"`.
        """
        if self.operator in {"in", "not in"}:
            # sorted() expects an Iterable[SupportsLessThan], but the union
            # type includes float|str which ty cannot prove is sortable as a
            # homogeneous collection.  At runtime self.value is always a
            # set[float] or set[str] here (enforced by the model validator),
            # so sorting is safe.
            sorted_values = ", ".join(str(v) for v in sorted(self.value))  # type: ignore[arg-type]
            return f"{self.variable} {self.operator} {{{sorted_values}}}"
        return f"{self.variable} {self.operator} {self.value}"

    def eval(self, x: float | str) -> bool:
        """Evaluate this predicate against a feature value.

        Args:
            x (float | str): The feature value to test.

        Returns:
            bool: `True` if the predicate holds for `x`, `False` otherwise.
        """
        return _apply_operator(self.operator, x, self.value)


class ClassificationRule(BaseModel):
    """A decision rule extracted from a classification tree leaf node.

    Represents the path from the root of the tree to one leaf, expressed as
    a list of :class:`Predicate` predicates, together with the predicted class
    label and the confidence at that leaf.

    Attributes:
        task_type (Literal["classification"]): Discriminator field; always
            `"classification"`.
        predicates (list[Predicate]): Predicates along the path from root to
            this leaf, e.g.
            `[Predicate(variable="tenure_months", operator=">", value=6),
            Predicate(variable="support_tickets", operator="<=", value=3)]`.
            An empty list means the tree has depth zero (a single leaf).
        prediction (str | float): Predicted class label for samples reaching
            this leaf.
        samples (int): Number of training samples that reached this leaf.
        confidence (float): Fraction of samples at this leaf belonging to the
            majority (predicted) class. Ranges from 0.0 to 1.0.

    Examples:
        >>> rule = ClassificationRule(
        ...     task_type="classification",
        ...     predicates=[
        ...         Predicate(variable="tenure_months", operator=">", value=6),
        ...         Predicate(variable="support_tickets", operator="<=", value=3),
        ...     ],
        ...     prediction="retained",
        ...     samples=342,
        ...     confidence=0.91,
        ... )
    """

    task_type: Literal["classification"] = Field(
        description='Discriminator field. Always "classification".',
    )
    predicates: list[Predicate] = Field(
        description=(
            "Predicates along the path from root to this leaf. "
            "Each Predicate captures a variable, operator, and threshold or value set. "
            "Empty list indicates a single-leaf tree with no splits."
        ),
    )
    prediction: str | float = Field(
        description="Predicted class label for samples reaching this leaf.",
    )
    samples: int = Field(
        ge=1,
        description="Number of training samples that reached this leaf node.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of samples at this leaf belonging to the majority (predicted) class. Ranges from 0.0 to 1.0."
        ),
    )


class RegressionRule(BaseModel):
    """A decision rule extracted from a regression tree leaf node.

    Represents the path from the root of the tree to one leaf, expressed as
    a list of :class:`Predicate` predicates, together with the predicted mean
    target value and the spread of target values at that leaf.

    Attributes:
        task_type (Literal["regression"]): Discriminator field; always
            `"regression"`.
        predicates (list[Predicate]): Predicates along the path from root to
            this leaf, e.g.
            `[Predicate(variable="age", operator=">", value=30),
            Predicate(variable="income", operator="<=", value=50000)]`.
            An empty list means the tree has depth zero (a single leaf).
        prediction (float): Mean target value for samples reaching this
            leaf.
        samples (int): Number of training samples that reached this leaf.
        std (float): Standard deviation of target values among samples at this
            leaf. Indicates prediction spread.

    Examples:
        >>> rule = RegressionRule(
        ...     task_type="regression",
        ...     predicates=[
        ...         Predicate(variable="age", operator=">", value=30),
        ...         Predicate(variable="income", operator="<=", value=50000),
        ...     ],
        ...     prediction=42500.0,
        ...     samples=187,
        ...     std=3200.5,
        ... )
    """

    task_type: Literal["regression"] = Field(
        description='Discriminator field. Always "regression".',
    )
    predicates: list[Predicate] = Field(
        description=(
            "Predicates along the path from root to this leaf. "
            "Each Predicate captures a variable, operator, and threshold or value set. "
            "Empty list indicates a single-leaf tree with no splits."
        ),
    )
    prediction: float = Field(
        description="Mean target value for samples reaching this leaf.",
    )
    samples: int = Field(
        ge=1,
        description="Number of training samples that reached this leaf node.",
    )
    std: float = Field(
        ge=0.0,
        description=("Standard deviation of target values among samples at this leaf. Indicates prediction spread."),
    )


# Union of ClassificationRule and RegressionRule, discriminated by the
# `task_type` field.  Use this alias when accepting a rule of either task
# type; Pydantic will select the correct model automatically.
type DecisionTreeRule = Annotated[
    ClassificationRule | RegressionRule,
    Field(discriminator="task_type"),
]


class DecisionTreeResult(BaseModel):
    """Structured output from the build_decision_tree tool.

    Captures the full result of fitting a decision tree, including the
    human-readable rules extracted from each leaf, feature importance scores,
    evaluation metrics, and tree structure metadata.

    Attributes:
        target (str): Target column name used as the prediction label.
        task_type (DecisionTreeTask): Either
            `"classification"` or `"regression"`.
        features_used (list[str]): Feature column names that were included when
            fitting the tree.
        features_excluded (list[str]): Feature column names that were excluded,
            each annotated with the reason for exclusion.
        rules (list[DecisionTreeRule]): One rule per leaf node, describing the
            predicates and prediction for that path through the tree. Each
            rule's `task_type` field identifies its concrete type.
        feature_importance (dict[str, float]): Mapping of feature name to
            importance score, sorted in descending order of importance. Scores
            must sum to 1.0 across all features used.
        metrics (dict[str, float]): Evaluation metrics for the fitted tree,
            e.g. `{"accuracy": 0.83}` for classification or
            `{"r_squared": 0.76, "rmse": 68200.0}` for regression.
        sample_count (int): Total number of samples used to fit the tree.
        depth (int): Actual depth of the fitted tree.
        leaf_count (int): Number of leaf nodes in the fitted tree.

    Examples:
        >>> result = DecisionTreeResult(
        ...     target="churn",
        ...     task_type="classification",
        ...     features_used=["tenure_months", "support_tickets"],
        ...     features_excluded=["customer_id (unique identifier)"],
        ...     rules=[
        ...         ClassificationRule(
        ...             task_type="classification",
        ...             predicates=[
        ...                 Predicate(variable="tenure_months", operator="<=", value=6),
        ...             ],
        ...             prediction="churned",
        ...             samples=210,
        ...             confidence=0.87,
        ...         ),
        ...     ],
        ...     feature_importance={"tenure_months": 0.7, "support_tickets": 0.3},
        ...     metrics={"accuracy": 0.89},
        ...     sample_count=500,
        ...     depth=3,
        ...     leaf_count=1,
        ... )
    """

    target: str = Field(
        description="Target column name used as the prediction label.",
    )
    task_type: DecisionTreeTask = Field(
        description='Task type: either "classification" or "regression".',
    )
    features_used: list[str] = Field(
        description="Feature column names that were included when fitting the tree.",
    )
    features_excluded: list[str] = Field(
        description=(
            "Feature column names that were excluded from fitting, each entry annotated with the reason for exclusion."
        ),
    )
    rules: list[DecisionTreeRule] = Field(
        description=(
            "One rule per leaf node, describing the path predicates and prediction "
            "for every reachable outcome of the tree. Each rule's task_type field "
            "identifies whether it is a ClassificationRule or RegressionRule."
        ),
    )
    feature_importance: dict[str, float] = Field(
        description=(
            "Mapping of feature name to importance score, sorted in descending "
            "order. Scores must sum to 1.0 across all features used."
        ),
    )
    metrics: dict[str, float] = Field(
        description=(
            'Evaluation metrics for the fitted tree, e.g. {"accuracy": 0.83} for '
            'classification or {"r_squared": 0.76, "rmse": 68200.0} for regression.'
        ),
    )
    sample_count: int = Field(
        ge=1,
        description="Total number of samples used to fit the tree.",
    )
    depth: int = Field(
        ge=0,
        description="Actual depth of the fitted tree.",
    )
    leaf_count: int = Field(
        ge=1,
        description="Number of leaf nodes in the fitted tree.",
    )

    @field_validator("feature_importance", mode="after")
    @classmethod
    def _validate_feature_importance_sums_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        """Validate that feature importance scores sum to 1.0.

        Args:
            value (dict[str, float]): The feature importance mapping to validate.

        Returns:
            dict[str, float]: The validated mapping, unchanged.

        Raises:
            ValueError: If the scores do not sum to 1.0 within a tolerance of 1e-6.
        """
        total = sum(value.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"feature_importance scores must sum to 1.0, got {total:.8f}")
        return value

    @model_validator(mode="after")
    def _validate_feature_importance_keys_in_features_used(self) -> DecisionTreeResult:
        """Validate that feature_importance keys exactly match features_used.

        Returns:
            DecisionTreeResult: The validated model instance.

        Raises:
            ValueError: If any key in `feature_importance` is missing from
                `features_used`, or any feature in `features_used` is missing
                from `feature_importance`.
        """
        importance_keys = set(self.feature_importance)
        features_used_set = set(self.features_used)
        extra_in_importance = importance_keys - features_used_set
        missing_from_importance = features_used_set - importance_keys
        errors: list[str] = []
        if extra_in_importance:
            errors.append(f"feature_importance contains keys not in features_used: {sorted(extra_in_importance)}")
        if missing_from_importance:
            errors.append(f"features_used contains keys not in feature_importance: {sorted(missing_from_importance)}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_rules_count_matches_leaf_count(self) -> DecisionTreeResult:
        """Validate that the number of rules equals the number of leaf nodes.

        Returns:
            DecisionTreeResult: The validated model instance.

        Raises:
            ValueError: If `len(rules)` does not equal `leaf_count`.
        """
        if len(self.rules) != self.leaf_count:
            raise ValueError(f"rules length ({len(self.rules)}) must equal leaf_count ({self.leaf_count})")
        return self

    @model_validator(mode="after")
    def _validate_rule_types_match_task_type(self) -> DecisionTreeResult:
        """Validate that every rule's task_type matches the result's task_type.

        Returns:
            DecisionTreeResult: The validated model instance.

        Raises:
            ValueError: If any rule has a `task_type` that does not match
                `self.task_type`.
        """
        if any(rule.task_type != self.task_type for rule in self.rules):
            raise ValueError(f"rules have task_type inconsistent with result task_type '{self.task_type}'")
        return self


# ---------------------------------------------------------------------------
# Private helpers -- Predicate operator evaluation
# ---------------------------------------------------------------------------

_SCALAR_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _apply_operator(
    op: PredicateOp,
    x: float | str,
    threshold: float | str | set[float] | set[str],
) -> bool:
    """Apply a comparison operator between a feature value and a threshold.

    Args:
        op (PredicateOp): The comparison operator to apply.
        x (float | str): The feature value to compare.
        threshold (float | str | set[float] | set[str]): The threshold or
            set of candidate values to compare against.

    Returns:
        bool: Result of applying `op` between `x` and `threshold`.

    Raises:
        ValueError: If `op` is not a recognized `PredicateOp` value.
    """
    _validate_operator_threshold_types(op, threshold)
    if op in _SCALAR_OPS:
        return _SCALAR_OPS[op](x, threshold)
    if op == "in" and isinstance(threshold, set):
        return x in threshold
    if op == "not in" and isinstance(threshold, set):
        return x not in threshold
    raise ValueError(f"Unexpected operator: {op!r}")


def _validate_operator_threshold_types(
    op: PredicateOp,
    threshold: float | str | set[float] | set[str],
) -> None:
    """Raise TypeError when operator and threshold types are incompatible.

    Args:
        op (PredicateOp): The comparison operator to validate.
        threshold (float | str | set[float] | set[str]): The threshold value
            to validate against the operator.

    Raises:
        TypeError: If a scalar operator is paired with a set threshold, or a
            membership operator is paired with a non-set threshold.
    """
    if op in _SCALAR_OPS and isinstance(threshold, set):
        raise TypeError(f"Scalar operator '{op}' cannot compare against a set")
    if op in {"in", "not in"} and not isinstance(threshold, set):
        raise TypeError(f"Membership operator '{op}' requires a set threshold")


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
    pl.Utf8: "categorical",
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
            :class:`_FeatureEncoder` objects (one per column).
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
    classification a :class:`sklearn.preprocessing.LabelEncoder` is used so
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
    category_mapping = dict(enumerate(str(label) for label in label_encoder.classes_))
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
    category_mapping = dict(enumerate(str(cat) for cat in ordinal_encoder.categories_[0]))
    return encoded_column, category_mapping
