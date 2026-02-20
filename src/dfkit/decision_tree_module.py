"""Pydantic result models and preprocessing pipeline for the decision tree module.

Provides structured output types for decision tree analysis, designed for
LLM readability. These models capture the rules, metrics, and feature
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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

type PredicateOp = Literal[">", ">=", "!=", "==", "<", "<=", "in", "not in"]

type ColumnType = Literal["numeric", "boolean", "categorical", "datetime", "duration", "excluded"]

type DecisionTreeTask = Literal["classification", "regression"]

type DecisionTree = DecisionTreeClassifier | DecisionTreeRegressor

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
            # At runtime self.value is always set[float] or set[str] (enforced by validator).
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
    a list of `Predicate` predicates, together with the predicted class
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
    a list of `Predicate` predicates, together with the predicted mean
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


# Use this alias when accepting a rule of either task type; Pydantic will select the correct model automatically.
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
        rules (list[ClassificationRule] | list[RegressionRule]): One rule per
            leaf node, describing the predicates and prediction for that path
            through the tree. Each rule's `task_type` field identifies its
            concrete type.
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
    rules: list[ClassificationRule] | list[RegressionRule] = Field(
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
# Private helpers -- Tree fitting
# ---------------------------------------------------------------------------


def _fit_tree(
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    task_type: DecisionTreeTask,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int | None = None,
) -> DecisionTree:
    """Fit a decision tree to the given feature matrix and target vector.

    Args:
        feature_matrix (np.ndarray): 2-D feature matrix with shape `(n_samples, n_features)`.
        target_array (np.ndarray): 1-D target vector with shape `(n_samples,)`.
        task_type (DecisionTreeTask): Whether to fit a classifier or regressor.
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        random_state (int | None): Random seed for reproducibility.

    Returns:
        DecisionTree: The fitted tree estimator.
    """
    tree_cls = DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor
    tree = tree_cls(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(feature_matrix, target_array)
    return tree


# ---------------------------------------------------------------------------
# Private helpers -- Rule extraction
# ---------------------------------------------------------------------------


def _extract_rules(
    tree: DecisionTree,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    feature_encoders: list[_FeatureEncoder],
    target_mapping: dict[int, str] | None,
    task_type: DecisionTreeTask,
) -> list[ClassificationRule] | list[RegressionRule]:
    """Extract human-readable rules from a fitted decision tree.

    Walks the `tree.tree_` internal structure recursively, building one
    `ClassificationRule` or `RegressionRule` per leaf node.

    For categorical features, the sklearn threshold is decoded back to a set
    of category labels: categories with ordinal codes `<= threshold` go to the
    left branch (`"in"` predicate) and those with codes `> threshold` go to
    the right branch (also an `"in"` predicate over the complementary label
    set, for symmetry and clarity).

    For regression tasks, the per-leaf standard deviation is computed from the
    actual training samples assigned to each leaf via `tree.apply(feature_matrix)`.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_matrix (np.ndarray): 2-D feature matrix used to fit the tree.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        feature_encoders (list[_FeatureEncoder]): Parallel list of encoders
            describing how each feature column was encoded.
        target_mapping (dict[int, str] | None): Maps integer codes back to
            class labels for classification; `None` for regression.
        task_type (DecisionTreeTask): Whether the tree is a classifier or regressor.

    Returns:
        list[ClassificationRule] | list[RegressionRule]: One rule per leaf node.
    """
    leaf_assignments = tree.apply(feature_matrix) if task_type == "regression" else None
    rules: list[ClassificationRule] | list[RegressionRule] = []
    _walk_tree(
        tree=tree,
        target_array=target_array,
        feature_encoders=feature_encoders,
        target_mapping=target_mapping,
        task_type=task_type,
        leaf_assignments=leaf_assignments,
        node_id=0,
        path_predicates=[],
        rules=rules,
    )
    return rules


def _walk_tree(
    *,
    tree: DecisionTree,
    target_array: np.ndarray,
    feature_encoders: list[_FeatureEncoder],
    target_mapping: dict[int, str] | None,
    task_type: DecisionTreeTask,
    leaf_assignments: np.ndarray | None,
    node_id: int,
    path_predicates: list[Predicate],
    rules: list[ClassificationRule] | list[RegressionRule],
) -> None:
    """Recursively walk a decision tree node and accumulate leaf rules.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        feature_encoders (list[_FeatureEncoder]): Parallel list of encoders.
        target_mapping (dict[int, str] | None): Class label mapping for classification.
        task_type (DecisionTreeTask): Whether the tree is a classifier or regressor.
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)`, used to compute per-leaf std for regression.
        node_id (int): The current node index in `tree.tree_`.
        path_predicates (list[Predicate]): Accumulated predicates from the root
            to `node_id`.
        rules (list[ClassificationRule] | list[RegressionRule]): Accumulator
            list; leaf rules are appended in-place.
    """
    sklearn_tree = tree.tree_
    left_child = sklearn_tree.children_left[node_id]
    right_child = sklearn_tree.children_right[node_id]
    is_leaf = left_child == right_child  # Both are TREE_LEAF (-1) at leaves

    if is_leaf:
        rule = _build_leaf_rule(
            sklearn_tree=sklearn_tree,
            node_id=node_id,
            path_predicates=path_predicates,
            task_type=task_type,
            target_mapping=target_mapping,
            target_array=target_array,
            leaf_assignments=leaf_assignments,
        )
        rules.append(rule)  # type: ignore[arg-type]
        return

    feature_index = sklearn_tree.feature[node_id]
    threshold = sklearn_tree.threshold[node_id]
    encoder = feature_encoders[feature_index]
    left_predicate, right_predicate = _build_split_predicates(encoder, threshold)

    shared_kwargs: dict[str, Any] = {
        "tree": tree,
        "target_array": target_array,
        "feature_encoders": feature_encoders,
        "target_mapping": target_mapping,
        "task_type": task_type,
        "leaf_assignments": leaf_assignments,
        "rules": rules,
    }
    _walk_tree(**shared_kwargs, node_id=left_child, path_predicates=[*path_predicates, left_predicate])
    _walk_tree(**shared_kwargs, node_id=right_child, path_predicates=[*path_predicates, right_predicate])


def _build_split_predicates(encoder: _FeatureEncoder, threshold: float) -> tuple[Predicate, Predicate]:
    """Build the left and right branch predicates for a decision tree split.

    For categorical features, the split threshold is decoded to a set of
    category labels via the encoder's `category_mapping`.  Categories with
    codes `<= threshold` form the left branch (`"in"` predicate); categories
    with codes `> threshold` form the right branch (also an `"in"` predicate
    over the complementary set, for symmetry and clarity).

    For all other feature types, numeric threshold predicates (`"<="` for the
    left branch and `">"` for the right branch) are created.

    Args:
        encoder (_FeatureEncoder): Encoder metadata for the feature being split.
        threshold (float): Raw sklearn split threshold value.

    Returns:
        tuple[Predicate, Predicate]: A 2-tuple of
            `(left_predicate, right_predicate)`.
    """
    feature_name = encoder.column_name

    if encoder.category_mapping is not None:
        left_labels: set[str] = {label for code, label in encoder.category_mapping.items() if code <= threshold}
        right_labels: set[str] = {label for code, label in encoder.category_mapping.items() if code > threshold}
        left_predicate = Predicate(variable=feature_name, operator="in", value=left_labels)
        right_predicate = Predicate(variable=feature_name, operator="in", value=right_labels)
        return left_predicate, right_predicate

    rounded_threshold = round(float(threshold), _THRESHOLD_DECIMAL_PLACES)
    left_predicate = Predicate(variable=feature_name, operator="<=", value=rounded_threshold)
    right_predicate = Predicate(variable=feature_name, operator=">", value=rounded_threshold)
    return left_predicate, right_predicate


def _build_leaf_rule(
    *,
    sklearn_tree: Any,
    node_id: int,
    path_predicates: list[Predicate],
    task_type: DecisionTreeTask,
    target_mapping: dict[int, str] | None,
    target_array: np.ndarray,
    leaf_assignments: np.ndarray | None,
) -> DecisionTreeRule:
    """Construct a leaf rule from a decision tree node's stored statistics.

    Args:
        sklearn_tree (Any): The `tree.tree_` internal structure from a fitted
            sklearn decision tree estimator.
        node_id (int): Index of the leaf node in `sklearn_tree`.
        path_predicates (list[Predicate]): Predicates accumulated along the
            path from root to this leaf.
        task_type (DecisionTreeTask): Whether to produce a classification or
            regression rule.
        target_mapping (dict[int, str] | None): Class label mapping for
            classification; `None` for regression.
        target_array (np.ndarray): 1-D target vector used to fit the tree (needed to
            compute per-leaf std for regression).
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)` for regression std computation.

    Returns:
        DecisionTreeRule: The constructed leaf rule.
    """
    node_value = sklearn_tree.value[node_id]
    n_samples = int(sklearn_tree.n_node_samples[node_id])

    if task_type == "classification":
        return _build_classification_rule(
            node_value, n_samples, path_predicates=path_predicates, target_mapping=target_mapping
        )
    return _build_regression_rule(
        node_value,
        n_samples,
        node_id,
        path_predicates=path_predicates,
        target_array=target_array,
        leaf_assignments=leaf_assignments,
    )


def _build_classification_rule(
    node_value: np.ndarray,
    n_samples: int,
    *,
    path_predicates: list[Predicate],
    target_mapping: dict[int, str] | None,
) -> ClassificationRule:
    """Construct a classification rule from a leaf node's class counts.

    Args:
        node_value (np.ndarray): The `tree.tree_.value[node_id]` array with
            shape `(1, n_classes)` containing per-class sample counts.
        n_samples (int): Total number of training samples at this leaf.
        path_predicates (list[Predicate]): Predicates along the root-to-leaf path.
        target_mapping (dict[int, str] | None): Maps integer class codes to
            original label strings.

    Returns:
        ClassificationRule: The constructed classification rule.
    """
    class_counts = node_value[0]
    class_index = int(np.argmax(class_counts))
    max_count = float(class_counts[class_index])
    total_count = float(class_counts.sum())
    confidence = max_count / total_count if total_count > 0 else 0.0

    if target_mapping is not None:
        prediction: str | float = target_mapping[class_index]
    else:
        prediction = class_index

    return ClassificationRule(
        task_type="classification",
        predicates=path_predicates,
        prediction=prediction,
        samples=n_samples,
        confidence=round(confidence, 4),
    )


def _build_regression_rule(
    node_value: np.ndarray,
    n_samples: int,
    node_id: int,
    *,
    path_predicates: list[Predicate],
    target_array: np.ndarray,
    leaf_assignments: np.ndarray | None,
) -> RegressionRule:
    """Construct a regression rule from a leaf node's stored mean and leaf samples.

    The standard deviation is computed from the actual training samples
    assigned to this leaf, identified via `leaf_assignments`.

    Args:
        node_value (np.ndarray): The `tree.tree_.value[node_id]` array with
            shape `(1, 1)` containing the leaf mean.
        n_samples (int): Total number of training samples at this leaf.
        node_id (int): Index of the leaf node in the internal tree structure.
        path_predicates (list[Predicate]): Predicates along the root-to-leaf path.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)`.

    Returns:
        RegressionRule: The constructed regression rule.
    """
    prediction = float(node_value[0, 0])

    leaf_std = 0.0
    if leaf_assignments is not None:
        leaf_mask = leaf_assignments == node_id
        leaf_target = target_array[leaf_mask]
        if len(leaf_target) > 0:
            leaf_std = float(np.std(leaf_target))

    return RegressionRule(
        task_type="regression",
        predicates=path_predicates,
        prediction=round(prediction, 4),
        samples=n_samples,
        std=round(leaf_std, 4),
    )


# ---------------------------------------------------------------------------
# Private helpers -- Metrics and feature importance
# ---------------------------------------------------------------------------


def _compute_metrics(
    tree: DecisionTree,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    task_type: DecisionTreeTask,
) -> dict[str, float]:
    """Compute evaluation metrics for a fitted decision tree.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_matrix (np.ndarray): 2-D feature matrix with shape `(n_samples, n_features)`.
        target_array (np.ndarray): 1-D target vector with shape `(n_samples,)`.
        task_type (DecisionTreeTask): Whether the tree is a classifier or regressor.

    Returns:
        dict[str, float]: For classification: `{"accuracy": <float>}`.
            For regression: `{"r_squared": <float>, "rmse": <float>}`.
    """
    predictions = tree.predict(feature_matrix)
    if task_type == "classification":
        return {"accuracy": float(accuracy_score(target_array, predictions))}
    return {
        "r_squared": float(r2_score(target_array, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target_array, predictions))),
    }


def _compute_feature_importance(
    tree: DecisionTree,
    feature_names: list[str],
) -> dict[str, float]:
    """Build a feature importance mapping from a fitted decision tree.

    Only features with non-zero importance are included. The remaining
    importances are renormalized to sum to 1.0 after excluding zero-importance
    features.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_names (list[str]): Feature names parallel to `tree.feature_importances_`.

    Returns:
        dict[str, float]: Mapping of feature name to rounded importance score,
            sorted in descending order of importance. Keys are exactly those
            features with importance > 0.
    """
    importances = tree.feature_importances_
    paired = [(name, round(float(importance), 4)) for name, importance in zip(feature_names, importances, strict=True)]
    filtered = [(name, importance) for name, importance in paired if importance > 0.0]
    filtered.sort(key=lambda item: item[1], reverse=True)
    total_rounded = sum(importance for _, importance in filtered)
    renormalized = [(name, round(importance / total_rounded, 4)) for name, importance in filtered]
    if renormalized:
        others_sum = sum(importance for _, importance in renormalized[:-1])
        last_name = renormalized[-1][0]
        renormalized[-1] = (last_name, round(1.0 - others_sum, 4))
    return dict(renormalized)


# ---------------------------------------------------------------------------
# Private helpers -- Pipeline orchestration
# ---------------------------------------------------------------------------


_MAX_TREE_DEPTH: int = 6  # Caps tree depth to prevent overfitting and keep rules human-readable.
_AUTO_MIN_SAMPLES_FRACTION: float = 0.02  # Scales the leaf floor with dataset size: 2% of n_rows.
_AUTO_MIN_SAMPLES_FLOOR: int = 5  # Absolute minimum leaf size regardless of dataset size.


def _build_decision_tree_result(
    df: pl.DataFrame,
    target: str,
    *,
    features: list[str] | None,
    max_depth: int,
    min_samples_leaf: int,
    task_type: str | None,
    random_state: int | None = None,
) -> DecisionTreeResult:
    """Orchestrate the full decision tree fitting pipeline.

    Validates inputs, preprocesses features and target, fits a decision tree,
    extracts rules, computes metrics and feature importance, then assembles
    and returns a `DecisionTreeResult`.

    Args:
        df (pl.DataFrame): The source DataFrame containing features and target.
        target (str): Name of the target column.
        features (list[str] | None): Feature column names to consider. When
            `None`, all columns except `target` are used.
        max_depth (int): Maximum tree depth; clamped to `_MAX_TREE_DEPTH`.
        min_samples_leaf (int): Minimum samples required at a leaf node.
            Auto-adjusted upward when the dataset is large.
        task_type (str | None): `"classification"`, `"regression"`, or `None`
            for automatic detection.
        random_state (int | None): Random seed passed to the sklearn tree for
            reproducibility.  `None` means non-deterministic.

    Returns:
        DecisionTreeResult: The fitted tree result.

    Raises:
        ValueError: On any validation failure (missing columns, degenerate
            target, no valid features, or insufficient samples).
    """
    _validate_inputs(df, target, features)

    feature_columns = features if features is not None else [col for col in df.columns if col != target]

    # Drop null-target rows before feature filtering so the cardinality ratio
    # uses the same denominator as the data the tree will be fitted on.
    df_clean = _prepare_clean_dataframe(df, target, min_samples_leaf)

    kept_columns, excluded_features = _filter_features(df_clean, feature_columns)

    if not kept_columns:
        excluded_labels = [f"{ef.name} ({ef.reason})" for ef in excluded_features]
        raise ValueError(f"No valid feature columns remain after filtering. Excluded: {excluded_labels}")

    n_rows = len(df_clean)
    return _fit_and_assemble_result(
        df_clean=df_clean,
        target=target,
        kept_columns=kept_columns,
        excluded_features=excluded_features,
        n_rows=n_rows,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        task_type=task_type,
        random_state=random_state,
    )


def _validate_inputs(
    df: pl.DataFrame,
    target: str,
    features: list[str] | None,
) -> None:
    """Validate that the target and requested feature columns exist in the DataFrame.

    Args:
        df (pl.DataFrame): The source DataFrame to check.
        target (str): The target column name.
        features (list[str] | None): Requested feature column names, or `None`
            to use all non-target columns.
    """
    _validate_target_column(df, target)
    feature_columns = features if features is not None else [col for col in df.columns if col != target]
    _validate_feature_columns_exist(df, feature_columns)


def _prepare_clean_dataframe(
    df: pl.DataFrame,
    target: str,
    min_samples_leaf: int,
) -> pl.DataFrame:
    """Validate the target series and drop rows with null targets.

    Args:
        df (pl.DataFrame): The source DataFrame.
        target (str): The target column name.
        min_samples_leaf (int): Minimum required non-null rows after dropping nulls.

    Returns:
        pl.DataFrame: The DataFrame with null-target rows removed.

    Raises:
        ValueError: If the target series is degenerate or too few non-null
            rows remain after dropping nulls.
    """
    _validate_target_series(df[target], target)

    df_clean = df.drop_nulls(subset=[target])
    n_rows = len(df_clean)

    if n_rows < min_samples_leaf:
        raise ValueError(
            f"Only {n_rows} non-null rows remain after dropping null targets, but min_samples_leaf={min_samples_leaf}."
        )

    return df_clean


def _fit_and_assemble_result(
    df_clean: pl.DataFrame,
    target: str,
    *,
    kept_columns: list[str],
    excluded_features: list[_ExcludedFeature],
    n_rows: int,
    max_depth: int,
    min_samples_leaf: int,
    task_type: str | None,
    random_state: int | None,
) -> DecisionTreeResult:
    """Encode data, fit a tree, extract rules, and assemble the result object.

    Args:
        df_clean (pl.DataFrame): DataFrame with null-target rows removed.
        target (str): Target column name.
        kept_columns (list[str]): Feature column names that survived filtering.
        excluded_features (list[_ExcludedFeature]): Excluded features with reasons.
        n_rows (int): Number of rows in `df_clean`.
        max_depth (int): Maximum tree depth; clamped internally to `_MAX_TREE_DEPTH`.
        min_samples_leaf (int): User-supplied minimum samples per leaf.
        task_type (str | None): Task type override or `None` for auto-detection.
        random_state (int | None): Random seed for the sklearn tree estimator.

    Returns:
        DecisionTreeResult: The fully assembled decision tree result.
    """
    detected_task_type = _detect_task_type(df_clean[target], task_type)
    feature_matrix, feature_encoders = _encode_features(df_clean, kept_columns)
    target_array, target_mapping = _encode_target(df_clean[target], detected_task_type)

    clamped_depth = min(max_depth, _MAX_TREE_DEPTH)
    effective_min_samples = _compute_effective_min_samples(n_rows, min_samples_leaf)

    fitted_tree = _fit_tree(
        feature_matrix,
        target_array,
        task_type=detected_task_type,
        max_depth=clamped_depth,
        min_samples_leaf=effective_min_samples,
        random_state=random_state,
    )
    rules = _extract_rules(
        fitted_tree,
        feature_matrix,
        target_array,
        feature_encoders=feature_encoders,
        target_mapping=target_mapping,
        task_type=detected_task_type,
    )
    metrics = _compute_metrics(fitted_tree, feature_matrix, target_array, task_type=detected_task_type)
    feature_importance = _compute_feature_importance(fitted_tree, kept_columns)
    # `kept_columns` that have zero importance are dropped from `features_used`;
    # only columns that actually contributed to splits appear in `feature_importance`.
    features_used = [col for col in kept_columns if col in feature_importance]
    zero_importance = [
        _ExcludedFeature(name=col, reason="zero importance") for col in kept_columns if col not in feature_importance
    ]
    features_excluded_labels = [f"{ef.name} ({ef.reason})" for ef in [*excluded_features, *zero_importance]]

    return DecisionTreeResult(
        target=target,
        task_type=detected_task_type,
        features_used=features_used,
        features_excluded=features_excluded_labels,
        rules=rules,
        feature_importance=feature_importance,
        metrics=metrics,
        sample_count=n_rows,
        depth=fitted_tree.get_depth(),
        leaf_count=fitted_tree.get_n_leaves(),
    )


def _validate_target_column(df: pl.DataFrame, target: str) -> None:
    """Raise `ValueError` if the target column does not exist in the DataFrame.

    Args:
        df (pl.DataFrame): The source DataFrame to check.
        target (str): The target column name to look up.

    Raises:
        ValueError: If `target` is not present in `df.columns`.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")


def _validate_feature_columns_exist(df: pl.DataFrame, feature_columns: list[str]) -> None:
    """Raise `ValueError` if any requested feature columns are missing.

    Args:
        df (pl.DataFrame): The source DataFrame to check.
        feature_columns (list[str]): The feature column names to look up.

    Raises:
        ValueError: If any column in `feature_columns` is absent from `df.columns`.
    """
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Requested feature columns not found in DataFrame: {missing_columns}")


def _validate_target_series(series: pl.Series, target: str) -> None:
    """Raise `ValueError` if the target series is degenerate.

    A target series is considered degenerate when all values are null or when
    it contains only a single unique non-null value.

    Args:
        series (pl.Series): The target column to inspect.
        target (str): The column name, used in error messages.

    Raises:
        ValueError: If all values are null or there is only one unique non-null value.
    """
    if series.is_null().all():
        raise ValueError(f"Target column '{target}' contains only null values.")
    non_null_series = series.drop_nulls()
    if non_null_series.n_unique() <= 1:
        raise ValueError(f"Target column '{target}' must have more than one unique value.")


def _compute_effective_min_samples(n_rows: int, user_min_samples_leaf: int) -> int:
    """Compute the effective `min_samples_leaf` to pass to the sklearn tree.

    The effective value is the maximum of the user-supplied value and an
    auto-adjusted floor of `max(_AUTO_MIN_SAMPLES_FLOOR, 2% of n_rows)`.
    This prevents overfitting on large datasets while respecting explicit
    user intent.

    Args:
        n_rows (int): Number of training samples after null-target rows are dropped.
        user_min_samples_leaf (int): The user-supplied `min_samples_leaf` parameter.

    Returns:
        int: The effective `min_samples_leaf` value to use.
    """
    auto_floor = max(_AUTO_MIN_SAMPLES_FLOOR, int(_AUTO_MIN_SAMPLES_FRACTION * n_rows))
    return max(user_min_samples_leaf, auto_floor)
