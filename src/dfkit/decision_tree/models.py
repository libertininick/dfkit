"""Pydantic result models and predicate logic for the decision tree module."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
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
