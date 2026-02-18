"""Pydantic result models for the decision tree module.

Provides structured output types for decision tree analysis, designed for
LLM readability. These models capture the rules, metrics, and feature
importance produced by the build_decision_tree tool.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ClassificationRule(BaseModel):
    """A decision rule extracted from a classification tree leaf node.

    Represents the path from the root of the tree to one leaf, expressed as
    a list of human-readable conditions, together with the predicted class
    label and the confidence at that leaf.

    Attributes:
        task_type (Literal["classification"]): Discriminator field; always
            ``"classification"``.
        conditions (list[str]): Human-readable conditions along the path from
            root to this leaf, e.g. ``["tenure_months > 6", "support_tickets <= 3"]``.
            An empty list means the tree has depth zero (a single leaf).
        prediction (str | float): Predicted class label for samples reaching
            this leaf.
        samples (int): Number of training samples that reached this leaf.
        confidence (float): Fraction of samples at this leaf belonging to the
            majority (predicted) class. Ranges from 0.0 to 1.0.

    Examples:
        >>> rule = ClassificationRule(
        ...     task_type="classification",
        ...     conditions=["tenure_months > 6", "support_tickets <= 3"],
        ...     prediction="retained",
        ...     samples=342,
        ...     confidence=0.91,
        ... )
    """

    task_type: Literal["classification"] = Field(
        description='Discriminator field. Always "classification".',
    )
    conditions: list[str] = Field(
        description=(
            "Human-readable conditions along the path from root to this leaf, "
            'e.g. ["tenure_months > 6", "support_tickets <= 3"]. '
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
    a list of human-readable conditions, together with the predicted mean
    target value and the spread of target values at that leaf.

    Attributes:
        task_type (Literal["regression"]): Discriminator field; always
            ``"regression"``.
        conditions (list[str]): Human-readable conditions along the path from
            root to this leaf, e.g. ``["age > 30", "income <= 50000"]``.
            An empty list means the tree has depth zero (a single leaf).
        prediction (str | float): Mean target value for samples reaching this
            leaf.
        samples (int): Number of training samples that reached this leaf.
        std (float): Standard deviation of target values among samples at this
            leaf. Indicates prediction spread.

    Examples:
        >>> rule = RegressionRule(
        ...     task_type="regression",
        ...     conditions=["age > 30", "income <= 50000"],
        ...     prediction=42500.0,
        ...     samples=187,
        ...     std=3200.5,
        ... )
    """

    task_type: Literal["regression"] = Field(
        description='Discriminator field. Always "regression".',
    )
    conditions: list[str] = Field(
        description=(
            "Human-readable conditions along the path from root to this leaf, "
            'e.g. ["age > 30", "income <= 50000"]. '
            "Empty list indicates a single-leaf tree with no splits."
        ),
    )
    prediction: str | float = Field(
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


DecisionTreeRule = Annotated[
    ClassificationRule | RegressionRule,
    Field(discriminator="task_type"),
]
"""Union of :class:`ClassificationRule` and :class:`RegressionRule`.

Discriminated by the ``task_type`` field. Use this type when accepting a rule
of either task type. Pydantic will select the correct model automatically.
"""


class DecisionTreeResult(BaseModel):
    """Structured output from the build_decision_tree tool.

    Captures the full result of fitting a decision tree, including the
    human-readable rules extracted from each leaf, feature importance scores,
    evaluation metrics, and tree structure metadata.

    Attributes:
        target (str): Target column name used as the prediction label.
        task_type (Literal["classification", "regression"]): Either
            ``"classification"`` or ``"regression"``.
        features_used (list[str]): Feature column names that were included when
            fitting the tree.
        features_excluded (list[str]): Feature column names that were excluded,
            each annotated with the reason for exclusion.
        rules (list[DecisionTreeRule]): One rule per leaf node, describing the
            conditions and prediction for that path through the tree. Each
            rule's ``task_type`` field identifies its concrete type.
        feature_importance (dict[str, float]): Mapping of feature name to
            importance score, sorted in descending order of importance. Scores
            must sum to 1.0 across all features used.
        metrics (dict[str, float]): Evaluation metrics for the fitted tree,
            e.g. ``{"accuracy": 0.83}`` for classification or
            ``{"r_squared": 0.76, "rmse": 68200.0}`` for regression.
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
        ...             conditions=["tenure_months <= 6"],
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
    task_type: Literal["classification", "regression"] = Field(
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
            "One rule per leaf node, describing the path conditions and prediction "
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
    def _validate_rules_count_matches_leaf_count(self) -> DecisionTreeResult:
        """Validate that the number of rules equals the number of leaf nodes.

        Returns:
            DecisionTreeResult: The validated model instance.

        Raises:
            ValueError: If ``len(rules)`` does not equal ``leaf_count``.
        """
        if len(self.rules) != self.leaf_count:
            raise ValueError(f"rules length ({len(self.rules)}) must equal leaf_count ({self.leaf_count})")
        return self
