"""Composable annotated field building blocks for evaluation schemas.

This module defines reusable annotated type aliases that encode validation
constraints and descriptions via `pydantic.Field`. Import these aliases
directly into Pydantic model definitions to avoid repeating constraint
metadata across evaluation result models.

Examples:
    >>> from dfkit.evaluation.fields import Correlation, Probability
    >>> from pydantic import BaseModel
    >>> class FeaturePair(BaseModel):
    ...     correlation: Correlation | None = None
    ...     p_value: Probability | None = None
    >>> pair_stats = FeaturePair(correlation=0.63)
"""

from typing import Annotated, Literal

from pydantic import Field

__all__ = [
    "Confounders",
    "Correlation",
    "Importance",
    "Metric",
    "NonNegativeMetric",
    "Probability",
    "Rank",
    "RelationshipStrength",
    "SampleCount",
]

# region Public type aliases

type Confounders = Annotated[
    list[Annotated[str, Field(description="Non-empty variable name.", min_length=1)]],
    Field(
        description="Other variables that confound relationship with target.",
    ),
]

type Correlation = Annotated[
    float,
    Field(
        description="Linear correlation between two variables.",
        ge=-1,
        le=1,
        allow_inf_nan=False,
    ),
]

type Importance = Annotated[
    float,
    Field(
        description="Score that quantifies the feature's contribution to the model's predictive power.",
        ge=0,
        le=1,
        allow_inf_nan=False,
    ),
]

type Metric = Annotated[
    float,
    Field(
        description="Finite numeric measurement or coefficient.",
        allow_inf_nan=False,
    ),
]

type NonNegativeMetric = Annotated[
    float,
    Field(
        description="Non-negative measurement such as an error, loss, or dispersion metric.",
        ge=0,
        allow_inf_nan=False,
    ),
]

type Probability = Annotated[
    float,
    Field(
        description="Statistical probability of an outcome or prediction.",
        ge=0,
        le=1,
        allow_inf_nan=False,
    ),
]

type Rank = Annotated[
    int,
    Field(
        description="Ordinal position in a ranked sequence.",
        ge=1,
    ),
]

type RelationshipStrength = Annotated[
    Literal["Strong", "Moderate", "Weak", "Negligible"],
    Field(
        description="Strength of relationship between two variables.",
    ),
]

type SampleCount = Annotated[
    int,
    Field(
        description="Count of observations or samples in a group.",
        ge=0,
    ),
]

# endregion
