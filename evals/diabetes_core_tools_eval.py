"""Reference eval using the `dfkit.evaluation` harness against core tools only.

This module demonstrates the standard pattern for writing evals with the dfkit
evaluation harness. It uses the scikit-learn diabetes dataset and evaluates how
well an LLM agent uses the core DataFrame tools to answer analytical questions.

Expected pass-rate range: 0.7-1.0. LLM variance is normal; results below 0.7
on a given run do not necessarily indicate a regression.

Agent/judge split:
- **Agent (Haiku)**: Executes tools and produces answers.
- **Judge (Sonnet)**: Extracts structured facts from the agent's free-text
  response and compares them against the expected `EvalCase` ground truth.

How to run::

    uv run python evals/diabetes_core_tools_eval.py

To add verbose tool-call logging, edit the ``__main__`` guard to::

    with enable_logging(): main()
"""

from __future__ import annotations

import sys
from typing import Final, Literal

import polars as pl
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel
from sklearn import datasets

from dfkit import DataFrameToolkit
from dfkit.chat_models_config import ModelName, get_chat_model
from dfkit.evaluation import (
    Confounders,
    Correlation,
    EvalCase,
    EvalResult,
    EvalSummary,
    Metric,
    NonNegativeMetric,
    RelationshipStrength,
    SampleCount,
    evaluate_agent,
    summarize_results,
)

RTOL: Final[float] = 0.1
ATOL: Final[float] = 0.05
CONFOUNDER_CORR_THRESHOLD: Final[float] = 0.2
PASS_RATE_THRESHOLD: Final[float] = 0.5
DATASET_NAME: Final[str] = "Diabetes Progression Dataset"


# region Load dataset


def load_diabetes_dataset() -> pl.DataFrame:
    """Load the scikit-learn diabetes dataset as a polars DataFrame.

    Mirrors the construction in `examples/diabetes_dataset_exploration.ipynb`:
    loads the unscaled diabetes data, maps the numeric `sex` column to
    `"male"`/`"female"` strings, and appends the `disease_progression` target.

    Returns:
        pl.DataFrame: Diabetes dataset with 11 columns and 442 rows.
    """
    data, target = datasets.load_diabetes(return_X_y=True, scaled=False)
    df = pl.DataFrame(
        data=data,
        schema=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
    )
    return df.with_columns(
        pl.col("sex").map_elements(lambda x: "male" if x == 1 else "female", return_dtype=pl.String),
        pl.Series(target).alias("disease_progression"),
    )


def register_dataset(toolkit: DataFrameToolkit, df: pl.DataFrame) -> None:
    """Register the diabetes dataset with the toolkit.

    Calls `toolkit.register_dataframe` with the canonical name, description,
    and column descriptions from `examples/diabetes_dataset_exploration.ipynb`.

    Args:
        toolkit (DataFrameToolkit): The toolkit instance to register the dataset with.
        df (pl.DataFrame): The diabetes DataFrame returned by `load_diabetes_dataset`.
    """
    toolkit.register_dataframe(
        name=DATASET_NAME,
        dataframe=df,
        description="""
    Ten baseline variables, age, sex, body mass index, average blood pressure,
    and six blood serum measurements were obtained for each diabetes patient,
    as well as the response of interest, a quantitative measure of disease
    progression one year after baseline.
    """,
        column_descriptions={
            "age": "Age of the patient in years.",
            "sex": "Sex of the patient",
            "bmi": "Body mass index.",
            "bp": "Average blood pressure.",
            "s1": "TC, total serum cholesterol.",
            "s2": "LDL, low-density lipoproteins.",
            "s3": "HDL, high-density lipoproteins.",
            "s4": "TCH, total cholesterol / HDL.",
            "s5": "LTG, possibly log of serum triglycerides level.",
            "s6": "GLU, blood sugar level.",
            "disease_progression": "A quantitative measure of disease progression one year after baseline.",
        },
    )


# endregion


# region Result schemas and helpers


class CorrelationResult(BaseModel):
    """Correlation between a feature and the target variable.

    Attributes:
        feature_name (str | None): Name of the feature being correlated.
        correlation (Correlation | None): Pearson correlation in [-1, 1].
        strength (RelationshipStrength | None): Categorical strength label.
        sample_count (SampleCount | None): Number of observations used.
    """

    feature_name: str | None = None
    correlation: Correlation | None = None
    strength: RelationshipStrength | None = None
    sample_count: SampleCount | None = None


class PartialCorrelationResult(BaseModel):
    """Partial correlation controlling for a confounding variable.

    Captures the raw bivariate correlation, the partial correlation after
    removing a control variable's influence, and the percentage change
    between them.

    Attributes:
        raw_correlation (Correlation | None): Bivariate Pearson correlation
            before controlling.
        partial_correlation (Correlation | None): Partial Pearson correlation
            after removing the control variable's linear effect.
        change_pct (Metric | None): Percentage change from raw to partial,
            computed as ``(partial - raw) / abs(raw) * 100``.
    """

    raw_correlation: Correlation | None = None
    partial_correlation: Correlation | None = None
    change_pct: Metric | None = None


class InteractionResult(BaseModel):
    """Interaction effect between two grouping variables on a target.

    Tests whether the effect of one variable (e.g. high vs low BMI) on the
    target differs across levels of another variable (e.g. sex).

    Attributes:
        male_effect (Metric | None): Effect size (high - low group mean)
            within the male subgroup.
        female_effect (Metric | None): Effect size (high - low group mean)
            within the female subgroup.
        interaction_magnitude (Metric | None): Signed difference between
            the male and female effects (male_effect - female_effect).
    """

    male_effect: Metric | None = None
    female_effect: Metric | None = None
    interaction_magnitude: Metric | None = None


class StratifiedImpactResult(BaseModel):
    """Comparison of a raw effect to an age-stratified (adjusted) effect.

    Tests for Simpson's paradox by comparing the naive group difference
    to the average within-stratum difference after stratifying by a
    potential confounder.

    Attributes:
        raw_effect (Metric | None): Naive difference in target mean between
            treatment and control groups (no stratification).
        adjusted_effect (Metric | None): Average of within-stratum effects
            after stratifying by a confounder (e.g. age tertiles).
        confounded (Literal["yes", "no"] | None): Whether the raw effect is
            materially confounded. ``"yes"`` if the absolute difference
            between raw and adjusted exceeds 10% of the raw effect magnitude.
    """

    raw_effect: Metric | None = None
    adjusted_effect: Metric | None = None
    confounded: Literal["yes", "no"] | None = None


class ConfoundingResult(BaseModel):
    """Confounder analysis for a primary correlation.

    Attributes:
        primary_correlation (Correlation | None): Pearson correlation of the
            primary variable with the target.
        confounders (Confounders | None): Candidate confounder variable names,
            sorted alphabetically. Must be in alphabetical order for exact-match
            comparison with ground truth.
    """

    primary_correlation: Correlation | None = None
    confounders: Confounders | None = None


class OutlierInfluenceResult(BaseModel):
    """Sensitivity of a correlation to extreme values.

    Measures how much a bivariate correlation changes when outliers
    (observations above a percentile threshold) are removed.

    Attributes:
        full_correlation (Correlation | None): Pearson correlation computed
            on the full dataset.
        trimmed_correlation (Correlation | None): Pearson correlation after
            removing observations above the 95th percentile of the feature.
        correlation_change (Metric | None): Signed difference
            ``trimmed_correlation - full_correlation``.
        outlier_count (SampleCount | None): Number of observations removed.
    """

    full_correlation: Correlation | None = None
    trimmed_correlation: Correlation | None = None
    correlation_change: Metric | None = None
    outlier_count: SampleCount | None = None


class TrendResult(BaseModel):
    """Trend of a target variable across ordered buckets of a feature.

    Extends basic trend detection with a monotonicity check to distinguish
    linear trends from noisy or non-monotonic patterns.

    Attributes:
        slope (Metric | None): Last-bucket-to-first-bucket difference in
            the target mean (signed).
        direction (Literal["increasing", "decreasing", "flat"] | None):
            Qualitative direction of the trend.
        monotonic (Literal["yes", "no"] | None): Whether the per-bucket
            means are monotonically ordered in the direction of the trend.
    """

    slope: Metric | None = None
    direction: Literal["increasing", "decreasing", "flat"] | None = None
    monotonic: Literal["yes", "no"] | None = None


class SubgroupGapResult(BaseModel):
    """Feature that creates the largest subgroup gap in a target variable.

    For each candidate feature, patients are split at the feature's median
    into above-median and below-median groups, and the gap in target mean
    is computed. This schema captures the feature with the largest gap.

    Attributes:
        feature_name (str | None): Name of the feature producing the
            largest absolute gap.
        high_group_mean (Metric | None): Mean target value for patients
            above the feature's median.
        low_group_mean (Metric | None): Mean target value for patients
            at or below the feature's median.
        gap (NonNegativeMetric | None): Absolute difference between the
            two group means.
    """

    feature_name: str | None = None
    high_group_mean: Metric | None = None
    low_group_mean: Metric | None = None
    gap: NonNegativeMetric | None = None


class PercentileProfileResult(BaseModel):
    """Comparison of target outcomes between extreme subgroups of a feature.

    Compares patients in the bottom quartile vs top quartile of a feature
    to quantify how extreme values relate to the target.

    Attributes:
        bottom_quartile_mean (Metric | None): Mean target value for patients
            at or below the 25th percentile of the feature.
        top_quartile_mean (Metric | None): Mean target value for patients
            at or above the 75th percentile of the feature.
        gap (Metric | None): Signed difference
            ``top_quartile_mean - bottom_quartile_mean``.
        ratio (Metric | None): Ratio ``top_quartile_mean / bottom_quartile_mean``.
    """

    bottom_quartile_mean: Metric | None = None
    top_quartile_mean: Metric | None = None
    gap: Metric | None = None
    ratio: Metric | None = None


class CorrelationStabilityResult(BaseModel):
    """Stability of a correlation across subgroups.

    Tests whether a bivariate relationship holds consistently by computing
    correlations separately in two subgroups and comparing them.

    Attributes:
        young_correlation (Correlation | None): Pearson correlation in the
            younger subgroup (below median age).
        old_correlation (Correlation | None): Pearson correlation in the
            older subgroup (at or above median age).
        difference (NonNegativeMetric | None): Absolute difference between
            the two subgroup correlations.
        stability (Literal["stable", "unstable"] | None): ``"stable"`` if
            the absolute difference is less than 0.1, ``"unstable"`` otherwise.
    """

    young_correlation: Correlation | None = None
    old_correlation: Correlation | None = None
    difference: NonNegativeMetric | None = None
    stability: Literal["stable", "unstable"] | None = None


def compute_confounders(
    df: pl.DataFrame,
    primary: str,
    target: str,
    *,
    threshold: float = CONFOUNDER_CORR_THRESHOLD,
) -> list[str]:
    """Identify confounding variables for a primary-target correlation.

    A candidate column is a confounder iff:
    (a) it is not ``primary`` or ``target`` itself,
    (b) it is numeric,
    (c) its absolute Pearson correlation with ``primary`` exceeds ``threshold``,
    and (d) its absolute Pearson correlation with ``target`` exceeds ``threshold``.

    Args:
        df (pl.DataFrame): The dataset.
        primary (str): Name of the primary variable.
        target (str): Name of the target variable.
        threshold (float): Minimum absolute correlation with both ``primary``
            and ``target`` for a variable to qualify as a confounder.

    Returns:
        list[str]: Sorted list of confounder column names.
    """
    numeric_cols = [col for col, dtype in df.schema.items() if dtype.is_numeric() and col not in {primary, target}]
    confounders: list[str] = []
    for col in numeric_cols:
        corr_with_primary = df.select(pl.corr(col, primary)).item()
        corr_with_target = df.select(pl.corr(col, target)).item()
        if abs(corr_with_primary) > threshold and abs(corr_with_target) > threshold:
            confounders.append(col)
    return sorted(confounders)


def get_numeric_features(df: pl.DataFrame, target: str) -> list[str]:
    """Return sorted names of numeric feature columns (excludes sex and target).

    Args:
        df (pl.DataFrame): The dataset.
        target (str): Name of the target column to exclude.

    Returns:
        list[str]: Sorted list of numeric feature column names.
    """
    return sorted(col for col, dtype in df.schema.items() if dtype.is_numeric() and col not in {"sex", target})


# endregion
