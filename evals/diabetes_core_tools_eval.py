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

To add verbose tool-call logging, edit the `__main__` guard to::

    with enable_logging(): main()
"""

from __future__ import annotations

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
            computed as `(partial - raw) / abs(raw) * 100`.
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
            materially confounded. `"yes"` if the absolute difference
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
            `trimmed_correlation - full_correlation`.
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
            `top_quartile_mean - bottom_quartile_mean`.
        ratio (Metric | None): Ratio `top_quartile_mean / bottom_quartile_mean`.
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
        stability (Literal["stable", "unstable"] | None): `"stable"` if
            the absolute difference is less than 0.1, `"unstable"` otherwise.
    """

    young_correlation: Correlation | None = None
    old_correlation: Correlation | None = None
    difference: NonNegativeMetric | None = None
    stability: Literal["stable", "unstable"] | None = None


type GroundTruthResult = (
    CorrelationResult
    | PartialCorrelationResult
    | InteractionResult
    | StratifiedImpactResult
    | ConfoundingResult
    | OutlierInfluenceResult
    | TrendResult
    | SubgroupGapResult
    | PercentileProfileResult
    | CorrelationStabilityResult
)


def compute_confounders(
    df: pl.DataFrame,
    primary: str,
    target: str,
    *,
    threshold: float = CONFOUNDER_CORR_THRESHOLD,
) -> list[str]:
    """Identify confounding variables for a primary-target correlation.

    A candidate column is a confounder iff:
    (a) it is not `primary` or `target` itself,
    (b) it is numeric,
    (c) its absolute Pearson correlation with `primary` exceeds `threshold`,
    and (d) its absolute Pearson correlation with `target` exceeds `threshold`.

    Args:
        df (pl.DataFrame): The dataset.
        primary (str): Name of the primary variable.
        target (str): Name of the target variable.
        threshold (float): Minimum absolute correlation with both `primary`
            and `target` for a variable to qualify as a confounder.

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


# region Ground truth


def build_ground_truth(df: pl.DataFrame) -> dict[str, GroundTruthResult]:
    """Precompute expected-result models for all 10 eval cases.

    Every numeric value is derived directly from `df` using polars; no
    hard-coded literals. Each private helper computes ground truth for one
    or more related cases.

    Args:
        df (pl.DataFrame): Diabetes dataset produced by `load_diabetes_dataset`.

    Returns:
        dict[str, GroundTruthResult]: Mapping of case id to expected-result model.
    """
    return {
        **_build_corr_strongest_case(df),
        **_build_partial_corr_case(df),
        **_build_interaction_case(df),
        **_build_simpson_case(df),
        **_build_confounder_case(df),
        **_build_outlier_influence_case(df),
        **_build_trend_case(df),
        **_build_subgroup_gap_case(df),
        **_build_percentile_profile_case(df),
        **_build_correlation_stability_case(df),
    }


_STRONG_THRESHOLD: Final[float] = 0.5
_MODERATE_THRESHOLD: Final[float] = 0.3
_WEAK_THRESHOLD: Final[float] = 0.1
_FLAT_SLOPE_EPSILON: Final[float] = 0.15
_CONFOUNDING_CHANGE_THRESHOLD: Final[float] = 0.1
_STABILITY_DIFFERENCE_THRESHOLD: Final[float] = 0.1


def _classify_correlation_strength(correlation: float) -> RelationshipStrength:
    """Map absolute correlation to a categorical relationship strength label.

    Args:
        correlation (float): Pearson correlation coefficient in [-1, 1].

    Returns:
        RelationshipStrength: One of `"Strong"`, `"Moderate"`, `"Weak"`,
            or `"Negligible"`.
    """
    value = abs(correlation)
    if value >= _STRONG_THRESHOLD:
        return "Strong"
    if value >= _MODERATE_THRESHOLD:
        return "Moderate"
    if value >= _WEAK_THRESHOLD:
        return "Weak"
    return "Negligible"


def _classify_trend_direction(slope: float) -> Literal["increasing", "decreasing", "flat"]:
    """Classify the direction of a trend slope.

    Uses an epsilon of 0.15 to classify near-zero slopes as `"flat"`.
    This threshold was chosen empirically to avoid labelling small noise in
    quintile means as a meaningful directional trend; it roughly corresponds
    to a one-unit change per quintile spread across the disease-progression
    scale.

    Args:
        slope (float): Last-bucket-to-first-bucket difference in target mean.

    Returns:
        Literal["increasing", "decreasing", "flat"]: Qualitative direction.
    """
    if abs(slope) < _FLAT_SLOPE_EPSILON:
        return "flat"
    return "increasing" if slope > 0 else "decreasing"


def _build_corr_strongest_case(df: pl.DataFrame) -> dict[str, CorrelationResult]:
    """Compute ground truth for the strongest-correlation eval case.

    Identifies the numeric feature with the highest absolute Pearson
    correlation to `disease_progression`.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, CorrelationResult]: Single-entry dict keyed `"corr-strongest"`.
    """
    target = "disease_progression"
    numeric_features = get_numeric_features(df, target)
    # Agent must: compute CORR(<col>, target) for each feature, then pick max |corr|
    feature_corrs = {col: df.select(pl.corr(col, target)).item() for col in numeric_features}
    strongest_feature = max(feature_corrs, key=lambda c: abs(feature_corrs[c]))
    strongest_corr = feature_corrs[strongest_feature]
    return {
        "corr-strongest": CorrelationResult(
            feature_name=strongest_feature,
            correlation=strongest_corr,
            strength=_classify_correlation_strength(strongest_corr),
            sample_count=df.height,
        ),
    }


def _build_partial_corr_case(df: pl.DataFrame) -> dict[str, PartialCorrelationResult]:
    """Compute ground truth for the partial-correlation eval case.

    Calculates the partial correlation between `s5` and
    `disease_progression` after controlling for `bmi`.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, PartialCorrelationResult]: Single-entry dict keyed `"partial-corr-s5"`.
    """
    target = "disease_progression"
    r_xy = df.select(pl.corr("s5", target)).item()
    r_xz = df.select(pl.corr("s5", "bmi")).item()
    r_yz = df.select(pl.corr("bmi", target)).item()
    partial = (r_xy - r_xz * r_yz) / ((1 - r_xz**2) * (1 - r_yz**2)) ** 0.5
    change_pct = (partial - r_xy) / abs(r_xy) * 100
    return {
        "partial-corr-s5": PartialCorrelationResult(
            raw_correlation=r_xy,
            partial_correlation=partial,
            change_pct=change_pct,
        ),
    }


def _build_interaction_case(df: pl.DataFrame) -> dict[str, InteractionResult]:
    """Compute ground truth for the sex-by-BMI interaction eval case.

    Measures how the BMI effect on disease progression differs between
    male and female patients (split at median BMI).

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, InteractionResult]: Single-entry dict keyed `"interaction-sex-bmi"`.
    """
    target = "disease_progression"
    median_bmi = df.select(pl.median("bmi")).item()
    male_high = df.filter((pl.col("sex") == "male") & (pl.col("bmi") > median_bmi)).select(pl.mean(target)).item()
    male_low = df.filter((pl.col("sex") == "male") & (pl.col("bmi") <= median_bmi)).select(pl.mean(target)).item()
    female_high = df.filter((pl.col("sex") == "female") & (pl.col("bmi") > median_bmi)).select(pl.mean(target)).item()
    female_low = df.filter((pl.col("sex") == "female") & (pl.col("bmi") <= median_bmi)).select(pl.mean(target)).item()
    male_effect = male_high - male_low
    female_effect = female_high - female_low
    return {
        "interaction-sex-bmi": InteractionResult(
            male_effect=male_effect,
            female_effect=female_effect,
            interaction_magnitude=male_effect - female_effect,
        ),
    }


def _build_simpson_case(df: pl.DataFrame) -> dict[str, StratifiedImpactResult]:
    """Compute ground truth for the Simpson's paradox eval case.

    Compares the raw BMI-threshold effect on disease progression against
    the age-stratified (adjusted) effect to detect confounding.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, StratifiedImpactResult]: Single-entry dict keyed `"simpson-bmi-age"`.
    """
    target = "disease_progression"
    raw_high = df.filter(pl.col("bmi") > 30).select(pl.mean(target)).item()
    raw_low = df.filter(pl.col("bmi") <= 30).select(pl.mean(target)).item()
    raw_effect = raw_high - raw_low

    df_with_tertile = df.with_columns(pl.col("age").qcut(3, labels=["young", "middle", "old"]).alias("age_tertile"))
    stratum_effects: list[float] = []
    for label in ["young", "middle", "old"]:
        stratum = df_with_tertile.filter(pl.col("age_tertile") == label)
        high = stratum.filter(pl.col("bmi") > 30)
        low = stratum.filter(pl.col("bmi") <= 30)
        if high.height > 0 and low.height > 0:
            stratum_effects.append(high.select(pl.mean(target)).item() - low.select(pl.mean(target)).item())
    adjusted_effect = sum(stratum_effects) / len(stratum_effects)
    confounded = "yes" if abs(raw_effect - adjusted_effect) / abs(raw_effect) > _CONFOUNDING_CHANGE_THRESHOLD else "no"
    return {
        "simpson-bmi-age": StratifiedImpactResult(
            raw_effect=raw_effect,
            adjusted_effect=adjusted_effect,
            confounded=confounded,
        ),
    }


def _build_confounder_case(df: pl.DataFrame) -> dict[str, ConfoundingResult]:
    """Compute ground truth for the BMI-confounder eval case.

    Identifies variables that confound the BMI-to-disease-progression
    relationship using the standard threshold defined by
    `CONFOUNDER_CORR_THRESHOLD`.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, ConfoundingResult]: Single-entry dict keyed `"confound-bmi"`.
    """
    target = "disease_progression"
    bmi_corr = df.select(pl.corr("bmi", target)).item()
    return {
        "confound-bmi": ConfoundingResult(
            primary_correlation=bmi_corr,
            confounders=compute_confounders(df, "bmi", target),
        ),
    }


def _build_outlier_influence_case(df: pl.DataFrame) -> dict[str, OutlierInfluenceResult]:
    """Compute ground truth for the outlier-influence eval case.

    Measures how removing observations above the 95th BMI percentile
    changes the BMI-to-disease-progression correlation.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, OutlierInfluenceResult]: Single-entry dict keyed `"outlier-influence"`.
    """
    target = "disease_progression"
    full_corr = df.select(pl.corr("bmi", target)).item()
    p95 = df.select(pl.col("bmi").quantile(0.95)).item()
    trimmed = df.filter(pl.col("bmi") <= p95)
    trimmed_corr = trimmed.select(pl.corr("bmi", target)).item()
    return {
        "outlier-influence": OutlierInfluenceResult(
            full_correlation=full_corr,
            trimmed_correlation=trimmed_corr,
            correlation_change=trimmed_corr - full_corr,
            outlier_count=df.height - trimmed.height,
        ),
    }


def _build_trend_case(df: pl.DataFrame) -> dict[str, TrendResult]:
    """Compute ground truth for the age-trend eval case.

    Divides patients into age quintiles and checks whether disease
    progression increases, decreases, or stays flat across them.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, TrendResult]: Single-entry dict keyed `"trend-age"`.
    """
    target = "disease_progression"
    quintile_means = (
        df
        .with_columns(pl.col("age").qcut(5, labels=[str(i) for i in range(5)]).alias("quintile"))
        .group_by("quintile")
        .agg(pl.mean(target).alias("mean_prog"))
        .sort("quintile")
        .get_column("mean_prog")
        .to_list()
    )
    slope = quintile_means[-1] - quintile_means[0]
    direction = _classify_trend_direction(slope)

    diffs = [quintile_means[i + 1] - quintile_means[i] for i in range(len(quintile_means) - 1)]
    if direction == "increasing":
        monotonic = "yes" if all(d >= 0 for d in diffs) else "no"
    elif direction == "decreasing":
        monotonic = "yes" if all(d <= 0 for d in diffs) else "no"
    else:
        monotonic = "yes"  # flat is trivially monotonic

    return {
        "trend-age": TrendResult(
            slope=slope,
            direction=direction,
            monotonic=monotonic,
        ),
    }


def _build_subgroup_gap_case(df: pl.DataFrame) -> dict[str, SubgroupGapResult]:
    """Compute ground truth for the maximum-subgroup-gap eval case.

    For each numeric feature, splits patients at the median and finds
    the feature that maximizes the gap in mean disease progression.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, SubgroupGapResult]: Single-entry dict keyed `"subgroup-max-gap"`.
    """
    target = "disease_progression"
    numeric_features = get_numeric_features(df, target)
    best_feature = ""
    best_high_mean = 0.0
    best_low_mean = 0.0
    best_gap = 0.0

    for col in numeric_features:
        median_val = df.select(pl.median(col)).item()
        high_mean = df.filter(pl.col(col) > median_val).select(pl.mean(target)).item()
        low_mean = df.filter(pl.col(col) <= median_val).select(pl.mean(target)).item()
        gap = abs(high_mean - low_mean)
        if gap > best_gap:
            best_feature = col
            best_high_mean = high_mean
            best_low_mean = low_mean
            best_gap = gap

    return {
        "subgroup-max-gap": SubgroupGapResult(
            feature_name=best_feature,
            high_group_mean=best_high_mean,
            low_group_mean=best_low_mean,
            gap=best_gap,
        ),
    }


def _build_percentile_profile_case(df: pl.DataFrame) -> dict[str, PercentileProfileResult]:
    """Compute ground truth for the BMI percentile-profile eval case.

    Compares mean disease progression in the bottom vs top BMI quartile.

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, PercentileProfileResult]: Single-entry dict keyed `"percentile-profile"`.
    """
    target = "disease_progression"
    p25 = df.select(pl.col("bmi").quantile(0.25)).item()
    p75 = df.select(pl.col("bmi").quantile(0.75)).item()
    bottom_mean = df.filter(pl.col("bmi") <= p25).select(pl.mean(target)).item()
    top_mean = df.filter(pl.col("bmi") >= p75).select(pl.mean(target)).item()
    return {
        "percentile-profile": PercentileProfileResult(
            bottom_quartile_mean=bottom_mean,
            top_quartile_mean=top_mean,
            gap=top_mean - bottom_mean,
            ratio=top_mean / bottom_mean,
        ),
    }


def _build_correlation_stability_case(df: pl.DataFrame) -> dict[str, CorrelationStabilityResult]:
    """Compute ground truth for the BMI correlation-stability eval case.

    Checks whether the BMI-to-disease-progression correlation is consistent
    between younger and older patient subgroups (split at median age).

    Args:
        df (pl.DataFrame): Diabetes dataset.

    Returns:
        dict[str, CorrelationStabilityResult]: Single-entry dict keyed `"correlation-stability"`.
    """
    target = "disease_progression"
    median_age = df.select(pl.median("age")).item()
    young_corr = df.filter(pl.col("age") < median_age).select(pl.corr("bmi", target)).item()
    old_corr = df.filter(pl.col("age") >= median_age).select(pl.corr("bmi", target)).item()
    difference = abs(young_corr - old_corr)
    stability = "stable" if difference < _STABILITY_DIFFERENCE_THRESHOLD else "unstable"
    return {
        "correlation-stability": CorrelationStabilityResult(
            young_correlation=young_corr,
            old_correlation=old_corr,
            difference=difference,
            stability=stability,
        ),
    }


# endregion
