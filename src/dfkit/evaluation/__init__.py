"""Evaluation primitives for assessing LLM agent responses against structured ground truth.

Examples:
    Define two Pydantic schemas — one for correlation analysis and one for feature-importance
    ranking — using the annotated field aliases provided by this package:

    ```python
    from pydantic import BaseModel
    from langchain_core.messages import HumanMessage

    from dfkit.evaluation import (
        EvalCase,
        Confounders,
        Correlation,
        Importance,
        Probability,
        Rank,
        RelationshipStrength,
        SampleCount,
        evaluate_agent,
        summarize_results,
    )

    class CorrelationResult(BaseModel):
        correlation: Correlation
        p_value: Probability
        sample_count: SampleCount
        strength: RelationshipStrength
        confounders: Confounders

    class FeatureImportanceResult(BaseModel):
        feature_name: str
        importance: Importance
        rank: Rank
    ```

    Create eval cases mixing both schemas:

    ```python
    corr_cases = [
        EvalCase(
            case_id="corr-age-income",
            input_messages=[HumanMessage(content="What is the correlation between age and income?")],
            expected_result=CorrelationResult(
                correlation=0.72,
                p_value=0.001,
                sample_count=1500,
                strength="Strong",
                confounders=["education", "occupation"],
            ),
        ),
        EvalCase(
            case_id="corr-height-weight",
            input_messages=[HumanMessage(content="Analyse the correlation between height and weight.")],
            expected_result=CorrelationResult(
                correlation=0.65,
                p_value=0.003,
                sample_count=800,
                strength="Moderate",
                confounders=["age", "sex"],
            ),
        ),
    ]

    importance_cases = [
        EvalCase(
            case_id="fi-age",
            input_messages=[HumanMessage(content="Rank age by feature importance for churn prediction.")],
            expected_result=FeatureImportanceResult(feature_name="age", importance=0.31, rank=1),
        ),
        EvalCase(
            case_id="fi-tenure",
            input_messages=[HumanMessage(content="Rank tenure by feature importance for churn prediction.")],
            expected_result=FeatureImportanceResult(feature_name="tenure", importance=0.24, rank=2),
        ),
        EvalCase(
            case_id="fi-balance",
            input_messages=[
                HumanMessage(content="Rank account balance by feature importance for churn prediction.")
            ],
            expected_result=FeatureImportanceResult(feature_name="balance", importance=0.18, rank=3),
        ),
    ]

    eval_cases = corr_cases + importance_cases
    ```

    Run the evaluation and inspect the summary:

    ```python
    results = evaluate_agent(agent, eval_cases, judge=judge_llm)
    summary = summarize_results(results)

    print(summary.pass_rate)      # e.g. 0.6  — fraction of fully-passing cases
    print(summary.mean_case_score)  # e.g. 0.83 — mean per-case field-match score
    print(summary.field_scores)   # e.g. {"correlation": 1.0, "rank": 0.67, ...}
    ```
"""

from dfkit.evaluation.aggregation import EvalSummary, summarize_results
from dfkit.evaluation.fields import (
    Confounders,
    Correlation,
    Importance,
    Metric,
    NonNegativeMetric,
    Probability,
    Rank,
    RelationshipStrength,
    SampleCount,
)
from dfkit.evaluation.models import EvalCase, EvalResult, FieldComparison
from dfkit.evaluation.runner import evaluate_agent

__all__ = [
    "Confounders",
    "Correlation",
    "EvalCase",
    "EvalResult",
    "EvalSummary",
    "FieldComparison",
    "Importance",
    "Metric",
    "NonNegativeMetric",
    "Probability",
    "Rank",
    "RelationshipStrength",
    "SampleCount",
    "evaluate_agent",
    "summarize_results",
]
