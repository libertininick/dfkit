"""Aggregation utilities for the dfkit evaluation pipeline.

This module provides `EvalSummary`, a Pydantic model that captures aggregate
statistics over a collection of `EvalResult` instances, and `summarize_results`,
the function that computes it.

Example:
    ``python
    results = evaluate_agent(agent, eval_cases, judge=judge_llm)
    summary = summarize_results(results)
    print(summary.pass_rate, summary.field_scores)
    ``
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from dfkit.evaluation.models import EvalResult

__all__ = [
    "EvalSummary",
    "FieldCounts",
    "summarize_results",
]


@dataclass(frozen=True)
class FieldCounts:
    """Per-field numerator and denominator for match-rate computation.

    Attributes:
        matched (int): Number of cases in which this field matched.
        total (int): Number of cases in which this field was compared.
    """

    matched: int
    total: int


class EvalSummary(BaseModel):
    """Aggregate statistics over a collection of evaluation results.

    Two score aggregations are exposed as computed properties:

    - `mean_case_score`: macro-average — each case contributes equally regardless
      of how many fields it contains.  Cases with no field comparisons contribute
      `1.0`, matching `EvalResult.score` semantics.
    - `micro_field_score`: micro-average — each *field comparison* contributes
      equally, so a 10-field case carries 10x the weight of a 1-field case.
      Cases with no field comparisons contribute nothing to the numerator or
      denominator.

    Attributes:
        total_cases (int): Total number of evaluation cases.
        passed_cases (int): Number of cases where all fields matched.
        total_score_sum (float): Sum of per-case scores across all cases.
        field_counts (dict[str, FieldCounts]): Per-field matched and total
            comparison counts.  Keys are field names; values hold the raw
            numerator (`matched`) and denominator (`total`) used to compute
            per-field match rates.  `total_matched_fields`,
            `total_field_comparisons`, and `field_scores` are all derived
            from this mapping.
    """

    total_cases: int = Field(description="Total number of evaluation cases.")
    passed_cases: int = Field(description="Number of cases where all fields matched.")
    total_score_sum: float = Field(description="Sum of per-case scores across all cases.")
    field_counts: dict[str, FieldCounts] = Field(
        description="Per-field matched and total comparison counts (field_name -> FieldCounts).",
    )

    @property
    def total_matched_fields(self) -> int:
        """Total field comparisons that matched across all cases.

        Returns:
            int: Sum of `matched` across all entries in `field_counts`.
        """
        return sum(fc.matched for fc in self.field_counts.values())

    @property
    def total_field_comparisons(self) -> int:
        """Total field comparisons across all cases.

        Returns:
            int: Sum of `total` across all entries in `field_counts`.
        """
        return sum(fc.total for fc in self.field_counts.values())

    @property
    def field_scores(self) -> dict[str, float]:
        """Per-field match rate derived from `field_counts`.

        Keys are field names; values are the fraction of cases that include the
        field in which that field matched.  When a field appears in only a subset
        of cases (heterogeneous schemas), the rate is computed over the cases that
        include it.

        Returns:
            dict[str, float]: Mapping of field name to match rate in `[0.0, 1.0]`.
        """
        return {field_name: fc.matched / fc.total for field_name, fc in self.field_counts.items()}

    @property
    def pass_rate(self) -> float:
        """Fraction of cases in which all fields matched.

        Returns:
            float: `passed_cases / total_cases`, or `0.0` when there are no cases.
        """
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def mean_case_score(self) -> float:
        """Macro-average score across all cases.

        Each case contributes equally to this average regardless of how many
        field comparisons it contains.  Cases with no field comparisons
        contribute `1.0` (vacuous pass), matching `EvalResult.score`
        semantics.

        Returns:
            float: `total_score_sum / total_cases`, or `0.0` when there are no
                cases.
        """
        if self.total_cases == 0:
            return 0.0
        return self.total_score_sum / self.total_cases

    @property
    def micro_field_score(self) -> float:
        """Micro-average score weighted by field comparison count.

        Each field comparison contributes equally, so a case with 10 fields has
        10x the influence of a case with 1 field.  Cases with no field
        comparisons contribute nothing to the numerator or denominator.

        Returns:
            float: `total_matched_fields / total_field_comparisons`, or `0.0`
                when there are no field comparisons.
        """
        if self.total_field_comparisons == 0:
            return 0.0
        return self.total_matched_fields / self.total_field_comparisons


def summarize_results(results: Sequence[EvalResult[Any, Any]]) -> EvalSummary:
    """Compute aggregate statistics over a sequence of evaluation results.

    Args:
        results (Sequence[EvalResult[Any, Any]]): Evaluation results to summarize.
            May be empty, in which case a zeroed summary is returned.

    Returns:
        EvalSummary: Aggregated statistics over all provided results.

    Examples:
        ```python
        results = evaluate_agent(agent, eval_cases, judge=judge_llm)
        summary = summarize_results(results)
        print(f"pass rate: {summary.pass_rate:.1%}")
        print(f"mean case score: {summary.mean_case_score:.2f}")
        print(f"per-field scores: {summary.field_scores}")
        ```
    """
    passed_cases = 0
    total_score_sum = 0.0
    field_match_counts: Counter[str] = Counter()
    field_total_counts: Counter[str] = Counter()

    for result in results:
        if result.passed:
            passed_cases += 1
        total_score_sum += result.score
        for field_comparison in result.field_comparisons:
            field_total_counts[field_comparison.field_name] += 1
            if field_comparison.matched:
                field_match_counts[field_comparison.field_name] += 1

    field_counts = {
        field_name: FieldCounts(matched=field_match_counts[field_name], total=count)
        for field_name, count in field_total_counts.items()
    }
    return EvalSummary(
        total_cases=len(results),
        passed_cases=passed_cases,
        total_score_sum=total_score_sum,
        field_counts=field_counts,
    )
