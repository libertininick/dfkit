"""Tests for the dfkit evaluation aggregation module.

Covers EvalSummary.pass_rate behavior and summarize_results aggregation logic
across multiple scenarios: all-passed, mixed, per-field score computation, empty
input, single result, and heterogeneous schemas.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.aggregation import EvalSummary, FieldCounts, summarize_results
from dfkit.evaluation.models import EvalCase, EvalResult, FieldComparison

# Placeholder value used as `extracted` when a FieldComparison should not match
NON_MATCHING_SENTINEL = "__sentinel__"


# region Module-level test schemas


class DummySchema(BaseModel):
    """Minimal Pydantic schema used as the result type in aggregation tests.

    All fields carry defaults so the same class can be used for both
    expected_result and extracted_result without requiring populated values.

    Attributes:
        a (int | None): An integer field, defaults to None.
        b (str | None): A string field, defaults to None.
    """

    a: int | None = None
    b: str | None = None


class OtherSchema(BaseModel):
    """Secondary Pydantic schema with fields disjoint from DummySchema.

    Used to test heterogeneous-schema cases where different EvalResults carry
    different result types. Fields here (`c`, `d`) intentionally do not
    overlap with DummySchema fields (`a`, `b`).

    Attributes:
        c (float | None): A float field, defaults to None.
        d (bool | None): A bool field, defaults to None.
    """

    c: float | None = None
    d: bool | None = None


# endregion

# region Factory helpers


def make_eval_result(
    field_comparisons: Sequence[FieldComparison],
    case_id: str = "case",
) -> EvalResult[HumanMessage, DummySchema]:
    """Build an EvalResult[HumanMessage, DummySchema] with the given field comparisons.

    Args:
        field_comparisons (Sequence[FieldComparison]): Field-level comparison records
            to attach to the result.
        case_id (str): Optional identifier for the embedded EvalCase.

    Returns:
        EvalResult[HumanMessage, DummySchema]: A fully constructed EvalResult
            using DummySchema for both expected and extracted results.
    """
    eval_case: EvalCase[HumanMessage, DummySchema] = EvalCase(
        case_id=case_id,
        input_messages=[HumanMessage(content="What is the answer?")],
        expected_result=DummySchema(a=1, b="x"),
    )
    return EvalResult(
        eval_case=eval_case,
        extracted_result=DummySchema(a=1, b="x"),
        field_comparisons=list(field_comparisons),
        agent_response="some response",
    )


def make_other_eval_result(
    field_comparisons: Sequence[FieldComparison],
    case_id: str = "case",
) -> EvalResult[HumanMessage, OtherSchema]:
    """Build an EvalResult[HumanMessage, OtherSchema] with the given field comparisons.

    Args:
        field_comparisons (Sequence[FieldComparison]): Field-level comparison records
            to attach to the result.
        case_id (str): Optional identifier for the embedded EvalCase.

    Returns:
        EvalResult[HumanMessage, OtherSchema]: A fully constructed EvalResult
            using OtherSchema for both expected and extracted results.
    """
    eval_case: EvalCase[HumanMessage, OtherSchema] = EvalCase(
        case_id=case_id,
        input_messages=[HumanMessage(content="What is the answer?")],
        expected_result=OtherSchema(c=1.0, d=True),
    )
    return EvalResult(
        eval_case=eval_case,
        extracted_result=OtherSchema(c=1.0, d=True),
        field_comparisons=list(field_comparisons),
        agent_response="some response",
    )


def make_field_comparison(
    field_name: str,
    *,
    matched: bool,
    expected: Any = "expected",
    strategy: str = "exact_match",
) -> FieldComparison:
    """Build a minimal FieldComparison with a given name and match status.

    Args:
        field_name (str): Name to assign to the field comparison.
        matched (bool): Whether the comparison should be considered matching.
        expected (Any): Expected value to record on the comparison.
        strategy (str): Comparison strategy label to record.

    Returns:
        FieldComparison: A FieldComparison instance with the given strategy.
    """
    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=expected if matched else NON_MATCHING_SENTINEL,
        matched=matched,
        strategy=strategy,
    )


# endregion

# region Parametrize helpers


class SummaryExpected(NamedTuple):
    """Expected field values for parametrized summarize_results tests.

    Field order mirrors `EvalSummary`: stored fields first (`total_cases`,
    `passed_cases`, `total_score_sum`), followed by the computed properties in
    declaration order (`total_matched_fields`, `total_field_comparisons`,
    `field_scores`, `pass_rate`, `mean_case_score`, `micro_field_score`).

    Attributes:
        total_cases (int): Expected total_cases on the summary.
        passed_cases (int): Expected passed_cases on the summary.
        total_score_sum (float): Expected total_score_sum on the summary.
        total_matched_fields (int): Expected total_matched_fields on the summary.
        total_field_comparisons (int): Expected total_field_comparisons on the summary.
        field_scores (dict[str, float]): Expected field_scores on the summary.
        pass_rate (float): Expected pass_rate on the summary.
        mean_case_score (float): Expected mean_case_score on the summary.
        micro_field_score (float): Expected micro_field_score on the summary.
    """

    total_cases: int
    passed_cases: int
    total_score_sum: float
    total_matched_fields: int
    total_field_comparisons: int
    field_scores: dict[str, float]
    pass_rate: float
    mean_case_score: float
    micro_field_score: float


# endregion


class TestEvalSummary:
    """Tests for the EvalSummary Pydantic model and its computed properties."""

    def test_construction_stores_all_fields(self) -> None:
        """EvalSummary should store all provided field values on construction."""
        # Arrange / Act
        summary = EvalSummary(
            total_cases=10,
            passed_cases=7,
            total_score_sum=8.0,
            field_counts={"a": FieldCounts(matched=9, total=10), "b": FieldCounts(matched=7, total=10)},
        )

        # Assert
        with check:
            assert summary.total_cases == 10
        with check:
            assert summary.passed_cases == 7
        with check:
            assert summary.total_score_sum == pytest.approx(8.0)
        with check:
            assert summary.total_matched_fields == 16
        with check:
            assert summary.total_field_comparisons == 20
        with check:
            assert summary.field_scores == {"a": pytest.approx(0.9), "b": pytest.approx(0.7)}

    def test_pass_rate_returns_zero_when_total_cases_is_zero(self) -> None:
        """EvalSummary.pass_rate should return 0.0 when total_cases is 0."""
        # Arrange
        summary = EvalSummary(
            total_cases=0,
            passed_cases=0,
            total_score_sum=0.0,
            field_counts={},
        )

        # Act / Assert
        assert summary.pass_rate == pytest.approx(0.0)

    def test_pass_rate_computes_fraction_for_nonzero_total_cases(self) -> None:
        """EvalSummary.pass_rate should compute passed_cases / total_cases correctly."""
        # Arrange
        summary = EvalSummary(
            total_cases=8,
            passed_cases=6,
            total_score_sum=6.0,
            field_counts={},
        )

        # Act / Assert
        assert summary.pass_rate == pytest.approx(6 / 8)

    def test_pass_rate_is_one_when_all_cases_passed(self) -> None:
        """EvalSummary.pass_rate should be 1.0 when passed_cases equals total_cases."""
        # Arrange
        summary = EvalSummary(
            total_cases=5,
            passed_cases=5,
            total_score_sum=5.0,
            field_counts={"a": FieldCounts(matched=5, total=5)},
        )

        # Act / Assert
        assert summary.pass_rate == pytest.approx(1.0)

    def test_model_dump_and_validate_round_trip(self) -> None:
        """EvalSummary should survive a model_dump / model_validate round-trip unchanged."""
        # Arrange
        original = EvalSummary(
            total_cases=3,
            passed_cases=2,
            total_score_sum=2.0,
            field_counts={"x": FieldCounts(matched=3, total=3), "y": FieldCounts(matched=1, total=3)},
        )

        # Act
        raw = original.model_dump()
        restored = EvalSummary.model_validate(raw)

        # Assert
        with check:
            assert restored.total_cases == original.total_cases
        with check:
            assert restored.passed_cases == original.passed_cases
        with check:
            assert restored.total_score_sum == pytest.approx(original.total_score_sum)
        with check:
            assert restored.total_matched_fields == original.total_matched_fields
        with check:
            assert restored.total_field_comparisons == original.total_field_comparisons
        with check:
            assert restored.field_scores == pytest.approx(original.field_scores)


class TestSummarizeResults:
    """Tests for the summarize_results aggregation function."""

    @pytest.mark.parametrize(
        ("results", "expected"),
        [
            pytest.param(
                [],
                SummaryExpected(
                    total_cases=0,
                    passed_cases=0,
                    mean_case_score=0.0,
                    pass_rate=0.0,
                    field_scores={},
                    micro_field_score=0.0,
                    total_score_sum=0.0,
                    total_matched_fields=0,
                    total_field_comparisons=0,
                ),
                id="empty",
            ),
            pytest.param(
                [
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=True)],
                        case_id="c1",
                    ),
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=True)],
                        case_id="c2",
                    ),
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=True)],
                        case_id="c3",
                    ),
                ],
                SummaryExpected(
                    total_cases=3,
                    passed_cases=3,
                    mean_case_score=1.0,
                    pass_rate=1.0,
                    field_scores={"a": 1.0, "b": 1.0},
                    micro_field_score=1.0,
                    # 3 cases x score 1.0 each
                    total_score_sum=3.0,
                    # 3 cases x 2 fields each, all matched
                    total_matched_fields=6,
                    total_field_comparisons=6,
                ),
                id="all-passed",
            ),
            pytest.param(
                [
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=True)],
                        case_id="pass",
                    ),
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=False)],
                        case_id="partial",
                    ),
                    make_eval_result(
                        [make_field_comparison("a", matched=False), make_field_comparison("b", matched=False)],
                        case_id="fail",
                    ),
                ],
                SummaryExpected(
                    total_cases=3,
                    passed_cases=1,
                    mean_case_score=(1.0 + 0.5 + 0.0) / 3,
                    pass_rate=1 / 3,
                    field_scores={"a": 2 / 3, "b": 1 / 3},
                    micro_field_score=0.5,
                    # pass=1.0, partial=0.5, fail=0.0
                    total_score_sum=1.5,
                    # a: pass+partial matched (2), b: pass matched (1)
                    total_matched_fields=3,
                    # 3 cases x 2 fields each
                    total_field_comparisons=6,
                ),
                id="mixed",
            ),
            pytest.param(
                [
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=False)],
                        case_id="only",
                    ),
                ],
                SummaryExpected(
                    total_cases=1,
                    passed_cases=0,
                    mean_case_score=0.5,
                    pass_rate=0.0,
                    field_scores={"a": 1.0, "b": 0.0},
                    micro_field_score=0.5,
                    # 1 matched out of 2 fields -> score 0.5
                    total_score_sum=0.5,
                    total_matched_fields=1,
                    total_field_comparisons=2,
                ),
                id="single-partial",
            ),
            pytest.param(
                [
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=False)],
                        case_id="partial-1",
                    ),
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=False)],
                        case_id="partial-2",
                    ),
                ],
                SummaryExpected(
                    total_cases=2,
                    passed_cases=0,
                    mean_case_score=0.5,
                    pass_rate=0.0,
                    field_scores={"a": 1.0, "b": 0.0},
                    micro_field_score=0.5,
                    # 2 cases x score 0.5 each
                    total_score_sum=1.0,
                    # 2 cases x 1 matched field each
                    total_matched_fields=2,
                    # 2 cases x 2 fields each
                    total_field_comparisons=4,
                ),
                id="all-partial",
            ),
            pytest.param(
                [
                    make_eval_result(
                        [
                            make_field_comparison("shared", matched=True),
                            make_field_comparison("only_first", matched=True),
                        ],
                        case_id="a",
                    ),
                    make_eval_result(
                        [
                            make_field_comparison("shared", matched=False),
                            make_field_comparison("only_second", matched=False),
                        ],
                        case_id="b",
                    ),
                ],
                SummaryExpected(
                    total_cases=2,
                    passed_cases=1,
                    mean_case_score=0.5,
                    pass_rate=0.5,
                    field_scores={"shared": 0.5, "only_first": 1.0, "only_second": 0.0},
                    micro_field_score=0.5,
                    # case-a: 1.0 (both matched), case-b: 0.0 (none matched)
                    total_score_sum=1.0,
                    # case-a: shared+only_first both matched (2), case-b: none (0)
                    total_matched_fields=2,
                    # 2 cases x 2 fields each
                    total_field_comparisons=4,
                ),
                id="partially-overlapping-fields",
            ),
            pytest.param(
                [
                    make_eval_result([], case_id="empty-a"),
                    make_eval_result([], case_id="empty-b"),
                ],
                SummaryExpected(
                    total_cases=2,
                    passed_cases=2,
                    mean_case_score=1.0,
                    pass_rate=1.0,
                    field_scores={},
                    # No field comparisons to weight, so micro_field_score is 0.0
                    # even though mean_case_score is 1.0 (vacuous passes).
                    micro_field_score=0.0,
                    # 2 empty cases each score 1.0 vacuously
                    total_score_sum=2.0,
                    # no field comparisons at all
                    total_matched_fields=0,
                    total_field_comparisons=0,
                ),
                id="all-empty-comparisons",
            ),
            pytest.param(
                [
                    make_eval_result(
                        [make_field_comparison("f", matched=True)],
                        case_id="one-field",
                    ),
                    make_eval_result(
                        [make_field_comparison(f"f{i}", matched=False) for i in range(10)],
                        case_id="ten-fields",
                    ),
                ],
                SummaryExpected(
                    total_cases=2,
                    passed_cases=1,
                    mean_case_score=0.5,
                    pass_rate=0.5,
                    field_scores={
                        "f": 1.0,
                        "f0": 0.0,
                        "f1": 0.0,
                        "f2": 0.0,
                        "f3": 0.0,
                        "f4": 0.0,
                        "f5": 0.0,
                        "f6": 0.0,
                        "f7": 0.0,
                        "f8": 0.0,
                        "f9": 0.0,
                    },
                    # mean_case_score diverges from micro_field_score when case sizes differ:
                    # mean weights each case equally (0.5), micro weights each field comparison
                    # equally (1 match out of 11 comparisons).
                    micro_field_score=1 / 11,
                    # case one-field: score 1.0, case ten-fields: score 0.0
                    total_score_sum=1.0,
                    # only "f" matched; f0..f9 all unmatched
                    total_matched_fields=1,
                    # 1 field + 10 fields
                    total_field_comparisons=11,
                ),
                id="micro-macro-divergence",
            ),
            pytest.param(
                [
                    # Empty-field case: no comparisons, passes vacuously with score 1.0.
                    make_eval_result([], case_id="vacuous"),
                    # Partial case: "a" matched, "b" unmatched -> score 0.5.
                    make_eval_result(
                        [make_field_comparison("a", matched=True), make_field_comparison("b", matched=False)],
                        case_id="partial",
                    ),
                ],
                SummaryExpected(
                    total_cases=2,
                    passed_cases=1,
                    # mean_case_score is macro: (1.0 + 0.5) / 2 = 0.75
                    mean_case_score=0.75,
                    pass_rate=0.5,
                    field_scores={"a": 1.0, "b": 0.0},
                    # micro_field_score ignores the empty case: 1 matched / 2 total = 0.5
                    # This intentionally diverges from mean_case_score (0.75 vs 0.5).
                    micro_field_score=0.5,
                    # vacuous case contributes 1.0; partial contributes 0.5
                    total_score_sum=1.5,
                    # only "a" from the partial case matched
                    total_matched_fields=1,
                    # empty case contributes 0 comparisons; partial contributes 2
                    total_field_comparisons=2,
                ),
                id="one-empty-one-partial",
            ),
        ],
    )
    def test_summarize_results_counts_and_scores(
        self,
        results: list[EvalResult[HumanMessage, DummySchema]],
        expected: SummaryExpected,
    ) -> None:
        """summarize_results should compute correct counts and scores for each scenario.

        Args:
            results (list[EvalResult[HumanMessage, DummySchema]]): Pre-built eval
                results for the scenario under test.
            expected (SummaryExpected): Expected aggregate values for the scenario.
        """
        # Act
        summary = summarize_results(results)

        # Assert
        with check:
            assert summary.total_cases == expected.total_cases
        with check:
            assert summary.passed_cases == expected.passed_cases
        with check:
            assert summary.mean_case_score == pytest.approx(expected.mean_case_score)
        with check:
            assert summary.pass_rate == pytest.approx(expected.pass_rate)
        with check:
            assert summary.field_scores == pytest.approx(expected.field_scores)
        with check:
            assert summary.micro_field_score == pytest.approx(expected.micro_field_score)
        with check:
            assert summary.total_score_sum == pytest.approx(expected.total_score_sum)
        with check:
            assert summary.total_matched_fields == expected.total_matched_fields
        with check:
            assert summary.total_field_comparisons == expected.total_field_comparisons

    def test_summarize_heterogeneous_schemas_disjoint_types_field_scores_per_schema(self) -> None:
        """summarize_results computes per-field scores correctly when schemas differ between results.

        One result uses DummySchema (fields "a", "b") and the other uses OtherSchema
        (fields "c", "d").  Because each field appears in exactly one result, each
        field score is computed over only that one case.
        """
        # Arrange
        result_dummy = make_eval_result(
            [make_field_comparison("a", matched=True), make_field_comparison("b", matched=False)],
            case_id="dummy",
        )
        result_other = make_other_eval_result(
            [make_field_comparison("c", matched=True), make_field_comparison("d", matched=True)],
            case_id="other",
        )

        # Act
        summary = summarize_results([result_dummy, result_other])

        # Assert — each field appears in exactly one case, so rate == match outcome
        with check:
            assert summary.field_scores == pytest.approx({"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0})
        with check:
            assert summary.total_cases == 2

    def test_summarize_double_counts_duplicate_field_names_within_a_case(self) -> None:
        """summarize_results treats duplicate field names in one case as separate comparisons.

        This test locks in the CURRENT behavior: when the same field name appears
        more than once in a single case's field_comparisons list, each occurrence
        is counted independently.  There is no deduplication.  The per-field score
        for that name is computed over all occurrences, not just the first.

        If intentional deduplication is ever introduced, this test should be updated
        to reflect the new semantics.
        """
        # Arrange — one case with two "a" entries (one matched, one not) plus a matched "b"
        eval_result = make_eval_result(
            [
                make_field_comparison("a", matched=True),
                make_field_comparison("a", matched=False),
                make_field_comparison("b", matched=True),
            ],
            case_id="dup-fields",
        )

        # Act
        summary = summarize_results([eval_result])

        # Assert — duplicates double-count: "a" appears twice, 1 matched out of 2
        with check:
            assert summary.field_scores == pytest.approx({"a": 0.5, "b": 1.0})
        with check:
            assert summary.total_matched_fields == 2
        with check:
            assert summary.total_field_comparisons == 3
