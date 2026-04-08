"""Tests for the field-level comparison engine in dfkit.evaluation.comparison."""

from typing import Literal

import pytest
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.comparison import (
    ATOL,
    compare_dict,
    compare_exact,
    compare_fields,
    compare_numeric,
    compare_numeric_sequence,
)
from dfkit.evaluation.fields import (
    Confounders,
    Correlation,
    RelationshipStrength,
)

# region Module-level test models


class NumericModel(BaseModel):
    """Pydantic model with numeric fields used in compare_fields tests.

    Attributes:
        score (float): A float score.
        count (int): An integer count.
    """

    score: float
    count: int


class MixedModel(BaseModel):
    """Pydantic model with diverse field types for dispatcher tests.

    Attributes:
        score (float): A float score.
        label (str): A text label.
        flag (bool): A boolean flag.
        tags (list[str]): A list of string tags.
        rank (int): An integer rank.
    """

    score: float
    label: str
    flag: bool
    tags: list[str]
    rank: int


class LiteralModel(BaseModel):
    """Pydantic model with a Literal field for exact-match strategy tests.

    Attributes:
        category (Literal["A", "B", "C"]): A categorical choice.
    """

    category: Literal["A", "B", "C"]


class OptionalFieldModel(BaseModel):
    """Pydantic model with Optional fields for None-handling tests.

    Attributes:
        score (float | None): An optional float score.
        tags (list[str] | None): An optional list of tags.
    """

    score: float | None = None
    tags: list[str] | None = None


class DictFieldModel(BaseModel):
    """Pydantic model with a dict field for dict-dispatch tests.

    Attributes:
        metrics (dict[str, float]): A mapping from metric name to numeric value.
    """

    metrics: dict[str, float]


class AnnotatedFieldModel(BaseModel):
    """Pydantic model using annotated field aliases from dfkit.evaluation.fields.

    Attributes:
        correlation (Correlation | None): Annotated float field wrapping Correlation.
        strength (RelationshipStrength | None): Annotated Literal field for relationship strength.
        confounders (Confounders | None): Annotated list[str] field for confounders.
    """

    correlation: Correlation | None = None
    strength: RelationshipStrength | None = None
    confounders: Confounders | None = None


class SimpleModel(BaseModel):
    """Simple model with score and label fields for parametrized tests.

    Attributes:
        score (float): A numeric score.
        label (str): A text label.
    """

    score: float
    label: str


class NumericSequenceModel(BaseModel):
    """Model with a list[float] field for numeric sequence dispatch tests.

    Attributes:
        weights (list[float]): A list of numeric weights.
    """

    weights: list[float]


class NestedDictFieldModel(BaseModel):
    """Pydantic model with a nested dict field for nested dict-dispatch tests.

    Attributes:
        nested_metrics (dict[str, dict[str, float]]): A mapping from group name to metric dict.
    """

    nested_metrics: dict[str, dict[str, float]]


class IntSequenceModel(BaseModel):
    """Model with a list[int] field for integer sequence dispatch tests.

    Attributes:
        ranks (list[int]): A list of integer ranks.
    """

    ranks: list[int]


# endregion


class TestCompareNumeric:
    """Tests for compare_numeric — numeric tolerance comparison strategy."""

    def test_exact_match_returns_matched(self) -> None:
        """Identical float values should produce a matched result."""
        result = compare_numeric("score", 0.5, 0.5)

        assert result.matched is True

    def test_within_atol_returns_matched(self) -> None:
        """Values within atol of each other should produce a matched result."""
        result = compare_numeric("score", 0.5, 0.5 + ATOL / 2)

        assert result.matched is True

    def test_outside_tolerances_returns_not_matched(self) -> None:
        """Values far outside rtol and atol should produce a not-matched result."""
        result = compare_numeric("score", 0.50, 0.60)

        assert result.matched is False

    def test_both_none_returns_matched(self) -> None:
        """Two None values should produce a matched result."""
        result = compare_numeric("score", None, None)

        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted value should produce a not-matched result."""
        result = compare_numeric("score", None, 0.5)

        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected value with None extracted should produce a not-matched result."""
        result = compare_numeric("score", 0.5, None)

        assert result.matched is False

    def test_strategy_is_numeric_tolerance(self) -> None:
        """Strategy field should always be 'numeric_tolerance'."""
        result = compare_numeric("score", 0.1, 0.1)

        assert result.strategy == "numeric_tolerance"

    def test_field_name_is_preserved(self) -> None:
        """The field_name argument should appear in the returned FieldComparison."""
        result = compare_numeric("my_field", 0.3, 0.3)

        assert result.field_name == "my_field"

    def test_custom_atol_widens_match(self) -> None:
        """A wider atol should cause a value outside the default tolerance to match."""
        result = compare_numeric("score", 0.5, 0.6, atol=0.15)

        assert result.matched is True

    def test_custom_rtol_widens_match(self) -> None:
        """A wider rtol should cause a proportionally close value to match."""
        result = compare_numeric("score", 1.0, 1.1, rtol=0.15)

        assert result.matched is True

    @pytest.mark.parametrize(
        ("expected", "extracted", "is_expected_match"),
        [
            pytest.param(1.0, 1.0, True, id="identical"),
            pytest.param(0.0, 1e-9, True, id="within_atol"),
            pytest.param(0.0, 0.5, False, id="outside_tolerances"),
            pytest.param(-0.5, -0.5, True, id="negative_identical"),
            pytest.param(-1.5, -2.5, False, id="negative_mismatch"),
            pytest.param(float("nan"), float("nan"), False, id="both_nan"),
            pytest.param(float("inf"), float("inf"), True, id="both_inf"),
            pytest.param(1e15, 1e15, True, id="very_large_identical"),
            pytest.param(1e15, 1e15 + 1.0, True, id="very_large_within_rtol"),
            pytest.param(float("nan"), 0.5, False, id="nan_expected_only"),
            pytest.param(0.5, float("nan"), False, id="nan_extracted_only"),
            pytest.param(float("inf"), float("-inf"), False, id="pos_inf_vs_neg_inf"),
            pytest.param(float("-inf"), float("-inf"), True, id="both_neg_inf"),
        ],
    )
    def test_various_numeric_combinations(
        self,
        *,
        expected: float,
        extracted: float,
        is_expected_match: bool,
    ) -> None:
        """compare_numeric should correctly classify a variety of numeric pairs.

        Args:
            expected (float): Expected numeric value.
            extracted (float): Extracted numeric value.
            is_expected_match (bool): Whether the comparison should be matched.
        """
        result = compare_numeric("x", expected, extracted)

        assert result.matched is is_expected_match


class TestCompareNumericSequence:
    """Tests for compare_numeric_sequence — element-wise float sequence comparison."""

    def test_identical_sequences_returns_matched(self) -> None:
        """Identical float sequences should produce a matched result."""
        result = compare_numeric_sequence("weights", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])

        assert result.matched is True

    def test_within_atol_returns_matched(self) -> None:
        """Sequences with all elements within atol should produce a matched result."""
        result = compare_numeric_sequence("weights", [0.1, 0.2], [0.1, 0.2 + ATOL / 2])

        assert result.matched is True

    def test_one_element_outside_tolerance_returns_not_matched(self) -> None:
        """A sequence with one element far outside tolerance should produce a not-matched result."""
        result = compare_numeric_sequence("weights", [0.1, 0.2, 0.3], [0.1, 0.2, 0.9])

        assert result.matched is False

    def test_length_mismatch_returns_not_matched(self) -> None:
        """Sequences of different lengths should produce a not-matched result."""
        result = compare_numeric_sequence("weights", [0.1, 0.2], [0.1, 0.2, 0.3])

        with check:
            assert result.matched is False
        with check:
            assert result.details["expected_length"] == 2
        with check:
            assert result.details["extracted_length"] == 3

    def test_both_none_returns_matched(self) -> None:
        """Two None sequences should produce a matched result."""
        result = compare_numeric_sequence("weights", None, None)

        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted sequence should produce a not-matched result."""
        result = compare_numeric_sequence("weights", None, [0.1])

        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected sequence with None extracted should produce a not-matched result."""
        result = compare_numeric_sequence("weights", [0.1], None)

        assert result.matched is False

    def test_strategy_is_numeric_sequence_tolerance(self) -> None:
        """Strategy field should always be 'numeric_sequence_tolerance'."""
        result = compare_numeric_sequence("weights", [0.1], [0.1])

        assert result.strategy == "numeric_sequence_tolerance"

    def test_details_contains_element_results_and_length(self) -> None:
        """Details should include 'element_results' and 'length' for same-length sequences."""
        result = compare_numeric_sequence("weights", [0.1, 0.2], [0.1, 0.9])

        with check:
            assert "element_results" in result.details
        with check:
            assert "length" in result.details
        with check:
            assert result.details["element_results"] == [True, False]
        with check:
            assert result.details["length"] == 2

    def test_empty_sequences_returns_matched(self) -> None:
        """Two empty sequences should produce a matched result."""
        result = compare_numeric_sequence("weights", [], [])

        assert result.matched is True

    def test_mixed_int_float_sequence_returns_matched(self) -> None:
        """A sequence mixing int and float elements should produce a matched result."""
        result = compare_numeric_sequence("values", [1, 0.5, 2.0], [1.0, 0.5, 2])

        with check:
            assert result.matched is True
        with check:
            assert result.strategy == "numeric_sequence_tolerance"

    def test_custom_atol_widens_element_match(self) -> None:
        """A wider atol should cause an element outside default tolerance to match."""
        result = compare_numeric_sequence("weights", [0.1, 0.5], [0.1, 0.6], atol=0.15)

        assert result.matched is True

    def test_single_element_mismatch_returns_not_matched(self) -> None:
        """Single-element sequences with differing values should produce a not-matched result."""
        result = compare_numeric_sequence("weights", [0.1], [0.9])

        with check:
            assert result.matched is False
        with check:
            assert result.details["element_results"] == [False]
        with check:
            assert result.details["length"] == 1

    def test_nan_elements_returns_not_matched(self) -> None:
        """Sequences containing NaN elements should produce a not-matched result."""
        result = compare_numeric_sequence("weights", [0.1, float("nan")], [0.1, float("nan")])

        assert result.matched is False

    def test_custom_rtol_widens_element_match(self) -> None:
        """A wider rtol should cause an element outside default tolerance to match."""
        result = compare_numeric_sequence("weights", [1.0, 2.0], [1.1, 2.0], rtol=0.15)

        assert result.matched is True


class TestCompareExact:
    """Tests for compare_exact — exact equality comparison strategy."""

    def test_equal_strings_returns_matched(self) -> None:
        """Identical string values should produce a matched result."""
        result = compare_exact("label", "hello", "hello")

        assert result.matched is True

    def test_unequal_strings_returns_not_matched(self) -> None:
        """Different string values should produce a not-matched result."""
        result = compare_exact("label", "hello", "world")

        assert result.matched is False

    def test_both_none_returns_matched(self) -> None:
        """Two None values should produce a matched result."""
        result = compare_exact("label", None, None)

        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted value should produce a not-matched result."""
        result = compare_exact("label", None, "value")

        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected value with None extracted should produce a not-matched result."""
        result = compare_exact("label", "value", None)

        assert result.matched is False

    def test_strategy_is_exact_match(self) -> None:
        """Strategy field should always be 'exact_match'."""
        result = compare_exact("label", "a", "a")

        assert result.strategy == "exact_match"

    def test_field_name_is_preserved(self) -> None:
        """The field_name argument should appear in the returned FieldComparison."""
        result = compare_exact("my_field", "x", "x")

        assert result.field_name == "my_field"

    def test_empty_string_returns_matched(self) -> None:
        """Two empty string values should produce a matched result."""
        result = compare_exact("label", "", "")

        assert result.matched is True

    @pytest.mark.parametrize(
        ("expected", "extracted", "is_expected_match"),
        [
            pytest.param(42, 42, True, id="equal_ints"),
            pytest.param(42, 43, False, id="unequal_ints"),
            pytest.param(True, True, True, id="equal_bools"),
            pytest.param(True, False, False, id="unequal_bools"),
            pytest.param("abc", "abc", True, id="equal_strings"),
            pytest.param("abc", "ABC", False, id="case_different_strings"),
            pytest.param(["a", "b"], ["a", "b"], True, id="equal_lists"),
            pytest.param(["a", "b"], ["a", "c"], False, id="unequal_lists"),
            pytest.param("42", 42, False, id="mixed_types"),
        ],
    )
    def test_various_scalar_types(
        self,
        *,
        expected: object,
        extracted: object,
        is_expected_match: bool,
    ) -> None:
        """compare_exact should handle all scalar types and list values correctly.

        Args:
            expected (object): Expected value.
            extracted (object): Extracted value.
            is_expected_match (bool): Whether the comparison should be matched.
        """
        result = compare_exact("field", expected, extracted)

        assert result.matched is is_expected_match


class TestCompareDict:
    """Tests for compare_dict — key-by-key dictionary comparison strategy."""

    def test_matching_dicts_returns_matched(self) -> None:
        """Identical dicts should produce a matched result."""
        result = compare_dict("metrics", {"a": "x", "b": "y"}, {"a": "x", "b": "y"})

        assert result.matched is True

    def test_mismatched_values_returns_not_matched(self) -> None:
        """Dicts with differing values for a shared key should produce a not-matched result."""
        result = compare_dict("metrics", {"a": "x"}, {"a": "z"})

        assert result.matched is False

    def test_missing_key_in_extracted_returns_not_matched(self) -> None:
        """A key present in expected but absent in extracted should produce a not-matched result."""
        result = compare_dict("metrics", {"a": "x", "b": "y"}, {"a": "x"})

        with check:
            assert result.matched is False
        with check:
            assert result.details["missing_keys"] == ["b"]

    def test_extra_key_in_extracted_returns_not_matched(self) -> None:
        """A key present in extracted but absent in expected should produce a not-matched result."""
        result = compare_dict("metrics", {"a": "x"}, {"a": "x", "b": "y"})

        with check:
            assert result.matched is False
        with check:
            assert result.details["extra_keys"] == ["b"]

    def test_both_none_returns_matched(self) -> None:
        """Two None dicts should produce a matched result."""
        result = compare_dict("metrics", None, None)

        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_both_empty_dicts_returns_matched(self) -> None:
        """Two empty dicts should produce a matched result."""
        result = compare_dict("metrics", {}, {})

        assert result.matched is True

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted dict should produce a not-matched result."""
        result = compare_dict("metrics", None, {"a": "x"})

        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected dict with None extracted should produce a not-matched result."""
        result = compare_dict("metrics", {"a": "x"}, None)

        assert result.matched is False

    def test_strategy_is_dict_comparison(self) -> None:
        """Strategy field should always be 'dict_comparison'."""
        result = compare_dict("metrics", {"a": "1"}, {"a": "1"})

        assert result.strategy == "dict_comparison"

    def test_details_contains_results_per_key_missing_and_extra(self) -> None:
        """Details should include 'results_per_key', 'missing_keys', and 'extra_keys'."""
        result = compare_dict("metrics", {"a": "x", "b": "y"}, {"a": "x", "c": "z"})

        with check:
            assert "results_per_key" in result.details
        with check:
            assert "missing_keys" in result.details
        with check:
            assert "extra_keys" in result.details
        with check:
            assert result.details["missing_keys"] == ["b"]
        with check:
            assert result.details["extra_keys"] == ["c"]

    def test_results_per_key_reflects_per_key_match(self) -> None:
        """results_per_key should record the matched flag for each shared key."""
        result = compare_dict("metrics", {"a": "x", "b": "y"}, {"a": "x", "b": "z"})

        with check:
            assert result.details["results_per_key"]["a"]["matched"] is True
        with check:
            assert result.details["results_per_key"]["b"]["matched"] is False

    def test_nested_float_values_auto_dispatch_to_numeric_tolerance(self) -> None:
        """compare_dict should auto-dispatch float values to numeric strategy."""
        result = compare_dict("metrics", {"score": 0.50}, {"score": 0.50})

        with check:
            assert result.matched is True
        with check:
            assert result.details["results_per_key"]["score"]["matched"] is True

    def test_nested_float_values_outside_tolerance_not_matched(self) -> None:
        """compare_dict with default dispatch should reject float values beyond default tolerance."""
        result = compare_dict("metrics", {"score": 0.50}, {"score": 0.70})

        assert result.matched is False

    def test_nested_dict_values_auto_dispatch_recurse(self) -> None:
        """compare_dict should recurse into nested dicts."""
        result = compare_dict(
            "metrics",
            {"inner": {"key": "val"}},
            {"inner": {"key": "val"}},
        )

        with check:
            assert result.matched is True
        with check:
            assert result.details["results_per_key"]["inner"]["matched"] is True

    def test_nested_dict_mismatch_propagates_to_parent(self) -> None:
        """A mismatch in a nested dict should cause the parent compare_dict to not match."""
        result = compare_dict(
            "metrics",
            {"inner": {"key": "expected"}},
            {"inner": {"key": "different"}},
        )

        assert result.matched is False

    def test_custom_atol_forwarded_to_nested_float(self) -> None:
        """A custom atol should be forwarded to nested float comparisons."""
        result = compare_dict("metrics", {"score": 0.50}, {"score": 0.60}, atol=0.15)

        assert result.matched is True

    def test_custom_rtol_forwarded_to_nested_float(self) -> None:
        """A custom rtol should be forwarded to nested float comparisons."""
        result = compare_dict("metrics", {"score": 1.0}, {"score": 1.1}, rtol=0.15)

        assert result.matched is True

    def test_mixed_value_types_in_single_dict(self) -> None:
        """A dict with string, float, and list values should produce a matched result when all values match."""
        result = compare_dict(
            "metrics",
            {"label": "high", "score": 0.5, "tags": ["a", "b"]},
            {"label": "high", "score": 0.5, "tags": ["a", "b"]},
        )

        assert result.matched is True


class TestCompareFields:
    """Tests for compare_fields — type-driven field comparison dispatcher."""

    def test_float_field_dispatches_to_numeric_strategy(self) -> None:
        """Float fields should be compared with the 'numeric_tolerance' strategy."""
        expected = NumericModel(score=0.5, count=3)
        extracted = NumericModel(score=0.5, count=3)

        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        with check:
            assert score_result.strategy == "numeric_tolerance"
        with check:
            assert score_result.matched is True

    def test_int_field_dispatches_to_numeric_strategy(self) -> None:
        """Integer fields should be compared with the 'numeric_tolerance' strategy."""
        expected = NumericModel(score=0.5, count=3)
        extracted = NumericModel(score=0.5, count=3)

        results = compare_fields(expected, extracted)
        count_result = next(r for r in results if r.field_name == "count")

        with check:
            assert count_result.strategy == "numeric_tolerance"
        with check:
            assert count_result.matched is True

    def test_str_field_dispatches_to_exact_strategy(self) -> None:
        """String fields should be compared with the 'exact_match' strategy."""
        expected = MixedModel(score=0.9, label="medium", flag=False, tags=["x"], rank=3)
        extracted = MixedModel(score=0.9, label="medium", flag=False, tags=["x"], rank=3)

        results = compare_fields(expected, extracted)
        label_result = next(r for r in results if r.field_name == "label")

        with check:
            assert label_result.strategy == "exact_match"
        with check:
            assert label_result.matched is True

    def test_bool_field_dispatches_to_exact_strategy(self) -> None:
        """Boolean fields should be compared with the 'exact_match' strategy."""
        expected = MixedModel(score=0.3, label="low", flag=False, tags=["y", "z"], rank=7)
        extracted = MixedModel(score=0.3, label="low", flag=False, tags=["y", "z"], rank=7)

        results = compare_fields(expected, extracted)
        flag_result = next(r for r in results if r.field_name == "flag")

        with check:
            assert flag_result.strategy == "exact_match"
        with check:
            assert flag_result.matched is True

    def test_literal_field_dispatches_to_exact_strategy(self) -> None:
        """Literal-typed fields should be compared with the 'exact_match' strategy."""
        expected = LiteralModel(category="A")
        extracted = LiteralModel(category="A")

        results = compare_fields(expected, extracted)
        cat_result = next(r for r in results if r.field_name == "category")

        with check:
            assert cat_result.strategy == "exact_match"
        with check:
            assert cat_result.matched is True

    def test_all_fields_present_in_results(self) -> None:
        """compare_fields should return one FieldComparison per field in the model."""
        expected = MixedModel(score=0.5, label="high", flag=True, tags=["a"], rank=1)
        extracted = MixedModel(score=0.5, label="high", flag=True, tags=["a"], rank=1)

        results = compare_fields(expected, extracted)
        field_names = {r.field_name for r in results}

        with check:
            assert len(results) == 5
        with check:
            assert field_names == {"score", "label", "flag", "tags", "rank"}

    def test_optional_float_field_both_none_returns_matched(self) -> None:
        """Optional float fields that are both None should produce a matched result with 'none_comparison' strategy."""
        expected = OptionalFieldModel(score=None, tags=None)
        extracted = OptionalFieldModel(score=None, tags=None)

        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        with check:
            assert score_result.matched is True
        with check:
            assert score_result.strategy == "none_comparison"

    def test_both_none_fields_use_none_comparison_strategy(self) -> None:
        """Fields where both expected and extracted are None should use 'none_comparison' strategy."""
        expected = OptionalFieldModel(score=None, tags=None)
        extracted = OptionalFieldModel(score=None, tags=None)

        results = compare_fields(expected, extracted)
        tags_result = next(r for r in results if r.field_name == "tags")

        with check:
            assert tags_result.matched is True
        with check:
            assert tags_result.strategy == "none_comparison"
        with check:
            assert tags_result.expected is None
        with check:
            assert tags_result.extracted is None

    def test_dict_field_dispatches_to_dict_comparison_strategy(self) -> None:
        """dict[str, float] fields should dispatch to the 'dict_comparison' strategy."""
        expected = DictFieldModel(metrics={"rmse": 0.1, "mae": 0.05})
        extracted = DictFieldModel(metrics={"rmse": 0.1, "mae": 0.05})

        results = compare_fields(expected, extracted)
        metrics_result = next(r for r in results if r.field_name == "metrics")

        with check:
            assert metrics_result.strategy == "dict_comparison"
        with check:
            assert metrics_result.matched is True

    def test_list_float_field_dispatches_to_numeric_sequence_strategy(self) -> None:
        """list[float] fields should dispatch to the 'numeric_sequence_tolerance' strategy."""
        expected = NumericSequenceModel(weights=[0.1, 0.2, 0.3])
        extracted = NumericSequenceModel(weights=[0.1, 0.2, 0.3])

        results = compare_fields(expected, extracted)
        weights_result = next(r for r in results if r.field_name == "weights")

        with check:
            assert weights_result.strategy == "numeric_sequence_tolerance"
        with check:
            assert weights_result.matched is True

    def test_list_int_field_dispatches_to_numeric_sequence_strategy(self) -> None:
        """list[int] fields should dispatch to the 'numeric_sequence_tolerance' strategy."""
        expected = IntSequenceModel(ranks=[1, 2, 3])
        extracted = IntSequenceModel(ranks=[1, 2, 3])

        results = compare_fields(expected, extracted)
        ranks_result = next(r for r in results if r.field_name == "ranks")

        with check:
            assert ranks_result.strategy == "numeric_sequence_tolerance"
        with check:
            assert ranks_result.matched is True

    def test_annotated_correlation_field_dispatches_to_numeric_strategy(self) -> None:
        """Annotated Correlation (float) fields should dispatch to 'numeric_tolerance' strategy."""
        expected = AnnotatedFieldModel(correlation=0.75, strength=None, confounders=None)
        extracted = AnnotatedFieldModel(correlation=0.75, strength=None, confounders=None)

        results = compare_fields(expected, extracted)
        corr_result = next(r for r in results if r.field_name == "correlation")

        with check:
            assert corr_result.strategy == "numeric_tolerance"
        with check:
            assert corr_result.matched is True

    def test_annotated_relationship_strength_field_dispatches_to_exact_strategy(self) -> None:
        """Annotated RelationshipStrength (Literal) fields should dispatch to 'exact_match' strategy."""
        expected = AnnotatedFieldModel(correlation=None, strength="Strong", confounders=None)
        extracted = AnnotatedFieldModel(correlation=None, strength="Strong", confounders=None)

        results = compare_fields(expected, extracted)
        strength_result = next(r for r in results if r.field_name == "strength")

        with check:
            assert strength_result.strategy == "exact_match"
        with check:
            assert strength_result.matched is True

    def test_matched_fields_have_correct_expected_and_extracted_values(self) -> None:
        """Each FieldComparison should carry the correct expected and extracted values."""
        expected = NumericModel(score=0.8, count=5)
        extracted = NumericModel(score=0.8, count=5)

        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")
        count_result = next(r for r in results if r.field_name == "count")

        with check:
            assert score_result.expected == pytest.approx(0.8)
        with check:
            assert score_result.extracted == pytest.approx(0.8)
        with check:
            assert count_result.expected == 5
        with check:
            assert count_result.extracted == 5

    def test_custom_atol_forwarded_to_float_field(self) -> None:
        """A custom atol should be forwarded to numeric field comparisons."""
        expected = NumericModel(score=0.50, count=1)
        extracted = NumericModel(score=0.60, count=1)

        results = compare_fields(expected, extracted, atol=0.15)
        score_result = next(r for r in results if r.field_name == "score")

        assert score_result.matched is True

    def test_custom_rtol_forwarded_to_float_field(self) -> None:
        """A custom rtol should be forwarded to numeric field comparisons."""
        expected = NumericModel(score=1.0, count=1)
        extracted = NumericModel(score=1.1, count=1)

        results = compare_fields(expected, extracted, rtol=0.15)
        score_result = next(r for r in results if r.field_name == "score")

        assert score_result.matched is True

    def test_nested_dict_field_dispatches_to_dict_comparison_and_recurses(self) -> None:
        """A nested dict field should dispatch to 'dict_comparison' and recurse into inner dicts."""
        expected = NestedDictFieldModel(nested_metrics={"group_a": {"rmse": 0.1, "mae": 0.05}})
        extracted = NestedDictFieldModel(nested_metrics={"group_a": {"rmse": 0.1, "mae": 0.05}})

        results = compare_fields(expected, extracted)
        nested_result = next(r for r in results if r.field_name == "nested_metrics")

        with check:
            assert nested_result.strategy == "dict_comparison"
        with check:
            assert nested_result.matched is True
        with check:
            assert nested_result.details["results_per_key"]["group_a"]["matched"] is True

    def test_nested_dict_field_inner_mismatch_propagates(self) -> None:
        """A mismatch in an inner dict value should cause the nested dict field to not match."""
        expected = NestedDictFieldModel(nested_metrics={"group_a": {"rmse": 0.1, "mae": 0.05}})
        extracted = NestedDictFieldModel(nested_metrics={"group_a": {"rmse": 0.1, "mae": 0.90}})

        results = compare_fields(expected, extracted)
        nested_result = next(r for r in results if r.field_name == "nested_metrics")

        with check:
            assert nested_result.matched is False
        with check:
            assert nested_result.details["results_per_key"]["group_a"]["matched"] is False

    def test_optional_float_field_expected_none_extracted_present_returns_not_matched(self) -> None:
        """When expected score is None and extracted is a float, result is not matched."""
        expected = OptionalFieldModel(score=None)
        extracted = OptionalFieldModel(score=0.5)

        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        with check:
            assert score_result.strategy == "numeric_tolerance"
        with check:
            assert score_result.matched is False

    def test_annotated_field_aliases_with_none_values_return_matched(self) -> None:
        """All Optional annotated fields set to None should produce matched results."""
        expected = AnnotatedFieldModel()
        extracted = AnnotatedFieldModel()

        results = compare_fields(expected, extracted)

        for result in results:
            with check:
                assert result.matched is True, f"Field {result.field_name!r} expected matched=True"

    def test_mismatched_model_types_raises_type_error(self) -> None:
        """Passing two different model types should raise TypeError with a clear message."""
        expected = NumericModel(score=0.5, count=1)
        extracted = SimpleModel(score=0.5, label="high")

        with pytest.raises(TypeError, match="expected and extracted must be the same model type"):
            compare_fields(expected, extracted)

    @pytest.mark.parametrize(
        ("exp_score", "ext_score", "exp_label", "ext_label", "is_score_matched", "is_label_matched"),
        [
            pytest.param(0.5, 0.5, "high", "high", True, True, id="all_match"),
            pytest.param(0.5, 0.9, "high", "high", False, True, id="score_mismatches"),
            pytest.param(0.5, 0.5, "high", "low", True, False, id="label_mismatches"),
            pytest.param(0.5, 0.9, "high", "low", False, False, id="both_mismatch"),
        ],
    )
    def test_field_match_combinations(
        self,
        *,
        exp_score: float,
        ext_score: float,
        exp_label: str,
        ext_label: str,
        is_score_matched: bool,
        is_label_matched: bool,
    ) -> None:
        """compare_fields should correctly record match/mismatch for each field independently.

        Args:
            exp_score (float): Expected score value.
            ext_score (float): Extracted score value.
            exp_label (str): Expected label value.
            ext_label (str): Extracted label value.
            is_score_matched (bool): Whether the score field should be matched.
            is_label_matched (bool): Whether the label field should be matched.
        """
        expected = SimpleModel(score=exp_score, label=exp_label)
        extracted = SimpleModel(score=ext_score, label=ext_label)

        results = compare_fields(expected, extracted)
        results_by_name = {result.field_name: result for result in results}

        with check:
            assert results_by_name["score"].matched is is_score_matched
        with check:
            assert results_by_name["label"].matched is is_label_matched
