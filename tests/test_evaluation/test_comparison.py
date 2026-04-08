"""Tests for the field-level comparison engine in dfkit.evaluation.comparison."""

from typing import Literal

import pytest
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.comparison import (
    DEFAULT_SET_OVERLAP_THRESHOLD,
    FieldConfig,
    _dispatch_comparison,
    compare_dict,
    compare_exact,
    compare_fields,
    compare_numeric,
    compare_set_overlap,
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


class TagsModel(BaseModel):
    """Model with a list[str] field for FieldConfig set overlap config tests.

    Attributes:
        score (float): A numeric score.
        tags (list[str]): A list of string tags.
    """

    score: float
    tags: list[str]


# endregion


class TestCompareNumeric:
    """Tests for compare_numeric — numeric tolerance comparison strategy."""

    def test_exact_match_returns_matched(self) -> None:
        """Identical float values should produce a matched result."""
        # Arrange / Act
        result = compare_numeric("score", 0.5, 0.5)

        # Assert
        assert result.matched is True

    def test_within_tolerance_returns_matched(self) -> None:
        """Values within the specified tolerance should produce a matched result."""
        # Arrange / Act
        result = compare_numeric("score", 0.50, 0.52, tolerance=0.05)

        # Assert
        assert result.matched is True

    def test_outside_tolerance_returns_not_matched(self) -> None:
        """Values exceeding the specified tolerance should produce a not-matched result."""
        # Arrange / Act
        result = compare_numeric("score", 0.50, 0.60, tolerance=0.05)

        # Assert
        assert result.matched is False

    def test_both_none_returns_matched(self) -> None:
        """Two None values should produce a matched result."""
        # Arrange / Act
        result = compare_numeric("score", None, None)

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted value should produce a not-matched result."""
        # Arrange / Act
        result = compare_numeric("score", None, 0.5)

        # Assert
        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected value with None extracted should produce a not-matched result."""
        # Arrange / Act
        result = compare_numeric("score", 0.5, None)

        # Assert
        assert result.matched is False

    def test_details_contains_difference_and_tolerance(self) -> None:
        """Details dict should include the absolute difference and the applied tolerance."""
        # Arrange / Act
        result = compare_numeric("score", 0.50, 0.52, tolerance=0.05)

        # Assert
        with check:
            assert "difference" in result.details
        with check:
            assert "tolerance" in result.details
        with check:
            assert result.details["difference"] == pytest.approx(0.02)
        with check:
            assert result.details["tolerance"] == pytest.approx(0.05)

    def test_strategy_is_numeric_tolerance(self) -> None:
        """Strategy field should always be 'numeric_tolerance'."""
        # Arrange / Act
        result = compare_numeric("score", 0.1, 0.1)

        # Assert
        assert result.strategy == "numeric_tolerance"

    def test_field_name_is_preserved(self) -> None:
        """The field_name argument should appear in the returned FieldComparison."""
        # Arrange / Act
        result = compare_numeric("my_field", 0.3, 0.3)

        # Assert
        assert result.field_name == "my_field"

    def test_at_tolerance_boundary_returns_matched(self) -> None:
        """A difference exactly equal to the tolerance should produce a matched result."""
        # Arrange — use integer-friendly fractions to avoid floating-point representation error
        # (0.55 - 0.50 = 0.050000000000000044 due to IEEE 754, so use 0.10 - 0.05 = 0.05 exactly)
        result = compare_numeric("score", 0.0, 0.05, tolerance=0.05)

        # Assert
        assert result.matched is True

    @pytest.mark.parametrize(
        ("expected", "extracted", "tolerance", "should_match"),
        [
            pytest.param(1.0, 1.0, 0.01, True, id="identical"),
            pytest.param(0.0, 0.04, 0.05, True, id="within_tolerance"),
            pytest.param(0.0, 0.06, 0.05, False, id="outside_tolerance"),
            pytest.param(-0.5, -0.45, 0.05, True, id="negative_within_tolerance"),
            pytest.param(-0.5, -0.44, 0.05, False, id="negative_outside_tolerance"),
            pytest.param(float("nan"), float("nan"), 0.05, False, id="both_nan"),
        ],
    )
    def test_various_numeric_combinations(
        self,
        *,
        expected: float,
        extracted: float,
        tolerance: float,
        should_match: bool,
    ) -> None:
        """compare_numeric should correctly classify a variety of numeric pairs.

        Args:
            expected (float): Expected numeric value.
            extracted (float): Extracted numeric value.
            tolerance (float): Tolerance threshold.
            should_match (bool): Whether the comparison should be matched.
        """
        # Arrange / Act
        result = compare_numeric("x", expected, extracted, tolerance=tolerance)

        # Assert
        assert result.matched is should_match


class TestCompareExact:
    """Tests for compare_exact — exact equality comparison strategy."""

    def test_equal_strings_returns_matched(self) -> None:
        """Identical string values should produce a matched result."""
        # Arrange / Act
        result = compare_exact("label", "hello", "hello")

        # Assert
        assert result.matched is True

    def test_unequal_strings_returns_not_matched(self) -> None:
        """Different string values should produce a not-matched result."""
        # Arrange / Act
        result = compare_exact("label", "hello", "world")

        # Assert
        assert result.matched is False

    def test_equal_literal_values_returns_matched(self) -> None:
        """Matching Literal-typed string values should produce a matched result."""
        # Arrange / Act
        result = compare_exact("strength", "Strong", "Strong")

        # Assert
        assert result.matched is True

    def test_both_none_returns_matched(self) -> None:
        """Two None values should produce a matched result."""
        # Arrange / Act
        result = compare_exact("label", None, None)

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted value should produce a not-matched result."""
        # Arrange / Act
        result = compare_exact("label", None, "value")

        # Assert
        assert result.matched is False

    def test_strategy_is_exact_match(self) -> None:
        """Strategy field should always be 'exact_match'."""
        # Arrange / Act
        result = compare_exact("label", "a", "a")

        # Assert
        assert result.strategy == "exact_match"

    def test_field_name_is_preserved(self) -> None:
        """The field_name argument should appear in the returned FieldComparison."""
        # Arrange / Act
        result = compare_exact("my_field", "x", "x")

        # Assert
        assert result.field_name == "my_field"

    @pytest.mark.parametrize(
        ("expected", "extracted", "should_match"),
        [
            pytest.param(42, 42, True, id="equal_ints"),
            pytest.param(42, 43, False, id="unequal_ints"),
            pytest.param(True, True, True, id="equal_bools"),
            pytest.param(True, False, False, id="unequal_bools"),
            pytest.param("abc", "abc", True, id="equal_strings"),
            pytest.param("abc", "ABC", False, id="case_different_strings"),
        ],
    )
    def test_various_scalar_types(
        self,
        expected: object,
        extracted: object,
        should_match: bool,  # noqa: FBT001
    ) -> None:
        """compare_exact should handle scalars (int, bool, str) correctly.

        Args:
            expected (object): Expected value.
            extracted (object): Extracted value.
            should_match (bool): Whether the comparison should be matched.
        """
        # Arrange / Act
        result = compare_exact("field", expected, extracted)

        # Assert
        assert result.matched is should_match


class TestCompareSetOverlap:
    """Tests for compare_set_overlap — Jaccard set overlap comparison strategy."""

    def test_identical_lists_returns_matched_with_ratio_one(self) -> None:
        """Identical lists should produce a matched result with overlap_ratio of 1.0."""
        # Arrange / Act
        result = compare_set_overlap("tags", ["a", "b", "c"], ["a", "b", "c"])

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details["overlap_ratio"] == pytest.approx(1.0)

    def test_partial_overlap_above_threshold_returns_matched(self) -> None:
        """Partial overlap exceeding the default threshold should produce a matched result."""
        # Arrange — union=4, intersection=3 → ratio=0.75 > 0.7
        result = compare_set_overlap("tags", ["a", "b", "c", "d"], ["a", "b", "c"])

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details["overlap_ratio"] == pytest.approx(0.75)

    def test_partial_overlap_below_threshold_returns_not_matched(self) -> None:
        """Partial overlap below the default threshold should produce a not-matched result."""
        # Arrange — union=4, intersection=1 → ratio=0.25 < 0.7
        result = compare_set_overlap("tags", ["a", "b", "c"], ["a", "d"])

        # Assert
        assert result.matched is False

    def test_case_insensitive_matching_by_default(self) -> None:
        """Mixed-case lists should match when case_sensitive=False (default)."""
        # Arrange / Act
        result = compare_set_overlap("tags", ["Rain"], ["rain"])

        # Assert
        assert result.matched is True

    def test_case_sensitive_matching_distinguishes_case(self) -> None:
        """Mixed-case lists should not match when case_sensitive=True."""
        # Arrange / Act
        result = compare_set_overlap("tags", ["Rain"], ["rain"], case_sensitive=True)

        # Assert
        assert result.matched is False

    def test_both_none_returns_matched(self) -> None:
        """Two None lists should produce a matched result."""
        # Arrange / Act
        result = compare_set_overlap("tags", None, None)

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted list should produce a not-matched result."""
        # Arrange / Act
        result = compare_set_overlap("tags", None, ["a"])

        # Assert
        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected list with None extracted should produce a not-matched result."""
        # Arrange / Act
        result = compare_set_overlap("tags", ["a"], None)

        # Assert
        assert result.matched is False

    def test_both_empty_lists_returns_matched(self) -> None:
        """Two empty lists should produce a matched result (empty union edge case)."""
        # Arrange / Act
        result = compare_set_overlap("tags", [], [])

        # Assert
        assert result.matched is True

    def test_details_contains_missing_and_extra(self) -> None:
        """Details should include 'missing' and 'extra' item lists."""
        # Arrange / Act
        result = compare_set_overlap("tags", ["a", "b"], ["b", "c"])

        # Assert
        with check:
            assert "missing" in result.details
        with check:
            assert "extra" in result.details
        with check:
            assert result.details["missing"] == ["a"]
        with check:
            assert result.details["extra"] == ["c"]

    def test_strategy_is_set_overlap(self) -> None:
        """Strategy field should always be 'set_overlap'."""
        # Arrange / Act
        result = compare_set_overlap("tags", ["a"], ["a"])

        # Assert
        assert result.strategy == "set_overlap"

    @pytest.mark.parametrize(
        ("expected_list", "extracted_list", "threshold", "should_match"),
        [
            pytest.param(["x", "y", "z"], ["x", "y", "z"], 0.7, True, id="full_match"),
            pytest.param(["x", "y"], ["x", "z"], 0.3, True, id="one_of_two_above_low_threshold"),
            pytest.param(["x", "y"], ["x", "z"], 0.6, False, id="one_of_two_below_high_threshold"),
            pytest.param(["a", "b", "c"], ["a", "b", "c", "d"], 0.7, True, id="extra_item_above_threshold"),
        ],
    )
    def test_various_overlap_scenarios(
        self,
        expected_list: list[str],
        extracted_list: list[str],
        threshold: float,
        should_match: bool,  # noqa: FBT001
    ) -> None:
        """compare_set_overlap should classify pairs correctly across threshold combinations.

        Args:
            expected_list (list[str]): Expected string list.
            extracted_list (list[str]): Extracted string list.
            threshold (float): Jaccard threshold to apply.
            should_match (bool): Whether the comparison should be matched.
        """
        # Arrange / Act
        result = compare_set_overlap("tags", expected_list, extracted_list, threshold=threshold)

        # Assert
        assert result.matched is should_match


class TestFieldConfig:
    """Tests for the FieldConfig model — per-field comparison configuration."""

    def test_all_fields_default_to_none(self) -> None:
        """A FieldConfig created with no arguments should have all fields set to None."""
        # Arrange / Act
        config = FieldConfig()

        # Assert
        with check:
            assert config.tolerance is None
        with check:
            assert config.threshold is None
        with check:
            assert config.case_sensitive is None

    def test_tolerance_can_be_set(self) -> None:
        """FieldConfig should accept and store a custom tolerance value."""
        # Arrange / Act
        config = FieldConfig(tolerance=0.10)

        # Assert
        assert config.tolerance == pytest.approx(0.10)

    def test_threshold_can_be_set(self) -> None:
        """FieldConfig should accept and store a custom threshold value."""
        # Arrange / Act
        config = FieldConfig(threshold=0.5)

        # Assert
        assert config.threshold == pytest.approx(0.5)

    def test_case_sensitive_can_be_set(self) -> None:
        """FieldConfig should accept and store a case_sensitive boolean."""
        # Arrange / Act
        config = FieldConfig(case_sensitive=True)

        # Assert
        assert config.case_sensitive is True

    def test_all_fields_can_be_set_together(self) -> None:
        """FieldConfig should accept tolerance, threshold, and case_sensitive simultaneously."""
        # Arrange / Act
        config = FieldConfig(tolerance=0.02, threshold=0.8, case_sensitive=False)

        # Assert
        with check:
            assert config.tolerance == pytest.approx(0.02)
        with check:
            assert config.threshold == pytest.approx(0.8)
        with check:
            assert config.case_sensitive is False


class TestDefaultSetOverlapThreshold:
    """Tests for DEFAULT_SET_OVERLAP_THRESHOLD constant."""

    def test_default_set_overlap_threshold_value(self) -> None:
        """DEFAULT_SET_OVERLAP_THRESHOLD should equal 0.7."""
        # Assert
        assert pytest.approx(0.7) == DEFAULT_SET_OVERLAP_THRESHOLD

    def test_default_set_overlap_threshold_applied_when_no_config(self) -> None:
        """compare_set_overlap with no threshold arg should use DEFAULT_SET_OVERLAP_THRESHOLD."""
        # Arrange — union=4, intersection=3 → ratio=0.75; passes default threshold 0.7
        result = compare_set_overlap("tags", ["a", "b", "c", "d"], ["a", "b", "c"])

        # Assert
        assert result.matched is True


class TestCompareDict:
    """Tests for compare_dict — key-by-key dictionary comparison strategy."""

    def test_matching_dicts_returns_matched(self) -> None:
        """Identical dicts should produce a matched result."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x", "b": "y"}, {"a": "x", "b": "y"})

        # Assert
        assert result.matched is True

    def test_mismatched_values_returns_not_matched(self) -> None:
        """Dicts with differing values for a shared key should produce a not-matched result."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x"}, {"a": "z"})

        # Assert
        assert result.matched is False

    def test_missing_key_in_extracted_returns_not_matched(self) -> None:
        """A key present in expected but absent in extracted should produce a not-matched result."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x", "b": "y"}, {"a": "x"})

        # Assert
        with check:
            assert result.matched is False
        with check:
            assert result.details["missing_keys"] == ["b"]

    def test_extra_key_in_extracted_returns_not_matched(self) -> None:
        """A key present in extracted but absent in expected should produce a not-matched result."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x"}, {"a": "x", "b": "y"})

        # Assert
        with check:
            assert result.matched is False
        with check:
            assert result.details["extra_keys"] == ["b"]

    def test_both_none_returns_matched(self) -> None:
        """Two None dicts should produce a matched result."""
        # Arrange / Act
        result = compare_dict("data", None, None)

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details == {}

    def test_both_empty_dicts_returns_matched(self) -> None:
        """Two empty dicts should produce a matched result."""
        # Arrange / Act
        result = compare_dict("data", {}, {})

        # Assert
        assert result.matched is True

    def test_expected_none_extracted_present_returns_not_matched(self) -> None:
        """None expected with a present extracted dict should produce a not-matched result."""
        # Arrange / Act
        result = compare_dict("data", None, {"a": "x"})

        # Assert
        assert result.matched is False

    def test_expected_present_extracted_none_returns_not_matched(self) -> None:
        """A present expected dict with None extracted should produce a not-matched result."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x"}, None)

        # Assert
        assert result.matched is False

    def test_strategy_is_dict_comparison(self) -> None:
        """Strategy field should always be 'dict_comparison'."""
        # Arrange / Act
        result = compare_dict("data", {"a": "1"}, {"a": "1"})

        # Assert
        assert result.strategy == "dict_comparison"

    def test_details_contains_key_results_missing_and_extra(self) -> None:
        """Details should include 'key_results', 'missing_keys', and 'extra_keys'."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x", "b": "y"}, {"a": "x", "c": "z"})

        # Assert
        with check:
            assert "key_results" in result.details
        with check:
            assert "missing_keys" in result.details
        with check:
            assert "extra_keys" in result.details
        with check:
            assert result.details["missing_keys"] == ["b"]
        with check:
            assert result.details["extra_keys"] == ["c"]

    def test_key_results_reflects_per_key_match(self) -> None:
        """key_results should record the matched flag for each shared key."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x", "b": "y"}, {"a": "x", "b": "z"})

        # Assert
        with check:
            assert result.details["key_results"]["a"]["matched"] is True
        with check:
            assert result.details["key_results"]["b"]["matched"] is False

    def test_nested_float_values_auto_dispatch_to_numeric_tolerance(self) -> None:
        """compare_dict with no explicit comparator should auto-dispatch float values to numeric strategy."""
        # Arrange — close floats that would fail exact match but pass numeric tolerance
        # Act
        result = compare_dict("data", {"score": 0.50}, {"score": 0.52})

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details["key_results"]["score"]["matched"] is True

    def test_nested_float_values_outside_tolerance_not_matched(self) -> None:
        """compare_dict with default dispatch should reject float values beyond default tolerance."""
        # Arrange — difference of 0.20 far exceeds the default numeric tolerance
        # Act
        result = compare_dict("data", {"score": 0.50}, {"score": 0.70})

        # Assert
        assert result.matched is False

    def test_nested_list_str_values_auto_dispatch_to_set_overlap(self) -> None:
        """compare_dict with no explicit comparator should auto-dispatch list[str] values to set_overlap strategy."""
        # Arrange — overlapping lists that pass set overlap threshold
        # union=4, intersection=3 → ratio=0.75 > default 0.7
        result = compare_dict(
            "data",
            {"tags": ["alpha", "beta", "gamma"]},
            {"tags": ["alpha", "beta", "gamma", "delta"]},
        )

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details["key_results"]["tags"]["matched"] is True

    def test_nested_dict_values_auto_dispatch_recurse(self) -> None:
        """compare_dict with no explicit comparator should recurse into nested dicts."""
        # Arrange — nested dicts with matching string values
        # Act
        result = compare_dict(
            "data",
            {"inner": {"key": "val"}},
            {"inner": {"key": "val"}},
        )

        # Assert
        with check:
            assert result.matched is True
        with check:
            assert result.details["key_results"]["inner"]["matched"] is True

    def test_nested_dict_mismatch_propagates_to_parent(self) -> None:
        """A mismatch in a nested dict should cause the parent compare_dict to not match."""
        # Arrange — inner dict has different value for shared key
        # Act
        result = compare_dict(
            "data",
            {"inner": {"key": "expected"}},
            {"inner": {"key": "different"}},
        )

        # Assert
        assert result.matched is False

    def test_config_parameter_is_accepted(self) -> None:
        """compare_dict should accept an optional config parameter without error."""
        # Arrange / Act
        result = compare_dict("data", {"a": "x"}, {"a": "x"}, config=FieldConfig())

        # Assert
        assert result.matched is True


class TestCompareFields:
    """Tests for compare_fields — type-driven field comparison dispatcher."""

    def test_float_field_dispatches_to_numeric_strategy(self) -> None:
        """Float fields should be compared with the 'numeric_tolerance' strategy."""
        # Arrange
        expected = NumericModel(score=0.5, count=3)
        extracted = NumericModel(score=0.52, count=3)

        # Act
        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        with check:
            assert score_result.strategy == "numeric_tolerance"
        with check:
            assert score_result.matched is True

    def test_int_field_dispatches_to_exact_strategy(self) -> None:
        """Integer fields should be compared with the 'exact_match' strategy."""
        # Arrange
        expected = NumericModel(score=0.5, count=3)
        extracted = NumericModel(score=0.5, count=3)

        # Act
        results = compare_fields(expected, extracted)
        count_result = next(r for r in results if r.field_name == "count")

        # Assert
        with check:
            assert count_result.strategy == "exact_match"
        with check:
            assert count_result.matched is True

    def test_str_field_dispatches_to_exact_strategy(self) -> None:
        """String fields should be compared with the 'exact_match' strategy."""
        # Arrange
        expected = MixedModel(score=0.5, label="high", flag=True, tags=[], rank=1)
        extracted = MixedModel(score=0.5, label="high", flag=True, tags=[], rank=1)

        # Act
        results = compare_fields(expected, extracted)
        label_result = next(r for r in results if r.field_name == "label")

        # Assert
        with check:
            assert label_result.strategy == "exact_match"
        with check:
            assert label_result.matched is True

    def test_bool_field_dispatches_to_exact_strategy(self) -> None:
        """Boolean fields should be compared with the 'exact_match' strategy."""
        # Arrange
        expected = MixedModel(score=0.5, label="high", flag=True, tags=[], rank=1)
        extracted = MixedModel(score=0.5, label="high", flag=True, tags=[], rank=1)

        # Act
        results = compare_fields(expected, extracted)
        flag_result = next(r for r in results if r.field_name == "flag")

        # Assert
        with check:
            assert flag_result.strategy == "exact_match"
        with check:
            assert flag_result.matched is True

    def test_list_str_field_dispatches_to_set_overlap_strategy(self) -> None:
        """list[str] fields should be compared with the 'set_overlap' strategy."""
        # Arrange
        expected = MixedModel(score=0.5, label="high", flag=True, tags=["a", "b"], rank=1)
        extracted = MixedModel(score=0.5, label="high", flag=True, tags=["a", "b"], rank=1)

        # Act
        results = compare_fields(expected, extracted)
        tags_result = next(r for r in results if r.field_name == "tags")

        # Assert
        with check:
            assert tags_result.strategy == "set_overlap"
        with check:
            assert tags_result.matched is True

    def test_literal_field_dispatches_to_exact_strategy(self) -> None:
        """Literal-typed fields should be compared with the 'exact_match' strategy."""
        # Arrange
        expected = LiteralModel(category="A")
        extracted = LiteralModel(category="A")

        # Act
        results = compare_fields(expected, extracted)
        cat_result = next(r for r in results if r.field_name == "category")

        # Assert
        with check:
            assert cat_result.strategy == "exact_match"
        with check:
            assert cat_result.matched is True

    def test_all_fields_present_in_results(self) -> None:
        """compare_fields should return one FieldComparison per field in the model."""
        # Arrange
        expected = MixedModel(score=0.5, label="high", flag=True, tags=["a"], rank=1)
        extracted = MixedModel(score=0.5, label="high", flag=True, tags=["a"], rank=1)

        # Act
        results = compare_fields(expected, extracted)
        field_names = {r.field_name for r in results}

        # Assert
        assert field_names == {"score", "label", "flag", "tags", "rank"}

    def test_field_config_tolerance_overrides_default_for_float_field(self) -> None:
        """A FieldConfig with custom tolerance should override the default 0.05 for that field."""
        # Arrange — difference of 0.15 is outside default 0.05 but inside custom 0.20
        expected = NumericModel(score=0.50, count=1)
        extracted = NumericModel(score=0.65, count=1)

        # Act
        results = compare_fields(expected, extracted, field_configs={"score": FieldConfig(tolerance=0.20)})
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        assert score_result.matched is True

    def test_field_config_tolerance_tighter_than_default_produces_mismatch(self) -> None:
        """A tighter FieldConfig tolerance should cause a value within the default range to not match."""
        # Arrange — difference of 0.03 is inside default 0.05 but outside custom 0.01
        expected = NumericModel(score=0.50, count=1)
        extracted = NumericModel(score=0.53, count=1)

        # Act
        results = compare_fields(expected, extracted, field_configs={"score": FieldConfig(tolerance=0.01)})
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        assert score_result.matched is False

    def test_default_tolerance_widens_numeric_match(self) -> None:
        """A wider default_tolerance should cause a float field to match when it otherwise would not."""
        # Arrange — difference of 0.15 exceeds the built-in default (0.05) but is within 0.20
        expected = NumericModel(score=0.50, count=1)
        extracted = NumericModel(score=0.65, count=1)

        # Act
        results = compare_fields(expected, extracted, default_tolerance=0.20)
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        assert score_result.matched is True

    def test_default_tolerance_tighter_than_builtin_produces_mismatch(self) -> None:
        """A tighter default_tolerance should reject a float difference that the built-in default would accept."""
        # Arrange — difference of 0.03 is inside the built-in default (0.05) but outside 0.01
        expected = NumericModel(score=0.50, count=1)
        extracted = NumericModel(score=0.53, count=1)

        # Act
        results = compare_fields(expected, extracted, default_tolerance=0.01)
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        assert score_result.matched is False

    def test_field_config_tolerance_takes_precedence_over_default_tolerance(self) -> None:
        """A field-specific FieldConfig tolerance should override default_tolerance."""
        # Arrange — default is tight (0.01), but score has a wide per-field override (0.20)
        expected = NumericModel(score=0.50, count=1)
        extracted = NumericModel(score=0.65, count=1)

        # Act
        results = compare_fields(
            expected,
            extracted,
            field_configs={"score": FieldConfig(tolerance=0.20)},
            default_tolerance=0.01,
        )
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        assert score_result.matched is True

    def test_optional_float_field_both_none_returns_matched(self) -> None:
        """Optional float fields that are both None should produce a matched result."""
        # Arrange
        expected = OptionalFieldModel(score=None, tags=None)
        extracted = OptionalFieldModel(score=None, tags=None)

        # Act
        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        assert score_result.matched is True

    def test_optional_list_str_field_dispatches_to_set_overlap(self) -> None:
        """Optional list[str] fields should dispatch to the 'set_overlap' strategy."""
        # Arrange
        expected = OptionalFieldModel(score=None, tags=["x", "y"])
        extracted = OptionalFieldModel(score=None, tags=["x", "y"])

        # Act
        results = compare_fields(expected, extracted)
        tags_result = next(r for r in results if r.field_name == "tags")

        # Assert
        assert tags_result.strategy == "set_overlap"

    def test_dict_field_dispatches_to_dict_comparison_strategy(self) -> None:
        """dict[str, float] fields should dispatch to the 'dict_comparison' strategy."""
        # Arrange
        expected = DictFieldModel(metrics={"rmse": 0.1, "mae": 0.05})
        extracted = DictFieldModel(metrics={"rmse": 0.1, "mae": 0.05})

        # Act
        results = compare_fields(expected, extracted)
        metrics_result = next(r for r in results if r.field_name == "metrics")

        # Assert
        with check:
            assert metrics_result.strategy == "dict_comparison"
        with check:
            assert metrics_result.matched is True

    def test_annotated_correlation_field_dispatches_to_numeric_strategy(self) -> None:
        """Annotated Correlation (float) fields should dispatch to 'numeric_tolerance' strategy."""
        # Arrange
        expected = AnnotatedFieldModel(correlation=0.75, strength=None, confounders=None)
        extracted = AnnotatedFieldModel(correlation=0.77, strength=None, confounders=None)

        # Act
        results = compare_fields(expected, extracted)
        corr_result = next(r for r in results if r.field_name == "correlation")

        # Assert
        with check:
            assert corr_result.strategy == "numeric_tolerance"
        with check:
            assert corr_result.matched is True

    def test_annotated_relationship_strength_field_dispatches_to_exact_strategy(self) -> None:
        """Annotated RelationshipStrength (Literal) fields should dispatch to 'exact_match' strategy."""
        # Arrange
        expected = AnnotatedFieldModel(correlation=None, strength="Strong", confounders=None)
        extracted = AnnotatedFieldModel(correlation=None, strength="Strong", confounders=None)

        # Act
        results = compare_fields(expected, extracted)
        strength_result = next(r for r in results if r.field_name == "strength")

        # Assert
        with check:
            assert strength_result.strategy == "exact_match"
        with check:
            assert strength_result.matched is True

    def test_annotated_confounders_field_dispatches_to_set_overlap_strategy(self) -> None:
        """Annotated Confounders (list[str]) fields should dispatch to 'set_overlap' strategy."""
        # Arrange
        expected = AnnotatedFieldModel(correlation=None, strength=None, confounders=["age", "sex"])
        extracted = AnnotatedFieldModel(correlation=None, strength=None, confounders=["age", "sex"])

        # Act
        results = compare_fields(expected, extracted)
        conf_result = next(r for r in results if r.field_name == "confounders")

        # Assert
        with check:
            assert conf_result.strategy == "set_overlap"
        with check:
            assert conf_result.matched is True

    def test_matched_fields_have_correct_expected_and_extracted_values(self) -> None:
        """Each FieldComparison should carry the correct expected and extracted values."""
        # Arrange
        expected = NumericModel(score=0.8, count=5)
        extracted = NumericModel(score=0.8, count=5)

        # Act
        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        with check:
            assert score_result.expected == pytest.approx(0.8)
        with check:
            assert score_result.extracted == pytest.approx(0.8)

    @pytest.mark.parametrize(
        ("exp_score", "ext_score", "exp_label", "ext_label", "score_matched", "label_matched"),
        [
            pytest.param(0.5, 0.5, "high", "high", True, True, id="all_match"),
            pytest.param(0.5, 0.9, "high", "high", False, True, id="score_mismatches"),
            pytest.param(0.5, 0.5, "high", "low", True, False, id="label_mismatches"),
            pytest.param(0.5, 0.9, "high", "low", False, False, id="both_mismatch"),
        ],
    )
    def test_field_match_combinations(  # noqa: PLR0917
        self,
        exp_score: float,
        ext_score: float,
        exp_label: str,
        ext_label: str,
        score_matched: bool,  # noqa: FBT001
        label_matched: bool,  # noqa: FBT001
    ) -> None:
        """compare_fields should correctly record match/mismatch for each field independently.

        Args:
            exp_score (float): Expected score value.
            ext_score (float): Extracted score value.
            exp_label (str): Expected label value.
            ext_label (str): Extracted label value.
            score_matched (bool): Whether the score field should be matched.
            label_matched (bool): Whether the label field should be matched.
        """
        # Arrange
        expected = SimpleModel(score=exp_score, label=exp_label)
        extracted = SimpleModel(score=ext_score, label=ext_label)

        # Act
        results = compare_fields(expected, extracted)
        results_by_name = {r.field_name: r for r in results}

        # Assert
        with check:
            assert results_by_name["score"].matched is score_matched
        with check:
            assert results_by_name["label"].matched is label_matched

    def test_annotated_field_aliases_with_none_values_return_matched(self) -> None:
        """All Optional annotated fields set to None should produce matched results."""
        # Arrange
        expected = AnnotatedFieldModel()
        extracted = AnnotatedFieldModel()

        # Act
        results = compare_fields(expected, extracted)

        # Assert
        for result in results:
            with check:
                assert result.matched is True, f"Field {result.field_name!r} expected matched=True"

    def test_optional_float_field_expected_none_extracted_present_returns_not_matched(self) -> None:
        """When expected score is None and extracted is a float, numeric strategy is used and result is not matched."""
        # Arrange
        expected = OptionalFieldModel(score=None)
        extracted = OptionalFieldModel(score=0.5)

        # Act
        results = compare_fields(expected, extracted)
        score_result = next(r for r in results if r.field_name == "score")

        # Assert
        with check:
            assert score_result.strategy == "numeric_tolerance"
        with check:
            assert score_result.matched is False

    def test_field_config_threshold_overrides_default_for_set_overlap_field(self) -> None:
        """A FieldConfig with custom threshold should override the default for set_overlap fields."""
        # Arrange — union=4, intersection=2 → ratio=0.5; fails default 0.7 but passes custom 0.4
        expected = TagsModel(score=0.5, tags=["alpha", "beta", "gamma"])
        extracted = TagsModel(score=0.5, tags=["alpha", "beta", "delta"])

        # Act
        results = compare_fields(expected, extracted, field_configs={"tags": FieldConfig(threshold=0.4)})
        tags_result = next(r for r in results if r.field_name == "tags")

        # Assert
        with check:
            assert tags_result.strategy == "set_overlap"
        with check:
            assert tags_result.matched is True

    def test_field_config_case_sensitive_overrides_default_for_set_overlap_field(self) -> None:
        """A FieldConfig with case_sensitive=True should cause case-sensitive set overlap comparison."""
        # Arrange — "Rain" vs "rain" differs only in case; case-sensitive would not match but default would
        expected = TagsModel(score=0.5, tags=["Rain", "Snow"])
        extracted = TagsModel(score=0.5, tags=["rain", "snow"])

        # Act
        results = compare_fields(
            expected,
            extracted,
            field_configs={"tags": FieldConfig(case_sensitive=True)},
        )
        tags_result = next(r for r in results if r.field_name == "tags")

        # Assert
        with check:
            assert tags_result.strategy == "set_overlap"
        with check:
            assert tags_result.matched is False

    def test_field_config_case_sensitive_false_matches_mixed_case(self) -> None:
        """A FieldConfig with case_sensitive=False should match mixed-case set overlap fields."""
        # Arrange — same items, different case; explicit case_sensitive=False should match
        expected = TagsModel(score=0.5, tags=["Alpha", "Beta"])
        extracted = TagsModel(score=0.5, tags=["ALPHA", "BETA"])

        # Act
        results = compare_fields(
            expected,
            extracted,
            field_configs={"tags": FieldConfig(case_sensitive=False)},
        )
        tags_result = next(r for r in results if r.field_name == "tags")

        # Assert
        assert tags_result.matched is True

    def test_mixed_field_configs_applies_only_to_configured_fields(self) -> None:
        """field_configs should apply per-field config only to listed fields; other fields use defaults."""
        # Arrange — score has tight config (0.01); tags has no config so uses default dispatch
        # score difference 0.03 exceeds custom tolerance 0.01 but is within default 0.05
        expected = TagsModel(score=0.50, tags=["x", "y"])
        extracted = TagsModel(score=0.53, tags=["x", "y"])

        # Act
        results = compare_fields(
            expected,
            extracted,
            field_configs={"score": FieldConfig(tolerance=0.01)},
        )
        results_by_name = {r.field_name: r for r in results}

        # Assert
        with check:
            assert results_by_name["score"].matched is False, "score should fail tight custom tolerance"
        with check:
            assert results_by_name["tags"].matched is True, "tags should match using default dispatch"

    def test_field_configs_none_falls_back_to_defaults(self) -> None:
        """Passing field_configs=None should behave identically to omitting field_configs."""
        # Arrange — difference of 0.03 within default 0.05 tolerance
        expected = NumericModel(score=0.50, count=2)
        extracted = NumericModel(score=0.53, count=2)

        # Act
        results_explicit_none = compare_fields(expected, extracted, field_configs=None)
        results_omitted = compare_fields(expected, extracted)

        # Assert
        score_explicit = next(r for r in results_explicit_none if r.field_name == "score")
        score_omitted = next(r for r in results_omitted if r.field_name == "score")
        with check:
            assert score_explicit.matched is score_omitted.matched
        with check:
            assert score_explicit.strategy == score_omitted.strategy


class TestDispatchComparison:
    """Tests for _dispatch_comparison — None expected_value fallback to extracted_value type."""

    def test_none_expected_float_extracted_dispatches_to_numeric(self) -> None:
        """When expected is None and extracted is float, strategy should be 'numeric_tolerance'."""
        # Arrange / Act
        result = _dispatch_comparison("score", None, 0.5, config=FieldConfig())

        # Assert
        assert result.strategy == "numeric_tolerance"

    def test_none_expected_float_extracted_returns_not_matched(self) -> None:
        """When expected is None and extracted is float, result should not be matched."""
        # Arrange / Act
        result = _dispatch_comparison("score", None, 0.5, config=FieldConfig())

        # Assert
        assert result.matched is False

    def test_none_expected_dict_extracted_dispatches_to_dict_comparison(self) -> None:
        """When expected is None and extracted is dict, strategy should be 'dict_comparison'."""
        # Arrange / Act
        result = _dispatch_comparison("metrics", None, {"a": 1.0}, config=FieldConfig())

        # Assert
        assert result.strategy == "dict_comparison"

    def test_none_expected_list_str_extracted_dispatches_to_set_overlap(self) -> None:
        """When expected is None and extracted is list[str], strategy should be 'set_overlap'."""
        # Arrange / Act
        result = _dispatch_comparison("tags", None, ["x", "y"], config=FieldConfig())

        # Assert
        assert result.strategy == "set_overlap"

    def test_both_none_dispatches_to_exact(self) -> None:
        """When both expected and extracted are None, strategy should be 'exact_match'."""
        # Arrange / Act
        result = _dispatch_comparison("field", None, None, config=FieldConfig())

        # Assert
        assert result.strategy == "exact_match"

    def test_float_expected_none_extracted_dispatches_to_numeric(self) -> None:
        """When expected is float and extracted is None, strategy should be 'numeric_tolerance'."""
        # Arrange / Act
        result = _dispatch_comparison("score", 0.5, None, config=FieldConfig())

        # Assert
        assert result.strategy == "numeric_tolerance"

    def test_list_int_dispatches_to_exact_match(self) -> None:
        """list[int] values should fall through to exact_match, not set_overlap."""
        # Arrange / Act
        result = _dispatch_comparison("ids", [1, 2, 3], [1, 2, 3], config=FieldConfig())

        # Assert
        with check:
            assert result.strategy == "exact_match"
        with check:
            assert result.matched is True

    def test_config_tolerance_applied_for_float_dispatch(self) -> None:
        """FieldConfig tolerance should be passed through to numeric comparison."""
        # Arrange — difference 0.15 exceeds default 0.05 but within custom 0.20
        result = _dispatch_comparison("score", 0.50, 0.65, config=FieldConfig(tolerance=0.20))

        # Assert
        assert result.matched is True

    def test_config_threshold_applied_for_list_str_dispatch(self) -> None:
        """FieldConfig threshold should be passed through to set_overlap comparison."""
        # Arrange — union=4, intersection=2 → ratio=0.5; fails default 0.7 but passes custom 0.4
        result = _dispatch_comparison(
            "tags",
            ["a", "b", "c"],
            ["a", "b", "d"],
            config=FieldConfig(threshold=0.4),
        )

        # Assert
        assert result.matched is True

    def test_config_case_sensitive_applied_for_list_str_dispatch(self) -> None:
        """FieldConfig case_sensitive should be passed through to set_overlap comparison."""
        # Arrange — "Alpha" and "alpha" match only when case_sensitive=False
        result_insensitive = _dispatch_comparison(
            "tags",
            ["Alpha"],
            ["alpha"],
            config=FieldConfig(case_sensitive=False),
        )
        result_sensitive = _dispatch_comparison(
            "tags",
            ["Alpha"],
            ["alpha"],
            config=FieldConfig(case_sensitive=True),
        )

        # Assert
        with check:
            assert result_insensitive.matched is True
        with check:
            assert result_sensitive.matched is False
