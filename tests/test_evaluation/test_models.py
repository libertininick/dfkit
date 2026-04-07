"""Tests for the evaluation pipeline models: EvalCase, FieldComparison, and EvalResult."""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.models import EvalCase, EvalResult, FieldComparison

# region Module-level test models


class SampleResult(BaseModel):
    """Minimal Pydantic model used as the result schema in evaluation test cases.

    Attributes:
        value (float): A numeric measurement.
        label (str): A text classification label.
    """

    value: float
    label: str


class AltResult(BaseModel):
    """Alternative Pydantic model used to verify EvalCase accepts any BaseModel subclass.

    Attributes:
        score (float): A numeric score between 0 and 1.
        category (str): A category name.
        count (int): A non-negative integer count.
    """

    score: float
    category: str
    count: int


# endregion


class TestEvalCase:
    """Tests for the EvalCase generic model."""

    def test_construction_with_human_message_and_sample_result_stores_fields(self) -> None:
        """EvalCase should store all provided fields when constructed with HumanMessage and SampleResult."""
        # Arrange
        messages = [HumanMessage(content="What is the value?")]
        expected = SampleResult(value=3.14, label="high")

        # Act
        case = EvalCase(input_messages=messages, expected_result=expected)

        # Assert
        with check:
            assert case.input_messages == messages
        with check:
            assert case.expected_result == expected

    def test_case_id_defaults_to_none(self) -> None:
        """EvalCase case_id should default to None when not provided."""
        # Arrange / Act
        case = EvalCase(
            input_messages=[HumanMessage(content="query")],
            expected_result=SampleResult(value=1.0, label="low"),
        )

        # Assert
        assert case.case_id is None

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """EvalCase metadata should default to an empty dict when not provided."""
        # Arrange / Act
        case = EvalCase(
            input_messages=[HumanMessage(content="query")],
            expected_result=SampleResult(value=0.5, label="medium"),
        )

        # Assert
        assert case.metadata == {}

    def test_explicit_case_id_and_metadata_are_stored(self) -> None:
        """EvalCase should retain explicitly provided case_id and metadata values."""
        # Arrange
        messages = [HumanMessage(content="Classify the sample.")]
        expected = SampleResult(value=2.0, label="medium")
        meta = {"split": "test", "difficulty": "easy"}

        # Act
        case = EvalCase(
            input_messages=messages,
            expected_result=expected,
            case_id="classify-medium",
            metadata=meta,
        )

        # Assert
        with check:
            assert case.case_id == "classify-medium"
        with check:
            assert case.metadata == {"split": "test", "difficulty": "easy"}

    def test_expected_result_accepts_any_base_model_subclass(self) -> None:
        """EvalCase should accept any BaseModel subclass as expected_result."""
        # Arrange
        alt = AltResult(score=0.9, category="premium", count=42)

        # Act
        case = EvalCase(
            input_messages=[HumanMessage(content="categorise this item")],
            expected_result=alt,
        )

        # Assert
        assert case.expected_result == alt

    def test_multiple_input_messages_are_preserved_in_order(self) -> None:
        """EvalCase should preserve all input messages in their original order."""
        # Arrange
        messages = [
            HumanMessage(content="first"),
            HumanMessage(content="second"),
            HumanMessage(content="third"),
        ]

        # Act
        case = EvalCase(
            input_messages=messages,
            expected_result=SampleResult(value=99.0, label="outlier"),
        )

        # Assert
        assert [m.content for m in case.input_messages] == ["first", "second", "third"]

    def test_empty_input_messages_are_stored(self) -> None:
        """EvalCase should accept and store an empty input_messages list."""
        # Arrange / Act
        case = EvalCase(
            input_messages=[],
            expected_result=SampleResult(value=0.0, label="none"),
        )

        # Assert
        assert case.input_messages == []


class TestFieldComparison:
    """Tests for the FieldComparison model."""

    def test_construction_with_all_required_fields_stores_values(self) -> None:
        """FieldComparison should store all required fields when constructed."""
        # Arrange / Act
        fc = FieldComparison(
            field_name="value",
            expected=3.14,
            extracted=3.14,
            matched=True,
            strategy="exact_match",
        )

        # Assert
        with check:
            assert fc.field_name == "value"
        with check:
            assert fc.expected == pytest.approx(3.14)
        with check:
            assert fc.extracted == pytest.approx(3.14)
        with check:
            assert fc.matched is True
        with check:
            assert fc.strategy == "exact_match"

    def test_details_defaults_to_empty_dict(self) -> None:
        """FieldComparison details should default to an empty dict when not provided."""
        # Arrange / Act
        fc = FieldComparison(
            field_name="label",
            expected="high",
            extracted="low",
            matched=False,
            strategy="exact_match",
        )

        # Assert
        assert fc.details == {}

    def test_serialization_round_trip_preserves_all_fields(self) -> None:
        """FieldComparison should survive a model_dump / model_validate round-trip unchanged."""
        # Arrange
        original = FieldComparison(
            field_name="score",
            expected=0.75,
            extracted=0.80,
            matched=False,
            strategy="numeric_tolerance",
            details={"tolerance": 0.05, "delta": 0.05},
        )

        # Act
        raw = original.model_dump()
        restored = FieldComparison.model_validate(raw)

        # Assert
        with check:
            assert restored.field_name == original.field_name
        with check:
            assert restored.expected == pytest.approx(original.expected)
        with check:
            assert restored.extracted == pytest.approx(original.extracted)
        with check:
            assert restored.matched == original.matched
        with check:
            assert restored.strategy == original.strategy
        with check:
            assert restored.details == original.details

    def test_populated_details_dict_is_stored_as_provided(self) -> None:
        """FieldComparison should store a non-empty details dict unchanged."""
        # Arrange
        details = {"overlap_ratio": 0.8, "missing": ["x"], "extra": []}

        # Act
        fc = FieldComparison(
            field_name="tags",
            expected=["a", "b", "c"],
            extracted=["a", "b"],
            matched=False,
            strategy="set_overlap",
            details=details,
        )

        # Assert
        assert fc.details == details

    def test_matched_false_is_stored_correctly(self) -> None:
        """FieldComparison should correctly represent a non-matching comparison."""
        # Arrange / Act
        fc = FieldComparison(
            field_name="category",
            expected="premium",
            extracted="standard",
            matched=False,
            strategy="exact_match",
        )

        # Assert
        assert fc.matched is False


class TestEvalResult:
    """Tests for the EvalResult generic model and its derived properties."""

    @pytest.mark.parametrize(
        ("comparisons", "expected_score"),
        [
            pytest.param(
                [
                    FieldComparison(field_name="a", expected=1, extracted=1, matched=True, strategy="exact"),
                    FieldComparison(field_name="b", expected=2, extracted=2, matched=True, strategy="exact"),
                    FieldComparison(field_name="c", expected=3, extracted=3, matched=True, strategy="exact"),
                    FieldComparison(field_name="d", expected=4, extracted=9, matched=False, strategy="exact"),
                ],
                0.75,
                id="three_of_four_matched",
            ),
            pytest.param(
                [
                    FieldComparison(field_name="value", expected=7.5, extracted=7.5, matched=True, strategy="exact"),
                    FieldComparison(
                        field_name="label", expected="medium", extracted="medium", matched=True, strategy="exact"
                    ),
                ],
                1.0,
                id="all_matched",
            ),
            pytest.param(
                [],
                1.0,
                id="no_comparisons",
            ),
            pytest.param(
                [
                    FieldComparison(field_name="value", expected=1.0, extracted=2.0, matched=False, strategy="exact"),
                    FieldComparison(
                        field_name="label", expected="high", extracted="low", matched=False, strategy="exact"
                    ),
                ],
                0.0,
                id="none_matched",
            ),
        ],
    )
    def test_score(
        self,
        comparisons: list[FieldComparison],
        expected_score: float,
    ) -> None:
        """EvalResult.score should return the fraction of matched field comparisons.

        Args:
            comparisons (list[FieldComparison]): List of FieldComparison instances to include in the result.
            expected_score (float): The expected score value as a fraction in [0.0, 1.0].
        """
        # Arrange
        case = EvalCase(
            input_messages=[HumanMessage(content="classify")],
            expected_result=SampleResult(value=1.0, label="low"),
        )

        # Act
        result = EvalResult(
            eval_case=case,
            extracted_result=SampleResult(value=1.0, label="low"),
            field_comparisons=comparisons,
            agent_response="extracted output",
        )

        # Assert
        assert result.score == pytest.approx(expected_score)

    @pytest.mark.parametrize(
        ("comparisons", "expected_passed"),
        [
            pytest.param(
                [
                    FieldComparison(field_name="value", expected=5.0, extracted=5.0, matched=True, strategy="exact"),
                    FieldComparison(
                        field_name="label", expected="high", extracted="high", matched=True, strategy="exact"
                    ),
                ],
                True,
                id="all_matched",
            ),
            pytest.param(
                [
                    FieldComparison(field_name="value", expected=2.0, extracted=2.0, matched=True, strategy="exact"),
                    FieldComparison(
                        field_name="label", expected="low", extracted="high", matched=False, strategy="exact"
                    ),
                ],
                False,
                id="one_unmatched",
            ),
        ],
    )
    def test_passed(
        self,
        comparisons: list[FieldComparison],
        expected_passed: bool,  # noqa: FBT001
    ) -> None:
        """EvalResult.passed should be True only when every field comparison matched.

        Args:
            comparisons (list[FieldComparison]): List of FieldComparison instances to include in the result.
            expected_passed (bool): The expected boolean value of the passed property.
        """
        # Arrange
        case = EvalCase(
            input_messages=[HumanMessage(content="classify")],
            expected_result=SampleResult(value=1.0, label="low"),
        )

        # Act
        result = EvalResult(
            eval_case=case,
            extracted_result=SampleResult(value=1.0, label="low"),
            field_comparisons=comparisons,
            agent_response="extracted output",
        )

        # Assert
        assert result.passed is expected_passed

    def test_failed_fields_returns_only_non_matching_comparisons(self) -> None:
        """EvalResult.failed_fields should contain only comparisons where matched=False."""
        # Arrange
        case = EvalCase(
            input_messages=[HumanMessage(content="mixed results")],
            expected_result=SampleResult(value=10.0, label="critical"),
        )
        match_a = FieldComparison(field_name="value", expected=10.0, extracted=10.0, matched=True, strategy="exact")
        fail_b = FieldComparison(
            field_name="label", expected="critical", extracted="warning", matched=False, strategy="exact"
        )
        fail_c = FieldComparison(field_name="extra", expected=True, extracted=False, matched=False, strategy="exact")

        # Act
        result = EvalResult(
            eval_case=case,
            extracted_result=SampleResult(value=10.0, label="warning"),
            field_comparisons=[match_a, fail_b, fail_c],
            agent_response="value=10.0 label=warning extra=False",
        )

        # Assert
        with check:
            assert len(result.failed_fields) == 2
        with check:
            assert result.failed_fields[0].field_name == "label"
        with check:
            assert result.failed_fields[1].field_name == "extra"

    def test_failed_fields_is_empty_when_all_comparisons_matched(self) -> None:
        """EvalResult.failed_fields should be an empty list when all comparisons matched."""
        # Arrange
        case = EvalCase(
            input_messages=[HumanMessage(content="all match")],
            expected_result=SampleResult(value=0.1, label="low"),
        )
        comparisons = [
            FieldComparison(
                field_name="value", expected=0.1, extracted=0.1, matched=True, strategy="numeric_tolerance"
            ),
            FieldComparison(field_name="label", expected="low", extracted="low", matched=True, strategy="exact_match"),
        ]

        # Act
        result = EvalResult(
            eval_case=case,
            extracted_result=SampleResult(value=0.1, label="low"),
            field_comparisons=comparisons,
            agent_response="value=0.1 label=low",
        )

        # Assert
        assert result.failed_fields == []

    def test_failed_fields_is_empty_when_no_comparisons(self) -> None:
        """EvalResult.failed_fields should be an empty list when field_comparisons is empty."""
        # Arrange
        case = EvalCase(
            input_messages=[HumanMessage(content="no fields")],
            expected_result=SampleResult(value=1.0, label="any"),
        )

        # Act
        result = EvalResult(
            eval_case=case,
            extracted_result=SampleResult(value=1.0, label="any"),
            field_comparisons=[],
            agent_response="",
        )

        # Assert
        assert result.failed_fields == []
