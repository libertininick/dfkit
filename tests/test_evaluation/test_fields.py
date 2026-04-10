"""Tests for the evaluation field type aliases."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from pytest_check import check

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


# region Models
class FieldModel(BaseModel):
    """Minimal Pydantic model that exercises all field type aliases.

    Attributes:
        correlation (Correlation | None): Linear correlation between two variables.
        importance (Importance | None): Feature importance score.
        probability (Probability | None): Statistical probability of an outcome.
        relationship_strength (RelationshipStrength | None): Strength of relationship between variables.
        confounders (Confounders | None): Other variables that confound relationship with target.
        non_negative_metric (NonNegativeMetric | None): Non-negative measurement such as an error or loss.
        metric (Metric | None): Unbounded numeric measurement or coefficient.
        sample_count (SampleCount | None): Count of observations or samples.
        rank (Rank | None): Ordinal position in a ranked sequence.
    """

    correlation: Correlation | None = None
    importance: Importance | None = None
    probability: Probability | None = None
    relationship_strength: RelationshipStrength | None = None
    confounders: Confounders | None = None
    non_negative_metric: NonNegativeMetric | None = None
    metric: Metric | None = None
    sample_count: SampleCount | None = None
    rank: Rank | None = None


class NestedFieldModel(BaseModel):
    """Pydantic model that nests field type aliases inside generic Python structures.

    Attributes:
        correlation_by_month (dict[int, Correlation | None]): Correlation values keyed by month number.
        importance_scores (list[Importance | None]): Sequence of importance scores.
        probabilities_by_name (dict[str, Probability | None]): Probability values keyed by feature name.
        strengths (list[RelationshipStrength | None]): Sequence of relationship strength labels.
        confounder_groups (dict[str, Confounders | None]): Named groups of confounder variable lists.
        metric_history (list[NonNegativeMetric | None]): Sequence of non-negative metric values over time.
        metrics_by_test (dict[str, Metric | None]): Metric values keyed by test name.
        sample_counts (list[SampleCount | None]): Sequence of sample counts.
        ranks_by_feature (dict[str, Rank | None]): Rank values keyed by feature name.
    """

    correlation_by_month: dict[int, Correlation | None] = {}
    importance_scores: list[Importance | None] = []
    probabilities_by_name: dict[str, Probability | None] = {}
    strengths: list[RelationshipStrength | None] = []
    confounder_groups: dict[str, Confounders | None] = {}
    metric_history: list[NonNegativeMetric | None] = []
    metrics_by_test: dict[str, Metric | None] = {}
    sample_counts: list[SampleCount | None] = []
    ranks_by_feature: dict[str, Rank | None] = {}


# endregion


class TestCorrelation:
    """Tests for the Correlation field type alias (float | None, -1 ≤ x ≤ 1)."""

    @pytest.mark.parametrize(
        "value",
        [-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    def test_valid_value_accepted(self, value: float) -> None:
        """Correlation should accept any float in [-1, 1].

        Args:
            value (float): A valid correlation value within the allowed range.
        """
        # Arrange / Act
        model = FieldModel(correlation=value)

        # Assert
        assert model.correlation == pytest.approx(value)

    def test_none_accepted(self) -> None:
        """Correlation should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(correlation=None)

        # Assert
        assert model.correlation is None

    @pytest.mark.parametrize(
        "value",
        [-1.1, 1.1, -2.0, 2.0, float("inf"), float("-inf"), float("nan")],
    )
    def test_out_of_range_value_raises_validation_error(self, value: float) -> None:
        """Correlation should reject values outside [-1, 1] with a ValidationError.

        Args:
            value (float): An out-of-range value that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(correlation=value)


class TestImportance:
    """Tests for the Importance field type alias (float | None, 0 ≤ x ≤ 1)."""

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.25, 0.5, 0.8, 1.0],
    )
    def test_valid_value_accepted(self, value: float) -> None:
        """Importance should accept any float in [0, 1].

        Args:
            value (float): A valid importance score within the allowed range.
        """
        # Arrange / Act
        model = FieldModel(importance=value)

        # Assert
        assert model.importance == pytest.approx(value)

    def test_none_accepted(self) -> None:
        """Importance should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(importance=None)

        # Assert
        assert model.importance is None

    @pytest.mark.parametrize(
        "value",
        [-0.1, 1.1, -1.0, 2.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_out_of_range_value_raises_validation_error(self, value: float) -> None:
        """Importance should reject values outside [0, 1] with a ValidationError.

        Args:
            value (float): An out-of-range value that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(importance=value)


class TestProbability:
    """Tests for the Probability field type alias (float | None, 0 ≤ x ≤ 1)."""

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.1, 0.5, 0.99, 1.0],
    )
    def test_valid_value_accepted(self, value: float) -> None:
        """Probability should accept any float in [0, 1].

        Args:
            value (float): A valid probability value within the allowed range.
        """
        # Arrange / Act
        model = FieldModel(probability=value)

        # Assert
        assert model.probability == pytest.approx(value)

    def test_none_accepted(self) -> None:
        """Probability should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(probability=None)

        # Assert
        assert model.probability is None

    @pytest.mark.parametrize(
        "value",
        [-0.01, 1.01, -1.0, 5.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_out_of_range_value_raises_validation_error(self, value: float) -> None:
        """Probability should reject values outside [0, 1] with a ValidationError.

        Args:
            value (float): An out-of-range value that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(probability=value)


class TestRelationshipStrength:
    """Tests for the RelationshipStrength field type alias (Literal | None)."""

    @pytest.mark.parametrize(
        "value",
        ["Strong", "Moderate", "Weak", "Negligible"],
    )
    def test_valid_literal_accepted(self, value: str) -> None:
        """RelationshipStrength should accept all four valid literal strings.

        Args:
            value (str): One of the four valid RelationshipStrength literal values.
        """
        # Arrange / Act
        model = FieldModel(relationship_strength=value)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.relationship_strength == value

    def test_none_accepted(self) -> None:
        """RelationshipStrength should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(relationship_strength=None)

        # Assert
        assert model.relationship_strength is None

    @pytest.mark.parametrize(
        "value",
        ["strong", "STRONG", "moderate", "None", "invalid", "", "weak"],
    )
    def test_invalid_literal_raises_validation_error(self, value: str) -> None:
        """RelationshipStrength should reject strings that are not one of the four valid literals.

        Args:
            value (str): An invalid string that should be rejected by Pydantic validation.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(relationship_strength=value)  # ty: ignore[invalid-argument-type]


class TestConfounders:
    """Tests for the Confounders field type alias (list[str] | None)."""

    def test_list_of_strings_accepted(self) -> None:
        """Confounders should accept a non-empty list of strings."""
        # Arrange / Act
        model = FieldModel(confounders=["age", "income", "tenure"])

        # Assert
        with check:
            assert model.confounders == ["age", "income", "tenure"]

    def test_empty_list_accepted(self) -> None:
        """Confounders should accept an empty list to represent 'no confounders exist'."""
        # Arrange / Act
        model = FieldModel(confounders=[])

        # Assert
        assert model.confounders == []

    def test_single_element_list_accepted(self) -> None:
        """Confounders should accept a list containing a single string."""
        # Arrange / Act
        model = FieldModel(confounders=["region"])

        # Assert
        assert model.confounders == ["region"]

    def test_none_accepted(self) -> None:
        """Confounders should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(confounders=None)

        # Assert
        assert model.confounders is None

    def test_list_of_non_strings_raises_validation_error(self) -> None:
        """Confounders should reject a list whose elements are not strings."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(confounders=[1, 2, 3])  # ty: ignore[invalid-argument-type]

    def test_empty_string_element_raises_validation_error(self) -> None:
        """Confounders should reject a list that contains an empty string."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(confounders=[""])


class TestNonNegativeMetric:
    """Tests for the NonNegativeMetric field type alias (float | None, x ≥ 0)."""

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.5, 1.0, 100.0, 1e6],
    )
    def test_valid_value_accepted(self, value: float) -> None:
        """NonNegativeMetric should accept any non-negative float.

        Args:
            value (float): A valid non-negative metric value.
        """
        # Arrange / Act
        model = FieldModel(non_negative_metric=value)

        # Assert
        assert model.non_negative_metric == pytest.approx(value)

    def test_none_accepted(self) -> None:
        """NonNegativeMetric should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(non_negative_metric=None)

        # Assert
        assert model.non_negative_metric is None

    @pytest.mark.parametrize(
        "value",
        [-0.1, -1.0, -100.0, float("-inf"), float("nan"), float("inf")],
    )
    def test_negative_value_raises_validation_error(self, value: float) -> None:
        """NonNegativeMetric should reject negative values with a ValidationError.

        Args:
            value (float): A negative value that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(non_negative_metric=value)


class TestMetric:
    """Tests for the Metric field type alias (float | None, unbounded)."""

    @pytest.mark.parametrize(
        "value",
        [-1e6, -1.0, 0.0, 1.0, 1e6, 1.96, -2.576],
    )
    def test_valid_value_accepted(self, value: float) -> None:
        """Metric should accept any finite float value, positive or negative.

        Args:
            value (float): A valid metric value.
        """
        # Arrange / Act
        model = FieldModel(metric=value)

        # Assert
        assert model.metric == pytest.approx(value)

    def test_none_accepted(self) -> None:
        """Metric should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(metric=None)

        # Assert
        assert model.metric is None

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan")],
    )
    def test_non_finite_value_raises_validation_error(self, value: float) -> None:
        """Metric should reject non-finite float values (inf, -inf, NaN).

        Args:
            value (float): A non-finite float value that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(metric=value)


class TestSampleCount:
    """Tests for the SampleCount field type alias (int | None, x ≥ 0)."""

    @pytest.mark.parametrize(
        "value",
        [0, 1, 2, 10, 100, 1000],
    )
    def test_valid_value_accepted(self, value: int) -> None:
        """SampleCount should accept any integer greater than or equal to 0.

        Args:
            value (int): A valid sample count value of at least 0.
        """
        # Arrange / Act
        model = FieldModel(sample_count=value)

        # Assert
        assert model.sample_count == value

    def test_none_accepted(self) -> None:
        """SampleCount should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(sample_count=None)

        # Assert
        assert model.sample_count is None

    @pytest.mark.parametrize(
        "value",
        [-1, -10],
    )
    def test_out_of_range_value_raises_validation_error(self, value: int) -> None:
        """SampleCount should reject negative integers with a ValidationError.

        Args:
            value (int): A negative integer that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(sample_count=value)


class TestRank:
    """Tests for the Rank field type alias (int | None, x ≥ 1)."""

    @pytest.mark.parametrize(
        "value",
        [1, 2, 3, 10, 100],
    )
    def test_valid_value_accepted(self, value: int) -> None:
        """Rank should accept any integer greater than or equal to 1.

        Args:
            value (int): A valid rank value of at least 1.
        """
        # Arrange / Act
        model = FieldModel(rank=value)

        # Assert
        assert model.rank == value

    def test_none_accepted(self) -> None:
        """Rank should accept None explicitly."""
        # Arrange / Act
        model = FieldModel(rank=None)

        # Assert
        assert model.rank is None

    @pytest.mark.parametrize(
        "value",
        [0, -1, -10],
    )
    def test_out_of_range_value_raises_validation_error(self, value: int) -> None:
        """Rank should reject zero and negative integers with a ValidationError.

        Args:
            value (int): A zero or negative integer that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            FieldModel(rank=value)


class TestNestedInStructures:
    """Tests verifying field type aliases enforce constraints when nested inside containers."""

    # region valid round-trips

    def test_correlation_by_month_valid_values_round_trip(self) -> None:
        """A representative correlation value inside a dict should be stored without modification."""
        # Arrange
        data = {1: 0.5}

        # Act
        model = NestedFieldModel(correlation_by_month=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.correlation_by_month[1] == pytest.approx(0.5)

    def test_importance_scores_valid_values_round_trip(self) -> None:
        """A representative importance score inside a list should be stored without modification."""
        # Arrange
        data = [0.5]

        # Act
        model = NestedFieldModel(importance_scores=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.importance_scores[0] == pytest.approx(0.5)

    def test_probabilities_by_name_valid_values_round_trip(self) -> None:
        """A representative probability value inside a dict should be stored without modification."""
        # Arrange
        data = {"feature_a": 0.5}

        # Act
        model = NestedFieldModel(probabilities_by_name=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.probabilities_by_name["feature_a"] == pytest.approx(0.5)

    def test_strengths_valid_literal_round_trips(self) -> None:
        """A representative RelationshipStrength literal should be accepted inside a list."""
        # Arrange
        value = "Moderate"

        # Act
        model = NestedFieldModel(strengths=[value])

        # Assert
        assert model.strengths[0] == value

    def test_confounder_groups_valid_values_round_trip(self) -> None:
        """A representative Confounders value inside a dict should be stored without modification."""
        # Arrange
        data = {"group_a": ["age", "income"]}

        # Act
        model = NestedFieldModel(confounder_groups=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.confounder_groups["group_a"] == ["age", "income"]

    def test_metric_history_valid_values_round_trip(self) -> None:
        """A representative NonNegativeMetric value inside a list should be stored without modification."""
        # Arrange
        data = [1.2]

        # Act
        model = NestedFieldModel(metric_history=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.metric_history[0] == pytest.approx(1.2)

    def test_metrics_by_test_valid_values_round_trip(self) -> None:
        """A representative Metric value inside a dict should be stored without modification."""
        # Arrange
        data = {"t_test": 1.96}

        # Act
        model = NestedFieldModel(metrics_by_test=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.metrics_by_test["t_test"] == pytest.approx(1.96)

    def test_sample_counts_valid_values_round_trip(self) -> None:
        """A representative SampleCount value inside a list should be stored without modification."""
        # Arrange
        data = [10]

        # Act
        model = NestedFieldModel(sample_counts=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.sample_counts[0] == 10

    def test_ranks_by_feature_valid_values_round_trip(self) -> None:
        """A representative Rank value inside a dict should be stored without modification."""
        # Arrange
        data = {"feature_a": 2}

        # Act
        model = NestedFieldModel(ranks_by_feature=data)  # ty: ignore[invalid-argument-type]

        # Assert
        assert model.ranks_by_feature["feature_a"] == 2

    # endregion

    # region validation still applies inside containers

    @pytest.mark.parametrize(
        "bad_value",
        [-1.1, 1.1, 2.0, -2.0, float("nan")],
    )
    def test_correlation_by_month_out_of_range_raises_validation_error(self, bad_value: float) -> None:
        """Correlation constraints should be enforced even when the value is inside a dict.

        Args:
            bad_value (float): A correlation value outside the allowed [-1, 1] range.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(correlation_by_month={1: bad_value})

    @pytest.mark.parametrize(
        "bad_value",
        [-0.1, 1.1, -1.0, 2.0, float("nan")],
    )
    def test_importance_scores_out_of_range_raises_validation_error(self, bad_value: float) -> None:
        """Importance constraints should be enforced even when the value is inside a list.

        Args:
            bad_value (float): An importance score outside the allowed [0, 1] range.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(importance_scores=[bad_value])

    @pytest.mark.parametrize(
        "bad_value",
        [-0.01, 1.01, -1.0, 5.0, float("nan")],
    )
    def test_probabilities_by_name_out_of_range_raises_validation_error(self, bad_value: float) -> None:
        """Probability constraints should be enforced even when the value is inside a dict.

        Args:
            bad_value (float): A probability value outside the allowed [0, 1] range.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(probabilities_by_name={"feat": bad_value})

    @pytest.mark.parametrize(
        "bad_value",
        ["strong", "STRONG", "moderate", "invalid", "", "weak"],
    )
    def test_strengths_invalid_literal_raises_validation_error(self, bad_value: str) -> None:
        """RelationshipStrength constraints should be enforced even when the value is inside a list.

        Args:
            bad_value (str): A string that is not one of the valid RelationshipStrength literals.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(strengths=[bad_value])  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        "bad_value",
        [-0.1, -1.0, -100.0, float("nan")],
    )
    def test_metric_history_negative_value_raises_validation_error(self, bad_value: float) -> None:
        """NonNegativeMetric constraints should be enforced even when the value is inside a list.

        Args:
            bad_value (float): A negative value that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(metric_history=[bad_value])

    @pytest.mark.parametrize(
        "bad_value",
        [-1, -10],
    )
    def test_sample_counts_out_of_range_raises_validation_error(self, bad_value: int) -> None:
        """SampleCount constraints should be enforced even when the value is inside a list.

        Args:
            bad_value (int): A negative integer that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(sample_counts=[bad_value])

    @pytest.mark.parametrize(
        "bad_value",
        [0, -1, -10],
    )
    def test_ranks_by_feature_out_of_range_raises_validation_error(self, bad_value: int) -> None:
        """Rank constraints should be enforced even when the value is inside a dict.

        Args:
            bad_value (int): A zero or negative integer that should be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            NestedFieldModel(ranks_by_feature={"feat": bad_value})

    # endregion

    # region none values inside containers

    def test_correlation_by_month_none_value_accepted(self) -> None:
        """A None correlation value inside a dict should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(correlation_by_month={1: None})

        # Assert
        assert model.correlation_by_month[1] is None

    def test_importance_scores_none_value_accepted(self) -> None:
        """A None importance value inside a list should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(importance_scores=[None])

        # Assert
        assert model.importance_scores[0] is None

    def test_probabilities_by_name_none_value_accepted(self) -> None:
        """A None probability value inside a dict should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(probabilities_by_name={"feat": None})

        # Assert
        assert model.probabilities_by_name["feat"] is None

    def test_strengths_none_value_accepted(self) -> None:
        """A None RelationshipStrength value inside a list should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(strengths=[None])

        # Assert
        assert model.strengths[0] is None

    def test_confounder_groups_none_value_accepted(self) -> None:
        """A None Confounders value inside a dict should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(confounder_groups={"g": None})

        # Assert
        assert model.confounder_groups["g"] is None

    def test_metric_history_none_value_accepted(self) -> None:
        """A None NonNegativeMetric value inside a list should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(metric_history=[None])

        # Assert
        assert model.metric_history[0] is None

    def test_metrics_by_test_none_value_accepted(self) -> None:
        """A None Metric value inside a dict should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(metrics_by_test={"t_test": None})

        # Assert
        assert model.metrics_by_test["t_test"] is None

    def test_sample_counts_none_value_accepted(self) -> None:
        """A None SampleCount value inside a list should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(sample_counts=[None])

        # Assert
        assert model.sample_counts[0] is None

    def test_ranks_by_feature_none_value_accepted(self) -> None:
        """A None Rank value inside a dict should be accepted without error."""
        # Arrange / Act
        model = NestedFieldModel(ranks_by_feature={"feat": None})

        # Assert
        assert model.ranks_by_feature["feat"] is None

    # endregion

    # region empty containers

    def test_empty_containers_accepted(self) -> None:
        """All container fields should accept empty collections as their default values."""
        # Arrange / Act
        model = NestedFieldModel()

        # Assert
        with check:
            assert model.correlation_by_month == {}
        with check:
            assert model.importance_scores == []
        with check:
            assert model.probabilities_by_name == {}
        with check:
            assert model.strengths == []
        with check:
            assert model.confounder_groups == {}
        with check:
            assert model.metric_history == []
        with check:
            assert model.metrics_by_test == {}
        with check:
            assert model.sample_counts == []
        with check:
            assert model.ranks_by_feature == {}

    # endregion
