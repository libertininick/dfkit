"""Tests for the decision tree module: Pydantic models and preprocessing pipeline functions."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from pytest_check import check

from dfkit.decision_tree_module import (
    ClassificationRule,
    DecisionTreeResult,
    DecisionTreeRule,
    Predicate,
    RegressionRule,
    _classify_column,
    _detect_task_type,
    _encode_features,
    _encode_target,
    _ExcludedFeature,
    _filter_features,
)


class TestPredicate:
    """Tests for the Predicate model: construction, repr/str, eval, and serialization."""

    # ------------------------------------------------------------------
    # Construction tests
    # ------------------------------------------------------------------

    def test_numeric_predicate_construction_sets_correct_fields(self) -> None:
        """Numeric predicate should populate all three fields from constructor arguments."""
        # Arrange / Act
        predicate = Predicate(variable="tenure_months", operator=">", value=6.0)

        # Assert
        with check:
            assert predicate.variable == "tenure_months"
        with check:
            assert predicate.operator == ">"
        with check:
            assert predicate.value == 6.0

    def test_string_predicate_construction_sets_correct_fields(self) -> None:
        """String-value predicate should populate all three fields from constructor arguments."""
        # Arrange / Act
        predicate = Predicate(variable="status", operator="==", value="active")

        # Assert
        with check:
            assert predicate.variable == "status"
        with check:
            assert predicate.operator == "=="
        with check:
            assert predicate.value == "active"

    def test_set_predicate_construction_sets_correct_fields(self) -> None:
        """Set-value predicate should populate all three fields including the set value."""
        # Arrange / Act
        predicate = Predicate(variable="plan_type", operator="in", value={"basic", "standard"})

        # Assert
        with check:
            assert predicate.variable == "plan_type"
        with check:
            assert predicate.operator == "in"
        with check:
            assert predicate.value == {"basic", "standard"}

    @pytest.mark.parametrize(
        "operator",
        [">", ">=", "!=", "==", "<", "<=", "in", "not in"],
    )
    def test_all_operators_accepted(self, operator: str) -> None:
        """Every operator in the Literal set should be accepted without a ValidationError.

        Args:
            operator (str): One of the eight valid comparison operators to test.
        """
        # Arrange
        is_membership_operator = operator in {"in", "not in"}
        predicate_value: float | str | set[float] | set[str] = {"a", "b"} if is_membership_operator else 1.0

        # Act
        predicate = Predicate(variable="x", operator=operator, value=predicate_value)  # type: ignore[arg-type]

        # Assert
        assert predicate.operator == operator

    def test_invalid_operator_rejected_raises_validation_error(self) -> None:
        """An operator not in the Literal set should raise a ValidationError."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            Predicate(variable="x", operator="~=", value=1.0)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Operator-value compatibility validation tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("operator", ["in", "not in"])
    def test_membership_operator_with_scalar_value_raises_validation_error(self, operator: str) -> None:
        """Membership operators paired with a numeric scalar should raise a ValidationError.

        The model validator must reject `"in"` and `"not in"` when the value
        is a plain float, because membership tests require a set of candidates.

        Args:
            operator (str): A membership operator (`"in"` or `"not in"`) to test.
        """
        # Arrange
        scalar_value = 5.0

        # Act / Assert
        with pytest.raises(ValidationError):
            Predicate(variable="x", operator=operator, value=scalar_value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("operator", ["in", "not in"])
    def test_membership_operator_with_string_scalar_raises_validation_error(self, operator: str) -> None:
        """Membership operators paired with a string scalar should raise a ValidationError.

        A bare string is not a set; the model validator must reject it when the
        operator is `"in"` or `"not in"`.

        Args:
            operator (str): A membership operator (`"in"` or `"not in"`) to test.
        """
        # Arrange
        string_value = "hello"

        # Act / Assert
        with pytest.raises(ValidationError):
            Predicate(variable="x", operator=operator, value=string_value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("operator", [">", ">=", "<", "<=", "==", "!="])
    def test_scalar_operator_with_set_value_raises_validation_error(self, operator: str) -> None:
        """Scalar operators paired with a set value should raise a ValidationError.

        Threshold comparisons (`>`, `>=`, `<`, `<=`, `==`, `!=`) require
        a scalar value; the model validator must reject a set.

        Args:
            operator (str): A scalar comparison operator to test.
        """
        # Arrange
        set_value: set[str] = {"a", "b"}

        # Act / Assert
        with pytest.raises(ValidationError):
            Predicate(variable="x", operator=operator, value=set_value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("operator", ["in", "not in"])
    def test_membership_operator_with_set_value_accepted(self, operator: str) -> None:
        """Membership operators paired with a set value should be accepted without error.

        Verifies that constructing a predicate with `"in"` or `"not in"`
        and a `set` value succeeds and preserves the set on the model.

        Args:
            operator (str): A membership operator (`"in"` or `"not in"`) to test.
        """
        # Arrange
        set_value: set[str] = {"x", "y"}

        # Act
        predicate = Predicate(variable="category", operator=operator, value=set_value)  # type: ignore[arg-type]

        # Assert
        with check:
            assert predicate.value == {"x", "y"}

    def test_scalar_operator_with_numeric_value_accepted(self) -> None:
        """A scalar operator paired with a numeric value should be accepted without error.

        Verifies that `">"` with a float threshold passes the operator-value
        compatibility validator and the value is preserved on the model.
        """
        # Arrange / Act
        predicate = Predicate(variable="score", operator=">", value=5.0)

        # Assert
        with check:
            assert predicate.value == 5.0

    def test_scalar_operator_with_string_value_accepted(self) -> None:
        """A scalar operator paired with a string value should be accepted without error.

        Verifies that `"=="` with a string scalar passes the operator-value
        compatibility validator and the value is preserved on the model.
        """
        # Arrange / Act
        predicate = Predicate(variable="status", operator="==", value="active")

        # Assert
        with check:
            assert predicate.value == "active"

    # ------------------------------------------------------------------
    # __str__ tests
    # ------------------------------------------------------------------

    def test_str_numeric_produces_expected_string(self) -> None:
        """str() of a numeric predicate should format as 'variable operator value'."""
        # Arrange
        predicate = Predicate(variable="tenure_months", operator=">", value=6.0)

        # Act
        representation = str(predicate)

        # Assert
        assert representation == "tenure_months > 6.0"

    def test_str_string_equality_produces_readable_output(self) -> None:
        """str() of a string equality predicate should include the string value."""
        # Arrange
        predicate = Predicate(variable="status", operator="==", value="active")

        # Act
        representation = str(predicate)

        # Assert
        assert representation == "status == active"

    def test_str_set_membership_produces_sorted_set_contents(self) -> None:
        """str() of an 'in' predicate should list set members in sorted order."""
        # Arrange
        predicate = Predicate(variable="plan_type", operator="in", value={"basic", "standard"})

        # Act
        representation = str(predicate)

        # Assert — sorted alphabetically: basic before standard
        assert representation == "plan_type in {basic, standard}"

    # ------------------------------------------------------------------
    # Eval tests
    # ------------------------------------------------------------------

    def test_eval_greater_than_returns_correct_truth_values(self) -> None:
        """Eval with '>' should return True when x exceeds the threshold and False otherwise."""
        # Arrange
        predicate = Predicate(variable="x", operator=">", value=5.0)

        # Act / Assert
        with check:
            assert predicate.eval(6.0) is True
        with check:
            assert predicate.eval(4.0) is False

    def test_eval_greater_equal_boundary_returns_true(self) -> None:
        """Eval with '>=' at the exact boundary value should return True."""
        # Arrange
        predicate = Predicate(variable="x", operator=">=", value=5.0)

        # Act / Assert
        with check:
            assert predicate.eval(5.0) is True
        with check:
            assert predicate.eval(4.9) is False

    def test_eval_less_than_returns_correct_truth_values(self) -> None:
        """Eval with '<' should return True when x is below the threshold and False otherwise."""
        # Arrange
        predicate = Predicate(variable="x", operator="<", value=5.0)

        # Act / Assert
        with check:
            assert predicate.eval(3.0) is True
        with check:
            assert predicate.eval(5.0) is False

    def test_eval_less_equal_boundary_returns_true(self) -> None:
        """Eval with '<=' at the exact boundary value should return True."""
        # Arrange
        predicate = Predicate(variable="x", operator="<=", value=5.0)

        # Act / Assert
        with check:
            assert predicate.eval(5.0) is True
        with check:
            assert predicate.eval(5.1) is False

    def test_eval_equal_matching_value_returns_true(self) -> None:
        """Eval with '==' should return True when x exactly matches the threshold."""
        # Arrange
        predicate = Predicate(variable="x", operator="==", value=5.0)

        # Act / Assert
        with check:
            assert predicate.eval(5.0) is True
        with check:
            assert predicate.eval(4.0) is False

    def test_eval_not_equal_differing_value_returns_true(self) -> None:
        """Eval with '!=' should return True when x differs from the threshold."""
        # Arrange
        predicate = Predicate(variable="x", operator="!=", value=5.0)

        # Act / Assert
        with check:
            assert predicate.eval(3.0) is True
        with check:
            assert predicate.eval(5.0) is False

    def test_eval_in_set_returns_correct_truth_values(self) -> None:
        """Eval with 'in' should return True when x is in the set and False otherwise."""
        # Arrange
        predicate = Predicate(variable="x", operator="in", value={"a", "b"})

        # Act / Assert
        with check:
            assert predicate.eval("a") is True
        with check:
            assert predicate.eval("c") is False

    def test_eval_not_in_set_absent_value_returns_true(self) -> None:
        """Eval with 'not in' should return True when x is absent from the set."""
        # Arrange
        predicate = Predicate(variable="x", operator="not in", value={"a", "b"})

        # Act / Assert
        with check:
            assert predicate.eval("c") is True
        with check:
            assert predicate.eval("a") is False

    def test_eval_in_set_float_membership_returns_correct_truth_values(self) -> None:
        """Eval with 'in' against a float set should return True for members and False for non-members."""
        # Arrange
        predicate = Predicate(variable="x", operator="in", value={1.0, 2.0})

        # Act / Assert
        with check:
            assert predicate.eval(1.0) is True
        with check:
            assert predicate.eval(3.0) is False

    def test_eval_string_equality_returns_correct_truth_values(self) -> None:
        """Eval with '==' against a string value should return True for a matching string and False otherwise."""
        # Arrange
        predicate = Predicate(variable="status", operator="==", value="active")

        # Act / Assert
        with check:
            assert predicate.eval("active") is True
        with check:
            assert predicate.eval("inactive") is False

    # ------------------------------------------------------------------
    # Serialization tests
    # ------------------------------------------------------------------

    def test_serialization_roundtrip_numeric_preserves_all_fields(self) -> None:
        """model_dump / model_validate roundtrip should preserve all fields for a numeric predicate."""
        # Arrange
        original = Predicate(variable="tenure_months", operator=">", value=6.0)

        # Act
        restored = Predicate.model_validate(original.model_dump())

        # Assert
        with check:
            assert restored.variable == original.variable
        with check:
            assert restored.operator == original.operator
        with check:
            assert restored.value == original.value

    def test_serialization_roundtrip_set_preserves_set_values(self) -> None:
        """model_dump / model_validate roundtrip should preserve set values for a membership predicate.

        Pydantic serializes sets to lists; model_validate must reconstruct the set correctly.
        """
        # Arrange
        original = Predicate(variable="plan_type", operator="in", value={"basic", "standard"})

        # Act
        restored = Predicate.model_validate(original.model_dump())

        # Assert
        with check:
            assert restored.variable == original.variable
        with check:
            assert restored.operator == original.operator
        with check:
            assert restored.value == original.value

    def test_json_roundtrip_preserves_all_fields(self) -> None:
        """model_dump_json / model_validate_json roundtrip should preserve all fields."""
        # Arrange
        original = Predicate(variable="score", operator=">=", value=0.75)

        # Act
        restored = Predicate.model_validate_json(original.model_dump_json())

        # Assert
        with check:
            assert restored.variable == original.variable
        with check:
            assert restored.operator == original.operator
        with check:
            assert restored.value == original.value


class TestDecisionTreeRule:
    """Tests for the ClassificationRule and RegressionRule discriminated union."""

    def test_classification_rule_construction_sets_correct_fields(self) -> None:
        """Classification rule should populate all fields from constructor arguments.

        Verifies that a rule built from a churn-prediction leaf node populates
        every field correctly via the ClassificationRule concrete type.
        """
        # Arrange
        predicates = [
            Predicate(variable="tenure_months", operator=">", value=12.0),
            Predicate(variable="support_tickets", operator="<=", value=2.0),
            Predicate(variable="monthly_charges", operator="<=", value=75.0),
        ]

        # Act
        rule = ClassificationRule(
            task_type="classification",
            predicates=predicates,
            prediction="retained",
            samples=342,
            confidence=0.91,
        )

        # Assert
        with check:
            assert rule.task_type == "classification"
        with check:
            assert rule.predicates == predicates
        with check:
            assert rule.prediction == "retained"
        with check:
            assert rule.samples == 342
        with check:
            assert rule.confidence == 0.91

    def test_regression_rule_construction_sets_correct_fields(self) -> None:
        """Regression rule should populate all fields from constructor arguments.

        Verifies that a rule built from a salary-prediction leaf node populates
        every field correctly via the RegressionRule concrete type.
        """
        # Arrange
        predicates = [
            Predicate(variable="years_experience", operator=">", value=5.0),
            Predicate(variable="education_level", operator="<=", value=2.0),
            Predicate(variable="department", operator="==", value="Engineering"),
        ]

        # Act
        rule = RegressionRule(
            task_type="regression",
            predicates=predicates,
            prediction=112_500.0,
            samples=87,
            std=18_430.75,
        )

        # Assert
        with check:
            assert rule.task_type == "regression"
        with check:
            assert rule.predicates == predicates
        with check:
            assert rule.prediction == 112_500.0
        with check:
            assert rule.samples == 87
        with check:
            assert rule.std == 18_430.75

    @pytest.mark.parametrize(
        ("rule_class", "kwargs", "expected_model"),
        [
            (
                ClassificationRule,
                {
                    "task_type": "classification",
                    "predicates": [
                        Predicate(variable="credit_score", operator=">", value=700.0),
                        Predicate(variable="loan_amount", operator="<=", value=50000.0),
                    ],
                    "prediction": "approved",
                    "samples": 519,
                    "confidence": 0.84,
                },
                ClassificationRule,
            ),
            (
                RegressionRule,
                {
                    "task_type": "regression",
                    "predicates": [
                        Predicate(variable="years_experience", operator=">", value=8.0),
                        Predicate(variable="management_level", operator="==", value="True"),
                    ],
                    "prediction": 148_300.0,
                    "samples": 59,
                    "std": 22_900.25,
                },
                RegressionRule,
            ),
        ],
    )
    def test_rule_serialization_roundtrip_produces_identical_object(
        self,
        rule_class: type[ClassificationRule | RegressionRule],
        kwargs: dict[str, Any],
        expected_model: type[ClassificationRule | RegressionRule],
    ) -> None:
        """Serializing a rule to a dict and validating it back should produce an equal model.

        Exercises model_dump followed by model_validate to confirm that no field
        is lost or mutated during a round-trip through the plain-Python representation,
        for both ClassificationRule and RegressionRule.

        Args:
            rule_class (type[ClassificationRule | RegressionRule]): The rule class to construct.
            kwargs (dict[str, Any]): Constructor keyword arguments for the rule.
            expected_model (type[ClassificationRule | RegressionRule]): The model class used for validation.
        """
        # Arrange
        original = rule_class(**kwargs)

        # Act
        restored = expected_model.model_validate(original.model_dump())

        # Assert
        assert restored == original

    def test_classification_rule_with_empty_predicates_represents_single_leaf(self) -> None:
        """A classification rule with no predicates should be accepted as a valid single-leaf tree."""
        # Act
        rule = ClassificationRule(
            task_type="classification",
            predicates=[],
            prediction="no_churn",
            samples=1000,
            confidence=0.95,
        )

        # Assert
        with check:
            assert rule.predicates == []
        with check:
            assert rule.prediction == "no_churn"
        with check:
            assert rule.samples == 1000

    @pytest.mark.parametrize(
        ("confidence", "predicates", "prediction", "samples"),
        [
            (
                0.0,
                [Predicate(variable="account_age_days", operator="<=", value=30.0)],
                "high_risk",
                45,
            ),
            (
                1.0,
                [
                    Predicate(variable="account_age_days", operator=">", value=365.0),
                    Predicate(variable="payment_failures", operator="==", value=0.0),
                ],
                "low_risk",
                312,
            ),
        ],
    )
    def test_classification_rule_with_boundary_confidence_values(
        self,
        confidence: float,
        predicates: list[Predicate],
        prediction: str,
        samples: int,
    ) -> None:
        """Confidence values of exactly 0.0 and 1.0 should both be accepted on a ClassificationRule.

        Args:
            confidence (float): The boundary confidence value to test (0.0 or 1.0).
            predicates (list[Predicate]): Path predicates for the rule.
            prediction (str): Predicted class label for the rule.
            samples (int): Number of training samples at this leaf.
        """
        # Act
        rule = ClassificationRule(
            task_type="classification",
            predicates=predicates,
            prediction=prediction,
            samples=samples,
            confidence=confidence,
        )

        # Assert
        with check:
            assert rule.confidence == confidence

    def test_regression_rule_with_zero_std(self) -> None:
        """A regression rule with std=0.0 should be accepted as a valid edge case."""
        # Act
        rule = RegressionRule(
            task_type="regression",
            predicates=[Predicate(variable="contract_type", operator="==", value="fixed_rate")],
            prediction=50_000.0,
            samples=12,
            std=0.0,
        )

        # Assert
        with check:
            assert rule.std == 0.0

    def test_classification_rule_rejects_missing_confidence_raises_validation_error(self) -> None:
        """A ClassificationRule constructed without confidence should raise a ValidationError.

        Confidence is a required field on ClassificationRule; omitting it must be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="confidence"):
            ClassificationRule(  # type: ignore[call-arg]
                task_type="classification",
                predicates=[Predicate(variable="product_category", operator="==", value="electronics")],
                prediction="medium_value",
                samples=78,
            )

    def test_regression_rule_rejects_missing_std_raises_validation_error(self) -> None:
        """A RegressionRule constructed without std should raise a ValidationError.

        std is a required field on RegressionRule; omitting it must be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="std"):
            RegressionRule(  # type: ignore[call-arg]
                task_type="regression",
                predicates=[Predicate(variable="years_experience", operator=">", value=5.0)],
                prediction=95_000.0,
                samples=120,
            )

    def test_classification_rule_rejects_negative_samples(self) -> None:
        """A classification rule with negative samples should raise a ValidationError."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="samples"):
            ClassificationRule(
                task_type="classification",
                predicates=[Predicate(variable="purchase_count", operator=">", value=5.0)],
                prediction="loyal",
                samples=-1,
                confidence=0.82,
            )

    def test_classification_rule_rejects_samples_zero(self) -> None:
        """A classification rule with samples=0 should raise a ValidationError.

        The ge=1 constraint requires at least one sample; zero must be rejected.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="samples"):
            ClassificationRule(
                task_type="classification",
                predicates=[Predicate(variable="tenure_months", operator=">", value=6.0)],
                prediction="retained",
                samples=0,
                confidence=0.75,
            )

    @pytest.mark.parametrize(
        "bad_confidence",
        [-0.1, -1.0, 1.5, 2.0],
    )
    def test_classification_rule_rejects_confidence_out_of_range(self, bad_confidence: float) -> None:
        """A classification rule with confidence outside [0.0, 1.0] should raise a ValidationError.

        Args:
            bad_confidence (float): An out-of-range confidence value to test.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            ClassificationRule(
                task_type="classification",
                predicates=[Predicate(variable="days_since_login", operator="<=", value=7.0)],
                prediction="active",
                samples=203,
                confidence=bad_confidence,
            )

    @pytest.mark.parametrize(
        "bad_std",
        [-0.001, -1.0, -100.0],
    )
    def test_regression_rule_rejects_negative_std(self, bad_std: float) -> None:
        """A regression rule with negative std should raise a ValidationError because ge=0.0 forbids it.

        Args:
            bad_std (float): A negative std value to test.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            RegressionRule(
                task_type="regression",
                predicates=[Predicate(variable="years_experience", operator=">", value=5.0)],
                prediction=95_000.0,
                samples=120,
                std=bad_std,
            )


class TestDecisionTreeResult:
    """Tests for the DecisionTreeResult model."""

    def test_classification_result_construction_sets_correct_fields(self) -> None:
        """Classification result should populate all fields from constructor arguments.

        Verifies a full churn-prediction result with two leaf rules, accuracy
        metric, and feature importance scores is constructed without error and
        every field is accessible with the expected value.
        """
        # Arrange
        rules: list[DecisionTreeRule] = [
            ClassificationRule(
                task_type="classification",
                predicates=[Predicate(variable="tenure_months", operator="<=", value=6.0)],
                prediction="churned",
                samples=210,
                confidence=0.87,
            ),
            ClassificationRule(
                task_type="classification",
                predicates=[
                    Predicate(variable="tenure_months", operator=">", value=6.0),
                    Predicate(variable="support_tickets", operator="<=", value=1.0),
                ],
                prediction="retained",
                samples=891,
                confidence=0.93,
            ),
        ]

        # Act
        tree_result = DecisionTreeResult(
            target="churn",
            task_type="classification",
            features_used=["tenure_months", "support_tickets", "monthly_charges"],
            features_excluded=["customer_id (unique identifier)", "phone_number (free-text)"],
            rules=rules,
            feature_importance={
                "tenure_months": 0.61,
                "support_tickets": 0.27,
                "monthly_charges": 0.12,
            },
            metrics={"accuracy": 0.89},
            sample_count=1101,
            depth=3,
            leaf_count=2,
        )

        # Assert
        with check:
            assert tree_result.target == "churn"
        with check:
            assert tree_result.task_type == "classification"
        with check:
            assert tree_result.features_used == ["tenure_months", "support_tickets", "monthly_charges"]
        with check:
            assert tree_result.features_excluded == [
                "customer_id (unique identifier)",
                "phone_number (free-text)",
            ]
        with check:
            assert tree_result.rules == rules
        with check:
            assert tree_result.feature_importance["tenure_months"] == 0.61
        with check:
            assert tree_result.metrics == {"accuracy": 0.89}
        with check:
            assert tree_result.sample_count == 1101
        with check:
            assert tree_result.depth == 3
        with check:
            assert tree_result.leaf_count == 2

    def test_regression_result_construction_sets_correct_fields(self) -> None:
        """Regression result should populate all fields from constructor arguments.

        Verifies a full salary-prediction result with three leaf rules, r_squared
        and rmse metrics, and feature importance scores is constructed without
        error and every field carries the expected value.
        """
        # Arrange
        rules: list[DecisionTreeRule] = [
            RegressionRule(
                task_type="regression",
                predicates=[Predicate(variable="years_experience", operator="<=", value=2.0)],
                prediction=54_200.0,
                samples=143,
                std=6_810.5,
            ),
            RegressionRule(
                task_type="regression",
                predicates=[
                    Predicate(variable="years_experience", operator=">", value=2.0),
                    Predicate(variable="years_experience", operator="<=", value=8.0),
                ],
                prediction=85_750.0,
                samples=298,
                std=11_340.0,
            ),
            RegressionRule(
                task_type="regression",
                predicates=[
                    Predicate(variable="years_experience", operator=">", value=8.0),
                    Predicate(variable="management_level", operator="==", value="True"),
                ],
                prediction=148_300.0,
                samples=59,
                std=22_900.25,
            ),
        ]

        # Act
        tree_result = DecisionTreeResult(
            target="annual_salary",
            task_type="regression",
            features_used=["years_experience", "management_level", "department_code"],
            features_excluded=["employee_id (unique identifier)"],
            rules=rules,
            feature_importance={
                "years_experience": 0.72,
                "management_level": 0.19,
                "department_code": 0.09,
            },
            metrics={"r_squared": 0.76, "rmse": 18_450.0},
            sample_count=500,
            depth=4,
            leaf_count=3,
        )

        # Assert
        with check:
            assert tree_result.target == "annual_salary"
        with check:
            assert tree_result.task_type == "regression"
        with check:
            assert tree_result.features_used == ["years_experience", "management_level", "department_code"]
        with check:
            assert tree_result.features_excluded == ["employee_id (unique identifier)"]
        with check:
            assert tree_result.rules == rules
        with check:
            assert tree_result.feature_importance["years_experience"] == 0.72
        with check:
            assert tree_result.metrics["r_squared"] == 0.76
        with check:
            assert tree_result.metrics["rmse"] == 18_450.0
        with check:
            assert tree_result.sample_count == 500
        with check:
            assert tree_result.depth == 4
        with check:
            assert tree_result.leaf_count == 3

    def test_classification_result_serialization_roundtrip_produces_identical_object(self) -> None:
        """Serializing a classification result to a dict and back should produce an equal model.

        Exercises model_dump followed by model_validate to confirm that no field
        is lost or mutated during a round-trip, including nested ClassificationRule
        objects embedded in the rules list.
        """
        # Arrange
        original = DecisionTreeResult(
            target="price_tier",
            task_type="classification",
            features_used=["sqft", "bedrooms", "neighbourhood_score"],
            features_excluded=["listing_url (free-text)"],
            rules=[
                ClassificationRule(
                    task_type="classification",
                    predicates=[
                        Predicate(variable="sqft", operator="<=", value=800.0),
                        Predicate(variable="neighbourhood_score", operator="<=", value=5.0),
                    ],
                    prediction="budget",
                    samples=405,
                    confidence=0.78,
                ),
                ClassificationRule(
                    task_type="classification",
                    predicates=[
                        Predicate(variable="sqft", operator=">", value=800.0),
                        Predicate(variable="bedrooms", operator=">=", value=3.0),
                    ],
                    prediction="premium",
                    samples=317,
                    confidence=0.88,
                ),
            ],
            feature_importance={"sqft": 0.55, "bedrooms": 0.30, "neighbourhood_score": 0.15},
            metrics={"accuracy": 0.83},
            sample_count=722,
            depth=2,
            leaf_count=2,
        )

        # Act
        restored = DecisionTreeResult.model_validate(original.model_dump())

        # Assert
        assert restored == original

    def test_regression_result_serialization_roundtrip_produces_identical_object(self) -> None:
        """Serializing a regression result to a dict and back should produce an equal model.

        Exercises model_dump followed by model_validate to confirm that no field
        is lost or mutated during a round-trip, including nested RegressionRule
        objects embedded in the rules list.
        """
        # Arrange
        original = DecisionTreeResult(
            target="house_price",
            task_type="regression",
            features_used=["sqft", "bedrooms", "garage_spaces"],
            features_excluded=["listing_id (unique identifier)"],
            rules=[
                RegressionRule(
                    task_type="regression",
                    predicates=[Predicate(variable="sqft", operator="<=", value=1200.0)],
                    prediction=285_000.0,
                    samples=230,
                    std=42_500.0,
                ),
                RegressionRule(
                    task_type="regression",
                    predicates=[
                        Predicate(variable="sqft", operator=">", value=1200.0),
                        Predicate(variable="bedrooms", operator=">=", value=3.0),
                    ],
                    prediction=520_000.0,
                    samples=180,
                    std=78_300.0,
                ),
            ],
            feature_importance={"sqft": 0.60, "bedrooms": 0.25, "garage_spaces": 0.15},
            metrics={"r_squared": 0.81, "rmse": 55_200.0},
            sample_count=410,
            depth=2,
            leaf_count=2,
        )

        # Act
        restored = DecisionTreeResult.model_validate(original.model_dump())

        # Assert
        assert restored == original

    def test_result_rejects_empty_rules_when_leaf_count_is_one(self) -> None:
        """A result with rules=[] and leaf_count=1 should raise a ValidationError.

        Verifies the invariant that len(rules) must equal leaf_count; an empty
        rules list contradicts a leaf_count of 1.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="rules"):
            DecisionTreeResult(
                target="subscription_tier",
                task_type="classification",
                features_used=["monthly_spend"],
                features_excluded=[],
                rules=[],
                feature_importance={"monthly_spend": 1.0},
                metrics={"accuracy": 0.0},
                sample_count=1,
                depth=0,
                leaf_count=1,
            )

    def test_result_rejects_rules_count_mismatched_with_leaf_count(self) -> None:
        """A result where len(rules) != leaf_count should raise a ValidationError.

        Verifies that providing two rules when leaf_count is three is rejected,
        enforcing the one-rule-per-leaf invariant.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="rules"):
            DecisionTreeResult(
                target="price_tier",
                task_type="classification",
                features_used=["sqft", "bedrooms"],
                features_excluded=[],
                rules=[
                    ClassificationRule(
                        task_type="classification",
                        predicates=[Predicate(variable="sqft", operator="<=", value=800.0)],
                        prediction="budget",
                        samples=200,
                        confidence=0.80,
                    ),
                    ClassificationRule(
                        task_type="classification",
                        predicates=[Predicate(variable="sqft", operator=">", value=800.0)],
                        prediction="premium",
                        samples=300,
                        confidence=0.85,
                    ),
                ],
                feature_importance={"sqft": 0.70, "bedrooms": 0.30},
                metrics={"accuracy": 0.82},
                sample_count=500,
                depth=1,
                leaf_count=3,
            )

    def test_result_with_depth_zero(self) -> None:
        """A result with depth=0 should be accepted as a valid single-node tree."""
        # Act
        tree_result = DecisionTreeResult(
            target="churn_risk",
            task_type="classification",
            features_used=["account_balance"],
            features_excluded=[],
            rules=[
                ClassificationRule(
                    task_type="classification",
                    predicates=[],
                    prediction="low_risk",
                    samples=500,
                    confidence=0.72,
                ),
            ],
            feature_importance={"account_balance": 1.0},
            metrics={"accuracy": 0.72},
            sample_count=500,
            depth=0,
            leaf_count=1,
        )

        # Assert
        with check:
            assert tree_result.depth == 0
        with check:
            assert tree_result.target == "churn_risk"
        with check:
            assert tree_result.task_type == "classification"

    def test_result_rejects_invalid_task_type(self) -> None:
        """A result with an invalid task_type should raise a ValidationError."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="task_type"):
            DecisionTreeResult(
                target="segment",
                task_type="clustering",  # type: ignore[arg-type]
                features_used=["age", "income"],
                features_excluded=[],
                rules=[
                    ClassificationRule(
                        task_type="classification",
                        predicates=[Predicate(variable="age", operator=">", value=30.0)],
                        prediction="group_a",
                        samples=100,
                        confidence=0.75,
                    ),
                ],
                feature_importance={"age": 0.5, "income": 0.5},
                metrics={},
                sample_count=100,
                depth=0,
                leaf_count=1,
            )

    def test_result_rejects_negative_depth(self) -> None:
        """A result with negative depth should raise a ValidationError."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="depth"):
            DecisionTreeResult(
                target="fraud_flag",
                task_type="classification",
                features_used=["transaction_amount"],
                features_excluded=[],
                rules=[
                    ClassificationRule(
                        task_type="classification",
                        predicates=[],
                        prediction="no_fraud",
                        samples=200,
                        confidence=0.9,
                    ),
                ],
                feature_importance={"transaction_amount": 1.0},
                metrics={"accuracy": 0.9},
                sample_count=200,
                depth=-1,
                leaf_count=1,
            )

    def test_result_rejects_sample_count_zero(self) -> None:
        """A result with sample_count=0 should raise a ValidationError because ge=1 requires at least one sample."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="sample_count"):
            DecisionTreeResult(
                target="conversion",
                task_type="classification",
                features_used=["page_views"],
                features_excluded=[],
                rules=[
                    ClassificationRule(
                        task_type="classification",
                        predicates=[],
                        prediction="converted",
                        samples=1,
                        confidence=1.0,
                    ),
                ],
                feature_importance={"page_views": 1.0},
                metrics={"accuracy": 0.0},
                sample_count=0,
                depth=0,
                leaf_count=1,
            )

    def test_result_rejects_leaf_count_zero(self) -> None:
        """A result with leaf_count=0 should raise a ValidationError because ge=1 requires at least one leaf."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="leaf_count"):
            DecisionTreeResult(
                target="revenue_band",
                task_type="regression",
                features_used=["order_value"],
                features_excluded=[],
                rules=[],
                feature_importance={"order_value": 1.0},
                metrics={"r_squared": 0.5},
                sample_count=100,
                depth=0,
                leaf_count=0,
            )

    def test_result_rejects_rules_with_task_type_inconsistent_with_result_task_type(self) -> None:
        """A result with rule task_types inconsistent with the result task_type should raise a ValidationError.

        Verifies that embedding a `ClassificationRule` inside a regression
        `DecisionTreeResult` is rejected, enforcing the invariant that every
        rule must share the same task_type as the enclosing result.

        The insurance claim domain is chosen to vary the realistic data from
        other tests in this class.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="task_type"):
            DecisionTreeResult(
                target="claim_amount",
                task_type="regression",
                features_used=["policy_age_years", "prior_claims_count"],
                features_excluded=[],
                rules=[
                    RegressionRule(
                        task_type="regression",
                        predicates=[Predicate(variable="prior_claims_count", operator="<=", value=1.0)],
                        prediction=4200.0,
                        samples=310,
                        std=850.0,
                    ),
                    ClassificationRule(  # wrong task_type: should be a RegressionRule
                        task_type="classification",
                        predicates=[Predicate(variable="prior_claims_count", operator=">", value=1.0)],
                        prediction="high_risk",
                        samples=140,
                        confidence=0.78,
                    ),
                ],
                feature_importance={"policy_age_years": 0.55, "prior_claims_count": 0.45},
                metrics={"r_squared": 0.71, "rmse": 1200.0},
                sample_count=450,
                depth=1,
                leaf_count=2,
            )

    def test_result_rejects_feature_importance_not_summing_to_one(self) -> None:
        """A result with feature_importance scores that do not sum to 1.0 should raise a ValidationError.

        Verifies the validator that enforces importance scores sum to 1.0 within
        a tolerance of 1e-6, catching data errors where importances are unnormalized.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match=r"feature_importance scores must sum to 1\.0"):
            DecisionTreeResult(
                target="churn",
                task_type="classification",
                features_used=["tenure_months", "support_tickets"],
                features_excluded=[],
                rules=[
                    ClassificationRule(
                        task_type="classification",
                        predicates=[],
                        prediction="retained",
                        samples=300,
                        confidence=0.80,
                    ),
                ],
                feature_importance={"tenure_months": 0.60, "support_tickets": 0.60},
                metrics={"accuracy": 0.80},
                sample_count=300,
                depth=0,
                leaf_count=1,
            )

    def test_result_accepts_feature_importance_within_floating_point_tolerance(self) -> None:
        """Feature importance scores summing to 1.0 within 1e-6 tolerance should be accepted.

        Verifies that minor floating-point rounding errors in importance scores
        (as produced by sklearn) do not cause spurious validation failures.
        """
        # Arrange / Act / Assert — values sum to 1.0000000005, within tolerance
        tree_result = DecisionTreeResult(
            target="churn",
            task_type="classification",
            features_used=["tenure_months", "support_tickets", "monthly_charges"],
            features_excluded=[],
            rules=[
                ClassificationRule(
                    task_type="classification",
                    predicates=[],
                    prediction="retained",
                    samples=500,
                    confidence=0.82,
                ),
            ],
            feature_importance={
                "tenure_months": 0.500_000_000_3,
                "support_tickets": 0.300_000_000_1,
                "monthly_charges": 0.199_999_999_6,
            },
            metrics={"accuracy": 0.82},
            sample_count=500,
            depth=0,
            leaf_count=1,
        )

        # Assert
        with check:
            assert tree_result.feature_importance["tenure_months"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase 2: Preprocessing Pipeline
# ---------------------------------------------------------------------------


class TestClassifyColumn:
    """Tests for _classify_column: maps Polars dtypes to broad feature categories."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        ],
    )
    def test_numeric_types_return_numeric(self, dtype: pl.DataType) -> None:
        """All integer and float dtypes should classify as 'numeric'.

        Args:
            dtype (pl.DataType): A Polars numeric dtype to classify.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "numeric"

    def test_boolean_type_returns_boolean(self) -> None:
        """Boolean dtype should classify as 'boolean'."""
        # Act
        column_type = _classify_column(pl.Boolean)  # type: ignore[arg-type]

        # Assert
        assert column_type == "boolean"

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.String,
            pl.Utf8,
            pl.Categorical,
        ],
    )
    def test_categorical_types_return_categorical(self, dtype: pl.DataType) -> None:
        """String, Utf8, and Categorical dtypes should classify as 'categorical'.

        Args:
            dtype (pl.DataType): A Polars categorical-like dtype to classify.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "categorical"

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.Enum(["a", "b"]),
        ],
    )
    def test_enum_type_returns_categorical(self, dtype: pl.DataType) -> None:
        """Parameterized Enum instances should classify as 'categorical'.

        Previously the lookup map used the bare `pl.Enum` class as its key,
        which caused a hash mismatch for parameterized instances like
        `Enum(['a', 'b'])`.  The `isinstance` fallback now ensures
        parameterized instances are correctly classified.

        Args:
            dtype (pl.DataType): A parameterized Polars Enum dtype to classify.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "categorical"

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.Date,
            pl.Datetime,
            pl.Datetime("us"),
            pl.Datetime("ns"),
        ],
    )
    def test_datetime_types_return_datetime(self, dtype: pl.DataType) -> None:
        """Date and Datetime dtypes (bare and parameterized) should classify as 'datetime'.

        Args:
            dtype (pl.DataType): A Polars date/datetime dtype to classify.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "datetime"

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.Duration,
            pl.Duration("us"),
            pl.Duration("ns"),
        ],
    )
    def test_duration_type_returns_duration(self, dtype: pl.DataType) -> None:
        """Duration dtypes (bare and parameterized) should classify as 'duration'.

        Args:
            dtype (pl.DataType): A Polars duration dtype to classify.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "duration"

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.Time,
            pl.Binary,
            pl.Null,
        ],
    )
    def test_excluded_scalar_types_return_excluded(self, dtype: pl.DataType) -> None:
        """Time, Binary, and Null dtypes have no supported encoding and should return 'excluded'.

        Args:
            dtype (pl.DataType): A Polars dtype that cannot be used as a feature.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "excluded"

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.List(pl.Int64),
            pl.Array(pl.Float32, 4),
        ],
    )
    def test_nested_types_return_excluded(self, dtype: pl.DataType) -> None:
        """Parameterized nested types (List, Array) are not in the lookup map and return 'excluded'.

        Args:
            dtype (pl.DataType): A Polars nested collection dtype.
        """
        # Act
        column_type = _classify_column(dtype)

        # Assert
        assert column_type == "excluded"


class TestFilterFeatures:
    """Tests for _filter_features: partition feature columns into kept and excluded sets."""

    def test_keeps_valid_features(self) -> None:
        """Numeric and categorical columns with variance should pass through to the kept list."""
        # Arrange
        df = pl.DataFrame({
            "age": pl.Series([25, 42, 31, 58, 19], dtype=pl.Int32),
            "status": pl.Series(["active", "inactive", "active", "inactive", "active"]),
        })

        # Act
        kept, excluded = _filter_features(df, ["age", "status"])

        # Assert
        with check:
            assert kept == ["age", "status"]
        with check:
            assert excluded == []

    def test_excludes_all_null_column(self) -> None:
        """A column where every value is null should be excluded with an 'all values are null' reason."""
        # Arrange
        df = pl.DataFrame({
            "revenue": pl.Series([1000.0, 2500.0, 500.0]),
            "missing_field": pl.Series([None, None, None], dtype=pl.Float64),
        })

        # Act
        kept, excluded = _filter_features(df, ["revenue", "missing_field"])

        # Assert
        with check:
            assert kept == ["revenue"]
        with check:
            assert len(excluded) == 1
        with check:
            assert excluded[0].name == "missing_field"
        with check:
            assert excluded[0].reason == "all values are null"

    def test_excludes_zero_variance_column(self) -> None:
        """A column with a single unique value (zero variance) should be excluded."""
        # Arrange
        df = pl.DataFrame({
            "score": pl.Series([88.5, 72.0, 91.5, 65.0, 78.5]),
            "constant_flag": pl.Series([1, 1, 1, 1, 1], dtype=pl.Int32),
        })

        # Act
        kept, excluded = _filter_features(df, ["score", "constant_flag"])

        # Assert
        with check:
            assert kept == ["score"]
        with check:
            assert len(excluded) == 1
        with check:
            assert excluded[0].name == "constant_flag"
        with check:
            assert excluded[0].reason == "single unique value"

    def test_excludes_high_cardinality_strings(self) -> None:
        """A string column with more than 90% unique values should be excluded as a likely identifier."""
        # Arrange — 50 rows, all unique strings → 100% unique ratio, well above the 0.9 threshold
        df = pl.DataFrame({
            "customer_id": [f"cust_{i:04d}" for i in range(50)],
            "region": ["North", "South"] * 25,
        })

        # Act
        kept, excluded = _filter_features(df, ["customer_id", "region"])

        # Assert
        with check:
            assert kept == ["region"]
        with check:
            assert len(excluded) == 1
        with check:
            assert excluded[0].name == "customer_id"
        with check:
            assert "high cardinality" in excluded[0].reason

    def test_excludes_complex_types(self) -> None:
        """List and Struct columns should be excluded with an 'unsupported dtype' reason."""
        # Arrange
        df = pl.DataFrame({
            "valid_col": pl.Series([1.0, 2.0, 3.0]),
            "tags": pl.Series([["a", "b"], ["c"], ["d", "e", "f"]]),
        })

        # Act
        kept, excluded = _filter_features(df, ["valid_col", "tags"])

        # Assert
        with check:
            assert kept == ["valid_col"]
        with check:
            assert excluded[0].name == "tags"
        with check:
            assert excluded[0].reason == "unsupported dtype"

    def test_returns_exclusion_reasons(self) -> None:
        """Each excluded feature should carry a non-empty human-readable reason string."""
        # Arrange — three columns each hitting a different exclusion path
        df = pl.DataFrame({
            "all_null_col": pl.Series([None, None, None], dtype=pl.Int64),
            "constant_col": pl.Series(["yes", "yes", "yes"]),
            "list_col": pl.Series([[10], [20], [30]]),
        })

        # Act
        _, excluded = _filter_features(df, ["all_null_col", "constant_col", "list_col"])

        # Assert — verify all reasons are meaningful non-empty strings
        for exc in excluded:
            with check:
                assert isinstance(exc.reason, str), f"Reason for {exc.name!r} should be str"
            with check:
                assert len(exc.reason) > 0, f"Reason for {exc.name!r} should not be empty"
            with check:
                assert isinstance(exc, _ExcludedFeature), f"{exc.name!r} should be _ExcludedFeature"

    def test_all_features_excluded_returns_empty_kept_list(self) -> None:
        """When every feature fails at least one filter, the kept list should be empty."""
        # Arrange — one all-null column and one zero-variance column
        df = pl.DataFrame({
            "all_null": pl.Series([None, None, None, None], dtype=pl.Float64),
            "constant": pl.Series([99, 99, 99, 99], dtype=pl.Int64),
        })

        # Act
        kept, excluded = _filter_features(df, ["all_null", "constant"])

        # Assert
        with check:
            assert kept == []
        with check:
            assert len(excluded) == 2


class TestDetectTaskType:
    """Tests for _detect_task_type: infer classification vs regression from target series."""

    def test_string_target_is_classification(self) -> None:
        """A target series with String dtype should be detected as classification."""
        # Arrange
        target = pl.Series("churn_label", ["churned", "retained", "churned", "retained"])

        # Act
        task_type = _detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_boolean_target_is_classification(self) -> None:
        """A target series with Boolean dtype should be detected as classification."""
        # Arrange
        target = pl.Series("is_fraud", [True, False, False, True, False], dtype=pl.Boolean)

        # Act
        task_type = _detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_string_target_with_utf8_alias_is_classification(self) -> None:
        """A target series created with pl.Utf8 (alias for String) should be detected as classification."""
        # Arrange — pl.Utf8 and pl.String are the same dtype
        target = pl.Series("churn_status", ["churned", "retained", "retained"], dtype=pl.Utf8)

        # Act
        task_type = _detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_float_target_is_regression(self) -> None:
        """A Float64 target series should be detected as regression."""
        # Arrange
        target = pl.Series(
            "house_price",
            [285_000.0, 420_000.0, 175_500.0, 610_000.0],
            dtype=pl.Float64,
        )

        # Act
        task_type = _detect_task_type(target, None)

        # Assert
        assert task_type == "regression"

    def test_low_cardinality_int_is_classification(self) -> None:
        """An integer series with 5 unique values across 200 rows should be classification.

        Low cardinality (≤20 unique values) triggers the classification heuristic.
        """
        # Arrange — 5 unique labels repeated across 200 rows
        target = pl.Series("rating", list(range(1, 6)) * 40, dtype=pl.Int32)

        # Act
        task_type = _detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_high_cardinality_int_is_regression(self) -> None:
        """An integer series with 150 unique values should be detected as regression.

        High cardinality (>20 unique values and >5% unique ratio) triggers regression.
        """
        # Arrange — 150 distinct integers, unique ratio = 1.0 >> 0.05 threshold
        target = pl.Series("transaction_id", list(range(150)), dtype=pl.Int64)

        # Act
        task_type = _detect_task_type(target, None)

        # Assert
        assert task_type == "regression"

    def test_override_to_classification(self) -> None:
        """A float target with a 'classification' override should return 'classification'."""
        # Arrange
        target = pl.Series("score", [1.0, 2.0, 3.0, 1.0, 2.0], dtype=pl.Float32)

        # Act
        task_type = _detect_task_type(target, "classification")

        # Assert
        assert task_type == "classification"

    def test_override_to_regression(self) -> None:
        """A string target with a 'regression' override should return 'regression'."""
        # Arrange
        target = pl.Series("label", ["yes", "no", "yes", "yes"])

        # Act
        task_type = _detect_task_type(target, "regression")

        # Assert
        assert task_type == "regression"

    def test_auto_override_behaves_same_as_none(self) -> None:
        """Both None and 'auto' should trigger automatic detection and produce the same result."""
        # Arrange
        target = pl.Series("category", ["A", "B", "A", "C", "B"])

        # Act
        task_type_none = _detect_task_type(target, None)
        task_type_auto = _detect_task_type(target, "auto")

        # Assert
        with check:
            assert task_type_none == "classification"
        with check:
            assert task_type_none == task_type_auto


class TestEncodeFeatures:
    """Tests for _encode_features: encode DataFrame columns into a float64 numpy matrix."""

    def test_numeric_passthrough_preserves_values_and_nulls_become_nan(self) -> None:
        """Numeric columns should be cast to float64; Polars nulls must become NaN in the matrix."""
        # Arrange
        df = pl.DataFrame({
            "weight_kg": pl.Series([55.5, 72.0, None, 90.1], dtype=pl.Float64),
        })

        # Act
        matrix, encoders = _encode_features(df, ["weight_kg"])

        # Assert
        with check:
            assert matrix.shape == (4, 1)
        with check:
            assert matrix.dtype == np.float64
        with check:
            assert matrix[0, 0] == pytest.approx(55.5)
        with check:
            assert matrix[1, 0] == pytest.approx(72.0)
        with check:
            assert np.isnan(matrix[2, 0]), "null should become NaN"
        with check:
            assert matrix[3, 0] == pytest.approx(90.1)
        with check:
            assert encoders[0].column_type == "numeric"
        with check:
            assert encoders[0].category_mapping is None

    def test_boolean_encoding_true_false_null(self) -> None:
        """Boolean columns should encode True→1.0, False→0.0, and null→NaN."""
        # Arrange
        df = pl.DataFrame({
            "has_subscription": pl.Series([True, False, None, True], dtype=pl.Boolean),
        })

        # Act
        matrix, encoders = _encode_features(df, ["has_subscription"])

        # Assert
        with check:
            assert matrix[0, 0] == pytest.approx(1.0), "True should encode to 1.0"
        with check:
            assert matrix[1, 0] == pytest.approx(0.0), "False should encode to 0.0"
        with check:
            assert np.isnan(matrix[2, 0]), "null should become NaN"
        with check:
            assert matrix[3, 0] == pytest.approx(1.0)
        with check:
            assert encoders[0].column_type == "boolean"

    def test_categorical_encoding_strings_to_ordinal_ints(self) -> None:
        """String columns should be ordinal-encoded and a category mapping returned."""
        # Arrange — three categories: apple < banana < cherry (alphabetical ordinal order)
        df = pl.DataFrame({
            "fruit": pl.Series(["cherry", "apple", "banana", "apple"], dtype=pl.String),
        })

        # Act
        matrix, encoders = _encode_features(df, ["fruit"])

        # Assert — ordinal codes assigned alphabetically
        with check:
            assert encoders[0].column_type == "categorical"
        with check:
            assert encoders[0].category_mapping is not None
        category_mapping = encoders[0].category_mapping
        assert category_mapping is not None  # narrowing for type checker
        with check:
            assert set(category_mapping.values()) == {"apple", "banana", "cherry"}
        with check:
            # cherry, apple, banana, apple should map to consistent integer codes
            assert matrix[1, 0] == matrix[3, 0], "both 'apple' rows should have same code"
        with check:
            assert matrix.dtype == np.float64

    def test_datetime_encoding_converts_to_epoch_microseconds(self) -> None:
        """Date columns should encode to epoch microseconds as float64."""
        # Arrange — epoch day 0 is 1970-01-01
        df = pl.DataFrame({
            "event_date": pl.Series(
                [
                    datetime.date(1970, 1, 1),
                    datetime.date(2020, 3, 15),
                ],
                dtype=pl.Date,
            ),
        })

        # Act
        matrix, encoders = _encode_features(df, ["event_date"])

        # Assert
        with check:
            assert matrix.dtype == np.float64
        with check:
            assert encoders[0].column_type == "datetime"
        with check:
            assert encoders[0].category_mapping is None
        # epoch is 1970-01-01 → 0 microseconds
        with check:
            assert matrix[0, 0] == pytest.approx(0.0), "1970-01-01 should encode to 0 epoch microseconds"
        # Later date must encode to a larger value
        with check:
            assert matrix[1, 0] > matrix[0, 0], "2020 date should have a larger epoch value"

    def test_duration_encoding_converts_to_microseconds(self) -> None:
        """Duration columns should encode to total microseconds as float64."""
        # Arrange
        df = pl.DataFrame({
            "response_time": pl.Series(
                [
                    datetime.timedelta(seconds=1),
                    datetime.timedelta(minutes=2),
                    datetime.timedelta(hours=1),
                ],
                dtype=pl.Duration,
            ),
        })

        # Act
        matrix, _encoders = _encode_features(df, ["response_time"])

        # Assert
        with check:
            assert matrix.dtype == np.float64
        with check:
            # 1 second = 1_000_000 microseconds
            assert matrix[0, 0] == pytest.approx(1_000_000.0)
        with check:
            # 2 minutes = 120_000_000 microseconds
            assert matrix[1, 0] == pytest.approx(120_000_000.0)
        with check:
            # 1 hour = 3_600_000_000 microseconds
            assert matrix[2, 0] == pytest.approx(3_600_000_000.0)

    def test_mixed_column_types_produce_correct_matrix_shape(self) -> None:
        """A DataFrame with float, boolean, and string columns should produce a (n_rows, 3) matrix."""
        # Arrange
        df = pl.DataFrame({
            "age": pl.Series([23.0, 45.0, 31.0, 67.0], dtype=pl.Float64),
            "is_premium": pl.Series([True, False, True, False], dtype=pl.Boolean),
            "plan": pl.Series(["basic", "pro", "basic", "enterprise"], dtype=pl.String),
        })

        # Act
        matrix, encoders = _encode_features(df, ["age", "is_premium", "plan"])

        # Assert
        with check:
            assert matrix.shape == (4, 3), "Matrix must have one column per feature"
        with check:
            assert matrix.dtype == np.float64
        with check:
            assert len(encoders) == 3
        with check:
            assert encoders[0].column_name == "age"
        with check:
            assert encoders[1].column_name == "is_premium"
        with check:
            assert encoders[2].column_name == "plan"

    def test_categorical_with_nulls_assigned_consistent_code(self) -> None:
        """Null values in a String column are converted to 'None' by Polars before encoding.

        Because Polars' to_numpy() materialises null as the Python string 'None',
        the OrdinalEncoder treats it as a distinct category and assigns it a
        consistent integer code.  Both null rows therefore share the same code,
        and non-null rows are unaffected.
        """
        # Arrange
        df = pl.DataFrame({
            "department": pl.Series(
                ["engineering", None, "marketing", "engineering", None],
                dtype=pl.String,
            ),
        })

        # Act
        matrix, encoders = _encode_features(df, ["department"])

        # Assert
        with check:
            assert encoders[0].column_type == "categorical"
        with check:
            # Both null rows should receive the same code
            assert matrix[1, 0] == matrix[4, 0], "both null rows should share the same code"
        with check:
            # Non-null 'engineering' rows should also share the same finite code
            assert matrix[0, 0] == matrix[3, 0], "both 'engineering' rows should share the same code"
        with check:
            assert matrix[0, 0] != matrix[1, 0], "'engineering' and null should have different codes"


class TestEncodeTarget:
    """Tests for _encode_target: encode the target column for regression or classification."""

    def test_regression_target_passthrough_as_float64(self) -> None:
        """Regression targets should be returned as a float64 numpy array with no category mapping."""
        # Arrange
        target = pl.Series("house_price", [285_000.0, 420_500.0, 175_000.0, 610_200.0])

        # Act
        encoded_array, category_mapping = _encode_target(target, "regression")

        # Assert
        with check:
            assert category_mapping is None
        with check:
            assert encoded_array.dtype == np.float64
        with check:
            assert encoded_array[0] == pytest.approx(285_000.0)
        with check:
            assert encoded_array[1] == pytest.approx(420_500.0)
        with check:
            assert encoded_array[2] == pytest.approx(175_000.0)
        with check:
            assert encoded_array[3] == pytest.approx(610_200.0)

    def test_classification_target_encodes_strings_to_integer_codes(self) -> None:
        """Classification targets should encode to float64 integer codes with a non-None mapping."""
        # Arrange — three classes: bird, cat, dog
        target = pl.Series("animal", ["cat", "dog", "cat", "bird", "dog"])

        # Act
        encoded_array, category_mapping = _encode_target(target, "classification")

        # Assert
        with check:
            assert category_mapping is not None
        with check:
            assert encoded_array.dtype == np.float64
        with check:
            # Consistent encoding: both 'cat' entries share the same code
            assert encoded_array[0] == encoded_array[2], "both 'cat' rows should have identical code"
        with check:
            # Consistent encoding: both 'dog' entries share the same code
            assert encoded_array[1] == encoded_array[4], "both 'dog' rows should have identical code"
        with check:
            # All three classes produce distinct codes
            assert len({encoded_array[0], encoded_array[1], encoded_array[3]}) == 3

    def test_classification_target_preserves_original_labels_in_mapping(self) -> None:
        """The category mapping should contain the original string labels as values."""
        # Arrange
        target = pl.Series("subscription_tier", ["free", "basic", "premium", "enterprise"])

        # Act
        _, category_mapping = _encode_target(target, "classification")

        # Assert
        assert category_mapping is not None  # narrowing for type checker
        label_values = set(category_mapping.values())
        with check:
            assert "free" in label_values
        with check:
            assert "basic" in label_values
        with check:
            assert "premium" in label_values
        with check:
            assert "enterprise" in label_values
        with check:
            # Mapping keys should be consecutive integers starting at 0
            assert set(category_mapping.keys()) == {0, 1, 2, 3}
