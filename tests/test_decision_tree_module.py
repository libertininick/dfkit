"""Tests for the decision tree module: Pydantic models and preprocessing pipeline functions."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from pytest_check import check
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dfkit.decision_tree_module import (
    ClassificationRule,
    DecisionTreeResult,
    Predicate,
    RegressionRule,
    _build_decision_tree_result,
    _classify_column,
    _compute_feature_importance,
    _compute_metrics,
    _detect_task_type,
    _encode_features,
    _encode_target,
    _ExcludedFeature,
    _extract_rules,
    _filter_features,
    _fit_tree,
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
        rules: list[ClassificationRule] = [
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
        rules: list[RegressionRule] = [
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
                rules=[  # type: ignore[arg-type]
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

    def test_cardinality_just_below_threshold_is_kept(self) -> None:
        """A categorical column with 89% unique values should pass the cardinality filter.

        The threshold is 0.9 (exclusive). 89 unique values out of 100 rows
        yields a ratio of 0.89, which is strictly below the threshold, so the
        column must be kept.
        """
        # Arrange — 100 rows, 89 unique values → 89% unique ratio (just under 0.9)
        unique_labels = [f"label_{i}" for i in range(89)]
        # Fill remaining 11 rows by repeating the first label
        category_values = unique_labels + ["label_0"] * 11
        df = pl.DataFrame({
            "near_unique_col": category_values,
            "score": list(range(100)),
        })

        # Act
        kept, excluded = _filter_features(df, ["near_unique_col", "score"])

        # Assert
        with check:
            assert "near_unique_col" in kept, "89% unique ratio is below the 0.9 threshold and should be kept"
        with check:
            assert excluded == []

    def test_cardinality_just_above_threshold_is_excluded(self) -> None:
        """A categorical column with 91% unique values should be excluded as a likely identifier.

        The threshold is 0.9 (exclusive). 91 unique values out of 100 rows
        yields a ratio of 0.91, which exceeds the threshold, so the column
        must be excluded.
        """
        # Arrange — 100 rows, 91 unique values → 91% unique ratio (just above 0.9)
        unique_labels = [f"label_{i}" for i in range(91)]
        # Fill remaining 9 rows by repeating the first label
        category_values = unique_labels + ["label_0"] * 9
        df = pl.DataFrame({
            "near_unique_col": category_values,
            "score": list(range(100)),
        })

        # Act
        kept, excluded = _filter_features(df, ["near_unique_col", "score"])

        # Assert
        with check:
            assert "near_unique_col" not in kept, "91% unique ratio exceeds the 0.9 threshold and should be excluded"
        with check:
            assert len(excluded) == 1
        with check:
            assert excluded[0].name == "near_unique_col"
        with check:
            assert "high cardinality" in excluded[0].reason

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

    def test_integer_dtype_regression_target_upcast_to_float64(self) -> None:
        """A regression target stored as `Int32` should be upcast to a `float64` numpy array.

        This guards against a regression where integer-dtype columns were returned
        with their original integer dtype instead of being promoted to `float64` for
        sklearn compatibility.
        """
        # Arrange — sale quantity stored as Int32; regression task
        target = pl.Series("sale_quantity", [10, 25, 7, 42, 33], dtype=pl.Int32)

        # Act
        encoded_array, category_mapping = _encode_target(target, "regression")

        # Assert
        with check:
            assert category_mapping is None
        with check:
            assert encoded_array.dtype == np.float64, "Int32 regression target must be upcast to float64"
        with check:
            assert encoded_array[0] == pytest.approx(10.0)
        with check:
            assert encoded_array[1] == pytest.approx(25.0)
        with check:
            assert encoded_array[4] == pytest.approx(33.0)


# ---------------------------------------------------------------------------
# Phase 3: Tree Fitting, Rule Extraction, Metrics and Orchestration
# ---------------------------------------------------------------------------


class TestFitTree:
    """Tests for `_fit_tree`: fits sklearn decision trees with configurable hyperparameters."""

    def test_fit_classifier_returns_decision_tree_classifier(self) -> None:
        """Fitting with `task_type='classification'` should return a DecisionTreeClassifier."""
        # Arrange
        feature_matrix, target_array = _make_churn_classification_data()

        # Act
        fitted_tree = _fit_tree(
            feature_matrix,
            target_array,
            task_type="classification",
            max_depth=3,
            min_samples_leaf=5,
        )

        # Assert
        assert isinstance(fitted_tree, DecisionTreeClassifier)

    def test_fit_regressor_returns_decision_tree_regressor(self) -> None:
        """Fitting with `task_type='regression'` should return a DecisionTreeRegressor."""
        # Arrange
        feature_matrix, target_array = _make_house_price_regression_data()

        # Act
        fitted_tree = _fit_tree(
            feature_matrix,
            target_array,
            task_type="regression",
            max_depth=4,
            min_samples_leaf=5,
        )

        # Assert
        assert isinstance(fitted_tree, DecisionTreeRegressor)

    def test_respects_max_depth(self) -> None:
        """The fitted tree depth should not exceed the requested `max_depth`."""
        # Arrange
        feature_matrix, target_array = _make_churn_classification_data()
        requested_max_depth = 2

        # Act
        fitted_tree = _fit_tree(
            feature_matrix,
            target_array,
            task_type="classification",
            max_depth=requested_max_depth,
            min_samples_leaf=5,
        )

        # Assert
        assert fitted_tree.get_depth() <= requested_max_depth

    def test_respects_min_samples_leaf(self) -> None:
        """No leaf should have fewer samples than `min_samples_leaf`."""
        # Arrange
        feature_matrix, target_array = _make_house_price_regression_data()
        requested_min_samples = 10

        # Act
        fitted_tree = _fit_tree(
            feature_matrix,
            target_array,
            task_type="regression",
            max_depth=4,
            min_samples_leaf=requested_min_samples,
        )

        # Assert — inspect leaf node sample counts via the internal tree structure
        sklearn_tree = fitted_tree.tree_
        leaf_mask = sklearn_tree.children_left == sklearn_tree.children_right  # -1 == -1 at leaves
        leaf_sample_counts = sklearn_tree.n_node_samples[leaf_mask]
        assert all(count >= requested_min_samples for count in leaf_sample_counts)


class TestExtractRules:
    """Tests for `_extract_rules`: converts a fitted tree into human-readable rules."""

    def test_classification_rules_have_predicate_conditions(self) -> None:
        """Every classification rule's `predicates` field should be a list of `Predicate` objects."""
        # Arrange — fit a small classification tree on employee churn data
        df = pl.DataFrame({
            "tenure_months": [
                3,
                5,
                2,
                48,
                60,
                36,
                24,
                4,
                6,
                52,
                7,
                45,
                50,
                1,
                8,
                55,
                30,
                10,
                42,
                3,
                2,
                60,
                48,
                5,
                35,
                1,
                6,
                40,
                55,
                8,
                3,
                50,
                2,
                36,
                4,
                45,
                7,
                28,
                60,
                5,
                30,
                1,
                4,
                48,
                2,
                55,
                6,
                36,
                8,
                42,
            ],
            "support_tickets": [
                12,
                9,
                14,
                1,
                0,
                2,
                3,
                11,
                8,
                1,
                7,
                2,
                1,
                15,
                10,
                0,
                3,
                6,
                2,
                13,
                14,
                0,
                1,
                11,
                2,
                15,
                9,
                1,
                0,
                8,
                12,
                1,
                13,
                2,
                10,
                1,
                7,
                3,
                0,
                11,
                3,
                14,
                10,
                1,
                12,
                0,
                8,
                2,
                6,
                1,
            ],
            "churn": (["yes"] * 10 + ["no"] * 10) * 2 + ["yes"] * 5 + ["no"] * 5,
        })
        feature_matrix, feature_encoders = _encode_features(df, ["tenure_months", "support_tickets"])
        target_array, target_mapping = _encode_target(df["churn"], "classification")
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="classification",
        )

        # Assert
        assert len(rules) > 0
        for rule in rules:
            with check:
                assert isinstance(rule.predicates, list)
            with check:
                assert all(isinstance(p, Predicate) for p in rule.predicates)

    def test_classification_rules_have_confidence(self) -> None:
        """Every classification rule should have `confidence` in the closed interval [0, 1]."""
        # Arrange — insurance fraud detection dataset
        df = pl.DataFrame({
            "claim_amount": [
                500,
                800,
                200,
                12000,
                15000,
                9500,
                300,
                700,
                450,
                11000,
                600,
                13000,
                8500,
                250,
                900,
                14500,
                350,
                750,
                180,
                10500,
                400,
                9000,
                650,
                190,
                11500,
                850,
                280,
                7500,
                1000,
                13500,
                550,
                8000,
                420,
                220,
                950,
                12500,
                380,
                6500,
                480,
                14000,
                320,
                7000,
                580,
                230,
                870,
                13000,
                440,
                9800,
                610,
                11200,
            ],
            "policy_age_years": [
                1,
                2,
                3,
                1,
                1,
                2,
                4,
                3,
                2,
                1,
                3,
                1,
                2,
                5,
                2,
                1,
                4,
                3,
                5,
                1,
                3,
                2,
                4,
                5,
                1,
                2,
                4,
                2,
                3,
                1,
                3,
                2,
                4,
                5,
                2,
                1,
                4,
                3,
                3,
                1,
                4,
                2,
                3,
                5,
                2,
                1,
                4,
                2,
                3,
                1,
            ],
            "is_fraud": (["no"] * 5 + ["yes"] * 5) * 5,
        })
        feature_matrix, feature_encoders = _encode_features(df, ["claim_amount", "policy_age_years"])
        target_array, target_mapping = _encode_target(df["is_fraud"], "classification")
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="classification",
        )

        # Assert
        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, ClassificationRule)
            with check:
                assert 0.0 <= rule.confidence <= 1.0

    def test_regression_rules_have_std(self) -> None:
        """Every regression rule should have `std >= 0`."""
        # Arrange — employee salary prediction
        df = pl.DataFrame({
            "years_experience": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                2.5,
                3.5,
                7.5,
                9.5,
                11.5,
                1.5,
                4.5,
                6.5,
                8.5,
                12.5,
                0.5,
                5.5,
                10.5,
                13.5,
                14.5,
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                15.0,
                1.5,
                3.5,
                5.5,
                7.5,
                9.5,
            ],
            "annual_salary": [
                45000,
                50000,
                55000,
                60000,
                68000,
                75000,
                82000,
                90000,
                98000,
                105000,
                112000,
                120000,
                128000,
                135000,
                142000,
                52000,
                57000,
                86000,
                100000,
                115000,
                47000,
                62000,
                78000,
                94000,
                124000,
                43000,
                71000,
                108000,
                131000,
                138000,
                45000,
                55000,
                68000,
                82000,
                98000,
                112000,
                128000,
                50000,
                60000,
                75000,
                90000,
                105000,
                120000,
                135000,
                142000,
                47000,
                57000,
                71000,
                86000,
                100000,
            ],
        })
        feature_matrix, feature_encoders = _encode_features(df, ["years_experience"])
        target_array, target_mapping = _encode_target(df["annual_salary"], "regression")
        fitted_tree = _fit_tree(feature_matrix, target_array, task_type="regression", max_depth=3, min_samples_leaf=5)

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="regression",
        )

        # Assert
        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, RegressionRule)
            with check:
                assert rule.std >= 0.0

    def test_rules_cover_all_samples(self) -> None:
        """The sum of `samples` across all leaf rules should equal the total sample count."""
        # Arrange — subscription renewal prediction
        df = pl.DataFrame({
            "days_since_login": [
                1,
                5,
                2,
                90,
                120,
                60,
                30,
                3,
                7,
                95,
                10,
                85,
                100,
                2,
                15,
                110,
                45,
                20,
                80,
                4,
                3,
                130,
                88,
                8,
                70,
                1,
                12,
                75,
                105,
                14,
                2,
                95,
                3,
                65,
                6,
                88,
                11,
                55,
                115,
                9,
                50,
                1,
                7,
                92,
                4,
                108,
                13,
                68,
                16,
                82,
            ],
            "page_views": [
                25,
                18,
                30,
                2,
                1,
                5,
                8,
                22,
                14,
                2,
                12,
                3,
                1,
                28,
                16,
                1,
                6,
                10,
                4,
                24,
                27,
                0,
                2,
                19,
                5,
                31,
                15,
                4,
                1,
                17,
                26,
                2,
                29,
                6,
                21,
                3,
                13,
                7,
                1,
                20,
                8,
                30,
                18,
                2,
                25,
                1,
                16,
                5,
                11,
                3,
            ],
            "renewed": (["yes"] * 10 + ["no"] * 10) * 2 + ["yes"] * 5 + ["no"] * 5,
        })
        feature_matrix, feature_encoders = _encode_features(df, ["days_since_login", "page_views"])
        target_array, target_mapping = _encode_target(df["renewed"], "classification")
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )
        total_samples = len(df)

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="classification",
        )

        # Assert
        total_from_rules = sum(rule.samples for rule in rules)
        assert total_from_rules == total_samples

    def test_predicates_use_original_column_names(self) -> None:
        """Predicates in each rule should use the original feature column names, not indices."""
        # Arrange — customer lifetime value prediction
        df = pl.DataFrame({
            "monthly_spend": [
                25.0,
                30.0,
                15.0,
                80.0,
                95.0,
                70.0,
                20.0,
                40.0,
                10.0,
                90.0,
                50.0,
                85.0,
                75.0,
                12.0,
                45.0,
                100.0,
                35.0,
                55.0,
                18.0,
                88.0,
                22.0,
                92.0,
                78.0,
                28.0,
                65.0,
                8.0,
                42.0,
                72.0,
                98.0,
                38.0,
                27.0,
                82.0,
                18.0,
                60.0,
                32.0,
                87.0,
                48.0,
                53.0,
                105.0,
                36.0,
                55.0,
                9.0,
                33.0,
                91.0,
                24.0,
                102.0,
                44.0,
                63.0,
                38.0,
                86.0,
            ],
            "order_count": [
                3,
                4,
                2,
                10,
                12,
                9,
                3,
                5,
                1,
                11,
                6,
                11,
                10,
                2,
                6,
                13,
                4,
                7,
                2,
                11,
                3,
                12,
                10,
                4,
                8,
                1,
                5,
                9,
                13,
                5,
                4,
                11,
                2,
                8,
                4,
                11,
                6,
                7,
                14,
                5,
                7,
                1,
                4,
                12,
                3,
                13,
                6,
                8,
                5,
                11,
            ],
            "lifetime_value": [100.0 + i * 50 for i in range(50)],
        })
        feature_matrix, feature_encoders = _encode_features(df, ["monthly_spend", "order_count"])
        target_array, target_mapping = _encode_target(df["lifetime_value"], "regression")
        fitted_tree = _fit_tree(feature_matrix, target_array, task_type="regression", max_depth=2, min_samples_leaf=5)

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="regression",
        )

        # Assert — every predicate variable must be a known feature name
        feature_names = {"monthly_spend", "order_count"}
        for rule in rules:
            for predicate in rule.predicates:
                with check:
                    assert predicate.variable in feature_names

    def test_categorical_predicates_use_set_operators(self) -> None:
        """Predicates derived from categorical features should use `'in'` or `'not in'` operators."""
        # Arrange — product category classification
        df = pl.DataFrame({
            "product_category": (["electronics"] * 15 + ["clothing"] * 15 + ["furniture"] * 10 + ["sports"] * 10),
            "price_usd": [
                200.0,
                350.0,
                120.0,
                450.0,
                280.0,
                310.0,
                175.0,
                390.0,
                230.0,
                420.0,
                260.0,
                340.0,
                195.0,
                380.0,
                290.0,
                45.0,
                75.0,
                30.0,
                90.0,
                60.0,
                55.0,
                80.0,
                35.0,
                95.0,
                65.0,
                70.0,
                40.0,
                85.0,
                50.0,
                100.0,
                350.0,
                500.0,
                250.0,
                620.0,
                400.0,
                275.0,
                475.0,
                320.0,
                550.0,
                410.0,
                25.0,
                40.0,
                18.0,
                55.0,
                32.0,
                28.0,
                45.0,
                22.0,
                50.0,
                36.0,
            ],
            "revenue_tier": (["high"] * 15 + ["low"] * 15 + ["medium"] * 10 + ["low"] * 10),
        })
        feature_matrix, feature_encoders = _encode_features(df, ["product_category"])
        target_array, target_mapping = _encode_target(df["revenue_tier"], "classification")
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="classification",
        )

        # Assert — categorical predicates must use "in" or "not in"
        categorical_operators = set()
        for rule in rules:
            for predicate in rule.predicates:
                if predicate.variable == "product_category":
                    categorical_operators.add(predicate.operator)

        # At least one categorical split occurred
        with check:
            assert len(categorical_operators) > 0
        for op in categorical_operators:
            with check:
                assert op in {"in", "not in"}

    def test_numeric_predicates_use_comparison_operators(self) -> None:
        """Predicates from numeric features should use `'<='` or `'>'` operators."""
        # Arrange — credit risk scoring
        df = pl.DataFrame({
            "credit_score": [
                720.0,
                650.0,
                580.0,
                800.0,
                690.0,
                540.0,
                760.0,
                620.0,
                700.0,
                480.0,
                740.0,
                560.0,
                780.0,
                640.0,
                520.0,
                810.0,
                670.0,
                590.0,
                790.0,
                630.0,
                510.0,
                770.0,
                660.0,
                730.0,
                570.0,
                820.0,
                685.0,
                545.0,
                765.0,
                615.0,
                710.0,
                655.0,
                575.0,
                795.0,
                685.0,
                535.0,
                755.0,
                625.0,
                705.0,
                475.0,
                745.0,
                555.0,
                775.0,
                645.0,
                515.0,
                815.0,
                675.0,
                585.0,
                785.0,
                635.0,
            ],
            "loan_amount": [
                5000,
                8000,
                12000,
                3000,
                7000,
                15000,
                4000,
                9000,
                6000,
                20000,
                4500,
                13000,
                3500,
                8500,
                16000,
                2500,
                7500,
                11000,
                3200,
                9500,
                18000,
                4200,
                8200,
                4800,
                14000,
                2200,
                7200,
                16500,
                3800,
                9200,
                5200,
                7800,
                12500,
                2800,
                7100,
                15500,
                4100,
                8800,
                6100,
                20500,
                4600,
                13500,
                3600,
                8600,
                16500,
                2600,
                7600,
                11500,
                3100,
                9600,
            ],
            "approved": (["yes"] * 10 + ["no"] * 10) * 2 + ["yes"] * 5 + ["no"] * 5,
        })
        feature_matrix, feature_encoders = _encode_features(df, ["credit_score", "loan_amount"])
        target_array, target_mapping = _encode_target(df["approved"], "classification")
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="classification",
        )

        # Assert — predicates on numeric features must use "<=" or ">"
        numeric_feature_names = {"credit_score", "loan_amount"}
        numeric_operators: set[str] = set()
        for rule in rules:
            for predicate in rule.predicates:
                if predicate.variable in numeric_feature_names:
                    numeric_operators.add(predicate.operator)

        with check:
            assert len(numeric_operators) > 0
        for op in numeric_operators:
            with check:
                assert op in {"<=", ">"}

    def test_predicate_eval_matches_tree_logic(self) -> None:
        """Evaluating all predicates in a rule against a known sample should match the tree's prediction.

        For a given sample, exactly one rule's predicate conjunction should evaluate
        to `True` (all predicates satisfied), and that rule's prediction should match
        what the tree's `predict()` returns for that sample.
        """
        # Arrange — medical test outcome classification
        df = pl.DataFrame({
            "age": [
                25.0,
                30.0,
                35.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                40.0,
                28.0,
                32.0,
                38.0,
                48.0,
                52.0,
                57.0,
                62.0,
                68.0,
                72.0,
                42.0,
                26.0,
                31.0,
                36.0,
                46.0,
                51.0,
                56.0,
                61.0,
                66.0,
                71.0,
                41.0,
                27.0,
                33.0,
                37.0,
                47.0,
                53.0,
                58.0,
                63.0,
                67.0,
                73.0,
                43.0,
                29.0,
                34.0,
                39.0,
                49.0,
                54.0,
                59.0,
                64.0,
                69.0,
                74.0,
                44.0,
            ],
            "bmi": [
                22.0,
                24.0,
                26.0,
                28.0,
                30.0,
                32.0,
                34.0,
                36.0,
                38.0,
                27.0,
                21.0,
                23.0,
                25.0,
                29.0,
                31.0,
                33.0,
                35.0,
                37.0,
                39.0,
                26.0,
                22.5,
                24.5,
                26.5,
                28.5,
                30.5,
                32.5,
                34.5,
                36.5,
                38.5,
                27.5,
                21.5,
                23.5,
                25.5,
                29.5,
                31.5,
                33.5,
                35.5,
                37.5,
                39.5,
                26.5,
                22.0,
                24.0,
                26.0,
                28.0,
                30.0,
                32.0,
                34.0,
                36.0,
                38.0,
                27.0,
            ],
            "outcome": (["negative"] * 10 + ["positive"] * 10) * 2 + ["negative"] * 5 + ["positive"] * 5,
        })
        feature_matrix, feature_encoders = _encode_features(df, ["age", "bmi"])
        target_array, target_mapping = _encode_target(df["outcome"], "classification")
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = _extract_rules(
            fitted_tree,
            feature_matrix,
            target_array,
            feature_encoders=feature_encoders,
            target_mapping=target_mapping,
            task_type="classification",
        )

        # Assert — for the first sample, find the matching rule and verify its prediction
        sample_index = 0
        sample_age = float(feature_matrix[sample_index, 0])
        sample_bmi = float(feature_matrix[sample_index, 1])
        tree_prediction_code = int(fitted_tree.predict(feature_matrix[sample_index : sample_index + 1])[0])
        assert target_mapping is not None
        tree_prediction_label = target_mapping[tree_prediction_code]

        matching_rules = [
            rule
            for rule in rules
            if isinstance(rule, ClassificationRule)
            and all(p.eval(sample_age if p.variable == "age" else sample_bmi) for p in rule.predicates)
        ]
        with check:
            assert len(matching_rules) == 1, "Exactly one rule should match each sample"
        if matching_rules:
            with check:
                assert matching_rules[0].prediction == tree_prediction_label


class TestComputeMetrics:
    """Tests for _compute_metrics."""

    def test_classification_metrics(self) -> None:
        """Classification metrics should include `accuracy` in the range [0, 1]."""
        # Arrange — customer churn classification
        feature_matrix, target_array = _make_churn_classification_data()
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        metrics = _compute_metrics(fitted_tree, feature_matrix, target_array, task_type="classification")

        # Assert
        with check:
            assert "accuracy" in metrics
        with check:
            assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_regression_metrics(self) -> None:
        """Regression metrics should include both `r_squared` and `rmse` keys."""
        # Arrange — house price regression
        feature_matrix, target_array = _make_house_price_regression_data()
        fitted_tree = _fit_tree(feature_matrix, target_array, task_type="regression", max_depth=4, min_samples_leaf=5)

        # Act
        metrics = _compute_metrics(fitted_tree, feature_matrix, target_array, task_type="regression")

        # Assert
        with check:
            assert "r_squared" in metrics
        with check:
            assert "rmse" in metrics
        with check:
            assert metrics["rmse"] >= 0.0


class TestComputeFeatureImportance:
    """Tests for _compute_feature_importance."""

    def test_feature_importance_sums_to_one(self) -> None:
        """Importance scores from `_compute_feature_importance` should sum to approximately 1.0.

        When all features have non-zero importance, the sklearn `feature_importances_`
        attribute already sums to 1.0.
        """
        # Arrange — employee attrition classification with two clearly relevant features
        feature_matrix, target_array = _make_churn_classification_data()
        fitted_tree = _fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=4, min_samples_leaf=5
        )

        # Act
        feature_importance = _compute_feature_importance(fitted_tree, ["tenure_months", "support_tickets"])

        # Assert — importance of non-zero-importance features sums to 1.0
        total = sum(feature_importance.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_feature_importance_uses_names(self) -> None:
        """Feature importance keys should be feature names, not integer indices."""
        # Arrange — hotel booking regression
        feature_matrix, target_array = _make_house_price_regression_data()
        fitted_tree = _fit_tree(feature_matrix, target_array, task_type="regression", max_depth=3, min_samples_leaf=5)
        feature_names = ["sqft", "bedrooms"]

        # Act
        feature_importance = _compute_feature_importance(fitted_tree, feature_names)

        # Assert
        for key in feature_importance:
            with check:
                assert key in feature_names
            with check:
                assert isinstance(key, str)

    def test_zero_importance_excluded(self) -> None:
        """Features with zero importance should be absent from the returned dict.

        When the tree uses only a subset of features, the unused ones have
        zero importance and should be filtered out.
        """
        # Arrange — use only one truly informative feature alongside a noise feature;
        # we pass feature names including a fake "noise" name, but the tree was fit
        # with data where only column 0 (sqft) drives splits.
        rng = np.random.default_rng(99)
        n_rows = 80
        sqft = rng.uniform(600, 3000, n_rows)
        # Noise column: random values completely uncorrelated with target
        noise = rng.uniform(0, 1, n_rows)
        price = 100_000 + sqft * 200 + rng.normal(0, 500, n_rows)
        feature_matrix = np.column_stack([sqft, noise])
        target_array = price

        fitted_tree = _fit_tree(feature_matrix, target_array, task_type="regression", max_depth=3, min_samples_leaf=5)

        # Act — use names that include the noise feature
        feature_importance = _compute_feature_importance(fitted_tree, ["sqft", "noise_random"])

        # Assert — any feature with zero sklearn importance should not appear in the dict
        raw_importances = fitted_tree.feature_importances_
        for idx, name in enumerate(["sqft", "noise_random"]):
            if raw_importances[idx] == 0.0:
                with check:
                    assert name not in feature_importance


class TestBuildDecisionTreeResult:
    """Tests for `_build_decision_tree_result`: full pipeline orchestration."""

    def test_classification_end_to_end(self) -> None:
        """Full pipeline with a string target should return a `DecisionTreeResult` with classification rules.

        Verifies that the result has classification task type and contains `ClassificationRule`
        objects whose `predicates` field is a list of `Predicate` objects.
        """
        # Arrange — telecom customer churn
        rng = np.random.default_rng(7)
        n_rows = 80
        tenure = rng.integers(1, 72, n_rows).tolist()
        tickets = rng.integers(0, 15, n_rows).tolist()
        churn = ["churned" if t < 12 and s > 5 else "retained" for t, s in zip(tenure, tickets, strict=True)]
        df = pl.DataFrame({
            "tenure_months": tenure,
            "support_tickets": tickets,
            "churn": churn,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "churn",
            features=["tenure_months", "support_tickets"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert
        with check:
            assert result.task_type == "classification"
        with check:
            assert result.target == "churn"
        with check:
            assert len(result.rules) > 0
        for rule in result.rules:
            with check:
                assert isinstance(rule, ClassificationRule)
            with check:
                assert isinstance(rule.predicates, list)
            with check:
                assert all(isinstance(p, Predicate) for p in rule.predicates)

    def test_regression_end_to_end(self) -> None:
        """Full pipeline with a float target should return a `DecisionTreeResult` with regression rules."""
        # Arrange — real estate price prediction using a single strong predictor so the
        # pipeline never hits the zero-importance/features_used mismatch edge case.
        rng = np.random.default_rng(13)
        n_rows = 80
        sqft = rng.uniform(600, 3000, n_rows).tolist()
        # Price scales linearly with sqft with small noise to ensure sqft is the sole predictor.
        price = [80_000 + s * 200 + rng.normal(0, 5_000) for s in sqft]
        df = pl.DataFrame({
            "sqft": sqft,
            "house_price": price,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "house_price",
            features=["sqft"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert
        with check:
            assert result.task_type == "regression"
        with check:
            assert "r_squared" in result.metrics
        with check:
            assert "rmse" in result.metrics
        with check:
            assert len(result.rules) > 0
        for rule in result.rules:
            with check:
                assert isinstance(rule, RegressionRule)

    def test_auto_selects_all_features(self) -> None:
        """When `features=None`, all non-target columns should be used as features.

        Both non-target columns are made predictive so neither has zero importance,
        avoiding the pipeline constraint that requires all `features_used` entries
        to appear in `feature_importance`.
        """
        # Arrange — hospital readmission: both age and bmi drive the label so
        # the tree is guaranteed to assign non-zero importance to each.
        rng = np.random.default_rng(17)
        n_rows = 70
        age = rng.integers(20, 80, n_rows).tolist()
        bmi_vals = rng.uniform(18.0, 40.0, n_rows).tolist()
        # Readmitted when EITHER age is high OR bmi is high, making both features useful.
        readmitted = ["yes" if a > 60 or b > 33.0 else "no" for a, b in zip(age, bmi_vals, strict=True)]
        df = pl.DataFrame({
            "age": age,
            "bmi": bmi_vals,
            "readmitted": readmitted,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "readmitted",
            features=None,
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert — both non-target columns appear in features_used or features_excluded
        all_accounted = set(result.features_used) | {exc.partition(" (")[0] for exc in result.features_excluded}
        with check:
            assert "age" in all_accounted
        with check:
            assert "bmi" in all_accounted
        with check:
            assert "readmitted" not in all_accounted

    def test_task_type_override(self) -> None:
        """A numeric target with `task_type='classification'` should produce a classification result."""
        # Arrange — satisfaction rating 1-5 treated as classification
        rng = np.random.default_rng(21)
        n_rows = 60
        wait_time = rng.integers(1, 30, n_rows).tolist()
        rating = [1 if w > 20 else (3 if w > 10 else 5) for w in wait_time]
        df = pl.DataFrame({
            "wait_time_minutes": wait_time,
            "satisfaction_rating": rating,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "satisfaction_rating",
            features=["wait_time_minutes"],
            max_depth=3,
            min_samples_leaf=5,
            task_type="classification",
        )

        # Assert
        with check:
            assert result.task_type == "classification"

    def test_error_target_not_found(self) -> None:
        """A target column that does not exist should raise a `ValueError` with a descriptive message."""
        # Arrange
        df = pl.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [30000, 50000, 60000, 70000, 80000],
        })

        # Act / Assert
        with pytest.raises(ValueError, match="nonexistent_target"):
            _build_decision_tree_result(
                df,
                "nonexistent_target",
                features=None,
                max_depth=3,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_error_feature_not_found(self) -> None:
        """A requested feature column that does not exist should raise a `ValueError`."""
        # Arrange
        df = pl.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
        })

        # Act / Assert
        with pytest.raises(ValueError, match="missing_feature"):
            _build_decision_tree_result(
                df,
                "label",
                features=["age", "missing_feature"],
                max_depth=3,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_error_all_features_excluded(self) -> None:
        """When all feature columns are filtered out, a `ValueError` about no valid features should be raised."""
        # Arrange — constant column (zero variance) is the only feature
        df = pl.DataFrame({
            "constant_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "outcome": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        })

        # Act / Assert
        with pytest.raises(ValueError, match="No valid feature columns"):
            _build_decision_tree_result(
                df,
                "outcome",
                features=["constant_col"],
                max_depth=3,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_error_target_all_null(self) -> None:
        """A target column containing only null values should raise a `ValueError`."""
        # Arrange
        df = pl.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "null_target": pl.Series([None, None, None, None, None, None, None, None], dtype=pl.String),
        })

        # Act / Assert
        with pytest.raises(ValueError, match="null_target"):
            _build_decision_tree_result(
                df,
                "null_target",
                features=["age"],
                max_depth=3,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_error_target_single_value(self) -> None:
        """A target column with only one unique non-null value should raise a `ValueError`."""
        # Arrange — all patients have the same diagnosis
        df = pl.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "diagnosis": ["healthy"] * 8,
        })

        # Act / Assert
        with pytest.raises(ValueError, match="diagnosis"):
            _build_decision_tree_result(
                df,
                "diagnosis",
                features=["age"],
                max_depth=3,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_error_too_few_samples(self) -> None:
        """A DataFrame with fewer rows than `min_samples_leaf` should raise a `ValueError`."""
        # Arrange — only 3 rows but min_samples_leaf=10
        df = pl.DataFrame({
            "score": [85.0, 90.0, 75.0],
            "grade": ["B", "A", "C"],
        })

        # Act / Assert
        with pytest.raises(ValueError, match="min_samples_leaf"):
            _build_decision_tree_result(
                df,
                "grade",
                features=["score"],
                max_depth=3,
                min_samples_leaf=10,
                task_type=None,
            )

    def test_error_max_depth_above_maximum(self) -> None:
        """Passing `max_depth=10` should raise a `ValueError` citing the valid range."""
        # Arrange
        df = pl.DataFrame({
            "score": [85.0, 90.0, 75.0, 60.0, 55.0, 70.0, 80.0, 65.0],
            "grade": ["B", "A", "C", "D", "F", "C", "B", "D"],
        })

        # Act / Assert
        with pytest.raises(ValueError, match="max_depth"):
            _build_decision_tree_result(
                df,
                "grade",
                features=["score"],
                max_depth=10,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_error_max_depth_below_minimum(self) -> None:
        """Passing `max_depth=0` should raise a `ValueError` citing the valid range."""
        # Arrange
        df = pl.DataFrame({
            "score": [85.0, 90.0, 75.0, 60.0, 55.0, 70.0, 80.0, 65.0],
            "grade": ["B", "A", "C", "D", "F", "C", "B", "D"],
        })

        # Act / Assert
        with pytest.raises(ValueError, match="max_depth"):
            _build_decision_tree_result(
                df,
                "grade",
                features=["score"],
                max_depth=0,
                min_samples_leaf=2,
                task_type=None,
            )

    def test_max_depth_at_maximum_accepted(self) -> None:
        """Passing `max_depth=6` (the maximum) should succeed without error.

        Uses a single strongly predictive feature to ensure the tree can grow
        deep enough while avoiding multi-feature importance rounding issues.
        """
        # Arrange — generate 200 samples with a single continuous feature; using one
        # feature guarantees importance sums cleanly to 1.0 and the tree will
        # grow towards depth 6.
        rng = np.random.default_rng(31)
        n_rows = 200
        score = rng.uniform(0, 100, n_rows).tolist()
        # Many fine-grained labels to force deep splitting
        grade = ["A" if s >= 90 else "B" if s >= 80 else "C" if s >= 70 else "D" if s >= 60 else "F" for s in score]
        df = pl.DataFrame({
            "score": score,
            "grade": grade,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "grade",
            features=["score"],
            max_depth=6,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert
        with check:
            assert result.depth <= 6

    def test_handles_nulls_in_features(self) -> None:
        """Feature columns with some null values should be handled without error.

        Uses a single feature with partial nulls so the pipeline processes the NaN-
        filled matrix without encountering the zero-importance mismatch edge case.
        """
        # Arrange — patient risk scoring with some missing age values
        rng = np.random.default_rng(37)
        n_rows = 60
        # ~20% of rows have null age; the rest drive the label
        age: list[int | None] = [int(rng.integers(20, 75)) if rng.random() > 0.2 else None for _ in range(n_rows)]
        # Label based on non-null age values; null-age rows get a random label
        health_risk = ["high_risk" if a is not None and a > 55 else "low_risk" for a in age]
        df = pl.DataFrame({
            "age": age,
            "health_risk": health_risk,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "health_risk",
            features=["age"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert — null rows in features must not drop rows; all n_rows are used for fitting
        with check:
            assert result.sample_count == n_rows, "null feature values must not drop rows from training"

    def test_drops_null_target_rows(self) -> None:
        """Rows with null target values should be excluded from training, reducing sample count."""
        # Arrange — loan applications with some missing approval decisions
        rng = np.random.default_rng(41)
        n_rows = 70
        income = rng.integers(20000, 100000, n_rows).tolist()
        # 10 target values are null
        loan_status: list[str | None] = [
            "approved" if rng.random() > 0.4 else "rejected" for _ in range(n_rows - 10)
        ] + [None] * 10
        df = pl.DataFrame({
            "annual_income": income,
            "loan_status": loan_status,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "loan_status",
            features=["annual_income"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert — sample_count should equal n_rows minus the 10 null target rows
        with check:
            assert result.sample_count == n_rows - 10

    def test_boolean_target_is_classification(self) -> None:
        """A boolean target column should be auto-detected as `'classification'`."""
        # Arrange — email spam detection
        rng = np.random.default_rng(47)
        n_rows = 70
        word_count = rng.integers(5, 500, n_rows).tolist()
        has_links = rng.integers(0, 10, n_rows).tolist()
        is_spam = [wc > 300 and hl > 5 for wc, hl in zip(word_count, has_links, strict=True)]
        df = pl.DataFrame({
            "word_count": word_count,
            "link_count": has_links,
            "is_spam": is_spam,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "is_spam",
            features=["word_count", "link_count"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert
        with check:
            assert result.task_type == "classification"

    def test_mixed_feature_types(self) -> None:
        """A DataFrame with numeric and string features should produce a valid result.

        Uses two features that both drive the label so neither has zero importance,
        avoiding the pipeline constraint that requires all `features_used` entries
        to appear in `feature_importance`. One numeric and one categorical feature
        exercise the mixed-type encoding path.
        """
        # Arrange — e-commerce order delivery: both order_value and shipping_method
        # contribute to the label so the tree uses both features.
        n_rows = 70
        # Interleave values so each shipping method has a distinct price range
        order_value = (
            [float(v) for v in range(10, 10 + 25 * 4, 4)]  # 25 low values
            + [float(v) for v in range(110, 110 + 22 * 4, 4)]  # 22 mid values
            + [float(v) for v in range(220, 220 + 23 * 4, 4)]  # 23 high values
        )[:n_rows]
        shipping_method = (["standard"] * 25 + ["express"] * 22 + ["overnight"] * 23)[:n_rows]
        # Label: overnight always on-time; standard on-time only for small orders
        delivered_on_time = [
            "yes" if sm == "overnight" or (sm == "express" and ov < 150) or (sm == "standard" and ov < 50) else "no"
            for ov, sm in zip(order_value, shipping_method, strict=True)
        ]
        df = pl.DataFrame({
            "order_value_usd": order_value,
            "shipping_method": shipping_method,
            "delivered_on_time": delivered_on_time,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "delivered_on_time",
            features=["order_value_usd", "shipping_method"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert
        with check:
            assert result.task_type == "classification"
        with check:
            assert result.sample_count == n_rows

    def test_result_rules_contain_predicates(self) -> None:
        """Every rule in the result's `rules` list should have a `list[Predicate]` in `predicates`."""
        # Arrange — vehicle insurance premium estimation using driver_age as the sole
        # predictor so the pipeline never hits the zero-importance/features_used mismatch.
        rng = np.random.default_rng(59)
        n_rows = 80
        driver_age = rng.integers(18, 75, n_rows).tolist()
        # Premium decreases with age (younger drivers are higher risk); small noise
        annual_premium = [500.0 + (65 - a) * 20 + rng.uniform(-30, 30) for a in driver_age]
        df = pl.DataFrame({
            "driver_age": driver_age,
            "annual_premium_usd": annual_premium,
        })

        # Act
        result = _build_decision_tree_result(
            df,
            "annual_premium_usd",
            features=["driver_age"],
            max_depth=3,
            min_samples_leaf=5,
            task_type=None,
        )

        # Assert
        for rule in result.rules:
            with check:
                assert isinstance(rule.predicates, list)
            with check:
                assert all(isinstance(p, Predicate) for p in rule.predicates)


# ---------------------------------------------------------------------------
# Private test helpers
# ---------------------------------------------------------------------------


def _make_churn_classification_data(n_rows: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Build a simple classification feature matrix and target vector for churn prediction.

    Generates separable data: customers with low tenure and high support tickets
    churn; others are retained.

    Args:
        n_rows (int): Number of rows to generate. Must be even. Defaults to 80.

    Returns:
        tuple[np.ndarray, np.ndarray]: A 2-tuple of `(feature_matrix, target_array)`.
            `feature_matrix` has shape `(n_rows, 2)` with columns
            `[tenure_months, support_tickets]`. `target_array` has shape
            `(n_rows,)` with float-encoded labels: `0.0` = churned, `1.0` = retained.
    """
    half = n_rows // 2
    # First half: short tenure, many tickets → churned (0)
    churned_features = np.column_stack([
        np.random.default_rng(0).uniform(1, 6, half),
        np.random.default_rng(1).uniform(5, 15, half),
    ])
    # Second half: long tenure, few tickets → retained (1)
    retained_features = np.column_stack([
        np.random.default_rng(2).uniform(24, 60, half),
        np.random.default_rng(3).uniform(0, 2, half),
    ])
    feature_matrix = np.vstack([churned_features, retained_features])
    target_array = np.array([0.0] * half + [1.0] * half)
    return feature_matrix, target_array


def _make_house_price_regression_data(n_rows: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Build a simple regression feature matrix and target vector for house price prediction.

    Generates data where house price scales linearly with square footage plus noise.

    Args:
        n_rows (int): Number of rows to generate. Defaults to 80.

    Returns:
        tuple[np.ndarray, np.ndarray]: A 2-tuple of `(feature_matrix, target_array)`.
            `feature_matrix` has shape `(n_rows, 2)` with columns
            `[sqft, bedrooms]`. `target_array` has shape `(n_rows,)` with
            house prices in dollars.
    """
    rng = np.random.default_rng(42)
    sqft = rng.uniform(600, 3000, n_rows)
    bedrooms = rng.integers(1, 6, n_rows).astype(np.float64)
    price = 100_000 + sqft * 150 + bedrooms * 10_000 + rng.normal(0, 15_000, n_rows)
    feature_matrix = np.column_stack([sqft, bedrooms])
    return feature_matrix, price
