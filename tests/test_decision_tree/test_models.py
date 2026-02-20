"""Tests for the decision tree models: Predicate, ClassificationRule, RegressionRule, DecisionTreeResult."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pytest_check import check

from dfkit.decision_tree.models import (
    ClassificationRule,
    DecisionTreeResult,
    Predicate,
    RegressionRule,
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
