"""Tests for the ClassificationRule, RegressionRule, and DecisionTreeResult Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest_check import check

from dfkit.decision_tree_module import (
    ClassificationRule,
    DecisionTreeResult,
    DecisionTreeRule,
    RegressionRule,
)


class TestDecisionTreeRule:
    """Tests for the ClassificationRule and RegressionRule discriminated union."""

    def test_classification_rule_construction_sets_correct_fields(self) -> None:
        """Classification rule should populate all fields from constructor arguments.

        Verifies that a rule built from a churn-prediction leaf node populates
        every field correctly via the ClassificationRule concrete type.
        """
        # Arrange
        conditions = ["tenure_months > 12", "support_tickets <= 2", "monthly_charges <= 75.0"]

        # Act
        rule = ClassificationRule(
            task_type="classification",
            conditions=conditions,
            prediction="retained",
            samples=342,
            confidence=0.91,
        )

        # Assert
        with check:
            assert rule.task_type == "classification"
        with check:
            assert rule.conditions == conditions
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
        conditions = ["years_experience > 5", "education_level <= 2", "department == Engineering"]

        # Act
        rule = RegressionRule(
            task_type="regression",
            conditions=conditions,
            prediction=112_500.0,
            samples=87,
            std=18_430.75,
        )

        # Assert
        with check:
            assert rule.task_type == "regression"
        with check:
            assert rule.conditions == conditions
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
                    "conditions": ["credit_score > 700", "loan_amount <= 50000"],
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
                    "conditions": ["years_experience > 8", "management_level == True"],
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
        rule_class: type[ClassificationRule] | type[RegressionRule],
        kwargs: dict,
        expected_model: type[ClassificationRule] | type[RegressionRule],
    ) -> None:
        """Serializing a rule to a dict and validating it back should produce an equal model.

        Exercises model_dump followed by model_validate to confirm that no field
        is lost or mutated during a round-trip through the plain-Python representation,
        for both ClassificationRule and RegressionRule.

        Args:
            rule_class (type[ClassificationRule] | type[RegressionRule]): The rule class to construct.
            kwargs (dict): Constructor keyword arguments for the rule.
            expected_model (type[ClassificationRule] | type[RegressionRule]): The model class used for validation.
        """
        # Arrange
        original = rule_class(**kwargs)

        # Act
        restored = expected_model.model_validate(original.model_dump())

        # Assert
        assert restored == original

    def test_classification_rule_with_empty_conditions_represents_single_leaf(self) -> None:
        """A classification rule with no conditions should be accepted as a valid single-leaf tree."""
        # Act
        rule = ClassificationRule(
            task_type="classification",
            conditions=[],
            prediction="no_churn",
            samples=1000,
            confidence=0.95,
        )

        # Assert
        with check:
            assert rule.conditions == []
        with check:
            assert rule.prediction == "no_churn"
        with check:
            assert rule.samples == 1000

    @pytest.mark.parametrize(
        ("confidence", "conditions", "prediction", "samples"),
        [
            (0.0, ["account_age_days <= 30"], "high_risk", 45),
            (1.0, ["account_age_days > 365", "payment_failures == 0"], "low_risk", 312),
        ],
    )
    def test_classification_rule_with_boundary_confidence_values(
        self,
        confidence: float,
        conditions: list[str],
        prediction: str,
        samples: int,
    ) -> None:
        """Confidence values of exactly 0.0 and 1.0 should both be accepted on a ClassificationRule.

        Args:
            confidence (float): The boundary confidence value to test (0.0 or 1.0).
            conditions (list[str]): Path conditions for the rule.
            prediction (str): Predicted class label for the rule.
            samples (int): Number of training samples at this leaf.
        """
        # Act
        rule = ClassificationRule(
            task_type="classification",
            conditions=conditions,
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
            conditions=["contract_type == fixed_rate"],
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
                conditions=["product_category == electronics"],
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
                conditions=["years_experience > 5"],
                prediction=95_000.0,
                samples=120,
            )

    def test_classification_rule_rejects_negative_samples(self) -> None:
        """A classification rule with negative samples should raise a ValidationError."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError, match="samples"):
            ClassificationRule(
                task_type="classification",
                conditions=["purchase_count > 5"],
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
                conditions=["tenure_months > 6"],
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
                conditions=["days_since_login <= 7"],
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
                conditions=["years_experience > 5"],
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
                conditions=["tenure_months <= 6"],
                prediction="churned",
                samples=210,
                confidence=0.87,
            ),
            ClassificationRule(
                task_type="classification",
                conditions=["tenure_months > 6", "support_tickets <= 1"],
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
                conditions=["years_experience <= 2"],
                prediction=54_200.0,
                samples=143,
                std=6_810.5,
            ),
            RegressionRule(
                task_type="regression",
                conditions=["years_experience > 2", "years_experience <= 8"],
                prediction=85_750.0,
                samples=298,
                std=11_340.0,
            ),
            RegressionRule(
                task_type="regression",
                conditions=["years_experience > 8", "management_level == True"],
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

    @pytest.mark.parametrize(
        "original",
        [
            DecisionTreeResult(
                target="price_tier",
                task_type="classification",
                features_used=["sqft", "bedrooms", "neighbourhood_score"],
                features_excluded=["listing_url (free-text)"],
                rules=[
                    ClassificationRule(
                        task_type="classification",
                        conditions=["sqft <= 800", "neighbourhood_score <= 5"],
                        prediction="budget",
                        samples=405,
                        confidence=0.78,
                    ),
                    ClassificationRule(
                        task_type="classification",
                        conditions=["sqft > 800", "bedrooms >= 3"],
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
            ),
            DecisionTreeResult(
                target="house_price",
                task_type="regression",
                features_used=["sqft", "bedrooms", "garage_spaces"],
                features_excluded=["listing_id (unique identifier)"],
                rules=[
                    RegressionRule(
                        task_type="regression",
                        conditions=["sqft <= 1200"],
                        prediction=285_000.0,
                        samples=230,
                        std=42_500.0,
                    ),
                    RegressionRule(
                        task_type="regression",
                        conditions=["sqft > 1200", "bedrooms >= 3"],
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
            ),
        ],
    )
    def test_result_serialization_roundtrip_produces_identical_object(
        self,
        original: DecisionTreeResult,
    ) -> None:
        """Serializing a result to a dict and validating it back should produce an equal model.

        Exercises model_dump followed by model_validate to confirm that no field
        is lost or mutated during a round-trip, including nested rule objects
        embedded in the rules list, for both classification and regression results.

        Args:
            original (DecisionTreeResult): A fully constructed result to round-trip through serialization.
        """
        # Act
        restored = DecisionTreeResult.model_validate(original.model_dump())

        # Assert
        assert restored == original

    def test_result_with_rules_matching_leaf_count_is_accepted(self) -> None:
        """A result where len(rules) equals leaf_count should be accepted.

        Verifies the invariant that every leaf node has a corresponding rule
        when the counts match exactly.
        """
        # Arrange
        rule = ClassificationRule(
            task_type="classification",
            conditions=[],
            prediction="no_churn",
            samples=500,
            confidence=0.95,
        )

        # Act
        tree_result = DecisionTreeResult(
            target="subscription_tier",
            task_type="classification",
            features_used=["monthly_spend"],
            features_excluded=[],
            rules=[rule],
            feature_importance={"monthly_spend": 1.0},
            metrics={"accuracy": 0.95},
            sample_count=500,
            depth=0,
            leaf_count=1,
        )

        # Assert
        with check:
            assert len(tree_result.rules) == tree_result.leaf_count

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
                        conditions=["sqft <= 800"],
                        prediction="budget",
                        samples=200,
                        confidence=0.80,
                    ),
                    ClassificationRule(
                        task_type="classification",
                        conditions=["sqft > 800"],
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
                    conditions=[],
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
                        conditions=["age > 30"],
                        prediction="group_a",
                        samples=100,
                        confidence=0.75,
                    )
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
                        conditions=[],
                        prediction="no_fraud",
                        samples=200,
                        confidence=0.9,
                    )
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
                        conditions=[],
                        prediction="converted",
                        samples=1,
                        confidence=1.0,
                    )
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
                        conditions=[],
                        prediction="retained",
                        samples=300,
                        confidence=0.80,
                    )
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
                    conditions=[],
                    prediction="retained",
                    samples=500,
                    confidence=0.82,
                )
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
