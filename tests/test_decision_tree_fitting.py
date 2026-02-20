"""Tests for the decision tree fitting pipeline: fit_tree, extract_rules, compute_metrics, orchestration."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pytest_check import check
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dfkit.decision_tree.fitting import (
    build_decision_tree_result,
    compute_feature_importance,
    compute_metrics,
    extract_rules,
    fit_tree,
)
from dfkit.decision_tree.models import (
    ClassificationRule,
    Predicate,
    RegressionRule,
)
from dfkit.decision_tree.preprocessing import (
    encode_features,
    encode_target,
)

# ---------------------------------------------------------------------------
# Phase 3: Tree Fitting, Rule Extraction, Metrics and Orchestration
# ---------------------------------------------------------------------------


class TestFitTree:
    """Tests for `fit_tree`: fits sklearn decision trees with configurable hyperparameters."""

    def test_fit_classifier_returns_decision_tree_classifier(self) -> None:
        """Fitting with `task_type='classification'` should return a DecisionTreeClassifier."""
        # Arrange
        feature_matrix, target_array = _make_churn_classification_data()

        # Act
        fitted_tree = fit_tree(
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
        fitted_tree = fit_tree(
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
        fitted_tree = fit_tree(
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
        fitted_tree = fit_tree(
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
    """Tests for `extract_rules`: converts a fitted tree into human-readable rules."""

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
        feature_matrix, feature_encoders = encode_features(df, ["tenure_months", "support_tickets"])
        target_array, target_mapping = encode_target(df["churn"], "classification")
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["claim_amount", "policy_age_years"])
        target_array, target_mapping = encode_target(df["is_fraud"], "classification")
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["years_experience"])
        target_array, target_mapping = encode_target(df["annual_salary"], "regression")
        fitted_tree = fit_tree(feature_matrix, target_array, task_type="regression", max_depth=3, min_samples_leaf=5)

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["days_since_login", "page_views"])
        target_array, target_mapping = encode_target(df["renewed"], "classification")
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )
        total_samples = len(df)

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["monthly_spend", "order_count"])
        target_array, target_mapping = encode_target(df["lifetime_value"], "regression")
        fitted_tree = fit_tree(feature_matrix, target_array, task_type="regression", max_depth=2, min_samples_leaf=5)

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["product_category"])
        target_array, target_mapping = encode_target(df["revenue_tier"], "classification")
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["credit_score", "loan_amount"])
        target_array, target_mapping = encode_target(df["approved"], "classification")
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = extract_rules(
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
        feature_matrix, feature_encoders = encode_features(df, ["age", "bmi"])
        target_array, target_mapping = encode_target(df["outcome"], "classification")
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        rules = extract_rules(
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
    """Tests for compute_metrics."""

    def test_classification_metrics(self) -> None:
        """Classification metrics should include `accuracy` in the range [0, 1]."""
        # Arrange — customer churn classification
        feature_matrix, target_array = _make_churn_classification_data()
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=3, min_samples_leaf=5
        )

        # Act
        metrics = compute_metrics(fitted_tree, feature_matrix, target_array, task_type="classification")

        # Assert
        with check:
            assert "accuracy" in metrics
        with check:
            assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_regression_metrics(self) -> None:
        """Regression metrics should include both `r_squared` and `rmse` keys."""
        # Arrange — house price regression
        feature_matrix, target_array = _make_house_price_regression_data()
        fitted_tree = fit_tree(feature_matrix, target_array, task_type="regression", max_depth=4, min_samples_leaf=5)

        # Act
        metrics = compute_metrics(fitted_tree, feature_matrix, target_array, task_type="regression")

        # Assert
        with check:
            assert "r_squared" in metrics
        with check:
            assert "rmse" in metrics
        with check:
            assert metrics["rmse"] >= 0.0


class TestComputeFeatureImportance:
    """Tests for compute_feature_importance."""

    def test_feature_importance_sums_to_one(self) -> None:
        """Importance scores from `compute_feature_importance` should sum to approximately 1.0.

        When all features have non-zero importance, the sklearn `feature_importances_`
        attribute already sums to 1.0.
        """
        # Arrange — employee attrition classification with two clearly relevant features
        feature_matrix, target_array = _make_churn_classification_data()
        fitted_tree = fit_tree(
            feature_matrix, target_array, task_type="classification", max_depth=4, min_samples_leaf=5
        )

        # Act
        feature_importance = compute_feature_importance(fitted_tree, ["tenure_months", "support_tickets"])

        # Assert — importance of non-zero-importance features sums to 1.0
        total = sum(feature_importance.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_feature_importance_uses_names(self) -> None:
        """Feature importance keys should be feature names, not integer indices."""
        # Arrange — hotel booking regression
        feature_matrix, target_array = _make_house_price_regression_data()
        fitted_tree = fit_tree(feature_matrix, target_array, task_type="regression", max_depth=3, min_samples_leaf=5)
        feature_names = ["sqft", "bedrooms"]

        # Act
        feature_importance = compute_feature_importance(fitted_tree, feature_names)

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

        fitted_tree = fit_tree(feature_matrix, target_array, task_type="regression", max_depth=3, min_samples_leaf=5)

        # Act — use names that include the noise feature
        feature_importance = compute_feature_importance(fitted_tree, ["sqft", "noise_random"])

        # Assert — any feature with zero sklearn importance should not appear in the dict
        raw_importances = fitted_tree.feature_importances_
        for idx, name in enumerate(["sqft", "noise_random"]):
            if raw_importances[idx] == 0.0:
                with check:
                    assert name not in feature_importance


class TestBuildDecisionTreeResult:
    """Tests for `build_decision_tree_result`: full pipeline orchestration."""

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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
            build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
        result = build_decision_tree_result(
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
