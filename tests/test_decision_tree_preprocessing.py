"""Tests for the decision tree preprocessing pipeline: column classification, filtering, encoding."""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest
from pytest_check import check

from dfkit.decision_tree.preprocessing import (
    ExcludedFeature,
    classify_column,
    detect_task_type,
    encode_features,
    encode_target,
    filter_features,
)

# ---------------------------------------------------------------------------
# Phase 2: Preprocessing Pipeline
# ---------------------------------------------------------------------------


class TestClassifyColumn:
    """Tests for classify_column: maps Polars dtypes to broad feature categories."""

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
        column_type = classify_column(dtype)

        # Assert
        assert column_type == "numeric"

    def test_boolean_type_returns_boolean(self) -> None:
        """Boolean dtype should classify as 'boolean'."""
        # Act
        column_type = classify_column(pl.Boolean)  # type: ignore[arg-type]

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
        column_type = classify_column(dtype)

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
        column_type = classify_column(dtype)

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
        column_type = classify_column(dtype)

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
        column_type = classify_column(dtype)

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
        column_type = classify_column(dtype)

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
        column_type = classify_column(dtype)

        # Assert
        assert column_type == "excluded"


class TestFilterFeatures:
    """Tests for filter_features: partition feature columns into kept and excluded sets."""

    def test_keeps_valid_features(self) -> None:
        """Numeric and categorical columns with variance should pass through to the kept list."""
        # Arrange
        df = pl.DataFrame({
            "age": pl.Series([25, 42, 31, 58, 19], dtype=pl.Int32),
            "status": pl.Series(["active", "inactive", "active", "inactive", "active"]),
        })

        # Act
        kept, excluded = filter_features(df, ["age", "status"])

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
        kept, excluded = filter_features(df, ["revenue", "missing_field"])

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
        kept, excluded = filter_features(df, ["score", "constant_flag"])

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
        kept, excluded = filter_features(df, ["near_unique_col", "score"])

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
        kept, excluded = filter_features(df, ["near_unique_col", "score"])

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
        kept, excluded = filter_features(df, ["customer_id", "region"])

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
        kept, excluded = filter_features(df, ["valid_col", "tags"])

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
        _, excluded = filter_features(df, ["all_null_col", "constant_col", "list_col"])

        # Assert — verify all reasons are meaningful non-empty strings
        for exc in excluded:
            with check:
                assert isinstance(exc.reason, str), f"Reason for {exc.name!r} should be str"
            with check:
                assert len(exc.reason) > 0, f"Reason for {exc.name!r} should not be empty"
            with check:
                assert isinstance(exc, ExcludedFeature), f"{exc.name!r} should be ExcludedFeature"

    def test_all_features_excluded_returns_empty_kept_list(self) -> None:
        """When every feature fails at least one filter, the kept list should be empty."""
        # Arrange — one all-null column and one zero-variance column
        df = pl.DataFrame({
            "all_null": pl.Series([None, None, None, None], dtype=pl.Float64),
            "constant": pl.Series([99, 99, 99, 99], dtype=pl.Int64),
        })

        # Act
        kept, excluded = filter_features(df, ["all_null", "constant"])

        # Assert
        with check:
            assert kept == []
        with check:
            assert len(excluded) == 2


class TestDetectTaskType:
    """Tests for detect_task_type: infer classification vs regression from target series."""

    def test_string_target_is_classification(self) -> None:
        """A target series with String dtype should be detected as classification."""
        # Arrange
        target = pl.Series("churn_label", ["churned", "retained", "churned", "retained"])

        # Act
        task_type = detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_boolean_target_is_classification(self) -> None:
        """A target series with Boolean dtype should be detected as classification."""
        # Arrange
        target = pl.Series("is_fraud", [True, False, False, True, False], dtype=pl.Boolean)

        # Act
        task_type = detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_string_target_with_utf8_alias_is_classification(self) -> None:
        """A target series created with pl.Utf8 (alias for String) should be detected as classification."""
        # Arrange — pl.Utf8 and pl.String are the same dtype
        target = pl.Series("churn_status", ["churned", "retained", "retained"], dtype=pl.Utf8)

        # Act
        task_type = detect_task_type(target, None)

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
        task_type = detect_task_type(target, None)

        # Assert
        assert task_type == "regression"

    def test_low_cardinality_int_is_classification(self) -> None:
        """An integer series with 5 unique values across 200 rows should be classification.

        Low cardinality (≤20 unique values) triggers the classification heuristic.
        """
        # Arrange — 5 unique labels repeated across 200 rows
        target = pl.Series("rating", list(range(1, 6)) * 40, dtype=pl.Int32)

        # Act
        task_type = detect_task_type(target, None)

        # Assert
        assert task_type == "classification"

    def test_high_cardinality_int_is_regression(self) -> None:
        """An integer series with 150 unique values should be detected as regression.

        High cardinality (>20 unique values and >5% unique ratio) triggers regression.
        """
        # Arrange — 150 distinct integers, unique ratio = 1.0 >> 0.05 threshold
        target = pl.Series("transaction_id", list(range(150)), dtype=pl.Int64)

        # Act
        task_type = detect_task_type(target, None)

        # Assert
        assert task_type == "regression"

    def test_override_to_classification(self) -> None:
        """A float target with a 'classification' override should return 'classification'."""
        # Arrange
        target = pl.Series("score", [1.0, 2.0, 3.0, 1.0, 2.0], dtype=pl.Float32)

        # Act
        task_type = detect_task_type(target, "classification")

        # Assert
        assert task_type == "classification"

    def test_override_to_regression(self) -> None:
        """A string target with a 'regression' override should return 'regression'."""
        # Arrange
        target = pl.Series("label", ["yes", "no", "yes", "yes"])

        # Act
        task_type = detect_task_type(target, "regression")

        # Assert
        assert task_type == "regression"

    def test_auto_override_behaves_same_as_none(self) -> None:
        """Both None and 'auto' should trigger automatic detection and produce the same result."""
        # Arrange
        target = pl.Series("category", ["A", "B", "A", "C", "B"])

        # Act
        task_type_none = detect_task_type(target, None)
        task_type_auto = detect_task_type(target, "auto")

        # Assert
        with check:
            assert task_type_none == "classification"
        with check:
            assert task_type_none == task_type_auto


class TestEncodeFeatures:
    """Tests for encode_features: encode DataFrame columns into a float64 numpy matrix."""

    def test_numeric_passthrough_preserves_values_and_nulls_become_nan(self) -> None:
        """Numeric columns should be cast to float64; Polars nulls must become NaN in the matrix."""
        # Arrange
        df = pl.DataFrame({
            "weight_kg": pl.Series([55.5, 72.0, None, 90.1], dtype=pl.Float64),
        })

        # Act
        matrix, encoders = encode_features(df, ["weight_kg"])

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
        matrix, encoders = encode_features(df, ["has_subscription"])

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
        matrix, encoders = encode_features(df, ["fruit"])

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
        matrix, encoders = encode_features(df, ["event_date"])

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
        matrix, _encoders = encode_features(df, ["response_time"])

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
        matrix, encoders = encode_features(df, ["age", "is_premium", "plan"])

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
        matrix, encoders = encode_features(df, ["department"])

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
    """Tests for encode_target: encode the target column for regression or classification."""

    def test_regression_target_passthrough_as_float64(self) -> None:
        """Regression targets should be returned as a float64 numpy array with no category mapping."""
        # Arrange
        target = pl.Series("house_price", [285_000.0, 420_500.0, 175_000.0, 610_200.0])

        # Act
        encoded_array, category_mapping = encode_target(target, "regression")

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
        encoded_array, category_mapping = encode_target(target, "classification")

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
        _, category_mapping = encode_target(target, "classification")

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
        encoded_array, category_mapping = encode_target(target, "regression")

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
