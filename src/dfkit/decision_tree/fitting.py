"""Decision tree fitting, rule extraction, metrics computation, and pipeline orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dfkit.decision_tree.models import (
    ClassificationRule,
    DecisionTree,
    DecisionTreeResult,
    DecisionTreeRule,
    DecisionTreeTask,
    Predicate,
    RegressionRule,
)
from dfkit.decision_tree.preprocessing import (
    THRESHOLD_DECIMAL_PLACES,
    ExcludedFeature,
    FeatureEncoder,
    detect_task_type,
    encode_features,
    encode_target,
    filter_features,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MIN_TREE_DEPTH: int = 1  # Lower bound on tree depth accepted by the public API.
_MAX_TREE_DEPTH: int = 6  # Caps tree depth to prevent overfitting and keep rules human-readable.
_AUTO_MIN_SAMPLES_FRACTION: float = 0.02  # Scales the leaf floor with dataset size: 2% of n_rows.
_AUTO_MIN_SAMPLES_FLOOR: int = 5  # Absolute minimum leaf size regardless of dataset size.


# ---------------------------------------------------------------------------
# Public interface -- Tree fitting
# ---------------------------------------------------------------------------


def fit_tree(
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    task_type: DecisionTreeTask,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int | None = None,
) -> DecisionTree:
    """Fit a decision tree to the given feature matrix and target vector.

    Args:
        feature_matrix (np.ndarray): 2-D feature matrix with shape `(n_samples, n_features)`.
        target_array (np.ndarray): 1-D target vector with shape `(n_samples,)`.
        task_type (DecisionTreeTask): Whether to fit a classifier or regressor.
        max_depth (int): Maximum depth of the tree. Must be between
            `_MIN_TREE_DEPTH` and `_MAX_TREE_DEPTH` (1 to 6), inclusive.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        random_state (int | None): Random seed for reproducibility.

    Returns:
        DecisionTree: The fitted tree estimator.

    Raises:
        ValueError: If `max_depth` is outside the valid range
            `[_MIN_TREE_DEPTH, _MAX_TREE_DEPTH]`.
    """
    if not (_MIN_TREE_DEPTH <= max_depth <= _MAX_TREE_DEPTH):
        raise ValueError(f"max_depth must be between {_MIN_TREE_DEPTH} and {_MAX_TREE_DEPTH}, got {max_depth}.")
    tree_cls = DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor
    tree = tree_cls(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(feature_matrix, target_array)
    return tree


# ---------------------------------------------------------------------------
# Public interface -- Rule extraction
# ---------------------------------------------------------------------------


def extract_rules(
    tree: DecisionTree,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    feature_encoders: list[FeatureEncoder],
    target_mapping: dict[int, str] | None,
    task_type: DecisionTreeTask,
) -> list[ClassificationRule] | list[RegressionRule]:
    """Extract human-readable rules from a fitted decision tree.

    Walks the `tree.tree_` internal structure recursively, building one
    `ClassificationRule` or `RegressionRule` per leaf node.

    For categorical features, the sklearn threshold is decoded back to a set
    of category labels: categories with ordinal codes `<= threshold` go to the
    left branch (`"in"` predicate) and those with codes `> threshold` go to
    the right branch (also an `"in"` predicate over the complementary label
    set, for symmetry and clarity).

    For regression tasks, the per-leaf standard deviation is computed from the
    actual training samples assigned to each leaf via `tree.apply(feature_matrix)`.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_matrix (np.ndarray): 2-D feature matrix used to fit the tree.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        feature_encoders (list[FeatureEncoder]): Parallel list of encoders
            describing how each feature column was encoded.
        target_mapping (dict[int, str] | None): Maps integer codes back to
            class labels for classification; `None` for regression.
        task_type (DecisionTreeTask): Whether the tree is a classifier or regressor.

    Returns:
        list[ClassificationRule] | list[RegressionRule]: One rule per leaf node.
    """
    leaf_assignments = tree.apply(feature_matrix) if task_type == "regression" else None
    rules: list[ClassificationRule] | list[RegressionRule] = []
    _walk_tree(
        tree=tree,
        target_array=target_array,
        feature_encoders=feature_encoders,
        target_mapping=target_mapping,
        task_type=task_type,
        leaf_assignments=leaf_assignments,
        node_id=0,
        path_predicates=[],
        rules=rules,
    )
    return rules


# ---------------------------------------------------------------------------
# Public interface -- Metrics and feature importance
# ---------------------------------------------------------------------------


def compute_metrics(
    tree: DecisionTree,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    task_type: DecisionTreeTask,
) -> dict[str, float]:
    """Compute evaluation metrics for a fitted decision tree.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_matrix (np.ndarray): 2-D feature matrix with shape `(n_samples, n_features)`.
        target_array (np.ndarray): 1-D target vector with shape `(n_samples,)`.
        task_type (DecisionTreeTask): Whether the tree is a classifier or regressor.

    Returns:
        dict[str, float]: For classification: `{"accuracy": <float>}`.
            For regression: `{"r_squared": <float>, "rmse": <float>}`.
    """
    predictions = tree.predict(feature_matrix)
    if task_type == "classification":
        return {"accuracy": float(accuracy_score(target_array, predictions))}
    return {
        "r_squared": float(r2_score(target_array, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target_array, predictions))),
    }


def compute_feature_importance(
    tree: DecisionTree,
    feature_names: list[str],
) -> dict[str, float]:
    """Build a feature importance mapping from a fitted decision tree.

    Only features with non-zero importance are included. The remaining
    importances are renormalized to sum to 1.0 after excluding zero-importance
    features.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_names (list[str]): Feature names parallel to `tree.feature_importances_`.

    Returns:
        dict[str, float]: Mapping of feature name to rounded importance score,
            sorted in descending order of importance. Keys are exactly those
            features with importance > 0.
    """
    importances = tree.feature_importances_
    paired = [(name, round(float(importance), 4)) for name, importance in zip(feature_names, importances, strict=True)]
    filtered = [(name, importance) for name, importance in paired if importance > 0.0]
    filtered.sort(key=lambda item: item[1], reverse=True)
    total_rounded = sum(importance for _, importance in filtered)
    renormalized = [(name, round(importance / total_rounded, 4)) for name, importance in filtered]
    if renormalized:
        others_sum = sum(importance for _, importance in renormalized[:-1])
        last_name = renormalized[-1][0]
        renormalized[-1] = (last_name, round(1.0 - others_sum, 4))
    return dict(renormalized)


# ---------------------------------------------------------------------------
# Public interface -- Pipeline orchestration
# ---------------------------------------------------------------------------


def build_decision_tree_result(
    df: pl.DataFrame,
    target: str,
    *,
    features: list[str] | None,
    max_depth: int,
    min_samples_leaf: int,
    task_type: str | None,
    random_state: int | None = None,
) -> DecisionTreeResult:
    """Orchestrate the full decision tree fitting pipeline.

    Validates inputs, preprocesses features and target, fits a decision tree,
    extracts rules, computes metrics and feature importance, then assembles
    and returns a `DecisionTreeResult`.

    Args:
        df (pl.DataFrame): The source DataFrame containing features and target.
        target (str): Name of the target column.
        features (list[str] | None): Feature column names to consider. When
            `None`, all columns except `target` are used.
        max_depth (int): Maximum tree depth; must be between 1 and
            `_MAX_TREE_DEPTH` (inclusive).
        min_samples_leaf (int): Minimum samples required at a leaf node.
            Auto-adjusted upward when the dataset is large.
        task_type (str | None): `"classification"`, `"regression"`, or `None`
            for automatic detection.
        random_state (int | None): Random seed passed to the sklearn tree for
            reproducibility.  `None` means non-deterministic.

    Returns:
        DecisionTreeResult: The fitted tree result.

    Raises:
        ValueError: On any validation failure (missing columns, degenerate
            target, no valid features, or insufficient samples).
    """
    _validate_inputs(df, target, features)

    feature_columns = features if features is not None else [col for col in df.columns if col != target]

    # Drop null-target rows before feature filtering so the cardinality ratio
    # uses the same denominator as the data the tree will be fitted on.
    df_clean = _prepare_clean_dataframe(df, target, min_samples_leaf)

    kept_columns, excluded_features = filter_features(df_clean, feature_columns)

    if not kept_columns:
        excluded_labels = [f"{ef.name} ({ef.reason})" for ef in excluded_features]
        raise ValueError(f"No valid feature columns remain after filtering. Excluded: {excluded_labels}")

    n_rows = len(df_clean)
    return _fit_and_assemble_result(
        df_clean=df_clean,
        target=target,
        kept_columns=kept_columns,
        excluded_features=excluded_features,
        n_rows=n_rows,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        task_type=task_type,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _walk_tree(
    *,
    tree: DecisionTree,
    target_array: np.ndarray,
    feature_encoders: list[FeatureEncoder],
    target_mapping: dict[int, str] | None,
    task_type: DecisionTreeTask,
    leaf_assignments: np.ndarray | None,
    node_id: int,
    path_predicates: list[Predicate],
    rules: list[ClassificationRule] | list[RegressionRule],
) -> None:
    """Recursively walk a decision tree node and accumulate leaf rules.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        feature_encoders (list[FeatureEncoder]): Parallel list of encoders.
        target_mapping (dict[int, str] | None): Class label mapping for classification.
        task_type (DecisionTreeTask): Whether the tree is a classifier or regressor.
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)`, used to compute per-leaf std for regression.
        node_id (int): The current node index in `tree.tree_`.
        path_predicates (list[Predicate]): Accumulated predicates from the root
            to `node_id`.
        rules (list[ClassificationRule] | list[RegressionRule]): Accumulator
            list; leaf rules are appended in-place.
    """
    sklearn_tree = tree.tree_
    left_child = sklearn_tree.children_left[node_id]
    right_child = sklearn_tree.children_right[node_id]
    is_leaf = left_child == right_child  # Both are TREE_LEAF (-1) at leaves

    if is_leaf:
        rule = _build_leaf_rule(
            sklearn_tree=sklearn_tree,
            node_id=node_id,
            path_predicates=path_predicates,
            task_type=task_type,
            target_mapping=target_mapping,
            target_array=target_array,
            leaf_assignments=leaf_assignments,
        )
        rules.append(rule)  # type: ignore[arg-type]
        return

    feature_index = sklearn_tree.feature[node_id]
    threshold = sklearn_tree.threshold[node_id]
    encoder = feature_encoders[feature_index]
    left_predicate, right_predicate = _build_split_predicates(encoder, threshold)

    shared_kwargs: dict[str, Any] = {
        "tree": tree,
        "target_array": target_array,
        "feature_encoders": feature_encoders,
        "target_mapping": target_mapping,
        "task_type": task_type,
        "leaf_assignments": leaf_assignments,
        "rules": rules,
    }
    _walk_tree(**shared_kwargs, node_id=left_child, path_predicates=[*path_predicates, left_predicate])
    _walk_tree(**shared_kwargs, node_id=right_child, path_predicates=[*path_predicates, right_predicate])


def _build_split_predicates(encoder: FeatureEncoder, threshold: float) -> tuple[Predicate, Predicate]:
    """Build the left and right branch predicates for a decision tree split.

    For categorical features, the split threshold is decoded to a set of
    category labels via the encoder's `category_mapping`.  Categories with
    codes `<= threshold` form the left branch (`"in"` predicate); categories
    with codes `> threshold` form the right branch (also an `"in"` predicate
    over the complementary set, for symmetry and clarity).

    For all other feature types, numeric threshold predicates (`"<="` for the
    left branch and `">"` for the right branch) are created.

    Args:
        encoder (FeatureEncoder): Encoder metadata for the feature being split.
        threshold (float): Raw sklearn split threshold value.

    Returns:
        tuple[Predicate, Predicate]: A 2-tuple of
            `(left_predicate, right_predicate)`.
    """
    feature_name = encoder.column_name

    if encoder.category_mapping is not None:
        left_labels: set[str] = {label for code, label in encoder.category_mapping.items() if code <= threshold}
        right_labels: set[str] = {label for code, label in encoder.category_mapping.items() if code > threshold}
        left_predicate = Predicate(variable=feature_name, operator="in", value=left_labels)
        right_predicate = Predicate(variable=feature_name, operator="in", value=right_labels)
        return left_predicate, right_predicate

    rounded_threshold = round(float(threshold), THRESHOLD_DECIMAL_PLACES)
    left_predicate = Predicate(variable=feature_name, operator="<=", value=rounded_threshold)
    right_predicate = Predicate(variable=feature_name, operator=">", value=rounded_threshold)
    return left_predicate, right_predicate


def _build_leaf_rule(
    *,
    sklearn_tree: Any,
    node_id: int,
    path_predicates: list[Predicate],
    task_type: DecisionTreeTask,
    target_mapping: dict[int, str] | None,
    target_array: np.ndarray,
    leaf_assignments: np.ndarray | None,
) -> DecisionTreeRule:
    """Construct a leaf rule from a decision tree node's stored statistics.

    Args:
        sklearn_tree (Any): The `tree.tree_` internal structure from a fitted
            sklearn decision tree estimator.
        node_id (int): Index of the leaf node in `sklearn_tree`.
        path_predicates (list[Predicate]): Predicates accumulated along the
            path from root to this leaf.
        task_type (DecisionTreeTask): Whether to produce a classification or
            regression rule.
        target_mapping (dict[int, str] | None): Class label mapping for
            classification; `None` for regression.
        target_array (np.ndarray): 1-D target vector used to fit the tree (needed to
            compute per-leaf std for regression).
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)` for regression std computation.

    Returns:
        DecisionTreeRule: The constructed leaf rule.
    """
    node_value = sklearn_tree.value[node_id]
    n_samples = int(sklearn_tree.n_node_samples[node_id])

    if task_type == "classification":
        return _build_classification_rule(
            node_value, n_samples, path_predicates=path_predicates, target_mapping=target_mapping
        )
    return _build_regression_rule(
        node_value,
        n_samples,
        node_id,
        path_predicates=path_predicates,
        target_array=target_array,
        leaf_assignments=leaf_assignments,
    )


def _build_classification_rule(
    node_value: np.ndarray,
    n_samples: int,
    *,
    path_predicates: list[Predicate],
    target_mapping: dict[int, str] | None,
) -> ClassificationRule:
    """Construct a classification rule from a leaf node's class counts.

    Args:
        node_value (np.ndarray): The `tree.tree_.value[node_id]` array with
            shape `(1, n_classes)` containing per-class sample counts.
        n_samples (int): Total number of training samples at this leaf.
        path_predicates (list[Predicate]): Predicates along the root-to-leaf path.
        target_mapping (dict[int, str] | None): Maps integer class codes to
            original label strings.

    Returns:
        ClassificationRule: The constructed classification rule.
    """
    class_counts = node_value[0]
    class_index = int(np.argmax(class_counts))
    max_count = float(class_counts[class_index])
    total_count = float(class_counts.sum())
    confidence = max_count / total_count if total_count > 0 else 0.0

    if target_mapping is not None:
        prediction: str | float = target_mapping[class_index]
    else:
        prediction = class_index

    return ClassificationRule(
        task_type="classification",
        predicates=path_predicates,
        prediction=prediction,
        samples=n_samples,
        confidence=round(confidence, 4),
    )


def _build_regression_rule(
    node_value: np.ndarray,
    n_samples: int,
    node_id: int,
    *,
    path_predicates: list[Predicate],
    target_array: np.ndarray,
    leaf_assignments: np.ndarray | None,
) -> RegressionRule:
    """Construct a regression rule from a leaf node's stored mean and leaf samples.

    The standard deviation is computed from the actual training samples
    assigned to this leaf, identified via `leaf_assignments`.

    Args:
        node_value (np.ndarray): The `tree.tree_.value[node_id]` array with
            shape `(1, 1)` containing the leaf mean.
        n_samples (int): Total number of training samples at this leaf.
        node_id (int): Index of the leaf node in the internal tree structure.
        path_predicates (list[Predicate]): Predicates along the root-to-leaf path.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)`.

    Returns:
        RegressionRule: The constructed regression rule.
    """
    prediction = float(node_value[0, 0])

    leaf_std = 0.0
    if leaf_assignments is not None:
        leaf_mask = leaf_assignments == node_id
        leaf_target = target_array[leaf_mask]
        if len(leaf_target) > 0:
            leaf_std = float(np.std(leaf_target))

    return RegressionRule(
        task_type="regression",
        predicates=path_predicates,
        prediction=round(prediction, 4),
        samples=n_samples,
        std=round(leaf_std, 4),
    )


def _validate_inputs(
    df: pl.DataFrame,
    target: str,
    features: list[str] | None,
) -> None:
    """Validate that the target and requested feature columns exist in the DataFrame.

    Args:
        df (pl.DataFrame): The source DataFrame to check.
        target (str): The target column name.
        features (list[str] | None): Requested feature column names, or `None`
            to use all non-target columns.
    """
    _validate_target_column(df, target)
    feature_columns = features if features is not None else [col for col in df.columns if col != target]
    _validate_feature_columns_exist(df, feature_columns)


def _prepare_clean_dataframe(
    df: pl.DataFrame,
    target: str,
    min_samples_leaf: int,
) -> pl.DataFrame:
    """Validate the target series and drop rows with null targets.

    Args:
        df (pl.DataFrame): The source DataFrame.
        target (str): The target column name.
        min_samples_leaf (int): Minimum required non-null rows after dropping nulls.

    Returns:
        pl.DataFrame: The DataFrame with null-target rows removed.

    Raises:
        ValueError: If the target series is degenerate or too few non-null
            rows remain after dropping nulls.
    """
    _validate_target_series(df[target], target)

    df_clean = df.drop_nulls(subset=[target])
    n_rows = len(df_clean)

    if n_rows < min_samples_leaf:
        raise ValueError(
            f"Only {n_rows} non-null rows remain after dropping null targets, but min_samples_leaf={min_samples_leaf}."
        )

    return df_clean


def _fit_and_assemble_result(
    df_clean: pl.DataFrame,
    target: str,
    *,
    kept_columns: list[str],
    excluded_features: list[ExcludedFeature],
    n_rows: int,
    max_depth: int,
    min_samples_leaf: int,
    task_type: str | None,
    random_state: int | None,
) -> DecisionTreeResult:
    """Encode data, fit a tree, extract rules, and assemble the result object.

    Args:
        df_clean (pl.DataFrame): DataFrame with null-target rows removed.
        target (str): Target column name.
        kept_columns (list[str]): Feature column names that survived filtering.
        excluded_features (list[ExcludedFeature]): Excluded features with reasons.
        n_rows (int): Number of rows in `df_clean`.
        max_depth (int): Maximum tree depth; must be between 1 and
            `_MAX_TREE_DEPTH` (inclusive).
        min_samples_leaf (int): User-supplied minimum samples per leaf.
        task_type (str | None): Task type override or `None` for auto-detection.
        random_state (int | None): Random seed for the sklearn tree estimator.

    Returns:
        DecisionTreeResult: The fully assembled decision tree result.
    """
    detected_task_type = detect_task_type(df_clean[target], task_type)
    feature_matrix, feature_encoders = encode_features(df_clean, kept_columns)
    target_array, target_mapping = encode_target(df_clean[target], detected_task_type)

    effective_min_samples = _compute_effective_min_samples(n_rows, min_samples_leaf)

    fitted_tree = fit_tree(
        feature_matrix,
        target_array,
        task_type=detected_task_type,
        max_depth=max_depth,
        min_samples_leaf=effective_min_samples,
        random_state=random_state,
    )
    rules = extract_rules(
        fitted_tree,
        feature_matrix,
        target_array,
        feature_encoders=feature_encoders,
        target_mapping=target_mapping,
        task_type=detected_task_type,
    )
    metrics = compute_metrics(fitted_tree, feature_matrix, target_array, task_type=detected_task_type)
    feature_importance = compute_feature_importance(fitted_tree, kept_columns)
    # `kept_columns` that have zero importance are dropped from `features_used`;
    # only columns that actually contributed to splits appear in `feature_importance`.
    features_used = [col for col in kept_columns if col in feature_importance]
    zero_importance = [
        ExcludedFeature(name=col, reason="zero importance") for col in kept_columns if col not in feature_importance
    ]
    features_excluded_labels = [f"{ef.name} ({ef.reason})" for ef in [*excluded_features, *zero_importance]]

    return DecisionTreeResult(
        target=target,
        task_type=detected_task_type,
        features_used=features_used,
        features_excluded=features_excluded_labels,
        rules=rules,
        feature_importance=feature_importance,
        metrics=metrics,
        sample_count=n_rows,
        depth=fitted_tree.get_depth(),
        leaf_count=fitted_tree.get_n_leaves(),
    )


def _validate_target_column(df: pl.DataFrame, target: str) -> None:
    """Raise `ValueError` if the target column does not exist in the DataFrame.

    Args:
        df (pl.DataFrame): The source DataFrame to check.
        target (str): The target column name to look up.

    Raises:
        ValueError: If `target` is not present in `df.columns`.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")


def _validate_feature_columns_exist(df: pl.DataFrame, feature_columns: list[str]) -> None:
    """Raise `ValueError` if any requested feature columns are missing.

    Args:
        df (pl.DataFrame): The source DataFrame to check.
        feature_columns (list[str]): The feature column names to look up.

    Raises:
        ValueError: If any column in `feature_columns` is absent from `df.columns`.
    """
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Requested feature columns not found in DataFrame: {missing_columns}")


def _validate_target_series(series: pl.Series, target: str) -> None:
    """Raise `ValueError` if the target series is degenerate.

    A target series is considered degenerate when all values are null or when
    it contains only a single unique non-null value.

    Args:
        series (pl.Series): The target column to inspect.
        target (str): The column name, used in error messages.

    Raises:
        ValueError: If all values are null or there is only one unique non-null value.
    """
    if series.is_null().all():
        raise ValueError(f"Target column '{target}' contains only null values.")
    non_null_series = series.drop_nulls()
    if non_null_series.n_unique() <= 1:
        raise ValueError(f"Target column '{target}' must have more than one unique value.")


def _compute_effective_min_samples(n_rows: int, user_min_samples_leaf: int) -> int:
    """Compute the effective `min_samples_leaf` to pass to the sklearn tree.

    The effective value is the maximum of the user-supplied value and an
    auto-adjusted floor of `max(_AUTO_MIN_SAMPLES_FLOOR, 2% of n_rows)`.
    This prevents overfitting on large datasets while respecting explicit
    user intent.

    Args:
        n_rows (int): Number of training samples after null-target rows are dropped.
        user_min_samples_leaf (int): The user-supplied `min_samples_leaf` parameter.

    Returns:
        int: The effective `min_samples_leaf` value to use.
    """
    auto_floor = max(_AUTO_MIN_SAMPLES_FLOOR, int(_AUTO_MIN_SAMPLES_FRACTION * n_rows))
    return max(user_min_samples_leaf, auto_floor)
