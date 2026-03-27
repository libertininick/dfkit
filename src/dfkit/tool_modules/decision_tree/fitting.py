"""Decision tree fitting, rule extraction, metrics computation, and pipeline orchestration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dfkit.tool_modules.decision_tree.models import (
    ClassificationRule,
    DecisionTree,
    DecisionTreeResult,
    DecisionTreeRule,
    DecisionTreeTask,
    Predicate,
    PredicateOp,
    RegressionRule,
)
from dfkit.tool_modules.decision_tree.preprocessing import (
    THRESHOLD_DECIMAL_PLACES,
    FeatureEncoder,
    encode_features,
    encode_target,
    filter_features,
    infer_task,
)

# region Module-level constants

MIN_TREE_DEPTH: int = 1  # Lower bound on tree depth accepted by the public API.
MAX_TREE_DEPTH: int = 6  # Caps tree depth to prevent overfitting and keep rules human-readable.
AUTO_MIN_SAMPLES_FRACTION: float = 0.02  # Scales the leaf floor with dataset size: 2% of n_rows.
AUTO_MIN_SAMPLES_FLOOR: int = 5  # Absolute minimum leaf size regardless of dataset size.

# endregion


# region Public interface


def analyze_with_decision_tree(
    df: pl.DataFrame,
    features: list[str] | None,
    target: str,
    *,
    task: str | None = None,
    max_depth: int = MAX_TREE_DEPTH,
    min_samples_leaf: int = AUTO_MIN_SAMPLES_FLOOR,
    random_state: int | None = None,
) -> DecisionTreeResult:
    """Analyze the relationship of target column to other column(s) using a decision tree.

    Validates inputs, preprocesses features and target, fits a decision tree,
    extracts rules, computes metrics and feature importance, then assembles
    and returns a `DecisionTreeResult`.

    Args:
        df (pl.DataFrame): The source DataFrame containing features and target.
        features (list[str] | None): Feature column names to consider. When
            `None`, all columns except `target` are used.
        target (str): Name of the target column.
        task (str | None): `"classification"`, `"regression"`, or `None`
            for automatic inference of task type.
        max_depth (int): Maximum tree depth; must be between 1 and
            `MAX_TREE_DEPTH` (inclusive).
        min_samples_leaf (int): Minimum samples required at a leaf node.
            Auto-adjusted upward when the dataset is large.
        random_state (int | None): Random seed passed to the sklearn tree for
            reproducibility.  `None` means non-deterministic.

    Returns:
        DecisionTreeResult: The fitted tree result.

    Raises:
        ValueError: On any validation failure (invalid task override, missing
            columns, degenerate target, no valid features, or insufficient
            samples).
    """
    _validate_task_override(task)
    _validate_target_column_exists(df, target)
    feature_columns = features if features is not None else [col for col in df.columns if col != target]
    _validate_feature_columns_exist(df, feature_columns)

    # Drop null-target rows before feature filtering so the cardinality ratio
    # uses the same denominator as the data the tree will be fitted on.
    df_clean = _prepare_clean_dataframe(df, target, min_samples_leaf)

    kept_columns, excluded_features = filter_features(df_clean, feature_columns)

    if not kept_columns:
        excluded_labels = [f"{ef.name} ({ef.reason})" for ef in excluded_features]
        raise ValueError(f"No valid feature columns remain after filtering. Excluded: {excluded_labels}")

    n_rows = len(df_clean)
    return _encode_fit_extract(
        df_clean=df_clean,
        target=target,
        kept_columns=kept_columns,
        n_rows=n_rows,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        task_override=task,
        random_state=random_state,
    )


def fit_tree(
    task: DecisionTreeTask,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    max_depth: int = MAX_TREE_DEPTH,
    min_samples_leaf: int = AUTO_MIN_SAMPLES_FLOOR,
    random_state: int | None = None,
) -> DecisionTree:
    """Fit a decision tree to the given feature matrix and target vector.

    Args:
        task (DecisionTreeTask): Whether to fit a classifier or regressor.
        feature_matrix (np.ndarray): 2-D feature matrix with shape `(n_samples, n_features)`.
        target_array (np.ndarray): 1-D target vector with shape `(n_samples,)`.
        max_depth (int): Maximum depth of the tree. Must be between
            `MIN_TREE_DEPTH` and `MAX_TREE_DEPTH` (1 to 6), inclusive.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        random_state (int | None): Random seed for reproducibility.

    Returns:
        DecisionTree: The fitted tree estimator.

    Raises:
        ValueError: If `max_depth` is outside the valid range
            `[MIN_TREE_DEPTH, MAX_TREE_DEPTH]`.
    """
    if not (MIN_TREE_DEPTH <= max_depth <= MAX_TREE_DEPTH):
        raise ValueError(f"max_depth must be between {MIN_TREE_DEPTH} and {MAX_TREE_DEPTH}, got {max_depth}.")
    tree_cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    tree = tree_cls(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(feature_matrix, target_array)
    return tree


def extract_rules(
    tree: DecisionTree,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    feature_encoders: list[FeatureEncoder],
    target_mapping: dict[int, str] | None,
    task: DecisionTreeTask,
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

    Redundant predicates on the same variable are simplified: multiple `<=`
    thresholds are reduced to the minimum, multiple `>` thresholds to the
    maximum, and multiple `in` sets are intersected.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_matrix (np.ndarray): 2-D feature matrix used to fit the tree.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        feature_encoders (list[FeatureEncoder]): Parallel list of encoders
            describing how each feature column was encoded.
        target_mapping (dict[int, str] | None): Maps integer codes back to
            class labels for classification; `None` for regression.
        task (DecisionTreeTask): Whether the tree is a classifier or regressor.

    Returns:
        list[ClassificationRule] | list[RegressionRule]: One rule per leaf node.
    """
    rules = _walk_tree(
        tree=tree,
        target_array=target_array,
        feature_encoders=feature_encoders,
        target_mapping=target_mapping,
        task=task,
        leaf_assignments=tree.apply(feature_matrix),
    )
    return rules


def compute_metrics(
    tree: DecisionTree,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    *,
    task: DecisionTreeTask,
) -> dict[str, float]:
    """Compute evaluation metrics for a fitted decision tree.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        feature_matrix (np.ndarray): 2-D feature matrix with shape `(n_samples, n_features)`.
        target_array (np.ndarray): 1-D target vector with shape `(n_samples,)`.
        task (DecisionTreeTask): Whether the tree is a classifier or regressor.

    Returns:
        dict[str, float]: For classification: `{"accuracy": <float>}`.
            For regression: `{"r_squared": <float>, "rmse": <float>}`.
    """
    predictions = tree.predict(feature_matrix)
    if task == "classification":
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
            features with importance > 0. Returns an empty dict for single-leaf
            trees where all feature importances are zero.
    """
    importances = tree.feature_importances_
    # Round each raw importance early so that the displayed values already reflect
    # the precision we will surface to callers. Features rounded to exactly 0.0 are
    # dropped, which means very small raw importances (<0.00005) are excluded.
    paired = [(name, round(float(importance), 4)) for name, importance in zip(feature_names, importances, strict=True)]
    filtered = [(name, importance) for name, importance in paired if importance > 0.0]
    filtered.sort(key=lambda item: item[1], reverse=True)
    total_rounded = sum(importance for _, importance in filtered)
    renormalized = [(name, round(importance / total_rounded, 4)) for name, importance in filtered]
    if renormalized:
        others_sum = sum(importance for _, importance in renormalized[:-1])
        last_name = renormalized[-1][0]
        # Force the last element to exactly 1.0 - (sum of all others) so that
        # accumulated rounding error does not prevent the displayed values from
        # summing to 1.0.
        renormalized[-1] = (last_name, round(1.0 - others_sum, 4))
    return dict(renormalized)

# endregion


# region Helpers


class PendingNode(NamedTuple):
    """A node queued for processing during iterative tree traversal.

    Attributes:
        node_id (int): Index of the node in the sklearn internal tree structure.
        path_predicates (tuple[Predicate, ...]): Predicates accumulated from
            the root to this node.
    """

    node_id: int
    path_predicates: tuple[Predicate, ...]


def _walk_tree(
    *,
    tree: DecisionTree,
    target_array: np.ndarray,
    feature_encoders: list[FeatureEncoder],
    target_mapping: dict[int, str] | None,
    task: DecisionTreeTask,
    leaf_assignments: np.ndarray | None,
) -> list[ClassificationRule] | list[RegressionRule]:
    """Walk a fitted decision tree iteratively and return one rule per leaf.

    Args:
        tree (DecisionTree): A fitted sklearn tree.
        target_array (np.ndarray): 1-D target vector used to fit the tree.
        feature_encoders (list[FeatureEncoder]): Parallel list of encoders.
        target_mapping (dict[int, str] | None): Class label mapping for classification.
        task (DecisionTreeTask): Whether the tree is a classifier or regressor.
        leaf_assignments (np.ndarray | None): Per-sample leaf node IDs from
            `tree.apply(feature_matrix)`, used to compute per-leaf std for regression.

    Returns:
        list[ClassificationRule] | list[RegressionRule]: One rule per leaf node.
    """
    sklearn_tree = tree.tree_
    rules: list[ClassificationRule] | list[RegressionRule] = []
    pending: list[PendingNode] = [PendingNode(node_id=0, path_predicates=())]

    while pending:
        node = pending.pop()
        left_child = sklearn_tree.children_left[node.node_id]
        right_child = sklearn_tree.children_right[node.node_id]

        # Check for a leaf node
        if left_child == right_child:
            rule = _build_leaf_rule(
                sklearn_tree=sklearn_tree,
                node_id=node.node_id,
                path_predicates=tuple(_simplify_predicates(node.path_predicates)),
                task=task,
                target_mapping=target_mapping,
                target_array=target_array,
                leaf_assignments=leaf_assignments,
            )
            rules.append(rule)
            continue

        encoder = feature_encoders[sklearn_tree.feature[node.node_id]]
        threshold = sklearn_tree.threshold[node.node_id]
        left_predicate, right_predicate = _build_split_predicates(encoder, threshold)

        # Right first so left is visited first (LIFO)
        pending.append(PendingNode(right_child, (*node.path_predicates, right_predicate)))
        pending.append(PendingNode(left_child, (*node.path_predicates, left_predicate)))

    return rules


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
    path_predicates: Sequence[Predicate],
    task: DecisionTreeTask,
    target_mapping: dict[int, str] | None,
    target_array: np.ndarray,
    leaf_assignments: np.ndarray | None,
) -> DecisionTreeRule:
    """Construct a leaf rule from a decision tree node's stored statistics.

    Args:
        sklearn_tree (Any): The `tree.tree_` internal structure from a fitted
            sklearn decision tree estimator.
        node_id (int): Index of the leaf node in `sklearn_tree`.
        path_predicates (Sequence[Predicate]): Predicates accumulated along the
            path from root to this leaf.
        task (DecisionTreeTask): Whether to produce a classification or
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

    if task == "classification":
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
    path_predicates: Sequence[Predicate],
    target_mapping: dict[int, str] | None,
) -> ClassificationRule:
    """Construct a classification rule from a leaf node's class counts.

    Args:
        node_value (np.ndarray): The `tree.tree_.value[node_id]` array with
            shape `(1, n_classes)` containing per-class sample counts.
        n_samples (int): Total number of training samples at this leaf.
        path_predicates (Sequence[Predicate]): Predicates along the root-to-leaf path.
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
        task="classification",
        predicates=list(path_predicates),
        prediction=prediction,
        samples=n_samples,
        confidence=round(confidence, 4),
    )


def _build_regression_rule(
    node_value: np.ndarray,
    n_samples: int,
    node_id: int,
    *,
    path_predicates: Sequence[Predicate],
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
        path_predicates (Sequence[Predicate]): Predicates along the root-to-leaf path.
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
            # ddof=0 (population std) intentionally matches sklearn's own impurity
            # calculation, which also uses the population formula internally.
            leaf_std = float(np.std(leaf_target))

    return RegressionRule(
        task="regression",
        predicates=list(path_predicates),
        prediction=round(prediction, 4),
        samples=n_samples,
        std=round(leaf_std, 4),
    )


def _simplify_predicates(predicates: Sequence[Predicate]) -> list[Predicate]:
    """Reduce a list of predicates to the tightest equivalent constraints.

    Groups predicates by `(variable, operator)` and applies operator-specific
    simplification rules:

    - `<=`: keeps the predicate with the smallest value (tightest upper bound).
    - `>`: keeps the predicate with the largest value (tightest lower bound).
    - `in`: replaces the group with a single predicate whose `value` is the
      intersection of all `value` sets.
    - Any other operator: passes all predicates through unchanged.

    The output ordering mirrors the first appearance of each variable in the
    input list, then preserves operator order within that variable.

    Args:
        predicates (Sequence[Predicate]): Predicates to simplify.

    Returns:
        list[Predicate]: Simplified predicates in first-appearance order.
    """
    groups: dict[tuple[str, PredicateOp], list[Predicate]] = defaultdict(list)
    for predicate in predicates:
        key = (predicate.variable, predicate.operator)
        groups[key].append(predicate)

    result: list[Predicate] = []
    for (variable, op), group in groups.items():
        result.extend(_simplify_predicate_group(variable, op, group))
    return result


def _simplify_predicate_group(variable: str, op: PredicateOp, group: list[Predicate]) -> list[Predicate]:
    """Simplify one `(variable, operator)` predicate group to its tightest constraints.

    Args:
        variable (str): The feature variable shared by all predicates in the group.
        op (PredicateOp): The operator shared by all predicates in the group.
        group (list[Predicate]): One or more predicates with the same variable and operator.

    Returns:
        list[Predicate]: Simplified predicates for this group.

    Raises:
        ValueError: If the intersection of ``"in"`` predicates for *variable*
            is empty, indicating a contradictory rule path.
    """
    if op == "<=":
        return [min(group, key=lambda p: cast(float, p.value))]
    if op == ">":
        return [max(group, key=lambda p: cast(float, p.value))]
    if op == "in":
        sets = [cast(set[float] | set[str], p.value) for p in group]
        intersection = sets[0].intersection(*sets[1:])
        if not intersection:
            raise ValueError(
                f"Intersection of 'in' predicates for '{variable}' is empty: "
                "the rule path is contradictory and can never match any sample."
            )
        return [Predicate(variable=variable, operator="in", value=intersection)]
    return list(group)


def _validate_task_override(task: str | None) -> None:
    """Raise `ValueError` if *task* is not a recognized task override value.

    Args:
        task (str | None): The task override to validate. `None` and `"auto"`
            trigger automatic detection and are therefore accepted. Only
            `"classification"` and `"regression"` are accepted as explicit
            overrides.

    Raises:
        ValueError: If *task* is a non-`None` string that is not
            `"classification"`, `"regression"`, or `"auto"`.
    """
    valid_values = {"classification", "regression", "auto", None}
    if task not in valid_values:
        raise ValueError(
            f"Invalid task override '{task}'. Valid values are 'classification', 'regression', 'auto', or None."
        )


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


def _encode_fit_extract(
    df_clean: pl.DataFrame,
    target: str,
    *,
    kept_columns: list[str],
    n_rows: int,
    max_depth: int,
    min_samples_leaf: int,
    task_override: str | None,
    random_state: int | None,
) -> DecisionTreeResult:
    """Encode data, fit a tree, extract rules, and assemble the result object.

    Args:
        df_clean (pl.DataFrame): DataFrame with null-target rows removed.
        target (str): Target column name.
        kept_columns (list[str]): Feature column names that survived filtering.
        n_rows (int): Number of rows in `df_clean`.
        max_depth (int): Maximum tree depth; must be between 1 and
            `MAX_TREE_DEPTH` (inclusive).
        min_samples_leaf (int): User-supplied minimum samples per leaf.
        task_override (str | None): Task type override or `None` for auto-detection.
        random_state (int | None): Random seed for the sklearn tree estimator.

    Returns:
        DecisionTreeResult: The fully assembled decision tree result.
    """
    task = infer_task(df_clean[target], task_override)
    feature_matrix, feature_encoders = encode_features(df_clean, kept_columns)
    target_array, target_mapping = encode_target(df_clean[target], task)

    effective_min_samples = _compute_effective_min_samples(n_rows, min_samples_leaf)

    fitted_tree = fit_tree(
        task,
        feature_matrix,
        target_array,
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
        task=task,
    )
    metrics = compute_metrics(fitted_tree, feature_matrix, target_array, task=task)
    feature_importances = compute_feature_importance(fitted_tree, kept_columns)
    # `kept_columns` that have zero importance are dropped from `features`;
    # only columns that actually contributed to splits appear in `feature_importances`.
    features = [col for col in kept_columns if col in feature_importances]

    return DecisionTreeResult(
        target=target,
        task=task,
        features=features,
        rules=rules,
        feature_importances=feature_importances,
        metrics=metrics,
        sample_count=n_rows,
        depth=fitted_tree.get_depth(),
        leaf_count=fitted_tree.get_n_leaves(),
    )


def _validate_target_column_exists(df: pl.DataFrame, target: str) -> None:
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
    auto-adjusted floor of `max(AUTO_MIN_SAMPLES_FLOOR, 2% of n_rows)`.
    This prevents overfitting on large datasets while respecting explicit
    user intent.

    Args:
        n_rows (int): Number of training samples after null-target rows are dropped.
        user_min_samples_leaf (int): The user-supplied `min_samples_leaf` parameter.

    Returns:
        int: The effective `min_samples_leaf` value to use.
    """
    auto_floor = max(AUTO_MIN_SAMPLES_FLOOR, int(AUTO_MIN_SAMPLES_FRACTION * n_rows))
    return max(user_min_samples_leaf, auto_floor)


# endregion
