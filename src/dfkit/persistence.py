"""State persistence and restoration for DataFrameToolkit.

This module contains functions for restoring a DataFrameToolkit from serialized
state. The main entry point is `restore_registry_from_state()`, which reconstructs all
base and derivative dataframes from a saved DataFrameToolkitState.

The restoration process:
1. Normalizes user-provided dataframe keys to DataFrameIds
2. Validates base dataframes match expected schema and statistics
3. Registers base dataframes with their original references
4. Reconstructs derivative dataframes by replaying SQL queries in dependency order
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from graphlib import CycleError, TopologicalSorter
from typing import Final

import polars as pl

from dfkit.identifier import (
    DATAFRAME_ID_PATTERN,
    DataFrameId,
)
from dfkit.models import (
    ColumnSummary,
    DataFrameReference,
    DataFrameToolkitState,
)
from dfkit.registry import DataFrameRegistry

__all__ = ["REL_TOL_DEFAULT", "restore_registry_from_state"]

REL_TOL_DEFAULT: Final[float] = 1e-9


def restore_registry_from_state(
    *,
    state: DataFrameToolkitState,
    base_dataframes: Mapping[str, pl.DataFrame],
    rel_tol: float = REL_TOL_DEFAULT,
) -> DataFrameRegistry:
    """Create a DataFrameRegistry from saved state by registering bases and reconstructing derivatives.

    This is the main entry point for state restoration, handling:

    1. Finding all base references in the state (those without parent dependencies)
    2. Normalizing user-provided dataframe keys to DataFrameIds
    3. Validating all base refs are provided and match expected schema/statistics
    4. Registering base dataframes with their references
    5. Reconstructing derivatives via SQL replay in dependency order

    Args:
        state (DataFrameToolkitState): Serialized state from export_state().
        base_dataframes (Mapping[str, pl.DataFrame]): Mapping of identifier to
            DataFrame for all base tables. Keys can be either names or IDs
            (df_xxxxxxxx format). DataFrames must match the schema and
            statistics from when the state was exported.
        rel_tol (float): Relative tolerance for floating point comparisons
            during validation. Defaults to 1e-9.

    Returns:
        DataFrameRegistry: A fully restored registry with all base and
            derivative dataframes registered.

    Raises:
        ValueError: If a provided key doesn't match any base reference,
            if required base dataframes are missing, or if a DataFrame's
            schema or statistics don't match the expected state.

    Examples:
        Restore a registry from saved state:

        >>> from dfkit.persistence import restore_registry_from_state
        >>> from dfkit.models import DataFrameToolkitState
        >>> import polars as pl
        >>> state = DataFrameToolkitState(references=[])
        >>> registry = restore_registry_from_state(state=state, base_dataframes={})
        >>> len(registry.references)
        0
    """
    registry = DataFrameRegistry()
    # 1. Find base references
    base_refs = {ref.id: ref for ref in state.references if ref.is_base}

    # 2. Normalize user keys to DataFrameId
    normalized_bases = _resolve_dataframe_keys_to_ids(
        dataframes=base_dataframes,
        names_to_ids={ref.name: ref.id for ref in base_refs.values()},
    )

    # 3. Validate all base refs in state are provided
    missing_ids = base_refs.keys() - normalized_bases.keys()
    if missing_ids:
        missing_info = [(base_refs[df_id].name, df_id) for df_id in missing_ids]
        msg = f"Missing base dataframes (name, id): {missing_info}"
        raise ValueError(msg)

    # 4. Validate each provided dataframe matches its reference
    for df_id, dataframe in normalized_bases.items():
        ref = base_refs[df_id]
        _validate_dataframe_matches_reference(dataframe, ref, rel_tol=rel_tol)

    # 5. Register base dataframes with their references
    for df_id, dataframe in normalized_bases.items():
        registry.register(base_refs[df_id], dataframe)

    # 6. Reconstruct derivative dataframes via SQL replay in dependency order
    _reconstruct_derivatives(state, registry, rel_tol=rel_tol)

    return registry


# Private helpers


def _resolve_dataframe_keys_to_ids(
    *,
    dataframes: Mapping[str, pl.DataFrame],
    names_to_ids: dict[str, DataFrameId],
) -> dict[DataFrameId, pl.DataFrame]:
    """Convert a mapping of names/IDs to DataFrames into a mapping of IDs to DataFrames.

    Note: Extra base dataframes in `dataframes` that aren't in `names_to_ids` are not allowed and will raise an error.

    Args:
        dataframes (Mapping[str, pl.DataFrame]): Dataframes keyed by name or ID (df_xxxxxxxx format).
        names_to_ids (dict[str, DataFrameId]): Mapping from names to DataFrameIds.

    Returns:
        dict[DataFrameId, pl.DataFrame]: Dataframes keyed by their DataFrameId.

    Raises:
        ValueError: If a key doesn't match any base reference, or if
            multiple keys resolve to the same DataFrameId.
    """
    ids = set(names_to_ids.values())
    name_or_id_to_id: dict[str | DataFrameId, DataFrameId] = names_to_ids | {df_id: df_id for df_id in ids}

    normalized: dict[DataFrameId, pl.DataFrame] = {}
    for name_or_id, dataframe in dataframes.items():
        # Check if the identifier (name or ID) matches any base reference, and get the corresponding ID
        if (df_id := name_or_id_to_id.get(name_or_id)) is None:
            msg = (
                f"ID '{name_or_id}' not in state's base references. Available IDs: {ids}"
                if DATAFRAME_ID_PATTERN.match(name_or_id)
                else f"Name '{name_or_id}' not in state's base references. Available names: {list(names_to_ids.keys())}"
            )
            raise ValueError(msg)

        # Check for duplicate IDs (multiple keys resolving to the same ID)
        if df_id in normalized:
            msg = f"Duplicate: key '{name_or_id}' resolves to ID '{df_id}' which was already provided"
            raise ValueError(msg)

        # Store the dataframe under its resolved ID
        normalized[df_id] = dataframe

    return normalized


def _validate_dataframe_matches_reference(
    dataframe: pl.DataFrame,
    reference: DataFrameReference,
    *,
    rel_tol: float = REL_TOL_DEFAULT,
) -> None:
    """Validate that a DataFrame matches the expected schema and statistics from a reference.

    Checks that column names, shape (rows, columns), and column summaries match
    the reference metadata. This ensures data consistency when reconstructing
    from saved state.

    Args:
        dataframe (pl.DataFrame): The DataFrame to validate.
        reference (DataFrameReference): The reference containing expected metadata.
        rel_tol (float): Relative tolerance for floating point comparisons. Defaults to 1e-9.

    Raises:
        ValueError: If column names, shape, or column summaries do not match.
    """
    # Check shapes match
    actual_shape = dataframe.shape
    expected_shape = (reference.num_rows, reference.num_columns)
    if actual_shape != expected_shape:
        msg = f"DataFrame '{reference.name}' shape mismatch. Expected: {expected_shape}, got: {actual_shape}"
        raise ValueError(msg)

    # Check columns match
    actual_columns = dataframe.columns
    expected_columns = reference.column_names

    if actual_columns != expected_columns:
        msg = f"DataFrame '{reference.name}' column mismatch. Expected: {expected_columns}, got: {actual_columns}"
        raise ValueError(msg)

    # Check column summaries match
    for col_name in actual_columns:
        actual_summary = ColumnSummary.from_series(dataframe[col_name])
        expected_summary = reference.column_summaries[col_name]

        mismatches = _compare_column_summaries(actual_summary, expected_summary, rel_tol=rel_tol)
        if mismatches:
            msg = f"DataFrame '{reference.name}' column '{col_name}' statistics mismatch. Differences: {mismatches}"
            raise ValueError(msg)


def _compare_column_summaries(
    actual: ColumnSummary,
    expected: ColumnSummary,
    *,
    exact_fields: Sequence[str] = ("dtype", "count", "null_count", "unique_count"),
    approx_fields: Sequence[str] = ("min", "max", "mean", "std", "p25", "p50", "p75"),
    rel_tol: float = REL_TOL_DEFAULT,
) -> dict[str, tuple[object, object]]:
    """Compare two column summaries and return any mismatches.

    Note: Any field not in exact_fields or approx_fields is ignored for comparison.
    This allows comparison of data-dependent fields while ignoring supplemental
    metadata fields (e.g., description) that may differ without indicating a data mismatch.

    Args:
        actual (ColumnSummary): The actual column summary from the DataFrame.
        expected (ColumnSummary): The expected column summary from the reference.
        exact_fields (Sequence[str]): Fields that must match exactly (e.g. dtype, count).
        approx_fields (Sequence[str]): Fields that can match approximately (e.g. min, max, mean).
            Defaults to common numeric summaries.
        rel_tol (float): Relative tolerance for floating point comparisons.

    Returns:
        dict[str, tuple[object, object]]: Dictionary of field names to (actual, expected)
            tuples for any fields that don't match. Empty if all fields match.
    """
    mismatches: dict[str, tuple[object, object]] = {}

    for field in exact_fields:
        actual_val = getattr(actual, field)
        expected_val = getattr(expected, field)
        if actual_val != expected_val:
            mismatches[field] = (actual_val, expected_val)

    for field in approx_fields:
        actual_val = getattr(actual, field)
        expected_val = getattr(expected, field)
        if not _values_nearly_equal(actual=actual_val, expected=expected_val, rel_tol=rel_tol):
            mismatches[field] = (actual_val, expected_val)

    return mismatches


def _floats_nearly_equal(actual: float, expected: float, *, rel_tol: float) -> bool:
    """Compare two floats for near-equality, treating NaN == NaN as True.

    Unlike standard float comparison where NaN != NaN, this function treats
    two NaN values as equal. This is useful for validating statistical summaries
    where NaN indicates missing/undefined values that should match.

    Args:
        actual (float): The actual value to compare.
        expected (float): The expected value to compare against.
        rel_tol (float): Relative tolerance for the comparison.

    Returns:
        bool: True if both values are NaN, if both are finite and within
            rel_tol of each other, or if both are the same infinity.
            False if only one value is NaN.
    """
    if math.isnan(actual) and math.isnan(expected):
        return True
    if math.isnan(actual) or math.isnan(expected):
        return False
    return math.isclose(actual, expected, rel_tol=rel_tol)


def _exact_values_equal(
    *,
    actual: int | str | None,
    expected: int | str | None,
) -> bool:
    """Compare non-float values for equality.

    Handles None, str, and bool comparisons. Returns False for numeric
    non-bool pairs that require float comparison.

    Args:
        actual (int | str | None): The actual value to compare.
        expected (int | str | None): The expected value to compare against.

    Returns:
        bool: True if values are equal, False otherwise.
    """
    if actual is None or expected is None:
        return actual is None and expected is None
    if type(actual) is type(expected):
        return actual == expected
    return False


def _values_nearly_equal(
    *,
    actual: float | str | None,
    expected: float | str | None,
    rel_tol: float = REL_TOL_DEFAULT,
) -> bool:
    """Check if two values are nearly equal, handling floats, strings, and None.

    Args:
        actual (float | str | None): The actual value to compare.
        expected (float | str | None): The expected value to compare against.
        rel_tol (float): Relative tolerance for float comparisons. Defaults to 1e-9.

    Returns:
        bool: True if values are considered equal.
    """
    if (
        isinstance(actual, numbers.Real)
        and isinstance(expected, numbers.Real)
        and not isinstance(actual, bool)
        and not isinstance(expected, bool)
    ):
        return _floats_nearly_equal(float(actual), float(expected), rel_tol=rel_tol)
    elif not isinstance(actual, float) and not isinstance(expected, float):
        return _exact_values_equal(actual=actual, expected=expected)
    else:
        return False


def _reconstruct_derivatives(
    state: DataFrameToolkitState,
    registry: DataFrameRegistry,
    *,
    rel_tol: float = REL_TOL_DEFAULT,
) -> None:
    """Reconstruct and register derivative dataframes from state.

    Replays SQL queries for derivatives in dependency order and validates that
    reconstructed dataframes match the expected statistics from the saved state.

    Note:
        Non-deterministic SQL queries may cause reconstruction validation to fail
        even when the state is valid. Examples include:

        - ORDER BY without LIMIT: Row order may differ, affecting percentile statistics
        - Floating-point aggregations: Accumulation order may differ slightly

        If you encounter validation failures with valid state, consider whether
        the original query was non-deterministic.

    Args:
        state (DataFrameToolkitState): The state containing derivative references.
        registry (DataFrameRegistry): The registry for SQL execution and registration.
        rel_tol (float): Relative tolerance for floating point comparisons
            during validation. Defaults to 1e-9.
    """
    for ref in _sort_references_by_dependency_order(state.references):
        if ref.id in registry.references:
            # Skip base dataframes that MUST already be registered,
            # and any derivatives that were already reconstructed by a previous reference
            # (e.g. if multiple references point to the same derivative ID)
            continue

        result_df = _reconstruct_dataframe(ref, registry)
        _validate_dataframe_matches_reference(result_df, ref, rel_tol=rel_tol)

        registry.register(ref, result_df)


def _sort_references_by_dependency_order(references: list[DataFrameReference]) -> list[DataFrameReference]:
    """Sort references by dependency order (parents before children).

    Args:
        references (list[DataFrameReference]): References to sort.

    Returns:
        list[DataFrameReference]: Sorted references with parents before children.

    Raises:
        ValueError: If cyclic dependencies are detected in the references,
            or if any reference has parent_ids pointing to non-existent references.
    """
    refs_by_id = {ref.id: ref for ref in references}

    # Validate all parent_ids exist before building the graph
    for ref in references:
        missing = [pid for pid in ref.parent_ids if pid not in refs_by_id]
        if missing:
            msg = f"Reference '{ref.name}' has unknown parent_ids: {missing}. State may be corrupted."
            raise ValueError(msg)

    graph = {ref.id: ref.parent_ids for ref in references}

    try:
        sorted_ids = list(TopologicalSorter(graph).static_order())
    except CycleError as e:
        msg = f"Cyclic dependency detected in references, state may be corrupted: {e}"
        raise ValueError(msg) from e

    return [refs_by_id[ref_id] for ref_id in sorted_ids]


def _reconstruct_dataframe(
    ref: DataFrameReference,
    registry: DataFrameRegistry,
) -> pl.DataFrame:
    """Reconstruct a single derivative dataframe from its reference.

    Args:
        ref (DataFrameReference): The reference to reconstruct.
        registry (DataFrameRegistry): The registry for SQL execution.

    Returns:
        pl.DataFrame: The reconstructed dataframe.

    Raises:
        RuntimeError: If invariants are violated (base dataframe reached
            reconstruction, or derivative missing source_query).
        ValueError: If required parents are missing or SQL execution fails.
    """
    if ref.is_base:
        msg = (
            f"Invariant violation: base dataframe '{ref.name}' (id={ref.id}) "
            f"reached _reconstruct_dataframe. The topological sort should "
            f"exclude base dataframes from reconstruction."
        )
        raise RuntimeError(msg)

    # Derivatives must have a source_query to replay
    if not ref.source_query:
        msg = (
            f"Invariant violation: derivative '{ref.name}' has parent_ids but no source_query - state may be corrupted"
        )
        raise RuntimeError(msg)

    missing_parents = [pid for pid in ref.parent_ids if pid not in registry.references]
    if missing_parents:
        available_ids = list(registry.references.keys())
        msg = (
            f"Cannot reconstruct '{ref.name}': missing parent dataframes {missing_parents}. "
            f"Available IDs: {available_ids}"
        )
        raise ValueError(msg)

    return _execute_reconstruction_query(ref.source_query, ref.name, registry)


def _execute_reconstruction_query(source_query: str, name: str, registry: DataFrameRegistry) -> pl.DataFrame:
    """Execute the SQL query to reconstruct a dataframe.

    The caller is responsible for validating that source_query is non-empty
    before calling this function.

    Args:
        source_query (str): The SQL query to execute.
        name (str): The reference name, used for error messages.
        registry (DataFrameRegistry): The registry for SQL execution.

    Returns:
        pl.DataFrame: The reconstructed dataframe.

    Raises:
        ValueError: If SQL execution fails.
        TypeError: If execute_sql(eager=True) returns non-DataFrame type.
    """
    try:
        result_df = registry.context.execute_sql(source_query, eager=True)
    except (ValueError, pl.exceptions.PolarsError, RuntimeError) as e:
        msg = f"SQL execution failed while reconstructing '{name}': {e}. Query: {source_query}"
        raise ValueError(msg) from e

    # Defensive: Polars execute(eager=True) should always return DataFrame per contract.
    if not isinstance(result_df, pl.DataFrame):
        msg = f"execute_sql(eager=True) returned {type(result_df).__name__}, expected DataFrame"
        raise TypeError(msg)

    return result_df
