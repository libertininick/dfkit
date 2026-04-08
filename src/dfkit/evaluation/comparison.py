"""Field-level comparison engine for the evaluation pipeline.

This module implements comparison strategy functions for different field types
and a dispatcher that selects the appropriate strategy based on the runtime
type of the expected value. Dispatch uses `isinstance` checks on the concrete
Python values that Pydantic has already resolved — no annotation introspection
is required.

Examples:
    >>> from dfkit.evaluation.comparison import compare_exact, compare_numeric
    >>> result = compare_exact("label", "A", "A")
    >>> result.matched
    True
    >>> result = compare_numeric("score", 0.5, 0.52, tolerance=DEFAULT_NUMERIC_TOLERANCE)
    >>> result.matched
    True
"""

from collections.abc import Collection
from typing import Any, Final

from pydantic import BaseModel

from dfkit.evaluation.models import FieldComparison

__all__ = [
    "DEFAULT_NUMERIC_TOLERANCE",
    "DEFAULT_SET_OVERLAP_THRESHOLD",
    "FieldConfig",
    "compare_dict",
    "compare_exact",
    "compare_fields",
    "compare_numeric",
    "compare_set_overlap",
]

DEFAULT_NUMERIC_TOLERANCE: Final[float] = 0.05
DEFAULT_SET_OVERLAP_THRESHOLD: Final[float] = 0.7

# region Public interface


class FieldConfig(BaseModel):
    """Per-field configuration for comparison strategies.

    All attributes are optional; when `None` the comparison strategy uses its
    built-in default.

    Attributes:
        tolerance (float | None): Absolute tolerance for numeric comparisons.
            Defaults to `DEFAULT_NUMERIC_TOLERANCE` when `None`.
        threshold (float | None): Minimum Jaccard index required for a set-overlap
            match.  Defaults to `DEFAULT_SET_OVERLAP_THRESHOLD` when `None`.
        case_sensitive (bool | None): Whether set-overlap comparison is
            case-sensitive.  Defaults to `False` when `None`.
    """

    tolerance: float | None = None
    threshold: float | None = None
    case_sensitive: bool | None = None


def compare_numeric(
    field_name: str,
    expected: float | None,
    extracted: float | None,
    *,
    tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> FieldComparison:
    """Compare two numeric values within an absolute tolerance.

    Args:
        field_name (str): Name of the field being compared.
        expected (float | None): Expected numeric value from ground truth.
        extracted (float | None): Numeric value extracted by the judge LLM.
        tolerance (float): Maximum absolute difference allowed for a match.

    Returns:
        FieldComparison: Comparison result with strategy `"numeric_tolerance"`.
            When both values are present, `details` includes the absolute
            `"difference"` and the `"tolerance"` applied.
    """
    if expected is None and extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=True,
            strategy="numeric_tolerance",
        )

    if expected is None or extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=False,
            strategy="numeric_tolerance",
        )

    difference = abs(expected - extracted)
    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=difference <= tolerance,
        strategy="numeric_tolerance",
        details={"difference": difference, "tolerance": tolerance},
    )


def compare_exact(
    field_name: str,
    expected: Any,
    extracted: Any,
) -> FieldComparison:
    """Compare two values for exact equality.

    Args:
        field_name (str): Name of the field being compared.
        expected (Any): Expected value from ground truth.
        extracted (Any): Value extracted by the judge LLM.

    Returns:
        FieldComparison: Comparison result with strategy `"exact_match"`.
    """
    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=expected == extracted,
        strategy="exact_match",
    )


def compare_set_overlap(
    field_name: str,
    expected: list[str] | None,
    extracted: list[str] | None,
    *,
    case_sensitive: bool = False,
    threshold: float = DEFAULT_SET_OVERLAP_THRESHOLD,
) -> FieldComparison:
    """Compare two string lists using Jaccard overlap.

    Args:
        field_name (str): Name of the field being compared.
        expected (list[str] | None): Expected list of strings from ground truth.
        extracted (list[str] | None): List of strings extracted by the judge LLM.
        case_sensitive (bool): When `False` (default), strings are lowercased
            before comparison.
        threshold (float): Minimum Jaccard index required for a match
            (default `DEFAULT_SET_OVERLAP_THRESHOLD`).

    Returns:
        FieldComparison: Comparison result with strategy `"set_overlap"`.
            `details` includes `"overlap_ratio"` (Jaccard index),
            `"missing"` (items in expected but not extracted), and
            `"extra"` (items in extracted but not expected).
    """
    if expected is None and extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=True,
            strategy="set_overlap",
        )

    if expected is None or extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=False,
            strategy="set_overlap",
        )

    expected_set = _to_str_set(expected, case_sensitive=case_sensitive)
    extracted_set = _to_str_set(extracted, case_sensitive=case_sensitive)

    intersection = expected_set & extracted_set
    union = expected_set | extracted_set
    overlap_ratio = len(intersection) / len(union) if union else 1.0  # Two empty sets are considered a perfect match

    missing = sorted(expected_set - extracted_set)
    extra = sorted(extracted_set - expected_set)

    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=overlap_ratio >= threshold,
        strategy="set_overlap",
        details={"overlap_ratio": overlap_ratio, "missing": missing, "extra": extra},
    )


def compare_dict(
    field_name: str,
    expected: dict[str, Any] | None,
    extracted: dict[str, Any] | None,
    *,
    config: FieldConfig | None = None,
) -> FieldComparison:
    """Compare two dictionaries key-by-key.

    Each value is compared using `_dispatch_comparison` so that nested floats,
    lists, and dicts are handled with the appropriate strategy automatically.

    Args:
        field_name (str): Name of the field being compared.
        expected (dict[str, Any] | None): Expected dictionary from ground truth.
        extracted (dict[str, Any] | None): Dictionary extracted by the judge LLM.
        config (FieldConfig | None): Per-field config forwarded to
            `_dispatch_comparison` for each key.

    Returns:
        FieldComparison: Comparison result with strategy `"dict_comparison"`.
            `details` includes `"key_results"` (per-key comparison info),
            `"missing_keys"` (keys present in expected but absent in
            extracted), and `"extra_keys"` (keys present in extracted but
            absent in expected).
    """
    if expected is None and extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=True,
            strategy="dict_comparison",
        )

    if expected is None or extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=False,
            strategy="dict_comparison",
        )

    resolved_config = config or FieldConfig()

    expected_keys = set(expected.keys())
    extracted_keys = set(extracted.keys())
    missing_keys = sorted(expected_keys - extracted_keys)
    extra_keys = sorted(extracted_keys - expected_keys)
    shared_keys = expected_keys & extracted_keys

    key_results: dict[str, dict[str, Any]] = {}
    for key in sorted(shared_keys):
        key_comparison = _dispatch_comparison(key, expected[key], extracted[key], resolved_config)
        key_results[key] = {"matched": key_comparison.matched, **key_comparison.details}

    all_values_matched = all(info["matched"] for info in key_results.values())

    is_matched = not missing_keys and not extra_keys and all_values_matched

    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=is_matched,
        strategy="dict_comparison",
        details={
            "key_results": key_results,
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
        },
    )


def compare_fields[M: BaseModel](
    expected: M,
    extracted: M,
    *,
    field_configs: dict[str, FieldConfig] | None = None,
    default_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> list[FieldComparison]:
    """Compare all fields of two Pydantic model instances.

    Iterates over the fields declared on `expected` and selects a comparison
    strategy for each one based on the runtime type of the expected value:

    - `float` → `compare_numeric`
    - `dict` → `compare_dict`
    - `list` where all elements are `str` → `compare_set_overlap`
    - Everything else (including `None`) → `compare_exact`

    Args:
        expected (M): Ground-truth model instance.
        extracted (M): Model instance produced by the judge LLM.
        field_configs (dict[str, FieldConfig] | None): Optional mapping of field
            names to per-field `FieldConfig` instances.  When a field is absent
            from this mapping an empty `FieldConfig()` is used, which causes each
            strategy to fall back to its built-in default.
        default_tolerance (float): Fallback absolute tolerance used for numeric
            fields whose `FieldConfig.tolerance` is `None`.  Defaults to
            `DEFAULT_NUMERIC_TOLERANCE`.

    Returns:
        list[FieldComparison]: One `FieldComparison` per field in `expected`.
    """
    resolved_configs = field_configs or {}
    comparisons = [
        _dispatch_comparison(
            field_name=field_name,
            expected_value=getattr(expected, field_name),
            extracted_value=getattr(extracted, field_name),
            config=_resolve_field_config(resolved_configs.get(field_name), default_tolerance),
        )
        for field_name in type(expected).model_fields
    ]

    return comparisons


# endregion

# region Private helpers


def _resolve_field_config(config: FieldConfig | None, default_tolerance: float) -> FieldConfig:
    """Merge a per-field config with the caller-level default tolerance.

    When `config` already has an explicit `tolerance`, it is kept as-is.
    When `config.tolerance` is `None`, `default_tolerance` is applied.

    Args:
        config (FieldConfig | None): Per-field config, or `None` if none was
            provided.
        default_tolerance (float): Tolerance to use when `config.tolerance` is
            `None`.

    Returns:
        FieldConfig: A `FieldConfig` with `tolerance` always set.
    """
    base = config or FieldConfig()
    if base.tolerance is None:
        return base.model_copy(update={"tolerance": default_tolerance})
    return base


def _to_str_set(items: Collection[str], *, case_sensitive: bool) -> set[str]:
    """Convert a collection of strings to a set, optionally lowercased.

    Args:
        items (Collection[str]): Input collection of strings.
        case_sensitive (bool): When `False`, all strings are lowercased.

    Returns:
        set[str]: Normalized set of strings.
    """
    if case_sensitive:
        return set(items)
    return {item.lower() for item in items}


def _dispatch_comparison(
    field_name: str,
    expected_value: Any,
    extracted_value: Any,
    config: FieldConfig,
) -> FieldComparison:
    """Select and invoke the appropriate comparison strategy based on runtime type.

    Dispatch is based on `isinstance` checks against the concrete Python type
    of `expected_value`.  When `expected_value` is `None` (e.g. for
    `Optional`-typed fields), dispatch falls back to the type of
    `extracted_value` so that the correct strategy is still selected:

    - `float` → :func:`compare_numeric`
    - `dict` → :func:`compare_dict`
    - `list` where all elements are `str` → :func:`compare_set_overlap`
    - Everything else (including both values being `None`) → :func:`compare_exact`

    Args:
        field_name (str): Name of the field being compared.
        expected_value (Any): Expected value from ground truth.
        extracted_value (Any): Value produced by the judge LLM.
        config (FieldConfig): Per-field configuration supplying optional
            overrides for tolerance, threshold, and case sensitivity.

    Returns:
        FieldComparison: Result of the selected comparison strategy.
    """
    dispatch_value = expected_value if expected_value is not None else extracted_value

    if isinstance(dispatch_value, float):
        tolerance = config.tolerance if config.tolerance is not None else DEFAULT_NUMERIC_TOLERANCE
        return compare_numeric(field_name, expected_value, extracted_value, tolerance=tolerance)

    if isinstance(dispatch_value, dict):
        return compare_dict(field_name, expected_value, extracted_value, config=config)

    if isinstance(dispatch_value, list) and all(isinstance(item, str) for item in dispatch_value):
        threshold = config.threshold if config.threshold is not None else DEFAULT_SET_OVERLAP_THRESHOLD
        case_sensitive = config.case_sensitive if config.case_sensitive is not None else False
        return compare_set_overlap(
            field_name,
            expected_value,
            extracted_value,
            threshold=threshold,
            case_sensitive=case_sensitive,
        )

    return compare_exact(field_name, expected_value, extracted_value)


# endregion
