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
    >>> result = compare_numeric("score", 0.5, 0.5)
    >>> result.matched
    True
"""

import math
from collections.abc import Sequence
from typing import Any, Final

from pydantic import BaseModel

from dfkit.evaluation.models import FieldComparison

__all__ = [
    "ATOL",
    "RTOL",
    "compare_dict",
    "compare_exact",
    "compare_fields",
    "compare_numeric",
    "compare_numeric_sequence",
]

RTOL: Final[float] = 1e-5
ATOL: Final[float] = 1e-8

# region Public interface


def compare_numeric(
    field_name: str,
    expected: int | float | None,
    extracted: int | float | None,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> FieldComparison:
    """Compare two numeric values within a relative and absolute tolerance.

    Args:
        field_name (str): Name of the field being compared.
        expected (int | float | None): Expected numeric value from ground truth.
        extracted (int | float | None): Numeric value extracted by the judge LLM.
        rtol (float): Relative tolerance for the numeric comparison. Values
            within `rtol * expected` of each other are considered equal.
        atol (float): Absolute tolerance for the numeric comparison. Values
            within `atol` of each other are considered equal regardless of
            magnitude.

    Returns:
        FieldComparison: Comparison result with strategy "numeric_tolerance".
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

    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=math.isclose(expected, extracted, rel_tol=rtol, abs_tol=atol),
        strategy="numeric_tolerance",
    )


def compare_numeric_sequence(
    field_name: str,
    expected: Sequence[float] | None,
    extracted: Sequence[float] | None,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> FieldComparison:
    """Compare two sequences of floats element-wise within a tolerance.

    Args:
        field_name (str): Name of the field being compared.
        expected (Sequence[float] | None): Expected sequence from ground truth.
        extracted (Sequence[float] | None): Sequence extracted by the judge LLM.
        rtol (float): Relative tolerance for the numeric comparison. Values
            within `rtol * expected` of each other are considered equal.
        atol (float): Absolute tolerance for the numeric comparison. Values
            within `atol` of each other are considered equal regardless of
            magnitude.

    Returns:
        FieldComparison: Comparison result with strategy "numeric_sequence_tolerance".
            When both sequences are present and have the same length, details
            includes "element_results" (a list of per-element match booleans)
            and "length". When the lengths differ, details includes
            "expected_length" and "extracted_length".
    """
    if expected is None and extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=True,
            strategy="numeric_sequence_tolerance",
        )

    if expected is None or extracted is None:
        return FieldComparison(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            matched=False,
            strategy="numeric_sequence_tolerance",
        )

    if len(expected) != len(extracted):
        return FieldComparison(
            field_name=field_name,
            expected=list(expected),
            extracted=list(extracted),
            matched=False,
            strategy="numeric_sequence_tolerance",
            details={
                "expected_length": len(expected),
                "extracted_length": len(extracted),
            },
        )

    element_results = [math.isclose(e, x, rel_tol=rtol, abs_tol=atol) for e, x in zip(expected, extracted, strict=True)]

    return FieldComparison(
        field_name=field_name,
        expected=list(expected),
        extracted=list(extracted),
        matched=all(element_results),
        strategy="numeric_sequence_tolerance",
        details={
            "element_results": element_results,
            "length": len(expected),
        },
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
        FieldComparison: Comparison result with strategy "exact_match".
    """
    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=expected == extracted,
        strategy="exact_match",
    )


def compare_dict(
    field_name: str,
    expected: dict[str, Any] | None,
    extracted: dict[str, Any] | None,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> FieldComparison:
    """Compare two dictionaries key-by-key.

    Each value is compared using _dispatch_comparison so that nested floats,
    lists, and dicts are handled with the appropriate strategy automatically.

    Args:
        field_name (str): Name of the field being compared.
        expected (dict[str, Any] | None): Expected dictionary from ground truth.
        extracted (dict[str, Any] | None): Dictionary extracted by the judge LLM.
        rtol (float): Relative tolerance forwarded to numeric comparison strategies.
        atol (float): Absolute tolerance forwarded to numeric comparison strategies.

    Returns:
        FieldComparison: Comparison result with strategy "dict_comparison".
            Details includes "key_results" (per-key comparison info),
            "missing_keys" (keys present in expected but absent in extracted),
            and "extra_keys" (keys present in extracted but absent in
            expected).
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

    # Compare keys from expected dict and extracted dict
    expected_keys = set(expected)
    extracted_keys = set(extracted)
    missing_keys = sorted(expected_keys - extracted_keys)
    extra_keys = sorted(extracted_keys - expected_keys)
    shared_keys = expected_keys & extracted_keys

    # Compare values of shared keys
    results_per_key: dict[str, dict[str, Any]] = {}
    for key in sorted(shared_keys):
        key_comparison = _dispatch_comparison(
            field_name=key,
            expected_value=expected[key],
            extracted_value=extracted[key],
            rtol=rtol,
            atol=atol,
        )
        results_per_key[key] = {"matched": key_comparison.matched, "details": key_comparison.details}

    return FieldComparison(
        field_name=field_name,
        expected=expected,
        extracted=extracted,
        matched=not missing_keys and not extra_keys and all(result["matched"] for result in results_per_key.values()),
        strategy="dict_comparison",
        details={
            "results_per_key": results_per_key,
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
        },
    )


def compare_fields[M: BaseModel](
    expected: M,
    extracted: M,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> list[FieldComparison]:
    """Compare all fields of two Pydantic model instances.

    Iterates over the fields declared on `expected` and selects a comparison
    strategy for each one based on the runtime type of the expected value:

    - both values are None -> "none_comparison" (matched, no further dispatch)
    - float -> compare_numeric
    - dict -> compare_dict
    - Sequence[float] -> compare_numeric_sequence
    - Everything else -> compare_exact

    Args:
        expected (M): Ground-truth model instance.
        extracted (M): Model instance produced by the judge LLM.
        rtol (float): Relative tolerance forwarded to numeric comparison strategies.
        atol (float): Absolute tolerance forwarded to numeric comparison strategies.

    Returns:
        list[FieldComparison]: One FieldComparison per field in expected.

    Raises:
        TypeError: If expected and extracted are not the same model type.

    Examples:
        Exact match on string fields:

        >>> from pydantic import BaseModel
        >>> class Label(BaseModel):
        ...     name: str
        >>> results = compare_fields(Label(name="A"), Label(name="A"))
        >>> len(results)
        1
        >>> results[0].matched
        True

        Numeric tolerance on float fields:

        >>> class Score(BaseModel):
        ...     value: float
        >>> results = compare_fields(Score(value=1.0), Score(value=1.0 + 1e-7))
        >>> results[0].matched
        True
        >>> results[0].strategy
        'numeric_tolerance'

        Mixed field types dispatched to different strategies:

        >>> class Mixed(BaseModel):
        ...     label: str
        ...     score: float
        >>> results = compare_fields(
        ...     Mixed(label="X", score=0.5),
        ...     Mixed(label="X", score=0.5),
        ... )
        >>> [(r.field_name, r.strategy, r.matched) for r in results]
        [('label', 'exact_match', True), ('score', 'numeric_tolerance', True)]
    """
    if type(expected) is not type(extracted):
        raise TypeError(
            f"expected and extracted must be the same model type, "
            f"got {type(expected).__name__} and {type(extracted).__name__}"
        )
    return [
        _dispatch_comparison(
            field_name=field_name,
            expected_value=getattr(expected, field_name),
            extracted_value=getattr(extracted, field_name),
            rtol=rtol,
            atol=atol,
        )
        for field_name in type(expected).model_fields
    ]


# endregion

# region Private helpers


def _dispatch_comparison(
    *,
    field_name: str,
    expected_value: Any,
    extracted_value: Any,
    rtol: float,
    atol: float,
) -> FieldComparison:
    """Select and invoke the appropriate comparison strategy based on runtime type.

    Dispatch is based on isinstance checks against the concrete Python type
    of expected_value. When expected_value is None (e.g. for Optional-typed
    fields), dispatch falls back to the type of extracted_value so that the
    correct strategy is still selected:

    - both None -> "none_comparison" (early return, no further dispatch)
    - int | float (excluding bool) -> compare_numeric
    - dict -> compare_dict
    - Sequence[int | float] (excluding bool) -> compare_numeric_sequence
    - Everything else -> compare_exact

    Args:
        field_name (str): Name of the field being compared.
        expected_value (Any): Expected value from ground truth.
        extracted_value (Any): Value produced by the judge LLM.
        rtol (float): Relative tolerance forwarded to numeric comparison strategies.
        atol (float): Absolute tolerance forwarded to numeric comparison strategies.

    Returns:
        FieldComparison: Result of the selected comparison strategy.
    """
    if expected_value is None and extracted_value is None:
        return FieldComparison(
            field_name=field_name,
            expected=None,
            extracted=None,
            matched=True,
            strategy="none_comparison",
        )

    dispatch_value = expected_value if expected_value is not None else extracted_value

    if isinstance(dispatch_value, (int, float)) and not isinstance(dispatch_value, bool):
        return compare_numeric(field_name, expected_value, extracted_value, rtol=rtol, atol=atol)

    if isinstance(dispatch_value, dict):
        return compare_dict(field_name, expected_value, extracted_value, rtol=rtol, atol=atol)

    is_numeric_sequence = (
        isinstance(dispatch_value, Sequence)
        and not isinstance(dispatch_value, str)
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in dispatch_value)
    )
    if is_numeric_sequence:
        return compare_numeric_sequence(field_name, expected_value, extracted_value, rtol=rtol, atol=atol)

    return compare_exact(field_name, expected_value, extracted_value)


# endregion
