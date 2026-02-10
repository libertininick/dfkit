"""SQL utilities for parsing and validating SQL queries.

This module provides functions for validating SQL syntax using SQLglot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, build_scope, find_all_in_scope

from dfkit.exceptions import (
    ParseErrorDict,
    SQLBlacklistedCommandError,
    SQLColumnError,
    SQLSyntaxError,
    SQLTableError,
)

__all__ = ["DESTRUCTIVE_COMMANDS", "extract_table_names", "parse_sql", "validate_sql"]


# =============================================================================
# Constants
# =============================================================================

# Common destructive SQL commands that modify or delete data/schema.
# Use with parse_sql's blacklist parameter to block these operations.
DESTRUCTIVE_COMMANDS: Final[frozenset[str]] = frozenset({
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "TRUNCATE",
    "ALTER",
    "CREATE",
})

# Mapping of sqlglot expression types to SQL command type strings for blacklist checking.
# Set operations (Union, Intersect, Except) are considered SELECT queries.
_EXPRESSION_TYPE_MAP: Final[dict[type[exp.Expression], str]] = {
    exp.Select: "SELECT",
    exp.Delete: "DELETE",
    exp.Insert: "INSERT",
    exp.Update: "UPDATE",
    exp.Drop: "DROP",
    exp.Create: "CREATE",
    exp.TruncateTable: "TRUNCATE",
    exp.Alter: "ALTER",
    exp.Union: "SELECT",
    exp.Intersect: "SELECT",
    exp.Except: "SELECT",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class _ColumnValidationResult:
    """Result of validating a single column reference.

    Attributes:
        col_name (str): The name of the column being validated.
        invalid_table (str | None): Table name if column is invalid, None otherwise.
        ambiguous_tables (list[str] | None): Tables containing ambiguous column, None otherwise.
        not_found_in_tables (list[str] | None): Tables searched when column not found in any, None otherwise.
    """

    col_name: str
    invalid_table: str | None = None
    ambiguous_tables: list[str] | None = None
    not_found_in_tables: list[str] | None = None


@dataclass
class _ColumnErrors:
    """Aggregated column validation errors from a SQL expression.

    Attributes:
        invalid_columns (dict[str, list[str]]): Mapping of table names to lists of
            invalid column names that don't exist in that table.
        ambiguous_columns (dict[str, list[str]]): Mapping of column names to lists of
            table names where the column exists but the reference is ambiguous.
        not_found_columns (dict[str, list[str]]): Mapping of column names to lists of
            table names that were searched when the column was not found in any table.
    """

    invalid_columns: dict[str, list[str]]
    ambiguous_columns: dict[str, list[str]]
    not_found_columns: dict[str, list[str]]

    @property
    def has_errors(self) -> bool:
        """Return True if any column errors were detected."""
        return bool(self.invalid_columns or self.ambiguous_columns or self.not_found_columns)


# =============================================================================
# Public Interface
# =============================================================================


def parse_sql(
    query: str, *, dialect: str | None = None, blacklist: Collection[str] | None = None
) -> sqlglot.Expression:
    """Parses the query using SQLglot to detect syntax errors and returns the parsed expression.

    If the query is syntactically valid, returns normally. If the query has syntax
    errors, raises SQLSyntaxError with details about the parse errors. Optionally
    validates the command type against a blacklist of disallowed commands.

    Args:
        query (str): The SQL query string to validate.
        dialect (str | None): Optional SQL dialect to use for parsing. Defaults to None.
        blacklist (Collection[str] | None): Optional collection of SQL command types to block
            (e.g., {"DELETE", "DROP"}). Matching is case-insensitive. Use
            DESTRUCTIVE_COMMANDS for a pre-defined set of data-modifying commands.
            Defaults to None (no blacklist checking).

    Returns:
        sqlglot.Expression: The parsed SQL expression if the query is valid.

    Raises:
        SQLSyntaxError: If the query is empty, contains only whitespace, or has
            invalid SQL syntax. The exception's `errors` attribute contains a list
            of details about each parse error (description, line, col, context).
        SQLBlacklistedCommandError: If the query's command type is in the blacklist.
            The exception includes the detected command_type and the blacklist.

    Examples:
        Valid SQL query:
        >>> expression = parse_sql("SELECT a FROM t")

        Invalid SQL query:
        >>> try:
        ...     parse_sql("SELECT * FROM (SELECT a FROM t") # Missing closing parenthesis
        ... except SQLSyntaxError as e:
        ...     print("Syntax error caught")
        Syntax error caught

        Blocking destructive commands:
        >>> try:
        ...     parse_sql("DELETE FROM users", blacklist=DESTRUCTIVE_COMMANDS)
        ... except SQLBlacklistedCommandError as e:
        ...     print(f"Blocked: {e.command_type}")
        Blocked: DELETE
    """
    if not query or not query.strip():
        raise SQLSyntaxError("SQL query cannot be empty or whitespace-only", query=query, errors=[])

    try:
        expression = sqlglot.parse_one(query, dialect=dialect)
    except sqlglot.errors.ParseError as e:
        errors: list[ParseErrorDict] = [
            ParseErrorDict(
                description=error_dict.get("description", ""),
                line=error_dict.get("line", 0),
                col=error_dict.get("col", 0),
                start_context=error_dict.get("start_context", ""),
                highlight=error_dict.get("highlight", ""),
                end_context=error_dict.get("end_context", ""),
            )
            for error_dict in e.errors
        ]

        raise SQLSyntaxError(
            message=f"SQL syntax error: {e}",
            query=query,
            errors=errors,
        ) from e

    if blacklist and (command_type := _get_sql_command_type(expression)) is not None:
        normalized_blacklist = {cmd.upper() for cmd in blacklist}
        if command_type.upper() in normalized_blacklist:
            raise SQLBlacklistedCommandError(
                message=f"SQL command '{command_type}' is not allowed.",
                query=query,
                command_type=command_type,
                blacklist=normalized_blacklist,
            )

    return expression


def validate_sql(
    query: str,
    table_columns: dict[str, set[str]],
    *,
    dialect: str | None = None,
    blacklist: Collection[str] | None = None,
) -> exp.Expression:
    """Validate a SQL query through comprehensive multi-step validation.

    Orchestrates full SQL validation by running parsing, table validation,
    and column validation in sequence. This provides a single entry point
    for complete SQL validation in the DataFrame toolkit.

    The validation steps are:
    1. Parse the query using `parse_sql()` with optional blacklist
    2. Validate table references
    3. Validate column references

    Args:
        query (str): The SQL query string to validate.
        table_columns (dict[str, set[str]]): Mapping of table names to their
            valid column names. This schema is used for both table validation
            (tables must be in the schema) and column validation.
        dialect (str | None): Optional SQL dialect to use for parsing.
            Defaults to None.
        blacklist (Collection[str] | None): Optional collection of SQL command
            types to block (e.g., {"DELETE", "DROP"}). Use `DESTRUCTIVE_COMMANDS`
            for a pre-defined set. Defaults to None (no blacklist checking).

    Returns:
        exp.Expression: The parsed and validated SQL expression.

    Raises:
        SQLSyntaxError: If the query has invalid SQL syntax or is empty.
        SQLBlacklistedCommandError: If the query's command type is in the blacklist.
        SQLTableError: If the query references tables not in `table_columns`.
        SQLColumnError: If the query references invalid or ambiguous columns.

    Examples:
        Successful validation:
        >>> expression = validate_sql(
        ...     "SELECT id, name FROM users",
        ...     {"users": {"id", "name", "email"}},
        ... )

        Invalid table reference:
        >>> try:
        ...     validate_sql(
        ...         "SELECT id FROM unknown_table",
        ...         {"users": {"id", "name"}},
        ...     )
        ... except SQLTableError as e:
        ...     print(f"Invalid tables: {e.invalid_tables}")
        Invalid tables: ['unknown_table']

        Invalid column reference:
        >>> try:
        ...     validate_sql(
        ...         "SELECT bad_col FROM users",
        ...         {"users": {"id", "name"}},
        ...     )
        ... except SQLColumnError as e:
        ...     print(e.format_details())
        Column "bad_col" not found in table "users". Available columns: id, name

        Ambiguous column reference:
        >>> try:
        ...     validate_sql(
        ...         "SELECT id FROM users, orders",
        ...         {"users": {"id", "name"}, "orders": {"id", "total"}},
        ...     )
        ... except SQLColumnError as e:
        ...     print(e.format_details())
        Column "id" is ambiguous. Found in tables: orders, users. Please qualify as "orders.id" or "users.id".

        Blocking destructive commands:
        >>> try:
        ...     validate_sql(
        ...         "DELETE FROM users",
        ...         {"users": {"id", "name"}},
        ...         blacklist=DESTRUCTIVE_COMMANDS,
        ...     )
        ... except SQLBlacklistedCommandError as e:
        ...     print(f"Blocked: {e.command_type}")
        Blocked: DELETE
    """  # noqa: DOC502 # We want to explicitly document exceptions raised by helpers.
    # Step 1: Parse the query
    expression = parse_sql(query, dialect=dialect, blacklist=blacklist)

    # Step 2: Validate table references
    _validate_sql_tables(expression, valid_tables=table_columns.keys(), query_str=query)

    # Step 3: Validate column references (includes ambiguity detection)
    _validate_sql_columns(expression, table_columns, query_str=query)

    return expression


def extract_table_names(expression: exp.Expression) -> list[str]:
    """Extract table names from a parsed SQL expression using scope traversal.

    Uses sqlglot's scope analysis to correctly distinguish actual database tables
    from CTEs and subqueries.

    Args:
        expression (exp.Expression): A parsed sqlglot expression.

    Returns:
        list[str]: List of lowercase table names referenced in the query.
            May contain duplicates if the same table is referenced multiple times
            (e.g., in self-joins). Returns empty list if no tables are found.
    """
    root = build_scope(expression)
    if root is None:
        # Defensive: build_scope returns None for non-queryable expressions (e.g., SHOW).
        # In practice, this shouldn't occur since validate_sql only processes parsed queries.
        return []  # pragma: no cover

    tables: list[exp.Table] = [
        source
        for scope in root.traverse()
        for _alias, (_node, source) in scope.selected_sources.items()
        if isinstance(source, exp.Table)
    ]

    return [table.name.lower() for table in tables]


# =============================================================================
# Private Helpers: Command Type
# =============================================================================


def _get_sql_command_type(expression: exp.Expression) -> str | None:
    """Map a sqlglot expression to its SQL command type string.

    Args:
        expression (exp.Expression): A parsed sqlglot expression.

    Returns:
        str | None: The SQL command type (e.g., "SELECT", "DELETE") or None if
            the expression type is not recognized.
    """
    return _EXPRESSION_TYPE_MAP.get(type(expression))


# =============================================================================
# Private Helpers: Table Validation
# =============================================================================


def _validate_sql_tables(expression: exp.Expression, valid_tables: Collection[str], query_str: str) -> None:
    """Validate that a SQL expression only references allowed tables.

    Extracts all table references using scope analysis to correctly distinguish
    actual database tables from CTEs. Validates that at least one valid table
    is referenced and no unknown tables are used.

    Args:
        expression (exp.Expression): A pre-parsed sqlglot Expression.
        valid_tables (Collection[str]): Collection of allowed table names. Matching is
            case-insensitive.
        query_str (str): The original query string for error messages.

    Raises:
        SQLTableError: If no valid tables are referenced, or if unknown tables
            are referenced. The exception includes the list of invalid table names.
    """
    referenced_table_names = extract_table_names(expression)

    if not referenced_table_names:
        raise SQLTableError(
            message="Query does not reference any tables. At least one table from valid_tables must be referenced.",
            query=query_str,
            invalid_tables=[],
        )

    normalized_valid_tables = {t.lower() for t in valid_tables}
    invalid_tables = sorted(set(referenced_table_names) - normalized_valid_tables)

    if invalid_tables:
        raise SQLTableError(
            message=f"Query references invalid tables: {invalid_tables}",
            query=query_str,
            invalid_tables=invalid_tables,
        )


# =============================================================================
# Private Helpers: Column Validation
# =============================================================================


def _validate_sql_columns(
    expression: exp.Expression,
    table_columns: dict[str, set[str]],
    query_str: str,
) -> None:
    """Validate that a SQL expression only references valid columns for base tables.

    Extracts column references and validates them against the provided schema.
    Detects two types of errors:

    1. Invalid columns: References to columns that don't exist in the table.
    2. Ambiguous columns: Unqualified column references that exist in multiple
       tables (e.g., `SELECT id FROM users, orders` where both have `id`).

    Only validates columns on base (real) tables; columns from derived tables
    (CTEs, subqueries) or tables not in the schema are intentionally skipped.

    Args:
        expression (exp.Expression): A pre-parsed sqlglot Expression.
        table_columns (dict[str, set[str]]): Mapping of table names to their
            valid column names. Matching is case-insensitive for both table
            names and column names.
        query_str (str): The original query string for error messages.

    Raises:
        SQLColumnError: If invalid or ambiguous columns are referenced. The
            exception includes invalid_columns (grouped by table),
            ambiguous_columns (column to tables mapping), and the schema.
    """
    normalized_schema: dict[str, set[str]] = {
        table_name.lower(): {col.lower() for col in columns} for table_name, columns in table_columns.items()
    }

    errors = _collect_column_errors(expression, normalized_schema)

    if errors.has_errors:
        raise SQLColumnError(
            message=_build_column_error_message(
                errors.invalid_columns, errors.ambiguous_columns, errors.not_found_columns, table_columns
            ),
            query=query_str,
            invalid_columns=errors.invalid_columns,
            ambiguous_columns=errors.ambiguous_columns,
            not_found_columns=errors.not_found_columns,
            table_columns=table_columns,
        )


def _collect_column_errors(
    expression: exp.Expression,
    normalized_schema: dict[str, set[str]],
) -> _ColumnErrors:
    """Collect invalid, ambiguous, and not-found column references from a SQL expression.

    Traverses all scopes in the expression and validates column references
    against the provided schema. Detects invalid columns (don't exist in the
    table), ambiguous columns (exist in multiple tables without qualification),
    and not-found columns (don't exist in any table for multi-table queries).

    Args:
        expression (exp.Expression): The parsed SQL expression.
        normalized_schema (dict[str, set[str]]): Lowercase schema mapping table names to column sets.

    Returns:
        _ColumnErrors: Aggregated column validation errors containing invalid_columns,
            ambiguous_columns, and not_found_columns mappings.
    """
    invalid_columns: dict[str, list[str]] = defaultdict(list)
    ambiguous_columns: dict[str, list[str]] = defaultdict(list)
    not_found_columns: dict[str, list[str]] = defaultdict(list)

    for result in _validate_column_references(expression, normalized_schema):
        if result.invalid_table:
            invalid_columns[result.invalid_table].append(result.col_name)
        elif result.ambiguous_tables:
            existing = set(ambiguous_columns[result.col_name])
            ambiguous_columns[result.col_name] = list(existing | set(result.ambiguous_tables))
        elif result.not_found_in_tables:
            existing = set(not_found_columns[result.col_name])
            not_found_columns[result.col_name] = list(existing | set(result.not_found_in_tables))

    return _ColumnErrors(
        invalid_columns=dict(invalid_columns),
        ambiguous_columns=dict(ambiguous_columns),
        not_found_columns=dict(not_found_columns),
    )


def _validate_column_references(
    expression: exp.Expression,
    normalized_schema: dict[str, set[str]],
) -> Iterator[_ColumnValidationResult]:
    """Yield validation results for all column references in a SQL expression.

    Args:
        expression (exp.Expression): The parsed SQL expression to validate.
        normalized_schema (dict[str, set[str]]): Schema with lowercase table and column names.

    Yields:
        _ColumnValidationResult: Validation result for each column reference found.
    """
    root = build_scope(expression)
    if root is None:
        return

    for scope in root.traverse():
        alias_to_table = _build_alias_to_table_map(scope)
        for column in find_all_in_scope(scope.expression, exp.Column):
            yield _validate_column_in_scope(column, alias_to_table, normalized_schema)


def _validate_column_in_scope(
    column: exp.Column,
    alias_to_table: dict[str, str],
    normalized_schema: dict[str, set[str]],
) -> _ColumnValidationResult:
    """Validate a single column reference against the schema.

    Args:
        column (exp.Column): The column expression to validate.
        alias_to_table (dict[str, str]): Mapping from alias to base table name.
        normalized_schema (dict[str, set[str]]): Lowercase schema mapping table names to column sets.

    Returns:
        _ColumnValidationResult: Result indicating if the column is invalid or ambiguous.
    """
    col_name = column.name.lower()
    table_alias = column.table.lower() if column.table else ""

    if table_alias:
        return _check_qualified_column(col_name, table_alias, alias_to_table, normalized_schema)
    return _check_unqualified_column(col_name, alias_to_table, normalized_schema)


def _check_qualified_column(
    col_name: str,
    table_alias: str,
    alias_to_table: dict[str, str],
    normalized_schema: dict[str, set[str]],
) -> _ColumnValidationResult:
    """Check if a qualified column reference (e.g., users.id) is valid.

    Args:
        col_name (str): The column name to validate.
        table_alias (str): The table alias used in the query.
        alias_to_table (dict[str, str]): Mapping from table aliases to base table names.
        normalized_schema (dict[str, set[str]]): Schema with lowercase table and column names.

    Returns:
        _ColumnValidationResult: Result indicating validity or the specific error.
    """
    base_table = alias_to_table.get(table_alias)
    if base_table is None or base_table not in normalized_schema:
        # No base table found for alias; skip validation
        return _ColumnValidationResult(col_name)
    if col_name not in normalized_schema[base_table]:
        # Column not found in the specified table
        return _ColumnValidationResult(col_name, invalid_table=base_table)
    # Column is valid
    return _ColumnValidationResult(col_name)


def _check_unqualified_column(
    col_name: str,
    alias_to_table: dict[str, str],
    normalized_schema: dict[str, set[str]],
) -> _ColumnValidationResult:
    """Check if an unqualified column reference is valid or ambiguous.

    Args:
        col_name (str): The column name to validate.
        alias_to_table (dict[str, str]): Mapping from table aliases to base table names.
        normalized_schema (dict[str, set[str]]): Schema with lowercase table and column names.

    Returns:
        _ColumnValidationResult: Result indicating validity or the specific error.

    Note:
        When alias_to_table is empty (e.g., for queries like `SELECT 1` that
        reference no tables), this function returns a valid result since there
        are no tables to validate against. This is correct behavior as column
        validation only applies to columns from base tables in the schema.
    """
    base_tables = list(alias_to_table.values())

    if len(base_tables) == 1:
        return _check_unqualified_column_single_table(col_name, base_tables[0], normalized_schema)

    return _check_unqualified_column_multi_table(col_name, base_tables, normalized_schema)


def _check_unqualified_column_single_table(
    col_name: str,
    base_table: str,
    normalized_schema: dict[str, set[str]],
) -> _ColumnValidationResult:
    """Check unqualified column against a single table.

    Args:
        col_name (str): The column name to validate.
        base_table (str): The table name to check against.
        normalized_schema (dict[str, set[str]]): Schema with lowercase table and column names.

    Returns:
        _ColumnValidationResult: Result indicating validity or the specific error.
    """
    if base_table in normalized_schema and col_name not in normalized_schema[base_table]:
        return _ColumnValidationResult(col_name, invalid_table=base_table)
    return _ColumnValidationResult(col_name)


def _check_unqualified_column_multi_table(
    col_name: str,
    base_tables: list[str],
    normalized_schema: dict[str, set[str]],
) -> _ColumnValidationResult:
    """Check unqualified column against multiple tables for ambiguity or not found.

    Args:
        col_name (str): The column name to validate.
        base_tables (list[str]): List of table names to check against.
        normalized_schema (dict[str, set[str]]): Schema with lowercase table and column names.

    Returns:
        _ColumnValidationResult: Result indicating validity, ambiguity, or not found.
    """
    tables_with_column = [t for t in base_tables if t in normalized_schema and col_name in normalized_schema[t]]

    if len(tables_with_column) > 1:
        return _ColumnValidationResult(col_name, ambiguous_tables=tables_with_column)

    if not tables_with_column:
        tables_searched = [t for t in base_tables if t in normalized_schema]
        if tables_searched:
            return _ColumnValidationResult(col_name, not_found_in_tables=tables_searched)

    return _ColumnValidationResult(col_name)


def _build_alias_to_table_map(scope: Scope) -> dict[str, str]:
    """Build a mapping from table aliases to base table names for a scope.

    Args:
        scope (Scope): A sqlglot scope object with selected_sources attribute.

    Returns:
        dict[str, str]: Mapping from lowercase alias to lowercase base table name.
            Only includes sources that are actual database tables, not CTEs or subqueries.
    """
    return {
        alias.lower(): source.name.lower()
        for alias, (_node, source) in scope.selected_sources.items()
        if isinstance(source, exp.Table)
    }


# =============================================================================
# Private Helpers: Error Messages
# =============================================================================


def _build_column_error_message(
    invalid_columns: dict[str, list[str]],
    ambiguous_columns: dict[str, list[str]],
    not_found_columns: dict[str, list[str]],
    table_columns: dict[str, set[str]],
) -> str:
    """Build a user-friendly error message for column reference errors.

    Args:
        invalid_columns (dict[str, list[str]]): Mapping of table names to lists of invalid column names.
        ambiguous_columns (dict[str, list[str]]): Mapping of column names to lists of tables where found.
        not_found_columns (dict[str, list[str]]): Mapping of column names to lists of tables searched.
        table_columns (dict[str, set[str]]): Original schema for looking up original-case table names.

    Returns:
        str: Semicolon-separated error message listing all column errors.
    """
    messages = []

    # Add invalid column messages
    for table_name, cols in sorted(invalid_columns.items()):
        original_table = _find_original_table_name(table_name, table_columns)
        for col in sorted(cols):
            messages.append(f'Column "{col}" not found in table "{original_table}"')

    # Add ambiguous column messages
    for col_name, tables in sorted(ambiguous_columns.items()):
        sorted_tables = sorted(tables)
        tables_str = ", ".join(sorted_tables)
        messages.append(f'Column "{col_name}" is ambiguous (found in: {tables_str})')

    # Add not-found column messages (for multi-table queries)
    for col_name, tables in sorted(not_found_columns.items()):
        sorted_tables = sorted(tables)
        tables_str = ", ".join(sorted_tables)
        messages.append(f'Column "{col_name}" not found in any table (searched: {tables_str})')

    return "; ".join(messages)


def _find_original_table_name(table_name: str, table_columns: dict[str, set[str]]) -> str:
    """Find the original-case table name from the schema.

    Args:
        table_name (str): Lowercase table name to look up.
        table_columns (dict[str, set[str]]): Original schema with potentially mixed-case table names.

    Returns:
        str: The original-case table name from the schema, or the input if not found.
    """
    return next(
        (orig_name for orig_name in table_columns if orig_name.lower() == table_name),
        table_name,
    )
