"""Custom exceptions for the DataFrame toolkit.

This module defines exceptions for DataFrame column validation and SQL validation:

Column validation exceptions (subclass ValueError):
- ColumnsNotFoundError: Raised when requested columns do not exist in a DataFrame.
- DuplicateColumnsError: Raised when duplicate column names are provided.

SQL validation exceptions (subclass Exception):
- SQLValidationError: Base class for all SQL validation errors. Catch this to
  handle any SQL validation failure.
- SQLSyntaxError: Raised when SQL has invalid syntax (parse errors).
- SQLTableError: Raised when SQL references non-existent tables.
- SQLColumnError: Raised when SQL references non-existent columns.
- SQLBlacklistedCommandError: Raised when SQL contains a blacklisted command type.
"""

from __future__ import annotations

from typing import TypedDict


class ParseErrorDict(TypedDict, total=False):
    """Typed dictionary representing a single SQL parse error.

    All fields are optional to support partial error information from different
    SQL parsers that may not provide all details.

    Attributes:
        description (str): Human-readable explanation of the error.
        line (int): Line number where the error occurred (1-indexed).
        col (int): Column number where the error occurred (1-indexed).
        start_context (str): Text appearing before the error location.
        highlight (str): The problematic text segment that caused the error.
        end_context (str): Text appearing after the error location.
    """

    description: str
    line: int
    col: int
    start_context: str
    highlight: str
    end_context: str


class ColumnsNotFoundError(ValueError):
    """Raised when requested columns do not exist in a DataFrame.

    Attributes:
        missing_columns (list[str]): Column names that were not found.
        available_columns (list[str]): Column names present in the DataFrame.

    Examples:
        >>> err = ColumnsNotFoundError(
        ...     missing_columns=["x", "y"],
        ...     available_columns=["a", "b", "c"],
        ... )
        >>> err.missing_columns
        ['x', 'y']
    """

    missing_columns: list[str]
    available_columns: list[str]

    def __init__(
        self,
        missing_columns: list[str],
        available_columns: list[str],
    ) -> None:
        """Initialize ColumnsNotFoundError.

        Args:
            missing_columns (list[str]): Column names not found in the DataFrame.
            available_columns (list[str]): Column names present in the DataFrame.
        """
        super().__init__(f"Columns not found in DataFrame: {sorted(missing_columns)}")
        self.missing_columns = missing_columns
        self.available_columns = available_columns


class DuplicateColumnsError(ValueError):
    """Raised when duplicate column names are provided.

    Attributes:
        columns (list[str]): The column list that contains duplicates.
        duplicate_columns (list[str]): The specific column names that are
            duplicated (each listed once).

    Examples:
        >>> err = DuplicateColumnsError(columns=["a", "a", "b"])
        >>> err.columns
        ['a', 'a', 'b']
        >>> err.duplicate_columns
        ['a']
    """

    columns: list[str]
    duplicate_columns: list[str]

    def __init__(self, columns: list[str]) -> None:
        """Initialize DuplicateColumnsError.

        Args:
            columns (list[str]): The column list containing duplicates.
        """
        super().__init__("Duplicate column names are not allowed")
        self.columns = columns
        seen: set[str] = set()
        self.duplicate_columns = []
        for col in columns:
            if col in seen:
                self.duplicate_columns.append(col)
            seen.add(col)


class SQLValidationError(Exception):
    """Base exception for all SQL validation errors.

    Use this as the base class for SQL-related validation failures. Catching
    this exception will catch all SQL validation errors from the toolkit.

    Attributes:
        query (str | None): The SQL query that failed validation.
    """

    query: str | None

    def __init__(self, message: str, query: str | None = None) -> None:
        """Initialize SQLValidationError.

        Args:
            message (str): Description of the validation error.
            query (str | None): The SQL query that failed validation.
        """
        super().__init__(message)
        self.query = query

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns:
            str: Detailed string representation including message and query.
        """
        return f"{self.__class__.__name__}(message={str(self)!r}, query={self.query!r})"


class SQLSyntaxError(SQLValidationError):
    """Raised when a SQL query has invalid syntax.

    This exception is raised when the SQL parser cannot parse the query
    due to malformed SQL syntax. Supports multiple parse errors similar to
    SQLglot's ParseError, with each error containing optional location and
    context information.

    Attributes:
        errors (list[ParseErrorDict]): List of parse error dictionaries, each
            containing optional keys: description, line, col, start_context,
            highlight, end_context.

    Examples:
        >>> err = SQLSyntaxError(
        ...     message="Multiple syntax errors",
        ...     errors=[
        ...         {"description": "Missing FROM", "line": 1},
        ...         {"description": "Invalid column", "line": 2},
        ...     ],
        ... )
        >>> len(err.errors)
        2
    """

    errors: list[ParseErrorDict]

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        errors: list[ParseErrorDict] | None = None,
    ) -> None:
        """Initialize SQLSyntaxError.

        Args:
            message (str): Description of the syntax error.
            query (str | None): The SQL query that failed parsing.
            errors (list[ParseErrorDict] | None): List of parse error dictionaries
                with optional keys: description, line, col, start_context,
                highlight, end_context.
        """
        super().__init__(message, query=query)
        self.errors = errors or []

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns:
            str: Detailed string representation including message, query, and errors.
        """
        return f"{self.__class__.__name__}(message={str(self)!r}, query={self.query!r}, errors={self.errors!r})"


class SQLTableError(SQLValidationError):
    """Raised when a SQL query references invalid tables.

    This exception is raised when the SQL query references table names
    that are not registered in the DataFrame context.

    Attributes:
        invalid_tables (list[str]): List of table names that are not registered.
    """

    invalid_tables: list[str]

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        invalid_tables: list[str] | None = None,
    ) -> None:
        """Initialize SQLTableError.

        Args:
            message (str): Description of the table error.
            query (str | None): The SQL query with invalid table references.
            invalid_tables (list[str] | None): List of table names that are not registered.
        """
        super().__init__(message, query=query)
        self.invalid_tables = invalid_tables or []

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns:
            str: Detailed string representation including message, query, and invalid tables.
        """
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, query={self.query!r}, "
            f"invalid_tables={self.invalid_tables!r})"
        )


class SQLColumnError(SQLValidationError):
    """Raised when a SQL query has column reference errors.

    This exception is raised when the SQL query references column names
    that do not exist in the specified tables, when column references
    are ambiguous (exist in multiple tables without qualification), or
    when columns are not found in any table for multi-table queries.

    Attributes:
        invalid_columns (dict[str, list[str]]): Mapping of table names to lists of
            invalid column names that don't exist in that table.
        ambiguous_columns (dict[str, list[str]]): Mapping of column names to lists of
            table names where the column exists but the reference is ambiguous.
        not_found_columns (dict[str, list[str]]): Mapping of column names to lists of
            table names that were searched when the column was not found in any table.
        table_columns (dict[str, set[str]]): The schema used for validation, mapping
            table names to their valid column sets.

    Examples:
        Invalid column:
        >>> err = SQLColumnError(
        ...     message="Invalid column references",
        ...     invalid_columns={"users": ["col_z"]},
        ...     table_columns={"users": {"id", "name", "email"}},
        ... )
        >>> print(err.format_details())
        Column "col_z" not found in table "users". Available columns: email, id, name

        Ambiguous column:
        >>> err = SQLColumnError(
        ...     message="Ambiguous column references",
        ...     ambiguous_columns={"id": ["users", "orders"]},
        ...     table_columns={"users": {"id", "name"}, "orders": {"id", "total"}},
        ... )
        >>> print(err.format_details())
        Column "id" is ambiguous. Found in tables: orders, users. Please qualify as "orders.id" or "users.id".

        Not found in any table:
        >>> err = SQLColumnError(
        ...     message="Column not found",
        ...     not_found_columns={"nonexistent": ["users", "orders"]},
        ...     table_columns={"users": {"id", "name"}, "orders": {"id", "total"}},
        ... )
        >>> print(err.format_details())
        Column "nonexistent" not found in any table. Searched tables: orders, users
    """

    invalid_columns: dict[str, list[str]]
    ambiguous_columns: dict[str, list[str]]
    not_found_columns: dict[str, list[str]]
    table_columns: dict[str, set[str]]

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        invalid_columns: dict[str, list[str]] | None = None,
        ambiguous_columns: dict[str, list[str]] | None = None,
        not_found_columns: dict[str, list[str]] | None = None,
        table_columns: dict[str, set[str]] | None = None,
    ) -> None:
        """Initialize SQLColumnError.

        Args:
            message (str): Description of the column error.
            query (str | None): The SQL query with column reference errors.
            invalid_columns (dict[str, list[str]] | None): Mapping of table names to
                lists of invalid column names referenced for that table.
            ambiguous_columns (dict[str, list[str]] | None): Mapping of column names
                to lists of table names where the column exists but is ambiguous.
            not_found_columns (dict[str, list[str]] | None): Mapping of column names
                to lists of table names that were searched.
            table_columns (dict[str, set[str]] | None): The schema used for validation,
                mapping table names to their valid column sets.
        """
        super().__init__(message, query=query)
        self.invalid_columns = invalid_columns or {}
        self.ambiguous_columns = ambiguous_columns or {}
        self.not_found_columns = not_found_columns or {}
        self.table_columns = table_columns or {}

    def format_details(self) -> str:
        """Format detailed error messages for invalid, ambiguous, and not-found columns.

        Returns:
            str: Multi-line string with one line per error, showing details
                about invalid columns (with available alternatives),
                ambiguous columns (with qualification suggestions), and
                not-found columns (with searched tables).

        Examples:
            >>> err = SQLColumnError(
            ...     message="Column errors",
            ...     invalid_columns={"users": ["foo"]},
            ...     ambiguous_columns={"id": ["users", "orders"]},
            ...     table_columns={"users": {"id", "name"}, "orders": {"id"}},
            ... )
            >>> print(err.format_details())
            Column "foo" not found in table "users". Available columns: id, name
            Column "id" is ambiguous. Found in tables: orders, users. Please qualify as "orders.id" or "users.id".
        """
        lines = []
        # Format invalid columns
        for table_name, columns in sorted(self.invalid_columns.items()):
            available = self.table_columns.get(table_name, set())
            available_str = ", ".join(sorted(available)) if available else "(none)"
            for col in sorted(columns):
                lines.append(f'Column "{col}" not found in table "{table_name}". Available columns: {available_str}')
        # Format ambiguous columns
        for col_name, tables in sorted(self.ambiguous_columns.items()):
            sorted_tables = sorted(tables)
            tables_str = ", ".join(sorted_tables)
            qualify_options = " or ".join(f'"{t}.{col_name}"' for t in sorted_tables)
            lines.append(
                f'Column "{col_name}" is ambiguous. Found in tables: {tables_str}. Please qualify as {qualify_options}.'
            )
        # Format not-found columns (multi-table queries)
        for col_name, tables in sorted(self.not_found_columns.items()):
            sorted_tables = sorted(tables)
            tables_str = ", ".join(sorted_tables)
            lines.append(f'Column "{col_name}" not found in any table. Searched tables: {tables_str}')
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns:
            str: Detailed string representation including message, query, and column errors.
        """
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, query={self.query!r}, "
            f"invalid_columns={self.invalid_columns!r}, "
            f"ambiguous_columns={self.ambiguous_columns!r}, "
            f"not_found_columns={self.not_found_columns!r}, "
            f"table_columns={self.table_columns!r})"
        )


class SQLBlacklistedCommandError(SQLValidationError):
    """Raised when a SQL query contains a blacklisted command type.

    This exception is raised when the SQL query contains a command type
    that is not allowed, such as DELETE, DROP, INSERT, UPDATE, or other
    data-modifying statements that may be restricted for safety.

    Attributes:
        command_type (str): The detected SQL command type (e.g., "DELETE", "DROP").
        blacklist (set[str]): The set of blacklisted commands that are not allowed.

    Example:
        >>> err = SQLBlacklistedCommandError(
        ...     message="Command 'DELETE' is not allowed",
        ...     query="DELETE FROM users WHERE id = 1",
        ...     command_type="DELETE",
        ...     blacklist={"DELETE", "DROP", "INSERT", "UPDATE"},
        ... )
        >>> err.command_type
        'DELETE'
        >>> "DELETE" in err.blacklist
        True
    """

    command_type: str
    blacklist: set[str]

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        command_type: str = "",
        blacklist: set[str] | None = None,
    ) -> None:
        """Initialize SQLBlacklistedCommandError.

        Args:
            message (str): Description of the blacklist violation.
            query (str | None): The SQL query containing the blacklisted command.
            command_type (str): The detected SQL command type (e.g., "DELETE", "DROP").
            blacklist (set[str] | None): The set of blacklisted commands that are not
                allowed. Defaults to an empty set if not provided.
        """
        super().__init__(message, query=query)
        self.command_type = command_type
        self.blacklist = blacklist or set()

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns:
            str: Detailed string representation including message, query, command type, and blacklist.
        """
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, query={self.query!r}, "
            f"command_type={self.command_type!r}, blacklist={self.blacklist!r})"
        )
