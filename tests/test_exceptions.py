"""Tests for custom exceptions.

This module tests the custom exception classes used for SQL validation errors
in the DataFrame toolkit, ensuring proper inheritance, attribute storage, and
catchability patterns.
"""

from __future__ import annotations

import pytest
from pytest_check import check

from dfkit.exceptions import (
    ParseErrorDict,
    SQLBlacklistedCommandError,
    SQLColumnError,
    SQLSyntaxError,
    SQLTableError,
    SQLValidationError,
)


class TestSQLValidationErrorBase:
    """Tests for SQLValidationError base exception class."""

    def test_sql_validation_error_is_base_exception(self) -> None:
        """Verify SQLValidationError can be raised and caught as Exception.

        The base SQLValidationError should be a proper subclass of Exception,
        allowing it to be raised and caught using standard exception handling.
        """
        query = "SELECT * FROM users"
        error = SQLValidationError("Validation failed", query=query)

        # Verify it's catchable as Exception
        with pytest.raises(Exception):  # noqa B017 # Intentional for test
            raise error

        # Verify instance checks
        with check:
            assert isinstance(error, Exception), "Should be an instance of Exception"
        with check:
            assert isinstance(error, SQLValidationError), "Should be an instance of SQLValidationError"

    def test_sql_validation_error_stores_query(self) -> None:
        """Verify SQLValidationError stores the query attribute correctly.

        The query that caused the validation error should be accessible
        via the query attribute for debugging and error reporting.
        """
        query = "SELECT id, name FROM products WHERE price > 100"
        message = "Validation failed"

        error = SQLValidationError(message, query=query)

        with check:
            assert error.query == query, "Query should be stored and retrievable"
        with check:
            assert str(error) == message, "Message should be accessible via str()"

    def test_sql_validation_error_with_none_query(self) -> None:
        """Verify SQLValidationError handles None query gracefully.

        When no query is provided, the query attribute should be None,
        supporting cases where the query is unknown or not applicable.
        """
        error = SQLValidationError("Validation failed")

        with check:
            assert error.query is None, "Query should be None when not provided"


class TestSQLSyntaxError:
    """Tests for SQLSyntaxError exception class."""

    def test_sql_syntax_error_inherits_from_validation_error(self) -> None:
        """Verify SQLSyntaxError can be caught as SQLValidationError.

        SQLSyntaxError should be a subclass of SQLValidationError, enabling
        catch-all handling for SQL validation errors while allowing specific
        handling when needed.
        """
        query = "SELEC * FORM users"  # Intentional typos
        errors: list[ParseErrorDict] = [{"description": "Unexpected token 'SELEC'", "line": 1, "col": 1}]
        error = SQLSyntaxError("Invalid SQL syntax", query=query, errors=errors)

        # Verify it's catchable as SQLValidationError
        with pytest.raises(SQLValidationError):
            raise error

        # Verify instance checks
        with check:
            assert isinstance(error, SQLValidationError), "Should be an instance of SQLValidationError"
        with check:
            assert isinstance(error, SQLSyntaxError), "Should be an instance of SQLSyntaxError"

    def test_sql_syntax_error_stores_errors(self) -> None:
        """Verify SQLSyntaxError stores the errors attribute correctly.

        The list of parse error dictionaries should be accessible via errors
        to provide detailed information about what went wrong during parsing,
        including location and context for each error.
        """
        query = "SELECT * FORM users"
        errors: list[ParseErrorDict] = [
            {
                "description": "Syntax error near 'FORM'. Expected 'FROM'.",
                "line": 1,
                "col": 10,
                "start_context": "SELECT * ",
                "highlight": "FORM",
                "end_context": " users",
            }
        ]
        message = "Invalid SQL syntax"

        error = SQLSyntaxError(message, query=query, errors=errors)

        with check:
            assert error.errors == errors, "Errors list should be stored and retrievable"
        with check:
            assert len(error.errors) == 1, "Errors list should contain one error"
        with check:
            assert error.errors[0]["description"] == "Syntax error near 'FORM'. Expected 'FROM'."
        with check:
            assert error.errors[0]["line"] == 1
        with check:
            assert error.errors[0]["col"] == 10
        with check:
            assert error.query == query, "Query should be inherited from base class"
        with check:
            assert str(error) == message, "Message should be accessible via str()"

    def test_sql_syntax_error_with_no_errors_defaults_to_empty_list(self) -> None:
        """Verify SQLSyntaxError defaults to empty list when no errors provided.

        When no errors list is provided, the attribute should default to
        an empty list rather than None for consistent iteration behavior.
        """
        error = SQLSyntaxError("Invalid SQL syntax", query="bad query")

        with check:
            assert error.errors == [], "Errors should default to empty list when not provided"

    def test_sql_syntax_error_with_multiple_errors(self) -> None:
        """Verify SQLSyntaxError correctly stores multiple parse errors.

        SQLSyntaxError should support multiple parse errors, similar to how
        SQLglot's ParseError can contain multiple error entries.
        """
        errors: list[ParseErrorDict] = [
            {"description": "Missing FROM clause", "line": 1},
            {"description": "Invalid column reference", "line": 2, "col": 5},
            {"description": "Unexpected end of query", "line": 3},
        ]
        error = SQLSyntaxError("Multiple syntax errors", query="bad sql", errors=errors)

        with check:
            assert len(error.errors) == 3, "Should store all three errors"
        with check:
            assert error.errors[0]["description"] == "Missing FROM clause"
        with check:
            assert error.errors[1]["line"] == 2
        with check:
            assert error.errors[1]["col"] == 5
        with check:
            assert "col" not in error.errors[0], "First error should not have col key"


class TestSQLTableError:
    """Tests for SQLTableError exception class."""

    def test_sql_table_error_inherits_from_validation_error(self) -> None:
        """Verify SQLTableError can be caught as SQLValidationError.

        SQLTableError should be a subclass of SQLValidationError, enabling
        unified exception handling across all SQL validation error types.
        """
        query = "SELECT * FROM nonexistent_table"
        invalid_tables = ["nonexistent_table"]
        error = SQLTableError("Invalid table reference", query=query, invalid_tables=invalid_tables)

        # Verify it's catchable as SQLValidationError
        with pytest.raises(SQLValidationError):
            raise error

        # Verify instance checks
        with check:
            assert isinstance(error, SQLValidationError), "Should be an instance of SQLValidationError"
        with check:
            assert isinstance(error, SQLTableError), "Should be an instance of SQLTableError"

    def test_sql_table_error_stores_invalid_tables(self) -> None:
        """Verify SQLTableError stores the invalid_tables attribute correctly.

        The list of invalid table names should be accessible via invalid_tables
        to enable error reporting that identifies exactly which tables were not found.
        """
        query = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        invalid_tables = ["users", "orders"]
        message = "Referenced tables not registered"

        error = SQLTableError(message, query=query, invalid_tables=invalid_tables)

        with check:
            assert error.invalid_tables == invalid_tables, "Invalid tables should be stored and retrievable"
        with check:
            assert error.query == query, "Query should be inherited from base class"
        with check:
            assert str(error) == message, "Message should be accessible via str()"

    def test_sql_table_error_with_none_invalid_tables(self) -> None:
        """Verify SQLTableError defaults to empty list when invalid_tables is None.

        When no invalid_tables is provided, the attribute should default to
        an empty list rather than None for consistent iteration behavior.
        """
        error = SQLTableError("Table error", query="SELECT * FROM tbl")

        with check:
            assert error.invalid_tables == [], "Invalid tables should default to empty list"

    def test_sql_table_error_with_single_table(self) -> None:
        """Verify SQLTableError handles single invalid table correctly.

        A common case is having just one invalid table reference, which
        should be stored and retrievable as a single-element list.
        """
        invalid_tables = ["missing_table"]
        error = SQLTableError("Table not found", query="SELECT * FROM missing_table", invalid_tables=invalid_tables)

        with check:
            assert len(error.invalid_tables) == 1, "Should have exactly one invalid table"
        with check:
            assert error.invalid_tables[0] == "missing_table", "Invalid table name should match"


class TestSQLColumnError:
    """Tests for SQLColumnError exception class."""

    def test_sql_column_error_inherits_from_validation_error(self) -> None:
        """Verify SQLColumnError can be caught as SQLValidationError.

        SQLColumnError should be a subclass of SQLValidationError, enabling
        unified exception handling across all SQL validation error types.
        """
        query = "SELECT nonexistent_col FROM users"
        invalid_columns = {"users": ["nonexistent_col"]}
        error = SQLColumnError("Invalid column reference", query=query, invalid_columns=invalid_columns)

        # Verify it's catchable as SQLValidationError
        with pytest.raises(SQLValidationError):
            raise error

        # Verify instance checks
        with check:
            assert isinstance(error, SQLValidationError), "Should be an instance of SQLValidationError"
        with check:
            assert isinstance(error, SQLColumnError), "Should be an instance of SQLColumnError"

    def test_sql_column_error_stores_invalid_columns(self) -> None:
        """Verify SQLColumnError stores the invalid_columns dict correctly.

        The mapping of table names to invalid column lists should be accessible
        via invalid_columns for detailed error reporting that identifies exactly
        which columns were not found in which tables.
        """
        query = "SELECT a.bad_col, b.missing FROM table_a a JOIN table_b b ON a.id = b.id"
        invalid_columns = {"table_a": ["bad_col"], "table_b": ["missing"]}
        message = "Referenced columns not found"

        error = SQLColumnError(message, query=query, invalid_columns=invalid_columns)

        with check:
            assert error.invalid_columns == invalid_columns, "Invalid columns dict should be stored and retrievable"
        with check:
            assert error.query == query, "Query should be inherited from base class"
        with check:
            assert str(error) == message, "Message should be accessible via str()"

    def test_sql_column_error_with_none_invalid_columns(self) -> None:
        """Verify SQLColumnError defaults to empty dict when invalid_columns is None.

        When no invalid_columns is provided, the attribute should default to
        an empty dict rather than None for consistent iteration behavior.
        """
        error = SQLColumnError("Column error", query="SELECT col FROM tbl")

        with check:
            assert error.invalid_columns == {}, "Invalid columns should default to empty dict"

    def test_sql_column_error_with_multiple_tables(self) -> None:
        """Verify SQLColumnError correctly stores columns for multiple tables.

        When multiple tables have invalid column references, all should be
        captured in the invalid_columns dict with proper table-to-columns mapping.
        """
        invalid_columns = {
            "users": ["nonexistent_field", "another_bad_col"],
            "orders": ["unknown_col"],
        }
        error = SQLColumnError(
            "Multiple column errors", query="SELECT u.nonexistent_field FROM users u", invalid_columns=invalid_columns
        )

        with check:
            assert "users" in error.invalid_columns, "Users table should be in invalid columns"
        with check:
            assert "orders" in error.invalid_columns, "Orders table should be in invalid columns"
        with check:
            assert len(error.invalid_columns["users"]) == 2, "Users should have 2 invalid columns"
        with check:
            assert len(error.invalid_columns["orders"]) == 1, "Orders should have 1 invalid column"


class TestSQLBlacklistedCommandError:
    """Tests for SQLBlacklistedCommandError exception class."""

    def test_sql_blacklisted_command_error_inherits_from_validation_error(self) -> None:
        """Verify SQLBlacklistedCommandError can be caught as SQLValidationError.

        SQLBlacklistedCommandError should be a subclass of SQLValidationError,
        enabling unified exception handling across all SQL validation error types.
        """
        query = "DELETE FROM users WHERE id = 1"
        command_type = "DELETE"
        blacklist = {"DELETE", "DROP", "INSERT", "UPDATE"}
        error = SQLBlacklistedCommandError(
            "Command 'DELETE' is not allowed",
            query=query,
            command_type=command_type,
            blacklist=blacklist,
        )

        # Verify it's catchable as SQLValidationError
        with pytest.raises(SQLValidationError):
            raise error

        # Verify instance checks
        with check:
            assert isinstance(error, SQLValidationError), "Should be an instance of SQLValidationError"
        with check:
            assert isinstance(error, SQLBlacklistedCommandError), "Should be an instance of SQLBlacklistedCommandError"

    def test_sql_blacklisted_command_error_stores_attributes(self) -> None:
        """Verify SQLBlacklistedCommandError stores all attributes correctly.

        The command_type and blacklist should be accessible via their respective
        attributes for detailed error reporting that identifies exactly which
        command was blocked and the full set of disallowed commands.
        """
        query = "DROP TABLE users"
        command_type = "DROP"
        blacklist = {"DELETE", "DROP", "INSERT", "UPDATE"}
        message = "Command 'DROP' is not allowed"

        error = SQLBlacklistedCommandError(message, query=query, command_type=command_type, blacklist=blacklist)

        with check:
            assert error.command_type == command_type, "Command type should be stored and retrievable"
        with check:
            assert error.blacklist == blacklist, "Blacklist should be stored and retrievable"
        with check:
            assert error.query == query, "Query should be inherited from base class"
        with check:
            assert str(error) == message, "Message should be accessible via str()"

    def test_sql_blacklisted_command_error_defaults(self) -> None:
        """Verify SQLBlacklistedCommandError uses correct default values.

        When only message is provided, command_type should default to empty string
        and blacklist should default to an empty set rather than None for
        consistent iteration behavior.
        """
        error = SQLBlacklistedCommandError("Blacklisted command detected")

        with check:
            assert not error.command_type, "Command type should default to empty string"
        with check:
            assert error.blacklist == set(), "Blacklist should default to empty set"
        with check:
            assert error.query is None, "Query should default to None"

    def test_sql_blacklisted_command_error_repr(self) -> None:
        """Verify SQLBlacklistedCommandError has informative repr output.

        The repr should include the class name, message, query, command_type,
        and blacklist for debugging purposes.
        """
        query = "INSERT INTO users VALUES (1, 'test')"
        command_type = "INSERT"
        blacklist = {"DELETE", "DROP", "INSERT"}
        message = "Command 'INSERT' is not allowed"

        error = SQLBlacklistedCommandError(message, query=query, command_type=command_type, blacklist=blacklist)

        repr_str = repr(error)

        with check:
            assert "SQLBlacklistedCommandError" in repr_str, "Repr should include class name"
        with check:
            assert message in repr_str, "Repr should include message"
        with check:
            assert query in repr_str, "Repr should include query"
        with check:
            assert command_type in repr_str, "Repr should include command_type"
        with check:
            assert "blacklist=" in repr_str, "Repr should include blacklist key"


class TestSQLColumnErrorAmbiguous:
    """Tests for SQLColumnError ambiguous column handling."""

    def test_sql_column_error_stores_ambiguous_columns(self) -> None:
        """Verify SQLColumnError stores the ambiguous_columns dict correctly.

        The mapping of column names to table lists should be accessible
        via ambiguous_columns for detailed error reporting.
        """
        query = "SELECT id FROM users, orders"
        ambiguous_columns = {"id": ["users", "orders"]}
        message = "Ambiguous column references"

        error = SQLColumnError(message, query=query, ambiguous_columns=ambiguous_columns)

        with check:
            assert error.ambiguous_columns == ambiguous_columns, "Ambiguous columns should be stored"
        with check:
            assert error.query == query, "Query should be inherited from base class"

    def test_sql_column_error_with_none_ambiguous_columns(self) -> None:
        """Verify SQLColumnError defaults to empty dict when ambiguous_columns is None."""
        error = SQLColumnError("Column error", query="SELECT col FROM tbl")

        with check:
            assert error.ambiguous_columns == {}, "Ambiguous columns should default to empty dict"

    def test_sql_column_error_format_details_ambiguous(self) -> None:
        """Verify format_details handles ambiguous columns correctly."""
        error = SQLColumnError(
            message="Ambiguous column references",
            ambiguous_columns={"id": ["users", "orders"]},
            table_columns={"users": {"id", "name"}, "orders": {"id", "total"}},
        )

        details = error.format_details()

        with check:
            assert 'Column "id" is ambiguous' in details, "Should mention column is ambiguous"
        with check:
            assert "orders" in details and "users" in details, "Should list tables"
        with check:
            assert "orders.id" in details or "users.id" in details, "Should show qualification options"

    def test_sql_column_error_format_details_both_invalid_and_ambiguous(self) -> None:
        """Verify format_details handles both invalid and ambiguous columns."""
        error = SQLColumnError(
            message="Column errors",
            invalid_columns={"users": ["foo"]},
            ambiguous_columns={"id": ["users", "orders"]},
            table_columns={"users": {"id", "name"}, "orders": {"id", "total"}},
        )

        details = error.format_details()

        with check:
            assert 'Column "foo" not found' in details, "Should mention invalid column"
        with check:
            assert 'Column "id" is ambiguous' in details, "Should mention ambiguous column"

    def test_sql_column_error_repr_includes_ambiguous_columns(self) -> None:
        """Verify repr includes ambiguous_columns."""
        error = SQLColumnError(
            message="Ambiguous",
            ambiguous_columns={"id": ["users", "orders"]},
        )

        repr_str = repr(error)

        with check:
            assert "ambiguous_columns=" in repr_str, "Repr should include ambiguous_columns"


class TestSQLColumnErrorNotFound:
    """Tests for SQLColumnError not-found column handling (multi-table queries)."""

    def test_sql_column_error_stores_not_found_columns(self) -> None:
        """Verify SQLColumnError stores the not_found_columns dict correctly.

        The mapping of column names to table lists (tables that were searched)
        should be accessible via not_found_columns for detailed error reporting.
        """
        query = "SELECT nonexistent FROM users, orders"
        not_found_columns = {"nonexistent": ["users", "orders"]}
        message = "Column not found in any table"

        error = SQLColumnError(message, query=query, not_found_columns=not_found_columns)

        with check:
            assert error.not_found_columns == not_found_columns, "Not found columns should be stored"
        with check:
            assert error.query == query, "Query should be inherited from base class"

    def test_sql_column_error_with_none_not_found_columns(self) -> None:
        """Verify SQLColumnError defaults to empty dict when not_found_columns is None."""
        error = SQLColumnError("Column error", query="SELECT col FROM tbl")

        with check:
            assert error.not_found_columns == {}, "Not found columns should default to empty dict"

    def test_sql_column_error_format_details_not_found(self) -> None:
        """Verify format_details handles not-found columns correctly."""
        error = SQLColumnError(
            message="Column not found",
            not_found_columns={"nonexistent": ["users", "orders"]},
            table_columns={"users": {"id", "name"}, "orders": {"id", "total"}},
        )

        details = error.format_details()

        with check:
            assert 'Column "nonexistent" not found in any table' in details, "Should mention column not found"
        with check:
            assert "orders" in details and "users" in details, "Should list searched tables"

    def test_sql_column_error_format_details_all_error_types(self) -> None:
        """Verify format_details handles invalid, ambiguous, and not-found columns."""
        error = SQLColumnError(
            message="Column errors",
            invalid_columns={"users": ["foo"]},
            ambiguous_columns={"id": ["users", "orders"]},
            not_found_columns={"nonexistent": ["users", "orders"]},
            table_columns={"users": {"id", "name"}, "orders": {"id", "total"}},
        )

        details = error.format_details()

        with check:
            assert 'Column "foo" not found' in details, "Should mention invalid column"
        with check:
            assert 'Column "id" is ambiguous' in details, "Should mention ambiguous column"
        with check:
            assert 'Column "nonexistent" not found in any table' in details, "Should mention not found column"

    def test_sql_column_error_repr_includes_not_found_columns(self) -> None:
        """Verify repr includes not_found_columns."""
        error = SQLColumnError(
            message="Not found",
            not_found_columns={"nonexistent": ["users", "orders"]},
        )

        repr_str = repr(error)

        with check:
            assert "not_found_columns=" in repr_str, "Repr should include not_found_columns"


class TestAllExceptionsStoreQuery:
    """Tests verifying all exception types store the query attribute."""

    @pytest.mark.parametrize(
        ("exception_class", "extra_kwargs"),
        [
            (SQLValidationError, {}),
            (SQLSyntaxError, {"errors": [{"description": "parse failed", "line": 1}]}),
            (SQLTableError, {"invalid_tables": ["tbl"]}),
            (SQLColumnError, {"invalid_columns": {"tbl": ["col"]}}),
            (SQLBlacklistedCommandError, {"command_type": "DELETE", "blacklist": {"DELETE", "DROP"}}),
        ],
        ids=[
            "SQLValidationError",
            "SQLSyntaxError",
            "SQLTableError",
            "SQLColumnError",
            "SQLBlacklistedCommandError",
        ],
    )
    def test_all_exceptions_store_query(
        self,
        exception_class: type[SQLValidationError],
        extra_kwargs: dict[str, object],
    ) -> None:
        """Verify all exception types in the hierarchy store the query attribute.

        All SQL validation exceptions should consistently store the query that
        caused the error, enabling uniform access to this debugging information.

        Args:
            exception_class: The exception class to test.
            extra_kwargs: Additional keyword arguments specific to each exception type.
        """
        query = "SELECT * FROM test_table WHERE id = 1"
        message = "Test error message"

        error = exception_class(message, query=query, **extra_kwargs)

        with check:
            assert error.query == query, f"{exception_class.__name__} should store query attribute"


class TestExceptionsCatchableByBaseClass:
    """Tests verifying multiple exception types can be caught by SQLValidationError."""

    def test_exceptions_catchable_by_base_class(self) -> None:
        """Verify a workflow where multiple exception types can be caught by SQLValidationError.

        This test simulates a real-world scenario where different SQL validation
        errors might occur, and all can be handled uniformly by catching the
        base SQLValidationError class.
        """
        # Test that all exception types are catchable by SQLValidationError
        errors_caught: list[str] = []
        error_types = [
            "syntax",
            "table",
            "column",
            "blacklist",
        ]
        for error_type in error_types:
            try:
                self._simulate_exceptions(error_type)
            except SQLValidationError as e:
                errors_caught.append(type(e).__name__)
                # Verify query is accessible on all caught exceptions
                with check:
                    assert e.query == "SELECT * FROM users", f"Query should be accessible on {type(e).__name__}"

        with check:
            assert len(errors_caught) == 4, "All four exceptions should be caught"
        with check:
            assert "SQLSyntaxError" in errors_caught, "SQLSyntaxError should be caught by base class"
        with check:
            assert "SQLTableError" in errors_caught, "SQLTableError should be caught by base class"
        with check:
            assert "SQLColumnError" in errors_caught, "SQLColumnError should be caught by base class"
        with check:
            assert "SQLBlacklistedCommandError" in errors_caught, (
                "SQLBlacklistedCommandError should be caught by base class"
            )

    def test_specific_exceptions_not_caught_by_siblings(self) -> None:
        """Verify that sibling exceptions do not catch each other.

        SQLSyntaxError, SQLTableError, and SQLColumnError are siblings in the
        hierarchy (all inherit from SQLValidationError), so one should not
        catch another.
        """  # noqa DOC501 # We are testing exception handling here.
        # SQLTableError should not be caught by SQLSyntaxError handler
        with pytest.raises(SQLTableError):
            try:
                raise SQLTableError("Table error", query="SELECT * FROM tbl", invalid_tables=["tbl"])
            except SQLSyntaxError:
                # This should not catch SQLTableError
                pass

        # SQLColumnError should not be caught by SQLTableError handler
        with pytest.raises(SQLColumnError):
            try:
                raise SQLColumnError("Column error", query="SELECT col FROM tbl", invalid_columns={"tbl": ["col"]})
            except SQLTableError:
                # This should not catch SQLColumnError
                pass

        # SQLSyntaxError should not be caught by SQLColumnError handler
        with pytest.raises(SQLSyntaxError):
            try:
                raise SQLSyntaxError(
                    "Syntax error", query="bad sql", errors=[{"description": "parse failed", "line": 1}]
                )
            except SQLColumnError:
                # This should not catch SQLSyntaxError
                pass

    @staticmethod
    def _simulate_exceptions(error_type: str) -> None:
        """Simulate different SQL validation errors.

        Raises a SQLValidationError subclass (SQLSyntaxError, SQLTableError,
        SQLColumnError, or SQLBlacklistedCommandError) based on error_type.

        Args:
            error_type (str): The type of error to simulate. One of "syntax", "table",
                "column", or "blacklist".

        Raises:
            ValueError: If an unknown error_type is provided.
        """
        query = "SELECT * FROM users"
        error_map: dict[str, SQLValidationError] = {
            "syntax": SQLSyntaxError("Syntax error", query=query, errors=[{"description": "Invalid token", "line": 1}]),
            "table": SQLTableError("Table not found", query=query, invalid_tables=["users"]),
            "column": SQLColumnError("Column not found", query=query, invalid_columns={"users": ["id"]}),
            "blacklist": SQLBlacklistedCommandError(
                "Blacklisted command", query=query, command_type="DELETE", blacklist={"DELETE", "DROP"}
            ),
        }
        if error_type not in error_map:
            raise ValueError(f"Unknown error_type: {error_type}")
        raise error_map[error_type]
