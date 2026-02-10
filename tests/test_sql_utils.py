"""Tests for SQL utility functions.

This module tests the SQL parsing and validation functions including:
- `parse_sql`: Parses SQL queries and validates syntax
- `validate_sql`: Comprehensive SQL validation (tables and columns)
- `_validate_sql_tables`: Private helper for table validation
- `_validate_sql_columns`: Private helper for column validation
"""

from __future__ import annotations

import pytest
import sqlglot
from pytest_check import check

from dfkit.exceptions import (
    SQLBlacklistedCommandError,
    SQLColumnError,
    SQLSyntaxError,
    SQLTableError,
    SQLValidationError,
)
from dfkit.sql_utils import (
    DESTRUCTIVE_COMMANDS,
    _get_sql_command_type,
    _validate_sql_columns,
    _validate_sql_tables,
    parse_sql,
    validate_sql,
)


class TestParseSQLValidQueries:
    """Tests for parse_sql with valid SQL queries.

    These tests verify that syntactically correct SQL queries return
    a sqlglot.Expression when parsed.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT a FROM t",
            "SELECT a, b, c FROM table1",
            "SELECT * FROM users",
            "SELECT DISTINCT name FROM employees",
            "SELECT COUNT(*) FROM orders",
            "SELECT a FROM t WHERE a > 1",
            "SELECT name, age FROM users WHERE active = true AND age >= 18",
            "SELECT * FROM products ORDER BY price DESC",
            "SELECT category, SUM(amount) FROM sales GROUP BY category",
            "SELECT * FROM items LIMIT 10 OFFSET 5",
        ],
        ids=[
            "simple_select",
            "select_multiple_columns",
            "select_star",
            "select_distinct",
            "select_count",
            "select_with_where",
            "select_with_compound_where",
            "select_with_order_by",
            "select_with_group_by",
            "select_with_limit_offset",
        ],
    )
    def test_parse_sql_valid_select_returns_expression(self, query: str) -> None:
        """Valid SELECT statements should return a sqlglot Expression.

        Verifies that various valid SELECT statement patterns are parsed
        successfully and return a sqlglot.Expression.

        Args:
            query: A syntactically valid SELECT statement.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid queries"

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT INTO users (name) VALUES ('John')",
            "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)",
            "INSERT INTO logs SELECT * FROM temp_logs",
        ],
        ids=[
            "insert_single_column",
            "insert_multiple_columns",
            "insert_from_select",
        ],
    )
    def test_parse_sql_valid_insert_returns_expression(self, query: str) -> None:
        """Valid INSERT statements should return a sqlglot Expression.

        Verifies that various valid INSERT statement patterns are parsed
        successfully and return a sqlglot.Expression.

        Args:
            query: A syntactically valid INSERT statement.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid INSERT"

    @pytest.mark.parametrize(
        "query",
        [
            "UPDATE users SET name = 'Jane'",
            "UPDATE products SET price = 19.99 WHERE id = 1",
            "UPDATE orders SET status = 'shipped', updated_at = NOW() WHERE order_id = 123",
        ],
        ids=[
            "update_simple",
            "update_with_where",
            "update_multiple_columns",
        ],
    )
    def test_parse_sql_valid_update_returns_expression(self, query: str) -> None:
        """Valid UPDATE statements should return a sqlglot Expression.

        Verifies that various valid UPDATE statement patterns are parsed
        successfully and return a sqlglot.Expression.

        Args:
            query: A syntactically valid UPDATE statement.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid UPDATE"

    @pytest.mark.parametrize(
        "query",
        [
            "DELETE FROM users",
            "DELETE FROM orders WHERE status = 'cancelled'",
            "DELETE FROM logs WHERE created_at < '2024-01-01'",
        ],
        ids=[
            "delete_all",
            "delete_with_where",
            "delete_with_date_condition",
        ],
    )
    def test_parse_sql_valid_delete_returns_expression(self, query: str) -> None:
        """Valid DELETE statements should return a sqlglot Expression.

        Verifies that various valid DELETE statement patterns are parsed
        successfully and return a sqlglot.Expression.

        Args:
            query: A syntactically valid DELETE statement.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid DELETE"


class TestParseSQLComplexQueries:
    """Tests for parse_sql with complex SQL patterns.

    These tests verify that advanced SQL features like JOINs, subqueries,
    and CTEs are properly parsed and return sqlglot.Expression.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT a.id, b.name FROM table_a a JOIN table_b b ON a.id = b.a_id",
            "SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            "SELECT * FROM products p LEFT JOIN categories c ON p.category_id = c.id",
            "SELECT * FROM employees e RIGHT JOIN departments d ON e.dept_id = d.id",
            "SELECT * FROM table1 t1 FULL OUTER JOIN table2 t2 ON t1.key = t2.key",
            "SELECT * FROM t1 CROSS JOIN t2",
            (
                "SELECT u.name, o.total, p.name AS product "
                "FROM users u "
                "JOIN orders o ON u.id = o.user_id "
                "JOIN order_items oi ON o.id = oi.order_id "
                "JOIN products p ON oi.product_id = p.id"
            ),
        ],
        ids=[
            "simple_join",
            "inner_join",
            "left_join",
            "right_join",
            "full_outer_join",
            "cross_join",
            "multi_table_join",
        ],
    )
    def test_parse_sql_valid_join_returns_expression(self, query: str) -> None:
        """Valid JOIN queries should return a sqlglot Expression.

        Verifies that various JOIN patterns (INNER, LEFT, RIGHT, FULL OUTER,
        CROSS, and multi-table JOINs) are parsed successfully.

        Args:
            query: A syntactically valid JOIN query.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid JOINs"

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)",
            "SELECT name, (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) AS order_count FROM users",
            "SELECT * FROM (SELECT id, name FROM users WHERE active = true) AS active_users",
            ("SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products)"),
            ("SELECT * FROM employees e WHERE EXISTS (SELECT 1 FROM managers m WHERE m.employee_id = e.id)"),
        ],
        ids=[
            "subquery_in_where",
            "correlated_subquery",
            "subquery_in_from",
            "scalar_subquery",
            "exists_subquery",
        ],
    )
    def test_parse_sql_valid_subquery_returns_expression(self, query: str) -> None:
        """Valid subqueries should return a sqlglot Expression.

        Verifies that various subquery patterns (IN, correlated, derived tables,
        scalar, EXISTS) are parsed successfully.

        Args:
            query: A syntactically valid query containing subqueries.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid subqueries"

    @pytest.mark.parametrize(
        "query",
        [
            "WITH temp AS (SELECT * FROM users) SELECT * FROM temp",
            (
                "WITH active_users AS (SELECT * FROM users WHERE active = true), "
                "recent_orders AS (SELECT * FROM orders WHERE date > '2024-01-01') "
                "SELECT * FROM active_users u JOIN recent_orders o ON u.id = o.user_id"
            ),
            (
                "WITH RECURSIVE tree AS ("
                "SELECT id, parent_id, name FROM categories WHERE parent_id IS NULL "
                "UNION ALL "
                "SELECT c.id, c.parent_id, c.name FROM categories c "
                "JOIN tree t ON c.parent_id = t.id"
                ") SELECT * FROM tree"
            ),
        ],
        ids=[
            "simple_cte",
            "multiple_ctes",
            "recursive_cte",
        ],
    )
    def test_parse_sql_valid_cte_returns_expression(self, query: str) -> None:
        """Valid CTEs (Common Table Expressions) should return a sqlglot Expression.

        Verifies that various CTE patterns (simple, multiple, recursive) are
        parsed successfully.

        Args:
            query: A syntactically valid query containing CTEs.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid CTEs"

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM t1 UNION SELECT * FROM t2",
            "SELECT * FROM t1 UNION ALL SELECT * FROM t2",
            "SELECT * FROM t1 INTERSECT SELECT * FROM t2",
            "SELECT * FROM t1 EXCEPT SELECT * FROM t2",
        ],
        ids=[
            "union",
            "union_all",
            "intersect",
            "except",
        ],
    )
    def test_parse_sql_valid_set_operations_returns_expression(self, query: str) -> None:
        """Valid set operations should return a sqlglot Expression.

        Verifies that UNION, UNION ALL, INTERSECT, and EXCEPT operations
        are parsed successfully.

        Args:
            query: A syntactically valid query with set operations.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "parse_sql should return Expression for valid set operations"

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT name, SUM(amount) OVER (PARTITION BY category) FROM sales",
            "SELECT name, ROW_NUMBER() OVER (ORDER BY created_at DESC) FROM users",
            (
                "SELECT name, amount, "
                "LAG(amount) OVER (PARTITION BY user_id ORDER BY date) AS prev_amount "
                "FROM transactions"
            ),
            ("SELECT *, RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank FROM employees"),
        ],
        ids=[
            "sum_over_partition",
            "row_number",
            "lag_function",
            "rank_function",
        ],
    )
    def test_parse_sql_valid_window_functions_returns_expression(self, query: str) -> None:
        """Valid window functions should return a sqlglot Expression.

        Verifies that various window function patterns are parsed successfully.

        Args:
            query: A syntactically valid query with window functions.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), (
                "parse_sql should return Expression for valid window functions"
            )


class TestParseSQLEmptyQueries:
    """Tests for parse_sql with empty or whitespace-only queries.

    These tests verify that empty and whitespace-only queries raise SQLSyntaxError
    with an empty errors list.
    """

    def test_parse_sql_empty_string_raises_sql_syntax_error(self) -> None:
        """Empty string should raise SQLSyntaxError.

        An empty query string is not valid SQL and should raise SQLSyntaxError
        with an empty errors list.
        """
        with pytest.raises(SQLSyntaxError):
            parse_sql("")

    def test_parse_sql_empty_string_has_empty_errors(self) -> None:
        """Empty string SQLSyntaxError should have empty errors list.

        When an empty query is provided, the SQLSyntaxError should contain
        an empty errors list since there are no parse errors to report.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql("")

        with check:
            assert exc_info.value.errors == [], "Empty query should have empty errors list"
        with check:
            assert not exc_info.value.query, "SQLSyntaxError should store the empty query"

    @pytest.mark.parametrize(
        "query",
        [
            " ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            "  \t\n  ",
            "\t\t\t",
            "\n\n\n",
        ],
        ids=[
            "single_space",
            "multiple_spaces",
            "tab",
            "newline",
            "crlf",
            "mixed_whitespace",
            "multiple_tabs",
            "multiple_newlines",
        ],
    )
    def test_parse_sql_whitespace_only_raises_sql_syntax_error(self, query: str) -> None:
        """Whitespace-only strings should raise SQLSyntaxError.

        Queries containing only whitespace characters (spaces, tabs, newlines)
        are not valid SQL and should raise SQLSyntaxError with an empty errors list.

        Args:
            query: A string containing only whitespace characters.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(query)

        with check:
            assert exc_info.value.errors == [], "Whitespace-only query should have empty errors list"


class TestParseSQLInvalidQueries:
    """Tests for parse_sql with invalid SQL syntax.

    These tests verify that syntactically incorrect SQL queries raise
    SQLSyntaxError with appropriate error information.

    Note: SQLglot is quite permissive and accepts many incomplete SQL statements.
    These tests use queries that are definitively rejected by SQLglot's parser.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM (SELECT a FROM t",  # unclosed parenthesis
            "SELECT * FROM t WHERE a = (",  # unclosed paren in expression
            "SELECT a FROM t WHERE a >",  # incomplete comparison operator
            "SELECT 1 + FROM t",  # invalid arithmetic expression
            "SELECT a FROM t JOIN",  # incomplete join (no table)
            "SELECT a FROM t WHERE IN (1)",  # missing left operand for IN
            "SELECT a FROM t WHERE a BETWEEN",  # incomplete BETWEEN
            "SELECT a FROM t ORDER BY ,",  # comma without column in ORDER BY
            "SELECT a FROM t WHERE a LIKE",  # incomplete LIKE
            "SELECT CASE WHEN FROM t",  # incomplete CASE expression
        ],
        ids=[
            "unclosed_parenthesis",
            "unclosed_paren_in_expression",
            "incomplete_comparison",
            "invalid_arithmetic",
            "incomplete_join",
            "missing_in_operand",
            "incomplete_between",
            "empty_order_by_column",
            "incomplete_like",
            "incomplete_case",
        ],
    )
    def test_parse_sql_invalid_query_raises_sql_syntax_error(self, query: str) -> None:
        """Invalid SQL syntax should raise SQLSyntaxError.

        Queries with syntax errors that SQLglot's parser definitively rejects
        should raise SQLSyntaxError with error details.

        Args:
            query: A syntactically invalid SQL query.
        """
        with pytest.raises(SQLSyntaxError):
            parse_sql(query)

    def test_parse_sql_unclosed_parenthesis_raises_sql_syntax_error(self) -> None:
        """Unclosed parenthesis should raise SQLSyntaxError.

        Mismatched parentheses are a common syntax error that should be
        detected and reported clearly.
        """
        with pytest.raises(SQLSyntaxError):
            parse_sql("SELECT * FROM (SELECT a FROM t")

    def test_parse_sql_invalid_expression_raises_sql_syntax_error(self) -> None:
        """Invalid expression with missing operand should raise SQLSyntaxError.

        Arithmetic or comparison expressions with missing operands should
        be caught by the parser.
        """
        with pytest.raises(SQLSyntaxError):
            parse_sql("SELECT 1 + FROM t")

    def test_parse_sql_incomplete_join_raises_sql_syntax_error(self) -> None:
        """Incomplete JOIN without table should raise SQLSyntaxError.

        A JOIN keyword without a following table name should be rejected.
        """
        with pytest.raises(SQLSyntaxError):
            parse_sql("SELECT a FROM t JOIN")

    def test_parse_sql_missing_in_operand_raises_sql_syntax_error(self) -> None:
        """IN clause without left operand should raise SQLSyntaxError.

        The IN operator requires a column or expression on its left side.
        """
        with pytest.raises(SQLSyntaxError):
            parse_sql("SELECT a FROM t WHERE IN (1, 2, 3)")


class TestParseSQLErrorDetails:
    """Tests for SQLSyntaxError attributes and error details.

    These tests verify that SQLSyntaxError contains the expected attributes
    including the original query and parse error details.
    """

    # Use a query that definitively causes a ParseError in SQLglot
    INVALID_QUERY = "SELECT * FROM (SELECT a FROM t"

    def test_parse_sql_error_contains_query(self) -> None:
        """SQLSyntaxError should contain the original query.

        The query attribute should store the exact query string that was
        validated, enabling debugging and error reporting.
        """
        invalid_query = self.INVALID_QUERY

        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(invalid_query)

        with check:
            assert exc_info.value.query == invalid_query, "SQLSyntaxError should store the original query"

    def test_parse_sql_error_contains_errors_list(self) -> None:
        """SQLSyntaxError should contain a non-empty errors list.

        The errors attribute should contain at least one ParseErrorDict
        describing what went wrong during parsing.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(self.INVALID_QUERY)

        with check:
            assert isinstance(exc_info.value.errors, list), "errors should be a list"
        with check:
            assert len(exc_info.value.errors) > 0, "errors list should not be empty for invalid SQL"

    def test_parse_sql_error_contains_description(self) -> None:
        """SQLSyntaxError errors should contain description field.

        Each error in the errors list should have a description field
        explaining what went wrong.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(self.INVALID_QUERY)

        error = exc_info.value.errors[0]
        with check:
            assert "description" in error, "Error dict should contain 'description' key"
        with check:
            assert isinstance(error["description"], str), "description should be a string"
        with check:
            assert len(error["description"]) > 0, "description should not be empty"

    def test_parse_sql_error_contains_location_info(self) -> None:
        """SQLSyntaxError errors should contain line and column information.

        Parse errors should include location information (line and col) to
        help identify where in the query the error occurred.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(self.INVALID_QUERY)

        error = exc_info.value.errors[0]
        with check:
            assert "line" in error, "Error dict should contain 'line' key"
        with check:
            assert "col" in error, "Error dict should contain 'col' key"
        with check:
            assert isinstance(error["line"], int), "line should be an integer"
        with check:
            assert isinstance(error["col"], int), "col should be an integer"

    def test_parse_sql_error_message_contains_details(self) -> None:
        """SQLSyntaxError message should include parse error details.

        The exception message should contain useful information about the
        parse error, not just a generic "syntax error" message.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(self.INVALID_QUERY)

        error_message = str(exc_info.value)
        with check:
            assert "SQL syntax error" in error_message, "Error message should indicate SQL syntax error"

    def test_parse_sql_error_context_fields(self) -> None:
        """SQLSyntaxError errors contain all ParseErrorDict fields.

        Parse errors include start_context, highlight, and end_context
        fields to show the problematic section of the query.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(self.INVALID_QUERY)

        error = exc_info.value.errors[0]
        # These fields should exist (even if empty) as per ParseErrorDict
        expected_keys = {"description", "line", "col", "start_context", "highlight", "end_context"}
        actual_keys = set(error.keys())

        with check:
            assert expected_keys == actual_keys, (
                f"Error dict should contain all ParseErrorDict keys. "
                f"Missing: {expected_keys - actual_keys}, Extra: {actual_keys - expected_keys}"
            )

    def test_parse_sql_error_has_meaningful_context(self) -> None:
        """SQLSyntaxError should provide context about where the error occurred.

        The start_context and highlight fields should contain text from
        the query to help locate the error.
        """
        with pytest.raises(SQLSyntaxError) as exc_info:
            parse_sql(self.INVALID_QUERY)

        error = exc_info.value.errors[0]
        # For unclosed parenthesis, start_context should contain query text
        with check:
            assert "start_context" in error, "Error should have start_context"
        with check:
            # The start_context should contain part of the query
            assert len(error.get("start_context", "")) > 0 or len(error.get("highlight", "")) > 0, (
                "At least one of start_context or highlight should have content"
            )


class TestParseSQLEdgeCases:
    """Tests for parse_sql edge cases.

    These tests verify proper handling of edge cases like queries with
    leading/trailing whitespace, comments, and special characters.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "  SELECT a FROM t",
            "SELECT a FROM t  ",
            "  SELECT a FROM t  ",
            "\nSELECT a FROM t\n",
            "\t\tSELECT a FROM t\t\t",
        ],
        ids=[
            "leading_spaces",
            "trailing_spaces",
            "both_spaces",
            "newlines",
            "tabs",
        ],
    )
    def test_parse_sql_query_with_whitespace_padding_returns_expression(self, query: str) -> None:
        """Valid queries with leading/trailing whitespace should return Expression.

        Whitespace around a valid SQL query should be ignored during parsing.

        Args:
            query: A valid SQL query with surrounding whitespace.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "Whitespace padding should not affect valid query parsing"

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT a FROM t -- this is a comment",
            "-- comment\nSELECT a FROM t",
            "SELECT /* inline comment */ a FROM t",
            "SELECT a /* comment */ FROM /* comment */ t",
        ],
        ids=[
            "trailing_line_comment",
            "leading_line_comment",
            "inline_block_comment",
            "multiple_block_comments",
        ],
    )
    def test_parse_sql_query_with_comments_returns_expression(self, query: str) -> None:
        """Valid queries with SQL comments should return Expression.

        SQL comments (both line and block) should be handled correctly
        by the parser.

        Args:
            query: A valid SQL query containing comments.
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "SQL comments should not affect valid query parsing"

    def test_parse_sql_multiline_query_returns_expression(self) -> None:
        """Valid multiline queries should return Expression.

        Queries formatted across multiple lines should be parsed correctly.
        """
        query = """
        SELECT
            a,
            b,
            c
        FROM
            table_name
        WHERE
            a > 1
            AND b < 10
        ORDER BY
            c DESC
        """
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "Multiline query formatting should not affect parsing"

    def test_parse_sql_case_insensitive_keywords(self) -> None:
        """SQL keywords in different cases should be accepted.

        SQL keywords are case-insensitive, so SELECT, select, and SeLeCt
        should all be valid.
        """
        queries = [
            "select a from t",
            "SELECT A FROM T",
            "SeLeCt A fRoM t",
        ]
        for query in queries:
            result = parse_sql(query)
            with check:
                assert isinstance(result, sqlglot.Expression), f"Case variation '{query}' should be valid"

    def test_parse_sql_special_characters_in_strings(self) -> None:
        """Queries with special characters in string literals should be valid.

        String literals may contain special characters that should not
        affect SQL parsing.
        """
        query = "SELECT * FROM users WHERE name = 'O''Brien' AND email LIKE '%@example.com'"
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "Special characters in strings should not affect parsing"

    def test_parse_sql_numeric_literals(self) -> None:
        """Queries with various numeric literal formats should be valid.

        Different numeric formats (integers, floats, scientific notation)
        should all be accepted.
        """
        query = "SELECT * FROM t WHERE a = 42 AND b = 3.14 AND c = 1.5e10 AND d = -99.9"
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "Numeric literals should be parsed correctly"

    def test_parse_sql_quoted_identifiers(self) -> None:
        """Queries with quoted identifiers should be valid.

        Table and column names may be quoted to allow reserved words or
        special characters as identifiers.
        """
        query = 'SELECT "select", "from" FROM "table" WHERE "order" = 1'
        result = parse_sql(query)

        with check:
            assert isinstance(result, sqlglot.Expression), "Quoted identifiers should be parsed correctly"


class TestGetSQLCommandType:
    """Tests for _get_sql_command_type helper function.

    These tests verify that sqlglot expression types are correctly mapped
    to their corresponding SQL command type strings.
    """

    def test_get_sql_command_type_select_returns_select(self) -> None:
        """SELECT query should return 'SELECT' command type.

        A simple SELECT statement parsed by sqlglot should map to the
        'SELECT' command type string.
        """
        expression = sqlglot.parse_one("SELECT a, b FROM t")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "SELECT", "SELECT query should return 'SELECT'"

    def test_get_sql_command_type_delete_returns_delete(self) -> None:
        """DELETE query should return 'DELETE' command type.

        A DELETE statement parsed by sqlglot should map to the
        'DELETE' command type string.
        """
        expression = sqlglot.parse_one("DELETE FROM users WHERE id = 1")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "DELETE", "DELETE query should return 'DELETE'"

    def test_get_sql_command_type_insert_returns_insert(self) -> None:
        """INSERT query should return 'INSERT' command type.

        An INSERT statement parsed by sqlglot should map to the
        'INSERT' command type string.
        """
        expression = sqlglot.parse_one("INSERT INTO users (name) VALUES ('John')")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "INSERT", "INSERT query should return 'INSERT'"

    def test_get_sql_command_type_update_returns_update(self) -> None:
        """UPDATE query should return 'UPDATE' command type.

        An UPDATE statement parsed by sqlglot should map to the
        'UPDATE' command type string.
        """
        expression = sqlglot.parse_one("UPDATE users SET name = 'Jane' WHERE id = 1")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "UPDATE", "UPDATE query should return 'UPDATE'"

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE users",
            "DROP INDEX idx_name",
        ],
        ids=[
            "drop_table",
            "drop_index",
        ],
    )
    def test_get_sql_command_type_drop_returns_drop(self, query: str) -> None:
        """DROP query should return 'DROP' command type.

        DROP TABLE and DROP INDEX statements parsed by sqlglot should
        both map to the 'DROP' command type string.

        Args:
            query: A DROP statement (TABLE or INDEX).
        """
        expression = sqlglot.parse_one(query)
        result = _get_sql_command_type(expression)

        with check:
            assert result == "DROP", f"'{query}' should return 'DROP'"

    def test_get_sql_command_type_create_returns_create(self) -> None:
        """CREATE query should return 'CREATE' command type.

        A CREATE TABLE statement parsed by sqlglot should map to the
        'CREATE' command type string.
        """
        expression = sqlglot.parse_one("CREATE TABLE users (id INT, name VARCHAR(100))")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "CREATE", "CREATE query should return 'CREATE'"

    def test_get_sql_command_type_truncate_returns_truncate(self) -> None:
        """TRUNCATE query should return 'TRUNCATE' command type.

        A TRUNCATE TABLE statement parsed by sqlglot should map to the
        'TRUNCATE' command type string.
        """
        expression = sqlglot.parse_one("TRUNCATE TABLE users")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "TRUNCATE", "TRUNCATE query should return 'TRUNCATE'"

    def test_get_sql_command_type_alter_returns_alter(self) -> None:
        """ALTER query should return 'ALTER' command type.

        An ALTER TABLE statement parsed by sqlglot should map to the
        'ALTER' command type string.
        """
        expression = sqlglot.parse_one("ALTER TABLE users ADD COLUMN email VARCHAR(255)")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "ALTER", "ALTER query should return 'ALTER'"

    def test_get_sql_command_type_union_returns_select(self) -> None:
        """UNION query should return 'SELECT' command type.

        A UNION set operation is considered a SELECT query and should
        map to the 'SELECT' command type string.
        """
        expression = sqlglot.parse_one("SELECT a FROM t1 UNION SELECT a FROM t2")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "SELECT", "UNION query should return 'SELECT'"

    def test_get_sql_command_type_intersect_returns_select(self) -> None:
        """INTERSECT query should return 'SELECT' command type.

        An INTERSECT set operation is considered a SELECT query and should
        map to the 'SELECT' command type string.
        """
        expression = sqlglot.parse_one("SELECT a FROM t1 INTERSECT SELECT a FROM t2")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "SELECT", "INTERSECT query should return 'SELECT'"

    def test_get_sql_command_type_except_returns_select(self) -> None:
        """EXCEPT query should return 'SELECT' command type.

        An EXCEPT set operation is considered a SELECT query and should
        map to the 'SELECT' command type string.
        """
        expression = sqlglot.parse_one("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
        result = _get_sql_command_type(expression)

        with check:
            assert result == "SELECT", "EXCEPT query should return 'SELECT'"

    def test_get_sql_command_type_unrecognized_returns_none(self) -> None:
        """Unrecognized expression type should return None.

        Expression types not in the mapping should return None to indicate
        the command type could not be determined.
        """
        # Create a mock expression type that won't be in the mapping
        # Using a SHOW statement which is not in _EXPRESSION_TYPE_MAP
        expression = sqlglot.parse_one("SHOW TABLES")
        result = _get_sql_command_type(expression)

        with check:
            assert result is None, "Unrecognized expression type should return None"


class TestParseSQLBlacklist:
    """Tests for parse_sql blacklist functionality.

    These tests verify the optional blacklist parameter that blocks specific
    SQL command types from being parsed successfully.
    """

    def test_parse_sql_without_blacklist_allows_destructive_commands(self) -> None:
        """DELETE without blacklist should succeed for backward compatibility.

        When no blacklist is provided, destructive commands like DELETE should
        be parsed successfully and return an Expression.
        """
        result = parse_sql("DELETE FROM t")

        with check:
            assert isinstance(result, sqlglot.Expression), "DELETE without blacklist should return Expression"

    def test_parse_sql_blacklist_allows_non_matching_commands(self) -> None:
        """SELECT with DELETE blacklisted should succeed.

        Commands not in the blacklist should be parsed successfully even when
        other commands are blacklisted.
        """
        result = parse_sql("SELECT * FROM t", blacklist={"DELETE"})

        with check:
            assert isinstance(result, sqlglot.Expression), "SELECT should succeed when DELETE is blacklisted"

    def test_parse_sql_blacklist_blocks_matching_commands(self) -> None:
        """DELETE with DELETE blacklisted should raise SQLBlacklistedCommandError.

        Commands that match entries in the blacklist should be blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("DELETE FROM t", blacklist={"DELETE"})

    def test_parse_sql_blacklist_case_insensitive_lowercase_blacklist(self) -> None:
        """DELETE with lowercase 'delete' blacklisted should raise.

        Blacklist matching should be case-insensitive, so lowercase blacklist
        entries should still block uppercase SQL commands.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("DELETE FROM t", blacklist={"delete"})

    def test_parse_sql_blacklist_case_insensitive_lowercase_query(self) -> None:
        """Lowercase 'delete' query with DELETE blacklisted should raise.

        Blacklist matching should be case-insensitive, so lowercase SQL commands
        should still be blocked by uppercase blacklist entries.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("delete from t", blacklist={"DELETE"})

    def test_parse_sql_blacklist_destructive_commands_allows_select(self) -> None:
        """SELECT with DESTRUCTIVE_COMMANDS blacklist should succeed.

        The DESTRUCTIVE_COMMANDS set should only block data-modifying
        commands, not read-only SELECT statements.
        """
        result = parse_sql("SELECT * FROM users", blacklist=DESTRUCTIVE_COMMANDS)

        with check:
            assert isinstance(result, sqlglot.Expression), "SELECT should succeed with DESTRUCTIVE_COMMANDS blacklist"

    def test_parse_sql_blacklist_destructive_commands_blocks_delete(self) -> None:
        """DELETE with DESTRUCTIVE_COMMANDS blacklist should raise.

        The DESTRUCTIVE_COMMANDS set includes DELETE, so it should be blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("DELETE FROM users WHERE id = 1", blacklist=DESTRUCTIVE_COMMANDS)

    def test_parse_sql_blacklist_error_attributes_set_correctly(self) -> None:
        """SQLBlacklistedCommandError attributes should be set correctly.

        The exception should have command_type and blacklist attributes
        populated with the correct values.
        """
        blacklist = {"DELETE", "DROP"}
        with pytest.raises(SQLBlacklistedCommandError) as exc_info:
            parse_sql("DELETE FROM t", blacklist=blacklist)

        with check:
            assert exc_info.value.command_type == "DELETE", "command_type should be 'DELETE'"
        with check:
            # Blacklist is normalized to uppercase
            assert exc_info.value.blacklist == {"DELETE", "DROP"}, (
                "blacklist should contain the normalized blacklisted commands"
            )
        with check:
            assert exc_info.value.query == "DELETE FROM t", "query should contain the original query string"

    def test_parse_sql_blacklist_insert_blocked(self) -> None:
        """INSERT with INSERT blacklisted should raise SQLBlacklistedCommandError.

        The INSERT command should be correctly identified and blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("INSERT INTO t VALUES (1)", blacklist={"INSERT"})

    def test_parse_sql_blacklist_update_blocked(self) -> None:
        """UPDATE with UPDATE blacklisted should raise SQLBlacklistedCommandError.

        The UPDATE command should be correctly identified and blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("UPDATE t SET a = 1", blacklist={"UPDATE"})

    def test_parse_sql_blacklist_drop_blocked(self) -> None:
        """DROP with DROP blacklisted should raise SQLBlacklistedCommandError.

        The DROP command should be correctly identified and blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("DROP TABLE t", blacklist={"DROP"})

    def test_parse_sql_blacklist_create_blocked(self) -> None:
        """CREATE with CREATE blacklisted should raise SQLBlacklistedCommandError.

        The CREATE command should be correctly identified and blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("CREATE TABLE t (id INT)", blacklist={"CREATE"})

    def test_parse_sql_blacklist_truncate_blocked(self) -> None:
        """TRUNCATE with TRUNCATE blacklisted should raise SQLBlacklistedCommandError.

        The TRUNCATE command should be correctly identified and blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("TRUNCATE TABLE t", blacklist={"TRUNCATE"})

    def test_parse_sql_blacklist_alter_blocked(self) -> None:
        """ALTER with ALTER blacklisted should raise SQLBlacklistedCommandError.

        The ALTER command should be correctly identified and blocked.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("ALTER TABLE t ADD COLUMN c INT", blacklist={"ALTER"})

    def test_parse_sql_blacklist_empty_set_allows_all(self) -> None:
        """DELETE with empty blacklist should succeed.

        An empty blacklist set should not block any commands.
        """
        result = parse_sql("DELETE FROM t", blacklist=set())

        with check:
            assert isinstance(result, sqlglot.Expression), "Empty blacklist should allow all commands"

    def test_parse_sql_blacklist_select_can_be_blocked(self) -> None:
        """SELECT with SELECT blacklisted should raise SQLBlacklistedCommandError.

        Even SELECT queries can be blacklisted if needed for specific use cases.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("SELECT * FROM t", blacklist={"SELECT"})

    def test_parse_sql_blacklist_union_queries_detected_as_select(self) -> None:
        """UNION query with SELECT blacklisted should raise.

        Set operations like UNION are treated as SELECT queries and should
        be blocked when SELECT is in the blacklist.
        """
        with pytest.raises(SQLBlacklistedCommandError):
            parse_sql("SELECT a FROM t1 UNION SELECT a FROM t2", blacklist={"SELECT"})

    def test_parse_sql_blacklist_error_inherits_from_sql_validation_error(self) -> None:
        """SQLBlacklistedCommandError should be catchable as SQLValidationError.

        The exception hierarchy allows catching all SQL validation errors
        with a single except clause.
        """
        with pytest.raises(SQLValidationError):
            parse_sql("DELETE FROM t", blacklist={"DELETE"})


class TestValidateSQLTables:
    """Tests for _validate_sql_tables private helper function.

    These tests verify that _validate_sql_tables correctly validates table
    references in SQL expressions against a set of allowed tables.
    """

    # -------------------------------------------------------------------------
    # Success Cases (should not raise)
    # -------------------------------------------------------------------------

    def test_validate_sql_tables_single_valid_table_succeeds(self) -> None:
        """Query referencing a single valid table should not raise.

        A simple SELECT query referencing a table that exists in the
        valid_tables set should pass validation without error.
        """
        # Arrange
        query = "SELECT price FROM products WHERE price > 0"
        valid_tables = {"products"}

        # Act & Assert - should not raise
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    def test_validate_sql_tables_multiple_valid_tables_join_succeeds(self) -> None:
        """Query joining multiple valid tables should not raise.

        A JOIN query referencing multiple tables that all exist in the
        valid_tables set should pass validation without error.
        """
        # Arrange
        query = "SELECT * FROM employees e JOIN departments d ON e.dept_id = d.id"
        valid_tables = {"employees", "departments"}

        # Act & Assert - should not raise
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    def test_validate_sql_tables_case_insensitive_query_uppercase_succeeds(self) -> None:
        """Query with uppercase table name should match lowercase valid_tables.

        Table name matching should be case-insensitive, so USERS in the
        query should match 'users' in valid_tables.
        """
        # Arrange
        query = "SELECT * FROM INVENTORY"
        valid_tables = {"inventory"}

        # Act & Assert - should not raise
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    def test_validate_sql_tables_case_insensitive_valid_tables_uppercase_succeeds(self) -> None:
        """Query with lowercase table name should match uppercase valid_tables.

        Table name matching should be case-insensitive, so transactions in the
        query should match 'TRANSACTIONS' in valid_tables.
        """
        # Arrange
        query = "SELECT * FROM transactions"
        valid_tables = {"TRANSACTIONS"}

        # Act & Assert - should not raise
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    def test_validate_sql_tables_with_table_alias_succeeds(self) -> None:
        """Query using table alias should validate the actual table name.

        Table aliases (e.g., 'u' for 'users') should not affect validation;
        the underlying table name should be checked against valid_tables.
        """
        # Arrange
        query = "SELECT p.name, p.price FROM products p"
        valid_tables = {"products"}

        # Act & Assert - should not raise
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    def test_validate_sql_tables_accepts_pre_parsed_expression_succeeds(self) -> None:
        """Pre-parsed sqlglot Expression should be accepted as expression parameter.

        The function accepts a pre-parsed sqlglot Expression from parse_sql().
        """
        # Arrange
        query = "SELECT COUNT(*) FROM orders WHERE status = 'shipped'"
        expression = parse_sql(query)
        valid_tables = {"orders"}

        # Act & Assert - should not raise
        _validate_sql_tables(expression, valid_tables, query)

    # -------------------------------------------------------------------------
    # Failure Cases (should raise SQLTableError)
    # -------------------------------------------------------------------------

    def test_validate_sql_tables_invalid_table_raises_error(self) -> None:
        """Query referencing only invalid tables should raise SQLTableError.

        When the query references a table not in valid_tables, SQLTableError
        should be raised with the invalid table name in the invalid_tables list.
        """
        # Arrange
        query = "SELECT * FROM nonexistent_data"
        valid_tables = {"products", "orders"}

        # Act & Assert
        with pytest.raises(SQLTableError) as exc_info:
            _validate_sql_tables(parse_sql(query), valid_tables, query)

        with check:
            assert "nonexistent_data" in exc_info.value.invalid_tables, (
                "invalid_tables should contain 'nonexistent_data'"
            )

    def test_validate_sql_tables_mix_valid_invalid_raises_error(self) -> None:
        """Query with mix of valid and invalid tables should raise SQLTableError.

        When some tables are valid but others are not, SQLTableError should
        be raised listing only the invalid tables.
        """
        # Arrange
        query = "SELECT * FROM customers JOIN ghost_table ON customers.id = ghost_table.cust_id"
        valid_tables = {"customers"}

        # Act & Assert
        with pytest.raises(SQLTableError) as exc_info:
            _validate_sql_tables(parse_sql(query), valid_tables, query)

        with check:
            assert exc_info.value.invalid_tables == ["ghost_table"], "invalid_tables should contain only 'ghost_table'"

    def test_validate_sql_tables_no_tables_raises_error(self) -> None:
        """Query without table references should raise SQLTableError.

        Queries like 'SELECT 1' that don't reference any tables should
        raise SQLTableError with an empty invalid_tables list.
        """
        # Arrange
        query = "SELECT 1"
        valid_tables = {"metrics"}

        # Act & Assert
        with pytest.raises(SQLTableError) as exc_info:
            _validate_sql_tables(parse_sql(query), valid_tables, query)

        with check:
            assert exc_info.value.invalid_tables == [], (
                "invalid_tables should be empty for queries with no table references"
            )

    def test_validate_sql_tables_all_invalid_tables_raises_error(self) -> None:
        """Query referencing only invalid tables should list all in error.

        When multiple tables are referenced and none are valid, all invalid
        table names should be included in the SQLTableError.
        """
        # Arrange
        query = "SELECT * FROM alpha JOIN beta ON alpha.id = beta.alpha_id"
        valid_tables = {"gamma", "delta"}

        # Act & Assert
        with pytest.raises(SQLTableError) as exc_info:
            _validate_sql_tables(parse_sql(query), valid_tables, query)

        with check:
            assert "alpha" in exc_info.value.invalid_tables, "invalid_tables should contain 'alpha'"
        with check:
            assert "beta" in exc_info.value.invalid_tables, "invalid_tables should contain 'beta'"

    # -------------------------------------------------------------------------
    # CTE and Subquery Tests
    # -------------------------------------------------------------------------

    def test_validate_sql_tables_cte_not_counted_as_table_succeeds(self) -> None:
        """CTE names should not be treated as database tables.

        Common Table Expressions (CTEs) define temporary result sets that
        should not be validated against valid_tables. Only the actual
        database tables referenced within the CTE should be validated.
        """
        # Arrange
        query = "WITH recent AS (SELECT * FROM events) SELECT * FROM recent"
        valid_tables = {"events"}

        # Act & Assert - should not raise because 'recent' is a CTE, not a table
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    def test_validate_sql_tables_subquery_table_validated_succeeds(self) -> None:
        """Tables in subqueries should be validated against valid_tables.

        Tables referenced inside subqueries should be included in the
        validation, not just tables in the main query.
        """
        # Arrange
        query = "SELECT * FROM products WHERE category_id IN (SELECT id FROM categories)"
        valid_tables = {"products", "categories"}

        # Act & Assert - should not raise
        _validate_sql_tables(parse_sql(query), valid_tables, query)

    # -------------------------------------------------------------------------
    # Error Attribute Tests
    # -------------------------------------------------------------------------

    def test_validate_sql_tables_error_contains_query(self) -> None:
        """SQLTableError should contain the original query string.

        The query attribute should store the exact query string that was
        validated, enabling debugging and error reporting.
        """
        # Arrange
        query = "SELECT * FROM phantom_records"
        valid_tables = {"accounts", "ledger"}

        # Act & Assert
        with pytest.raises(SQLTableError) as exc_info:
            _validate_sql_tables(parse_sql(query), valid_tables, query)

        with check:
            assert exc_info.value.query == query, "SQLTableError should store the original query"

    def test_validate_sql_tables_error_contains_invalid_tables_list(self) -> None:
        """SQLTableError invalid_tables attribute should be a list.

        The invalid_tables attribute should be a list type containing
        the names of tables that failed validation.
        """
        # Arrange
        query = "SELECT * FROM missing_data"
        valid_tables = {"reports"}

        # Act & Assert
        with pytest.raises(SQLTableError) as exc_info:
            _validate_sql_tables(parse_sql(query), valid_tables, query)

        with check:
            assert isinstance(exc_info.value.invalid_tables, list), "invalid_tables should be a list"


class TestValidateSQLColumnsValidCases:
    """Tests for _validate_sql_columns with valid queries.

    These tests verify that _validate_sql_columns correctly accepts expressions
    with valid column references and does not raise errors.
    """

    def test_validate_sql_columns_single_table_valid_columns_succeeds(self) -> None:
        """Query with valid columns from single table should not raise."""
        query = "SELECT id, name FROM users"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

    def test_validate_sql_columns_table_qualified_columns_succeeds(self) -> None:
        """Query with table-qualified columns should not raise."""
        query = "SELECT users.id, users.name FROM users"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

    def test_validate_sql_columns_table_alias_succeeds(self) -> None:
        """Query with table aliases should validate base table columns."""
        query = "SELECT u.id, u.name FROM users u"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

    def test_validate_sql_columns_join_multiple_tables_succeeds(self) -> None:
        """JOIN with columns from multiple base tables should not raise."""
        query = "SELECT u.id, u.name, o.order_date, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        table_columns = {"users": {"id", "name", "email"}, "orders": {"id", "user_id", "order_date", "total"}}
        _validate_sql_columns(parse_sql(query), table_columns, query)

    def test_validate_sql_columns_invalid_table_skipped(self) -> None:
        """Query with table not in provided schema should be skipped."""
        query = "SELECT id, name FROM nonexistent_table"
        table_columns = {"users": {"id", "name", "email"}}
        _validate_sql_columns(parse_sql(query), table_columns, query)

    @pytest.mark.parametrize(
        ("query", "table_columns"),
        [
            ("SELECT COUNT(id) FROM users", {"users": {"id", "name"}}),
            ("SELECT SUM(total) FROM orders", {"orders": {"id", "total", "user_id"}}),
            ("SELECT AVG(price), MAX(price), MIN(price) FROM products", {"products": {"id", "name", "price"}}),
        ],
        ids=["count_column", "sum_column", "multiple_aggregates"],
    )
    def test_validate_sql_columns_aggregate_functions_succeeds(
        self, query: str, table_columns: dict[str, set[str]]
    ) -> None:
        """Aggregate functions referencing valid columns should not raise."""
        _validate_sql_columns(parse_sql(query), table_columns, query)

    def test_validate_sql_columns_cte_columns_skipped_base_table_skipped(self) -> None:
        """CTE columns should be skipped; base table columns skipped."""
        query = (
            "WITH active_users AS (SELECT id, name FROM users WHERE active = true) SELECT id, name FROM active_users"
        )
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email", "active"}}, query)

    def test_validate_sql_columns_subquery_alias_columns_skipped(self) -> None:
        """Columns from subquery aliases should be skipped."""
        query = "SELECT sub.id, sub.name FROM (SELECT id, name FROM users WHERE active = true) AS sub"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email", "active"}}, query)

    def test_validate_sql_columns_self_join_different_aliases_succeeds(self) -> None:
        """Self-join with same table aliased differently should not raise."""
        query = (
            "SELECT e.id, e.name, m.name AS manager_name FROM employees e LEFT JOIN employees m ON e.manager_id = m.id"
        )
        _validate_sql_columns(parse_sql(query), {"employees": {"id", "name", "manager_id"}}, query)

    def test_validate_sql_columns_pre_parsed_expression_succeeds(self) -> None:
        """Pre-parsed Expression input should be accepted."""
        query = "SELECT id, name FROM users"
        expression = parse_sql(query)
        _validate_sql_columns(expression, {"users": {"id", "name", "email"}}, query)


class TestValidateSQLColumnsInvalidCases:
    """Tests for _validate_sql_columns with invalid queries.

    These tests verify that _validate_sql_columns raises SQLColumnError
    for expressions with invalid column references.
    """

    def test_validate_sql_columns_invalid_column_raises_error(self) -> None:
        """Query with invalid column should raise SQLColumnError."""
        query = "SELECT id, nonexistent_col FROM users"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

        with check:
            assert "users" in exc_info.value.invalid_columns
        with check:
            assert "nonexistent_col" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_columns_multiple_invalid_columns_different_tables_raises_error(self) -> None:
        """Query with multiple invalid columns across tables should list all."""
        query = "SELECT u.id, u.bad_col, o.order_date, o.fake_col FROM users u JOIN orders o ON u.id = o.user_id"
        table_columns = {"users": {"id", "name", "email"}, "orders": {"id", "user_id", "order_date", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), table_columns, query)

        with check:
            assert "bad_col" in exc_info.value.invalid_columns.get("users", [])
        with check:
            assert "fake_col" in exc_info.value.invalid_columns.get("orders", [])

    def test_validate_sql_columns_table_alias_invalid_column_shows_base_table(self) -> None:
        """Table alias with invalid column should report base table name."""
        query = "SELECT u.id, u.fake_column FROM users u"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

        with check:
            assert "users" in exc_info.value.invalid_columns
        with check:
            assert "fake_column" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_columns_invalid_column_in_aggregate_raises_error(self) -> None:
        """Invalid column inside aggregate function should raise SQLColumnError."""
        query = "SELECT COUNT(nonexistent) FROM users"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

        with check:
            assert "nonexistent" in exc_info.value.invalid_columns.get("users", [])


class TestValidateSQLColumnsErrorDetails:
    """Tests for SQLColumnError error message formatting.

    These tests verify that SQLColumnError provides detailed and useful
    error information including table context and available columns.
    """

    def test_validate_sql_columns_error_includes_table_name(self) -> None:
        """Error should include which table the invalid column was from."""
        query = "SELECT bad_column FROM users"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

        with check:
            assert "users" in exc_info.value.format_details()

    def test_validate_sql_columns_error_includes_available_columns(self) -> None:
        """Error format_details should include available columns for the table."""
        query = "SELECT bad_column FROM users"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

        error_details = exc_info.value.format_details()
        with check:
            assert "id" in error_details
        with check:
            assert "name" in error_details
        with check:
            assert "email" in error_details

    def test_validate_sql_columns_multiple_invalid_columns_listed_in_error(self) -> None:
        """Multiple invalid columns should produce multiple entries."""
        query = "SELECT id, col_a, col_b FROM users"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

        invalid_cols = exc_info.value.invalid_columns.get("users", [])
        with check:
            assert "col_a" in invalid_cols
        with check:
            assert "col_b" in invalid_cols

    def test_validate_sql_columns_error_contains_query(self) -> None:
        """SQLColumnError should contain the original query string."""
        query = "SELECT bad_col FROM users"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

        with check:
            assert exc_info.value.query == query

    def test_validate_sql_columns_error_contains_table_columns_schema(self) -> None:
        """SQLColumnError should contain the table_columns schema."""
        query = "SELECT bad_col FROM users"
        table_columns = {"users": {"id", "name"}}
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), table_columns, query)

        with check:
            assert exc_info.value.table_columns == table_columns


class TestValidateSQLColumnsCaseInsensitivity:
    """Tests for case-insensitive matching in _validate_sql_columns.

    These tests verify that table and column names are matched
    case-insensitively.
    """

    def test_validate_sql_columns_column_names_case_insensitive_succeeds(self) -> None:
        """Column names should be matched case-insensitively."""
        query = "SELECT ID, NAME FROM users"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

    def test_validate_sql_columns_table_names_case_insensitive_succeeds(self) -> None:
        """Table names should be matched case-insensitively."""
        query = "SELECT id, name FROM USERS"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

    def test_validate_sql_columns_mixed_case_matching_succeeds(self) -> None:
        """Mixed case in both query and schema should match."""
        query = "SELECT Id, NaMe FROM Users"
        _validate_sql_columns(parse_sql(query), {"USERS": {"ID", "NAME", "EMAIL"}}, query)


class TestValidateSQLColumnsEdgeCases:
    """Tests for edge cases in _validate_sql_columns.

    These tests cover edge cases like SELECT *, various SQL clauses,
    and ambiguous column references.
    """

    def test_validate_sql_columns_star_select_succeeds(self) -> None:
        """SELECT * should not raise errors since no specific columns are referenced."""
        query = "SELECT * FROM users"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

    def test_validate_sql_columns_where_clause_columns_succeeds(self) -> None:
        """Columns in WHERE clause should be validated."""
        query = "SELECT id FROM users WHERE bad_filter = 1"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name", "email"}}, query)

        with check:
            assert "bad_filter" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_columns_order_by_columns_succeeds(self) -> None:
        """Columns in ORDER BY clause should be validated."""
        query = "SELECT id FROM users ORDER BY fake_column"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

        with check:
            assert "fake_column" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_columns_group_by_columns_succeeds(self) -> None:
        """Columns in GROUP BY clause should be validated."""
        query = "SELECT category, COUNT(*) FROM products GROUP BY fake_category"
        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(
                parse_sql(query),
                {"products": {"id", "name", "category"}},
                query,
            )

        with check:
            assert "fake_category" in exc_info.value.invalid_columns.get("products", [])

    def test_validate_sql_columns_join_condition_columns_succeeds(self) -> None:
        """Columns in JOIN ON conditions should be validated."""
        query = "SELECT u.id FROM users u JOIN orders o ON u.fake_join_col = o.user_id"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "user_id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), table_columns, query)

        with check:
            assert "fake_join_col" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_columns_empty_schema_for_unknown_table_skips_validation(self) -> None:
        """Columns for tables not in schema should be skipped."""
        query = "SELECT u.id, ext.data FROM users u JOIN external ext ON u.id = ext.user_id"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

    def test_validate_sql_columns_no_scope_returns_without_error(self) -> None:
        """Query without scope should not raise."""
        query = "SELECT 1"
        _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)

    def test_validate_sql_columns_unqualified_column_multiple_tables_ambiguous(self) -> None:
        """Unqualified column existing in multiple tables should raise SQLColumnError."""
        query = "SELECT id FROM users u JOIN orders o ON u.id = o.user_id"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "user_id"}}

        with pytest.raises(SQLColumnError) as exc_info:
            _validate_sql_columns(parse_sql(query), table_columns, query)

        with check:
            assert "id" in exc_info.value.ambiguous_columns, "Should detect 'id' as ambiguous"

    def test_validate_sql_columns_inherits_from_sql_validation_error(self) -> None:
        """SQLColumnError should be catchable as SQLValidationError."""
        query = "SELECT bad_col FROM users"
        with pytest.raises(SQLValidationError):
            _validate_sql_columns(parse_sql(query), {"users": {"id", "name"}}, query)


# =============================================================================
# Tests for validate_sql function
# =============================================================================


class TestValidateSQLSuccessCases:
    """Tests for validate_sql with valid queries.

    These tests verify that validate_sql correctly accepts queries that pass
    all validation steps without raising errors.
    """

    def test_validate_sql_simple_query_succeeds(self) -> None:
        """Valid simple query with matching schema should not raise."""
        validate_sql("SELECT id FROM users", {"users": {"id", "name", "email"}})

    def test_validate_sql_multiple_columns_succeeds(self) -> None:
        """Valid query selecting multiple columns should not raise."""
        validate_sql("SELECT id, name, email FROM users", {"users": {"id", "name", "email"}})

    def test_validate_sql_join_query_succeeds(self) -> None:
        """Valid JOIN query with all tables and columns in schema should not raise."""
        query = "SELECT u.id, u.name, o.order_date FROM users u JOIN orders o ON u.id = o.user_id"
        table_columns = {"users": {"id", "name", "email"}, "orders": {"id", "user_id", "order_date", "total"}}
        validate_sql(query, table_columns)

    def test_validate_sql_only_accepts_string_query(self) -> None:
        """validate_sql should only accept string queries."""
        # String query should work
        validate_sql("SELECT id, name FROM users", {"users": {"id", "name", "email"}})

    def test_validate_sql_with_table_alias_succeeds(self) -> None:
        """Query with table aliases should validate correctly."""
        validate_sql("SELECT u.id, u.name FROM users u", {"users": {"id", "name", "email"}})

    def test_validate_sql_with_dialect_succeeds(self) -> None:
        """Valid query with dialect parameter should not raise."""
        validate_sql("SELECT id, name FROM users", {"users": {"id", "name"}}, dialect="duckdb")

    def test_validate_sql_with_where_clause_succeeds(self) -> None:
        """Valid query with WHERE clause using valid columns should not raise."""
        validate_sql("SELECT id, name FROM users WHERE id > 10", {"users": {"id", "name", "email"}})

    def test_validate_sql_with_order_by_succeeds(self) -> None:
        """Valid query with ORDER BY using valid columns should not raise."""
        validate_sql("SELECT id, name FROM users ORDER BY name", {"users": {"id", "name", "email"}})

    def test_validate_sql_with_aggregate_succeeds(self) -> None:
        """Valid query with aggregate functions should not raise."""
        validate_sql("SELECT COUNT(id) FROM users", {"users": {"id", "name", "email"}})

    def test_validate_sql_complex_join_succeeds(self) -> None:
        """Complex multi-table JOIN should validate correctly."""
        query = """
            SELECT u.id, u.name, o.order_date, p.product_name
            FROM users u
            JOIN orders o ON u.id = o.user_id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE u.id > 0
            ORDER BY o.order_date DESC
        """
        table_columns = {
            "users": {"id", "name", "email"},
            "orders": {"id", "user_id", "order_date"},
            "order_items": {"id", "order_id", "product_id"},
            "products": {"id", "product_name", "price"},
        }
        validate_sql(query, table_columns)


class TestValidateSQLSyntaxErrors:
    """Tests for validate_sql with syntax errors.

    These tests verify that parse_sql errors propagate correctly through validate_sql.
    """

    def test_validate_sql_invalid_syntax_raises_sql_syntax_error(self) -> None:
        """Query with invalid syntax should raise SQLSyntaxError."""
        with pytest.raises(SQLSyntaxError):
            validate_sql("SELECT * FROM (SELECT a FROM t", {"users": {"id", "name"}})

    def test_validate_sql_empty_query_raises_sql_syntax_error(self) -> None:
        """Empty string query should raise SQLSyntaxError."""
        with pytest.raises(SQLSyntaxError):
            validate_sql("", {"users": {"id", "name"}})

    def test_validate_sql_whitespace_query_raises_sql_syntax_error(self) -> None:
        """Whitespace-only query should raise SQLSyntaxError."""
        with pytest.raises(SQLSyntaxError):
            validate_sql("   ", {"users": {"id", "name"}})

    def test_validate_sql_blacklisted_command_raises_error(self) -> None:
        """DELETE query with DESTRUCTIVE_COMMANDS blacklist should raise SQLBlacklistedCommandError."""
        with pytest.raises(SQLBlacklistedCommandError) as exc_info:
            validate_sql("DELETE FROM users WHERE id = 1", {"users": {"id", "name"}}, blacklist=DESTRUCTIVE_COMMANDS)

        with check:
            assert exc_info.value.command_type == "DELETE"

    def test_validate_sql_blacklisted_drop_raises_error(self) -> None:
        """DROP query with DESTRUCTIVE_COMMANDS blacklist should raise SQLBlacklistedCommandError."""
        with pytest.raises(SQLBlacklistedCommandError) as exc_info:
            validate_sql("DROP TABLE users", {"users": {"id", "name"}}, blacklist=DESTRUCTIVE_COMMANDS)

        with check:
            assert exc_info.value.command_type == "DROP"


class TestValidateSQLTableErrors:
    """Tests for validate_sql with table errors.

    These tests verify that validate_sql_tables errors propagate correctly.
    """

    def test_validate_sql_invalid_table_raises_sql_table_error(self) -> None:
        """Query referencing table not in schema should raise SQLTableError."""
        with pytest.raises(SQLTableError) as exc_info:
            validate_sql("SELECT id FROM unknown_table", {"users": {"id", "name"}})

        with check:
            assert "unknown_table" in exc_info.value.invalid_tables

    def test_validate_sql_multiple_invalid_tables_raises_sql_table_error(self) -> None:
        """Query referencing multiple invalid tables should list all."""
        with pytest.raises(SQLTableError) as exc_info:
            validate_sql(
                "SELECT a.id, b.name FROM table_a a JOIN table_b b ON a.id = b.a_id", {"users": {"id", "name"}}
            )

        with check:
            assert "table_a" in exc_info.value.invalid_tables
        with check:
            assert "table_b" in exc_info.value.invalid_tables

    def test_validate_sql_no_tables_raises_sql_table_error(self) -> None:
        """Query without table references should raise SQLTableError."""
        with pytest.raises(SQLTableError):
            validate_sql("SELECT 1", {"users": {"id", "name"}})

    def test_validate_sql_table_error_contains_query(self) -> None:
        """SQLTableError should contain the original query string."""
        query = "SELECT id FROM bad_table"
        with pytest.raises(SQLTableError) as exc_info:
            validate_sql(query, {"users": {"id", "name"}})

        with check:
            assert exc_info.value.query == query


class TestValidateSQLColumnErrors:
    """Tests for validate_sql with column errors.

    These tests verify that validate_sql_columns errors propagate correctly.
    """

    def test_validate_sql_invalid_column_raises_sql_column_error(self) -> None:
        """Query referencing invalid column should raise SQLColumnError."""
        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql("SELECT id, bad_col FROM users", {"users": {"id", "name"}})

        with check:
            assert "bad_col" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_multiple_invalid_columns_raises_sql_column_error(self) -> None:
        """Query with multiple invalid columns should list all."""
        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql("SELECT id, col_a, col_b FROM users", {"users": {"id", "name"}})

        invalid_cols = exc_info.value.invalid_columns.get("users", [])
        with check:
            assert "col_a" in invalid_cols
        with check:
            assert "col_b" in invalid_cols

    def test_validate_sql_invalid_column_in_where_raises_error(self) -> None:
        """Invalid column in WHERE clause should raise SQLColumnError."""
        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql("SELECT id FROM users WHERE bad_filter = 1", {"users": {"id", "name"}})

        with check:
            assert "bad_filter" in exc_info.value.invalid_columns.get("users", [])

    def test_validate_sql_invalid_column_different_tables_raises_error(self) -> None:
        """Invalid columns across multiple tables should all be reported."""
        query = "SELECT u.id, u.bad_col, o.fake_col FROM users u JOIN orders o ON u.id = o.user_id"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "user_id", "order_date"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        with check:
            assert "bad_col" in exc_info.value.invalid_columns.get("users", [])
        with check:
            assert "fake_col" in exc_info.value.invalid_columns.get("orders", [])

    def test_validate_sql_column_error_contains_schema(self) -> None:
        """SQLColumnError should contain the table_columns schema."""
        table_columns = {"users": {"id", "name"}}
        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql("SELECT bad_col FROM users", table_columns)

        with check:
            assert exc_info.value.table_columns == table_columns


class TestValidateSQLAmbiguousColumns:
    """Tests for validate_sql with ambiguous column references.

    These tests verify that ambiguous column references (unqualified columns
    that exist in multiple tables) are properly detected and reported.
    """

    def test_validate_sql_ambiguous_column_raises_sql_column_error(self) -> None:
        """Ambiguous column reference should raise SQLColumnError."""
        query = "SELECT id FROM users, orders"  # 'id' exists in both tables
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        with check:
            assert "id" in exc_info.value.ambiguous_columns, "Should report 'id' as ambiguous"

    def test_validate_sql_ambiguous_column_lists_tables(self) -> None:
        """Ambiguous column error should list all tables where column exists."""
        query = "SELECT id FROM users u, orders o"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        tables = exc_info.value.ambiguous_columns.get("id", [])
        with check:
            assert "users" in tables, "Should list 'users' table"
        with check:
            assert "orders" in tables, "Should list 'orders' table"

    def test_validate_sql_ambiguous_column_contains_query(self) -> None:
        """SQLColumnError for ambiguous columns should contain the query."""
        query = "SELECT id FROM users, orders"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        with check:
            assert exc_info.value.query == query

    def test_validate_sql_ambiguous_column_format_details(self) -> None:
        """format_details should provide actionable feedback for ambiguous columns."""
        query = "SELECT id FROM users, orders"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        details = exc_info.value.format_details()
        with check:
            assert "ambiguous" in details.lower(), "Should mention ambiguous"
        with check:
            assert "users" in details and "orders" in details, "Should list tables"

    def test_validate_sql_qualified_column_not_ambiguous(self) -> None:
        """Qualified column reference should not be considered ambiguous."""
        query = "SELECT u.id, o.id FROM users u, orders o"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        # Should not raise - columns are properly qualified
        validate_sql(query, table_columns)

    def test_validate_sql_unambiguous_column_single_table_match(self) -> None:
        """Column existing in only one table should not be ambiguous."""
        query = "SELECT name FROM users, orders"  # 'name' only in users
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        # Should not raise - 'name' only exists in 'users'
        validate_sql(query, table_columns)

    def test_validate_sql_column_not_found_in_any_table_raises_error(self) -> None:
        """Unqualified column not found in any table should raise SQLColumnError."""
        query = "SELECT nonexistent FROM users, orders"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        with check:
            assert "nonexistent" in exc_info.value.not_found_columns, "Should report 'nonexistent' as not found"
        with check:
            searched_tables = exc_info.value.not_found_columns.get("nonexistent", [])
            assert "users" in searched_tables and "orders" in searched_tables, "Should list searched tables"

    def test_validate_sql_column_not_found_format_details(self) -> None:
        """format_details should provide actionable feedback for not-found columns."""
        query = "SELECT nonexistent FROM users, orders"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "total"}}

        with pytest.raises(SQLColumnError) as exc_info:
            validate_sql(query, table_columns)

        details = exc_info.value.format_details()
        with check:
            assert "not found in any table" in details.lower(), "Should mention not found"
        with check:
            assert "users" in details and "orders" in details, "Should list searched tables"


class TestValidateSQLErrorPrecedence:
    """Tests verifying error precedence in validate_sql.

    These tests verify that errors are raised in the expected order:
    syntax -> blacklist -> table -> column -> qualify.
    """

    def test_validate_sql_syntax_error_before_table_error(self) -> None:
        """Syntax errors should be raised before table errors."""
        # This query has both syntax error (unclosed paren) and would have table error
        with pytest.raises(SQLSyntaxError):
            validate_sql("SELECT * FROM (SELECT a FROM bad_table", {"users": {"id"}})

    def test_validate_sql_blacklist_error_before_table_error(self) -> None:
        """Blacklist errors should be raised before table errors."""
        # DELETE is blacklisted, and the table doesn't exist
        with pytest.raises(SQLBlacklistedCommandError):
            validate_sql("DELETE FROM nonexistent_table", {"users": {"id"}}, blacklist={"DELETE"})

    def test_validate_sql_table_error_before_column_error(self) -> None:
        """Table errors should be raised before column errors."""
        # Table doesn't exist, so column validation shouldn't run
        with pytest.raises(SQLTableError):
            validate_sql("SELECT bad_col FROM nonexistent_table", {"users": {"id", "name"}})


class TestValidateSQLEdgeCases:
    """Tests for edge cases in validate_sql."""

    def test_validate_sql_empty_blacklist_allows_all(self) -> None:
        """Empty blacklist should not block any commands."""
        validate_sql("SELECT id FROM users", {"users": {"id", "name"}}, blacklist=set())

    def test_validate_sql_none_blacklist_allows_all(self) -> None:
        """None blacklist should not block any commands."""
        validate_sql("SELECT id FROM users", {"users": {"id", "name"}}, blacklist=None)

    def test_validate_sql_case_insensitive_table_names(self) -> None:
        """Table names should be matched case-insensitively."""
        validate_sql("SELECT id FROM USERS", {"users": {"id", "name"}})

    def test_validate_sql_case_insensitive_column_names(self) -> None:
        """Column names should be matched case-insensitively."""
        validate_sql("SELECT ID, NAME FROM users", {"users": {"id", "name"}})

    def test_validate_sql_with_cte_succeeds(self) -> None:
        """Query with CTE should validate correctly."""
        query = "WITH active AS (SELECT id, name FROM users WHERE active = true) SELECT id, name FROM active"
        validate_sql(query, {"users": {"id", "name", "active"}})

    def test_validate_sql_with_subquery_succeeds(self) -> None:
        """Query with subquery should validate correctly."""
        query = "SELECT id, name FROM users WHERE id IN (SELECT user_id FROM orders)"
        table_columns = {"users": {"id", "name"}, "orders": {"id", "user_id"}}
        validate_sql(query, table_columns)

    def test_validate_sql_star_select_succeeds(self) -> None:
        """SELECT * should not raise errors."""
        validate_sql("SELECT * FROM users", {"users": {"id", "name", "email"}})
