"""Tests for DataFrameToolkit tools.

Covers get_tools, get_dataframe_id, list_dataframes, execute_sql, and view_as_markdown_table.
"""

from __future__ import annotations

import polars as pl
import pytest
from pytest_check import check

from dfkit.models import DataFrameReference, ToolCallError
from dfkit.toolkit import DataFrameToolkit


class TestGetTools:
    """Tests for DataFrameToolkit.get_tools and get_core_tools methods."""

    def test_get_tools_returns_list(self) -> None:
        """Given toolkit, When get_tools called, Then returns list of StructuredTool."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_tools()

        # Assert
        with check:
            assert isinstance(tools, list)
        with check:
            assert len(tools) >= 1

    def test_get_core_tools_returns_list(self) -> None:
        """Given toolkit, When get_core_tools called, Then returns list of StructuredTool."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_core_tools()

        # Assert
        with check:
            assert isinstance(tools, list)
        with check:
            assert len(tools) >= 1

    def test_get_tools_contains_core_tools(self) -> None:
        """Given toolkit, When get_tools called, Then contains all core tools."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        all_tools = toolkit.get_tools()
        core_tools = toolkit.get_core_tools()

        # Assert - all core tools should be in get_tools()
        all_tool_names = {t.name for t in all_tools}
        core_tool_names = {t.name for t in core_tools}
        with check:
            assert core_tool_names.issubset(all_tool_names)

    def test_tool_schema_excludes_self(self) -> None:
        """Given toolkit, When tool created, Then schema does not include 'self' parameter."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_core_tools()
        tool_get_dataframe_id = next(t for t in tools if t.name == "get_dataframe_id")
        tool_schema = tool_get_dataframe_id.args_schema.model_json_schema()

        # Assert - 'self' should not be in the properties
        with check:
            assert "self" not in tool_schema.get("properties", {})
        with check:
            assert "name" in tool_schema.get("properties", {})


class TestGetDataFrameId:
    """Tests for get_dataframe_id and get_dataframe_reference via the LangChain tool interface.

    Direct method tests live in test_registration.py (TestGetDataFrameId, TestGetDataFrameReference).
    """

    def test_get_dataframe_id_tool_invoke(self) -> None:
        """Given toolkit with registered DataFrame, When get_dataframe_id tool invoked, Then returns correct ID."""
        # Arrange
        toolkit = DataFrameToolkit()
        reference = toolkit.register_dataframe("sales", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_core_tools()
        tool_get_dataframe_id = next(t for t in tools if t.name == "get_dataframe_id")
        result = tool_get_dataframe_id.invoke({"name": "sales"})

        # Assert
        with check:
            assert result == reference.id

    def test_get_dataframe_reference_tool_invoke(self) -> None:
        """Given toolkit with registered DataFrame, When get_dataframe_reference tool invoked, Returns reference."""
        # Arrange
        toolkit = DataFrameToolkit()
        expected_reference = toolkit.register_dataframe(
            "sales",
            pl.DataFrame({"amount": [100, 200, 300]}),
        )

        # Act
        tools = toolkit.get_core_tools()
        tool_get_reference = next(t for t in tools if t.name == "get_dataframe_reference")
        result = tool_get_reference.invoke({"identifier": "sales"})

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result == expected_reference
        with check:
            assert result.name == "sales"


class TestListDataFrames:
    """Tests for DataFrameToolkit.list_dataframes method."""

    def test_list_empty_toolkit_returns_empty_list(self) -> None:
        """Given empty toolkit, When list_dataframes called, Then returns empty list."""
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        result = toolkit.list_dataframes()

        # Assert
        with check:
            assert isinstance(result, list)
        with check:
            assert len(result) == 0

    def test_list_with_registered_dataframes_returns_all_references(self) -> None:
        """Given toolkit with registered DataFrames, When list_dataframes called, Then returns all references."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("users", pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}))
        toolkit.register_dataframe("orders", pl.DataFrame({"order_id": [10, 20], "user_id": [1, 2]}))

        # Act
        result = toolkit.list_dataframes()

        # Assert
        with check:
            assert isinstance(result, list)
        with check:
            assert len(result) == 2
        names = {ref.name for ref in result}
        with check:
            assert names == {"users", "orders"}

    def test_list_references_contain_schema_info(self) -> None:
        """Given registered DataFrames, When list_dataframes called, Then references have column info."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe(
            "products",
            pl.DataFrame({"sku": ["A1", "B2"], "price": [9.99, 19.99]}),
            description="Product catalog",
        )

        # Act
        result = toolkit.list_dataframes()

        # Assert
        ref = result[0]
        with check:
            assert ref.name == "products"
        with check:
            assert ref.description == "Product catalog"
        with check:
            assert ref.column_names == ["sku", "price"]
        with check:
            assert ref.num_rows == 2
        with check:
            assert ref.num_columns == 2
        with check:
            assert "sku" in ref.column_summaries
        with check:
            assert "price" in ref.column_summaries

    def test_list_returns_all_types_including_derivatives(self) -> None:
        """Given toolkit with base and derived DataFrames, When list_dataframes called, Then returns all."""
        # Arrange
        toolkit = DataFrameToolkit()
        base_ref = toolkit.register_dataframe("base", pl.DataFrame({"x": [1, 2, 3]}))

        # Create derivative via public API
        derived_ref = toolkit.execute_sql(
            query=f"SELECT x FROM {base_ref.id} WHERE x < 3",  # noqa: S608
            result_name="derived",
        )
        assert isinstance(derived_ref, DataFrameReference)

        # Act
        result = toolkit.list_dataframes()

        # Assert
        with check:
            assert len(result) == 2
        names = {ref.name for ref in result}
        with check:
            assert names == {"base", "derived"}

    def test_list_dataframes_tool_invoke(self) -> None:
        """Given toolkit with registered DataFrames, When list_dataframes tool invoked, Then returns list."""
        # Arrange
        toolkit = DataFrameToolkit()
        toolkit.register_dataframe("users", pl.DataFrame({"id": [1, 2]}))
        toolkit.register_dataframe("orders", pl.DataFrame({"order_id": [10]}))

        # Act
        tools = toolkit.get_core_tools()
        tool_list_dataframes = next(t for t in tools if t.name == "list_dataframes")
        result = tool_list_dataframes.invoke({})

        # Assert
        with check:
            assert isinstance(result, list)
        with check:
            assert len(result) == 2
        names = {ref.name for ref in result}
        with check:
            assert names == {"users", "orders"}


class TestExecuteSQL:
    """Tests for DataFrameToolkit.execute_sql method."""

    @pytest.fixture
    def toolkit(self) -> DataFrameToolkit:
        """Create a toolkit instance for testing.

        Returns:
            DataFrameToolkit: Fresh toolkit instance with no DataFrames registered.
        """
        return DataFrameToolkit()

    def test_execute_success(self, toolkit: DataFrameToolkit) -> None:
        """Given valid query, When called, Then returns DataFrameReference with correct metadata.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "amount": [100, 200, 150, 300, 250]})
        ref = toolkit.register_dataframe("sales", df)
        query = f"SELECT * FROM {ref.id} WHERE amount > 150"  # noqa: S608

        # Act
        result = toolkit.execute_sql(
            query=query,
            result_name="high_sales",
            result_description="Sales over $150",
        )

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.name == "high_sales"
        with check:
            assert result.description == "Sales over $150"
        with check:
            assert result.num_rows == 3  # IDs 2, 4, 5 have amount > 150
        with check:
            assert result.column_names == ["id", "amount"]

    def test_execute_registers_result(self, toolkit: DataFrameToolkit) -> None:
        """Given valid query, When called, Then result accessible via list_dataframes.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"value": [10, 20, 30]})
        ref = toolkit.register_dataframe("data", df)
        query = f"SELECT value * 2 AS doubled FROM {ref.id}"  # noqa: S608

        # Act
        result = toolkit.execute_sql(query=query, result_name="doubled_data")

        # Assert - verify result is registered
        with check:
            assert isinstance(result, DataFrameReference)

        all_refs = toolkit.list_dataframes()
        names = {r.name for r in all_refs}
        with check:
            assert "doubled_data" in names
        with check:
            assert len(all_refs) == 2  # Original + result

    def test_execute_tracks_parent_ids(self, toolkit: DataFrameToolkit) -> None:
        """Given query referencing tables, When called, Then parent_ids set correctly.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df1 = pl.DataFrame({"id": [1, 2], "value": [100, 200]})
        df2 = pl.DataFrame({"id": [1, 2], "category": ["A", "B"]})
        ref1 = toolkit.register_dataframe("table1", df1)
        ref2 = toolkit.register_dataframe("table2", df2)

        # Act - query references both tables via JOIN
        query = f"""
            SELECT t1.id, t1.value, t2.category
            FROM {ref1.id} AS t1
            JOIN {ref2.id} AS t2 ON t1.id = t2.id
        """  # noqa: S608
        result = toolkit.execute_sql(query=query, result_name="joined_result")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.parent_ids is not None
        with check:
            assert set(result.parent_ids) == {ref1.id, ref2.id}

    def test_execute_sets_source_query(self, toolkit: DataFrameToolkit) -> None:
        """Given valid query, When called, Then result has source_query set.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        ref = toolkit.register_dataframe("source", df)
        query = f"SELECT x, y FROM {ref.id} WHERE x > 1"  # noqa: S608

        # Act
        result = toolkit.execute_sql(query=query, result_name="filtered")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.source_query == query
        with check:
            assert result.source_query is not None

    def test_execute_duplicate_name_returns_error(self, toolkit: DataFrameToolkit) -> None:
        """Given existing result name, When called, Then ToolCallError with DuplicateName.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3]})
        ref = toolkit.register_dataframe("data", df)
        toolkit.register_dataframe("existing_result", pl.DataFrame({"b": [4, 5, 6]}))

        query = f"SELECT * FROM {ref.id}"  # noqa: S608

        # Act - try to create result with existing name
        result = toolkit.execute_sql(query=query, result_name="existing_result")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "DuplicateName"
        with check:
            assert "existing_result" in result.message
        with check:
            assert "already registered" in result.message.lower()

    def test_execute_syntax_error_returns_error(self, toolkit: DataFrameToolkit) -> None:
        """Given invalid SQL, When called, Then ToolCallError with SQLSyntaxError.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2, 3]})
        ref = toolkit.register_dataframe("data", df)

        # Act - intentionally malformed SQL (missing closing parenthesis)
        bad_query = f"SELECT * FROM (SELECT id FROM {ref.id}"  # noqa: S608
        result = toolkit.execute_sql(query=bad_query, result_name="broken")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "SQLSyntaxError"
        with check:
            assert "errors" in result.details
        with check:
            assert "query" in result.details

    def test_execute_table_error_returns_error(self, toolkit: DataFrameToolkit) -> None:
        """Given unknown table, When called, Then ToolCallError with SQLTableError and available_tables.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2, 3]})
        ref = toolkit.register_dataframe("known_table", df)

        # Act - query references non-existent table
        bad_query = "SELECT * FROM unknown_table_xyz"
        result = toolkit.execute_sql(query=bad_query, result_name="result")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "SQLTableError"
        with check:
            assert "invalid_tables" in result.details
        invalid_tables = result.details["invalid_tables"]
        assert isinstance(invalid_tables, list)
        with check:
            assert "unknown_table_xyz" in invalid_tables
        with check:
            assert "available_tables" in result.details
        available_tables = result.details["available_tables"]
        assert isinstance(available_tables, list)
        with check:
            assert ref.id in available_tables

    def test_execute_column_error_returns_error(self, toolkit: DataFrameToolkit) -> None:
        """Given invalid column, When called, Then ToolCallError with SQLColumnError.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        ref = toolkit.register_dataframe("users", df)

        # Act - query references non-existent column
        bad_query = f"SELECT id, nonexistent_column FROM {ref.id}"  # noqa: S608
        result = toolkit.execute_sql(query=bad_query, result_name="result")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "SQLColumnError"
        with check:
            assert "invalid_columns" in result.details
        with check:
            assert "table_columns" in result.details
        # Verify the error message identifies the problematic column
        with check:
            assert "nonexistent_column" in result.message.lower() or any(
                "nonexistent_column" in str(v).lower() for v in result.details.values()
            )

    def test_execute_blacklisted_command_returns_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DELETE/DROP/etc, When called, Then ToolCallError with SQLBlacklistedCommand.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        ref = toolkit.register_dataframe("data", df)

        # Act - try each destructive command
        delete_query = f"DELETE FROM {ref.id} WHERE id = 1"  # noqa: S608
        result_delete = toolkit.execute_sql(query=delete_query, result_name="result1")

        drop_query = f"DROP TABLE {ref.id}"
        result_drop = toolkit.execute_sql(query=drop_query, result_name="result2")

        insert_query = f"INSERT INTO {ref.id} (id, value) VALUES (4, 40)"  # noqa: S608
        result_insert = toolkit.execute_sql(query=insert_query, result_name="result3")

        # Assert DELETE
        with check:
            assert isinstance(result_delete, ToolCallError)
        with check:
            assert result_delete.error_type == "SQLBlacklistedCommand"
        with check:
            assert result_delete.details.get("blocked_command") == "DELETE"

        # Assert DROP
        with check:
            assert isinstance(result_drop, ToolCallError)
        with check:
            assert result_drop.error_type == "SQLBlacklistedCommand"
        with check:
            assert result_drop.details.get("blocked_command") == "DROP"

        # Assert INSERT
        with check:
            assert isinstance(result_insert, ToolCallError)
        with check:
            assert result_insert.error_type == "SQLBlacklistedCommand"
        with check:
            assert result_insert.details.get("blocked_command") == "INSERT"

    def test_execute_join_query_success(self, toolkit: DataFrameToolkit) -> None:
        """Given JOIN query across two tables, When called, Then returns correct result.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        users = pl.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        orders = pl.DataFrame({"order_id": [101, 102, 103], "user_id": [1, 1, 2], "amount": [50.0, 75.0, 100.0]})

        ref_users = toolkit.register_dataframe("users", users)
        ref_orders = toolkit.register_dataframe("orders", orders)

        # Act - JOIN users and orders
        query = f"""
            SELECT u.name, o.order_id, o.amount
            FROM {ref_users.id} AS u
            JOIN {ref_orders.id} AS o ON u.user_id = o.user_id
            ORDER BY o.order_id
        """  # noqa: S608
        result = toolkit.execute_sql(query=query, result_name="user_orders")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.num_rows == 3
        with check:
            assert set(result.column_names) == {"name", "order_id", "amount"}
        with check:
            assert result.parent_ids is not None
        with check:
            assert set(result.parent_ids) == {ref_users.id, ref_orders.id}

    def test_execute_result_can_be_queried(self, toolkit: DataFrameToolkit) -> None:
        """Given executed query result, When new query references result, Then works.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange - create base DataFrame
        base_df = pl.DataFrame({"category": ["A", "A", "B", "B"], "value": [10, 20, 30, 40]})
        base_ref = toolkit.register_dataframe("base", base_df)

        # Act - first query: aggregate by category
        query1 = f"SELECT category, SUM(value) AS total FROM {base_ref.id} GROUP BY category"  # noqa: S608
        result1 = toolkit.execute_sql(query=query1, result_name="category_totals")

        with check:
            assert isinstance(result1, DataFrameReference)

        # Act - second query: filter the aggregated result
        query2 = f"SELECT * FROM {result1.id} WHERE total > 50"  # noqa: S608
        result2 = toolkit.execute_sql(query=query2, result_name="high_totals")

        # Assert
        with check:
            assert isinstance(result2, DataFrameReference)
        with check:
            assert result2.num_rows == 1  # Only category B has total > 50 (70)
        with check:
            assert result2.parent_ids is not None
        with check:
            assert result1.id in result2.parent_ids

    def test_execute_aggregation_query_success(self, toolkit: DataFrameToolkit) -> None:
        """Given aggregation query with GROUP BY, When called, Then returns correct aggregated results.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        sales = pl.DataFrame({
            "region": ["North", "North", "South", "South", "East"],
            "product": ["Widget", "Gadget", "Widget", "Widget", "Gadget"],
            "revenue": [1000.0, 1500.0, 800.0, 1200.0, 2000.0],
        })
        ref = toolkit.register_dataframe("sales", sales)

        # Act - aggregate by region
        query = f"""
            SELECT region, SUM(revenue) AS total_revenue, COUNT(*) AS sale_count
            FROM {ref.id}
            GROUP BY region
            ORDER BY region
        """  # noqa: S608
        result = toolkit.execute_sql(query=query, result_name="regional_summary")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.num_rows == 3  # North, South, East
        with check:
            assert set(result.column_names) == {"region", "total_revenue", "sale_count"}

    def test_execute_with_cte_success(self, toolkit: DataFrameToolkit) -> None:
        """Given query with CTE (Common Table Expression), When called, Then returns correct result.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        scores = pl.DataFrame({"student": ["Alice", "Bob", "Charlie", "Diana"], "score": [85, 92, 78, 95]})
        ref = toolkit.register_dataframe("scores", scores)

        # Act - use CTE to calculate and filter by average
        query = f"""
            WITH avg_score AS (
                SELECT AVG(score) AS avg FROM {ref.id}
            )
            SELECT student, score
            FROM {ref.id}
            WHERE score >= 90
            ORDER BY score DESC
        """  # noqa: S608
        result = toolkit.execute_sql(query=query, result_name="high_scorers")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.num_rows == 2  # Diana (95) and Bob (92) have scores >= 90
        with check:
            assert result.column_names == ["student", "score"]

    def test_execute_empty_result_success(self, toolkit: DataFrameToolkit) -> None:
        """Given query that returns no rows, When called, Then returns DataFrameReference with zero rows.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2, 3], "status": ["active", "active", "active"]})
        ref = toolkit.register_dataframe("data", df)

        # Act - query that matches nothing
        query = f"SELECT * FROM {ref.id} WHERE status = 'inactive'"  # noqa: S608
        result = toolkit.execute_sql(query=query, result_name="no_matches")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.num_rows == 0
        with check:
            assert result.column_names == ["id", "status"]

    def test_execute_union_query_success(self, toolkit: DataFrameToolkit) -> None:
        """Given UNION query combining two tables, When called, Then returns combined result.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df1 = pl.DataFrame({"id": [1, 2], "type": ["A", "A"]})
        df2 = pl.DataFrame({"id": [3, 4], "type": ["B", "B"]})
        ref1 = toolkit.register_dataframe("data1", df1)
        ref2 = toolkit.register_dataframe("data2", df2)

        # Act - UNION query
        query = f"""
            SELECT id, type FROM {ref1.id}
            UNION ALL
            SELECT id, type FROM {ref2.id}
            ORDER BY id
        """  # noqa: S608
        result = toolkit.execute_sql(query=query, result_name="combined")

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.num_rows == 4
        with check:
            assert result.column_names == ["id", "type"]

    def test_execute_sql_tool_invoke(self, toolkit: DataFrameToolkit) -> None:
        """Given toolkit with registered DataFrame, When execute_sql tool invoked, Then returns DataFrameReference.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        ref = toolkit.register_dataframe("sales", df)

        # Act
        tools = toolkit.get_core_tools()
        tool_execute_sql = next(t for t in tools if t.name == "execute_sql")
        result = tool_execute_sql.invoke({
            "query": f"SELECT * FROM {ref.id} WHERE amount > 150",  # noqa: S608
            "result_name": "high_sales",
            "result_description": "Sales over $150",
        })

        # Assert
        with check:
            assert isinstance(result, DataFrameReference)
        with check:
            assert result.name == "high_sales"
        with check:
            assert result.description == "Sales over $150"
        with check:
            assert result.num_rows == 2  # IDs 2 and 3 have amount > 150


class TestViewAsMarkdownTable:
    """Tests for DataFrameToolkit.view_as_markdown_table method."""

    @pytest.fixture
    def toolkit(self) -> DataFrameToolkit:
        """Create a toolkit instance for testing.

        Returns:
            DataFrameToolkit: Fresh toolkit instance with no DataFrames registered.
        """
        return DataFrameToolkit()

    def test_view_by_name_returns_string(self, toolkit: DataFrameToolkit) -> None:
        """Given registered DataFrame, When view_as_markdown_table by name, Then returns markdown table with data.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"product": ["Widget", "Gadget", "Gizmo"], "price": [4.99, 12.50, 7.25]})
        toolkit.register_dataframe("sales", df)

        # Act
        result = toolkit.view_as_markdown_table("sales")

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert "|" in result
        with check:
            assert "---" in result
        with check:
            assert "Widget" in result
        with check:
            assert "12.5" in result
        with check:
            assert "Gizmo" in result

    def test_view_by_id_returns_string(self, toolkit: DataFrameToolkit) -> None:
        """Given registered DataFrame, When view_as_markdown_table by ID, Then returns markdown table with data.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"city": ["Oslo", "Lima", "Cairo"], "temp_c": [-5, 22, 35]})
        ref = toolkit.register_dataframe("sales", df)

        # Act
        result = toolkit.view_as_markdown_table(ref.id)

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert "|" in result
        with check:
            assert "---" in result
        with check:
            assert "Oslo" in result
        with check:
            assert "22" in result
        with check:
            assert "35" in result

    def test_view_not_found_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given empty toolkit, When view_as_markdown_table with nonexistent identifier, Then returns ToolCallError.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange - empty toolkit

        # Act
        result = toolkit.view_as_markdown_table("nonexistent")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "DataFrameNotFound"
        with check:
            assert "nonexistent" in result.message

    def test_view_not_found_error_has_available_info(self, toolkit: DataFrameToolkit) -> None:
        """Given toolkit with DataFrames, When not found, Then error has available names and IDs.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        ref1 = toolkit.register_dataframe("df1", pl.DataFrame({"a": [1, 2]}))
        ref2 = toolkit.register_dataframe("df2", pl.DataFrame({"b": [3, 4]}))

        # Act
        result = toolkit.view_as_markdown_table("nonexistent")

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert "available_names" in result.details
        available_names = result.details["available_names"]
        assert isinstance(available_names, list)
        with check:
            assert set(available_names) == {"df1", "df2"}
        with check:
            assert "available_ids" in result.details
        available_ids = result.details["available_ids"]
        assert isinstance(available_ids, list)
        with check:
            assert set(available_ids) == {ref1.id, ref2.id}

    def test_view_with_columns_filter(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame with columns [a, b, c], When columns=["a", "c"], Then result contains only a and c.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", columns=["a", "c"])

        # Assert
        assert isinstance(result, str)
        header_line = result.strip().split("\n")[0]
        header_cols = [col.strip() for col in header_line.strip("|").split("|")]
        with check:
            assert "a" in header_cols
        with check:
            assert "c" in header_cols
        with check:
            assert "b" not in header_cols

    def test_view_with_invalid_columns_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When columns contain nonexistent column, Then returns ToolCallError.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"name": ["Ava", "Ben", "Cal"], "age": [28, 34, 45]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", columns=["nonexistent"])

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidColumns"

    def test_view_with_invalid_columns_error_has_details(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When invalid columns, Then error has available and invalid columns.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"color": ["red", "blue", "green"], "hex": ["#FF0000", "#0000FF", "#00FF00"]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", columns=["color", "nonexistent", "missing"])

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert "available_columns" in result.details
        available_columns = result.details["available_columns"]
        assert isinstance(available_columns, list)
        with check:
            assert set(available_columns) == {"color", "hex"}
        with check:
            assert "requested_columns" in result.details
        with check:
            assert result.details["requested_columns"] == ["color", "nonexistent", "missing"]
        with check:
            assert "invalid_columns" in result.details
        invalid = result.details["invalid_columns"]
        assert invalid is not None
        assert isinstance(invalid, list)
        with check:
            assert set(invalid) == {"nonexistent", "missing"}

    def test_view_with_duplicate_columns_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When columns contain duplicates, Then returns ToolCallError with DuplicateColumns.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"sku": ["A1", "B2", "C3"], "qty": [50, 120, 75]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", columns=["sku", "sku"])

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "DuplicateColumns"
        with check:
            assert "duplicate_columns" in result.details
        with check:
            assert result.details["duplicate_columns"] == ["sku"]
        with check:
            assert result.details["requested_columns"] == ["sku", "sku"]

    def test_view_with_num_rows_zero_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When num_rows=0, Then returns ToolCallError with InvalidArgument.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"item": ["pen", "book", "tape"], "cost": [1.50, 8.99, 3.25]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", num_rows=0)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "num_rows" in result.message

    def test_view_with_negative_num_rows_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When num_rows=-1, Then returns ToolCallError with InvalidArgument.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"fruit": ["apple", "banana", "cherry"], "count": [12, 7, 25]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", num_rows=-1)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"

    def test_view_with_num_rows_exceeding_dataframe_size(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame with 3 rows, When num_rows=100, Then returns all rows without error.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"country": ["Japan", "Brazil", "Kenya"], "code": ["JP", "BR", "KE"]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", num_rows=100)

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert "Japan" in result
        with check:
            assert "BR" in result
        with check:
            assert "Kenya" in result

    def test_view_with_empty_columns_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When columns=[], Then returns ToolCallError with InvalidArgument.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"dept": ["HR", "Eng", "Sales"], "budget": [50000, 120000, 80000]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", columns=[])

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "empty" in result.message

    def test_view_with_num_rows(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame with 20 rows, When num_rows=5, Then result shows 5 data rows plus ellipsis.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": list(range(1, 21)), "value": list(range(100, 120))})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", num_rows=5)

        # Assert
        assert isinstance(result, str)
        with check:
            assert "|" in result
        with check:
            assert "---" in result
        # When num_rows < total rows, table should include ellipsis indicator
        with check:
            assert "…" in result or "..." in result
        # Verify row count: header + separator + 5 data rows + ellipsis row = content lines with pipes
        data_lines = [line for line in result.strip().split("\n") if line.strip().startswith("|")]
        # Subtract header and separator lines
        content_rows = len(data_lines) - 2
        with check:
            assert content_rows <= 6  # 5 data + 1 ellipsis

    def test_view_with_sample(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame with 100 rows, When sample=True, num_rows=5, Then returns string (smoke test).

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": list(range(1, 101)), "value": list(range(1000, 1100))})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", num_rows=5, sample=True)

        # Assert - smoke test that it returns valid markdown
        with check:
            assert isinstance(result, str)
        with check:
            assert "|" in result
        with check:
            assert "---" in result

    def test_view_with_sample_and_seed_reproducible(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When called twice with same seed, Then results are identical.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"id": list(range(1, 101)), "value": list(range(1000, 1100))})
        toolkit.register_dataframe("data", df)

        # Act
        result1 = toolkit.view_as_markdown_table("data", num_rows=10, sample=True, seed=42)
        result2 = toolkit.view_as_markdown_table("data", num_rows=10, sample=True, seed=42)

        # Assert
        with check:
            assert isinstance(result1, str)
        with check:
            assert isinstance(result2, str)
        with check:
            assert result1 == result2  # Same seed produces same result

    def test_view_with_seed_without_sample_returns_tool_call_error(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame, When seed provided without sample=True, Then returns ToolCallError.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"name": ["Alice", "Bob"], "score": [95.5, 87.3]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", seed=42)

        # Assert
        with check:
            assert isinstance(result, ToolCallError)
        with check:
            assert result.error_type == "InvalidArgument"
        with check:
            assert "seed" in result.message
        with check:
            assert result.details["seed"] == 42
        with check:
            assert result.details["sample"] is False

    def test_view_with_num_rows_equal_to_dataframe_size(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame with 3 rows, When num_rows=3, Then returns all rows without ellipsis.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"city": ["NYC", "LA", "CHI"], "pop": [8_336_817, 3_979_576, 2_693_976]})
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data", num_rows=3)

        # Assert
        assert isinstance(result, str)
        with check:
            assert "NYC" in result
        with check:
            assert "LA" in result
        with check:
            assert "CHI" in result
        with check:
            assert "…" not in result and "..." not in result

    def test_view_with_varied_data_types(self, toolkit: DataFrameToolkit) -> None:
        """Given DataFrame with strings, nulls, and large numbers, When viewed, Then renders correctly.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({
            "label": ["hello world", None, "special: <>&\"'"],
            "amount": [0, 999_999_999, -42],
        })
        toolkit.register_dataframe("data", df)

        # Act
        result = toolkit.view_as_markdown_table("data")

        # Assert
        assert isinstance(result, str)
        with check:
            assert "hello world" in result
        with check:
            assert "null" in result
        with check:
            assert "999999999" in result
        with check:
            assert "-42" in result

    def test_view_tool_invoke(self, toolkit: DataFrameToolkit) -> None:
        """Given toolkit, When view_as_markdown_table tool invoked via LangChain, Then returns markdown string.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        df = pl.DataFrame({"region": ["North", "South", "East"], "revenue": [4500, 3200, 5800]})
        toolkit.register_dataframe("sales", df)

        # Act
        tools = toolkit.get_core_tools()
        tool_view = next(t for t in tools if t.name == "view_as_markdown_table")
        result = tool_view.invoke({"identifier": "sales"})

        # Assert
        with check:
            assert isinstance(result, str)
        with check:
            assert "|" in result
        with check:
            assert "---" in result

    def test_view_tool_schema_excludes_self(self, toolkit: DataFrameToolkit) -> None:
        """Given toolkit, When tool schema inspected, Then 'self' not in properties.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_core_tools()
        tool_view = next(t for t in tools if t.name == "view_as_markdown_table")
        tool_schema = tool_view.args_schema.model_json_schema()

        # Assert - 'self' should not be in the properties
        with check:
            assert "self" not in tool_schema.get("properties", {})
        with check:
            assert "identifier" in tool_schema.get("properties", {})

    def test_view_included_in_get_tools(self, toolkit: DataFrameToolkit) -> None:
        """Given toolkit, When get_tools called, Then view_as_markdown_table in tool names.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_tools()
        tool_names = {t.name for t in tools}

        # Assert
        with check:
            assert "view_as_markdown_table" in tool_names

    def test_view_included_in_get_core_tools(self, toolkit: DataFrameToolkit) -> None:
        """Given toolkit, When get_core_tools called, Then view_as_markdown_table in tool names.

        Args:
            toolkit (DataFrameToolkit): Toolkit instance from fixture.
        """
        # Arrange
        toolkit.register_dataframe("test", pl.DataFrame({"a": [1, 2, 3]}))

        # Act
        tools = toolkit.get_core_tools()
        tool_names = {t.name for t in tools}

        # Assert
        with check:
            assert "view_as_markdown_table" in tool_names
