"""Integration test simulating a complete LLM agent workflow."""

from __future__ import annotations

import polars as pl
from pytest_check import check

from dfkit.models import DataFrameReference, ToolCallError
from dfkit.toolkit import DataFrameToolkit


def test_full_workflow() -> None:
    """Simulate complete LLM interaction: list, inspect, query, query result."""
    # Arrange - create toolkit with sample DataFrames
    toolkit = DataFrameToolkit()
    toolkit.register_dataframe(
        "employees",
        pl.DataFrame({
            "emp_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "dept": ["eng", "eng", "sales", "sales", "eng"],
            "salary": [90000, 85000, 70000, 75000, 95000],
        }),
        description="Employee records",
    )
    toolkit.register_dataframe(
        "departments",
        pl.DataFrame({
            "dept": ["eng", "sales", "hr"],
            "budget": [500000, 300000, 200000],
        }),
        description="Department budgets",
    )

    tools = {t.name: t for t in toolkit.get_tools()}

    # Step 1: LLM lists available DataFrames
    available = tools["list_dataframes"].invoke({})
    assert isinstance(available, list)
    with check:
        assert len(available) == 2
    names = {ref.name for ref in available}
    with check:
        assert names == {"employees", "departments"}

    # Step 2: LLM gets reference for employees to inspect schema
    emp_ref = tools["get_dataframe_reference"].invoke({"identifier": "employees"})
    assert isinstance(emp_ref, DataFrameReference)
    with check:
        assert emp_ref.column_names == ["emp_id", "name", "dept", "salary"]

    # Step 2.5: LLM views a sample of the employees data
    md_table = tools["view_as_markdown_table"].invoke({"identifier": "employees", "num_rows": 3})
    assert isinstance(md_table, str)
    with check:
        assert "Alice" in md_table or "Bob" in md_table  # At least some data visible
    with check:
        assert "|" in md_table  # Markdown table format

    # Step 3: LLM gets ID for SQL query
    emp_id = tools["get_dataframe_id"].invoke({"name": "employees"})
    assert isinstance(emp_id, str)
    with check:
        assert emp_id == emp_ref.id

    dept_id = tools["get_dataframe_id"].invoke({"name": "departments"})
    assert isinstance(dept_id, str)

    # Step 4: LLM executes SQL query with JOIN
    result = tools["execute_sql"].invoke({
        "query": (
            f"SELECT e.name, e.salary, d.budget "  # noqa: S608
            f"FROM {emp_id} e JOIN {dept_id} d ON e.dept = d.dept "
            f"WHERE e.salary > 80000"
        ),
        "result_name": "high_earners",
        "result_description": "Employees earning over 80k with dept budget",
    })
    assert isinstance(result, DataFrameReference)
    with check:
        assert result.name == "high_earners"
    with check:
        assert result.num_rows == 3  # Alice, Bob, Eve
    with check:
        assert set(result.column_names) == {"name", "salary", "budget"}

    # Step 5: LLM queries the derived result
    result2 = tools["execute_sql"].invoke({
        "query": f"SELECT name FROM {result.id} WHERE salary > 90000",  # noqa: S608
        "result_name": "top_earners",
    })
    assert isinstance(result2, DataFrameReference)
    with check:
        assert result2.num_rows == 1  # Only Eve (95000)

    # Step 5.5: LLM views the derived result
    md_result = tools["view_as_markdown_table"].invoke({"identifier": result2.id})
    assert isinstance(md_result, str)

    # Step 6: Verify all DataFrames are visible
    final_list = tools["list_dataframes"].invoke({})
    assert isinstance(final_list, list)
    with check:
        assert len(final_list) == 4  # 2 base + 2 derived

    # Step 7: Error handling - duplicate name
    error_result = tools["execute_sql"].invoke({
        "query": f"SELECT * FROM {emp_id}",  # noqa: S608
        "result_name": "high_earners",  # Already used
    })
    assert isinstance(error_result, ToolCallError)
    with check:
        assert error_result.error_type == "DuplicateName"
