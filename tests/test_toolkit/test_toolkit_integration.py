"""Integration test simulating a complete LLM agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest
from pytest_check import check

from dfkit.models import DataFrameReference, ToolCallError
from dfkit.toolkit import DataFrameToolkit


@dataclass
class WorkflowFixture:
    """Fixture container for integration test toolkit and tools.

    Attributes:
        toolkit (DataFrameToolkit): The toolkit instance with registered DataFrames.
        tools (dict[str, Any]): Tool name to tool instance mapping.
    """

    toolkit: DataFrameToolkit
    tools: dict[str, Any]


@pytest.fixture
def workflow_toolkit() -> WorkflowFixture:
    """Create a toolkit with sample DataFrames and tool dict for workflow tests.

    Returns:
        WorkflowFixture: Fixture with toolkit and tools.
    """
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
    return WorkflowFixture(toolkit=toolkit, tools=tools)


def test_workflow_list_inspect_view(workflow_toolkit: WorkflowFixture) -> None:
    """Simulate LLM listing, inspecting, and viewing DataFrames.

    Args:
        workflow_toolkit (WorkflowFixture): Fixture with toolkit and tools.
    """
    tools = workflow_toolkit.tools

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

    # Step 3: LLM views employees data
    md_table = tools["view_as_markdown_table"].invoke({"identifier": "employees", "num_rows": 5})
    assert isinstance(md_table, str)
    with check:
        assert "Alice" in md_table
    with check:
        assert "Bob" in md_table
    with check:
        assert "Charlie" in md_table
    with check:
        assert "Diana" in md_table
    with check:
        assert "Eve" in md_table
    with check:
        assert "|" in md_table


def test_workflow_query_derive_error(workflow_toolkit: WorkflowFixture) -> None:
    """Simulate LLM querying, deriving results, viewing, and error handling.

    Args:
        workflow_toolkit (WorkflowFixture): Fixture with toolkit and tools.
    """
    tools = workflow_toolkit.tools

    # Step 1: LLM gets IDs for SQL query
    emp_id = tools["get_dataframe_id"].invoke({"name": "employees"})
    assert isinstance(emp_id, str)
    dept_id = tools["get_dataframe_id"].invoke({"name": "departments"})
    assert isinstance(dept_id, str)

    # Step 2: LLM executes SQL query with JOIN
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

    # Step 3: LLM queries the derived result
    result2 = tools["execute_sql"].invoke({
        "query": f"SELECT name FROM {result.id} WHERE salary > 90000",  # noqa: S608
        "result_name": "top_earners",
    })
    assert isinstance(result2, DataFrameReference)
    with check:
        assert result2.num_rows == 1  # Only Eve (95000)

    # Step 4: LLM views the derived result
    md_result = tools["view_as_markdown_table"].invoke({"identifier": result2.id})
    assert isinstance(md_result, str)

    # Step 5: Verify all DataFrames are visible
    final_list = tools["list_dataframes"].invoke({})
    assert isinstance(final_list, list)
    with check:
        assert len(final_list) == 4  # 2 base + 2 derived

    # Step 6: Error handling - duplicate name
    error_result = tools["execute_sql"].invoke({
        "query": f"SELECT * FROM {emp_id}",  # noqa: S608
        "result_name": "high_earners",  # Already used
    })
    assert isinstance(error_result, ToolCallError)
    with check:
        assert error_result.error_type == "DuplicateName"
