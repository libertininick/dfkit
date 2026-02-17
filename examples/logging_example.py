"""Example script to showcase logging in dfkit."""

import polars as pl

from dfkit import DataFrameToolkit, enable_logging
from dfkit.models import ToolCallError

# Enable logging at TOOL_CALL level (and above) with full log format for better visibility of log details
with enable_logging(
    level="TOOL_CALL",
    log_format="full",
):
    # Create toolkit and register some DataFrames
    toolkit = DataFrameToolkit()

    # Register a DataFrame
    df_sales = pl.DataFrame({
        "product": ["A", "B", "C"],
        "quantity": [10, 20, 15],
        "price": [100.0, 200.0, 150.0],
    })

    toolkit.register_dataframe("sales", df_sales)

    # Get DataFrame ID
    df_id = toolkit.get_dataframe_id("sales")
    print(f"\nDataFrame ID: {df_id}\n")

    # List DataFrames
    toolkit.list_dataframes()

    # Execute SQL query
    if isinstance(df_id, ToolCallError):
        raise RuntimeError(df_id.message)
    result = toolkit.execute_sql(
        query=f"SELECT * FROM {df_id} WHERE quantity > 10",  # noqa: S608 - Demo with literal string, not user input
        result_name="filtered_sales",
    )

    # View as markdown table
    toolkit.view_as_markdown_table("sales", num_rows=3)

    # Try an error to show error logging
    toolkit.get_dataframe_id("nonexistent")

# Logging automatically disabled here
