"""Demonstrates how to enable and configure logging in dfkit.

dfkit logging is disabled by default. Users opt in by calling ``enable_logging()``,
which returns a ``LoggingHandle``. The handle can be used as a context manager
(``with enable_logging(): ...``) or disabled manually via ``handle.disable()``.
When the last active handle is disabled, dfkit logging is automatically turned off.

Key concepts shown here:

- ``level``: controls the minimum log level. The custom ``TOOL_CALL`` level
  (numeric value 25, between INFO and WARNING) surfaces toolkit method
  invocations and is the default.
- ``log_format``: ``"short"`` shows ``timestamp | level | function - message``;
  ``"full"`` adds the module and line number.
- Error logging: failed operations (e.g. looking up a nonexistent DataFrame)
  are logged automatically without raising.
- Automatic cleanup: logging is re-disabled when the context manager exits.
"""

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
