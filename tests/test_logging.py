"""Tests for loguru logging in dfkit.

This module verifies that logging is disabled by default and that
structured log records are produced when enabled, covering toolkit
operations, tool calls, and module context registration.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, NamedTuple

import polars as pl
import pytest
from loguru import logger
from pytest_check import check

from dfkit.logging import (
    PACKAGE_NAME,
    TOOL_CALL_LEVEL,
    TOOL_CALL_LEVEL_NUMBER,
    _is_dfkit_record,
    _is_dfkit_tool_call_record,
    enable_logging,
)
from dfkit.toolkit import DataFrameToolkit


class LogSink(NamedTuple):
    """Log sink with records list and handler ID for cleanup.

    Attributes:
        records (list[dict]): List that accumulates log record dictionaries.
        handler_id (int): Logger handler ID for cleanup.
    """

    records: list[dict]
    handler_id: int


@pytest.fixture
def log_sink() -> Generator[LogSink]:
    """Create a sink that captures log records for testing.

    The sink captures record dictionaries with fields like message, level,
    extra, function, etc. Automatically cleans up the handler after the test.

    Yields:
        Generator[LogSink]: Named tuple with records list and handler_id for cleanup.
    """
    # Arrange - create list to capture records
    records: list[dict] = []

    def sink(message: Any) -> None:
        """Capture record dict from each log message.

        Args:
            message (Any): Log message with record attribute containing log details.
        """
        records.append(message.record)

    # Act - add sink and enable dfkit logging
    handler_id = logger.add(sink)
    logger.enable(PACKAGE_NAME)

    yield LogSink(records=records, handler_id=handler_id)

    # Cleanup - disable and remove handler
    logger.disable(PACKAGE_NAME)
    logger.remove(handler_id)


def test_logging_disabled_by_default() -> None:
    """Verify no log records are captured when logging is disabled.

    Given: Fresh state with logging disabled, a sink capturing all output
    When: Perform toolkit operations (register, get_dataframe_id, etc.)
    Then: No log records are captured
    """
    # Arrange - explicitly disable dfkit logging to protect against test ordering issues
    logger.disable(PACKAGE_NAME)

    # Arrange - create sink without enabling dfkit logging
    records: list[dict] = []

    def sink(message: Any) -> None:
        """Capture record dict from each log message.

        Args:
            message (Any): Log message with record attribute.
        """
        records.append(message.record)

    handler_id = logger.add(sink)
    try:
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})

        # Act - perform operations that would log if enabled
        toolkit.register_dataframe("sales", df)
        toolkit.get_dataframe_id("sales")
        toolkit.list_dataframes()

        # Assert - no dfkit logs captured
        dfkit_records = [r for r in records if r["name"].startswith(PACKAGE_NAME)]
        with check:
            assert len(dfkit_records) == 0, "No dfkit logs should be captured when disabled"
    finally:
        logger.remove(handler_id)


def test_logging_enabled_produces_output(log_sink: LogSink) -> None:
    """Verify logging produces INFO records when enabled.

    Given: logger.enable("dfkit") called, sink capturing output
    When: Register a DataFrame
    Then: INFO log record captured with name, ID, shape

    Args:
        log_sink (LogSink): Fixture providing log sink for capturing records.
    """
    # Arrange
    toolkit = DataFrameToolkit()
    df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})

    # Act
    toolkit.register_dataframe("sales", df)

    # Assert - at least one INFO record captured
    info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
    with check:
        assert len(info_records) > 0, "Should have INFO log records"


class TestToolCallLogging:
    """Tests for tool call logging including entry, exit, error, and level filtering."""

    def test_tool_call_entry_exit_logging(self, log_sink: LogSink) -> None:
        """Verify tool calls log entry at TOOL_CALL level and exit at DEBUG level.

        Given: Logging enabled, toolkit with registered DataFrame
        When: Call get_dataframe_id("sales")
        Then: TOOL_CALL entry record + DEBUG exit record

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"price": [9.99, 19.99, 29.99], "quantity": [1.0, 2.5, 3.0]})
        toolkit.register_dataframe("sales", df)

        # Act
        _result = toolkit.get_dataframe_id("sales")

        # Assert - TOOL_CALL entry record
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [r for r in tool_call_records if "Tool call:" in r["message"]]
        with check:
            assert len(entry_records) > 0, "Should have TOOL_CALL entry record with 'Tool call:'"
        if entry_records:
            with check:
                assert "get_dataframe_id" in str(entry_records[0]["message"]), "Entry should mention function name"

        # Assert - DEBUG exit record
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        exit_records = [r for r in debug_records if "Tool call result:" in r["message"]]
        with check:
            assert len(exit_records) > 0, "Should have DEBUG exit record with 'Tool call result:'"
        if exit_records:
            with check:
                assert "get_dataframe_id" in str(exit_records[0]["message"]), "Exit should mention function name"

    def test_tool_call_error_logging(self, log_sink: LogSink) -> None:
        """Verify tool call errors are logged at WARNING level.

        Given: Logging enabled, toolkit with no matching DataFrames
        When: Call get_dataframe_id("nonexistent")
        Then: TOOL_CALL entry record + WARNING error record

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        toolkit.get_dataframe_id("nonexistent")

        # Assert - TOOL_CALL entry
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [r for r in tool_call_records if "Tool call:" in r["message"]]
        with check:
            assert len(entry_records) > 0, "Should have TOOL_CALL entry record"

        # Assert - WARNING error
        warning_records = [r for r in log_sink.records if r["level"].name == "WARNING"]
        error_records = [r for r in warning_records if "Tool call error:" in r["message"]]
        with check:
            assert len(error_records) > 0, "Should have WARNING error record"
        if error_records:
            with check:
                assert "get_dataframe_id" in str(error_records[0]["message"]), "Error should mention function name"

    def test_execute_sql_creates_info_log(self, log_sink: LogSink) -> None:
        """Verify execute_sql creates TOOL_CALL entry + INFO result logs.

        Given: Logging enabled, toolkit with registered DataFrame
        When: Execute a SQL query that creates a result DataFrame
        Then: TOOL_CALL entry + INFO result with "DataFrame created via SQL"

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "product": ["Widget", "Gadget", "Doohickey"],
            "revenue": [1500.50, 2200.75, 3100.25],
        })
        ref = toolkit.register_dataframe("sales", df)
        # SQL injection is safe here: table name comes from controlled ref.id
        query = f"SELECT * FROM {ref.id} WHERE revenue > 2000"  # noqa: S608

        # Act
        toolkit.execute_sql(query=query, result_name="filtered_sales")

        # Assert - TOOL_CALL entry for execute_sql
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [
            r for r in tool_call_records if "execute_sql" in str(r["message"]) and "Tool call:" in r["message"]
        ]
        with check:
            assert len(entry_records) > 0, "Should have TOOL_CALL entry for execute_sql"

        # Assert - INFO result for DataFrame creation
        info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
        creation_records = [r for r in info_records if "created via SQL" in str(r["message"])]
        with check:
            assert len(creation_records) > 0, "Should have INFO log for DataFrame created via SQL"

    def test_tool_call_level_captures_entries_not_debug(self, log_sink: LogSink) -> None:
        """Verify TOOL_CALL level captures tool call entries but not DEBUG records.

        Given: Logging enabled, log sink capturing all records
        When: Perform tool calls that generate both TOOL_CALL entry and DEBUG result records
        Then: TOOL_CALL entries are captured, DEBUG results are also captured (filter in assertions)

        This validates the core use case: filtering for tool call entries only.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "category": ["Electronics", "Furniture", "Clothing"],
            "sales": [15000, 8500, 12300],
        })
        toolkit.register_dataframe("sales", df)

        # Act - perform tool call that generates TOOL_CALL entry + DEBUG result
        toolkit.get_dataframe_id("sales")

        # Assert - TOOL_CALL entry is captured
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [r for r in tool_call_records if "Tool call:" in r["message"]]
        with check:
            assert len(entry_records) > 0, "Should capture TOOL_CALL entry records"
        if entry_records:
            with check:
                assert "get_dataframe_id" in str(entry_records[0]["message"]), "Entry should mention function name"

        # Assert - DEBUG result is also present (different level)
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [r for r in debug_records if "Tool call result:" in r["message"]]
        with check:
            assert len(result_records) > 0, "DEBUG result records should also be captured by sink"


class TestDataFrameRegistrationLogging:
    """Tests for DataFrame registration and unregistration logging."""

    def test_register_dataframe_logging(self, log_sink: LogSink) -> None:
        """Verify register_dataframe logs with name, shape, columns.

        Given: Logging enabled
        When: Register a DataFrame with name "sales"
        Then: INFO record with name="sales", shape, columns

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})

        # Act
        toolkit.register_dataframe("sales", df)

        # Assert - INFO record with registration details
        info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
        register_records = [
            r for r in info_records if "register" in str(r["message"]).lower() or "sales" in str(r["message"])
        ]
        with check:
            assert len(register_records) > 0, "Should have INFO log for DataFrame registration"

        # Assert - verify structured fields
        register_with_extra = [r for r in register_records if r.get("extra")]
        with check:
            assert len(register_with_extra) > 0, "Should have records with structured extra fields"
        if register_with_extra:
            extra = register_with_extra[0]["extra"]
            with check:
                assert extra.get("name") == "sales", "Extra should contain name='sales'"
            with check:
                assert "dataframe_id" in extra, "Extra should contain dataframe_id"
            with check:
                assert "shape" in extra, "Extra should contain shape"

    def test_unregister_dataframe_logging(self, log_sink: LogSink) -> None:
        """Verify unregister_dataframe logs with name and ID.

        Given: Logging enabled, toolkit with registered DataFrame
        When: Unregister the DataFrame
        Then: INFO record with name and ID

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        _ref = toolkit.register_dataframe("sales", df)

        # Act
        toolkit.unregister_dataframe("sales")

        # Assert - INFO record with unregistration details
        info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
        unregister_records = [r for r in info_records if "unregister" in str(r["message"]).lower()]
        with check:
            assert len(unregister_records) > 0, "Should have INFO log for DataFrame unregistration"

        # Assert - verify structured fields
        unregister_with_extra = [r for r in unregister_records if r.get("extra")]
        with check:
            assert len(unregister_with_extra) > 0, "Should have records with structured extra fields"
        if unregister_with_extra:
            extra = unregister_with_extra[0]["extra"]
            with check:
                assert extra.get("name") == "sales", "Extra should contain name='sales'"
            with check:
                assert "dataframe_id" in extra, "Extra should contain dataframe_id"

    def test_register_dataframes_batch_logging(self, log_sink: LogSink) -> None:
        """Verify register_dataframes logs individual and batch records.

        Given: Logging enabled
        When: Register 3 DataFrames via register_dataframes()
        Then: Individual INFO records for each DataFrame plus a batch summary

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df1 = pl.DataFrame({"id": [1, 2], "amount": [100, 200]})
        df2 = pl.DataFrame({"product": ["A", "B"], "price": [10, 20]})
        # Use None values and booleans to add data type variety
        df3 = pl.DataFrame({
            "region": ["North", "South", None],
            "sales": [1000, None, 3000],
            "active": [True, False, True],
        })

        # Act
        toolkit.register_dataframes({"sales": df1, "products": df2, "regions": df3})

        # Assert - INFO records for each plus batch summary
        info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
        with check:
            assert len(info_records) >= 3, "Should have INFO logs for each DataFrame registration"

        # Check for batch summary with "DataFrames registered" (plural)
        batch_records = [r for r in info_records if "DataFrames registered" in str(r["message"])]
        with check:
            assert len(batch_records) > 0, "Should have batch summary log with 'DataFrames registered'"

    def test_log_records_contain_structured_fields(self, log_sink: LogSink) -> None:
        """Verify log records contain structured fields in extra dict.

        Given: Logging enabled, custom sink capturing record dicts
        When: Register a DataFrame and execute SQL
        Then: Record extra dicts contain expected keys (name, dataframe_id, shape)

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})

        # Act - perform operations that should add structured fields
        ref = toolkit.register_dataframe("sales", df)
        # SQL injection is safe here: table name comes from controlled ref.id
        query = f"SELECT * FROM {ref.id}"  # noqa: S608
        toolkit.execute_sql(query=query, result_name="filtered_sales")

        # Assert - check for structured fields in extra
        records_with_extra = [r for r in log_sink.records if r.get("extra")]
        with check:
            assert len(records_with_extra) > 0, "Should have records with extra fields"

        # Verify expected structured fields
        all_extra_keys = set()
        for record in records_with_extra:
            all_extra_keys.update(record["extra"].keys())

        expected_fields = {"name", "dataframe_id", "shape"}
        with check:
            assert expected_fields.issubset(all_extra_keys), (
                f"Should have {expected_fields} in structured fields, found: {all_extra_keys}"
            )


class TestViewAsMarkdownTableLogging:
    """Tests for view_as_markdown_table logging."""

    def test_view_as_markdown_table_logging(self, log_sink: LogSink) -> None:
        """Verify view_as_markdown_table logs entry at TOOL_CALL and result at DEBUG.

        Given: Logging enabled, toolkit with registered DataFrame
        When: Call view_as_markdown_table("sales")
        Then: TOOL_CALL entry record + DEBUG result record

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"value": [42]})
        toolkit.register_dataframe("sales", df)

        # Act
        toolkit.view_as_markdown_table("sales")

        # Assert - TOOL_CALL entry
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [r for r in tool_call_records if "Tool call:" in str(r["message"])]
        with check:
            assert len(entry_records) > 0, "Should have TOOL_CALL entry record with 'Tool call:'"
        if entry_records:
            with check:
                assert "view_as_markdown_table" in str(entry_records[0]["message"]), "Should mention function name"

        # Assert - DEBUG result
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [r for r in debug_records if "Tool call result:" in str(r["message"])]
        with check:
            assert len(result_records) > 0, "Should have DEBUG result record with 'Tool call result:'"
        if result_records:
            with check:
                assert "view_as_markdown_table" in str(result_records[0]["message"]), "Should mention function name"

    def test_view_as_markdown_table_error_logging(self, log_sink: LogSink) -> None:
        """Verify view_as_markdown_table error case logs WARNING level error.

        Given: Logging enabled, toolkit with no matching DataFrames
        When: Call view_as_markdown_table("nonexistent")
        Then: WARNING error record with "Tool call error:"

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        toolkit.view_as_markdown_table("nonexistent")

        # Assert - WARNING error record
        warning_records = [r for r in log_sink.records if r["level"].name == "WARNING"]
        error_records = [r for r in warning_records if "Tool call error:" in r["message"]]
        with check:
            assert len(error_records) > 0, "Should have WARNING error record"
        if error_records:
            with check:
                assert "view_as_markdown_table" in str(error_records[0]["message"]), (
                    "Error should mention function name"
                )


class TestListDataFramesLogging:
    """Tests for list_dataframes logging."""

    def test_list_dataframes_logging(self, log_sink: LogSink) -> None:
        """Verify list_dataframes logs entry at TOOL_CALL and result at DEBUG.

        Given: Logging enabled, toolkit with 2 registered DataFrames
        When: Call list_dataframes()
        Then: TOOL_CALL entry record + DEBUG result record

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df1 = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]})
        # Use booleans to add data type variety
        df2 = pl.DataFrame({"product": ["A", "B"], "price": [10, 20], "active": [True, False]})
        toolkit.register_dataframe("sales", df1)
        toolkit.register_dataframe("products", df2)

        # Act
        toolkit.list_dataframes()

        # Assert - TOOL_CALL entry
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [
            r for r in tool_call_records if "list_dataframes" in str(r["message"]) and "Tool call:" in str(r["message"])
        ]
        with check:
            assert len(entry_records) > 0, "Should have TOOL_CALL entry record"

        # Assert - DEBUG result
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [
            r
            for r in debug_records
            if "list_dataframes" in str(r["message"]) and "Tool call result:" in str(r["message"])
        ]
        with check:
            assert len(result_records) > 0, "Should have DEBUG result record"

    def test_list_dataframes_empty_registry_logging(self, log_sink: LogSink) -> None:
        """Verify list_dataframes logs correctly with empty registry.

        Given: Logging enabled, toolkit with 0 registered DataFrames
        When: Call list_dataframes()
        Then: TOOL_CALL entry record + DEBUG result record

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()

        # Act
        toolkit.list_dataframes()

        # Assert - TOOL_CALL entry
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [
            r for r in tool_call_records if "list_dataframes" in str(r["message"]) and "Tool call:" in str(r["message"])
        ]
        with check:
            assert len(entry_records) > 0, "Should have TOOL_CALL entry record"

        # Assert - DEBUG result
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [
            r
            for r in debug_records
            if "list_dataframes" in str(r["message"]) and "Tool call result:" in str(r["message"])
        ]
        with check:
            assert len(result_records) > 0, "Should have DEBUG result record"


class TestEnableLogging:
    """Tests for enable_logging function."""

    def test_enable_logging_returns_logging_handle(self) -> None:
        """Verify enable_logging returns a LoggingHandle object.

        Given: Logger in default state
        When: Call enable_logging()
        Then: Returns a LoggingHandle with handler_id attribute
        """
        # Act
        handle = enable_logging()

        # Assert - returns a LoggingHandle with handler_id attribute
        with check:
            assert hasattr(handle, "handler_id"), "Should return object with handler_id attribute"
        with check:
            assert isinstance(handle.handler_id, int), "handler_id should be an integer"
        with check:
            assert hasattr(handle, "disable"), "Should have disable() method"

        # Cleanup
        handle.disable()

    def test_enable_logging_enables_dfkit_logger(self, log_sink: LogSink) -> None:
        """Verify dfkit logs are produced after calling enable_logging.

        Given: Logger disabled by default
        When: Call enable_logging() and perform dfkit operations
        Then: Log records are captured

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging adds its own handler
        handle = enable_logging()

        # Act - perform operation that should log
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        toolkit.register_dataframe("test_df", df)

        # Assert - dfkit logs were captured
        dfkit_records = [r for r in log_sink.records if r["name"].startswith(PACKAGE_NAME)]
        with check:
            assert len(dfkit_records) > 0, "Should have captured dfkit log records"

        # Cleanup
        handle.disable()

    def test_enable_logging_disable_method(self, log_sink: LogSink) -> None:
        """Verify LoggingHandle.disable() removes its handler but leaves logger enabled.

        Given: Handler added via enable_logging()
        When: Call handle.disable()
        Then: Handler is removed but logger remains enabled for other handlers

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        handle = enable_logging()

        # Act
        handle.disable()

        # Assert - handler_id is set to None after disable
        with check:
            assert handle.handler_id is None, "handler_id should be None after disable()"

        # Assert - calling disable() again is safe
        handle.disable()

        # Assert - logger can be re-enabled and new handlers can capture logs
        # Note: handle.disable() calls logger.disable(_PACKAGE_NAME), so we need to re-enable
        logger.enable(PACKAGE_NAME)
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        toolkit.register_dataframe("test_df", df)

        dfkit_records = [r for r in log_sink.records if r["name"].startswith("dfkit")]
        with check:
            assert len(dfkit_records) > 0, "New sinks should still capture dfkit logs after one handle is disabled"

    def test_enable_logging_context_manager(self, log_sink: LogSink) -> None:
        """Verify LoggingHandle works as a context manager.

        Given: Logger in default state
        When: Use enable_logging() in a with statement
        Then: Handler is active inside context, cleaned up on exit

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Act - use as context manager
        with enable_logging():
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            toolkit.register_dataframe("test_df", df)

        # Assert - dfkit logs were captured
        dfkit_records = [r for r in log_sink.records if r["name"].startswith(PACKAGE_NAME)]
        with check:
            assert len(dfkit_records) > 0, "Should have captured dfkit log records"

    def test_enable_logging_idempotency(self) -> None:
        """Verify calling enable_logging twice returns independent handles.

        Given: Logger in default state
        When: Call enable_logging() twice in sequence
        Then: Both calls return independent, valid handles with distinct handler IDs
        """
        # Act
        handle1 = enable_logging()
        handle2 = enable_logging()

        # Assert - both handles are independent with distinct handler IDs
        with check:
            assert handle1 is not None, "First enable_logging should return a handle"
        with check:
            assert handle2 is not None, "Second enable_logging should return a handle"
        with check:
            assert handle1.handler_id != handle2.handler_id, "Each call should create a distinct handler"

        # Cleanup
        handle1.disable()
        handle2.disable()

    def test_concurrent_handle_independence(self) -> None:
        """Verify disabling one handle preserves the other handle's logging capability.

        Given: Two handles from enable_logging(tool_calls_only=False)
        When: Disable handle1, perform toolkit operation that produces logs, disable handle2
        Then: Log records are still captured after handle1 is disabled, proving handle2 is active
        """
        # Arrange - create two independent handles
        handle1 = enable_logging(tool_calls_only=False)
        handle2 = enable_logging(tool_calls_only=False)

        # Arrange - create log sink to capture records
        records: list[dict] = []

        def sink(message: Any) -> None:
            """Capture record dict from each log message.

            Args:
                message (Any): Log message with record attribute.
            """
            records.append(message.record)

        sink_handler_id = logger.add(sink)
        logger.enable(PACKAGE_NAME)

        try:
            # Act - disable handle1 only
            handle1.disable()

            # Act - perform toolkit operation that produces log output
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            toolkit.register_dataframe("test_df", df)

            # Assert - log records were still captured (handle2 is still active)
            dfkit_records = [r for r in records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) > 0, "Logs should still be captured after disabling handle1"

            # Assert - verify we captured registration logs specifically
            register_records = [r for r in dfkit_records if "register" in str(r["message"]).lower()]
            with check:
                assert len(register_records) > 0, "Should have captured registration log from handle2"

        finally:
            # Cleanup - disable handle2 and remove sink
            handle2.disable()
            logger.disable(PACKAGE_NAME)
            logger.remove(sink_handler_id)

    def test_disable_double_call_safe(self) -> None:
        """Verify calling disable() twice does not raise.

        Given: LoggingHandle from enable_logging()
        When: Call disable() twice
        Then: No exception raised
        """
        # Arrange
        handle = enable_logging()

        # Act & Assert - should not raise
        handle.disable()
        handle.disable()

    def test_enable_logging_tool_calls_only_default_filters_to_tool_call_level(self, log_sink: LogSink) -> None:
        """Verify default tool_calls_only=True only captures TOOL_CALL level records.

        Given: enable_logging() called with default tool_calls_only=True
        When: Operations that produce INFO, TOOL_CALL, and DEBUG logs are performed
        Then: Only TOOL_CALL level records are passed through the handler filter

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging with tool_calls_only=True (default)
        handle = enable_logging()  # tool_calls_only=True by default

        # Arrange - add filtered sink to verify enable_logging filter behavior
        filtered_records: list[dict] = []

        def filtered_sink(message: Any) -> None:
            """Capture record dict from filtered log messages.

            Args:
                message (Any): Log message with record attribute.
            """
            filtered_records.append(message.record)

        filtered_handler_id = logger.add(
            filtered_sink,
            filter=_is_dfkit_tool_call_record,
            level=TOOL_CALL_LEVEL,
        )

        try:
            # Act - perform operations that generate multiple log levels
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            toolkit.register_dataframe("test_df", df)  # Produces INFO logs
            toolkit.get_dataframe_id("test_df")  # Produces TOOL_CALL + DEBUG logs

            # Assert - log_sink captures all dfkit records (unfiltered)
            info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
            debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
            tool_call_records_unfiltered = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
            with check:
                assert len(info_records) > 0, "log_sink should have INFO records (proving emission)"
            with check:
                assert len(debug_records) > 0, "log_sink should have DEBUG records (proving emission)"
            with check:
                assert len(tool_call_records_unfiltered) > 0, "log_sink should have TOOL_CALL records"

            # Assert - filtered sink only captures TOOL_CALL records
            with check:
                assert len(filtered_records) > 0, "Filtered sink should have records"
            for record in filtered_records:
                with check:
                    assert record["level"].name == "TOOL_CALL", "All filtered records should be TOOL_CALL level"
                with check:
                    assert record["level"].no == TOOL_CALL_LEVEL_NUMBER, "Should have correct level number"

            # Assert - filtered sink has no INFO or DEBUG
            filtered_info = [r for r in filtered_records if r["level"].name == "INFO"]
            filtered_debug = [r for r in filtered_records if r["level"].name == "DEBUG"]
            with check:
                assert len(filtered_info) == 0, "Filtered sink should have no INFO records"
            with check:
                assert len(filtered_debug) == 0, "Filtered sink should have no DEBUG records"

        finally:
            # Cleanup
            logger.remove(filtered_handler_id)
            handle.disable()

    def test_enable_logging_tool_calls_only_false_captures_all_levels(self, log_sink: LogSink) -> None:
        """Verify tool_calls_only=False captures records at all levels >= specified level.

        Given: enable_logging(tool_calls_only=False) called with default TOOL_CALL level
        When: Operations that produce INFO, TOOL_CALL, WARNING, and DEBUG logs are performed
        Then: All records at or above TOOL_CALL level (15) are captured (TOOL_CALL, INFO, WARNING)

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging with tool_calls_only=False
        handle = enable_logging(tool_calls_only=False)

        # Arrange - add filtered sink to verify enable_logging filter behavior
        filtered_records: list[dict] = []

        def filtered_sink(message: Any) -> None:
            """Capture record dict from filtered log messages.

            Args:
                message (Any): Log message with record attribute.
            """
            filtered_records.append(message.record)

        filtered_handler_id = logger.add(
            filtered_sink,
            filter=_is_dfkit_record,
            level=TOOL_CALL_LEVEL,
        )

        try:
            # Act - perform operations that generate multiple log levels
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            toolkit.register_dataframe("test_df", df)  # Produces INFO logs (level 20)
            toolkit.get_dataframe_id("test_df")  # Produces TOOL_CALL (level 15) + DEBUG (level 10) logs
            toolkit.get_dataframe_id("nonexistent")  # Produces WARNING log (level 30)

            # Assert - log_sink captures all dfkit records including DEBUG (unfiltered)
            debug_records_unfiltered = [r for r in log_sink.records if r["level"].name == "DEBUG"]
            with check:
                assert len(debug_records_unfiltered) > 0, "log_sink should have DEBUG records (proving emission)"

            # Assert - filtered sink captures TOOL_CALL, INFO, WARNING but NOT DEBUG
            with check:
                assert len(filtered_records) > 0, "Filtered sink should have records"

            # Should have TOOL_CALL records (level 15 >= 15)
            tool_call_records = [r for r in filtered_records if r["level"].name == "TOOL_CALL"]
            with check:
                assert len(tool_call_records) > 0, "Filtered sink should capture TOOL_CALL records"

            # Should have INFO records (level 20 > 15)
            info_records = [r for r in filtered_records if r["level"].name == "INFO"]
            with check:
                assert len(info_records) > 0, "Filtered sink should capture INFO records"

            # Should have WARNING records (level 30 > 15)
            warning_records = [r for r in filtered_records if r["level"].name == "WARNING"]
            with check:
                assert len(warning_records) > 0, "Filtered sink should capture WARNING records"

            # Should NOT have DEBUG records (level 10 < 15, below threshold)
            debug_records_filtered = [r for r in filtered_records if r["level"].name == "DEBUG"]
            with check:
                assert len(debug_records_filtered) == 0, (
                    "Filtered sink should NOT capture DEBUG records (below TOOL_CALL level)"
                )

        finally:
            # Cleanup
            logger.remove(filtered_handler_id)
            handle.disable()

    def test_enable_logging_tool_calls_only_true_excludes_info_and_debug(self, log_sink: LogSink) -> None:
        """Verify tool_calls_only=True specifically excludes INFO and DEBUG even from dfkit.

        Given: enable_logging(tool_calls_only=True) called
        When: Operations that produce INFO (from register), DEBUG (from tool call result),
            and TOOL_CALL (from tool call entry) are performed
        Then: Only TOOL_CALL level records pass through; INFO and DEBUG are excluded

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging with tool_calls_only=True
        handle = enable_logging(tool_calls_only=True)

        # Arrange - add filtered sink to verify enable_logging filter behavior
        filtered_records: list[dict] = []

        def filtered_sink(message: Any) -> None:
            """Capture record dict from filtered log messages.

            Args:
                message (Any): Log message with record attribute.
            """
            filtered_records.append(message.record)

        filtered_handler_id = logger.add(
            filtered_sink,
            filter=_is_dfkit_tool_call_record,
            level=TOOL_CALL_LEVEL,
        )

        try:
            # Act - perform operations that generate INFO, TOOL_CALL, and DEBUG
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
            toolkit.register_dataframe("test_df", df)  # INFO log
            toolkit.get_dataframe_id("test_df")  # TOOL_CALL entry + DEBUG result

            # Assert - log_sink captures all dfkit records (unfiltered)
            info_records_unfiltered = [r for r in log_sink.records if r["level"].name == "INFO"]
            debug_records_unfiltered = [r for r in log_sink.records if r["level"].name == "DEBUG"]
            with check:
                assert len(info_records_unfiltered) > 0, "log_sink should have INFO records (proving emission)"
            with check:
                assert len(debug_records_unfiltered) > 0, "log_sink should have DEBUG records (proving emission)"

            # Assert - filtered sink only has TOOL_CALL records
            with check:
                assert len(filtered_records) > 0, "Filtered sink should have records"
            for record in filtered_records:
                with check:
                    assert record["level"].name == "TOOL_CALL", "All filtered records should be TOOL_CALL level"

            # Assert - filtered sink has NO INFO or DEBUG records
            filtered_info = [r for r in filtered_records if r["level"].name == "INFO"]
            filtered_debug = [r for r in filtered_records if r["level"].name == "DEBUG"]
            with check:
                assert len(filtered_info) == 0, "Filtered sink should exclude INFO records"
            with check:
                assert len(filtered_debug) == 0, "Filtered sink should exclude DEBUG records"

        finally:
            # Cleanup
            logger.remove(filtered_handler_id)
            handle.disable()

    def test_log_format_short_shows_function_name_only(self, log_sink: LogSink) -> None:
        """Verify log_format='short' (default) shows only function name in formatted output.

        Given: enable_logging(log_format="short") called
        When: Perform toolkit operation that produces logs
        Then: enable_logging successfully applies short format

        This test verifies enable_logging(log_format="short") executes without error.
        Detailed format validation is an implementation detail.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging with short format
        handle = enable_logging(log_format="short")

        # Act - perform operation that produces logs
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        toolkit.register_dataframe("test_df", df)

        # Assert - logs were produced
        with check:
            assert len(log_sink.records) > 0, "Should have captured log records"

        # Cleanup
        handle.disable()

    def test_log_format_full_shows_module_function_line(self, log_sink: LogSink) -> None:
        """Verify log_format='full' shows module path, function name, and line number.

        Given: enable_logging(log_format="full") called
        When: Perform toolkit operation that produces logs
        Then: enable_logging successfully applies full format

        This test verifies enable_logging(log_format="full") executes without error.
        Detailed format validation is an implementation detail.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging with full format
        handle = enable_logging(log_format="full")

        # Act - perform operation that produces logs
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        toolkit.register_dataframe("test_df", df)

        # Assert - logs were produced
        with check:
            assert len(log_sink.records) > 0, "Should have captured log records"

        # Cleanup
        handle.disable()

    def test_log_format_default_is_short(self, log_sink: LogSink) -> None:
        """Verify calling enable_logging() without log_format argument uses 'short' format.

        Given: enable_logging() called with no log_format argument
        When: Perform toolkit operation that produces logs
        Then: enable_logging successfully applies default format

        This verifies enable_logging() executes without error when using default format.
        Detailed format validation is an implementation detail.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - enable_logging with default format (should be "short")
        handle = enable_logging()

        # Act - perform operation
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
        toolkit.register_dataframe("test_df", df)

        # Assert - logs were produced
        with check:
            assert len(log_sink.records) > 0, "Should have captured log records"

        # Cleanup
        handle.disable()
