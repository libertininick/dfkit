"""Tests for loguru logging in dfkit.

This module verifies that logging is disabled by default and that
structured log records are produced when enabled, covering toolkit
operations, tool calls, and module context registration.
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from collections.abc import Generator
from typing import Any, NamedTuple
from unittest import mock

import polars as pl
import pytest
from loguru import logger
from pytest_check import check

from dfkit.logging import (
    PACKAGE_NAME,
    TOOL_CALL_LEVEL,
    TOOL_CALL_LEVEL_NUMBER,
    LoggingHandle,
    _register_tool_call_level,
    enable_logging,
)
from dfkit.toolkit import DataFrameToolkit


class LogSink(NamedTuple):
    """Log sink with records list and handler ID for cleanup.

    Attributes:
        records (list[dict[str, Any]]): List that accumulates log record dictionaries.
        handler_id (int): Logger handler ID for cleanup.
    """

    records: list[dict[str, Any]]
    handler_id: int


@contextlib.contextmanager
def capturing_sink(*, enable_dfkit: bool = True) -> Generator[list[dict[str, Any]]]:
    """Context manager that adds a loguru sink and yields the captured records list.

    Registers a new loguru handler that appends each raw record dict to a list,
    optionally enables the dfkit logger for the duration of the block, then removes
    the handler (and disables the dfkit logger when ``enable_dfkit=True``) on exit.

    Use this in tests that need a raw sink independent of the ``log_sink`` fixture.
    Pass ``enable_dfkit=False`` when testing state *after* a ``LoggingHandle`` has
    already been disabled — in that case the sink observes whether dfkit records
    flow without this helper re-enabling the logger.

    Args:
        enable_dfkit (bool): When True (default), calls ``logger.enable(PACKAGE_NAME)``
            before yielding and ``logger.disable(PACKAGE_NAME)`` on exit. Set to False
            when the test must observe the logger state left by the code under test.

    Yields:
        Generator[list[dict[str, Any]]]: Mutable list that accumulates record dicts
            while the context is active. Records are appended in arrival order.

    Examples:
        >>> with capturing_sink() as captured_records:  # doctest: +SKIP
        ...     toolkit.register_dataframe("sales", df)
        ...     assert len(captured_records) > 0
    """
    captured_records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        """Append the raw record dict to the accumulated list.

        Args:
            message (Any): Loguru message object; its ``record`` attribute holds the raw dict.
        """
        captured_records.append(message.record)

    handler_id = logger.add(_sink)
    if enable_dfkit:
        logger.enable(PACKAGE_NAME)
    try:
        yield captured_records
    finally:
        if enable_dfkit:
            logger.disable(PACKAGE_NAME)
        logger.remove(handler_id)


@pytest.fixture(autouse=True)
def restore_active_ids() -> Generator[None]:
    """Save and restore LoggingHandle._active_ids around each test.

    Because LoggingHandle._active_ids is a ClassVar[set[int]] shared across
    the entire process, tests that call enable_logging() can leak handler IDs
    into subsequent tests if they fail before cleanup. This fixture snapshots
    the set before each test and unconditionally restores it afterwards,
    removing any handler IDs that were added during the test.

    Yields:
        None: Nothing; used only for setup/teardown side effects.
    """
    # Arrange - snapshot active IDs before the test runs
    saved_ids: set[int] = set(LoggingHandle._active_ids)

    yield

    # Cleanup - find IDs added during the test and remove their handlers
    added_ids = LoggingHandle._active_ids - saved_ids
    for handler_id in added_ids:
        with contextlib.suppress(ValueError):
            logger.remove(handler_id)
    # Mutate the existing set in-place to preserve object identity (ClassVar
    # is shared; reassigning it would break other references to the same set).
    LoggingHandle._active_ids.clear()
    LoggingHandle._active_ids.update(saved_ids)


@pytest.fixture
def log_sink() -> Generator[LogSink]:
    """Create a sink that captures log records for testing.

    The sink captures record dictionaries with fields like message, level,
    extra, function, etc. Automatically cleans up the handler after the test.

    Yields:
        Generator[LogSink]: Named tuple with records list and handler_id for cleanup.
    """
    # Arrange - create list to capture records
    captured_records: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        """Capture record dict from each log message.

        Args:
            message (Any): Log message with record attribute containing log details.
        """
        captured_records.append(message.record)

    # Act - add sink and enable dfkit logging
    handler_id = logger.add(sink)
    logger.enable(PACKAGE_NAME)

    yield LogSink(records=captured_records, handler_id=handler_id)

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
    captured_records: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        """Capture record dict from each log message.

        Args:
            message (Any): Log message with record attribute.
        """
        captured_records.append(message.record)

    handler_id = logger.add(sink)
    try:
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "user_id": ["alice@example.com", "bob@test.com", "charlie@domain.org"],
            "login_count": [42, 137, 89],
        })

        # Act - perform operations that would log if enabled
        toolkit.register_dataframe("user_logins", df)
        toolkit.get_dataframe_id("user_logins")
        toolkit.list_dataframes()

        # Assert - no dfkit logs captured
        dfkit_records = [r for r in captured_records if r["name"].startswith(PACKAGE_NAME)]
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
    df = pl.DataFrame({
        "timestamp": ["2024-01-15 09:30:00", "2024-01-15 14:22:00", "2024-01-16 11:45:00"],
        "temperature": [22.5, 23.1, 21.8],
    })

    # Act
    toolkit.register_dataframe("sensor_readings", df)

    # Assert - at least one INFO record captured
    info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
    with check:
        assert len(info_records) > 0, "Should have INFO log records"

    # Assert - at least one record references the registered DataFrame name
    name_referenced = any(
        "sensor_readings" in str(r["message"]) or r.get("extra", {}).get("name") == "sensor_readings"
        for r in log_sink.records
    )
    with check:
        assert name_referenced, "At least one record should reference the registered DataFrame name 'sensor_readings'"


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
        _ = toolkit.get_dataframe_id("sales")

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
        """Verify TOOL_CALL level is distinct from DEBUG and that filtering works correctly.

        Given: Logging enabled, log sink capturing all records
        When: Perform tool calls that generate both TOOL_CALL entry and DEBUG result records
        Then: TOOL_CALL and DEBUG records are at different numeric levels; filtering by level
              name correctly separates entry records (TOOL_CALL) from result records (DEBUG)

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - use different data shape from test_tool_call_entry_exit_logging to vary coverage
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "category": ["Electronics", "Furniture", "Clothing"],
            "sales": [15000, 8500, 12300],
            "in_stock": [True, False, True],
        })
        toolkit.register_dataframe("inventory", df)

        # Act - perform tool call that generates TOOL_CALL entry + DEBUG result
        toolkit.get_dataframe_id("inventory")

        # Assert - TOOL_CALL entry is captured
        tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
        entry_records = [r for r in tool_call_records if "Tool call:" in r["message"]]
        with check:
            assert len(entry_records) > 0, "Should capture TOOL_CALL entry records"
        if entry_records:
            with check:
                assert "get_dataframe_id" in str(entry_records[0]["message"]), "Entry should mention function name"

        # Assert - DEBUG result records are at a lower numeric level than TOOL_CALL records
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [r for r in debug_records if "Tool call result:" in r["message"]]
        with check:
            assert len(result_records) > 0, "DEBUG result records should also be captured by sink"

        # Assert - level filtering correctly separates the two sets: no entry records appear in
        # debug_records and no result records appear in tool_call_records
        tool_call_level_nos = {r["level"].no for r in tool_call_records}
        debug_level_nos = {r["level"].no for r in debug_records}
        with check:
            assert tool_call_level_nos.isdisjoint(debug_level_nos), (
                "TOOL_CALL and DEBUG records must have different numeric level values"
            )

        # Assert - filtering by TOOL_CALL excludes DEBUG result records (key filtering property)
        debug_msgs_in_tool_call = [r for r in tool_call_records if "Tool call result:" in r["message"]]
        with check:
            assert len(debug_msgs_in_tool_call) == 0, (
                "DEBUG 'Tool call result:' records should NOT appear when filtering by TOOL_CALL level"
            )


class TestDataFrameRegistrationLogging:
    """Tests for DataFrame registration and unregistration logging."""

    def test_register_dataframe_logging(self, log_sink: LogSink) -> None:
        """Verify register_dataframe logs with name, shape, columns.

        Given: Logging enabled
        When: Register a DataFrame with name "orders"
        Then: INFO record with name="orders", shape, columns

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "order_id": [1001, 1002, 1003, 1004],
            "customer": ["Alice", "Bob", "Carol", "Dave"],
            "total": [299.99, 49.50, 1250.00, 89.99],
            "shipped": [True, True, False, True],
        })

        # Act
        toolkit.register_dataframe("orders", df)

        # Assert - INFO record with registration details
        info_records = [r for r in log_sink.records if r["level"].name == "INFO"]
        register_records = [
            r for r in info_records if "register" in str(r["message"]).lower() or "orders" in str(r["message"])
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
                assert extra.get("name") == "orders", "Extra should contain name='orders'"
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
        df = pl.DataFrame({
            "event_id": [501, 502],
            "event_name": ["Conference", "Workshop"],
            "attendees": [250, 75],
        })
        toolkit.register_dataframe("events", df)

        # Act
        toolkit.unregister_dataframe("events")

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
                assert extra.get("name") == "events", "Extra should contain name='events'"
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
        df = pl.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "spend": [1200.50, 340.00, 5600.25, 890.75, 2100.00],
        })

        # Act - perform operations that should add structured fields
        ref = toolkit.register_dataframe("customers", df)
        # SQL injection is safe here: table name comes from controlled ref.id
        query = f"SELECT * FROM {ref.id}"  # noqa: S608
        toolkit.execute_sql(query=query, result_name="all_customers")

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
        df = pl.DataFrame({"metric": ["conversion_rate"], "value": [0.0342]})
        toolkit.register_dataframe("kpis", df)

        # Act
        toolkit.view_as_markdown_table("kpis")

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

    @pytest.mark.parametrize(
        ("dataframe_count", "expected_names"),
        [
            (0, []),
            (2, ["sales", "products"]),
        ],
        ids=["empty-registry", "two-dataframes"],
    )
    def test_list_dataframes_logs_entry_and_result(
        self, log_sink: LogSink, dataframe_count: int, expected_names: list[str]
    ) -> None:
        """Verify list_dataframes logs TOOL_CALL entry and DEBUG result for any registry size.

        The DEBUG result record's extra fields must reflect the actual count and
        names of registered DataFrames, differentiating the empty and non-empty cases.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
            dataframe_count (int): Number of DataFrames to register before calling list_dataframes.
            expected_names (list[str]): DataFrame names that should appear in the DEBUG result extra.
        """
        # Arrange - register the requested number of DataFrames
        toolkit = DataFrameToolkit()
        if dataframe_count >= 1:
            df1 = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]})
            toolkit.register_dataframe("sales", df1)
        if dataframe_count >= 2:
            # Use booleans to add data type variety
            df2 = pl.DataFrame({"product": ["A", "B"], "price": [10, 20], "active": [True, False]})
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

        # Assert - DEBUG result record is present
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [
            r
            for r in debug_records
            if "list_dataframes" in str(r["message"]) and "Tool call result:" in str(r["message"])
        ]
        with check:
            assert len(result_records) > 0, "Should have DEBUG result record"

        # Assert - per-case: DEBUG result extra fields reflect the actual registry state
        if result_records:
            extra = result_records[0].get("extra", {})
            with check:
                assert extra.get("count") == dataframe_count, (
                    f"Result extra 'count' should be {dataframe_count}, got {extra.get('count')}"
                )
            actual_names = extra.get("names", [])
            with check:
                assert sorted(actual_names) == sorted(expected_names), (
                    f"Result extra 'names' should be {expected_names}, got {actual_names}"
                )


class TestToolCallLevelRegistration:
    """Tests for TOOL_CALL custom log level registration edge cases."""

    def test_tool_call_level_registered_with_correct_number(self) -> None:
        """Verify the TOOL_CALL level is registered with the expected numeric value at import time.

        Given: dfkit.logging has been imported
        When: The TOOL_CALL level is queried from the loguru logger
        Then: The level exists with the correct numeric value (25)
        """
        # Act - query the registered level
        level = logger.level(TOOL_CALL_LEVEL)

        # Assert - level registered with expected numeric value
        with check:
            assert level.no == TOOL_CALL_LEVEL_NUMBER, (
                f"TOOL_CALL level should have numeric value {TOOL_CALL_LEVEL_NUMBER}, got {level.no}"
            )

    def test_duplicate_level_wrong_number_warns_not_raises(self) -> None:
        """Verify a numeric mismatch on TOOL_CALL registration issues a warning, not an exception.

        Given: The TOOL_CALL level already exists with a numeric value different from expected
        When: _register_tool_call_level() runs and detects the conflict
        Then: A UserWarning is issued with the conflict details; no ValueError is raised
        """
        # Arrange - build a fake level object whose .no is wrong so the conflict branch fires
        conflicting_number = TOOL_CALL_LEVEL_NUMBER + 1
        fake_level = mock.MagicMock(spec=["no"])
        fake_level.no = conflicting_number

        # Patch logger.level so the 1-arg lookup returns our fake level (simulating the level
        # already being registered with a different numeric value).  We target
        # dfkit.logging.logger.level because that is the loguru external boundary used by the
        # production helper.
        with (
            mock.patch("dfkit.logging.logger.level", return_value=fake_level),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            # Act - invoke the real conflict-detection helper with warnings capture
            _register_tool_call_level()

        # Assert - a UserWarning was issued, not a ValueError
        with check:
            assert len(caught) == 1, "Should have issued exactly one warning"
        with check:
            assert issubclass(caught[0].category, UserWarning), "Should issue UserWarning, not raise ValueError"
        with check:
            assert "already registered with numeric value" in str(caught[0].message), (
                "Warning message should describe the conflict"
            )
        with check:
            assert str(TOOL_CALL_LEVEL_NUMBER) in str(caught[0].message), (
                "Warning message should mention the expected numeric value"
            )


class TestEnableLoggingLifecycle:
    """Tests for enable_logging handle creation, disable, context manager, and idempotency."""

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
        df = pl.DataFrame({
            "created_at": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"],
            "status": ["pending", "approved", "rejected", "pending"],
        })
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
        logger.enable(PACKAGE_NAME)
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "device_id": ["DEV-01", "DEV-02"],
            "battery_level": [0.85, 0.42],
        })
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
            df = pl.DataFrame({
                "transaction_id": ["TXN-001"],
                "amount_usd": [1599.99],
                "currency": ["USD"],
            })
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

        Given: Two handles from enable_logging(level="DEBUG")
        When: Disable handle1, perform toolkit operation that produces logs, disable handle2
        Then: Log records are still captured after handle1 is disabled, proving handle2 is active
        """
        # Arrange - create two independent handles with DEBUG level so all dfkit records including
        # INFO registration logs are captured
        handle1 = enable_logging(level="DEBUG")
        handle2 = enable_logging(level="DEBUG")

        with capturing_sink() as captured_records:
            # Act - disable handle1 only
            handle1.disable()

            # Act - perform toolkit operation that produces log output
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "session_id": ["S-001", "S-002", "S-003", "S-004", "S-005"],
                "duration_sec": [120, 340, 58, 1920, 245],
            })
            toolkit.register_dataframe("test_df", df)

            # Assert - log records were still captured (handle2 is still active)
            dfkit_records = [r for r in captured_records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) > 0, "Logs should still be captured after disabling handle1"

            # Assert - verify we captured registration logs specifically
            register_records = [r for r in dfkit_records if "register" in str(r["message"]).lower()]
            with check:
                assert len(register_records) > 0, "Should have captured registration log from handle2"

        # Cleanup - disable handle2 after the sink context exits
        handle2.disable()

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

    def test_disable_re_disables_logger_when_last_handle(self) -> None:
        """Verify dfkit logging is re-disabled after the last active handle is disabled.

        Given: A single active handle, then disabled
        When: A new sink is added and toolkit operations are performed
        Then: No dfkit log records are captured because logger is disabled
        """
        # Arrange - create and immediately disable the only handle
        handle = enable_logging()
        handle.disable()

        # Arrange - add a new sink after the handle is disabled (do NOT re-enable dfkit logger)
        with capturing_sink(enable_dfkit=False) as captured_records:
            # Act - perform toolkit operations that would log if enabled
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "order_id": [2001, 2002, 2003],
                "total_usd": [149.99, 89.50, 499.00],
            })
            toolkit.register_dataframe("orders_after_disable", df)
            toolkit.list_dataframes()

            # Assert - no dfkit records should appear (logger is re-disabled)
            dfkit_records = [r for r in captured_records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) == 0, "No dfkit logs should be captured after last handle is disabled"

    def test_concurrent_handles_keep_logger_enabled(self) -> None:
        """Verify disabling one handle while another is active does not disable the logger.

        Given: Two active handles
        When: Handle1 is disabled but handle2 is still active
        Then: dfkit log records are still captured; only after handle2 is disabled do they stop
        """
        # Arrange - create two independent handles with DEBUG level so all dfkit records including
        # INFO registration logs are captured
        handle1 = enable_logging(level="DEBUG")
        handle2 = enable_logging(level="DEBUG")

        with capturing_sink() as records_after_disable1:
            # Act - disable handle1 only, leaving handle2 active
            handle1.disable()

            # Act - perform operations with handle2 still active
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "sensor_id": ["S-100", "S-200", "S-300"],
                "reading_mv": [1200.5, 980.0, 1450.75],
            })
            toolkit.register_dataframe("sensor_data", df)

            # Assert - dfkit logs are still captured because handle2 is active
            dfkit_records_mid = [r for r in records_after_disable1 if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records_mid) > 0, "dfkit logs should still flow while handle2 is active"

            # Act - now disable handle2 (last active handle)
            records_after_disable1.clear()
            handle2.disable()

            # Act - perform more operations after all handles disabled
            toolkit2 = DataFrameToolkit()
            df2 = pl.DataFrame({
                "flight_id": ["FL-001", "FL-002"],
                "altitude_ft": [35000, 28000],
            })
            toolkit2.register_dataframe("flights_post_disable", df2)

            # Assert - no dfkit logs after all handles are disabled
            dfkit_records_after = [r for r in records_after_disable1 if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records_after) == 0, "dfkit logs should stop after all handles are disabled"

    def test_active_ids_cleaned_on_disable(self) -> None:
        """Verify logging stops after all handles are disabled.

        Given: Two handles created from enable_logging()
        When: Both handles are disabled
        Then: A new sink captures no dfkit records from subsequent toolkit operations
        """
        # Arrange - create two independent handles then disable both
        handle1 = enable_logging()
        handle2 = enable_logging()
        handle1.disable()
        handle2.disable()

        # Arrange - add a new sink after all handles are disabled (do NOT re-enable dfkit logger)
        with capturing_sink(enable_dfkit=False) as captured_records:
            # Act - perform toolkit operations that would log if dfkit were enabled
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "product_sku": ["SKU-001", "SKU-002", "SKU-003"],
                "units_sold": [142, 87, 310],
                "revenue_usd": [2130.00, 1044.00, 4650.00],
            })
            toolkit.register_dataframe("product_sales", df)
            toolkit.list_dataframes()

            # Assert - no dfkit records captured because both handles were disabled
            dfkit_records = [r for r in captured_records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) == 0, "No dfkit logs should be captured after all handles are disabled"

    def test_context_manager_re_disables_on_exit(self) -> None:
        """Verify the context manager triggers logger re-disable on exit.

        Given: enable_logging() used as a context manager
        When: The with block exits
        Then: dfkit logging is disabled; a new sink captures no dfkit records
        """
        # Act - use enable_logging as context manager, then exit
        with enable_logging():
            pass  # Block exits, handle.disable() is called automatically

        # Arrange - add a new sink after the context manager exits (do NOT re-enable dfkit logger)
        with capturing_sink(enable_dfkit=False) as captured_records:
            # Act - perform operations that would log if dfkit were still enabled
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "experiment_id": ["EXP-001", "EXP-002", "EXP-003"],
                "result_score": [0.92, 0.87, 0.95],
                "passed": [True, False, True],
            })
            toolkit.register_dataframe("experiment_results", df)
            toolkit.get_dataframe_id("experiment_results")

            # Assert - no dfkit records captured after context manager exited
            dfkit_records = [r for r in captured_records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) == 0, "No dfkit logs should be captured after context manager exits"


class TestEnableLoggingFiltering:
    """Tests for enable_logging default filtering behavior."""

    def test_enable_logging_default_handler_captures_tool_call_and_above(self) -> None:
        """Verify enable_logging() handler captures TOOL_CALL and above, excluding INFO and DEBUG.

        Given: enable_logging() creates a stderr handler at the default TOOL_CALL level (25)
        When: Operations produce DEBUG, INFO, TOOL_CALL, and WARNING logs
        Then: TOOL_CALL and WARNING records reach stderr; INFO and DEBUG are excluded because
              their numeric values (20 and 10) fall below the TOOL_CALL threshold (25)

        This test captures stderr output from the actual handler created by enable_logging
        to verify the default level wiring is correct.
        """
        # Arrange - capture stderr to intercept enable_logging handler output
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            # Arrange - enable_logging with default level (TOOL_CALL = 25)
            handle = enable_logging()

            # Act - perform operations that generate DEBUG, INFO, TOOL_CALL, and WARNING logs
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "product_id": ["P-001", "P-002"],
                "price": [19.99, 39.99],
            })
            toolkit.register_dataframe("test_df", df)  # INFO (level 20) - below TOOL_CALL threshold
            toolkit.get_dataframe_id("test_df")  # TOOL_CALL (25) entry + DEBUG (10) result
            toolkit.get_dataframe_id("nonexistent")  # TOOL_CALL (25) entry + WARNING (30) error

            # Get captured output
            stderr_output = captured_stderr.getvalue()

            # Assert - stderr should contain TOOL_CALL entries (level 25 = at threshold)
            with check:
                assert "TOOL_CALL" in stderr_output, "Should have TOOL_CALL level logs in stderr"

            # Assert - stderr should contain WARNING level logs (level 30 > threshold)
            with check:
                assert "WARNING" in stderr_output, "Should have WARNING level logs in stderr"

            # Assert - stderr should NOT contain INFO logs (level 20 < TOOL_CALL threshold 25)
            with check:
                assert "INFO" not in stderr_output, (
                    "Should NOT have INFO level logs in stderr (below TOOL_CALL threshold)"
                )

            # Assert - stderr should NOT contain DEBUG logs (level 10 < TOOL_CALL threshold 25)
            with check:
                assert "DEBUG" not in stderr_output, (
                    "Should NOT have DEBUG level logs in stderr (below TOOL_CALL threshold)"
                )

            # Cleanup
            handle.disable()

        finally:
            sys.stderr = original_stderr

    def test_enable_logging_debug_level_captures_all_levels(self) -> None:
        """Verify enable_logging(level="DEBUG") passes DEBUG, INFO, TOOL_CALL, and WARNING records.

        Given: enable_logging(level="DEBUG") creates a stderr handler at the DEBUG threshold (10)
        When: Operations produce DEBUG, INFO, TOOL_CALL, and WARNING logs
        Then: All four levels appear in stderr because DEBUG (10) is the lowest threshold
        """
        # Arrange - capture stderr to intercept enable_logging handler output
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            # Arrange - enable_logging with DEBUG level (threshold = 10)
            handle = enable_logging(level="DEBUG")

            # Act - perform operations that generate all level types
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "store_id": ["S-01", "S-02", "S-03"],
                "revenue": [45000.0, 32000.0, 61000.0],
            })
            toolkit.register_dataframe("stores", df)  # INFO (level 20)
            toolkit.get_dataframe_id("stores")  # TOOL_CALL (25) entry + DEBUG (10) result
            toolkit.get_dataframe_id("nonexistent")  # TOOL_CALL (25) entry + WARNING (30) error

            # Get captured output
            stderr_output = captured_stderr.getvalue()

            # Assert - DEBUG threshold means all four levels are included
            with check:
                assert "DEBUG" in stderr_output, "Should have DEBUG level logs (at threshold)"
            with check:
                assert "INFO" in stderr_output, "Should have INFO level logs (above DEBUG threshold)"
            with check:
                assert "TOOL_CALL" in stderr_output, "Should have TOOL_CALL level logs (above DEBUG threshold)"
            with check:
                assert "WARNING" in stderr_output, "Should have WARNING level logs (above DEBUG threshold)"

            # Cleanup
            handle.disable()

        finally:
            sys.stderr = original_stderr

    def test_enable_logging_warning_level_excludes_tool_call_info_debug(self) -> None:
        """Verify enable_logging(level="WARNING") only passes WARNING and above.

        Given: enable_logging(level="WARNING") creates a stderr handler at threshold 30
        When: Operations produce DEBUG, INFO, TOOL_CALL, and WARNING logs
        Then: Only WARNING appears in stderr; DEBUG, INFO, TOOL_CALL are excluded because
              their numeric values (10, 20, 25) fall below the WARNING threshold (30)
        """
        # Arrange - capture stderr to intercept enable_logging handler output
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            # Arrange - enable_logging with WARNING level (threshold = 30)
            handle = enable_logging(level="WARNING")

            # Act - trigger all level types
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "campaign_id": ["C-001", "C-002"],
                "impressions": [120000, 85000],
                "clicks": [3400, 2100],
            })
            toolkit.register_dataframe("campaigns", df)  # INFO (level 20) - below WARNING
            toolkit.get_dataframe_id("campaigns")  # TOOL_CALL (25) entry + DEBUG (10) result - below WARNING
            toolkit.get_dataframe_id("nonexistent")  # TOOL_CALL (25) entry + WARNING (30) error

            # Get captured output
            stderr_output = captured_stderr.getvalue()

            # Assert - WARNING level (30) is at threshold and should appear
            with check:
                assert "WARNING" in stderr_output, "Should have WARNING level logs (at threshold)"

            # Assert - TOOL_CALL (25), INFO (20), DEBUG (10) are all below WARNING threshold
            with check:
                assert "TOOL_CALL" not in stderr_output, (
                    "Should NOT have TOOL_CALL logs (level 25 < WARNING threshold 30)"
                )
            with check:
                assert "INFO" not in stderr_output, "Should NOT have INFO logs (level 20 < WARNING threshold 30)"
            with check:
                assert "DEBUG" not in stderr_output, "Should NOT have DEBUG logs (level 10 < WARNING threshold 30)"

            # Cleanup
            handle.disable()

        finally:
            sys.stderr = original_stderr


class TestEnableLoggingFormatting:
    """Tests for enable_logging log_format parameter."""

    def test_log_format_short_shows_function_name_only(self, log_sink: LogSink) -> None:
        """Verify log_format='short' renders only the function name, not the module path.

        The short format string is ``{function} - {message}``. It must include the
        function name (e.g. ``register_dataframe``) but must NOT include the full
        module path (e.g. ``dfkit.toolkit``).

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - capture stderr to inspect the rendered format output
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            # Arrange - enable_logging with short format
            handle = enable_logging(log_format="short")

            # Act - perform operation that produces TOOL_CALL-level logs
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "region": ["North", "South", "East", "West"],
                "sales_ytd": [125000.50, 98000.25, 145000.00, 112000.75],
            })
            toolkit.register_dataframe("regions", df)
            toolkit.get_dataframe_id("regions")  # produces TOOL_CALL entry

            stderr_output = captured_stderr.getvalue()

            # Assert - raw records were captured (log_sink still receives them)
            dfkit_records = [r for r in log_sink.records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) > 0, "Should have captured dfkit log records"

            # Assert - rendered output contains the function name
            with check:
                assert "get_dataframe_id" in stderr_output, "Short format should include the calling function name"

            # Assert - rendered output does NOT contain the module path (short format omits {name})
            with check:
                assert "dfkit.toolkit" not in stderr_output, (
                    "Short format should NOT include the module path (dfkit.toolkit)"
                )

        finally:
            sys.stderr = original_stderr
            handle.disable()

    def test_log_format_full_shows_module_function_line(self, log_sink: LogSink) -> None:
        """Verify log_format='full' renders module path, function name, and line number.

        The full format string is ``{name}:{function}:{line} - {message}``. All three
        source-location tokens must appear in the rendered stderr output.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - capture stderr to inspect the rendered format output
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            # Arrange - enable_logging with full format
            handle = enable_logging(log_format="full")

            # Act - perform operation that produces TOOL_CALL-level logs
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "log_level": ["INFO", "WARNING", "ERROR"],
                "event_count": [1500, 45, 3],
            })
            toolkit.register_dataframe("log_counts", df)
            toolkit.get_dataframe_id("log_counts")  # produces TOOL_CALL entry

            stderr_output = captured_stderr.getvalue()

            # Assert - raw records were captured (log_sink still receives them)
            dfkit_records = [r for r in log_sink.records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) > 0, "Should have captured dfkit log records"

            # Assert - rendered output contains the full module path
            with check:
                assert "dfkit.toolkit" in stderr_output, "Full format should include the module path (dfkit.toolkit)"

            # Assert - rendered output contains the function name
            with check:
                assert "get_dataframe_id" in stderr_output, "Full format should include the calling function name"

            # Assert - rendered output contains a numeric line number (digit after the last colon
            # in the source-location segment, e.g. ``dfkit.toolkit:get_dataframe_id:482``)
            import re  # noqa: PLC0415 - local import intentional; re is stdlib

            has_line_number = bool(re.search(r"dfkit\.toolkit:\w+:\d+", stderr_output))
            with check:
                assert has_line_number, "Full format should include a line number in 'module:function:line' pattern"

        finally:
            sys.stderr = original_stderr
            handle.disable()

    def test_log_format_default_is_short(self, log_sink: LogSink) -> None:
        """Verify enable_logging() without log_format uses 'short' format by default.

        The default format must behave identically to ``log_format='short'``:
        function name appears in rendered output, module path does not.

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
        """
        # Arrange - capture stderr to inspect the rendered format output
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr

            # Arrange - enable_logging with default format (should be "short")
            handle = enable_logging()

            # Act - perform operation that produces TOOL_CALL-level logs
            toolkit = DataFrameToolkit()
            df = pl.DataFrame({
                "ip_address": ["192.168.1.1", "192.168.1.2", "192.168.1.3"],
                "request_count": [340, 125, 890],
            })
            toolkit.register_dataframe("web_traffic", df)
            toolkit.get_dataframe_id("web_traffic")  # produces TOOL_CALL entry

            stderr_output = captured_stderr.getvalue()

            # Assert - raw records were captured (log_sink still receives them)
            dfkit_records = [r for r in log_sink.records if r["name"].startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) > 0, "Should have captured dfkit log records"

            # Assert - default behaves like short: function name appears
            with check:
                assert "get_dataframe_id" in stderr_output, (
                    "Default format should include the calling function name (same as 'short')"
                )

            # Assert - default behaves like short: module path is absent
            with check:
                assert "dfkit.toolkit" not in stderr_output, (
                    "Default format should NOT include module path (same as 'short', omits {name})"
                )

        finally:
            sys.stderr = original_stderr
            handle.disable()
