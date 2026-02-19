"""Tests for loguru logging in dfkit.

This module verifies that logging is disabled by default and that
structured log records are produced when enabled, covering toolkit
operations, tool calls, and module context registration.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import warnings
from collections.abc import Generator
from typing import NamedTuple
from unittest import mock

import loguru
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
        records (list[loguru.Record]): List that accumulates log record dictionaries.
        handler_id (int): Logger handler ID for cleanup.
    """

    records: list[loguru.Record]
    handler_id: int


@contextlib.contextmanager
def capturing_sink(*, enable_dfkit: bool = True) -> Generator[list[loguru.Record]]:
    """Context manager that adds a loguru sink and yields the captured records list.

    Registers a new loguru handler that appends each raw record dict to a list,
    optionally enables the dfkit logger for the duration of the block, then removes
    the handler (and disables the dfkit logger when `enable_dfkit=True`) on exit.

    Use this in tests that need a raw sink independent of the `log_sink` fixture.
    Pass `enable_dfkit=False` when testing state *after* a `LoggingHandle` has
    already been disabled — in that case the sink observes whether dfkit records
    flow without this helper re-enabling the logger.

    Args:
        enable_dfkit (bool): When True (default), calls `logger.enable(PACKAGE_NAME)`
            before yielding and `logger.disable(PACKAGE_NAME)` on exit. Set to False
            when the test must observe the logger state left by the code under test.

    Yields:
        Generator[list[loguru.Record]]: Mutable list that accumulates record dicts
            while the context is active. Records are appended in arrival order.

    Examples:
        >>> with capturing_sink() as captured_records:  # doctest: +SKIP
        ...     toolkit.register_dataframe("sales", df)
        ...     assert len(captured_records) > 0
    """
    captured_records: list[loguru.Record] = []

    def _sink(message: loguru.Message) -> None:
        """Append the raw record dict to the accumulated list.

        Args:
            message (loguru.Message): Loguru message object; its `record` attribute holds the raw dict.
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
    captured_records: list[loguru.Record] = []

    def sink(message: loguru.Message) -> None:
        """Capture record dict from each log message.

        Args:
            message (loguru.Message): Log message with record attribute containing log details.
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

    toolkit = DataFrameToolkit()
    df = pl.DataFrame({
        "user_id": ["alice@example.com", "bob@test.com", "charlie@domain.org"],
        "login_count": [42, 137, 89],
    })

    # Act & Assert - use capturing_sink(enable_dfkit=False) to observe the disabled state
    with capturing_sink(enable_dfkit=False) as captured_records:
        toolkit.register_dataframe("user_logins", df)
        toolkit.get_dataframe_id("user_logins")
        toolkit.list_dataframes()

        # Assert - no dfkit logs captured
        dfkit_records = [r for r in captured_records if (r["name"] or "").startswith(PACKAGE_NAME)]
        with check:
            assert len(dfkit_records) == 0, "No dfkit logs should be captured when disabled"


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


@pytest.mark.parametrize(
    ("function_name", "args_dict", "result_level", "result_marker"),
    [
        ("get_dataframe_id", {"name": "sales"}, "DEBUG", "Tool call result:"),
        ("view_as_markdown_table", {"identifier": "sales"}, "DEBUG", "Tool call result:"),
        ("list_dataframes", {}, "DEBUG", "Tool call result:"),
        ("execute_sql", {"query": None, "result_name": "filtered_sales"}, "INFO", "created via SQL"),
    ],
    ids=["get_dataframe_id", "view_as_markdown_table", "list_dataframes", "execute_sql"],
)
def test_toolkit_function_logs_tool_call_entry_and_result(
    log_sink: LogSink,
    function_name: str,
    args_dict: dict[str, object],
    result_level: str,
    result_marker: str,
) -> None:
    """Verify each toolkit function logs a TOOL_CALL entry and a result record at the expected level.

    Covers the shared pattern across get_dataframe_id, view_as_markdown_table, list_dataframes,
    and execute_sql: each call must produce exactly one TOOL_CALL entry record containing
    the function name, and a result record at the expected level containing result_marker.

    Args:
        log_sink (LogSink): Fixture providing log sink for capturing records.
        function_name (str): The toolkit method to call (e.g. "get_dataframe_id").
        args_dict (dict[str, object]): Keyword arguments to pass to the toolkit method.
            For execute_sql, the "query" value is filled in with the registered DataFrame ID.
        result_level (str): The expected log level name for the result record (e.g. "DEBUG" or "INFO").
        result_marker (str): A substring expected in the result record's message.
    """
    # Arrange - set up toolkit with a registered DataFrame for all functions that need one
    toolkit = DataFrameToolkit()
    df = pl.DataFrame({"price": [9.99, 19.99, 29.99], "quantity": [1.0, 2.5, 3.0]})
    ref = toolkit.register_dataframe("sales", df)

    # For execute_sql, fill in the query using the registered ref ID
    call_args = dict(args_dict)
    if function_name == "execute_sql":
        # SQL injection is safe here: table name comes from controlled ref.id
        call_args["query"] = f"SELECT * FROM {ref.id}"  # noqa: S608

    # Act - call the toolkit function by name using getattr
    getattr(toolkit, function_name)(**call_args)

    # Assert - TOOL_CALL entry record contains the function name
    tool_call_records = [r for r in log_sink.records if r["level"].name == "TOOL_CALL"]
    entry_records = [r for r in tool_call_records if "Tool call:" in str(r["message"])]
    with check:
        assert len(entry_records) > 0, f"Should have TOOL_CALL entry record for {function_name}"
    with check:
        assert len(entry_records) > 0 and function_name in str(entry_records[0]["message"]), (
            f"TOOL_CALL entry should mention {function_name}"
        )

    # Assert - result record exists at the expected level and contains the expected marker
    result_records = [
        r for r in log_sink.records if r["level"].name == result_level and result_marker in str(r["message"])
    ]
    with check:
        assert len(result_records) > 0, (
            f"Should have {result_level} result record containing '{result_marker}' for {function_name}"
        )


def test_execute_sql_invalid_query_logs_warning(log_sink: LogSink) -> None:
    """Verify execute_sql logs a WARNING record when given an invalid query.

    Given: Logging enabled, toolkit with a registered DataFrame
    When: Call execute_sql with a syntactically invalid SQL query
    Then: A WARNING error record is captured containing "Tool call error:"

    Args:
        log_sink (LogSink): Fixture providing log sink for capturing records.
    """
    # Arrange
    toolkit = DataFrameToolkit()
    df = pl.DataFrame({
        "account_id": ["ACC-001", "ACC-002", "ACC-003"],
        "balance_usd": [15000.0, 3200.50, 87000.25],
    })
    toolkit.register_dataframe("accounts", df)

    # Act - execute a query with invalid SQL syntax
    toolkit.execute_sql(query="SELECT FROM WHERE", result_name="invalid_result")

    # Assert - WARNING error record is present
    warning_records = [r for r in log_sink.records if r["level"].name == "WARNING"]
    error_records = [r for r in warning_records if "Tool call error:" in str(r["message"])]
    with check:
        assert len(error_records) > 0, "Should have WARNING error record for invalid SQL query"


class TestToolCallLogging:
    """Tests for tool call logging including entry, exit, error, and level filtering."""

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
        with check:
            assert len(error_records) > 0 and "get_dataframe_id" in str(error_records[0]["message"]), (
                "Error should mention function name"
            )

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
        with check:
            assert len(entry_records) > 0 and "get_dataframe_id" in str(entry_records[0]["message"]), (
                "Entry should mention function name"
            )

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
        extra = register_with_extra[0]["extra"] if register_with_extra else {}
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
        extra = unregister_with_extra[0]["extra"] if unregister_with_extra else {}
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
        with check:
            assert len(error_records) > 0 and "view_as_markdown_table" in str(error_records[0]["message"]), (
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
    def test_list_dataframes_result_extra_reflects_registry_state(
        self, log_sink: LogSink, dataframe_count: int, expected_names: list[str]
    ) -> None:
        """Verify the list_dataframes DEBUG result extra fields reflect the actual registry state.

        The TOOL_CALL entry and DEBUG result logging for list_dataframes is covered by
        `test_toolkit_function_logs_tool_call_entry_and_result`. This test focuses
        exclusively on the per-case invariant: the DEBUG result's extra dict must contain
        the correct `count` and `names` values for both empty and non-empty registries.

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

        # Assert - DEBUG result extra fields reflect the actual registry state
        debug_records = [r for r in log_sink.records if r["level"].name == "DEBUG"]
        result_records = [
            r
            for r in debug_records
            if "list_dataframes" in str(r["message"]) and "Tool call result:" in str(r["message"])
        ]
        with check:
            assert len(result_records) > 0, "Should have DEBUG result record for list_dataframes"
        extra = result_records[0].get("extra", {}) if result_records else {}
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
        dfkit_records = [r for r in log_sink.records if (r["name"] or "").startswith(PACKAGE_NAME)]
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

        dfkit_records = [r for r in log_sink.records if (r["name"] or "").startswith("dfkit")]
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
        dfkit_records = [r for r in log_sink.records if (r["name"] or "").startswith(PACKAGE_NAME)]
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

    def test_concurrent_handle_independence_and_lifecycle(self) -> None:
        """Verify handle independence across the full lifecycle: disable one, then both.

        Given: Two handles from enable_logging(level="DEBUG")
        When: Disable handle1, assert logs still flow; then disable handle2, assert logs stop
        Then: Log records are still captured after handle1 is disabled (handle2 active);
              no records are captured after handle2 is also disabled
        """
        # Arrange - create two independent handles with DEBUG level so all dfkit records
        # including INFO registration logs are captured
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
            toolkit.register_dataframe("session_data", df)

            # Assert - log records were still captured (handle2 is still active)
            dfkit_records_mid = [r for r in captured_records if (r["name"] or "").startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records_mid) > 0, "Logs should still be captured after disabling handle1"
            register_records = [r for r in dfkit_records_mid if "register" in str(r["message"]).lower()]
            with check:
                assert len(register_records) > 0, "Should have captured registration log from handle2"

            # Act - disable handle2 (last active handle) and clear previous records
            captured_records.clear()
            handle2.disable()

            # Act - perform more operations after all handles disabled
            post_toolkit = DataFrameToolkit()
            flights_df = pl.DataFrame({
                "flight_id": ["FL-001", "FL-002"],
                "altitude_ft": [35000, 28000],
            })
            post_toolkit.register_dataframe("flights_post_disable", flights_df)

            # Assert - no dfkit logs after all handles are disabled
            dfkit_records_after = [r for r in captured_records if (r["name"] or "").startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records_after) == 0, "dfkit logs should stop after all handles are disabled"

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

    @pytest.mark.parametrize(
        "handle_count",
        [1, 2],
        ids=["single-handle", "two-handles"],
    )
    def test_disable_all_handles_stops_logging(self, handle_count: int) -> None:
        """Verify dfkit logging is re-disabled after all active handles are disabled.

        Given: One or two active handles, all subsequently disabled
        When: A new sink is added and toolkit operations are performed
        Then: No dfkit log records are captured because the logger is disabled

        Args:
            handle_count (int): Number of handles to create and disable before asserting.
        """
        # Arrange - create and disable all handles
        handles = [enable_logging() for _ in range(handle_count)]
        for handle in handles:
            handle.disable()

        # Arrange - add a new sink after all handles are disabled (do NOT re-enable dfkit logger)
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
            dfkit_records = [r for r in captured_records if (r["name"] or "").startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) == 0, (
                    f"No dfkit logs should be captured after all {handle_count} handle(s) are disabled"
                )

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
            dfkit_records = [r for r in captured_records if (r["name"] or "").startswith(PACKAGE_NAME)]
            with check:
                assert len(dfkit_records) == 0, "No dfkit logs should be captured after context manager exits"

    def test_context_manager_exit_cleans_up_on_exception(self) -> None:
        """Verify __exit__ performs handler cleanup even when an exception is raised inside the block.

        Given: enable_logging() used as a context manager
        When: An exception is raised inside the with block
        Then: The handler is cleaned up (handler_id is None) and the exception propagates

        Raises:
            RuntimeError: Intentionally raised inside the context to test cleanup under failure.
        """
        # Arrange
        handle_ref: list[LoggingHandle] = []

        # Act & Assert - exception propagates and cleanup still occurs
        with pytest.raises(RuntimeError, match="simulated error"), enable_logging() as handle:
            handle_ref.append(handle)
            raise RuntimeError("simulated error")

        # Assert - handler was cleaned up even though an exception was raised
        with check:
            assert len(handle_ref) == 1, "Should have captured the handle"
        with check:
            assert handle_ref[0].handler_id is None, "handler_id should be None after exception in context manager"

    def test_get_active_handle_count_increments_on_enable(self) -> None:
        """Verify get_active_handle_count increments when enable_logging is called.

        Given: No active handles from this test
        When: enable_logging() is called once, then again
        Then: The count increases by one after each call, and returns to baseline after disable
        """
        # Arrange - snapshot the count before this test adds any handles
        baseline_count = LoggingHandle.get_active_handle_count()

        # Act - add first handle
        handle1 = enable_logging()

        # Assert - count increased by 1
        with check:
            assert LoggingHandle.get_active_handle_count() == baseline_count + 1, (
                "Count should be baseline + 1 after first enable_logging"
            )

        # Act - add second handle
        handle2 = enable_logging()

        # Assert - count increased by another 1
        with check:
            assert LoggingHandle.get_active_handle_count() == baseline_count + 2, (
                "Count should be baseline + 2 after second enable_logging"
            )

        # Act - disable first handle
        handle1.disable()

        # Assert - count decreased by 1
        with check:
            assert LoggingHandle.get_active_handle_count() == baseline_count + 1, (
                "Count should be baseline + 1 after disabling first handle"
            )

        # Act - disable second handle
        handle2.disable()

        # Assert - count back to baseline
        with check:
            assert LoggingHandle.get_active_handle_count() == baseline_count, (
                "Count should return to baseline after all handles disabled"
            )


class TestEnableLoggingFiltering:
    """Tests for enable_logging default filtering behavior."""

    @pytest.mark.parametrize(
        ("level", "present_levels", "absent_levels"),
        [
            (
                "TOOL_CALL",
                ["TOOL_CALL", "WARNING"],
                ["INFO", "DEBUG"],
            ),
            (
                "DEBUG",
                ["DEBUG", "INFO", "TOOL_CALL", "WARNING"],
                [],
            ),
            (
                "WARNING",
                ["WARNING"],
                ["TOOL_CALL", "INFO", "DEBUG"],
            ),
        ],
        ids=["default-tool-call-level", "debug-level-captures-all", "warning-level-excludes-tool-call"],
    )
    def test_enable_logging_level_filtering(
        self,
        monkeypatch: pytest.MonkeyPatch,
        level: str,
        present_levels: list[str],
        absent_levels: list[str],
    ) -> None:
        """Verify enable_logging handler captures only records at or above the configured level.

        Given: enable_logging() called with a specific level threshold
        When: Operations produce DEBUG, INFO, TOOL_CALL, and WARNING logs
        Then: Only level names in present_levels appear in stderr; absent_levels do not

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching sys.stderr safely.
            level (str): The log level string to pass to enable_logging (e.g. "DEBUG", "WARNING").
            present_levels (list[str]): Level name strings that must appear in stderr output.
            absent_levels (list[str]): Level name strings that must NOT appear in stderr output.
        """
        # Arrange - capture stderr to intercept enable_logging handler output
        captured_stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured_stderr)

        # Arrange - enable_logging with the parametrized level
        handle = enable_logging(level=level)  # type: ignore[arg-type]

        # Act - perform operations that generate DEBUG, INFO, TOOL_CALL, and WARNING logs
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "product_id": ["P-001", "P-002"],
            "price": [19.99, 39.99],
        })
        toolkit.register_dataframe("products", df)  # INFO (level 20)
        toolkit.get_dataframe_id("products")  # TOOL_CALL (25) entry + DEBUG (10) result
        toolkit.get_dataframe_id("nonexistent")  # TOOL_CALL (25) entry + WARNING (30) error

        # Cleanup before reading output so handle.disable() writes don't contaminate assertions
        handle.disable()

        stderr_output = captured_stderr.getvalue()

        # Assert - all expected levels appear in stderr
        for expected_level in present_levels:
            with check:
                assert expected_level in stderr_output, (
                    f"Should have {expected_level} level logs in stderr (level={level})"
                )

        # Assert - all excluded levels are absent from stderr
        for excluded_level in absent_levels:
            with check:
                assert excluded_level not in stderr_output, (
                    f"Should NOT have {excluded_level} level logs in stderr (level={level})"
                )


class TestEnableLoggingFormatting:
    """Tests for enable_logging log_format parameter."""

    @pytest.mark.parametrize(
        ("log_format", "expected_present", "expected_absent"),
        [
            (
                "short",
                ["get_dataframe_id"],
                ["dfkit.toolkit"],
            ),
            (
                "full",
                ["dfkit.toolkit", "get_dataframe_id"],
                [],
            ),
            (
                None,
                ["get_dataframe_id"],
                ["dfkit.toolkit"],
            ),
        ],
        ids=["short-format", "full-format", "default-is-short"],
    )
    def test_enable_logging_format_renders_expected_tokens(
        self,
        log_sink: LogSink,
        monkeypatch: pytest.MonkeyPatch,
        log_format: str | None,
        expected_present: list[str],
        expected_absent: list[str],
    ) -> None:
        """Verify enable_logging log_format controls which source-location tokens appear in stderr.

        Given: enable_logging() called with a specific log_format (or default when None)
        When: Operations produce TOOL_CALL-level logs
        Then: expected_present substrings appear in rendered stderr; expected_absent do not

        Args:
            log_sink (LogSink): Fixture providing log sink for capturing records.
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching sys.stderr safely.
            log_format (str | None): The log_format to pass to enable_logging, or None to use default.
            expected_present (list[str]): Substrings that must appear in rendered stderr output.
            expected_absent (list[str]): Substrings that must NOT appear in rendered stderr output.
        """
        # Arrange - capture stderr to inspect the rendered format output
        captured_stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured_stderr)

        # Arrange - enable_logging with specified or default format
        handle = enable_logging() if log_format is None else enable_logging(log_format=log_format)  # type: ignore[arg-type]

        # Act - perform operation that produces TOOL_CALL-level logs
        toolkit = DataFrameToolkit()
        df = pl.DataFrame({
            "region": ["North", "South", "East", "West"],
            "sales_ytd": [125000.50, 98000.25, 145000.00, 112000.75],
        })
        toolkit.register_dataframe("regional_sales", df)
        toolkit.get_dataframe_id("regional_sales")  # produces TOOL_CALL entry

        # Cleanup before reading output so handle.disable() writes don't contaminate assertions
        handle.disable()

        stderr_output = captured_stderr.getvalue()

        # Assert - raw records were captured (log_sink still receives them)
        dfkit_records = [r for r in log_sink.records if (r["name"] or "").startswith(PACKAGE_NAME)]
        with check:
            assert len(dfkit_records) > 0, "Should have captured dfkit log records"

        # Assert - expected tokens appear
        for token in expected_present:
            with check:
                assert token in stderr_output, f"Format '{log_format}' should include '{token}' in stderr"

        # Assert - excluded tokens are absent
        for token in expected_absent:
            with check:
                assert token not in stderr_output, f"Format '{log_format}' should NOT include '{token}' in stderr"

        # For full format, additionally verify the module:function:line pattern
        if log_format == "full":
            has_line_number = bool(re.search(r"dfkit\.toolkit:\w+:\d+", stderr_output))
            with check:
                assert has_line_number, "Full format should include a line number in 'module:function:line' pattern"
