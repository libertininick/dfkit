"""Logging utilities for dfkit.

This module provides a custom TOOL_CALL log level and a context manager for
enabling/disabling dfkit logging with loguru.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, Final, Literal

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

PACKAGE_NAME: Final[str] = __name__.split(".")[0]

# Remove loguru's default stderr handler to prevent duplicate output
# Only removes handler 0 (the default); other user-added handlers remain intact
with contextlib.suppress(ValueError):
    logger.remove(0)

# Register custom TOOL_CALL level (between DEBUG=10 and INFO=20)
TOOL_CALL_LEVEL: Final[str] = "TOOL_CALL"
TOOL_CALL_LEVEL_NUMBER: Final[int] = 15  # Between DEBUG (10) and INFO (20)
try:
    existing_level = logger.level(TOOL_CALL_LEVEL)
except ValueError:
    # Level does not exist yet; register it
    logger.level(TOOL_CALL_LEVEL, no=TOOL_CALL_LEVEL_NUMBER, icon="🔧")
else:
    # Level exists; verify numeric value matches
    if existing_level.no != TOOL_CALL_LEVEL_NUMBER:
        msg = (
            f"TOOL_CALL level already registered with numeric value {existing_level.no},"
            f" expected {TOOL_CALL_LEVEL_NUMBER}"
        )
        raise ValueError(msg)

# Define valid log levels for enable_logging
type LogLevel = Literal[
    "TRACE",
    "DEBUG",
    "TOOL_CALL",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

# Define valid log format styles
type LogFormat = Literal["short", "full"]


class LoggingHandle:
    """Handle for managing dfkit logging lifecycle.

    Stores the handler ID from logger.add() and provides cleanup via disable()
    or automatic cleanup through context manager protocol.

    Examples:
        >>> with enable_logging():  # doctest: +SKIP
        ...     logger.info("Temporary logging enabled")

        >>> handle = enable_logging()  # doctest: +SKIP
        >>> logger.info("Logging enabled")  # doctest: +SKIP
        >>> handle.disable()  # doctest: +SKIP
    """

    def __init__(self, handler_id: int) -> None:
        """Initialize the logging handle.

        Args:
            handler_id (int): The loguru handler ID from logger.add().
        """
        self.handler_id: int | None = handler_id

    def disable(self) -> None:
        """Remove the handler associated with this logging handle."""
        if self.handler_id is None:
            return
        with contextlib.suppress(ValueError):
            logger.remove(self.handler_id)
        self.handler_id = None

    def __enter__(self) -> LoggingHandle:
        """Enter context manager.

        Returns:
            LoggingHandle: This handle instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and disable logging.

        Args:
            *args (object): Exception information (type, value, traceback).
        """
        self.disable()


def enable_logging(
    *,
    level: LogLevel = TOOL_CALL_LEVEL,
    tool_calls_only: bool = True,
    log_format: LogFormat = "short",
) -> LoggingHandle:
    """Enable dfkit logging with loguru's default format.

    Use this to observe which toolkit methods are invoked during an LLM agent
    session. Each call returns an independent handle that manages its own
    handler; use the handle's disable() method or context manager protocol to
    clean up when finished.

    Args:
        level (LogLevel): Minimum log level to display. Defaults to "TOOL_CALL"
            which surfaces toolkit method invocations. Lower to "DEBUG" or
            "TRACE" when diagnosing unexpected behavior inside dfkit internals.
            Valid values: "TRACE", "DEBUG", "TOOL_CALL", "INFO", "WARNING",
            "ERROR", "CRITICAL".
        tool_calls_only (bool): When True (default), filters output to only
            TOOL_CALL-level records so you see a clean stream of toolkit method
            invocations. Set to False when you need the full picture—warnings,
            errors, and debug messages—to troubleshoot issues.
        log_format (LogFormat): Controls how much source location context
            appears in each log line. Use "short" (default) for everyday
            monitoring where just the function name is enough. Switch to "full"
            when you need module:function:line to pinpoint where a log
            originated. Valid values: "short", "full".

    Returns:
        LoggingHandle: Independent handle for managing the logging handler.

    Examples:
        >>> with enable_logging():  # doctest: +SKIP
        ...     logger.info("Logging is enabled")
    """
    logger.enable(PACKAGE_NAME)

    # Choose filter based on tool_calls_only parameter
    filter_func = _is_dfkit_tool_call_record if tool_calls_only else _is_dfkit_record

    # Choose format based on log_format parameter
    if log_format == "short":
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "  # noqa: RUF027  # loguru format string
            "<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        )
    else:  # "full"
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "  # noqa: RUF027  # loguru format string
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    handler_id = logger.add(
        sys.stderr,
        level=level,
        filter=filter_func,
        format=format_str,
    )

    return LoggingHandle(handler_id)


def _is_dfkit_record(record: Record) -> bool:
    """Filter log records to only include dfkit module records.

    Used by enable_logging when tool_calls_only=False to pass all dfkit records
    at or above the configured level.

    Args:
        record (Record): The loguru Record object to filter.

    Returns:
        bool: True if the record is from the dfkit module, False otherwise.
    """
    name = record["name"]
    return name is not None and name.startswith(PACKAGE_NAME)


def _is_dfkit_tool_call_record(record: Record) -> bool:
    """Filter log records to only include dfkit TOOL_CALL level records.

    Used by enable_logging when tool_calls_only=True (the default) to pass only
    TOOL_CALL-level records from dfkit.

    Args:
        record (Record): The loguru Record object to filter.

    Returns:
        bool: True if the record is from dfkit and at exactly TOOL_CALL level,
            False otherwise.
    """
    name = record["name"]
    level_no = record["level"].no
    return name is not None and name.startswith(PACKAGE_NAME) and level_no == TOOL_CALL_LEVEL_NUMBER
