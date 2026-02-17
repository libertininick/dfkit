"""Logging utilities for dfkit.

This module provides a custom TOOL_CALL log level and a context manager for
enabling/disabling dfkit logging with loguru.

Note:
    Importing this module removes loguru's default stderr handler (ID 0) to
    prevent duplicate output when ``enable_logging()`` adds its own handler.
    If your application configures loguru handlers *before* importing dfkit,
    handler 0 may no longer be the default — in that case the removal is a
    no-op (the ``ValueError`` is suppressed). To be safe, configure loguru
    handlers *after* importing dfkit, or re-add a stderr handler explicitly
    if needed.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import warnings
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from loguru import logger

if TYPE_CHECKING:
    from types import TracebackType

    from loguru import Record

PACKAGE_NAME: Final[str] = __name__.split(".")[0]

# Remove loguru's default stderr handler to prevent duplicate output when
# enable_logging() adds its own filtered handler. Handler ID 0 is the default
# stderr handler that loguru creates at import time. If another library has
# already removed or replaced handler 0 before this module is imported, the
# ValueError is suppressed and no handler is removed. See module docstring.
with contextlib.suppress(ValueError):
    logger.remove(0)

# Register custom TOOL_CALL level (between INFO=20 and WARNING=30)
TOOL_CALL_LEVEL: Final[str] = "TOOL_CALL"
TOOL_CALL_LEVEL_NUMBER: Final[int] = 25  # Between INFO (20) and WARNING (30)


def _register_tool_call_level() -> None:
    """Register the TOOL_CALL custom log level with loguru.

    Attempts to look up the TOOL_CALL level. If it does not exist, registers it
    with the configured numeric value. If it already exists with a different
    numeric value, emits a UserWarning because loguru does not permit changing
    the numeric value of an existing level.
    """
    try:
        existing_level = logger.level(TOOL_CALL_LEVEL)
    except ValueError:
        # Level does not exist yet; register it
        logger.level(TOOL_CALL_LEVEL, no=TOOL_CALL_LEVEL_NUMBER, icon="🔧")
    else:
        # Level exists; warn if numeric value differs (loguru does not allow changing no)
        if existing_level.no != TOOL_CALL_LEVEL_NUMBER:
            msg = (
                f"TOOL_CALL level already registered with numeric value {existing_level.no},"
                f" expected {TOOL_CALL_LEVEL_NUMBER}"
            )
            warnings.warn(msg, stacklevel=2)


_register_tool_call_level()

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

    _active_ids: ClassVar[set[int]] = set()
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, handler_id: int) -> None:
        """Initialize the logging handle.

        Args:
            handler_id (int): The loguru handler ID from logger.add().
        """
        self.handler_id: int | None = handler_id
        with LoggingHandle._lock:
            LoggingHandle._active_ids.add(handler_id)

    def disable(self) -> None:
        """Remove the handler associated with this logging handle.

        When this is the last active handle, ``logger.disable("dfkit")`` is
        called to suppress dfkit log messages. This will also suppress messages
        routed to any handler added independently via ``logger.enable("dfkit")``.
        See ``enable_logging`` for details.
        """
        with LoggingHandle._lock:
            if self.handler_id is None:
                return
            LoggingHandle._active_ids.discard(self.handler_id)
            with contextlib.suppress(ValueError):
                logger.remove(self.handler_id)
            self.handler_id = None
            if not LoggingHandle._active_ids:
                logger.disable(PACKAGE_NAME)

    def __enter__(self) -> LoggingHandle:
        """Enter context manager.

        Returns:
            LoggingHandle: This handle instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and disable logging.

        Args:
            exc_type (type[BaseException] | None): The exception type, if raised.
            exc_val (BaseException | None): The exception instance, if raised.
            exc_tb (TracebackType | None): The traceback, if raised.
        """
        self.disable()

    @classmethod
    def get_active_handle_count(cls) -> int:
        """Return the number of currently active logging handles.

        Returns:
            int: Count of active handles that have not been disabled.
        """
        with cls._lock:
            return len(cls._active_ids)


def enable_logging(
    *,
    level: LogLevel = TOOL_CALL_LEVEL,
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
        log_format (LogFormat): Controls how much source location context
            appears in each log line. Use "short" (default) for everyday
            monitoring where just the function name is enough. Switch to "full"
            when you need module:function:line to pinpoint where a log
            originated. Valid values: "short", "full".

    Returns:
        LoggingHandle: Independent handle for managing the logging handler.

    Note:
        This function calls ``logger.enable("dfkit")`` so that log messages
        reach the newly added handler. When the last active ``LoggingHandle``
        is disabled, ``logger.disable("dfkit")`` is called automatically. If
        your application independently calls ``logger.enable("dfkit")`` (e.g.
        to route dfkit messages to your own handler), be aware that disabling
        the last handle will re-disable the ``"dfkit"`` logger, suppressing
        your handler as well. In that case, call ``logger.enable("dfkit")``
        again after the handle is disabled.

    Examples:
        >>> with enable_logging():  # doctest: +SKIP
        ...     logger.info("Logging is enabled")
    """
    logger.enable(PACKAGE_NAME)

    if log_format == "short":
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "  # noqa: RUF027 - loguru format string
            "<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        )
    else:  # "full"
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "  # noqa: RUF027 - loguru format string
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    handler_id = logger.add(
        sys.stderr,
        level=level,
        filter=_is_dfkit_record,
        format=format_str,
    )

    return LoggingHandle(handler_id)


def _is_dfkit_record(record: Record) -> bool:
    """Filter to pass all dfkit module records.

    Default filter for enable_logging; passes all dfkit records at or above
    the configured level. Loguru's built-in level filtering handles numeric
    thresholds.

    Args:
        record (Record): The loguru Record object to filter.

    Returns:
        bool: True if the record is from the dfkit module, False otherwise.
    """
    name = record["name"]
    return name is not None and name.startswith(PACKAGE_NAME)
