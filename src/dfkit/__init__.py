"""dfkit: An agent toolkit for interacting with DataFrames."""

from loguru import logger

from dfkit.tool_modules.tool_module import ToolModule
from dfkit.toolkit import DataFrameToolkit
from dfkit.utils.logging import PACKAGE_NAME, enable_logging

logger.disable(PACKAGE_NAME)  # noqa: RUF067 - Disable logging for the dfkit module by default

__all__ = [
    "DataFrameToolkit",
    "ToolModule",
    "enable_logging",
]
