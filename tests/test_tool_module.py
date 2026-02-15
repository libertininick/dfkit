"""Tests for ToolModule Protocol."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pytest_check import check

from dfkit.tool_module import ToolModule


class TestToolModuleProtocol:
    """Tests for ToolModule Protocol conformance checking."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        """Verify class with get_tools() and system_prompt satisfies ToolModule.

        A class that implements both get_tools() returning list[BaseTool] and
        system_prompt as a property should satisfy the ToolModule protocol
        when checked with isinstance().
        """

        # Arrange
        class ConformingModule:
            """A minimal module that satisfies the ToolModule protocol."""

            def __init__(self, context: object) -> None:
                """Initialize the module.

                Args:
                    context (object): The ToolModuleContext instance.
                """
                self._context = context
                self._tools: list[BaseTool] = []

            @property
            def system_prompt(self) -> str:
                """Return the system prompt for this module.

                Returns:
                    str: The system prompt text.
                """
                return "This is a test module."

            def get_tools(self) -> list[BaseTool]:
                """Return the tools provided by this module.

                Returns:
                    list[BaseTool]: The list of tools.
                """
                return list(self._tools)

        # Act
        instance = ConformingModule(context=None)
        result = isinstance(instance, ToolModule)

        # Assert
        with check:
            assert result is True, "conforming class should satisfy ToolModule protocol"

    def test_missing_get_tools_fails_protocol(self) -> None:
        """Verify class missing get_tools() does not satisfy ToolModule.

        A class that only has system_prompt but lacks get_tools() should not
        satisfy the ToolModule protocol.
        """

        # Arrange
        class MissingGetTools:
            """A class missing the get_tools() method."""

            @property
            def system_prompt(self) -> str:
                """Return the system prompt.

                Returns:
                    str: The system prompt text.
                """
                return "This module is missing get_tools()."

        # Act
        instance = MissingGetTools()
        result = isinstance(instance, ToolModule)

        # Assert
        with check:
            assert result is False, "class without get_tools() should not satisfy protocol"

    def test_missing_system_prompt_fails_protocol(self) -> None:
        """Verify class missing system_prompt does not satisfy ToolModule.

        A class that only has get_tools() but lacks system_prompt should not
        satisfy the ToolModule protocol.
        """

        # Arrange
        class MissingSystemPrompt:
            """A class missing the system_prompt property."""

            def get_tools(self) -> list[BaseTool]:
                """Return the tools.

                Returns:
                    list[BaseTool]: The list of tools.
                """
                return []

        # Act
        instance = MissingSystemPrompt()
        result = isinstance(instance, ToolModule)

        # Assert
        with check:
            assert result is False, "class without system_prompt should not satisfy protocol"
