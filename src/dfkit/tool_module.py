"""Protocol for tool modules that extend the DataFrame toolkit.

This module defines the ToolModule Protocol, which tool modules must implement
to integrate with the DataFrame toolkit. Modules provide LangChain tools and
system prompts that guide LLM agents in using those tools.

The protocol defines two required members (system_prompt and get_tools) but does
NOT enforce constructor signatures. See the ToolModule class docstring for
implementation details and the recommended constructor pattern.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.tools import BaseTool


@runtime_checkable
class ToolModule(Protocol):
    """Protocol for tool modules that extend the DataFrame toolkit.

    Modules must implement two members:
    - system_prompt: Property returning LLM guidance text for using the module's tools
    - get_tools(): Method returning a list of LangChain BaseTool instances

    Constructor Contract (NOT enforced by protocol):
        Modules should accept a ToolModuleContext in their ``__init__`` method::

            def __init__(self, context: ToolModuleContext) -> None:
                self._context = context

        Note: runtime_checkable protocols cannot verify ``__init__`` signatures.
        This is a convention that module authors must follow.

    Implementation Steps:
        1. Create a class that implements the ToolModule protocol
        2. Accept ToolModuleContext in __init__
        3. Store the context as an instance variable
        4. Implement system_prompt property returning guidance text
        5. Implement get_tools() returning list of LangChain tools
        6. Use the context to access DataFrame state in tool implementations

    For configurable modules needing additional parameters beyond the context,
    use a factory function that returns a configured class::

        from langchain_core.tools import tool
        from dfkit.models import ToolCallError

        def create_module(param: str) -> type:
            class ConfiguredModule:
                def __init__(self, context: ToolModuleContext) -> None:
                    self._context = context
                    self._param = param

                @property
                def system_prompt(self) -> str:
                    return f"Module with parameter: {self._param}."

                def get_tools(self) -> list[BaseTool]:
                    @tool
                    def example_tool(name: str) -> str:
                        df = self._context.get_dataframe(name)
                        if isinstance(df, ToolCallError):
                            return str(df)
                        return f"DataFrame has {len(df)} rows"
                    return [example_tool]
            return ConfiguredModule

        # Usage in toolkit:
        toolkit.get_tools(create_module("value"))
    """

    @property
    def system_prompt(self) -> str:
        """LLM guidance for using this module's tools.

        Returns:
            str: System prompt text explaining when and how to use the module's tools.
        """
        ...

    def get_tools(self) -> list[BaseTool]:
        """Get LangChain tools provided by this module.

        Returns:
            list[BaseTool]: List of LangChain BaseTool instances.
        """
        ...
