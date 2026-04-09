"""Fake LLM chat models for use in evaluation test suites.

Provides drop-in fake implementations of LangChain chat models that support
structured output without making real network calls.
"""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


class FakeStructuredChatModel(FakeListChatModel):
    """Fake chat model that supports structured output via JSON parsing.

    Wraps `FakeListChatModel` to implement `with_structured_output`.
    Each response in `responses` must be a JSON string that can be
    parsed by the target schema's `model_validate_json`.

    Examples:
        >>> from pydantic import BaseModel
        >>> class MySchema(BaseModel):
        ...     name: str
        ...     score: float
        >>> llm = FakeStructuredChatModel(responses=['{"name": "x", "score": 0.9}'])
        >>> chain = llm.with_structured_output(MySchema)
        >>> result = chain.invoke("rate x")
        >>> result
        MySchema(name='x', score=0.9)
    """

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        _include_raw: bool = False,
        **_kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Return a runnable that parses this model's next response as schema.

        Args:
            schema (dict[str, Any] | type): Pydantic model class to validate the JSON response against.
            _include_raw (bool): Ignored; accepted for interface compatibility.
            **_kwargs (Any): Ignored; accepted for interface compatibility.

        Returns:
            Runnable[Any, Any]: A runnable that invokes this model and parses
                the result as an instance of `schema`.

        Raises:
            TypeError: If `schema` is not a BaseModel subclass.
        """
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError(f"schema must be a BaseModel subclass, got {schema!r}")

        def parse(prompt_input: Any) -> Any:
            msg = self.invoke(prompt_input)
            return schema.model_validate_json(str(msg.content))

        return RunnableLambda(parse)


class NonSchemaModel(FakeListChatModel):
    """Fake chat model that returns a raw dict from with_structured_output.

    Subclasses FakeListChatModel directly so the with_structured_output
    override can use the exact BaseChatModel signature without conflicting
    with FakeStructuredChatModel's stricter implementation.
    """

    def with_structured_output(  # type: ignore[override]
        self,
        _schema: dict[str, Any] | type,
        *,
        _include_raw: bool = False,
        **_kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Return a runnable that always yields a raw dict, not a schema instance.

        Args:
            _schema (dict[str, Any] | type): Ignored; accepted for interface compatibility.
            _include_raw (bool): Ignored; accepted for interface compatibility.
            **_kwargs (Any): Ignored; accepted for interface compatibility.

        Returns:
            Runnable[Any, Any]: A runnable that always returns a raw dict.
        """
        return RunnableLambda(lambda _input: {"name": "Alice", "age": 30})
