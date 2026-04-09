"""Tests for the judge prompt template and extract_facts function.

Covers DEFAULT_JUDGE_PROMPT formatting and the extract_facts function's
schema binding, chain invocation, custom-prompt routing, and error handling.
All tests use FakeStructuredChatModel; no real network calls are made.
"""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.judge import DEFAULT_JUDGE_PROMPT, extract_facts

# region Fake LLM


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


# endregion

# region Module-level test schemas


class FactSchema(BaseModel):
    """Minimal Pydantic schema used as the extraction target in judge tests.

    All fields have defaults so that `schema()` succeeds, satisfying the
    schema-default validation in `extract_facts`.

    Attributes:
        name (str | None): A person's name extracted from the agent response.
        age (int | None): An optional age value extracted from the agent response.
    """

    name: str | None = None
    age: int | None = None


class AltFactSchema(BaseModel):
    """Alternative schema to verify extract_facts accepts any BaseModel subclass.

    All fields have defaults so that `schema()` succeeds.

    Attributes:
        score (float | None): A numeric score extracted from the agent response.
        label (str | None): A text label extracted from the agent response.
    """

    score: float | None = None
    label: str | None = None


class RequiredFieldSchema(BaseModel):
    """Schema with a required field, used to test TypeError on schema validation.

    Attributes:
        name (str): Required name field with no default.
    """

    name: str


# endregion


class TestDefaultJudgePrompt:
    """Tests for the DEFAULT_JUDGE_PROMPT module-level constant."""

    def test_default_prompt_has_two_messages(self) -> None:
        """Formatted output should contain exactly two messages (system + human)."""
        # Arrange / Act
        messages = DEFAULT_JUDGE_PROMPT.format_messages(agent_response="some agent output")

        # Assert
        assert len(messages) == 2

    def test_default_prompt_includes_agent_response_text(self) -> None:
        """Formatted messages should contain the agent_response string verbatim."""
        # Arrange
        agent_response = "The temperature is 42 degrees and the unit is Celsius."

        # Act
        messages = DEFAULT_JUDGE_PROMPT.format_messages(agent_response=agent_response)

        # Assert
        all_content = " ".join(str(m.content) for m in messages)
        assert agent_response in all_content

    def test_default_prompt_system_message_contains_judge_instruction(self) -> None:
        """The first message should carry the evaluation judge instruction text."""
        # Arrange / Act
        messages = DEFAULT_JUDGE_PROMPT.format_messages(agent_response="any response")

        # Assert — system message is first
        system_content = str(messages[0].content)
        with check:
            assert "evaluation judge" in system_content
        with check:
            assert "None" in system_content

    def test_default_prompt_accepts_empty_agent_response(self) -> None:
        """DEFAULT_JUDGE_PROMPT should format without error when agent_response is empty."""
        # Arrange / Act
        messages = DEFAULT_JUDGE_PROMPT.format_messages(agent_response="")

        # Assert — formatting succeeds and message count is unchanged
        assert len(messages) == 2

    def test_default_prompt_preserves_multiline_agent_response(self) -> None:
        """Multi-line agent responses should be embedded verbatim in the formatted prompt."""
        # Arrange
        agent_response = "Line one.\nLine two.\nLine three with unicode: \u00e9l\u00e8ve."

        # Act
        messages = DEFAULT_JUDGE_PROMPT.format_messages(agent_response=agent_response)

        # Assert
        all_content = " ".join(str(m.content) for m in messages)
        assert agent_response in all_content


class TestExtractFacts:
    """Tests for the extract_facts function."""

    @pytest.mark.parametrize(
        ("schema_class", "expected", "agent_response", "prompt"),
        [
            pytest.param(
                FactSchema,
                FactSchema(name="Bob"),
                "Bob was mentioned but no age given.",
                None,
                id="fact-schema-none-optional-field",
            ),
            pytest.param(
                AltFactSchema,
                AltFactSchema(score=0.72, label="standard"),
                "Score was 0.72; category is standard.",
                None,
                id="alt-schema-different-schema-class",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Carol", age=25),
                "Carol is 25 years old.",
                ChatPromptTemplate.from_messages([
                    ("human", "Custom extraction request: {agent_response}"),
                ]),
                id="fact-schema-custom-prompt",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Dana"),
                "Dana was mentioned.",
                None,
                id="fact-schema-prompt-none-uses-default",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name=None),
                "",
                None,
                id="fact-schema-empty-agent-response",
            ),
        ],
    )
    def test_extract_facts_builds_chain_and_returns_schema_instance(
        self,
        schema_class: type[BaseModel],
        expected: BaseModel,
        agent_response: str,
        prompt: ChatPromptTemplate | None,
    ) -> None:
        """extract_facts should build the LLM chain and return a populated schema instance.

        Covers schema variation, optional-field handling, custom prompts, prompt=None fallback,
        and empty agent responses. Each parametrize case isolates one behavioral dimension.

        Args:
            schema_class (type[BaseModel]): The Pydantic model class passed as the extraction target.
            expected (BaseModel): The pre-built model instance used both as LLM response source and assertion target.
            agent_response (str): The agent response string forwarded to `extract_facts`.
            prompt (ChatPromptTemplate | None): Custom prompt passed to `extract_facts`, or `None` to use the default.
        """
        # Arrange
        judge = FakeStructuredChatModel(responses=[expected.model_dump_json()])

        # Act
        result = extract_facts(judge, agent_response, schema_class, prompt=prompt)

        # Assert
        assert result == expected

    def test_extract_facts_missing_agent_response_variable_raises_value_error(self) -> None:
        """extract_facts should raise ValueError when custom prompt lacks agent_response variable."""
        # Arrange
        bad_prompt = ChatPromptTemplate.from_messages([
            ("human", "Evaluate this: {user_query}"),
        ])
        judge = FakeStructuredChatModel(responses=['{"name": "Eve", "age": 22}'])

        # Act / Assert
        with pytest.raises(ValueError, match="agent_response"):
            extract_facts(judge, "Eve is 22 years old.", FactSchema, prompt=bad_prompt)

    def test_extract_facts_invalid_json_response_returns_default(self) -> None:
        """extract_facts should return schema() when the LLM returns non-JSON content.

        FakeStructuredChatModel's with_structured_output calls model_validate_json,
        which raises a pydantic ValidationError for non-JSON responses. extract_facts
        catches this and returns the default schema instance.
        """
        # Arrange — the model returns plain text, not valid JSON
        judge = FakeStructuredChatModel(responses=["not valid json at all"])

        # Act
        result = extract_facts(judge, "Some response text.", FactSchema)

        # Assert — fallback to default instance
        assert result == FactSchema()

    def test_extract_facts_wrong_schema_json_returns_default(self) -> None:
        """extract_facts should return schema() when the LLM returns JSON that doesn't match the schema.

        A JSON object whose fields don't match FactSchema triggers a pydantic
        ValidationError inside FakeStructuredChatModel. extract_facts catches
        this and returns the default schema instance.
        """
        # Arrange — score/label fields don't match FactSchema
        judge = FakeStructuredChatModel(responses=['{"score": 0.5, "label": "high"}'])

        # Act
        result = extract_facts(judge, "Some response text.", FactSchema)

        # Assert — fallback to default instance
        assert result == FactSchema()

    def test_extract_facts_schema_with_required_field_raises_type_error(self) -> None:
        """extract_facts should raise TypeError when schema() cannot be instantiated.

        A schema with required fields (no defaults) cannot produce a fallback
        instance, so extract_facts raises TypeError before doing any other work.
        """
        # Arrange
        judge = FakeStructuredChatModel(responses=['{"name": "Eve"}'])

        # Act / Assert
        with pytest.raises(TypeError, match="RequiredFieldSchema"):
            extract_facts(judge, "Eve was mentioned.", RequiredFieldSchema)
