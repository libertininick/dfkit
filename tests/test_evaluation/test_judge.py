"""Tests for the judge prompt template and extract_facts function.

Covers DEFAULT_JUDGE_PROMPT formatting and the extract_facts function's
schema binding, chain invocation, custom-prompt routing, and error handling.
All tests use FakeStructuredChatModel; no real network calls are made.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.judge import DEFAULT_JUDGE_PROMPT, extract_facts
from tests.test_evaluation.fake_chat_models import FakeStructuredChatModel, NonSchemaModel


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


# region Tests
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
            # The system prompt must instruct the model to leave missing facts
            # unset (None/null); check for the concept rather than a literal word.
            assert any(
                keyword in system_content.lower()
                for keyword in ("none", "null", "not present", "leave", "unset", "missing")
            )

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
        ("schema_class", "expected", "agent_response", "prompt", "prompt_variables"),
        [
            pytest.param(
                FactSchema,
                FactSchema(name="Bob"),
                "Bob was mentioned but no age given.",
                None,
                None,
                id="fact-schema-none-optional-field",
            ),
            pytest.param(
                AltFactSchema,
                AltFactSchema(score=0.72, label="standard"),
                "Score was 0.72; category is standard.",
                None,
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
                None,
                id="fact-schema-custom-prompt",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Dana"),
                "Dana was mentioned.",
                None,
                None,
                id="fact-schema-prompt-none-uses-default",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name=None),
                "",
                None,
                None,
                id="fact-schema-empty-agent-response",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Maria"),
                "Name: Maria; score=9/10 (top-tier). Tags: <urgent>, [review], {pending}. Cost: $4.99 & \u20ac3.50.",
                None,
                None,
                id="fact-schema-special-chars-unicode",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Ivan", age=38),
                (
                    "Ivan is a 38-year-old software engineer with over a decade of experience "
                    "in distributed systems. He joined the team in January and has already "
                    "delivered three major features ahead of schedule. His colleagues describe "
                    "him as thorough and dependable, and he has received two commendations from "
                    "senior leadership this quarter alone."
                ),
                None,
                None,
                id="fact-schema-long-multi-sentence-response",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Julia", age=29),
                (
                    "## Summary\n\n"
                    "The agent identified **Julia** (age: 29) as the primary contact.\n\n"
                    "### Details\n\n"
                    "```json\n"
                    '{"name": "Julia", "age": 29, "role": "lead"}\n'
                    "```\n\n"
                    "Additional context: she is fluent in three languages."
                ),
                None,
                None,
                id="fact-schema-markdown-with-embedded-json",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Frank", age=40),
                "Frank is 40 years old.",
                ChatPromptTemplate.from_messages([
                    ("human", "Context: {context}\n\nAgent response: {agent_response}\n\nExtract facts."),
                ]),
                {"context": "Medical intake form"},
                id="fact-schema-custom-prompt-extra-variables",
            ),
            pytest.param(
                FactSchema,
                FactSchema(name="Hank", age=55),
                "Hank is 55 years old.",
                ChatPromptTemplate.from_messages([
                    ("human", "Context: {context}\n\nAgent response: {agent_response}\n\nExtract facts."),
                ]),
                {"context": "HR record", "agent_response": "stale value — should be ignored"},
                id="fact-schema-agent-response-precedence",
            ),
        ],
    )
    def test_extract_facts_returns_populated_schema(
        self,
        schema_class: type[BaseModel],
        expected: BaseModel,
        agent_response: str,
        prompt: ChatPromptTemplate | None,
        prompt_variables: dict[str, str] | None,
    ) -> None:
        """extract_facts should build the LLM chain and return a populated schema instance.

        Covers schema variation, optional-field handling, custom prompts, prompt=None fallback,
        empty agent responses, unicode/special characters, long multi-sentence text, structured
        content like markdown with embedded JSON, extra prompt variables supplied via
        prompt_variables, and agent_response precedence over any agent_response key in
        prompt_variables. Each parametrize case isolates one behavioral dimension.

        Args:
            schema_class (type[BaseModel]): The Pydantic model class passed as the extraction target.
            expected (BaseModel): The pre-built model instance used both as LLM response source and assertion target.
            agent_response (str): The agent response string forwarded to `extract_facts`.
            prompt (ChatPromptTemplate | None): Custom prompt passed to `extract_facts`, or `None` to use the default.
            prompt_variables (dict[str, str] | None): Extra template variables forwarded to `extract_facts`, or `None`.
        """
        # Arrange
        judge = FakeStructuredChatModel(responses=[expected.model_dump_json()])

        # Act
        result = extract_facts(judge, agent_response, schema_class, prompt=prompt, prompt_variables=prompt_variables)

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

        A JSON object whose field values cannot be coerced to FactSchema types
        triggers a pydantic ValidationError inside FakeStructuredChatModel.
        extract_facts catches this and returns the default schema instance.
        """
        # Arrange — list cannot be coerced to str | None, triggering ValidationError
        judge = FakeStructuredChatModel(responses=['{"name": ["not", "a", "string"]}'])

        # Act
        result = extract_facts(judge, "Some response text.", FactSchema)

        # Assert — fallback to default instance
        assert result == FactSchema()

    def test_extract_facts_schema_with_required_field_raises_value_error(self) -> None:
        """extract_facts should raise ValueError when schema() cannot be instantiated.

        A schema with required fields (no defaults) cannot produce a fallback
        instance, so extract_facts raises ValueError before doing any other work.
        """
        # Arrange
        judge = FakeStructuredChatModel(responses=['{"name": "Eve"}'])

        # Act / Assert
        with pytest.raises(ValueError, match="RequiredFieldSchema"):
            extract_facts(judge, "Eve was mentioned.", RequiredFieldSchema)

    def test_extract_facts_custom_prompt_extra_variables_missing_prompt_variables_raises(self) -> None:
        """extract_facts should raise when extra prompt variables are not provided.

        When a custom prompt references a variable beyond {agent_response} (e.g. {context})
        and prompt_variables is not supplied, the template cannot be rendered. The function
        must propagate the resulting KeyError (or equivalent) so the caller learns that
        prompt_variables is required.
        """
        # Arrange
        custom_prompt = ChatPromptTemplate.from_messages([
            ("human", "Context: {context}\n\nAgent response: {agent_response}\n\nExtract facts."),
        ])
        judge = FakeStructuredChatModel(responses=['{"name": "Grace", "age": 28}'])

        # Act / Assert — missing {context} must surface an error, not silently return defaults
        with pytest.raises((KeyError, Exception)):
            extract_facts(judge, "Grace is 28 years old.", FactSchema, prompt=custom_prompt)

    def test_extract_facts_non_schema_result_returns_default(self) -> None:
        """extract_facts should return schema() when with_structured_output yields a non-schema type.

        This test covers the isinstance fallback branch (lines 100-101 of judge.py).
        When chain.invoke returns a value that is not an instance of the target schema
        (here a raw dict), extract_facts must return the default schema instance
        rather than forwarding the mistyped result to the caller.
        """
        # Arrange — a fake model whose with_structured_output ignores the schema
        # and returns a plain dict, bypassing model_validate_json entirely.
        judge = NonSchemaModel(responses=["unused"])

        # Act
        result = extract_facts(judge, "Alice is 30 years old.", FactSchema)

        # Assert
        assert result == FactSchema()


# endregion
