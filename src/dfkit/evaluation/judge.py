"""Judge prompt template and structured fact extraction for the evaluation pipeline.

This module provides the default `ChatPromptTemplate` used to instruct a judge
LLM to extract structured facts from an agent's free-text response, and the
`extract_facts` function that wires the prompt to a structured-output LLM and
invokes the chain.

The prompt is intentionally split into a system and human message to give the
model a clearer signal — "here's your role" vs "here's the task".

The judge is intentionally conservative: it extracts only facts that are
explicitly stated or strongly implied by the response, leaving unmentioned
fields as `None`.  This avoids inflating extraction accuracy with inferred
values that the agent did not actually produce.
"""

from typing import Final

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

__all__ = [
    "DEFAULT_JUDGE_PROMPT",
    "extract_facts",
]

# region Public interface

DEFAULT_JUDGE_PROMPT: Final[ChatPromptTemplate] = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an evaluation judge. Your task is to extract specific factual claims"
            " from the following agent response. Extract ONLY the facts that are explicitly"
            " stated or strongly implied. If a fact is not present in the response, use None"
            " for that field. Do not infer or assume values that are not clearly stated."
        ),
    ),
    (
        "human",
        "Agent response to evaluate:\n\n{agent_response}\n\nExtract the facts according to the required schema.",
    ),
])


def extract_facts[R: BaseModel](
    judge: BaseChatModel,
    agent_response: str,
    schema: type[R],
    *,
    prompt: ChatPromptTemplate | None = None,
    prompt_variables: dict[str, str] | None = None,
) -> R:
    """Extract structured facts from an agent response using a judge LLM.

    Chains the given (or default) prompt with a structured-output version of
    `judge` configured to populate `schema`, then invokes the chain with
    the agent's response text.

    All fields on `schema` must have default values (typically `None`) so that
    a default instance can be returned when extraction fails.

    Args:
        judge (BaseChatModel): The LLM used to perform structured extraction.
        agent_response (str): Raw free-text response produced by the agent
            under evaluation.
        schema (type[R]): Pydantic model class defining the facts to extract.
            The judge LLM is constrained to return an instance of this type.
            All fields must have default values.
        prompt (ChatPromptTemplate | None): Optional custom prompt template.
            Must accept an `agent_response` input variable.  When `None`,
            `DEFAULT_JUDGE_PROMPT` is used.
        prompt_variables (dict[str, str] | None): Optional additional template
            variables required by a custom `prompt` beyond `agent_response`.
            Values for `agent_response` in this dict are ignored — the explicit
            `agent_response` parameter always takes precedence.

    Returns:
        R: A populated instance of `schema` containing the facts extracted
            from `agent_response`, or a default instance (`schema()`) if the
            LLM output fails Pydantic validation against `schema`.

    Raises:
        ValueError: If `schema` cannot be instantiated with no arguments
            (one or more fields are missing default values), or if `prompt`
            does not contain an `agent_response` input variable.
    """
    try:
        default_result = schema()
    except ValidationError as exc:
        raise ValueError(
            f"{schema.__name__} cannot be instantiated with no arguments. All schema fields must have default values."
        ) from exc

    prompt = prompt or DEFAULT_JUDGE_PROMPT
    if "agent_response" not in prompt.input_variables:
        raise ValueError(
            f"Custom prompt must contain an 'agent_response' input variable, got: {prompt.input_variables!r}"
        )

    invoke_vars = {**(prompt_variables or {}), "agent_response": agent_response}
    chain = prompt | judge.with_structured_output(schema)
    try:
        result = chain.invoke(invoke_vars)
        if not isinstance(result, schema):
            return default_result
    except ValidationError:
        return default_result

    return result


# endregion
