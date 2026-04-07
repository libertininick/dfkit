"""Core data models for the evaluation pipeline.

This module defines the Pydantic models that represent the inputs and outputs
of an evaluation run: the ground-truth case (`EvalCase`), a per-field
comparison record (`FieldComparison`), and the aggregated result produced by
comparing an agent's structured output against the ground truth (`EvalResult`).

These models are generic over the message type `M` (a `BaseMessage` subclass)
and the result schema `R` (a Pydantic `BaseModel` subclass), so the same
infrastructure can be reused across different agent tasks and output schemas.
"""

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

__all__ = [
    "EvalCase",
    "EvalResult",
    "FieldComparison",
]

# region Public models


class EvalCase[M: BaseMessage, R: BaseModel](BaseModel, arbitrary_types_allowed=True):
    """A single evaluation case pairing input messages with an expected result.

    `EvalCase` is the ground-truth record for one evaluation scenario.  It
    stores the messages to replay to the agent alongside a fully populated
    Pydantic model that represents the correct structured output.

    The model is generic over two type parameters:

    - `M` — the `BaseMessage` subclass used for the conversation (e.g.
      `HumanMessage`, `AIMessage`).
    - `R` — the Pydantic `BaseModel` subclass that defines the expected
      output schema.  The judge LLM extracts a value of this type from the
      agent's free-text response, and each field is then compared against
      `expected_result` using a strategy appropriate for its type.

    `arbitrary_types_allowed` is enabled because `BaseMessage` is not
    itself a Pydantic model.

    Attributes:
        case_id (str | None): Optional identifier for this eval case.
        input_messages (list[M]): Ordered list of messages to send to the agent.
        expected_result (R): Ground-truth structured result.
        metadata (dict[str, Any]): Arbitrary key-value pairs attached to the case
            (e.g. dataset split, difficulty tier, source reference).
    """

    case_id: str | None = Field(default=None, description="Optional identifier for this eval case.")
    input_messages: list[M] = Field(description="Messages to send to the agent.")
    expected_result: R = Field(description="Ground truth as a populated Pydantic model instance.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for this eval case.",
    )


class FieldComparison(BaseModel):
    """The outcome of comparing one field between the expected and extracted results.

    Attributes:
        field_name (str): Name of the compared field.
        expected (Any): Expected value from the ground truth.
        extracted (Any): Value extracted by the judge LLM.
        matched (bool): Whether the extracted value matches the expected value.
        strategy (str): Comparison strategy used (e.g. `"numeric_tolerance"`,
            `"exact_match"`, `"set_overlap"`).
        details (dict[str, Any]): Strategy-specific comparison details such as
            the tolerance applied or the overlap ratio computed.
    """

    field_name: str = Field(description="Name of the compared field.")
    expected: Any = Field(description="Expected value from ground truth.")
    extracted: Any = Field(description="Value extracted by judge LLM.")
    matched: bool = Field(description="Whether the extracted value matches expected.")
    strategy: str = Field(
        description="Comparison strategy used (e.g., 'numeric_tolerance', 'exact_match', 'set_overlap').",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific comparison details (e.g., tolerance, overlap_ratio).",
    )


class EvalResult[M: BaseMessage, R: BaseModel](BaseModel, arbitrary_types_allowed=True):
    """The outcome of running an agent against a single `EvalCase`.

    `EvalResult` aggregates the raw agent response, the structured result
    extracted by the judge LLM, and a field-level comparison against the
    ground truth stored in `eval_case`.

    `arbitrary_types_allowed` is enabled because the embedded `EvalCase`
    contains `BaseMessage` instances.

    Attributes:
        eval_case (EvalCase[M, R]): The evaluation case that produced this result.
        extracted_result (R): The full structured result extracted by the judge LLM.
        field_comparisons (list[FieldComparison]): Field-level comparison results.
        agent_response (str): Raw text response from the agent.
    """

    eval_case: EvalCase[M, R] = Field(description="The evaluation case that produced this result.")
    extracted_result: R = Field(description="The full structured result extracted by the judge LLM.")
    field_comparisons: list[FieldComparison] = Field(description="Field-level comparison results.")
    agent_response: str = Field(description="Raw text response from the agent.")

    @property
    def score(self) -> float:
        """Fraction of fields that matched between expected and extracted results.

        Returns:
            float: A value in `[0.0, 1.0]`.  Returns `1.0` when there are
                no field comparisons (vacuously true).
        """
        if not self.field_comparisons:
            return 1.0
        return sum(fc.matched for fc in self.field_comparisons) / len(self.field_comparisons)

    @property
    def passed(self) -> bool:
        """Whether every field comparison matched.

        Returns:
            bool: `True` if all field comparisons matched, `False` otherwise.
        """
        return all(fc.matched for fc in self.field_comparisons)

    @property
    def failed_fields(self) -> list[FieldComparison]:
        """Field comparisons where the extracted value did not match the expected value.

        Returns:
            list[FieldComparison]: Comparisons with `matched` set to `False`.
        """
        return [fc for fc in self.field_comparisons if not fc.matched]


# endregion
