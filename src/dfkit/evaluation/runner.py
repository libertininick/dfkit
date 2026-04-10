"""Agent evaluation runner for the dfkit evaluation pipeline.

This module implements `evaluate_agent`, which drives a full evaluation loop:
replaying input messages to an agent, extracting structured facts from the
response via a judge LLM, and comparing the extracted facts against ground-truth
`EvalCase` data using `compare_fields`.

Each evaluation case produces an `EvalResult`. Errors during agent invocation or
fact extraction are caught and surfaced as `EvalResult` instances with all fields
marked unmatched, so a partial run still returns a result for every case.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, filter_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from loguru import logger
from pydantic import BaseModel, ValidationError

from dfkit.evaluation.comparison import ATOL, RTOL, compare_fields
from dfkit.evaluation.judge import extract_facts
from dfkit.evaluation.models import EvalCase, EvalResult, FieldComparison

__all__ = [
    "evaluate_agent",
]

# region Public interface


def evaluate_agent[M: BaseMessage, R: BaseModel](
    agent: Runnable,
    eval_cases: Sequence[EvalCase[M, R]],
    *,
    judge: BaseChatModel,
    judge_prompt: ChatPromptTemplate | None = None,
    prompt_variables: dict[str, str] | None = None,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> list[EvalResult[M, R]]:
    """Evaluate an agent against a sequence of eval cases.

    For each `EvalCase`, this function invokes the agent with its input
    messages, extracts structured facts from the response via `extract_facts`,
    compares them against ground-truth expectations via `compare_fields`, and
    wraps everything into an `EvalResult`.

    When agent invocation or fact extraction raises an exception, an `EvalResult`
    is still returned for that case with all fields marked `matched=False` and
    strategy `"error"`, so the caller always receives one result per input case.

    Args:
        agent (Runnable): The LangChain runnable (agent) to evaluate.
        eval_cases (Sequence[EvalCase[M, R]]): Ordered sequence of evaluation
            cases to run.
        judge (BaseChatModel): LLM used by `extract_facts` to parse structured
            facts from the agent's free-text response.
        judge_prompt (ChatPromptTemplate | None): Optional custom prompt for the
            judge.  Forwarded directly to `extract_facts`.
        prompt_variables (dict[str, str] | None): Additional template variables
            for the judge prompt.  Forwarded directly to `extract_facts`.
        rtol (float): Relative tolerance forwarded to `compare_fields`.
        atol (float): Absolute tolerance forwarded to `compare_fields`.

    Returns:
        list[EvalResult[M, R]]: One `EvalResult` per input `EvalCase`, in the
            same order as `eval_cases`.

    Raises:
        ValueError: If the agent response contains no AI messages. This is
            caught internally and surfaced as an error `EvalResult`.
    """
    results: list[EvalResult[M, R]] = []
    for index, eval_case in enumerate(eval_cases):
        case_label = eval_case.case_id or f"case[{index}]"
        logger.info("Evaluating {}", case_label)

        agent_input = {"messages": eval_case.input_messages}
        schema = type(eval_case.expected_result)

        try:
            agent_response_raw = agent.invoke(agent_input)
            messages = agent_response_raw.get("messages", [])
            ai_messages = filter_messages(messages, include_types="ai")
            if not ai_messages:
                raise ValueError("Agent response contained no AI messages")
            agent_answer = ai_messages[-1].content
            logger.debug("{}: agent response length={}", case_label, len(agent_answer))
        except (ValueError, KeyError, TypeError, RuntimeError, AttributeError, ValidationError) as exc:
            logger.warning("{}: agent invocation failed: {}", case_label, exc)
            result = _error_result(eval_case, schema, error=exc, stage="agent")
            results.append(result)
            continue

        try:
            extracted = extract_facts(
                judge,
                agent_answer,
                schema,
                prompt=judge_prompt,
                prompt_variables=prompt_variables,
            )
            logger.debug("{}: extraction result={}", case_label, extracted)
        except (ValueError, RuntimeError) as exc:
            logger.warning("{}: fact extraction failed: {}", case_label, exc)
            result = _error_result(eval_case, schema, error=exc, stage="extraction", agent_response=agent_answer)
            results.append(result)
            continue

        field_comparisons = compare_fields(eval_case.expected_result, extracted, rtol=rtol, atol=atol)
        result = EvalResult(
            eval_case=eval_case,
            extracted_result=extracted,
            field_comparisons=field_comparisons,
            agent_response=agent_answer,
        )
        logger.info("{}: score={:.2f}", case_label, result.score)
        results.append(result)

    return results


# endregion

# region Private helpers


def _error_result[M: BaseMessage, R: BaseModel](
    eval_case: EvalCase[M, R],
    schema: type[R],
    *,
    error: Exception,
    stage: Literal["agent", "extraction"],
    agent_response: str | None = None,
) -> EvalResult[M, R]:
    """Build an EvalResult representing a failed evaluation case.

    Args:
        eval_case (EvalCase[M, R]): The evaluation case that failed.
        schema (type[R]): Pydantic model class used to create a default extracted
            result.
        error (Exception): The exception that caused the failure.
        stage (Literal["agent", "extraction"]): Human-readable label identifying where the error occurred.
        agent_response (str | None): Raw agent response text, if available.

    Returns:
        EvalResult[M, R]: An EvalResult with all fields marked `matched=False`
            and strategy `"error"`.
    """
    error_message = f"{stage} error: {error}"
    field_comparisons = [
        FieldComparison(
            field_name=field_name,
            expected=getattr(eval_case.expected_result, field_name),
            extracted=None,
            matched=False,
            strategy="error",
            details={"error": error_message},
        )
        for field_name in schema.model_fields
    ]

    return EvalResult(
        eval_case=eval_case,
        extracted_result=schema(),
        field_comparisons=field_comparisons,
        agent_response=agent_response or error_message,
    )


# endregion
