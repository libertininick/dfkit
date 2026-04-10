"""Tests for the evaluate_agent function in dfkit.evaluation.runner.

Covers the full evaluation loop: building agent input, invoking the agent,
extracting structured facts via a judge LLM, comparing fields, and returning
EvalResult instances. All tests use real LangChain fakes (RunnableLambda,
FakeStructuredChatModel); no mocks or monkeypatching of internal code are used.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from pytest_check import check

from dfkit.evaluation.models import EvalCase
from dfkit.evaluation.runner import evaluate_agent
from tests.test_evaluation.fake_chat_models import FakeStructuredChatModel

# region Module-level test schemas


class SimpleResult(BaseModel):
    """Minimal Pydantic schema used as the extraction target in runner tests.

    All fields have defaults so that schema() succeeds and error-path fallback
    works without raising.

    Attributes:
        name (str | None): A name value extracted from the agent response.
        score (float | None): A numeric score extracted from the agent response.
    """

    name: str | None = None
    score: float | None = None


class DetailedResult(BaseModel):
    """Schema with four-plus fields and non-scalar types for runner integration tests.

    All fields have None defaults so that extract_facts can create a default
    instance on fallback — a runner constraint. Tests populate every field with
    non-None values to exercise required-field semantics and non-scalar dispatch.

    Attributes:
        label (str | None): A string label (exact comparison).
        score (float | None): A numeric score (numeric_tolerance comparison).
        tags (list[str] | None): A list of string tags (exact_match comparison).
        metrics (list[float] | None): A list of numeric values (numeric_sequence comparison).
        metadata (dict[str, Any] | None): An optional dictionary (dict_comparison).
    """

    label: str | None = None
    score: float | None = None
    tags: list[str] | None = None
    metrics: list[float] | None = None
    metadata: dict[str, Any] | None = None


# endregion


# region Tests


class TestEvaluateAgent:
    """Verify evaluate_agent scoring, error handling, and multi-case aggregation."""

    def test_evaluate_agent_single_case_all_fields_match_returns_score_one(self) -> None:
        """evaluate_agent with one matching case should return score 1.0.

        The fake agent returns a response string; the fake judge returns JSON
        that exactly matches the expected result, so all field comparisons must
        be matched=True and the overall score must be 1.0.
        """
        # Arrange
        expected = SimpleResult(name="Alice", score=0.9)
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="Alice scored 0.9")]})
        judge = FakeStructuredChatModel(responses=[expected.model_dump_json()])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="case-1",
            input_messages=[HumanMessage(content="Who scored highest?")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(1.0)
        with check:
            assert result.passed is True
        with check:
            assert all(fc.matched for fc in result.field_comparisons)

    def test_evaluate_agent_partial_match_returns_fractional_score(self) -> None:
        """evaluate_agent should return fractional score when only some fields match.

        The judge returns JSON where name matches the expected value but score
        does not. With two fields and one match, score must be 0.5.
        """
        # Arrange
        expected = SimpleResult(name="Bob", score=5.0)
        extracted = SimpleResult(name="Bob", score=99.0)
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="Bob's score was different")]})
        judge = FakeStructuredChatModel(responses=[extracted.model_dump_json()])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="case-partial",
            input_messages=[HumanMessage(content="What did Bob score?")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(0.5)
        with check:
            assert result.passed is False
        with check:
            name_fc = next(fc for fc in result.field_comparisons if fc.field_name == "name")
            assert name_fc.matched is True
        with check:
            score_fc = next(fc for fc in result.field_comparisons if fc.field_name == "score")
            assert score_fc.matched is False

    def test_evaluate_agent_multiple_cases_returns_result_per_case(self) -> None:
        """evaluate_agent should return one EvalResult per input EvalCase, in order.

        Two cases are supplied. The judge responses are consumed in order, one per
        case. The returned list must have the same length and order as the input.
        """
        # Arrange
        expected_a = SimpleResult(name="Carol", score=1.0)
        expected_b = SimpleResult(name="Dave", score=2.0)
        agent = RunnableLambda(lambda inp: {"messages": [AIMessage(content=inp["messages"][0].content)]})
        judge = FakeStructuredChatModel(
            responses=[
                expected_a.model_dump_json(),
                expected_b.model_dump_json(),
            ]
        )
        cases: list[EvalCase[HumanMessage, SimpleResult]] = [
            EvalCase(
                case_id="a",
                input_messages=[HumanMessage(content="Carol info")],
                expected_result=expected_a,
            ),
            EvalCase(
                case_id="b",
                input_messages=[HumanMessage(content="Dave info")],
                expected_result=expected_b,
            ),
        ]

        # Act
        results = evaluate_agent(agent, cases, judge=judge)

        # Assert
        assert len(results) == 2
        with check:
            assert results[0].eval_case.case_id == "a"
        with check:
            assert results[0].score == pytest.approx(1.0)
        with check:
            assert results[1].eval_case.case_id == "b"
        with check:
            assert results[1].score == pytest.approx(1.0)

    def test_evaluate_agent_custom_tolerances_numeric_comparison_uses_them(self) -> None:
        """evaluate_agent should forward rtol/atol to compare_fields for numeric fields.

        With expected score=1.0 and extracted score=1.05, the default RTOL (1e-5)
        would produce matched=False. With rtol=0.1 (10%), the values are within
        tolerance and the comparison must match.
        """
        # Arrange
        expected = SimpleResult(name="Eve", score=1.0)
        extracted = SimpleResult(name="Eve", score=1.05)
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="Eve scored about 1.0")]})
        judge = FakeStructuredChatModel(responses=[extracted.model_dump_json()])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="tolerance",
            input_messages=[HumanMessage(content="What did Eve score?")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge, rtol=0.1, atol=0.0)

        # Assert
        assert len(results) == 1
        result = results[0]
        score_fc = next(fc for fc in result.field_comparisons if fc.field_name == "score")
        assert score_fc.matched is True

    def test_evaluate_agent_custom_prompt_forwards_to_extract_facts(self) -> None:
        """evaluate_agent should accept a custom judge_prompt and produce valid results.

        Passing a ChatPromptTemplate with {agent_response} variable must not raise.
        The result must be a valid EvalResult, verifying the prompt was forwarded
        to extract_facts without error.
        """
        # Arrange
        expected = SimpleResult(name="Frank", score=None)
        custom_prompt = ChatPromptTemplate.from_messages([
            ("human", "Custom extraction request: {agent_response}"),
        ])
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="Frank was mentioned")]})
        judge = FakeStructuredChatModel(responses=[expected.model_dump_json()])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="custom-prompt",
            input_messages=[HumanMessage(content="Who was mentioned?")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge, judge_prompt=custom_prompt)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.agent_response == "Frank was mentioned"
        with check:
            assert result.extracted_result == expected

    def test_evaluate_agent_prompt_variables_forwarded_to_extract_facts(self) -> None:
        """evaluate_agent should forward prompt_variables to extract_facts successfully.

        A custom prompt with {agent_response} and {context} variables requires
        prompt_variables={"context": "..."} to render without error. The result
        must be a valid EvalResult, confirming the variables were forwarded.
        """
        # Arrange
        expected = SimpleResult(name="Grace", score=None)
        custom_prompt = ChatPromptTemplate.from_messages([
            ("human", "Context: {context}\n\nAgent response: {agent_response}\n\nExtract facts."),
        ])
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="Grace was mentioned")]})
        judge = FakeStructuredChatModel(responses=[expected.model_dump_json()])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="prompt-vars",
            input_messages=[HumanMessage(content="Who was mentioned?")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(
            agent,
            [eval_case],
            judge=judge,
            judge_prompt=custom_prompt,
            prompt_variables={"context": "test context"},
        )

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.extracted_result == expected
        with check:
            assert result.agent_response == "Grace was mentioned"

    def test_evaluate_agent_agent_error_returns_error_result(self) -> None:
        """evaluate_agent should handle agent invocation errors gracefully.

        When the agent raises an exception, evaluate_agent must still return one
        EvalResult for the case. All field comparisons must have matched=False
        and strategy="error", preventing a partial run from raising. The
        agent_response field must contain the error message string.
        """
        # Arrange
        expected = SimpleResult(name="Hank", score=3.0)

        def _failing_agent(_: object) -> dict:
            raise RuntimeError("agent exploded")

        agent = RunnableLambda(_failing_agent)
        judge = FakeStructuredChatModel(responses=['{"name": null, "score": null}'])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="error-case",
            input_messages=[HumanMessage(content="Any question")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(0.0)
        with check:
            assert result.passed is False
        with check:
            assert all(fc.matched is False for fc in result.field_comparisons)
        with check:
            assert all(fc.strategy == "error" for fc in result.field_comparisons)
        with check:
            assert result.agent_response == "agent error: agent exploded"

    def test_evaluate_agent_no_ai_messages_returns_error_result(self) -> None:
        """evaluate_agent should handle agents that return no AIMessage gracefully.

        When the agent returns a messages list containing only non-AI messages
        (e.g. a HumanMessage), evaluate_agent must still return one EvalResult for
        the case. All field comparisons must have matched=False and strategy="error",
        and the overall score must be 0.0 with passed=False.
        """
        # Arrange
        expected = SimpleResult(name="Ivan", score=7.0)
        agent = RunnableLambda(lambda _: {"messages": [HumanMessage(content="oops")]})
        judge = FakeStructuredChatModel(responses=['{"name": null, "score": null}'])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="no-ai-messages",
            input_messages=[HumanMessage(content="Any question")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(0.0)
        with check:
            assert result.passed is False
        with check:
            assert all(fc.matched is False for fc in result.field_comparisons)
        with check:
            assert all(fc.strategy == "error" for fc in result.field_comparisons)
        with check:
            assert result.agent_response == "agent error: Agent response contained no AI messages"

    def test_evaluate_agent_extraction_error_returns_error_result_with_agent_response(self) -> None:
        """evaluate_agent should handle fact-extraction errors and preserve the agent response.

        When the agent succeeds but the judge's structured-output chain raises a
        RuntimeError, evaluate_agent must still return one EvalResult for the case.
        All field comparisons must have matched=False and strategy="error". The
        agent_response field must contain the original agent answer (not the error
        message), because the runner captures the agent answer before extraction runs.
        """
        # Arrange
        expected = SimpleResult(name="Judy", score=2.5)

        class _FailingJudge(FakeStructuredChatModel):
            """FakeStructuredChatModel whose with_structured_output always raises."""

            def with_structured_output(  # type: ignore[override]
                self,
                _schema: dict | type,
                *,
                _include_raw: bool = False,
                **_kwargs: object,
            ) -> RunnableLambda:
                """Return a runnable that raises RuntimeError on invoke.

                Args:
                    _schema (dict | type): Ignored; accepted for interface compatibility.
                    _include_raw (bool): Ignored; accepted for interface compatibility.
                    **_kwargs (object): Ignored; accepted for interface compatibility.

                Returns:
                    RunnableLambda: A runnable that always raises RuntimeError.
                """

                def _raise(_: object) -> None:
                    raise RuntimeError("extraction broke")

                return RunnableLambda(_raise)

        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="Judy scored 2.5")]})
        judge = _FailingJudge(responses=[])
        eval_case: EvalCase[HumanMessage, SimpleResult] = EvalCase(
            case_id="extraction-error",
            input_messages=[HumanMessage(content="What did Judy score?")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(0.0)
        with check:
            assert result.passed is False
        with check:
            assert all(fc.matched is False for fc in result.field_comparisons)
        with check:
            assert all(fc.strategy == "error" for fc in result.field_comparisons)
        with check:
            assert result.agent_response == "Judy scored 2.5"

    def test_evaluate_agent_empty_cases_returns_empty_list(self) -> None:
        """evaluate_agent with an empty case list should return an empty list.

        No agent invocation or judge call should occur, and the result must be
        an empty list without raising.
        """
        # Arrange
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="unused")]})
        judge = FakeStructuredChatModel(responses=[])

        # Act
        results = evaluate_agent(agent, [], judge=judge)

        # Assert
        assert results == []

    def test_evaluate_agent_required_fields_all_match_returns_score_one(self) -> None:
        """evaluate_agent with all required DetailedResult fields matching returns score 1.0.

        The fake judge returns JSON that exactly matches the expected DetailedResult.
        All five field comparisons (label, score, tags, metrics, metadata) must be
        matched=True and the overall score must be 1.0. This validates that required-
        field schemas (no defaults) work end-to-end through the runner.
        """
        # Arrange
        expected = DetailedResult(
            label="alpha",
            score=0.95,
            tags=["fast", "accurate"],
            metrics=[1.0, 2.0, 3.0],
            metadata=None,
        )
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="alpha scored 0.95")]})
        judge = FakeStructuredChatModel(responses=[expected.model_dump_json()])
        eval_case: EvalCase[HumanMessage, DetailedResult] = EvalCase(
            case_id="detailed-all-match",
            input_messages=[HumanMessage(content="Describe the result.")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(1.0)
        with check:
            assert result.passed is True
        with check:
            assert all(fc.matched for fc in result.field_comparisons)

    def test_evaluate_agent_list_and_dict_fields_partial_match(self) -> None:
        """evaluate_agent returns fractional score when list[float] field mismatches.

        The judge returns a DetailedResult where label, score, tags, and metadata
        (both None) match but metrics differs ([1.0, 2.0] vs [1.0, 9.0]). With 5
        fields and 4 matched (label, score, tags, metadata via none_comparison),
        the score must be 0.8. This validates that compare_numeric_sequence is
        dispatched for list[float] and compare_exact for list[str] through the runner.
        """
        # Arrange
        expected = DetailedResult(
            label="beta",
            score=0.7,
            tags=["quick", "cheap"],
            metrics=[1.0, 2.0],
            metadata=None,
        )
        extracted = DetailedResult(
            label="beta",
            score=0.7,
            tags=["quick", "cheap"],
            metrics=[1.0, 9.0],
            metadata=None,
        )
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="beta scored 0.7")]})
        judge = FakeStructuredChatModel(responses=[extracted.model_dump_json()])
        eval_case: EvalCase[HumanMessage, DetailedResult] = EvalCase(
            case_id="detailed-partial-match",
            input_messages=[HumanMessage(content="Describe the beta result.")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        with check:
            assert result.score == pytest.approx(0.8)
        with check:
            assert result.passed is False
        metrics_fc = next(fc for fc in result.field_comparisons if fc.field_name == "metrics")
        with check:
            assert metrics_fc.matched is False
        with check:
            assert metrics_fc.strategy == "numeric_sequence_tolerance"
        tags_fc = next(fc for fc in result.field_comparisons if fc.field_name == "tags")
        with check:
            assert tags_fc.matched is True
        with check:
            assert tags_fc.strategy == "exact_match"
        metadata_fc = next(fc for fc in result.field_comparisons if fc.field_name == "metadata")
        with check:
            assert metadata_fc.strategy == "none_comparison"

    def test_evaluate_agent_dict_field_mismatch(self) -> None:
        """evaluate_agent marks metadata field unmatched when dict values differ.

        The expected DetailedResult has metadata={"key": "val"} and the extracted
        result has metadata={"key": "other"}. The metadata field comparison must use
        strategy "dict_comparison" and have matched=False, confirming that compare_dict
        is dispatched through the runner for dict[str, Any] fields.
        """
        # Arrange
        expected = DetailedResult(
            label="gamma",
            score=0.5,
            tags=["slow"],
            metrics=[0.1],
            metadata={"key": "val"},
        )
        extracted = DetailedResult(
            label="gamma",
            score=0.5,
            tags=["slow"],
            metrics=[0.1],
            metadata={"key": "other"},
        )
        agent = RunnableLambda(lambda _: {"messages": [AIMessage(content="gamma scored 0.5")]})
        judge = FakeStructuredChatModel(responses=[extracted.model_dump_json()])
        eval_case: EvalCase[HumanMessage, DetailedResult] = EvalCase(
            case_id="detailed-dict-mismatch",
            input_messages=[HumanMessage(content="Describe the gamma result.")],
            expected_result=expected,
        )

        # Act
        results = evaluate_agent(agent, [eval_case], judge=judge)

        # Assert
        assert len(results) == 1
        result = results[0]
        metadata_fc = next(fc for fc in result.field_comparisons if fc.field_name == "metadata")
        with check:
            assert metadata_fc.matched is False
        with check:
            assert metadata_fc.strategy == "dict_comparison"


# endregion
