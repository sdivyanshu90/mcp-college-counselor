from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from client.client import (
    FinalRecommendation,
    format_recommendation_output,
    parse_final_recommendation,
    probability_to_label,
    run_conversation,
)
from config import settings
from llm_provider import ChatCompletionResult, OpenAICompatibleClient, ToolUseRequest


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list[object]


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def create(self, **kwargs: Any) -> FakeResponse:
        _ = kwargs
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


class FakeAnthropicClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.messages = FakeMessages(responses)


class FakeOpenAIClient:
    def __init__(self, responses: list[ChatCompletionResult]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def create_chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        _ = kwargs
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


class FakeMcpResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = [FakeTextBlock(text=json.dumps(payload))]


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> FakeMcpResult:
        payload = arguments or {}
        self.calls.append((name, payload))
        if name == "evaluate_chances":
            return FakeMcpResult(
                {
                    "probability": 0.71,
                    "component_scores": {"gpa": 1.0, "test": 1.0, "ap_classes": 0.8, "lor": 0.67},
                    "university_name": "MIT EECS",
                    "confidence": "high",
                }
            )
        return FakeMcpResult(
            {
                "gaps": [
                    "AP classes: your 4 is 1.00 points/points below MIT EECS minimum of 5.",
                    "Letters of recommendation: your 2 is 1.00 points/points below MIT EECS minimum of 3.",
                ],
                "upcoming_deadlines": [],
                "is_competitive": True,
                "university_name": "MIT EECS",
            }
        )


@pytest.fixture(autouse=True)
def default_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "model_provider", "anthropic")


@pytest.mark.asyncio
async def test_agentic_loop_calls_tools_then_finishes() -> None:
    final_json = json.dumps(
        {
            "student_name": "Alex Rivera",
            "target_university": "MIT EECS",
            "admission_probability": 0.71,
            "probability_label": "High",
            "top_gaps": ["Gap 1", "Gap 2", "Gap 3"],
            "upcoming_deadlines": [],
            "strategic_recommendation": "Prioritize academic rigor and stronger recommendations.",
            "tool_calls_made": 0,
            "total_latency_ms": 0,
        }
    )
    anthropic_client = FakeAnthropicClient(
        [
            FakeResponse([FakeToolUseBlock(name="evaluate_chances", input={"target_university_id": "mit-eecs"}, id="1")]),
            FakeResponse([FakeToolUseBlock(name="get_action_items", input={"target_university_id": "mit-eecs"}, id="2")]),
            FakeResponse([FakeTextBlock(text=final_json)]),
        ]
    )
    session = FakeSession()

    result = await run_conversation(session, [], anthropic_client=anthropic_client)  # type: ignore[arg-type]

    assert [name for name, _ in session.calls] == ["evaluate_chances", "get_action_items"]
    assert result.target_university == "MIT EECS"
    assert result.tool_calls_made == 3


@pytest.mark.asyncio
async def test_agentic_loop_terminates_when_no_tool_use_blocks() -> None:
    final_json = json.dumps(
        {
            "student_name": "Alex Rivera",
            "target_university": "MIT EECS",
            "admission_probability": 0.5,
            "probability_label": "Medium",
            "top_gaps": ["Gap 1", "Gap 2", "Gap 3"],
            "upcoming_deadlines": [],
            "strategic_recommendation": "Continue strengthening quantitative signals.",
            "tool_calls_made": 0,
            "total_latency_ms": 0,
        }
    )

    result = await run_conversation(
        FakeSession(),
        [],
        anthropic_client=FakeAnthropicClient([FakeResponse([FakeTextBlock(text=final_json)])]),  # type: ignore[arg-type]
    )

    assert result.admission_probability == pytest.approx(0.5)
    assert result.probability_label == "Medium"


@pytest.mark.asyncio
async def test_agentic_loop_stops_after_ten_iterations() -> None:
    anthropic_client = FakeAnthropicClient(
        [FakeResponse([FakeToolUseBlock(name="evaluate_chances", input={}, id=str(index))]) for index in range(12)]
    )
    session = FakeSession()

    result = await run_conversation(session, [], anthropic_client=anthropic_client)  # type: ignore[arg-type]

    assert result.tool_calls_made == 10
    assert len(session.calls) == 10


@pytest.mark.asyncio
async def test_openai_compatible_loop_calls_tools_then_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "model_provider", "openai")
    openai_client = FakeOpenAIClient(
        [
            ChatCompletionResult(
                text="",
                tool_uses=[ToolUseRequest(id="1", name="evaluate_chances", input={"target_university_id": "mit-eecs"})],
            ),
            ChatCompletionResult(
                text="",
                tool_uses=[ToolUseRequest(id="2", name="get_action_items", input={"target_university_id": "mit-eecs"})],
            ),
            ChatCompletionResult(
                text=json.dumps(
                    {
                        "student_name": "Alex Rivera",
                        "target_university": "MIT EECS",
                        "admission_probability": 0.71,
                        "probability_label": "High",
                        "top_gaps": ["Gap 1", "Gap 2", "Gap 3"],
                        "upcoming_deadlines": [],
                        "strategic_recommendation": "Prioritize academic rigor and stronger recommendations.",
                        "tool_calls_made": 0,
                        "total_latency_ms": 0,
                    }
                ),
                tool_uses=[],
            ),
        ]
    )
    session = FakeSession()

    result = await run_conversation(session, [], openai_client=openai_client)

    assert [name for name, _ in session.calls] == ["evaluate_chances", "get_action_items"]
    assert result.target_university == "MIT EECS"
    assert result.tool_calls_made == 3


@pytest.mark.asyncio
async def test_openrouter_client_raises_on_http_error(mocker: Any) -> None:
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    error_response = httpx.Response(
        429,
        request=request,
        json={"error": {"message": "Rate limit exceeded"}},
    )
    mocker.patch.object(
        client,
        "_post_chat_completion",
        side_effect=httpx.HTTPStatusError(
            "rate limit",
            request=request,
            response=error_response,
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.create_chat_completion(
            model="anthropic/claude-3-haiku",
            system="You are a test assistant.",
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_completion_tokens=32,
        )


def test_final_recommendation_model_validation() -> None:
    recommendation = parse_final_recommendation(
        json.dumps(
            {
                "student_name": "Alex Rivera",
                "target_university": "MIT EECS",
                "admission_probability": 0.8,
                "probability_label": "High",
                "top_gaps": ["Gap 1", "Gap 2", "Gap 3"],
                "upcoming_deadlines": [],
                "strategic_recommendation": "Strong profile with one notable gap.",
                "tool_calls_made": 0,
                "total_latency_ms": 0,
            }
        ),
        iterations=2,
        total_latency_ms=50,
    )

    assert isinstance(recommendation, FinalRecommendation)
    assert recommendation.tool_calls_made == 2


def test_final_recommendation_accepts_percentage_strings() -> None:
    recommendation = parse_final_recommendation(
        json.dumps(
            {
                "student_name": "Alex Rivera",
                "target_university": "MIT EECS",
                "admission_probability": "91%",
                "top_gaps": ["Gap 1", "Gap 2", "Gap 3"],
                "upcoming_deadlines": [],
                "strategic_recommendation": "Strong fit with a few targeted improvements.",
            }
        ),
        iterations=2,
        total_latency_ms=50,
    )

    assert recommendation.admission_probability == pytest.approx(0.91)
    assert recommendation.probability_label == "High"


def test_final_recommendation_normalizes_stringified_json_recommendation() -> None:
    recommendation = parse_final_recommendation(
        json.dumps(
            {
                "student_name": "Alex Rivera",
                "target_university": "MIT EECS",
                "admission_probability": 1.0,
                "top_gaps": [],
                "upcoming_deadlines": [],
                "strategic_recommendation": json.dumps(
                    {
                        "name": "Alex Rivera",
                        "probability": "Very High",
                        "recommendation": "Currently strong odds of admission to MIT EECS",
                        "action_plan": [
                            {
                                "step": "Submit application before the early action deadline to maximize consideration",
                                "priority": 1,
                            },
                            {
                                "step": "Prepare a compelling essay highlighting leadership in tech projects and future research goals",
                                "priority": 2,
                            },
                        ],
                    }
                ),
            }
        ),
        iterations=2,
        total_latency_ms=50,
    )

    assert recommendation.strategic_recommendation.startswith(
        "Currently strong odds of admission to MIT EECS"
    )
    assert "Submit application before the early action deadline to maximize consideration" in (
        recommendation.strategic_recommendation
    )
    assert not recommendation.strategic_recommendation.startswith("{")


def test_final_recommendation_normalizes_nested_action_plan_recommendation() -> None:
    recommendation = parse_final_recommendation(
        json.dumps(
            {
                "student_name": "Alex Rivera",
                "target_university": "MIT EECS",
                "admission_probability": 1.0,
                "top_gaps": [],
                "upcoming_deadlines": [],
                "strategic_recommendation": json.dumps(
                    {
                        "name": "Alex Rivera",
                        "target_university": "MIT EECS",
                        "admission_probability": "High",
                        "overall_score": 1,
                        "specifics": {
                            "action_plan": [
                                {
                                    "priority": 1,
                                    "action": "Submit application by the earliest deadline to maximize visibility",
                                },
                                {
                                    "priority": 2,
                                    "step": "Obtain a third, highly detailed Letter of Recommendation from a STEM professor",
                                },
                            ]
                        },
                    }
                ),
            }
        ),
        iterations=2,
        total_latency_ms=50,
    )

    assert recommendation.strategic_recommendation.startswith(
        "Admission outlook for MIT EECS is High."
    )
    assert "Submit application by the earliest deadline to maximize visibility" in (
        recommendation.strategic_recommendation
    )
    assert "Obtain a third, highly detailed Letter of Recommendation from a STEM professor" in (
        recommendation.strategic_recommendation
    )
    assert not recommendation.strategic_recommendation.startswith("{")


def test_final_recommendation_recovers_malformed_json_like_wrapper() -> None:
    recommendation = parse_final_recommendation(
        json.dumps(
            {
                "student_name": "Alex Rivera",
                "target_university": "MIT EECS",
                "admission_probability": 1.0,
                "top_gaps": [],
                "upcoming_deadlines": [],
                "strategic_recommendation": '{"final_recommendation":"Given the data, Alex is competitive.\\n1. Submit early.\\n2. Strengthen letters.")',
            }
        ),
        iterations=2,
        total_latency_ms=50,
    )

    assert recommendation.strategic_recommendation.startswith(
        "Given the data, Alex is competitive."
    )
    assert "Submit early." in recommendation.strategic_recommendation
    assert not recommendation.strategic_recommendation.startswith("{")


def test_final_recommendation_normalizes_camel_case_action_plan() -> None:
    recommendation = parse_final_recommendation(
        json.dumps(
            {
                "student_name": "Alex Rivera",
                "target_university": "MIT EECS",
                "admission_probability": 1.0,
                "top_gaps": [],
                "upcoming_deadlines": [],
                "strategic_recommendation": json.dumps(
                    {
                        "probability": "High",
                        "confidence": "moderate",
                        "actionPlan": [
                            {
                                "priority": 1,
                                "task": "Compile and submit a strong statement of purpose highlighting research interests and relevant projects",
                            },
                            {
                                "priority": 2,
                                "task": "Obtain 3-4 high-quality letters of recommendation, prioritizing professors who can speak to advanced coursework and research",
                            },
                        ],
                    }
                ),
            }
        ),
        iterations=2,
        total_latency_ms=50,
    )

    assert recommendation.strategic_recommendation.startswith(
        "Current admission probability is High."
    )
    assert (
        "Compile and submit a strong statement of purpose highlighting research interests and relevant projects"
        in recommendation.strategic_recommendation
    )
    assert not recommendation.strategic_recommendation.startswith("{")


def test_format_recommendation_output_is_human_readable() -> None:
    recommendation = FinalRecommendation(
        student_name="Alex Rivera",
        target_university="MIT EECS",
        admission_probability=0.912,
        probability_label="High",
        top_gaps=["Gap 1", "Gap 2", "Gap 3"],
        upcoming_deadlines=[
            {
                "name": "MIT Research Fellowship",
                "deadline": "2026-06-11",
                "days_remaining": 45,
            }
        ],
        strategic_recommendation="Retake the GRE and strengthen research alignment.",
        tool_calls_made=3,
        total_latency_ms=16357,
    )

    output = format_recommendation_output(recommendation)

    assert "University Admission Recommendation" in output
    assert "Admission probability: High (91.2%)" in output
    assert "- MIT Research Fellowship (2026-06-11; 45 days remaining)" in output
    assert "Recommendation\nRetake the GRE and strengthen research alignment." in output


@pytest.mark.parametrize(
    ("probability", "expected_label"),
    [(0.3, "Low"), (0.5, "Medium"), (0.8, "High")],
)
def test_probability_label_mapping(probability: float, expected_label: str) -> None:
    assert probability_to_label(probability) == expected_label