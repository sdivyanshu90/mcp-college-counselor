from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from textwrap import fill
from typing import Any, Literal, Protocol, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ToolParam, ToolResultBlockParam
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool
from pydantic import BaseModel, model_validator

from config import settings
from llm_provider import ChatCompletionResult, OpenAICompatibleClient, ToolUseRequest


STUDENT_PROFILE: dict[str, Any] = {
    "name": "Alex Rivera",
    "gpa": 3.8,
    "sat_score": 1600,
    "gre_verbal": 155,
    "gre_quant": 165,
    "ap_classes_completed": 4,
    "lor_count": 2,
    "target_university_id": "umich-cs",
}
SYSTEM_PROMPT = (
    "You are an expert university admissions counselor. "
    "You have access to real admission data via tools. "
    "Always call evaluate_chances first, then get_action_items, "
    "then synthesize a final recommendation. "
    "Your final response must be a single JSON object matching the "
    "FinalRecommendation schema. Output nothing before or after the JSON."
)
USER_MESSAGE = (
    f"Student profile: {json.dumps(STUDENT_PROFILE, indent=2)}\n\n"
    f"Please evaluate admission chances at {STUDENT_PROFILE['target_university_id']} and provide a "
    "specific, prioritized action plan."
)


class FinalRecommendation(BaseModel):
    student_name: str
    target_university: str
    admission_probability: float
    probability_label: Literal["Low", "Medium", "High"]
    top_gaps: list[str]
    upcoming_deadlines: list[dict[str, Any]]
    strategic_recommendation: str
    tool_calls_made: int
    total_latency_ms: int

    @model_validator(mode="after")
    def normalize_top_gaps(self) -> FinalRecommendation:
        normalized = [gap for gap in self.top_gaps if gap][:3]
        while len(normalized) < 3:
            normalized.append("No major gap identified.")
        self.top_gaps = normalized
        return self


class ToolCallingSession(Protocol):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        ...


class OpenAICompatibleConversationClient(Protocol):
    async def create_chat_completion(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, Any]] | None = None,
        max_completion_tokens: int,
        response_format: dict[str, object] | None = None,
    ) -> ChatCompletionResult:
        ...


def convert_mcp_tool(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.inputSchema,
    }


def probability_to_label(probability: float) -> Literal["Low", "Medium", "High"]:
    if probability < 0.40:
        return "Low"
    if probability <= 0.65:
        return "Medium"
    return "High"


def parse_final_recommendation(
    raw_text: str,
    iterations: int,
    total_latency_ms: int,
    evaluation_result: dict[str, Any] | None = None,
    action_items_result: dict[str, Any] | None = None,
) -> FinalRecommendation:
    default_payload = _build_default_recommendation_payload(
        raw_text,
        iterations,
        total_latency_ms,
        evaluation_result,
        action_items_result,
    )

    try:
        parsed_payload = json.loads(raw_text)
        if isinstance(parsed_payload, str):
            default_payload["strategic_recommendation"] = parsed_payload
            return FinalRecommendation.model_validate(default_payload)
        if not isinstance(parsed_payload, dict):
            raise ValueError("Final response was not a JSON object")
    except (json.JSONDecodeError, ValueError):
        return FinalRecommendation.model_validate(default_payload)

    if "strategic_recommendation" not in parsed_payload:
        final_recommendation = parsed_payload.get("final_recommendation")
        if isinstance(final_recommendation, str) and final_recommendation.strip():
            parsed_payload["strategic_recommendation"] = final_recommendation.strip()
    normalized_recommendation = _normalize_recommendation_text(
        parsed_payload.get("strategic_recommendation")
    )
    if normalized_recommendation is None:
        normalized_recommendation = _normalize_recommendation_text(parsed_payload)
    if normalized_recommendation is not None:
        parsed_payload["strategic_recommendation"] = normalized_recommendation

    payload = dict(default_payload)
    payload.update(parsed_payload)
    payload["tool_calls_made"] = iterations
    payload["total_latency_ms"] = total_latency_ms
    probability = _normalize_probability_value(
        payload.get("admission_probability", 0.0),
        default_payload["admission_probability"],
    )
    payload["admission_probability"] = probability
    payload["probability_label"] = probability_to_label(probability)
    return FinalRecommendation.model_validate(payload)


def _build_default_recommendation_payload(
    raw_text: str,
    iterations: int,
    total_latency_ms: int,
    evaluation_result: dict[str, Any] | None,
    action_items_result: dict[str, Any] | None,
) -> dict[str, Any]:
    probability = _coerce_probability(evaluation_result)
    target_university = _coerce_university_name(evaluation_result, action_items_result)
    top_gaps = _coerce_gap_list(action_items_result)
    upcoming_deadlines = _coerce_deadlines(action_items_result)

    if raw_text.strip():
        recommendation_text = raw_text.strip()
    else:
        recommendation_text = _synthesize_recommendation_text(
            probability,
            target_university,
            top_gaps,
        )

    return {
        "student_name": str(STUDENT_PROFILE["name"]),
        "target_university": target_university,
        "admission_probability": probability,
        "probability_label": probability_to_label(probability),
        "top_gaps": top_gaps,
        "upcoming_deadlines": upcoming_deadlines,
        "strategic_recommendation": recommendation_text,
        "tool_calls_made": iterations,
        "total_latency_ms": total_latency_ms,
    }


def _coerce_probability(evaluation_result: dict[str, Any] | None) -> float:
    if evaluation_result is None:
        return 0.0
    return _normalize_probability_value(evaluation_result.get("probability"), 0.0)


def _normalize_probability_value(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        has_percent_suffix = stripped.endswith("%")
        numeric_portion = stripped.removesuffix("%").strip()
        try:
            numeric_value = float(numeric_portion)
        except ValueError:
            return fallback
        if has_percent_suffix or numeric_value > 1.0:
            return numeric_value / 100
        return numeric_value
    return fallback


def _coerce_university_name(
    evaluation_result: dict[str, Any] | None,
    action_items_result: dict[str, Any] | None,
) -> str:
    for candidate in (
        action_items_result.get("university_name") if action_items_result is not None else None,
        evaluation_result.get("university_name") if evaluation_result is not None else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return str(STUDENT_PROFILE["target_university_id"])


def _coerce_gap_list(action_items_result: dict[str, Any] | None) -> list[str]:
    if action_items_result is None:
        return [
            "No major gap identified.",
            "No major gap identified.",
            "No major gap identified.",
        ]

    gaps = action_items_result.get("gaps")
    if not isinstance(gaps, list):
        return [
            "No major gap identified.",
            "No major gap identified.",
            "No major gap identified.",
        ]

    normalized_gaps = [gap for gap in gaps if isinstance(gap, str) and gap.strip()]
    if not normalized_gaps:
        return [
            "No major gap identified.",
            "No major gap identified.",
            "No major gap identified.",
        ]
    return normalized_gaps[:3]


def _coerce_deadlines(action_items_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if action_items_result is None:
        return []
    deadlines = action_items_result.get("upcoming_deadlines")
    if not isinstance(deadlines, list):
        return []
    normalized_deadlines: list[dict[str, Any]] = []
    for deadline in deadlines:
        if isinstance(deadline, dict):
            normalized_deadlines.append(deadline)
    return normalized_deadlines


def _synthesize_recommendation_text(
    probability: float,
    target_university: str,
    top_gaps: list[str],
) -> str:
    if top_gaps and top_gaps[0] != "No major gap identified.":
        return (
            f"Admission probability for {target_university} is {probability:.2%}. "
            f"Prioritize the highest-impact gap first: {top_gaps[0]}"
        )
    return (
        f"Admission probability for {target_university} is {probability:.2%}. "
        "No material admissions gap was identified from the current tool outputs."
    )

def _normalize_recommendation_text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed_value = json.loads(stripped)
        except json.JSONDecodeError:
            for field_name in ("strategic_recommendation", "final_recommendation", "recommendation"):
                extracted_value = _extract_json_like_string_field(stripped, field_name)
                if extracted_value is not None:
                    return extracted_value
            return stripped
        normalized = _normalize_recommendation_text(parsed_value)
        if normalized is not None:
            return normalized
        return stripped

    if not isinstance(value, dict):
        return None

    recommendation_text: str | None = None
    for key in ("strategic_recommendation", "final_recommendation", "recommendation"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            recommendation_text = candidate.strip()
            break

    if recommendation_text is None:
        target_university = value.get("target_university")
        admission_probability = value.get("admission_probability", value.get("probability"))
        if isinstance(target_university, str) and target_university.strip():
            if isinstance(admission_probability, str) and admission_probability.strip():
                recommendation_text = (
                    f"Admission outlook for {target_university.strip()} is "
                    f"{admission_probability.strip()}."
                )
            elif isinstance(admission_probability, (int, float)):
                recommendation_text = (
                    f"Admission outlook for {target_university.strip()} is "
                    f"{float(admission_probability):.0%}."
                )
        elif isinstance(admission_probability, str) and admission_probability.strip():
            recommendation_text = f"Current admission probability is {admission_probability.strip()}."
        elif isinstance(admission_probability, (int, float)):
            recommendation_text = f"Current admission probability is {float(admission_probability):.0%}."

    action_steps: list[str] = []
    action_plan = value.get("action_plan", value.get("actionPlan"))
    if not isinstance(action_plan, list):
        specifics = value.get("specifics")
        if isinstance(specifics, dict):
            action_plan = specifics.get("action_plan", specifics.get("actionPlan"))
    if isinstance(action_plan, list):
        for item in action_plan:
            if isinstance(item, dict):
                step = item.get("step")
                if not isinstance(step, str) or not step.strip():
                    step = item.get("action")
                if not isinstance(step, str) or not step.strip():
                    step = item.get("task")
            else:
                step = item
            if isinstance(step, str) and step.strip():
                action_steps.append(step.strip().rstrip("."))

    if recommendation_text and action_steps:
        return f"{recommendation_text} Next steps: {'; '.join(action_steps[:3])}."
    if recommendation_text:
        return recommendation_text
    if action_steps:
        return f"Next steps: {'; '.join(action_steps[:3])}."
    return None


def _extract_json_like_string_field(raw_text: str, field_name: str) -> str | None:
    token = f'"{field_name}"'
    field_index = raw_text.find(token)
    if field_index < 0:
        return None

    colon_index = raw_text.find(":", field_index + len(token))
    if colon_index < 0:
        return None

    remainder = raw_text[colon_index + 1 :].lstrip()
    if not remainder.startswith('"'):
        return None

    try:
        parsed_text, _ = json.JSONDecoder().raw_decode(remainder)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_text, str) and parsed_text.strip():
        return parsed_text.strip()
    return None


def extract_last_text_block(content_blocks: list[object]) -> str:
    text_blocks = [
        cast(str, getattr(block, "text"))
        for block in content_blocks
        if getattr(block, "type", "") == "text" and isinstance(getattr(block, "text", None), str)
    ]
    return text_blocks[-1] if text_blocks else ""


async def run_conversation(
    session: ToolCallingSession,
    anthropic_tools: list[dict[str, Any]],
    anthropic_client: AsyncAnthropic | None = None,
    openai_client: OpenAICompatibleConversationClient | None = None,
) -> FinalRecommendation:
    if settings.model_provider in {"openai", "openrouter"}:
        return await _run_openai_conversation(
            session,
            anthropic_tools,
            openai_client or OpenAICompatibleClient(
                api_key=settings.openai_compatible_api_key,
                base_url=settings.openai_compatible_base_url,
            ),
        )

    return await _run_anthropic_conversation(session, anthropic_tools, anthropic_client)


async def _run_anthropic_conversation(
    session: ToolCallingSession,
    anthropic_tools: list[dict[str, Any]],
    anthropic_client: AsyncAnthropic | None,
) -> FinalRecommendation:
    messages: list[MessageParam] = [{"role": "user", "content": USER_MESSAGE}]
    client = anthropic_client or AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    iterations = 0
    started_at = time.monotonic()
    response_content: list[object] = []
    evaluation_result: dict[str, Any] | None = None
    action_items_result: dict[str, Any] | None = None

    while iterations < 10:
        iterations += 1
        response = await client.messages.create(
            model=settings.client_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=cast(list[ToolParam], anthropic_tools),
            messages=messages,
        )
        response_content = list(response.content)
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if getattr(block, "type", "") == "tool_use"]
        if not tool_uses:
            break

        tool_results: list[ToolResultBlockParam] = []
        for block in tool_uses:
            tool_name = getattr(block, "name", "")
            tool_input = getattr(block, "input", {})
            tool_use_id = getattr(block, "id", "")
            tool_request = ToolUseRequest(
                id=cast(str, tool_use_id),
                name=cast(str, tool_name),
                input=cast(dict[str, Any], tool_input),
            )
            content = await _call_mcp_tool(session, tool_request)
            evaluation_result, action_items_result = _update_tool_context(
                tool_request.name,
                content,
                evaluation_result,
                action_items_result,
            )
            tool_result_content: list[TextBlockParam] = [
                {"type": "text", "text": content}
            ]
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": cast(str, tool_use_id),
                    "content": tool_result_content,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    total_latency_ms = round((time.monotonic() - started_at) * 1000)
    final_text = extract_last_text_block(response_content)
    return _emit_recommendation(
        final_text,
        iterations,
        total_latency_ms,
        evaluation_result,
        action_items_result,
    )


async def _run_openai_conversation(
    session: ToolCallingSession,
    anthropic_tools: list[dict[str, Any]],
    openai_client: OpenAICompatibleConversationClient,
) -> FinalRecommendation:
    messages: list[dict[str, object]] = [{"role": "user", "content": USER_MESSAGE}]
    iterations = 0
    started_at = time.monotonic()
    final_text = ""
    evaluation_result: dict[str, Any] | None = None
    action_items_result: dict[str, Any] | None = None

    while iterations < 10:
        iterations += 1
        response = await openai_client.create_chat_completion(
            model=settings.client_model,
            system=SYSTEM_PROMPT,
            tools=anthropic_tools,
            messages=messages,
            max_completion_tokens=2048,
        )

        final_text = response.text
        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": response.text,
        }
        if response.tool_uses:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_use.id,
                    "type": "function",
                    "function": {
                        "name": tool_use.name,
                        "arguments": json.dumps(tool_use.input),
                    },
                }
                for tool_use in response.tool_uses
            ]
        messages.append(assistant_message)

        if not response.tool_uses:
            break

        for tool_use in response.tool_uses:
            content = await _call_mcp_tool(session, tool_use)
            evaluation_result, action_items_result = _update_tool_context(
                tool_use.name,
                content,
                evaluation_result,
                action_items_result,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use.id,
                    "name": tool_use.name,
                    "content": content,
                }
            )

    total_latency_ms = round((time.monotonic() - started_at) * 1000)
    return _emit_recommendation(
        final_text,
        iterations,
        total_latency_ms,
        evaluation_result,
        action_items_result,
    )


async def _call_mcp_tool(
    session: ToolCallingSession,
    tool_use: ToolUseRequest,
) -> str:
    try:
        mcp_result = await session.call_tool(tool_use.name, tool_use.input)
        if mcp_result.content:
            content = getattr(mcp_result.content[0], "text", "{}")
        else:
            content = "{}"
    except Exception as exc:
        content = json.dumps({"error": str(exc)})
    return cast(str, content)


def _emit_recommendation(
    final_text: str,
    iterations: int,
    total_latency_ms: int,
    evaluation_result: dict[str, Any] | None = None,
    action_items_result: dict[str, Any] | None = None,
) -> FinalRecommendation:
    recommendation = parse_final_recommendation(
        final_text,
        iterations,
        total_latency_ms,
        evaluation_result,
        action_items_result,
    )
    print(format_recommendation_output(recommendation))
    return recommendation


def format_recommendation_output(recommendation: FinalRecommendation) -> str:
    lines = [
        "University Admission Recommendation",
        "=================================",
        f"Student: {recommendation.student_name}",
        f"Target university: {recommendation.target_university}",
        (
            "Admission probability: "
            f"{recommendation.probability_label} ({recommendation.admission_probability:.1%})"
        ),
        f"Tool calls: {recommendation.tool_calls_made}",
        f"Total latency: {recommendation.total_latency_ms / 1000:.1f}s",
        "",
        "Top gaps",
    ]

    for gap in recommendation.top_gaps:
        lines.append(f"- {gap}")

    lines.extend(["", "Upcoming deadlines"])
    if recommendation.upcoming_deadlines:
        for deadline in recommendation.upcoming_deadlines:
            lines.append(_format_deadline_output(deadline))
    else:
        lines.append("- None")

    lines.extend(["", "Recommendation"])
    lines.extend(_wrap_output_block(recommendation.strategic_recommendation))
    return "\n".join(lines)


def _format_deadline_output(deadline: dict[str, Any]) -> str:
    name = str(deadline.get("name") or "Unnamed deadline").strip()
    details: list[str] = []

    deadline_value = deadline.get("deadline")
    if isinstance(deadline_value, str) and deadline_value.strip():
        details.append(deadline_value.strip())

    days_remaining = deadline.get("days_remaining")
    if isinstance(days_remaining, int):
        details.append(f"{days_remaining} days remaining")
    elif isinstance(days_remaining, float):
        details.append(f"{int(days_remaining)} days remaining")
    elif isinstance(days_remaining, str) and days_remaining.strip():
        details.append(days_remaining.strip())

    if details:
        return f"- {name} ({'; '.join(details)})"
    return f"- {name}"


def _wrap_output_block(text: str) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        return ["No recommendation provided."]
    return [fill(paragraph, width=88) for paragraph in paragraphs]


def _update_tool_context(
    tool_name: str,
    content: str,
    evaluation_result: dict[str, Any] | None,
    action_items_result: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parsed = _maybe_parse_json_object(content)
    if parsed is None:
        return evaluation_result, action_items_result
    if tool_name == "evaluate_chances":
        return parsed, action_items_result
    if tool_name == "get_action_items":
        return evaluation_result, parsed
    return evaluation_result, action_items_result


def _maybe_parse_json_object(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


async def run_agent() -> FinalRecommendation:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=dict(os.environ),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            anthropic_tools = [convert_mcp_tool(tool) for tool in tools_result.tools]
            return await run_conversation(session, anthropic_tools)


if __name__ == "__main__":
    asyncio.run(run_agent())