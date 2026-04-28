from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


ConversationMessage = dict[str, object]
KNOWN_TOOL_NAMES: tuple[str, ...] = ("evaluate_chances", "get_action_items")


@dataclass(frozen=True)
class ToolUseRequest:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    tool_uses: list[ToolUseRequest]


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")

    async def create_chat_completion(
        self,
        *,
        model: str,
        system: str,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]] | None = None,
        max_completion_tokens: int,
        response_format: dict[str, object] | None = None,
    ) -> ChatCompletionResult:
        base_payload: dict[str, object] = {
            "model": model,
            "messages": self._build_messages(system, messages),
            "max_completion_tokens": max_completion_tokens,
        }
        if tools:
            base_payload["tools"] = self._convert_tools(tools)
            base_payload["tool_choice"] = "auto"
            base_payload["parallel_tool_calls"] = False
        if response_format is not None:
            base_payload["response_format"] = response_format

        model_candidates = [model]
        for index, candidate in enumerate(model_candidates):
            payload = dict(base_payload)
            payload["model"] = candidate
            try:
                response_payload = await self._post_chat_completion(payload)
                message = self._extract_message(response_payload)
                return ChatCompletionResult(
                    text=self._coerce_text(message.get("content")),
                    tool_uses=self._parse_tool_uses(message.get("tool_calls")),
                )
            except httpx.HTTPStatusError:
                has_more_candidates = index < len(model_candidates) - 1
                if not has_more_candidates:
                    raise

        raise RuntimeError("No usable model candidates were available")

    async def create_json_completion(
        self,
        *,
        model: str,
        system: str,
        user_prompt: str,
        max_completion_tokens: int,
    ) -> str:
        result = await self.create_chat_completion(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"},
        )
        return result.text

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(4),
        retry=retry_if_exception(lambda exc: _is_retryable_http_error(exc)),
        reraise=True,
    )
    async def _post_chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            raw_payload = response.json()

        if not isinstance(raw_payload, dict):
            raise ValueError("Chat completion response was not a JSON object")
        return raw_payload

    def _build_messages(
        self,
        system: str,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        request_messages: list[ConversationMessage] = []
        if system:
            request_messages.append({"role": "system", "content": system})
        request_messages.extend(messages)
        return request_messages

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, object]]:
        converted_tools: list[dict[str, object]] = []
        for tool in tools:
            name = tool.get("name")
            description = tool.get("description", "")
            input_schema = tool.get("input_schema")
            if not isinstance(name, str) or not isinstance(description, str):
                continue
            if not isinstance(input_schema, dict):
                continue
            converted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": input_schema,
                    },
                }
            )
        return converted_tools

    def _extract_message(self, payload: dict[str, object]) -> dict[str, object]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat completion response did not include choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Chat completion choice was not a JSON object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat completion message was not a JSON object")
        return message

    def _coerce_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)
        return ""

    def _parse_tool_uses(self, tool_calls: object) -> list[ToolUseRequest]:
        parsed_tool_calls: list[ToolUseRequest] = []
        if not isinstance(tool_calls, list):
            return parsed_tool_calls

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            identifier = tool_call.get("id")
            function = tool_call.get("function")
            if not isinstance(identifier, str) or not isinstance(function, dict):
                continue

            name = function.get("name")
            arguments = function.get("arguments", "{}")
            if not isinstance(name, str):
                continue
            normalized_name = _normalize_tool_name(name)

            parsed_tool_calls.append(
                ToolUseRequest(
                    id=identifier,
                    name=normalized_name,
                    input=self._parse_arguments(arguments),
                )
            )
        return parsed_tool_calls

    def _parse_arguments(self, arguments: object) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if not stripped:
                return {}
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Tool call arguments were not valid JSON")

def _is_retryable_http_error(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    return exc.response.status_code in {429, 500, 502, 503, 504}


def _normalize_tool_name(name: str) -> str:
    stripped = name.strip()
    for known_tool_name in KNOWN_TOOL_NAMES:
        if known_tool_name in stripped:
            return known_tool_name
    return stripped