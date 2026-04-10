"""Conversation platform for Mistral AI."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

import aiohttp
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    AGENT_CAPABLE_MODELS,
    CONF_CONTINUE_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_WEB_SEARCH,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEB_SEARCH,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
    MISTRAL_API_BASE,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively make obj fully JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)

def _format_tool(tool: llm.Tool, custom_serializer: Any = None) -> dict[str, Any]:
    """Convert an HA LLM tool to Mistral function-calling format."""
    try:
        from voluptuous_openapi import convert
        parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    except Exception:
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(tool.description or ""),
            "parameters": parameters,
        },
    }

def _to_mistral_id(ha_id: str) -> str:
    """Convert an HA tool_call ID to a Mistral-compatible ID."""
    import hashlib
    return hashlib.md5(ha_id.encode()).hexdigest()[:9]

def _convert_chat_log_to_messages(chat_log: conversation.ChatLog) -> list[dict[str, Any]]:
    """Convert HA ChatLog content into Mistral messages."""
    messages: list[dict[str, Any]] = []
    tool_results: dict[str, conversation.ToolResultContent] = {
        c.tool_call_id: c for c in chat_log.content if isinstance(c, conversation.ToolResultContent)
    }

    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": content.content})
        elif isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": content.content})
        elif isinstance(content, conversation.AssistantContent):
            if content.tool_calls:
                msg = {
                    "role": "assistant",
                    "content": content.content or "",
                    "tool_calls": [
                        {
                            "id": _to_mistral_id(tc.id),
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.tool_args),
                            },
                        } for tc in content.tool_calls
                    ],
                }
                messages.append(msg)
                for tc in content.tool_calls:
                    if res := tool_results.get(tc.id):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": _to_mistral_id(res.tool_call_id),
                            "name": res.tool_name,
                            "content": json.dumps(res.tool_result),
                        })
            else:
                messages.append({"role": "assistant", "content": content.content or ""})
    return messages

async def _async_stream_delta(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[dict[str, Any], None]:
    """Parse SSE stream and yield dicts with proper HA objects."""
    buffer = b""
    current_tool_calls: dict[int, dict] = {}

    async def _flush_tool_calls():
        for tc in current_tool_calls.values():
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            
            tool_input = llm.ToolInput(
                tool_name=tc["name"],
                tool_args=args,
                id=tc["id"],
            )
            # Voor compatibiliteit met de nieuwste HA versies
            if not hasattr(tool_input, 'external'):
                setattr(tool_input, 'external', False)
                
            yield {"tool_calls": [tool_input]}
        current_tool_calls.clear()

    async for raw_chunk in resp.content.iter_any():
        buffer += raw_chunk
        while b"\n\n" in buffer:
            frame, buffer = buffer.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                line_str = line.decode("utf-8", errors="replace")
                if not line_str.startswith("data: "): continue
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    async for delta in _flush_tool_calls(): yield delta
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError: continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if "content" in delta and delta["content"]:
                    yield {"content": delta["content"]}

                if "tool_calls" in delta:
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "name": tc_delta.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        if tc_delta.get("id"): current_tool_calls[idx]["id"] = tc_delta["id"]
                        if tc_delta.get("function", {}).get("name"): 
                            current_tool_calls[idx]["name"] = tc_delta["function"]["name"]
                        if tc_delta.get("function", {}).get("arguments"):
                            current_tool_calls[idx]["arguments"] += tc_delta["function"]["arguments"]

                if choice.get("finish_reason") in ("tool_calls", "stop") and current_tool_calls:
                    async for delta in _flush_tool_calls(): yield delta

# ---------------------------------------------------------------------------
# Entity Class
# ---------------------------------------------------------------------------

class MistralConversationEntity(ConversationEntity):
    """Mistral AI conversation agent entity."""
    _attr_has_entity_name = True
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = ConversationEntityFeature.CONTROL

    @property
    def _runtime(self):
        return self.hass.data[DOMAIN][self._entry.entry_id]

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_conversation")},
            name="Mistral AI Conversation",
            manufacturer="Mistral AI",
            entry_type=DeviceEntryType.SERVICE,
        )

    async def _async_handle_message(
        self, user_input: ConversationInput, chat_log: conversation.ChatLog
    ) -> ConversationResult:
        opts = self._entry.options
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                opts.get(CONF_LLM_HASS_API),
                opts.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools = None
        if chat_log.llm_api:
            tools = [_format_tool(t, chat_log.llm_api.custom_serializer) for t in chat_log.llm_api.tools]

        for _ in range(MAX_TOOL_ITERATIONS):
            payload = _sanitize({
                "model": opts.get(CONF_MODEL, DEFAULT_MODEL),
                "messages": _convert_chat_log_to_messages(chat_log),
                "max_tokens": int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
                "temperature": float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
                "stream": True,
            })
            if tools:
                payload["tools"] = tools

            await self._stream_and_collect(payload, chat_log, user_input)
            if not chat_log.unresponded_tool_results:
                break

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _stream_and_collect(self, payload, chat_log, user_input):
        runtime = self._runtime
        async with runtime.session.post(
            f"{MISTRAL_API_BASE}/chat/completions",
            headers=runtime.headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=90)
        ) as resp:
            if resp.status >= 400:
                raise HomeAssistantError(f"Mistral error {resp.status}: {await resp.text()}")

            async for _ in chat_log.async_add_delta_content_stream(
                user_input.agent_id,
                _async_stream_delta(resp),
            ):
                pass

# ---------------------------------------------------------------------------
# Setup Entry (verplaatst naar beneden om NameError te voorkomen)
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI conversation entity."""
    async_add_entities([MistralConversationEntity(hass, config_entry)])
