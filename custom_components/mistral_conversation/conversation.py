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

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mistral AI conversation entity."""
    async_add_entities([MistralConversationEntity(hass, config_entry)])

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
