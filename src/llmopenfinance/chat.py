#!/usr/bin/env python3
"""Interactive chat interfaces for language models via Ollama (local) or Claude (Anthropic API).

Modelled on ``apexmusic.llm.chat`` (``AbstractChat`` / ``OllamaChat`` / ``ClaudeChat`` with
``chat_stream`` / ``chat`` / ``set_system_message`` / ``reset_conversation``) and extended with
**tool calling** so the assistant can query the local financial database.

Design notes:

* Each provider keeps its **native** transcript (Anthropic content blocks incl. thinking
  signatures; Ollama messages with ``tool_calls``/``tool_name``). Only *events* and *tool specs*
  are normalized: :class:`TextDelta`, :class:`ToolCallsReady`, :class:`TurnEnd`.
* Claude: the system prompt is the top-level ``system`` parameter (a ``{"role": "system"}``
  message at ``messages[0]`` is rejected), no ``temperature`` (Claude 5.5 models reject
  non-default sampling parameters), ``tool_choice: none`` to force a final answer while keeping
  the tool list (and the prompt cache prefix) unchanged; history is append-only.
* Ollama: ``think`` is set explicitly (qwen3.5 thinks by default and its tool template has known
  issues with thinking), ``num_ctx`` is set explicitly (the default is only 4k on <24 GB GPUs),
  and a fallback parses tool calls that small models emit as JSON text.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from llmopenfinance.constants import (
    CLAUDE_API_URL,
    CLAUDE_API_VERSION,
    CLAUDE_MAX_TOKENS,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_SERVER,
    ENV_ANTHROPIC_KEYS,
)
from llmopenfinance.tools import ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

Message = dict[str, Any]

FINAL_ANSWER_NOTE = (
    "Limite de chamadas de ferramentas atingido. Responda agora ao usuário com os dados que você já obteve."
)


class LLMError(RuntimeError):
    """Raised when the LLM backend fails (connection, HTTP error, stream error)."""


class _RetryableError(Exception):
    """Internal: a retryable HTTP status was received before streaming started."""

    def __init__(self, status_code: int, wait: float) -> None:
        super().__init__(f"retryable status {status_code}")
        self.status_code = status_code
        self.wait = wait


@dataclass(frozen=True)
class TextDelta:
    """A chunk of assistant text to display.

    Attributes:
        text: Text fragment.
    """

    text: str


@dataclass(frozen=True)
class ToolCallsReady:
    """The model requested tool calls in this turn.

    Attributes:
        calls: Calls to execute (in order).
    """

    calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class TurnEnd:
    """End of one model turn.

    Attributes:
        stop_reason: ``end_turn``, ``tool_use``, ``max_tokens``, ``refusal``...
        usage: Provider token usage counters.
    """

    stop_reason: str
    usage: Mapping[str, Any] = field(default_factory=dict)


Event = TextDelta | ToolCallsReady | TurnEnd


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a mapping or an object attribute (Ollama returns pydantic objects).

    Args:
        obj: Mapping or object.
        name: Field name.

    Returns:
        Any: The value or ``None``.
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


class AbstractChat(ABC):
    """Abstract base class for chat interfaces.

    Defines the interface for different chat implementations (Ollama, Claude).
    The system prompt is kept apart from the history and sent the provider's way.
    """

    provider = "abstract"

    def __init__(self, model: str) -> None:
        """Initialize the chat interface.

        Args:
            model: Model identifier.
        """
        self.model = model
        self.system_message = ""
        self.conversation_history: list[Message] = []

    @abstractmethod
    def _verify_connection(self) -> bool:
        """Verify connection to the chat service.

        Returns:
            bool: True if the service and model are reachable.
        """

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a user message to the history.

        Args:
            text: User text.
        """

    @abstractmethod
    def stream_turn(self, tools: Sequence[ToolSpec] | None = None, *, final: bool = False) -> Iterator[Event]:
        """Run one model turn over the current history.

        Args:
            tools: Tools the model may call (``None`` = plain chat).
            final: Force a text answer (no more tool calls) while keeping tool definitions.

        Yields:
            Event: Text deltas, then optionally :class:`ToolCallsReady`, then :class:`TurnEnd`.
        """

    @abstractmethod
    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append tool results for the calls of the last turn.

        Args:
            results: Results in call order.
        """

    def compact_history(self) -> None:
        """Drop intermediate tool rounds to save context (no-op where not allowed)."""
        logger.debug("%s keeps its full history", self.provider)

    def chat_stream(self, user_message: str) -> Iterator[str]:
        """Send a message and stream the response text (no tools).

        Args:
            user_message: User's input message.

        Yields:
            str: Response text fragments.
        """
        self.add_user_message(user_message)
        for event in self.stream_turn(None):
            if isinstance(event, TextDelta):
                yield event.text

    def chat(self, user_message: str) -> str:
        """Send a message and get the complete response (no tools).

        Args:
            user_message: User's input message.

        Returns:
            str: Complete response.
        """
        return "".join(self.chat_stream(user_message))

    def reset_conversation(self) -> None:
        """Clear conversation history (the system message is kept)."""
        self.conversation_history = []

    def truncate_history(self, length: int) -> None:
        """Roll the history back to ``length`` messages (used after a failed turn).

        Args:
            length: Number of messages to keep.
        """
        del self.conversation_history[length:]

    def get_history(self) -> list[Message]:
        """Get conversation history.

        Returns:
            list[Message]: Shallow copy of the provider-native history.
        """
        return self.conversation_history.copy()

    def set_system_message(self, system_message: str) -> None:
        """Set or update the system message.

        Args:
            system_message: System prompt to guide model behavior.
        """
        self.system_message = system_message


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def parse_text_tool_call(content: str, tool_names: Sequence[str]) -> dict[str, Any] | None:
    """Recover a tool call that a small model wrote as JSON text instead of ``tool_calls``.

    Args:
        content: Full assistant text.
        tool_names: Valid tool names.

    Returns:
        dict[str, Any] | None: ``{"name", "arguments"}`` or ``None`` if the text is not a tool call.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("function"), Mapping):
        data = data["function"]
    name = data.get("name")
    arguments = data.get("arguments", data.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if name not in tool_names or not isinstance(arguments, Mapping):
        return None
    return {"name": name, "arguments": dict(arguments)}


@dataclass
class _OllamaTurn:
    """Accumulates one streamed Ollama response.

    Attributes:
        hold_json: Withhold text that starts like a JSON tool call (so it is not printed raw).
        content: Assistant text so far.
        thinking: Thinking text (never displayed).
        raw_calls: Native tool calls.
        usage: Token counters from the final chunk.
        held_back: Whether text is currently being withheld.
    """

    hold_json: bool
    content: str = ""
    thinking: str = ""
    raw_calls: list[Any] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    held_back: bool = False

    def feed(self, chunk: Any) -> str:
        """Consume one stream chunk.

        Args:
            chunk: ``ollama.ChatResponse`` (or an equivalent mapping).

        Returns:
            str: Text to display now (empty when nothing or withheld).
        """
        message = _field(chunk, "message")
        text = _field(message, "content") or ""
        self.thinking += _field(message, "thinking") or ""
        self.raw_calls.extend(_field(message, "tool_calls") or [])
        if _field(chunk, "done"):
            self.usage = {
                "prompt_eval_count": _field(chunk, "prompt_eval_count"),
                "eval_count": _field(chunk, "eval_count"),
            }
        if not text:
            return ""
        self.content += text
        self.held_back = self.hold_json and self.content.lstrip().startswith(("{", "```"))
        return "" if self.held_back else text


class OllamaChat(AbstractChat):
    """Chat interface for Ollama-hosted language models (local, private).

    Args:
        model: Ollama model name (default ``qwen3.5:9b``).
        base_url: Ollama server URL (env ``OLLAMA_SERVER``/``OLLAMA_HOST`` or localhost).
        think: Thinking mode (``False`` recommended for tool loops with qwen3.5).
        num_ctx: Context window to request.
        temperature: Sampling temperature.
        client: Pre-built ``ollama.Client`` (tests inject a fake).
    """

    provider = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        think: bool | str = False,
        num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
        temperature: float = 0.3,
        client: Any = None,
    ) -> None:
        super().__init__(model or DEFAULT_OLLAMA_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_SERVER") or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_SERVER
        self.think = think
        self.num_ctx = num_ctx
        self.temperature = temperature
        if client is None:
            import ollama

            client = ollama.Client(host=self.base_url, timeout=1200)
        self.client = client
        self._call_counter = 0

    def _verify_connection(self) -> bool:
        """Verify that the Ollama server is up and the model is installed.

        Returns:
            bool: True if the model can be used.
        """
        try:
            self.client.show(self.model)
            return True
        except Exception:
            logger.debug("Ollama check failed", exc_info=True)
            return False

    def add_user_message(self, text: str) -> None:
        """Append a user message.

        Args:
            text: User text.
        """
        self.conversation_history.append({"role": "user", "content": text})

    def _messages(self, *, final: bool) -> list[Message]:
        messages: list[Message] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages += self.conversation_history
        if final:
            messages.append({"role": "system", "content": FINAL_ANSWER_NOTE})
        return messages

    def _next_call_id(self) -> str:
        self._call_counter += 1
        return f"call_{self._call_counter}"

    def _to_tool_call(self, raw: Any) -> ToolCall:
        function = _field(raw, "function")
        arguments = _field(function, "arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return ToolCall(self._next_call_id(), str(_field(function, "name")), dict(arguments))

    def stream_turn(self, tools: Sequence[ToolSpec] | None = None, *, final: bool = False) -> Iterator[Event]:
        """Run one Ollama turn, streaming text and collecting tool calls.

        Args:
            tools: Tools the model may call.
            final: Omit tools and ask for a final answer.

        Yields:
            Event: Text deltas, tool calls and the turn end.

        Raises:
            LLMError: When Ollama cannot be reached or the stream fails.
        """
        offer_tools = bool(tools) and not final
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(final=final and bool(tools)),
            "stream": True,
            "think": self.think,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if offer_tools and tools:
            kwargs["tools"] = [spec.ollama() for spec in tools]
        turn = _OllamaTurn(hold_json=offer_tools)
        try:
            for chunk in self.client.chat(**kwargs):
                text = turn.feed(chunk)
                if text:
                    yield TextDelta(text)
        except Exception as exc:
            raise LLMError(f"Erro de comunicação com o Ollama: {exc}") from exc

        content = turn.content
        calls = [self._to_tool_call(raw) for raw in turn.raw_calls]
        if offer_tools and tools and not calls:
            recovered = parse_text_tool_call(content, [spec.name for spec in tools])
            if recovered is not None:
                calls = [ToolCall(self._next_call_id(), recovered["name"], recovered["arguments"])]
                content = ""
            elif turn.held_back:
                yield TextDelta(content)

        message: Message = {"role": "assistant", "content": content}
        if turn.thinking:
            message["thinking"] = turn.thinking
        if calls:
            message["tool_calls"] = [{"function": {"name": c.name, "arguments": dict(c.arguments)}} for c in calls]
        self.conversation_history.append(message)

        usage = turn.usage
        prompt_tokens = usage.get("prompt_eval_count") or 0
        if prompt_tokens and prompt_tokens >= 0.9 * self.num_ctx:
            logger.warning("Ollama context almost full (%d/%d tokens); consider 'reset'.", prompt_tokens, self.num_ctx)
        if calls:
            yield ToolCallsReady(tuple(calls))
        yield TurnEnd("tool_use" if calls else "end_turn", usage)

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append tool results as ``role: tool`` messages.

        Args:
            results: Results in call order.
        """
        for result in results:
            self.conversation_history.append({"role": "tool", "tool_name": result.name, "content": result.content})

    def compact_history(self) -> None:
        """Keep only user messages and final assistant answers (saves the small local context)."""
        self.conversation_history = [
            m
            for m in self.conversation_history
            if m["role"] == "user" or (m["role"] == "assistant" and not m.get("tool_calls") and m.get("content"))
        ]


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class SSEAccumulator:
    """Rebuild Anthropic content blocks from Messages API streaming events.

    Handles ``text``, ``thinking`` (+ ``signature``), ``redacted_thinking`` and ``tool_use``
    blocks (``input_json_delta`` fragments are joined and parsed at ``content_block_stop``).
    Unknown events are ignored, as the API requires.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._partial_json: dict[int, list[str]] = {}
        self.stop_reason: str | None = None
        self.usage: dict[str, Any] = {}
        self.error: str | None = None

    def feed(self, event: Mapping[str, Any]) -> str:
        """Consume one decoded SSE ``data`` payload.

        Args:
            event: Decoded JSON event.

        Returns:
            str: Text to display (empty for non-text events).
        """
        kind = event.get("type")
        index = int(event.get("index", 0))
        if kind == "message_start":
            self.usage.update(_field(event.get("message"), "usage") or {})
        elif kind == "content_block_start":
            return self._start(index, event.get("content_block") or {})
        elif kind == "content_block_delta":
            return self._delta(index, event.get("delta") or {})
        elif kind == "content_block_stop":
            self._stop(index)
        elif kind == "message_delta":
            self.stop_reason = (event.get("delta") or {}).get("stop_reason") or self.stop_reason
            self.usage.update(event.get("usage") or {})
        elif kind == "error":
            error = event.get("error") or {}
            self.error = f"{error.get('type', 'error')}: {error.get('message', '')}"
        return ""

    def _start(self, index: int, content_block: Mapping[str, Any]) -> str:
        block = dict(content_block)
        if block.get("type") == "tool_use":
            block["input"] = {}
            self._partial_json[index] = []
        self._blocks[index] = block
        return str(block.get("text") or "") if block.get("type") == "text" else ""

    def _stop(self, index: int) -> None:
        if index not in self._partial_json:
            return
        raw = "".join(self._partial_json.pop(index)).strip()
        try:
            self._blocks[index]["input"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.warning("Invalid tool input JSON from Claude: %s", raw[:200])
            self._blocks[index]["input"] = {}

    def _delta(self, index: int, delta: Mapping[str, Any]) -> str:
        block = self._blocks.setdefault(index, {"type": "text", "text": ""})
        kind = delta.get("type")
        if kind == "text_delta":
            text = str(delta.get("text") or "")
            block["text"] = str(block.get("text") or "") + text
            return text
        if kind == "thinking_delta":
            block["thinking"] = str(block.get("thinking") or "") + str(delta.get("thinking") or "")
        elif kind == "signature_delta":
            block["signature"] = delta.get("signature")
        elif kind == "input_json_delta":
            self._partial_json.setdefault(index, []).append(str(delta.get("partial_json") or ""))
        return ""

    def content(self) -> list[dict[str, Any]]:
        """Return the completed content blocks in order (empty text blocks dropped).

        Returns:
            list[dict[str, Any]]: Anthropic content blocks.
        """
        blocks = [self._blocks[i] for i in sorted(self._blocks)]
        return [b for b in blocks if not (b.get("type") == "text" and not b.get("text"))]


def iter_sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Decode ``data:`` lines of a server-sent-events stream.

    Args:
        lines: Raw text lines.

    Yields:
        dict[str, Any]: Decoded JSON payloads (malformed lines are skipped).
    """
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


class ClaudeChat(AbstractChat):
    """Chat interface for the Claude Messages API (raw HTTP + SSE, like the reference).

    Args:
        model: Claude model id (default ``claude-sonnet-5-5``).
        api_key: API key (env ``ANTHROPIC_API_KEY`` or ``CLAUDE_TOKEN``).
        http: Pre-built ``httpx.Client`` (tests inject a ``MockTransport``).
        max_tokens: Output token cap per turn (thinking counts against it).
        max_retries: Retries for 429/5xx/529 before streaming starts.
        sleep: Sleep function (injectable for tests).

    Raises:
        ValueError: If no API key is configured.
    """

    provider = "claude"
    _RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        max_tokens: int = CLAUDE_MAX_TOKENS,
        max_retries: int = 2,
        sleep: Any = time.sleep,
    ) -> None:
        super().__init__(model or DEFAULT_CLAUDE_MODEL)
        self.api_key = api_key or next((os.environ[k] for k in ENV_ANTHROPIC_KEYS if os.environ.get(k)), "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.http = http or httpx.Client(base_url=CLAUDE_API_URL, timeout=httpx.Timeout(600.0, connect=10.0))
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._sleep = sleep

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": CLAUDE_API_VERSION,
            "content-type": "application/json",
        }

    def _verify_connection(self) -> bool:
        """Verify the API key and model with the free ``GET /v1/models/{id}`` endpoint.

        Returns:
            bool: True if the model is available to this key.
        """
        try:
            response = self.http.get(f"/v1/models/{self.model}", headers=self._headers(), timeout=10.0)
            return response.status_code == 200
        except httpx.HTTPError:
            logger.debug("Claude check failed", exc_info=True)
            return False

    def add_user_message(self, text: str) -> None:
        """Append a user message.

        Args:
            text: User text.
        """
        self.conversation_history.append({"role": "user", "content": text})

    def _payload(self, tools: Sequence[ToolSpec] | None, *, final: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.conversation_history,
            "stream": True,
        }
        if self.system_message:
            payload["system"] = [
                {"type": "text", "text": self.system_message, "cache_control": {"type": "ephemeral"}},
            ]
        if tools:
            payload["tools"] = [spec.anthropic() for spec in tools]
            if final:
                payload["tool_choice"] = {"type": "none"}
        return payload

    def _stream_events(self, payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """POST the request (with retries before the first byte) and yield SSE events.

        Args:
            payload: Request body.

        Yields:
            dict[str, Any]: Decoded SSE events.

        Raises:
            LLMError: On HTTP or transport errors.
        """
        attempt = 0
        while True:
            try:
                yield from self._stream_once(payload, attempt=attempt)
                return
            except _RetryableError as exc:
                logger.warning("Claude API %d; retrying in %.1fs", exc.status_code, exc.wait)
                attempt += 1
                self._sleep(exc.wait)
            except httpx.HTTPError as exc:
                raise LLMError(f"Erro de comunicação com a API do Claude: {exc}") from exc

    def _stream_once(self, payload: Mapping[str, Any], *, attempt: int) -> Iterator[dict[str, Any]]:
        """Send one streaming request.

        Args:
            payload: Request body.
            attempt: Zero-based attempt number.

        Yields:
            dict[str, Any]: Decoded SSE events.

        Raises:
            _RetryableError: On a retryable status while retries remain.
            LLMError: On any other HTTP error status.
        """
        with self.http.stream("POST", "/v1/messages", json=payload, headers=self._headers()) as response:
            if response.status_code in self._RETRY_STATUS and attempt < self.max_retries:
                response.read()
                wait = min(float(response.headers.get("retry-after") or 2**attempt), 60.0)
                raise _RetryableError(response.status_code, wait)
            if response.status_code >= 400:
                response.read()
                raise LLMError(f"Claude API {response.status_code}: {response.text[:500]}")
            yield from iter_sse_events(response.iter_lines())

    def stream_turn(self, tools: Sequence[ToolSpec] | None = None, *, final: bool = False) -> Iterator[Event]:
        """Run one Claude turn over SSE.

        Args:
            tools: Tools the model may call.
            final: Send ``tool_choice: none`` to force a text answer.

        Yields:
            Event: Text deltas, tool calls and the turn end.

        Raises:
            LLMError: On API/stream errors (the turn is not added to history).
        """
        accumulator = SSEAccumulator()
        for event in self._stream_events(self._payload(tools, final=final)):
            text = accumulator.feed(event)
            if text:
                yield TextDelta(text)
        if accumulator.error:
            raise LLMError(f"Erro no streaming do Claude: {accumulator.error}")

        stop_reason = accumulator.stop_reason or "end_turn"
        blocks = accumulator.content()
        calls: list[ToolCall] = []
        if stop_reason == "tool_use":
            calls = [
                ToolCall(str(b["id"]), str(b["name"]), dict(b.get("input") or {}))
                for b in blocks
                if b.get("type") == "tool_use"
            ]
        else:
            # Never leave unanswered tool_use blocks in history (e.g. truncated by max_tokens).
            blocks = [b for b in blocks if b.get("type") != "tool_use"]
        if not any(b.get("type") == "text" for b in blocks) and not calls:
            blocks.append({"type": "text", "text": "(sem resposta)"})
        self.conversation_history.append({"role": "assistant", "content": blocks})
        if calls:
            yield ToolCallsReady(tuple(calls))
        yield TurnEnd(stop_reason, accumulator.usage)

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append all tool results in ONE user message (keeps parallel tool use working).

        Args:
            results: Results in call order.
        """
        self.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                    for r in results
                ],
            }
        )


def get_llm(provider: str, model: str = "", **kwargs: Any) -> AbstractChat:
    """Instantiate the configured LLM chat backend.

    Args:
        provider: ``claude`` or ``ollama``.
        model: Model id (empty = provider default).
        **kwargs: Extra constructor arguments.

    Returns:
        AbstractChat: Chat backend.

    Raises:
        ValueError: For an unknown provider.
    """
    match provider.lower():
        case "claude" | "anthropic":
            return ClaudeChat(model=model or None, **kwargs)
        case "ollama":
            return OllamaChat(model=model or None, **kwargs)
        case _:
            raise ValueError(f"Provedor desconhecido '{provider}'. Use: claude | ollama")
