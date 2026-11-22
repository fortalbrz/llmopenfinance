"""Tests for the LLM chat backends (Ollama via a fake client, Claude via httpx.MockTransport + SSE)."""

import json
from types import SimpleNamespace

import httpx
import pytest

from llmopenfinance.chat import (
    ClaudeChat,
    LLMError,
    OllamaChat,
    SSEAccumulator,
    TextDelta,
    ToolCallsReady,
    TurnEnd,
    get_llm,
    iter_sse_events,
    parse_text_tool_call,
)
from llmopenfinance.tools import BASE_TOOLS, ToolResult

TOOLS = list(BASE_TOOLS)
TOOL_NAMES = [t.name for t in TOOLS]

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class FakeOllamaClient:
    def __init__(self, responses=None, *, fail_show=False, fail_chat=False):
        self.responses = list(responses or [])
        self.requests = []
        self.fail_show = fail_show
        self.fail_chat = fail_chat

    def show(self, model):
        if self.fail_show:
            raise ConnectionError("down")
        return {"model": model}

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail_chat:
            raise ConnectionError("refused")
        return iter(self.responses.pop(0))


def chunk(content="", *, tool_calls=None, thinking=None, done=False, **extra):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if thinking is not None:
        message["thinking"] = thinking
    return {"message": message, "done": done, **extra}


def test_ollama_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_SERVER", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    chat = OllamaChat(client=FakeOllamaClient())
    assert chat.model == "qwen3.5:9b"
    assert chat.base_url == "http://localhost:11434"
    assert chat.think is False


def test_ollama_real_client_constructed(monkeypatch):
    monkeypatch.setenv("OLLAMA_SERVER", "http://example:1234")
    chat = OllamaChat(model="m")
    assert chat.base_url == "http://example:1234"
    assert chat.client is not None


def test_ollama_verify_connection():
    assert OllamaChat(client=FakeOllamaClient())._verify_connection() is True
    assert OllamaChat(client=FakeOllamaClient(fail_show=True))._verify_connection() is False


def test_ollama_plain_chat_stream():
    client = FakeOllamaClient([[chunk("Olá"), chunk(", tudo bem?", done=True, prompt_eval_count=10, eval_count=3)]])
    chat = OllamaChat(client=client)
    chat.set_system_message("sys")
    assert chat.chat("oi") == "Olá, tudo bem?"
    request = client.requests[0]
    assert request["messages"][0] == {"role": "system", "content": "sys"}
    assert request["think"] is False
    assert request["options"]["num_ctx"] == 16384
    assert "tools" not in request
    assert chat.get_history()[-1] == {"role": "assistant", "content": "Olá, tudo bem?"}


def test_ollama_tool_round():
    calls = [
        {"function": {"name": "get_accounts", "arguments": {}}},
        SimpleNamespace(function=SimpleNamespace(name="get_loans", arguments='{"x": 1}')),
    ]
    client = FakeOllamaClient([[chunk(thinking="hmm"), chunk(tool_calls=calls, done=True)]])
    chat = OllamaChat(client=client)
    chat.add_user_message("saldo?")
    events = list(chat.stream_turn(TOOLS))
    ready = next(e for e in events if isinstance(e, ToolCallsReady))
    assert [c.name for c in ready.calls] == ["get_accounts", "get_loans"]
    assert ready.calls[1].arguments == {"x": 1}
    assert ready.calls[0].id != ready.calls[1].id
    assert isinstance(events[-1], TurnEnd)
    assert isinstance(events[-1], TurnEnd)
    assert events[-1].stop_reason == "tool_use"
    assert client.requests[0]["tools"][0]["type"] == "function"
    assistant = chat.conversation_history[-1]
    assert assistant["thinking"] == "hmm"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_accounts"
    chat.add_tool_results([ToolResult(c.id, c.name, "{}") for c in ready.calls])
    assert chat.conversation_history[-1] == {"role": "tool", "tool_name": "get_loans", "content": "{}"}


def test_ollama_invalid_json_arguments_become_empty():
    calls = [{"function": {"name": "get_accounts", "arguments": "{bad"}}]
    chat = OllamaChat(client=FakeOllamaClient([[chunk(tool_calls=calls, done=True)]]))
    chat.add_user_message("x")
    ready = next(e for e in chat.stream_turn(TOOLS) if isinstance(e, ToolCallsReady))
    assert ready.calls[0].arguments == {}


def test_ollama_json_text_fallback_is_hidden_and_converted():
    text = '{"name": "get_accounts", "arguments": {}}'
    chat = OllamaChat(client=FakeOllamaClient([[chunk(text[:10]), chunk(text[10:], done=True)]]))
    chat.add_user_message("x")
    events = list(chat.stream_turn(TOOLS))
    assert not any(isinstance(e, TextDelta) for e in events)
    assert next(e for e in events if isinstance(e, ToolCallsReady)).calls[0].name == "get_accounts"
    assert not chat.conversation_history[-1]["content"]


def test_ollama_held_back_text_released_when_not_a_tool_call():
    chat = OllamaChat(client=FakeOllamaClient([[chunk('{"resposta": 1}', done=True)]]))
    chat.add_user_message("x")
    events = list(chat.stream_turn(TOOLS))
    assert events[0] == TextDelta('{"resposta": 1}')


def test_ollama_final_round_omits_tools_and_adds_note():
    client = FakeOllamaClient([[chunk("resposta final", done=True)]])
    chat = OllamaChat(client=client)
    chat.add_user_message("x")
    list(chat.stream_turn(TOOLS, final=True))
    assert "tools" not in client.requests[0]
    assert "Limite de chamadas" in client.requests[0]["messages"][-1]["content"]


def test_ollama_context_warning(caplog):
    chat = OllamaChat(client=FakeOllamaClient([[chunk("ok", done=True, prompt_eval_count=16000)]]))
    chat.add_user_message("x")
    list(chat.stream_turn(None))
    assert "context almost full" in caplog.text


def test_ollama_errors_become_llm_error():
    chat = OllamaChat(client=FakeOllamaClient(fail_chat=True))
    chat.add_user_message("x")
    with pytest.raises(LLMError, match="Ollama"):
        list(chat.stream_turn(None))


def test_ollama_compact_history():
    chat = OllamaChat(client=FakeOllamaClient())
    chat.conversation_history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{}]},
        {"role": "tool", "tool_name": "t", "content": "{}"},
        {"role": "assistant", "content": "answer"},
    ]
    chat.compact_history()
    assert [m["role"] for m in chat.conversation_history] == ["user", "assistant"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"name": "get_accounts", "arguments": {}}', {"name": "get_accounts", "arguments": {}}),
        ('```json\n{"name": "get_loans", "parameters": {"a": 1}}\n```', {"name": "get_loans", "arguments": {"a": 1}}),
        (
            '{"function": {"name": "get_loans", "arguments": "{\\"a\\": 2}"}}',
            {"name": "get_loans", "arguments": {"a": 2}},
        ),
        ('{"name": "unknown", "arguments": {}}', None),
        ('{"name": "get_loans", "arguments": "{bad"}', None),
        ('{"name": "get_loans", "arguments": [1]}', None),
        ("[1, 2]", None),
        ("{not json", None),
        ("texto normal", None),
    ],
)
def test_parse_text_tool_call(text, expected):
    assert parse_text_tool_call(text, TOOL_NAMES) == expected


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


def sse(*events):
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return (": comment\n" + body).encode()


def text_stream(text, stop="end_turn"):
    return sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[:3]}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[3:]}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )


def tool_stream():
    return sse(
        {"type": "message_start", "message": {"usage": {}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "pensando"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "Vou ver. "}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "spending_by_category", "input": {}},
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ""}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"months": '},
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "3}"}},
        {"type": "content_block_stop", "index": 2},
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {"type": "tool_use", "id": "tu_2", "name": "get_accounts", "input": {}},
        },
        {"type": "content_block_stop", "index": 3},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
    )


class ClaudeApi:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(200 if request.url.path.endswith("claude-sonnet-5-5") else 404, json={})
        status, body, headers = self.responses.pop(0)
        return httpx.Response(status, content=body, headers=headers or {})

    def payload(self, index=-1):
        return json.loads(self.requests[index].content)


def claude(api, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(api), base_url="https://api.anthropic.com")
    return ClaudeChat(api_key="sk-test", http=http, sleep=lambda _s: None, **kwargs)


def test_claude_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeChat()


def test_claude_reads_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_TOKEN", "tok")
    chat = ClaudeChat()
    assert chat.api_key == "tok"
    assert chat.model == "claude-sonnet-5-5"


def test_claude_verify_connection():
    assert claude(ClaudeApi())._verify_connection() is True
    assert claude(ClaudeApi(), model="claude-unknown")._verify_connection() is False

    def broken(request):
        raise httpx.ConnectError("x", request=request)

    http = httpx.Client(transport=httpx.MockTransport(broken), base_url="https://api.anthropic.com")
    assert ClaudeChat(api_key="k", http=http)._verify_connection() is False


def test_claude_plain_stream_payload():
    api = ClaudeApi((200, text_stream("Olá mundo"), None))
    chat = claude(api)
    chat.set_system_message("Você é um assistente.")
    assert chat.chat("oi") == "Olá mundo"
    payload = api.payload()
    assert payload["system"][0]["text"] == "Você é um assistente."
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"] == [{"role": "user", "content": "oi"}]
    assert "temperature" not in payload
    assert "tools" not in payload
    assert payload["stream"] is True
    headers = api.requests[-1].headers
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["x-api-key"] == "sk-test"
    assert chat.conversation_history[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Olá mundo"}]}


def test_claude_tool_use_turn_and_results():
    api = ClaudeApi((200, tool_stream(), None))
    chat = claude(api)
    chat.add_user_message("gastos?")
    events = list(chat.stream_turn(TOOLS))
    assert events[0] == TextDelta("Vou ver. ")
    ready = next(e for e in events if isinstance(e, ToolCallsReady))
    assert [(c.id, c.name, dict(c.arguments)) for c in ready.calls] == [
        ("tu_1", "spending_by_category", {"months": 3}),
        ("tu_2", "get_accounts", {}),
    ]
    assert isinstance(events[-1], TurnEnd)
    assert events[-1].stop_reason == "tool_use"
    blocks = chat.conversation_history[-1]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "pensando", "signature": "sig"}
    assert blocks[2]["input"] == {"months": 3}
    payload = api.payload()
    assert {t["name"] for t in payload["tools"]} == set(TOOL_NAMES)
    assert "tool_choice" not in payload
    chat.add_tool_results(
        [ToolResult("tu_1", "spending_by_category", "[]"), ToolResult("tu_2", "get_accounts", "x", True)]
    )
    results = chat.conversation_history[-1]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["tu_1", "tu_2"]
    assert results["content"][1]["is_error"] is True


def test_claude_final_round_sets_tool_choice_none():
    api = ClaudeApi((200, text_stream("fim"), None))
    chat = claude(api)
    chat.add_user_message("x")
    list(chat.stream_turn(TOOLS, final=True))
    assert api.payload()["tool_choice"] == {"type": "none"}


def test_claude_truncated_tool_use_is_dropped():
    stream = sse(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "get_accounts"},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
    )
    chat = claude(ClaudeApi((200, stream, None)))
    chat.add_user_message("x")
    events = list(chat.stream_turn(TOOLS))
    assert not any(isinstance(e, ToolCallsReady) for e in events)
    assert isinstance(events[-1], TurnEnd)
    assert events[-1].stop_reason == "max_tokens"
    assert chat.conversation_history[-1]["content"] == [{"type": "text", "text": "(sem resposta)"}]


def test_claude_retries_then_succeeds():
    api = ClaudeApi((529, b'{"error": "overloaded"}', {"retry-after": "1"}), (200, text_stream("ok!"), None))
    chat = claude(api)
    assert chat.chat("x") == "ok!"
    assert len(api.requests) == 2


def test_claude_http_error_raises():
    chat = claude(ClaudeApi((400, b'{"error": {"message": "bad request"}}', None)))
    with pytest.raises(LLMError, match="400"):
        chat.chat("x")


def test_claude_retry_exhausted_raises():
    api = ClaudeApi(*[(429, b"{}", None)] * 3)
    with pytest.raises(LLMError, match="429"):
        claude(api).chat("x")


def test_claude_transport_error_raises():
    def broken(request):
        raise httpx.ReadTimeout("slow", request=request)

    http = httpx.Client(transport=httpx.MockTransport(broken), base_url="https://api.anthropic.com")
    with pytest.raises(LLMError, match="comunicação"):
        ClaudeChat(api_key="k", http=http).chat("x")


def test_claude_stream_error_event_raises_and_keeps_history_clean():
    stream = sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    chat = claude(ClaudeApi((200, stream, None)))
    chat.add_user_message("x")
    with pytest.raises(LLMError, match="overloaded_error"):
        list(chat.stream_turn(None))
    assert chat.conversation_history == [{"role": "user", "content": "x"}]


def test_sse_accumulator_edge_cases(caplog):
    acc = SSEAccumulator()
    assert acc.feed({"type": "content_block_delta", "index": 5, "delta": {"type": "text_delta", "text": "a"}}) == "a"
    acc.feed({"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "x"}})
    acc.feed({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t", "name": "n"}})
    acc.feed(
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{oops"}}
    )
    acc.feed({"type": "content_block_stop", "index": 2})
    acc.feed({"type": "content_block_stop", "index": 1})
    acc.feed({"type": "unknown_future_event"})
    acc.feed({"type": "content_block_delta", "index": 1, "delta": {"type": "future_delta"}})
    assert "Invalid tool input JSON" in caplog.text
    content = acc.content()
    assert [b["type"] for b in content] == ["redacted_thinking", "tool_use", "text"]
    assert content[1]["input"] == {}


def test_iter_sse_events_skips_noise():
    lines = iter(["event: x", "data: ", "data: [DONE]", "data: {bad", "data: [1]", 'data: {"type": "ok"}', ""])
    assert list(iter_sse_events(lines)) == [{"type": "ok"}]


def test_history_helpers():
    chat = claude(ClaudeApi())
    chat.add_user_message("a")
    chat.add_user_message("b")
    chat.truncate_history(1)
    assert chat.get_history() == [{"role": "user", "content": "a"}]
    chat.compact_history()  # no-op for Claude (append-only history)
    assert len(chat.conversation_history) == 1
    chat.reset_conversation()
    assert chat.get_history() == []


def test_get_llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(get_llm("claude"), ClaudeChat)
    assert isinstance(get_llm("Anthropic", "claude-haiku-5-5"), ClaudeChat)
    assert get_llm("ollama", "m", client=FakeOllamaClient()).model == "m"
    with pytest.raises(ValueError, match="desconhecido"):
        get_llm("gpt")
