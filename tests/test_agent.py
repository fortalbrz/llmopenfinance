"""Tests for the provider-agnostic assistant loop with a scripted fake LLM."""

from collections.abc import Iterator, Sequence

import pytest

from llmopenfinance.agent import AssistantError, FinanceAssistant, ToolStatus, build_system_prompt
from llmopenfinance.chat import AbstractChat, Event, LLMError, TextDelta, ToolCallsReady, TurnEnd
from llmopenfinance.tools import ToolCall, ToolRegistry, ToolResult, ToolSpec
from tests.conftest import TODAY


class ScriptedChat(AbstractChat):
    """Fake backend: each turn replays the next scripted list of events (or raises)."""

    provider = "fake"

    def __init__(self, turns):
        super().__init__("fake-model")
        self.turns = list(turns)
        self.finals: list[bool] = []
        self.tool_results: list[ToolResult] = []
        self.compacted = 0

    def _verify_connection(self) -> bool:
        return True

    def add_user_message(self, text: str) -> None:
        self.conversation_history.append({"role": "user", "content": text})

    def stream_turn(self, tools: Sequence[ToolSpec] | None = None, *, final: bool = False) -> Iterator[Event]:
        self.finals.append(final)
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        self.conversation_history.append({"role": "assistant", "content": "…"})
        yield from turn

    def add_tool_results(self, results) -> None:
        self.tool_results.extend(results)
        self.conversation_history.append({"role": "user", "content": "results"})

    def compact_history(self) -> None:
        self.compacted += 1


def answer(text):
    return [TextDelta(text), TurnEnd("end_turn")]


def tool_turn(*calls):
    return [ToolCallsReady(tuple(calls)), TurnEnd("tool_use")]


def make(con, turns, **kwargs):
    chat = ScriptedChat(turns)
    assistant = FinanceAssistant(chat, ToolRegistry(lambda: con), lambda: "SNAPSHOT", **kwargs)
    return chat, assistant


def test_build_system_prompt():
    prompt = build_system_prompt("SNAP", today=TODAY)
    assert "português do Brasil" in prompt
    assert "22/11/2026" in prompt
    assert prompt.endswith("SNAP")
    # Regression: without this rule qwen3.5:9b answered "top expenses" from the aggregates and invented values.
    assert "APENAS agregados" in prompt
    assert "CHAME UMA FERRAMENTA" in prompt


def test_refresh_sets_system_message_and_resets(con):
    chat, assistant = make(con, [])
    chat.conversation_history.append({"role": "user", "content": "old"})
    assistant.refresh()
    assert "SNAPSHOT" in chat.system_message
    assert chat.conversation_history == []


def test_direct_answer(con):
    chat, assistant = make(con, [answer("Seu saldo é R$ 10,00.")])
    assert assistant.answer("saldo?") == "Seu saldo é R$ 10,00."
    assert chat.compacted == 1


def test_tool_round_then_answer(con):
    call = ToolCall("c1", "get_accounts", {})
    chat, assistant = make(con, [tool_turn(call), answer("ok")])
    events = list(assistant.ask("saldo?"))
    assert events[0] == ToolStatus(call)
    assert events[-1] == TextDelta("ok")
    assert chat.tool_results[0].call_id == "c1"
    assert not chat.tool_results[0].is_error
    assert chat.finals == [False, False]


def test_round_budget_forces_final_answer(con):
    call = ToolCall("c", "get_loans", {})
    turns = [tool_turn(call), tool_turn(call), answer("final")]
    chat, assistant = make(con, turns, max_rounds=2)
    text = assistant.answer("x")
    assert text == "final"
    assert chat.finals == [False, False, True]
    assert chat.tool_results[1].content.startswith("[Você já chamou")


def test_stop_reasons_reported(con):
    _, assistant = make(con, [[TextDelta("parcial"), TurnEnd("max_tokens")]])
    assert "limite de tokens" in assistant.answer("x")
    _, assistant = make(con, [[TurnEnd("refusal")]])
    events = list(assistant.ask("x"))
    assert isinstance(events[-1], AssistantError)


def test_llm_error_rolls_back_history(con):
    chat, assistant = make(con, [LLMError("API fora do ar")])
    chat.conversation_history.append({"role": "user", "content": "previous"})
    events = list(assistant.ask("x"))
    assert events == [AssistantError("API fora do ar")]
    assert chat.conversation_history == [{"role": "user", "content": "previous"}]
    assert chat.compacted == 0


def test_llm_error_mid_tool_loop_rolls_back(con):
    call = ToolCall("c", "get_accounts", {})
    chat, assistant = make(con, [tool_turn(call), LLMError("caiu")])
    assert "[erro] caiu" in assistant.answer("x")
    assert chat.conversation_history == []


def test_unexpected_error_rolls_back_history(con):
    """Regression: only LLMError rolled back, leaving a dangling tool_use after other failures."""
    call = ToolCall("c", "get_accounts", {})
    chat, assistant = make(con, [tool_turn(call), RuntimeError("bug")])
    with pytest.raises(RuntimeError):
        list(assistant.ask("x"))
    assert chat.conversation_history == []


def test_interrupted_answer_rolls_back_history(con):
    call = ToolCall("c", "get_accounts", {})
    chat, assistant = make(con, [tool_turn(call), answer("never")])
    stream = assistant.ask("x")
    assert isinstance(next(stream), ToolStatus)
    stream.close()  # e.g. Ctrl+C while rendering
    assert chat.conversation_history == []
    assert chat.compacted == 0
