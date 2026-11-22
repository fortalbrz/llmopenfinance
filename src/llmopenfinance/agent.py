"""The financial assistant: system prompt + provider-agnostic tool loop.

For each user question: stream a model turn; if the model requests tools, execute them against
the read-only DuckDB connection, feed results back and repeat (at most ``max_rounds`` tool
rounds, then force a final answer). Answers follow the process language (Brazilian Portuguese by default).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import date

from llmopenfinance.analytics import today_brt
from llmopenfinance.chat import AbstractChat, LLMError, TextDelta, ToolCallsReady, TurnEnd
from llmopenfinance.constants import MAX_TOOL_ROUNDS
from llmopenfinance.i18n import format_date, t
from llmopenfinance.tools import ToolCall, ToolRegistry

logger = logging.getLogger(__name__)


def system_prompt() -> str:
    """Return the localised system prompt (without date or snapshot).

    Evaluated on every call so it follows the process language; never cache it in a module constant.

    Returns:
        str: The instruction text in the process language.
    """
    return t("prompt.system")


def build_system_prompt(snapshot: str, today: date | None = None) -> str:
    """Compose the system prompt with the current financial snapshot.

    Args:
        snapshot: Output of :func:`llmopenfinance.analytics.build_snapshot`.
        today: Reference date.

    Returns:
        str: Full system prompt.
    """
    ref = today or today_brt()
    return f"{system_prompt()}\n{t('prompt.today', date=format_date(ref))}\n\n{snapshot}"


@dataclass(frozen=True)
class ToolStatus:
    """A tool call about to run (for the UI).

    Attributes:
        call: The call.
    """

    call: ToolCall


@dataclass(frozen=True)
class AssistantError:
    """A recoverable failure while answering (shown to the user; the loop continues).

    Attributes:
        message: Human-readable error.
    """

    message: str


AssistantEvent = TextDelta | ToolStatus | AssistantError


def stop_reason_message(reason: str) -> str | None:
    """Return the localised notice for a model stop reason that deserves one.

    Args:
        reason: Provider-normalised stop reason (``end_turn``, ``max_tokens``, ``refusal`` ...).

    Returns:
        str | None: The message, or ``None`` when the reason needs no notice.
    """
    if reason == "max_tokens":
        return t("agent.stop.max_tokens")
    if reason == "refusal":
        return t("agent.stop.refusal")
    return None


class FinanceAssistant:
    """Tool-using financial assistant over one chat backend.

    Args:
        chat: LLM backend.
        registry: Tool registry bound to the read-only store.
        snapshot: Callable returning the current financial snapshot text.
        max_rounds: Maximum tool rounds per question.
    """

    def __init__(
        self,
        chat: AbstractChat,
        registry: ToolRegistry,
        snapshot: Callable[[], str],
        *,
        max_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.chat = chat
        self.registry = registry
        self._snapshot = snapshot
        self.max_rounds = max_rounds
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the system prompt from fresh data and start a new conversation.

        The conversation is reset because Claude's prompt cache / preserved thinking require the
        system prompt of an ongoing conversation to stay byte-identical.
        """
        self.chat.set_system_message(build_system_prompt(self._snapshot()))
        self.chat.reset_conversation()
        self.registry.reset_cache()

    def ask(self, question: str) -> Generator[AssistantEvent]:
        """Answer one question, streaming text and tool-status events.

        Never raises for LLM failures: they are yielded as :class:`AssistantError` and the
        history is rolled back so the next question starts clean.

        Args:
            question: User question.

        Yields:
            AssistantEvent: Text deltas, tool statuses and errors.
        """
        history_length = len(self.chat.conversation_history)
        self.registry.reset_cache()
        self.chat.add_user_message(question)
        completed = False
        try:
            yield from self._rounds()
            completed = True
        except LLMError as exc:
            logger.warning("LLM failure: %s", exc)
            self.chat.truncate_history(history_length)
            yield AssistantError(str(exc))
            return
        finally:
            # Ctrl+C, a closed generator or any unexpected error must not leave a dangling tool_use turn,
            # which would make every later request invalid.
            if not completed:
                self.chat.truncate_history(history_length)
        self.chat.compact_history()

    def _rounds(self) -> Iterator[AssistantEvent]:
        """Run model turns and tool rounds until the model answers (or the budget is spent).

        Yields:
            AssistantEvent: Text deltas, tool statuses and stop-reason errors.
        """
        for round_index in range(self.max_rounds + 1):
            calls: tuple[ToolCall, ...] = ()
            stop_reason = "end_turn"
            for event in self.chat.stream_turn(self.registry.specs, final=round_index == self.max_rounds):
                if isinstance(event, TextDelta):
                    yield event
                elif isinstance(event, ToolCallsReady):
                    calls = event.calls
                elif isinstance(event, TurnEnd):
                    stop_reason = event.stop_reason
            if not calls:
                message = stop_reason_message(stop_reason)
                if message is not None:
                    yield AssistantError(message)
                return
            for call in calls:
                yield ToolStatus(call)
            self.chat.add_tool_results(self.registry.run_all(calls))

    def answer(self, question: str) -> str:
        """Answer a question and return only the final text (no tool statuses).

        Args:
            question: User question.

        Returns:
            str: Concatenated answer text (errors included as ``[erro] ...`` lines).
        """
        parts: list[str] = []
        for event in self.ask(question):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, AssistantError):
                parts.append(f"\n[erro] {event.message}")
        return "".join(parts)
