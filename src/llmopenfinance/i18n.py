"""Language resolution and message lookup (pt-BR default, English optional).

The language is process-wide: it is resolved once from ``LLMOPENFINANCE_LANG`` on the first
:func:`get_language` or :func:`t` call and then frozen, so one process never mixes languages.
Catalogs live in ``messages_pt`` and ``messages_en``; they are read on every lookup, never copied.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from types import ModuleType

from llmopenfinance import messages_en, messages_pt
from llmopenfinance.constants import DEFAULT_LANGUAGE, ENV_LANG, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

_PT = "pt-BR"
_EN = "en"
_CATALOGS: dict[str, ModuleType] = {_PT: messages_pt, _EN: messages_en}
_DATE_FORMATS = {_PT: "%d/%m/%Y", _EN: "%Y-%m-%d"}


@dataclass
class _State:
    """Mutable process-wide state, guarded by ``_LOCK`` (the web server calls :func:`t` from threads).

    Attributes:
        language: Frozen language code, or ``None`` until first use.
        warned: ``(kind, key)`` pairs already logged, so each problem warns once.
    """

    language: str | None = None
    warned: set[tuple[str, str]] = field(default_factory=set)


_STATE = _State()
_LOCK = threading.Lock()


def normalize_language(value: str) -> str:
    """Map a user-supplied language code to a supported one.

    Case-insensitive, surrounding whitespace ignored, ``_`` treated as ``-``. The primary subtag decides:
    ``pt`` gives ``pt-BR`` and ``en`` gives ``en``.

    Args:
        value: Raw code such as ``"pt"``, ``"PT_br"`` or ``" en-US "``.

    Returns:
        ``"pt-BR"`` or ``"en"``.

    Raises:
        ValueError: If the primary subtag is neither ``pt`` nor ``en``; the message lists the supported values.
    """
    primary = value.strip().lower().replace("_", "-").split("-")[0]
    if primary == "pt":
        return _PT
    if primary == "en":
        return _EN
    raise ValueError(f"Unsupported language {value!r}; supported values: {', '.join(SUPPORTED_LANGUAGES)}")


def get_language() -> str:
    """Return the process language, resolving it from the environment on first use.

    An unset, empty or blank ``LLMOPENFINANCE_LANG`` means the default. An invalid value raises and leaves
    the language unresolved, so a later call can still succeed.

    Returns:
        ``"pt-BR"`` or ``"en"``.
    """
    with _LOCK:
        if _STATE.language is None:
            raw = os.environ.get(ENV_LANG, "").strip()
            _STATE.language = normalize_language(raw or DEFAULT_LANGUAGE)
        return _STATE.language


def reset_language(code: str | None = None) -> None:
    """Clear or force the frozen language. Meant for tests only.

    Also forgets which warnings were already logged, so tests do not depend on each other.

    Args:
        code: Language to freeze (normalised first), or ``None`` to resolve again from the environment.
    """
    language = None if code is None else normalize_language(code)
    with _LOCK:
        _STATE.language = language
        _STATE.warned.clear()


def _warn_once(kind: str, key: str, message: str, *args: object) -> None:
    """Log a warning the first time a ``(kind, key)`` problem is seen.

    Args:
        kind: Problem category (``missing`` or ``format``).
        key: Catalog key the problem is about.
        message: Lazy ``%``-style log message.
        *args: Arguments for ``message``.
    """
    with _LOCK:
        if (kind, key) in _STATE.warned:
            return
        _STATE.warned.add((kind, key))
    logger.warning(message, *args)


def t(key: str, **values: object) -> str:
    """Translate ``key`` into the process language and fill its ``{name}`` placeholders.

    A key missing from the active catalog falls back to the pt-BR text; a key missing everywhere returns the
    key itself; a template that cannot be formatted with ``values`` (missing placeholder, malformed template,
    or a value its format spec rejects, such as ``None`` under ``{n:.2f}``) is returned unformatted. Each
    problem is logged once per key. The only error that can escape is the ``ValueError`` that
    :func:`get_language` raises on first use when ``LLMOPENFINANCE_LANG`` is invalid. It is not listed in a
    ``Raises:`` section because ruff's DOC502 rejects exceptions the function does not raise itself.

    Args:
        key: Dotted catalog key such as ``"area.name"``.
        **values: Values for the template placeholders.

    Returns:
        The localised, formatted text.
    """
    language = get_language()
    template = _CATALOGS[language].MESSAGES.get(key)
    if template is None:
        template = _CATALOGS[_PT].MESSAGES.get(key)
        if template is None:
            _warn_once("missing", key, "Missing message key %r in every catalog", key)
            return key
        _warn_once("missing", key, "Message key %r missing from the %s catalog; using pt-BR", key, language)
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
        _warn_once(
            "format", key, "Cannot format message %r (%s: %s); returning the template", key, type(exc).__name__, exc
        )
        return template


def format_date(value: date) -> str:
    """Format a date for people: ``DD/MM/YYYY`` in pt-BR, ISO ``YYYY-MM-DD`` in English.

    Args:
        value: The date to show.

    Returns:
        The formatted date.
    """
    return value.strftime(_DATE_FORMATS[get_language()])
