"""
PII redaction middleware.

Scans request prompts (and optionally responses) for personally identifiable
information and replaces detected entities with placeholder tokens before
they reach the provider or the caller.

Two engines are available:

* **Regex engine** (default, no extra deps) — fast, covers email, phone,
  SSN, credit cards, IPv4.
* **Presidio engine** (optional, NER-based) — covers names, addresses, and
  20+ entity types via spaCy models.  Install with::

      pip install llm-std-lib[presidio]

Detected patterns (regex engine):
    - ``[EMAIL]``       — e-mail addresses
    - ``[PHONE]``       — US/international phone numbers
    - ``[SSN]``         — US Social Security Numbers (XXX-XX-XXXX)
    - ``[CREDIT_CARD]`` — 13–16-digit card numbers (Visa, MC, Amex, Discover)
    - ``[IP_ADDRESS]``  — IPv4 addresses
"""

from __future__ import annotations

import re
from typing import Any

from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext

# ---------------------------------------------------------------------------
# Compiled patterns (broad for high recall)
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # E-mail
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        "[EMAIL]",
    ),
    # Credit card (13–16 digits, optional separators every 4 digits)
    (
        re.compile(
            r"\b(?:\d[ \-]?){13,15}\d\b"
        ),
        "[CREDIT_CARD]",
    ),
    # SSN  XXX-XX-XXXX
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN]",
    ),
    # Phone — US and international (+1 optional, various separators)
    (
        re.compile(
            r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]\d{4}\b"
        ),
        "[PHONE]",
    ),
    # IPv4
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP_ADDRESS]",
    ),
]


def _redact(text: str) -> str:
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


class PIIRedactorMiddleware(BaseMiddleware):
    """Redacts PII from prompts and optionally from provider responses.

    By default uses a fast regex engine (no extra dependencies). Pass a
    :class:`~llm_std_lib.middleware.builtins.presidio_engine.PresidioPIIEngine`
    instance to enable NER-based detection of names, addresses, and 20+ entity
    types via Microsoft Presidio.

    Args:
        engine: Optional PII engine with a ``redact(text, language) -> str``
            method. When ``None``, the built-in regex engine is used.
        language: Language hint forwarded to the engine (e.g. ``"en"``,
            ``"ru"``). Ignored by the regex engine.
        redact_prompt: Redact ``ctx.prompt`` before sending to the provider.
            Default: ``True``.
        redact_response: Redact ``response.text`` before returning to caller.
            Default: ``False``.

    Example — regex (default)::

        pii = PIIRedactorMiddleware()
        # "My email is alice@example.com" → "My email is [EMAIL]"

    Example — Presidio (NER, names + addresses)::

        from llm_std_lib.middleware.builtins.presidio_engine import PresidioPIIEngine

        engine = PresidioPIIEngine(languages=["en", "ru"])
        pii = PIIRedactorMiddleware(engine=engine, language="ru")
        # "Меня зовут Иван Петров" → "Меня зовут <PERSON>"
    """

    def __init__(
        self,
        engine: Any | None = None,
        language: str = "en",
        redact_prompt: bool = True,
        redact_response: bool = False,
    ) -> None:
        self._engine = engine
        self._language = language
        self._redact_prompt = redact_prompt
        self._redact_response = redact_response

    def _do_redact(self, text: str) -> str:
        if self._engine is not None:
            return self._engine.redact(text, language=self._language)  # type: ignore[no-any-return]
        return _redact(text)

    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        if self._redact_prompt:
            ctx.prompt = self._do_redact(ctx.prompt)
            if ctx.system_prompt:
                ctx.system_prompt = self._do_redact(ctx.system_prompt)
        return ctx

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        if self._redact_response:
            response.text = self._do_redact(response.text)
        return response
