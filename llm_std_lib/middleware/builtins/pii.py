"""
PII redaction middleware.

Scans request prompts (and optionally responses) for personally identifiable
information and replaces detected entities with placeholder tokens before
they reach the provider or the caller.

Detected patterns:
    - ``[EMAIL]``       — e-mail addresses
    - ``[PHONE]``       — US/international phone numbers
    - ``[SSN]``         — US Social Security Numbers (XXX-XX-XXXX)
    - ``[CREDIT_CARD]`` — 13–16-digit card numbers (Visa, MC, Amex, Discover)
    - ``[IP_ADDRESS]``  — IPv4 addresses
"""

from __future__ import annotations

import re

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

    Args:
        redact_prompt: Redact ``ctx.prompt`` before sending to the provider.
            Default: ``True``.
        redact_response: Redact ``response.text`` before returning to caller.
            Default: ``False`` (responses rarely contain raw PII, but can be
            enabled for extra caution).

    Example::

        stack = MiddlewareStack([PIIRedactorMiddleware()])
        # "My email is alice@example.com" → "My email is [EMAIL]"
    """

    def __init__(
        self,
        redact_prompt: bool = True,
        redact_response: bool = False,
    ) -> None:
        self._redact_prompt = redact_prompt
        self._redact_response = redact_response

    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        if self._redact_prompt:
            ctx.prompt = _redact(ctx.prompt)
            if ctx.system_prompt:
                ctx.system_prompt = _redact(ctx.system_prompt)
        return ctx

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        if self._redact_response:
            response.text = _redact(response.text)
        return response
