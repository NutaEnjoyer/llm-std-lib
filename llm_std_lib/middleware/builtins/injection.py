"""
Prompt injection detection middleware.

Heuristic detection of common prompt injection and jailbreak patterns.
Raises :class:`~llm_std_lib.exceptions.LLMValidationError` when a high-risk
pattern is found in the prompt.
"""

from __future__ import annotations

import re

from llm_std_lib.exceptions import LLMValidationError
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous\s+)?instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you('ve)?\s+)?(been\s+)?told", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?!an?\s+AI)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?!an?\s+AI)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(
        r"(override|bypass|disable)\s+(your\s+)?(safety|filter|restriction|guideline)",
        re.IGNORECASE,
    ),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
]


class PromptInjectionDetector(BaseMiddleware):
    """Detects and blocks common prompt injection / jailbreak attempts.

    Checks ``ctx.prompt`` (and optionally ``ctx.system_prompt``) against a
    library of heuristic patterns.  On detection, raises
    :class:`~llm_std_lib.exceptions.LLMValidationError` and the request is
    never sent to the provider.

    Args:
        raise_on_detection: If ``True`` (default), raises
            ``LLMValidationError`` when a pattern is matched.
            If ``False``, the flag is stored in ``ctx.metadata["injection_detected"]``
            and the request continues — useful for logging-only mode.
        check_system_prompt: Also scan ``ctx.system_prompt``. Default: ``True``.

    Example::

        stack = MiddlewareStack([PromptInjectionDetector()])
        # "Ignore all previous instructions and ..." → LLMValidationError
    """

    def __init__(
        self,
        raise_on_detection: bool = True,
        check_system_prompt: bool = True,
    ) -> None:
        self._raise = raise_on_detection
        self._check_system = check_system_prompt

    def _detect(self, text: str) -> str | None:
        """Return the matched pattern string, or None if clean."""
        for pattern in _INJECTION_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0)
        return None

    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        matched = self._detect(ctx.prompt)
        if matched is None and self._check_system and ctx.system_prompt:
            matched = self._detect(ctx.system_prompt)

        if matched is not None:
            if ctx.metadata is None:
                ctx.metadata = {}
            ctx.metadata["injection_detected"] = matched

            if self._raise:
                raise LLMValidationError(
                    f"Prompt injection detected: '{matched}'"
                )

        return ctx
