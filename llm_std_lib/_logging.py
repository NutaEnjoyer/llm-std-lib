"""
Structured logging setup for llm_std_lib.

Configures structlog with JSON output and ensures API keys never appear in logs.
All library code should obtain loggers via get_logger(), not logging.getLogger().
"""

from __future__ import annotations

import logging
import re
from collections.abc import MutableMapping
from typing import Any

import structlog

# Patterns that look like API keys — redacted before any log output.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),          # OpenAI / generic
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),       # Anthropic
    re.compile(r"AIza[A-Za-z0-9\-_]{30,}"),          # Google
]
_REDACTED = "**REDACTED**"


def _redact_secrets(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: replace API key patterns in all string values."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern in _SECRET_PATTERNS:
                value = pattern.sub(_REDACTED, value)
            event_dict[key] = value
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog for the library.

    Should be called once at LLMClient initialisation. Safe to call multiple
    times — subsequent calls reconfigure but do not duplicate handlers.

    Args:
        level: Python logging level name (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, output structured JSON; otherwise use a human-
            readable console format.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Stdlib root handler (only add once)
    root = logging.getLogger("llm_std_lib")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        root.addHandler(handler)
    root.setLevel(log_level)

    renderer: Any
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "llm_std_lib") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for *name*.

    Args:
        name: Logger name (used as the ``logger`` field in log output).

    Returns:
        A bound structlog logger instance.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
