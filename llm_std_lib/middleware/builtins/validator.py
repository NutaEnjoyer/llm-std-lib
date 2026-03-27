"""
Response validator middleware.

Validates ``response.text`` against a Pydantic model or a plain callable,
raising :class:`~llm_std_lib.exceptions.LLMValidationError` on failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from llm_std_lib.exceptions import LLMValidationError
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext


class ResponseValidatorMiddleware(BaseMiddleware):
    """Validates the provider response against a Pydantic schema.

    The response text is expected to be valid JSON that can be parsed into
    the given Pydantic model.  If validation fails,
    :class:`~llm_std_lib.exceptions.LLMValidationError` is raised.

    Args:
        schema: A Pydantic :class:`~pydantic.BaseModel` subclass.
        custom_validator: Optional callable ``(text: str) -> None`` that is
            called *after* the Pydantic check.  Raise any exception to signal
            validation failure; it will be wrapped in ``LLMValidationError``.

    Example::

        class Answer(BaseModel):
            answer: str
            confidence: float

        stack = MiddlewareStack([ResponseValidatorMiddleware(schema=Answer)])
        # response.text must be JSON-parseable into Answer
    """

    def __init__(
        self,
        schema: type[BaseModel],
        custom_validator: Callable[[str], None] | None = None,
    ) -> None:
        self._schema = schema
        self._custom_validator = custom_validator

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        try:
            self._schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMValidationError(
                f"Response failed schema validation ({self._schema.__name__}): {exc}"
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMValidationError(
                f"Response is not valid JSON: {exc}"
            ) from exc

        if self._custom_validator is not None:
            try:
                self._custom_validator(response.text)
            except Exception as exc:
                raise LLMValidationError(
                    f"Custom validator failed: {exc}"
                ) from exc

        return response
