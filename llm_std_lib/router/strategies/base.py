"""
Abstract routing strategy interface.

All strategy implementations must subclass BaseStrategy and implement route().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_std_lib.types import RequestContext


@dataclass
class RouteResult:
    """The routing decision returned by a strategy.

    Attributes:
        provider: Provider name (e.g. ``"openai"``).
        model: Bare model name (e.g. ``"gpt-4o-mini"``).
        reason: Human-readable explanation of the decision.
    """

    provider: str
    model: str
    reason: str = ""

    @property
    def full_model(self) -> str:
        """``provider/model`` string, e.g. ``"openai/gpt-4o-mini"``."""
        return f"{self.provider}/{self.model}"


class BaseStrategy(ABC):
    """Abstract base class for routing strategies.

    A strategy receives a :class:`~llm_std_lib.types.RequestContext` and
    returns a :class:`RouteResult` that tells the router which provider and
    model to use for the request.
    """

    @abstractmethod
    def route(self, ctx: RequestContext) -> RouteResult:
        """Select the best provider/model for *ctx*.

        Args:
            ctx: The fully populated request context.

        Returns:
            :class:`RouteResult` with provider, model and optional reason.
        """
