"""
Cost tracking middleware.

Accumulates LLM call costs from ``response.cost_usd`` and exposes
per-tag and global totals for budgeting and observability.
"""

from __future__ import annotations

from collections import defaultdict

from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext


class CostTrackerMiddleware(BaseMiddleware):
    """Accumulates request costs and exposes per-tag breakdowns.

    Cost is read from :attr:`~llm_std_lib.types.ResponseContext.cost_usd`
    which is already calculated by the provider layer.  Tags are read from
    :attr:`~llm_std_lib.types.RequestContext.tags` (a plain ``dict``).

    Args:
        tag_keys: Tag keys to group costs by (e.g. ``["user_id", "project_id"]``).
            Defaults to ``["user_id"]``.

    Example::

        tracker = CostTrackerMiddleware(tag_keys=["user_id", "project_id"])
        stack = MiddlewareStack([tracker])

        # after some requests:
        print(tracker.total_cost)                        # 0.0042 USD
        print(tracker.cost_by_tag("user_id", "alice"))  # 0.0021 USD
    """

    def __init__(self, tag_keys: list[str] | None = None) -> None:
        self._tag_keys = tag_keys or ["user_id"]
        self._total: float = 0.0
        # tag_key → tag_value → accumulated cost
        self._by_tag: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        cost = response.cost_usd
        self._total += cost

        for key in self._tag_keys:
            value = ctx.tags.get(key) or (ctx.metadata or {}).get(key)
            if value is not None:
                self._by_tag[key][str(value)] += cost

        return response

    # ------------------------------------------------------------------
    # Reporting API
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Total accumulated cost across all requests (USD)."""
        return self._total

    def cost_by_tag(self, tag_key: str, tag_value: str) -> float:
        """Return accumulated cost for a specific tag value.

        Args:
            tag_key: The tag key (e.g. ``"user_id"``).
            tag_value: The tag value (e.g. ``"alice"``).

        Returns:
            Accumulated cost in USD, or ``0.0`` if no matching requests.
        """
        return self._by_tag.get(tag_key, {}).get(tag_value, 0.0)

    def breakdown(self) -> dict[str, dict[str, float]]:
        """Return the full cost breakdown as a nested dict."""
        return {k: dict(v) for k, v in self._by_tag.items()}

    def reset(self) -> None:
        """Reset all accumulated counters to zero."""
        self._total = 0.0
        self._by_tag.clear()
