"""Unit tests for the router layer (ComplexityScorer, strategies, ModelRouter)."""

from __future__ import annotations

import pytest

from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.complexity import ComplexityBreakdown, ComplexityScorer
from llm_std_lib.router.model_router import CostCalculator, ModelRouter
from llm_std_lib.router.strategies.base import RouteResult
from llm_std_lib.router.strategies.complexity_based import ComplexityBasedStrategy, Tier
from llm_std_lib.router.strategies.cost_optimized import CostOptimizedStrategy
from llm_std_lib.router.strategies.latency_optimized import LatencyOptimizedStrategy
from llm_std_lib.router.strategies.round_robin import RoundRobinStrategy
from llm_std_lib.types import RequestContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(prompt: str) -> RequestContext:
    return RequestContext(prompt=prompt)


# ---------------------------------------------------------------------------
# ComplexityScorer — must classify ≥85% of the test dataset correctly
# ---------------------------------------------------------------------------

class TestComplexityScorer:
    """Test dataset: (prompt, expected_band) where band is 'low'|'mid'|'high'."""

    DATASET: list[tuple[str, str]] = [
        # Low complexity (score < 0.35)
        ("What is 2 + 2?", "low"),
        ("Hello, how are you?", "low"),
        ("Tell me a joke.", "low"),
        ("What is the capital of France?", "low"),
        ("Translate 'hello' to Spanish.", "low"),
        ("Give me a recipe for pasta.", "low"),
        ("What day is it today?", "low"),
        # Mid complexity (0.35 ≤ score < 0.65)
        (
            "Explain step by step how to set up a Python virtual environment.",
            "mid",
        ),
        (
            "Compare and contrast REST vs GraphQL APIs.",
            "mid",
        ),
        (
            "Write a Python function that calculates the Fibonacci sequence.",
            "mid",
        ),
        (
            "Analyse the pros and cons of microservices architecture.",
            "mid",
        ),
        # High complexity (score ≥ 0.65)
        (
            "```python\ndef matrix_multiply(A, B):\n    pass\n```\n"
            "Implement the above function using NumPy and explain the time complexity "
            "step by step, showing the derivation $O(n^3)$ vs $O(n^{2.37})$ for Strassen.",
            "high",
        ),
        (
            "In the context of securitization and portfolio liquidity, analyse the "
            "fiduciary obligations under IFRS 9 and explain the arbitrage implications "
            "for a collateral portfolio with high volatility. Step by step reasoning required.",
            "high",
        ),
        (
            "Provide a differential diagnosis for a patient presenting with metastasis "
            "in the lymph nodes, elevated CA-125, and contraindications to chemotherapy. "
            "Include etiology, prognosis, and recommended biopsy protocols.",
            "high",
        ),
    ]

    def _band(self, score: float) -> str:
        if score < 0.35:
            return "low"
        if score < 0.65:
            return "mid"
        return "high"

    def test_classification_accuracy(self) -> None:
        scorer = ComplexityScorer()
        correct = 0
        for prompt, expected_band in self.DATASET:
            score = scorer.score(prompt)
            got = self._band(score)
            if got == expected_band:
                correct += 1
        accuracy = correct / len(self.DATASET)
        assert accuracy >= 0.85, (
            f"ComplexityScorer accuracy {accuracy:.0%} < 85% on test dataset."
        )

    def test_score_range(self) -> None:
        scorer = ComplexityScorer()
        for prompt, _ in self.DATASET:
            s = scorer.score(prompt)
            assert 0.0 <= s <= 1.0, f"Score {s} out of range for: {prompt[:40]}"

    def test_score_with_breakdown_returns_breakdown(self) -> None:
        scorer = ComplexityScorer()
        bd = scorer.score_with_breakdown("Step by step, derive the integral $\\int x^2 dx$.")
        assert isinstance(bd, ComplexityBreakdown)
        assert bd.math_score > 0
        assert bd.cot_score > 0
        assert 0.0 <= bd.total <= 1.0

    def test_simple_prompt_lower_than_complex(self) -> None:
        scorer = ComplexityScorer()
        simple = scorer.score("Hi")
        complex_ = scorer.score(
            "```python\nclass NeuralNet:\n    pass\n```\n"
            "Analyse this step by step, considering $O(n^2)$ complexity "
            "and fiduciary implications of the derivative portfolio."
        )
        assert simple < complex_

    def test_code_signal(self) -> None:
        scorer = ComplexityScorer()
        bd_no_code = scorer.score_with_breakdown("Write a sorting algorithm.")
        bd_with_code = scorer.score_with_breakdown(
            "```python\ndef sort(arr): pass\n```\nExplain this code."
        )
        assert bd_with_code.code_score > bd_no_code.code_score

    def test_math_signal(self) -> None:
        scorer = ComplexityScorer()
        bd = scorer.score_with_breakdown("Compute $\\int_0^1 x^2 dx$.")
        assert bd.math_score > 0
        assert bd.signals.get("has_math") is True

    def test_domain_signal(self) -> None:
        scorer = ComplexityScorer()
        bd = scorer.score_with_breakdown("The patient requires a biopsy for diagnosis.")
        assert bd.domain_score > 0
        assert bd.signals.get("has_domain_terms") is True

    def test_multilang_signal(self) -> None:
        scorer = ComplexityScorer()
        bd = scorer.score_with_breakdown("Переведи этот текст на английский язык.")
        assert bd.multilang_score > 0
        assert bd.signals.get("has_multilang") is True

    def test_empty_prompt(self) -> None:
        scorer = ComplexityScorer()
        s = scorer.score("")
        assert s == 0.0

    def test_very_long_prompt_saturates_length(self) -> None:
        scorer = ComplexityScorer()
        long_prompt = "word " * 3000
        bd = scorer.score_with_breakdown(long_prompt)
        assert bd.length_score == 1.0


# ---------------------------------------------------------------------------
# ComplexityBasedStrategy
# ---------------------------------------------------------------------------

class TestComplexityBasedStrategy:
    def _tiers(self) -> list[Tier]:
        return [
            Tier(models=["openai/gpt-4o-mini"], max_complexity=0.3),
            Tier(models=["openai/gpt-4o"],      max_complexity=0.7),
            Tier(models=["openai/gpt-4-turbo"], max_complexity=1.0),
        ]

    def test_simple_routes_to_tier1(self) -> None:
        strategy = ComplexityBasedStrategy(tiers=self._tiers())
        result = strategy.route(_ctx("Hi, how are you?"))
        assert result.model == "gpt-4o-mini"
        assert result.provider == "openai"

    def test_complex_routes_to_tier3(self) -> None:
        strategy = ComplexityBasedStrategy(tiers=self._tiers())
        complex_prompt = (
            "```python\nclass Transformer:\n    pass\n```\n"
            "Analyse step by step the $O(n^2)$ attention complexity "
            "and securitization implications for a fiduciary portfolio."
        )
        result = strategy.route(_ctx(complex_prompt))
        assert result.model == "gpt-4-turbo"

    def test_empty_tiers_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            ComplexityBasedStrategy(tiers=[])

    def test_tier_empty_models_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            Tier(models=[], max_complexity=0.5)

    def test_tier_invalid_max_complexity_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            Tier(models=["openai/gpt-4o"], max_complexity=1.5)

    def test_result_has_reason(self) -> None:
        strategy = ComplexityBasedStrategy(tiers=self._tiers())
        result = strategy.route(_ctx("Hello"))
        assert "complexity=" in result.reason

    def test_tiers_sorted_by_complexity(self) -> None:
        # Provide tiers in reverse order — strategy must sort them
        tiers = [
            Tier(models=["openai/gpt-4-turbo"], max_complexity=1.0),
            Tier(models=["openai/gpt-4o-mini"], max_complexity=0.3),
        ]
        strategy = ComplexityBasedStrategy(tiers=tiers)
        result = strategy.route(_ctx("Hi"))
        assert result.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# CostOptimizedStrategy
# ---------------------------------------------------------------------------

class TestCostOptimizedStrategy:
    def test_cheapest_model_selected(self) -> None:
        strategy = CostOptimizedStrategy(models=[
            "openai/gpt-4-turbo",   # expensive
            "openai/gpt-4o-mini",   # cheap
            "anthropic/claude-opus-3",  # very expensive
        ])
        result = strategy.route(_ctx("Hello"))
        assert result.model == "gpt-4o-mini"

    def test_empty_models_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            CostOptimizedStrategy(models=[])

    def test_quality_threshold_excludes_cheapest(self) -> None:
        strategy = CostOptimizedStrategy(
            models=["openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4-turbo"],
            quality_threshold=0.5,
        )
        result = strategy.route(_ctx("Hello"))
        # With threshold=0.5, skip bottom 50% (cheapest 1 model), next is gpt-4o
        assert result.model in ("gpt-4o", "gpt-4-turbo")

    def test_invalid_quality_threshold_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            CostOptimizedStrategy(models=["openai/gpt-4o"], quality_threshold=1.5)

    def test_unknown_price_deprioritised(self) -> None:
        strategy = CostOptimizedStrategy(models=[
            "unknown-provider/mystery-model",
            "openai/gpt-4o-mini",
        ])
        result = strategy.route(_ctx("Hello"))
        assert result.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# LatencyOptimizedStrategy
# ---------------------------------------------------------------------------

class TestLatencyOptimizedStrategy:
    def test_no_history_returns_first_model(self) -> None:
        strategy = LatencyOptimizedStrategy(models=[
            "openai/gpt-4o-mini", "anthropic/claude-haiku-3"
        ])
        # No observations yet — all p95 = inf, first (min) is returned
        result = strategy.route(_ctx("Hello"))
        assert result.full_model in ("openai/gpt-4o-mini", "anthropic/claude-haiku-3")

    def test_lower_latency_selected(self) -> None:
        strategy = LatencyOptimizedStrategy(models=[
            "openai/gpt-4o-mini", "anthropic/claude-haiku-3"
        ])
        for _ in range(10):
            strategy.record("openai/gpt-4o-mini", 300.0)
        for _ in range(10):
            strategy.record("anthropic/claude-haiku-3", 100.0)

        result = strategy.route(_ctx("Hello"))
        assert result.full_model == "anthropic/claude-haiku-3"

    def test_record_ignored_for_unknown_model(self) -> None:
        strategy = LatencyOptimizedStrategy(models=["openai/gpt-4o-mini"])
        strategy.record("unknown/model", 100.0)  # should not crash
        result = strategy.route(_ctx("Hello"))
        assert result.model == "gpt-4o-mini"

    def test_get_stats(self) -> None:
        strategy = LatencyOptimizedStrategy(models=["openai/gpt-4o-mini"])
        for ms in [100, 200, 300, 400, 500]:
            strategy.record("openai/gpt-4o-mini", float(ms))
        stats = strategy.get_stats()
        assert "openai/gpt-4o-mini" in stats
        assert stats["openai/gpt-4o-mini"]["count"] == 5
        assert stats["openai/gpt-4o-mini"]["p95"] >= stats["openai/gpt-4o-mini"]["p50"]

    def test_empty_models_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            LatencyOptimizedStrategy(models=[])


# ---------------------------------------------------------------------------
# RoundRobinStrategy
# ---------------------------------------------------------------------------

class TestRoundRobinStrategy:
    def test_cycles_through_models(self) -> None:
        models = ["openai/gpt-4o-mini", "anthropic/claude-haiku-3", "groq/llama-3-8b-8192"]
        strategy = RoundRobinStrategy(models=models)
        results = [strategy.route(_ctx("hi")).full_model for _ in range(6)]
        assert results == models + models  # two full cycles

    def test_single_model(self) -> None:
        strategy = RoundRobinStrategy(models=["openai/gpt-4o"])
        for _ in range(5):
            r = strategy.route(_ctx("hi"))
            assert r.model == "gpt-4o"

    def test_empty_models_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            RoundRobinStrategy(models=[])


# ---------------------------------------------------------------------------
# CostCalculator
# ---------------------------------------------------------------------------

class TestCostCalculator:
    def test_known_model(self) -> None:
        calc = CostCalculator()
        cost = calc.estimate("openai", "gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
        assert cost > 0

    def test_unknown_model_returns_zero(self) -> None:
        calc = CostCalculator()
        cost = calc.estimate("unknown", "model-xyz", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0

    def test_cheapest(self) -> None:
        calc = CostCalculator()
        cheapest = calc.cheapest(["openai/gpt-4-turbo", "openai/gpt-4o-mini", "openai/gpt-4o"])
        assert cheapest == "openai/gpt-4o-mini"

    def test_custom_weights(self) -> None:
        calc_default = CostCalculator()
        calc_output_heavy = CostCalculator(cost_weights={"input": 1.0, "output": 10.0})
        # gpt-4-turbo has high output cost; with high output weight it should be more expensive
        c1 = calc_default.estimate("openai", "gpt-4-turbo", 100, 500)
        c2 = calc_output_heavy.estimate("openai", "gpt-4-turbo", 100, 500)
        assert c2 > c1


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class TestModelRouter:
    def test_route_updates_context(self) -> None:
        router = ModelRouter.round_robin(models=["openai/gpt-4o-mini"])
        ctx = _ctx("Hello")
        result = router.route(ctx)
        assert ctx.provider == "openai"
        assert ctx.model == "gpt-4o-mini"
        assert result.provider == "openai"

    def test_custom_callable_strategy(self) -> None:
        def my_strategy(ctx: RequestContext) -> RouteResult:
            return RouteResult(provider="anthropic", model="claude-haiku-3", reason="always")

        router = ModelRouter(strategy=my_strategy)
        result = router.route(_ctx("Hello"))
        assert result.provider == "anthropic"
        assert result.model == "claude-haiku-3"

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            ModelRouter(strategy=42)  # type: ignore[arg-type]

    def test_factory_complexity_based(self) -> None:
        router = ModelRouter.complexity_based(tiers=[
            {"models": ["openai/gpt-4o-mini"], "max_complexity": 0.5},
            {"models": ["openai/gpt-4o"],      "max_complexity": 1.0},
        ])
        result = router.route(_ctx("Hi"))
        assert result.model == "gpt-4o-mini"

    def test_factory_cost_optimized(self) -> None:
        router = ModelRouter.cost_optimized(
            models=["openai/gpt-4-turbo", "openai/gpt-4o-mini"]
        )
        result = router.route(_ctx("Hello"))
        assert result.model == "gpt-4o-mini"

    def test_factory_latency_optimized(self) -> None:
        router = ModelRouter.latency_optimized(
            models=["openai/gpt-4o-mini", "anthropic/claude-haiku-3"]
        )
        assert router.strategy is not None

    def test_factory_round_robin(self) -> None:
        router = ModelRouter.round_robin(models=["openai/gpt-4o-mini", "anthropic/claude-haiku-3"])
        r1 = router.route(_ctx("a"))
        r2 = router.route(_ctx("b"))
        assert r1.full_model != r2.full_model

    def test_strategy_property(self) -> None:
        from llm_std_lib.router.strategies.round_robin import RoundRobinStrategy
        strategy = RoundRobinStrategy(models=["openai/gpt-4o-mini"])
        router = ModelRouter(strategy=strategy)
        assert router.strategy is strategy


# ---------------------------------------------------------------------------
# RouteResult
# ---------------------------------------------------------------------------

class TestRouteResult:
    def test_full_model(self) -> None:
        r = RouteResult(provider="openai", model="gpt-4o-mini")
        assert r.full_model == "openai/gpt-4o-mini"

    def test_default_reason(self) -> None:
        r = RouteResult(provider="openai", model="gpt-4o")
        assert r.reason == ""
