"""
Prompt complexity estimation.

ComplexityScorer analyses an incoming prompt and returns a float in [0.0, 1.0]
where 0.0 = trivially simple and 1.0 = maximally complex. The score is used by
the complexity_based routing strategy to pick the right model tier.

Six weighted signals are combined:

1. **Token length** — longer prompts are typically harder.
2. **Code / JSON / XML blocks** — technical structured content.
3. **Mathematical expressions** — LaTeX, equations, formulas.
4. **Chain-of-thought indicators** — "step by step", "reason", "explain why".
5. **Multilingual content** — non-ASCII characters hint at translation tasks.
6. **Domain-specific terminology** — medical, legal, financial vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Heuristic signal weights (sum = 1.0)
# ---------------------------------------------------------------------------
_W_LENGTH = 0.03
_W_CODE = 0.15
_W_MATH = 0.15
_W_COT = 0.40
_W_MULTILANG = 0.02
_W_DOMAIN = 0.25

# Token length normalisation cap (prompts ≥ this are scored 1.0 on length)
_MAX_TOKENS_FOR_NORM = 500

# Regex patterns
_CODE_PATTERN = re.compile(
    r"```[\s\S]*?```"           # fenced code blocks
    r"|`[^`\n]{2,}`"           # inline code
    r"|\{[\s\S]{10,}\}"        # JSON / dict-like structures
    r"|<[a-zA-Z][^>]*>[\s\S]*?</[a-zA-Z]+>",  # XML/HTML tags
    re.MULTILINE,
)
_MATH_PATTERN = re.compile(
    r"\$\$[\s\S]+?\$\$"        # display math
    r"|\$[^$\n]+\$"            # inline math
    r"|\\[a-zA-Z]+\{"          # LaTeX commands
    r"|\b\d+\s*[+\-*/^=]\s*\d+"  # arithmetic expressions
    r"|\bintegral\b|\bderivative\b|\bmatrix\b|\beigen\b",
    re.IGNORECASE,
)
_COT_PATTERN = re.compile(
    r"\bstep[\s\-]by[\s\-]step\b"
    r"|\breason(?:ing)?\b"
    r"|\bexplain\s+(?:why|how)\b"
    r"|\banalyse\b|\banalyze\b"
    r"|\bbreak\s+(?:it\s+)?down\b"
    r"|\bchain\s+of\s+thought\b"
    r"|\bfirst[,\s].+then\b"
    r"|\bpros?\s+and\s+cons?\b"
    r"|\bcompare\s+and\s+contrast\b",
    re.IGNORECASE,
)
_MULTILANG_PATTERN = re.compile(r"[^\x00-\x7F]")  # any non-ASCII character

# Specialised domain vocabulary (medical, legal, financial)
_DOMAIN_TERMS: frozenset[str] = frozenset({
    # Medical
    "diagnosis", "prognosis", "pathology", "pharmacology", "contraindication",
    "etiology", "metastasis", "immunotherapy", "auscultation", "biopsy",
    # Legal
    "jurisdiction", "plaintiff", "defendant", "subpoena", "litigation",
    "arbitration", "indemnification", "precedent", "injunction", "affidavit",
    # Financial
    "amortization", "arbitrage", "collateral", "derivative", "fiduciary",
    "liquidity", "portfolio", "volatility", "solvency", "securitization",
})
_DOMAIN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _DOMAIN_TERMS) + r")\b",
    re.IGNORECASE,
)


@dataclass
class ComplexityBreakdown:
    """Detailed per-signal scores from ComplexityScorer.

    All individual scores are in [0.0, 1.0]. ``total`` is the weighted sum.
    """

    length_score: float = 0.0
    code_score: float = 0.0
    math_score: float = 0.0
    cot_score: float = 0.0
    multilang_score: float = 0.0
    domain_score: float = 0.0
    total: float = 0.0
    signals: dict[str, bool] = field(default_factory=dict)


class ComplexityScorer:
    """Heuristic prompt complexity estimator.

    Returns a float in ``[0.0, 1.0]`` representing the estimated difficulty of
    a prompt. Combine with :class:`ModelRouter` to route simple prompts to cheap
    models and complex ones to flagship models.

    Args:
        max_tokens_for_norm: Token count at which the length signal saturates
            at 1.0 (default: 2000).

    Example::

        scorer = ComplexityScorer()
        score = scorer.score("What is 2 + 2?")          # → ~0.05
        score = scorer.score("Step by step, derive ...")  # → ~0.70
    """

    def __init__(self, max_tokens_for_norm: int = _MAX_TOKENS_FOR_NORM) -> None:
        self._max_tokens = max_tokens_for_norm

    def score(self, prompt: str) -> float:
        """Compute a complexity score in ``[0.0, 1.0]`` for *prompt*.

        Args:
            prompt: The raw prompt text to evaluate.

        Returns:
            Complexity score between 0.0 (simple) and 1.0 (very complex).
        """
        return self.score_with_breakdown(prompt).total

    def score_with_breakdown(self, prompt: str) -> ComplexityBreakdown:
        """Compute the complexity score and return per-signal details.

        Args:
            prompt: The raw prompt text to evaluate.

        Returns:
            :class:`ComplexityBreakdown` with individual signal scores and total.
        """
        bd = ComplexityBreakdown()

        # 1. Token-length signal (approximate: split on whitespace)
        approx_tokens = len(prompt.split())
        bd.length_score = min(1.0, approx_tokens / self._max_tokens)

        # 2. Code / JSON / XML
        code_matches = _CODE_PATTERN.findall(prompt)
        bd.code_score = min(1.0, len(code_matches) * 1.0)
        bd.signals["has_code"] = bool(code_matches)

        # 3. Mathematical expressions
        math_matches = _MATH_PATTERN.findall(prompt)
        bd.math_score = min(1.0, len(math_matches) * 0.5)
        bd.signals["has_math"] = bool(math_matches)

        # 4. Chain-of-thought indicators
        cot_matches = _COT_PATTERN.findall(prompt)
        bd.cot_score = min(1.0, len(cot_matches) * 0.9)
        bd.signals["has_cot"] = bool(cot_matches)

        # 5. Multilingual (fraction of non-ASCII chars, capped at 0.3 → score 1.0)
        non_ascii = len(_MULTILANG_PATTERN.findall(prompt))
        ratio = non_ascii / max(1, len(prompt))
        bd.multilang_score = min(1.0, ratio / 0.3)
        bd.signals["has_multilang"] = non_ascii > 3

        # 6. Domain-specific terminology (unique matches)
        domain_hits = set(m.lower() for m in _DOMAIN_PATTERN.findall(prompt))
        bd.domain_score = min(1.0, len(domain_hits) * 0.4)
        bd.signals["has_domain_terms"] = bool(domain_hits)

        bd.total = (
            _W_LENGTH * bd.length_score
            + _W_CODE * bd.code_score
            + _W_MATH * bd.math_score
            + _W_COT * bd.cot_score
            + _W_MULTILANG * bd.multilang_score
            + _W_DOMAIN * bd.domain_score
        )
        # Clamp to [0.0, 1.0] just in case
        bd.total = max(0.0, min(1.0, bd.total))
        return bd
