"""
Presidio-backed PII redaction engine.

Uses Microsoft Presidio for NER-based detection of names, addresses, and
other entities that regex cannot reliably catch.

Requires the ``presidio`` optional dependency::

    pip install llm-std-lib[presidio]
    python -m spacy download en_core_web_lg
    python -m spacy download ru_core_news_sm  # for Russian

Usage::

    from llm_std_lib.middleware.builtins.pii import PIIRedactorMiddleware
    from llm_std_lib.middleware.builtins.presidio_engine import PresidioPIIEngine

    engine = PresidioPIIEngine(languages=["en", "ru"])
    pii = PIIRedactorMiddleware(engine=engine)
"""

from __future__ import annotations

from typing import Any


class PresidioPIIEngine:
    """NER-based PII engine powered by Microsoft Presidio.

    Detects names, addresses, phone numbers, emails, credit cards, IDs,
    and many more entity types using spaCy NER models under the hood.

    Args:
        languages: List of language codes to analyse. Each language requires
            a spaCy model installed (e.g. ``en_core_web_lg``,
            ``ru_core_news_sm``). Default: ``["en"]``.
        entities: Presidio entity types to redact. ``None`` = all detected.
            Examples: ``["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "LOCATION", "CREDIT_CARD"]``.
        default_language: Language used when ``redact()`` is called without
            an explicit *language* argument. Default: ``"en"``.

    Raises:
        ImportError: If ``presidio-analyzer`` or ``presidio-anonymizer``
            are not installed.

    Example::

        engine = PresidioPIIEngine(languages=["en", "ru"])
        clean = engine.redact("Меня зовут Иван, email: ivan@mail.ru", language="ru")
        # → "Меня зовут <PERSON>, email: <EMAIL_ADDRESS>"
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        entities: list[str] | None = None,
        default_language: str = "en",
    ) -> None:
        self._languages = languages or ["en"]
        self._entities = entities
        self._default_language = default_language
        self._analyzer: Any = None
        self._anonymizer: Any = None

    def _load(self) -> None:
        """Lazily load Presidio engines (heavy import, done once)."""
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "presidio is not installed. "
                "Run: pip install llm-std-lib[presidio]"
            ) from exc

        try:
            self._analyzer = AnalyzerEngine()
        except OSError as exc:
            # spaCy model not found
            models = {
                "en": "en_core_web_lg",
                "ru": "ru_core_news_sm",
            }
            hints = " ".join(
                f"python -m spacy download {m}"
                for lang, m in models.items()
                if lang in self._languages
            ) or "python -m spacy download en_core_web_lg"
            raise OSError(
                f"spaCy model not found. Download the required model(s):\n  {hints}"
            ) from exc

        self._anonymizer = AnonymizerEngine()

    def redact(self, text: str, language: str | None = None) -> str:
        """Detect and anonymise PII in *text*.

        Args:
            text: Raw input text.
            language: ISO language code (e.g. ``"en"``, ``"ru"``).
                Falls back to ``default_language`` if not provided.

        Returns:
            Text with PII replaced by ``<ENTITY_TYPE>`` placeholders.
        """
        if not text:
            return text

        if self._analyzer is None:
            self._load()

        lang = language or self._default_language
        results = self._analyzer.analyze(
            text=text,
            language=lang,
            entities=self._entities,
        )
        if not results:
            return text

        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return str(anonymized.text)
