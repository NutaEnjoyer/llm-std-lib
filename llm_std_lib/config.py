"""
Configuration models for llm_std_lib.

LLMConfig is the top-level configuration object. It can be built programmatically,
loaded from environment variables via LLMConfig.from_env(), or from a YAML file.
Validation is performed at construction time (fail-fast).
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llm_std_lib.exceptions import LLMConfigError

# ---------------------------------------------------------------------------
# Supported providers
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = frozenset({
    "openai",
    "anthropic",
    "google",
    "azure",
    "bedrock",
    "ollama",
    "groq",
    "lm_studio",
})

# Token price in USD per 1k tokens: {provider: {model: {input, output}}}
PROVIDER_PRICES: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    },
    "anthropic": {
        "claude-opus-3": {"input": 0.015, "output": 0.075},
        "claude-sonnet-3-5": {"input": 0.003, "output": 0.015},
        "claude-haiku-3": {"input": 0.00025, "output": 0.00125},
    },
    "google": {
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    },
    "groq": {
        "llama-3-8b-8192": {"input": 0.00005, "output": 0.00008},
        "mixtral-8x7b": {"input": 0.00027, "output": 0.00027},
    },
}


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider.

    Args:
        api_key: API key for the provider.
        base_url: Optional custom base URL (e.g. for Azure or Ollama).
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum number of retries before raising.
        extra: Provider-specific extra parameters (e.g. Azure deployment name).
    """

    api_key: str | None = None
    base_url: str | None = None
    timeout_ms: int = 30_000
    max_retries: int = 3
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Top-level configuration for LLMClient.

    Args:
        default_model: Model to use when no routing is configured.
            Format: ``"provider/model-name"`` (e.g. ``"openai/gpt-4o-mini"``).
        providers: Mapping of provider name → ProviderConfig.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        cache: SemanticCache instance (optional, injected at runtime).
        router: ModelRouter instance (optional, injected at runtime).
        fallback: FallbackChain instance (optional, injected at runtime).
        middleware: Ordered list of middleware instances.
        metrics: Metrics exporter config (e.g. ``{"exporter": "prometheus", "port": 9090}``).

    Example::

        config = LLMConfig(
            default_model="anthropic/claude-haiku-3",
            providers={"anthropic": ProviderConfig(api_key="sk-ant-...")},
        )
    """

    default_model: str = "openai/gpt-4o-mini"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    log_level: str = "INFO"
    # Heavy objects are typed as Any to avoid circular imports at module level.
    cache: Any = None
    router: Any = None
    fallback: Any = None
    middleware: list[Any] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate(self) -> LLMConfig:
        self._validate_default_model()
        self._validate_log_level()
        self._warn_unknown_providers()
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_default_model(self) -> None:
        if not self.default_model or not self.default_model.strip():
            raise LLMConfigError("default_model must be a non-empty string.")

    def _validate_log_level(self) -> None:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid:
            raise LLMConfigError(
                f"Invalid log_level '{self.log_level}'. Must be one of {sorted(valid)}."
            )

    def _warn_unknown_providers(self) -> None:
        import warnings

        for name in self.providers:
            if name not in SUPPORTED_PROVIDERS:
                warnings.warn(
                    f"Unknown provider '{name}'. Supported: {sorted(SUPPORTED_PROVIDERS)}.",
                    UserWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def default_provider(self) -> str:
        """Extract provider name from default_model (``'openai/gpt-4o'`` → ``'openai'``)."""
        if "/" in self.default_model:
            return self.default_model.split("/", 1)[0]
        return "openai"

    @property
    def default_model_name(self) -> str:
        """Extract bare model name from default_model (``'openai/gpt-4o'`` → ``'gpt-4o'``)."""
        if "/" in self.default_model:
            return self.default_model.split("/", 1)[1]
        return self.default_model

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Return ProviderConfig for *provider*, creating a default one if absent."""
        return self.providers.get(provider, ProviderConfig())

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build LLMConfig from environment variables.

        Reads the following variables (all optional):

        - ``OPENAI_API_KEY``
        - ``ANTHROPIC_API_KEY``
        - ``GOOGLE_API_KEY``
        - ``GROQ_API_KEY``
        - ``LLM_STD_DEFAULT_MODEL`` (default: ``openai/gpt-4o-mini``)
        - ``LLM_STD_LOG_LEVEL`` (default: ``INFO``)
        - ``LLM_STD_METRICS_PORT``

        Raises:
            LLMConfigError: If the assembled configuration is invalid.
        """
        providers: dict[str, ProviderConfig] = {}

        for env_var, provider_name in [
            ("OPENAI_API_KEY", "openai"),
            ("ANTHROPIC_API_KEY", "anthropic"),
            ("GOOGLE_API_KEY", "google"),
            ("GROQ_API_KEY", "groq"),
        ]:
            key = os.environ.get(env_var)
            if key:
                providers[provider_name] = ProviderConfig(api_key=key)

        # Azure: requires endpoint + deployment via env vars
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if azure_endpoint and azure_deployment:
            providers["azure"] = ProviderConfig(
                api_key=azure_key,
                base_url=azure_endpoint,
                extra={"deployment": azure_deployment},
            )

        # Ollama: auto-detected if OLLAMA_BASE_URL is set (or running on default port)
        ollama_url = os.environ.get("OLLAMA_BASE_URL")
        if ollama_url:
            providers["ollama"] = ProviderConfig(base_url=ollama_url)

        # LM Studio: auto-detected if LM_STUDIO_BASE_URL is set
        lm_studio_url = os.environ.get("LM_STUDIO_BASE_URL")
        if lm_studio_url:
            providers["lm_studio"] = ProviderConfig(base_url=lm_studio_url)

        # AWS Bedrock: no key needed (uses AWS credential chain), opt-in via flag
        if os.environ.get("AWS_BEDROCK_ENABLED", "").lower() in ("1", "true", "yes"):
            providers["bedrock"] = ProviderConfig(
                extra={
                    "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                }
            )

        default_model = os.environ.get("LLM_STD_DEFAULT_MODEL", "openai/gpt-4o-mini")
        log_level = os.environ.get("LLM_STD_LOG_LEVEL", "INFO")

        metrics: dict[str, Any] = {}
        if port := os.environ.get("LLM_STD_METRICS_PORT"):
            metrics["port"] = int(port)

        return cls(
            default_model=default_model,
            providers=providers,
            log_level=log_level,
            metrics=metrics,
        )

    @classmethod
    def from_file(cls, path: str) -> LLMConfig:
        """Load LLMConfig from a YAML or TOML file.

        The file format is detected from the extension (``.yaml``/``.yml``
        or ``.toml``).  Environment variables present in the loaded config
        are **not** overridden — use :meth:`from_env` if you need that.

        Args:
            path: Path to the config file (absolute or relative to cwd).

        Returns:
            Fully validated :class:`LLMConfig` instance.

        Raises:
            LLMConfigError: If the file cannot be read, parsed, or is invalid.
            ImportError: If the required parser (``pyyaml`` / ``tomllib``) is
                missing.

        Example YAML::

            default_model: openai/gpt-4o-mini
            log_level: INFO
            providers:
              openai:
                api_key: sk-...
              anthropic:
                api_key: sk-ant-...
        """
        import pathlib

        p = pathlib.Path(path)
        if not p.exists():
            raise LLMConfigError(f"Config file not found: {path}")

        suffix = p.suffix.lower()
        try:
            raw_text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMConfigError(f"Cannot read config file '{path}': {exc}") from exc

        try:
            if suffix in (".yaml", ".yml"):
                data = _parse_yaml(raw_text)
            elif suffix == ".toml":
                data = _parse_toml(raw_text)
            else:
                raise LLMConfigError(
                    f"Unsupported config file format '{suffix}'. "
                    "Use .yaml, .yml, or .toml."
                )
        except LLMConfigError:
            raise
        except Exception as exc:
            raise LLMConfigError(f"Failed to parse config file '{path}': {exc}") from exc

        # Normalise providers: dict[str, dict] → dict[str, ProviderConfig]
        raw_providers = data.pop("providers", {}) or {}
        providers: dict[str, ProviderConfig] = {
            name: ProviderConfig(**cfg) if isinstance(cfg, dict) else cfg
            for name, cfg in raw_providers.items()
        }

        try:
            return cls(providers=providers, **data)
        except Exception as exc:
            raise LLMConfigError(f"Invalid config in '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# HashiCorp Vault integration
# ---------------------------------------------------------------------------


class VaultConfig:
    """Reads API keys from a HashiCorp Vault KV secret and returns a
    :class:`LLMConfig`.

    Requires ``hvac`` (``pip install llm-std-lib[vault]``).

    Args:
        url: Vault server URL (e.g. ``"http://vault:8200"``).
        token: Vault token. Falls back to ``VAULT_TOKEN`` env var.
        secret_path: KV v2 secret path (e.g. ``"secret/data/llm-keys"``).
        mount_point: KV mount point (default ``"secret"``).

    Example::

        cfg = VaultConfig(
            url="http://vault:8200",
            token="s.xxx",
            secret_path="secret/data/llm-keys",
        ).load()
        client = LLMClient(cfg)
    """

    def __init__(
        self,
        url: str,
        secret_path: str,
        token: str | None = None,
        mount_point: str = "secret",
    ) -> None:
        self._url = url
        self._secret_path = secret_path
        self._token = token or os.environ.get("VAULT_TOKEN")
        self._mount_point = mount_point

    def load(self, base_config: LLMConfig | None = None) -> LLMConfig:
        """Fetch secrets from Vault and return an :class:`LLMConfig`.

        Merges Vault secrets on top of *base_config* (or a blank config).

        Args:
            base_config: Optional base config to merge into.

        Returns:
            :class:`LLMConfig` with provider API keys populated from Vault.

        Raises:
            ImportError: If ``hvac`` is not installed.
            LLMConfigError: If Vault is unreachable or the secret is missing.
        """
        secrets = self._fetch()
        cfg = base_config or LLMConfig()

        provider_keys: dict[str, str] = {
            "openai": secrets.get("OPENAI_API_KEY", ""),
            "anthropic": secrets.get("ANTHROPIC_API_KEY", ""),
            "google": secrets.get("GOOGLE_API_KEY", ""),
            "groq": secrets.get("GROQ_API_KEY", ""),
        }

        providers = dict(cfg.providers)
        for name, key in provider_keys.items():
            if key:
                existing = providers.get(name, ProviderConfig())
                providers[name] = existing.model_copy(update={"api_key": key})

        return cfg.model_copy(update={"providers": providers})

    def _fetch(self) -> dict[str, Any]:
        try:
            import hvac
        except ImportError as exc:
            raise ImportError(
                "hvac is required for VaultConfig. "
                "Install it with: pip install llm-std-lib[vault]"
            ) from exc

        try:
            client = hvac.Client(url=self._url, token=self._token)
            # Strip "secret/data/" prefix if present — hvac adds it automatically
            path = self._secret_path
            if path.startswith(f"{self._mount_point}/data/"):
                path = path[len(f"{self._mount_point}/data/"):]
            elif path.startswith(f"{self._mount_point}/"):
                path = path[len(f"{self._mount_point}/"):]

            response = client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self._mount_point
            )
            return response["data"]["data"]  # type: ignore[no-any-return]
        except Exception as exc:
            raise LLMConfigError(f"Vault error reading '{self._secret_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# YAML / TOML parsing helpers
# ---------------------------------------------------------------------------


def _parse_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc
    result = yaml.safe_load(text)
    return result if isinstance(result, dict) else {}


def _parse_toml(text: str) -> dict[str, Any]:
    # tomllib is built-in since Python 3.11; fall back to tomli for 3.10
    try:
        import tomllib
        result: dict[str, Any] = tomllib.loads(text)
        return result
    except ImportError:
        pass
    try:
        import tomli
        result2: dict[str, Any] = tomli.loads(text)
        return result2
    except ImportError as exc:
        raise ImportError(
            "tomli is required to load TOML config files on Python <3.11. "
            "Install it with: pip install tomli"
        ) from exc
