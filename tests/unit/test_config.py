"""Unit tests for llm_std_lib.config."""

from __future__ import annotations

import os
import textwrap

import pytest

from llm_std_lib.config import LLMConfig, ProviderConfig, VaultConfig, _parse_toml, _parse_yaml
from llm_std_lib.exceptions import LLMConfigError


class TestProviderConfig:
    def test_defaults(self) -> None:
        cfg = ProviderConfig()
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.timeout_ms == 30_000
        assert cfg.max_retries == 3
        assert cfg.extra == {}

    def test_with_key(self) -> None:
        cfg = ProviderConfig(api_key="sk-test", timeout_ms=10_000)
        assert cfg.api_key == "sk-test"
        assert cfg.timeout_ms == 10_000


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.default_model == "openai/gpt-4o-mini"
        assert cfg.log_level == "INFO"
        assert cfg.providers == {}
        assert cfg.middleware == []

    def test_default_provider_with_slash(self) -> None:
        cfg = LLMConfig(default_model="anthropic/claude-haiku-3")
        assert cfg.default_provider == "anthropic"
        assert cfg.default_model_name == "claude-haiku-3"

    def test_default_provider_without_slash(self) -> None:
        cfg = LLMConfig(default_model="gpt-4o-mini")
        assert cfg.default_provider == "openai"
        assert cfg.default_model_name == "gpt-4o-mini"

    def test_empty_default_model_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="default_model"):
            LLMConfig(default_model="")

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="log_level"):
            LLMConfig(log_level="VERBOSE")

    def test_get_provider_config_existing(self) -> None:
        pcfg = ProviderConfig(api_key="sk-test")
        cfg = LLMConfig(providers={"openai": pcfg})
        assert cfg.get_provider_config("openai").api_key == "sk-test"

    def test_get_provider_config_missing_returns_default(self) -> None:
        cfg = LLMConfig()
        default = cfg.get_provider_config("openai")
        assert isinstance(default, ProviderConfig)
        assert default.api_key is None

    def test_arbitrary_objects_in_cache_field(self) -> None:
        """cache/router/fallback accept any object (injected at runtime)."""
        cfg = LLMConfig(cache=object(), router=object(), fallback=object())
        assert cfg.cache is not None

    def test_unknown_provider_warns(self) -> None:
        with pytest.warns(UserWarning, match="Unknown provider"):
            LLMConfig(providers={"mycloud": ProviderConfig(api_key="x")})


class TestLLMConfigFromEnv:
    def test_reads_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig.from_env()
        assert "openai" in cfg.providers
        assert cfg.providers["openai"].api_key == "sk-openai-test"

    def test_reads_anthropic_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig.from_env()
        assert "anthropic" in cfg.providers

    def test_reads_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_STD_DEFAULT_MODEL", "anthropic/claude-haiku-3")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = LLMConfig.from_env()
        assert cfg.default_model == "anthropic/claude-haiku-3"

    def test_reads_metrics_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_STD_METRICS_PORT", "9090")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = LLMConfig.from_env()
        assert cfg.metrics.get("port") == 9090

    def test_no_keys_returns_empty_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.providers == {}

    def test_reads_azure_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        cfg = LLMConfig.from_env()
        assert "azure" in cfg.providers
        assert cfg.providers["azure"].extra["deployment"] == "gpt-4o-mini"

    def test_reads_ollama_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cfg = LLMConfig.from_env()
        assert "ollama" in cfg.providers

    def test_reads_bedrock_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_BEDROCK_ENABLED", "true")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        cfg = LLMConfig.from_env()
        assert "bedrock" in cfg.providers
        assert cfg.providers["bedrock"].extra["region"] == "eu-west-1"


# ---------------------------------------------------------------------------
# LLMConfig.from_file (YAML / TOML)
# ---------------------------------------------------------------------------


class TestLLMConfigFromFile:
    def test_from_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        yaml_content = textwrap.dedent("""\
            default_model: anthropic/claude-haiku-3
            log_level: DEBUG
            providers:
              anthropic:
                api_key: sk-ant-test
                timeout_ms: 5000
        """)
        f = tmp_path / "config.yaml"  # type: ignore[operator]
        f.write_text(yaml_content)
        cfg = LLMConfig.from_file(str(f))
        assert cfg.default_model == "anthropic/claude-haiku-3"
        assert cfg.log_level == "DEBUG"
        assert cfg.providers["anthropic"].api_key == "sk-ant-test"
        assert cfg.providers["anthropic"].timeout_ms == 5000

    def test_from_yml_extension(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "cfg.yml"  # type: ignore[operator]
        f.write_text("default_model: openai/gpt-4o-mini\n")
        cfg = LLMConfig.from_file(str(f))
        assert cfg.default_model == "openai/gpt-4o-mini"

    def test_from_toml(self, tmp_path: pytest.TempPathFactory) -> None:
        toml_content = textwrap.dedent("""\
            default_model = "openai/gpt-4o"
            log_level = "WARNING"

            [providers.openai]
            api_key = "sk-toml-test"
        """)
        f = tmp_path / "config.toml"  # type: ignore[operator]
        f.write_text(toml_content)
        cfg = LLMConfig.from_file(str(f))
        assert cfg.default_model == "openai/gpt-4o"
        assert cfg.providers["openai"].api_key == "sk-toml-test"

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="not found"):
            LLMConfig.from_file("/nonexistent/path/config.yaml")

    def test_unsupported_extension_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "config.json"  # type: ignore[operator]
        f.write_text("{}")
        with pytest.raises(LLMConfigError, match="Unsupported"):
            LLMConfig.from_file(str(f))

    def test_invalid_yaml_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "bad.yaml"  # type: ignore[operator]
        f.write_text("default_model: [\nbad yaml")
        with pytest.raises(LLMConfigError):
            LLMConfig.from_file(str(f))

    def test_yaml_missing_pyyaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"yaml": None}):
            with pytest.raises((ImportError, LLMConfigError)):
                _parse_yaml("default_model: openai/gpt-4o-mini")


# ---------------------------------------------------------------------------
# VaultConfig
# ---------------------------------------------------------------------------


class TestVaultConfig:
    def test_raises_import_error_without_hvac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"hvac": None}):
            vc = VaultConfig(url="http://vault:8200", secret_path="secret/data/keys")
            with pytest.raises((ImportError, TypeError)):
                vc._fetch()

    def test_load_merges_secrets(self) -> None:
        import unittest.mock as mock

        vc = VaultConfig(url="http://vault:8200", secret_path="secret/data/keys")
        fake_secrets = {
            "OPENAI_API_KEY": "sk-vault-openai",
            "ANTHROPIC_API_KEY": "sk-vault-ant",
        }
        with mock.patch.object(vc, "_fetch", return_value=fake_secrets):
            cfg = vc.load()

        assert cfg.providers["openai"].api_key == "sk-vault-openai"
        assert cfg.providers["anthropic"].api_key == "sk-vault-ant"

    def test_load_preserves_base_config(self) -> None:
        import unittest.mock as mock

        base = LLMConfig(
            default_model="anthropic/claude-haiku-3",
            providers={"openai": ProviderConfig(api_key="old-key", timeout_ms=99_000)},
        )
        vc = VaultConfig(url="http://vault:8200", secret_path="secret/data/keys")
        with mock.patch.object(vc, "_fetch", return_value={"OPENAI_API_KEY": "new-key"}):
            cfg = vc.load(base_config=base)

        assert cfg.providers["openai"].api_key == "new-key"
        assert cfg.default_model == "anthropic/claude-haiku-3"

    def test_load_skips_empty_secrets(self) -> None:
        import unittest.mock as mock

        vc = VaultConfig(url="http://vault:8200", secret_path="secret/data/keys")
        with mock.patch.object(vc, "_fetch", return_value={"OPENAI_API_KEY": ""}):
            cfg = vc.load()

        assert "openai" not in cfg.providers
