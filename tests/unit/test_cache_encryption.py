"""Unit tests for AES-256-GCM CacheEncryption."""

from __future__ import annotations

import pytest

from llm_std_lib.cache.encryption import CacheEncryption


class TestCacheEncryptionImportError:
    def test_raises_when_cryptography_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import unittest.mock as mock

        with mock.patch.dict(sys.modules, {"cryptography": None,
                                            "cryptography.hazmat": None,
                                            "cryptography.hazmat.primitives": None,
                                            "cryptography.hazmat.primitives.ciphers": None,
                                            "cryptography.hazmat.primitives.ciphers.aead": None}):
            from llm_std_lib.cache.encryption import _import_crypto
            with pytest.raises((ImportError, TypeError)):
                _import_crypto()


class TestCacheEncryptionInit:
    def test_valid_32_byte_key(self) -> None:
        enc = CacheEncryption(b"a" * 32)
        assert enc is not None

    def test_wrong_key_length_raises(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            CacheEncryption(b"short")

    def test_wrong_key_length_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            CacheEncryption(b"x" * 64)


class TestCacheEncryptionFactory:
    def test_generate_returns_instance(self) -> None:
        enc = CacheEncryption.generate()
        assert isinstance(enc, CacheEncryption)

    def test_two_generated_keys_are_different(self) -> None:
        a = CacheEncryption.generate()
        b = CacheEncryption.generate()
        assert a.export_key() != b.export_key()

    def test_from_key_str_pads_to_32(self) -> None:
        enc = CacheEncryption.from_key("short")
        assert enc is not None

    def test_from_key_str_truncates_to_32(self) -> None:
        enc = CacheEncryption.from_key("x" * 100)
        assert enc is not None

    def test_from_key_bytes(self) -> None:
        enc = CacheEncryption.from_key(b"a" * 32)
        assert enc is not None

    def test_from_base64_roundtrip(self) -> None:
        orig = CacheEncryption.generate()
        exported = orig.export_key()
        restored = CacheEncryption.from_base64(exported)
        assert restored.export_key() == exported


class TestCacheEncryptionRoundtrip:
    def test_encrypt_decrypt_roundtrip(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

        enc = CacheEncryption.generate()
        plaintext = "Hello, this is a sensitive LLM response!"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_different_ciphertext_each_call(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

        enc = CacheEncryption.generate()
        ct1 = enc.encrypt("same text")
        ct2 = enc.encrypt("same text")
        assert ct1 != ct2  # different nonces

    def test_wrong_key_fails_decryption(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

        enc1 = CacheEncryption.generate()
        enc2 = CacheEncryption.generate()
        ciphertext = enc1.encrypt("secret")
        with pytest.raises(ValueError, match="Decryption failed"):
            enc2.decrypt(ciphertext)

    def test_empty_string_roundtrip(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

        enc = CacheEncryption.generate()
        assert enc.decrypt(enc.encrypt("")) == ""

    def test_unicode_roundtrip(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

        enc = CacheEncryption.generate()
        text = "Привет мир! 🌍 日本語テスト"
        assert enc.decrypt(enc.encrypt(text)) == text

    def test_export_import_key_preserves_decryption(self) -> None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")

        enc = CacheEncryption.generate()
        ciphertext = enc.encrypt("keep this secret")
        restored = CacheEncryption.from_base64(enc.export_key())
        assert restored.decrypt(ciphertext) == "keep this secret"
