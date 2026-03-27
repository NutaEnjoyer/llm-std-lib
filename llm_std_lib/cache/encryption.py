"""
AES-256-GCM encryption for cached LLM responses.

Wraps the ``cryptography`` package (``pip install cryptography``) to provide
authenticated encryption of cache values before they are stored in any
backend.

Usage::

    enc = CacheEncryption.from_key("my-32-byte-secret-key-here!!!!!")
    ciphertext = enc.encrypt("sensitive response text")
    plaintext  = enc.decrypt(ciphertext)

    # Or generate a new key
    enc = CacheEncryption.generate()
    print(enc.export_key())   # store this somewhere safe
"""

from __future__ import annotations

import base64
import os


def _import_crypto() -> object:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import (
            AESGCM,
        )
        return AESGCM
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for CacheEncryption. "
            "Install it with: pip install cryptography"
        ) from exc


class CacheEncryption:
    """AES-256-GCM authenticated encryption for cache values.

    Args:
        key: 32-byte AES key (raw bytes).

    Raises:
        ValueError: If *key* is not exactly 32 bytes.
    """

    _NONCE_SIZE = 12  # 96-bit nonce recommended for GCM

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError(
                f"AES-256 key must be exactly 32 bytes, got {len(key)}."
            )
        self._key = key

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls) -> CacheEncryption:
        """Generate a random 256-bit key and return a new :class:`CacheEncryption`.

        Store the key securely (e.g. in HashiCorp Vault or an env var) so
        you can decrypt cached values after a restart.
        """
        return cls(os.urandom(32))

    @classmethod
    def from_key(cls, key: str | bytes) -> CacheEncryption:
        """Create from a raw key string or bytes.

        If *key* is a ``str``, it is UTF-8 encoded and zero-padded / truncated
        to 32 bytes.  For production, pass a proper 32-byte random key.

        Args:
            key: 32-byte key as ``bytes``, or a passphrase ``str`` (will be
                 padded to 32 bytes).
        """
        if isinstance(key, str):
            raw = key.encode("utf-8")
            raw = raw[:32].ljust(32, b"\x00")
        else:
            raw = key
        return cls(raw)

    @classmethod
    def from_base64(cls, b64_key: str) -> CacheEncryption:
        """Create from a base64-encoded key (as returned by :meth:`export_key`).

        Args:
            b64_key: Base64-encoded 32-byte key.
        """
        return cls(base64.b64decode(b64_key))

    # ------------------------------------------------------------------
    # Encrypt / Decrypt
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* and return a base64-encoded ciphertext string.

        The nonce is prepended to the ciphertext so :meth:`decrypt` is
        self-contained.

        Args:
            plaintext: UTF-8 string to encrypt.

        Returns:
            Base64-encoded ``nonce || ciphertext`` string.
        """
        AESGCM = _import_crypto()
        aesgcm = AESGCM(self._key)  # type: ignore[operator]
        nonce = os.urandom(self._NONCE_SIZE)
        ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ct).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext produced by :meth:`encrypt`.

        Args:
            ciphertext: Base64-encoded ``nonce || ciphertext`` string.

        Returns:
            Decrypted UTF-8 plaintext.

        Raises:
            ValueError: If decryption or authentication fails.
        """
        from cryptography.exceptions import InvalidTag

        AESGCM = _import_crypto()
        aesgcm = AESGCM(self._key)  # type: ignore[operator]
        raw = base64.b64decode(ciphertext)
        nonce, ct = raw[: self._NONCE_SIZE], raw[self._NONCE_SIZE :]
        try:
            return aesgcm.decrypt(nonce, ct, None).decode("utf-8")  # type: ignore[no-any-return]
        except InvalidTag as exc:
            raise ValueError("Decryption failed — wrong key or corrupted data.") from exc

    # ------------------------------------------------------------------
    # Key export
    # ------------------------------------------------------------------

    def export_key(self) -> str:
        """Return the encryption key as a base64 string for safe storage."""
        return base64.b64encode(self._key).decode("ascii")
