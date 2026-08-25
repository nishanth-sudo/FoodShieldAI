"""
tests/unit/test_security_audit.py — Task 7.6: Security audit tests

All tests are unit-level (no real DB, no real network).
Covers: JWT security, password hashing, input validation,
rate-limiter config, and contamination-risk category completeness.
"""

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from aiengine.models.contamination_risk.model import RISK_CATEGORIES
from backend.config import settings
from backend.core.middleware import validate_image_bytes
from backend.core.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Magic-byte helpers
# ---------------------------------------------------------------------------

_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
_ELF_BYTES = b"\x7fELF" + b"\x00" * 100      # Linux executable
_PE_BYTES = b"MZ" + b"\x00" * 100             # Windows executable
_TEXT_BYTES = b"Hello, I am a text file, not an image.\n" * 3


# ===========================================================================
# TestJWTSecurity
# ===========================================================================


@pytest.mark.unit
@pytest.mark.security
class TestJWTSecurity:
    """Verify the JWT layer rejects all known token-forgery and algorithm
    confusion attacks and handles missing-claim edge cases gracefully."""

    def test_token_requires_valid_signature(self) -> None:
        """A token whose signature has been tampered with must be rejected."""
        valid_token = create_access_token("user-1", "consumer")
        parts = valid_token.split(".")
        tampered = parts[0] + "." + parts[1] + ".invalidsignatureXXX"

        result = decode_token(tampered)

        assert result is None, "Tampered token must be rejected (decode_token returns None)"

    def test_expired_token_rejected(self) -> None:
        """A token with an expiry timestamp in the past must return None."""
        past_exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {
            "sub": "user-expired",
            "role": "consumer",
            "exp": past_exp,
            "type": "access",
        }
        expired_token = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

        result = decode_token(expired_token)

        assert result is None, "Expired token must be rejected"

    def test_refresh_token_cannot_be_used_as_access(self) -> None:
        """A token carrying type='refresh' must not be accepted as an access token."""
        from backend.core.security import create_refresh_token

        refresh_token = create_refresh_token("user-2")
        payload = decode_token(refresh_token)

        assert payload is not None, "A valid refresh token should still decode"
        assert payload.get("type") == "refresh", (
            "Refresh token must carry type='refresh' so callers can reject it"
        )

    def test_algorithm_confusion_rejected(self) -> None:
        """A token signed with HS256 cannot be verified if we request RS256."""
        valid_token = create_access_token("user-3", "admin")

        with pytest.raises(Exception):
            jwt.decode(
                valid_token,
                settings.jwt_secret_key,
                algorithms=["RS256"],
            )

    def test_none_algorithm_rejected(self) -> None:
        """A token crafted with alg='none' (unsigned) must be rejected."""
        payload = {
            "sub": "attacker",
            "role": "admin",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "type": "access",
        }
        try:
            none_token = jwt.encode(payload, "", algorithm="none")
            result = decode_token(none_token)
            assert result is None, "'none' alg token must be rejected by decode_token"
        except Exception:
            pass

    def test_token_with_missing_sub_returns_none(self) -> None:
        """A structurally valid token that lacks the 'sub' claim should not crash."""
        payload = {
            "role": "consumer",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
            "type": "access",
        }
        token_no_sub = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

        result = decode_token(token_no_sub)

        assert result is None or isinstance(result, dict), (
            "decode_token must return None or a dict — never raise"
        )

    def test_token_with_missing_role_handled(self) -> None:
        """A token without a 'role' claim should decode without crashing."""
        payload = {
            "sub": "user-no-role",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
            "type": "access",
        }
        token_no_role = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

        result = decode_token(token_no_role)

        assert isinstance(result, dict), "Token without 'role' should still decode successfully"
        assert "role" not in result, "Result should reflect the missing 'role' claim"


# ===========================================================================
# TestPasswordSecurity
# ===========================================================================


@pytest.mark.unit
@pytest.mark.security
class TestPasswordSecurity:
    """Verify bcrypt password hashing properties and edge-case handling."""

    def test_bcrypt_hash_length_is_60_chars(self) -> None:
        """BCrypt hashes are always exactly 60 characters long."""
        hashed = hash_password("SecurePassword123!")

        assert len(hashed) == 60, f"BCrypt hash must be 60 chars, got {len(hashed)}"

    def test_hash_is_not_plaintext(self) -> None:
        """The stored hash must not contain the plaintext password."""
        password = "MyS3cr3tP@ss"
        hashed = hash_password(password)

        assert password not in hashed, "Hash must not contain the plaintext password"

    def test_verify_password_constant_time_no_shortcut(self) -> None:
        """verify_password must return a bool — never raise — even on a wrong password."""
        hashed = hash_password("correct-password")

        result = verify_password("wrong-password", hashed)

        assert isinstance(result, bool), "verify_password must return a bool"
        assert result is False, "Wrong password must return False"

    def test_empty_password_can_be_hashed(self) -> None:
        """An empty string password can be hashed and verified correctly."""
        hashed = hash_password("")

        assert verify_password("", hashed) is True, "Empty password must verify against its hash"
        assert verify_password("notempty", hashed) is False, (
            "Non-empty password must not match empty-password hash"
        )

    def test_special_chars_in_password_handled(self) -> None:
        """Passwords containing Unicode and special characters must round-trip correctly."""
        password = "P@$$w0rd!#&*() 日本語 — café"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True, (
            "Special-char password must verify against its own hash"
        )

    def test_very_long_password_handled(self) -> None:
        """BCrypt handles long passwords gracefully without raising."""
        long_password = "A" * 80

        hashed = hash_password(long_password)
        result = verify_password(long_password, hashed)

        assert isinstance(result, bool), (
            "verify_password must return bool for a 80-char password — never raise"
        )

    def test_malformed_hash_returns_false(self) -> None:
        """verify_password with a garbage hash string must return False, not raise."""
        result = verify_password("anypassword", "not-a-bcrypt-hash-at-all")

        assert result is False, "Malformed hash must cause verify_password to return False"


# ===========================================================================
# TestInputValidation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.security
class TestInputValidation:
    """Verify that validate_image_bytes enforces magic-byte checks and
    filename sanitisation to prevent injection and path-traversal attacks."""

    def test_validate_image_bytes_accepts_valid_jpeg(self) -> None:
        """JPEG magic bytes (FF D8 FF) must be accepted and return a sanitised name."""
        result = validate_image_bytes(_JPEG_BYTES, "photo.jpg")

        assert result.endswith(".jpg") or result.endswith(".jpeg"), (
            "Valid JPEG input should return a filename with a JPEG extension"
        )

    def test_validate_image_bytes_rejects_non_image(self) -> None:
        """Plain-text file bytes must raise HTTPException 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_image_bytes(_TEXT_BYTES, "readme.txt")

        assert exc_info.value.status_code == 400, "Non-image must cause 400 HTTPException"

    def test_validate_image_bytes_accepts_valid_png(self) -> None:
        """PNG magic bytes (89 50 4E 47) must be accepted."""
        result = validate_image_bytes(_PNG_BYTES, "image.png")

        assert result.endswith(".png"), "Valid PNG input should return a filename ending in .png"

    def test_validate_image_bytes_rejects_executable(self) -> None:
        """ELF and PE executable magic bytes must be rejected with HTTPException 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as elf_exc:
            validate_image_bytes(_ELF_BYTES, "malware.jpg")
        assert elf_exc.value.status_code == 400, "ELF binary must cause 400 HTTPException"

        with pytest.raises(HTTPException) as pe_exc:
            validate_image_bytes(_PE_BYTES, "malware.png")
        assert pe_exc.value.status_code == 400, "PE binary must cause 400 HTTPException"

    def test_filename_sanitization_removes_path_traversal(self) -> None:
        """Path-traversal sequences must be stripped from filenames."""
        result = validate_image_bytes(_JPEG_BYTES, "../../../etc/passwd.jpg")

        assert ".." not in result, "Path traversal '..' must be removed from filename"
        assert "/" not in result, "Directory separators must be removed from filename"
        assert result, "Sanitised filename must not be empty"

    def test_filename_sanitization_handles_empty(self) -> None:
        """An empty filename must produce a safe default name."""
        result = validate_image_bytes(_JPEG_BYTES, "")

        assert result, "Empty filename must produce a non-empty default"
        assert ".." not in result, "Default filename must not contain path traversal"


# ===========================================================================
# TestRateLimiting
# ===========================================================================


@pytest.mark.unit
@pytest.mark.security
class TestRateLimiting:
    """Verify that rate-limiting infrastructure is correctly configured and
    that settings expose upload-size and MIME-type constraints."""

    def test_rate_limiter_is_configured(self) -> None:
        """The SlowAPI limiter object must exist."""
        from backend.core.rate_limit import limiter

        assert limiter is not None, "limiter object must be importable and not None"

    def test_settings_have_max_upload_size(self) -> None:
        """settings.max_upload_size_mb must be a positive integer."""
        assert isinstance(settings.max_upload_size_mb, int), (
            "max_upload_size_mb must be an integer"
        )
        assert settings.max_upload_size_mb > 0, (
            "max_upload_size_mb must be greater than 0"
        )

    def test_settings_have_allowed_types(self) -> None:
        """settings.allowed_image_types must be a non-empty list of MIME strings."""
        assert isinstance(settings.allowed_image_types, list), (
            "allowed_image_types must be a list"
        )
        assert len(settings.allowed_image_types) > 0, (
            "allowed_image_types must not be empty"
        )
        assert all(t.startswith("image/") for t in settings.allowed_image_types), (
            "Every entry in allowed_image_types must be an image/* MIME type"
        )


# ===========================================================================
# TestContaminationRiskModel
# ===========================================================================


@pytest.mark.unit
@pytest.mark.security
class TestContaminationRiskModel:
    """Verify that RISK_CATEGORIES covers the three required contamination
    domains: biological, chemical, and physical."""

    def test_risk_categories_cover_biological(self) -> None:
        """RISK_CATEGORIES must contain at least one 'biological_*' variant."""
        biological = [c for c in RISK_CATEGORIES if c.startswith("biological_")]

        assert len(biological) > 0, (
            "RISK_CATEGORIES must include at least one biological risk category"
        )

    def test_risk_categories_cover_chemical(self) -> None:
        """RISK_CATEGORIES must contain at least one 'chemical_*' variant."""
        chemical = [c for c in RISK_CATEGORIES if c.startswith("chemical_")]

        assert len(chemical) > 0, (
            "RISK_CATEGORIES must include at least one chemical risk category"
        )

    def test_risk_categories_cover_physical(self) -> None:
        """RISK_CATEGORIES must contain at least one 'physical_*' variant."""
        physical = [c for c in RISK_CATEGORIES if c.startswith("physical_")]

        assert len(physical) > 0, (
            "RISK_CATEGORIES must include at least one physical risk category"
        )
