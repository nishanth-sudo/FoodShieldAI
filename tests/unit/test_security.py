import time
from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from backend.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from backend.config import settings


class TestPasswordHashing:
    def test_hash_password(self):
        hashed = hash_password("securepass123")
        assert hashed != "securepass123"
        assert hashed.startswith("$2b$")

    def test_verify_correct_password(self):
        hashed = hash_password("securepass123")
        assert verify_password("securepass123", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("securepass123")
        assert verify_password("wrongpass", hashed) is False

    def test_different_hashes_for_same_password(self):
        h1 = hash_password("samepass")
        h2 = hash_password("samepass")
        assert h1 != h2


class TestJWTTokens:
    def test_create_access_token(self):
        token = create_access_token(user_id="user-123", role="consumer")
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        assert payload["sub"] == "user-123"
        assert payload["role"] == "consumer"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_create_refresh_token(self):
        token = create_refresh_token(user_id="user-123")
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        assert payload["sub"] == "user-123"
        assert payload["type"] == "refresh"
        assert "exp" in payload

    def test_access_token_expiry(self):
        token = create_access_token(user_id="u1", role="admin")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "u1"
        assert payload["role"] == "admin"

    def test_decode_invalid_token(self):
        payload = decode_token("invalid-token")
        assert payload is None

    def test_decode_expired_token(self):
        expired_payload = {
            "sub": "u1",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "type": "access",
        }
        expired_token = jwt.encode(
            expired_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        payload = decode_token(expired_token)
        assert payload is None

    def test_access_and_refresh_tokens_differ(self):
        access = create_access_token("u1", "consumer")
        refresh = create_refresh_token("u1")
        assert access != refresh
