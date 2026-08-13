import pytest
from fastapi import HTTPException
from backend.core.middleware import validate_image_bytes


def test_valid_jpeg_magic_bytes():
    # FF D8 FF + padding to at least 12 bytes
    data = b'\xff\xd8\xff' + b'\x00' * 12
    result = validate_image_bytes(data, "food.jpg")
    assert result.endswith(".jpg") or result.endswith(".jpeg")


def test_valid_png_magic_bytes():
    # 89 50 4E 47 + padding
    data = b'\x89\x50\x4e\x47' + b'\x00' * 12
    result = validate_image_bytes(data, "food.png")
    assert result.endswith(".png")


def test_valid_webp_magic_bytes():
    # RIFF + 4 size bytes + WEBP
    data = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 4
    result = validate_image_bytes(data, "food.webp")
    assert result.endswith(".webp")


def test_invalid_magic_bytes():
    data = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    with pytest.raises(HTTPException) as exc_info:
        validate_image_bytes(data, "bad.jpg")
    assert exc_info.value.status_code == 400


def test_filename_sanitization():
    # Path traversal attempt should be sanitized to just the base filename
    data = b'\xff\xd8\xff' + b'\x00' * 12
    result = validate_image_bytes(data, "path/../../evil.jpg")
    assert "/" not in result
    assert ".." not in result


def test_file_too_small():
    with pytest.raises(HTTPException) as exc_info:
        validate_image_bytes(b'', "empty.jpg")
    assert exc_info.value.status_code == 400


def test_file_exactly_at_minimum_size():
    # Less than 12 bytes also raises
    with pytest.raises(HTTPException):
        validate_image_bytes(b'\xff\xd8\xff', "small.jpg")
