from backend.infrastructure.cache import compute_image_hash
from backend.core.rate_limit import InMemoryRateLimiter

def test_compute_image_hash_consistency():
    data = b'testdata'
    assert compute_image_hash(data) == compute_image_hash(data)

def test_compute_image_hash_uniqueness():
    assert compute_image_hash(b'data1') != compute_image_hash(b'data2')

def test_compute_image_hash_type():
    assert isinstance(compute_image_hash(b'data'), str)

def test_in_memory_rate_limiter_allows_requests():
    limiter = InMemoryRateLimiter(3, 60)
    assert limiter.is_allowed('test') is True
    assert limiter.is_allowed('test') is True
    assert limiter.is_allowed('test') is True

def test_in_memory_rate_limiter_blocks_when_exceeded():
    limiter = InMemoryRateLimiter(3, 60)
    limiter.is_allowed('test2')
    limiter.is_allowed('test2')
    limiter.is_allowed('test2')
    assert limiter.is_allowed('test2') is False
