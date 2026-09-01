from __future__ import annotations

import logging
import time
import functools
from typing import Any, Callable, TypeVar, Generic
from aiengine.llm.provider import LLMProvider, LLMResponse, ChatMessage, GenerationOptions

logger = logging.getLogger(__name__)

T = TypeVar("T")

class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is in OPEN state."""
    pass

class CircuitBreaker:
    """
    Simple circuit breaker to prevent cascading failures.
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.perf_counter()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.error("Circuit breaker transitioned to OPEN state")

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if self.last_failure_time and (time.perf_counter() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info("Circuit breaker transitioned to HALF-OPEN state")
                return True
            return False

        if self.state == "HALF-OPEN":
            return True

        return False

class ResilientLLMProvider:
    """
    Wrapper for LLMProvider that adds retries, timeouts, and circuit breaking.
    """
    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker = CircuitBreaker()

    def generate(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions
    ) -> LLMResponse:
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for provider {self.provider.provider_name}")

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.provider.generate(messages, options)
                self.circuit_breaker.record_success()
                return response
            except Exception as exc:
                last_exception = exc
                # Only retry on transient errors (simplified here)
                logger.warning("LLM request attempt %d failed: %s", attempt + 1, exc)

                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    break

        self.circuit_breaker.record_failure()
        raise last_exception or RuntimeError("LLM generation failed after retries")

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    def is_available(self) -> bool:
        return self.provider.is_available()
