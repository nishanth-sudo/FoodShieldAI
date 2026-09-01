from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, Literal

# ---------------------------------------------------------------------------
# Domain Types
# ---------------------------------------------------------------------------

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass(frozen=True)
class GenerationOptions:
    """Configuration for LLM generation."""
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    response_format: str | None = None  # e.g., "json_object"

@dataclass(frozen=True)
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float = 0.0
    request_id: str = "unknown"

# ---------------------------------------------------------------------------
# Provider Interface
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """
    Interface for LLM providers (vLLM, Ollama, OpenAI, etc.).
    Ensures the report generator is decoupled from specific API implementations.
    """
    def generate(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: The conversation history/prompt.
            options: Generation hyperparameters.

        Returns:
            A standardized LLMResponse object.
        """
        ...

    @property
    def provider_name(self) -> str:
        """Returns the name of the provider (e.g., 'vllm', 'ollama')."""
        ...
