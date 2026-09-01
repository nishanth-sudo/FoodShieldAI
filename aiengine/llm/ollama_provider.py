from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from aiengine.llm.provider import ChatMessage, GenerationOptions, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_TIMEOUT = 120

class OllamaProvider(LLMProvider):
    """
    Ollama provider implementing the LLMProvider interface.
    Uses the OpenAI-compatible REST API provided by Ollama.
    """
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", str(_DEFAULT_TIMEOUT)))
        self._client = None

        self._init_client()

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",
                timeout=self.timeout,
            )
            logger.info("OllamaProvider initialised — base_url=%s model=%s", self.base_url, self.model)
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai>=1.0.0")
            self._client = None

    def generate(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions,
    ) -> LLMResponse:
        """
        Generate a response using the Ollama server.
        """
        if self._client is None:
            raise RuntimeError("OllamaProvider client not initialised (missing 'openai' package).")

        start_time = time.perf_counter()
        try:
            # Convert ChatMessage TypedDict to what openai-python expects
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]

            response = self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            )

            latency = time.perf_counter() - start_time
            content = response.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                latency_seconds=latency,
                request_id=response.id,
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
            )

        except Exception as exc:
            logger.error("OllamaProvider generation failed: %s", exc)
            raise RuntimeError(f"Ollama generation error: {exc}") from exc

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            import httpx
            resp = httpx.get(
                self.base_url.replace("/v1", "/api/tags"),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
