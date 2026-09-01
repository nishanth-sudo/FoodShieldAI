from __future__ import annotations

import logging
import os
import time
from typing import Any

from aiengine.llm.provider import ChatMessage, GenerationOptions, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementing the LLMProvider interface.
    """
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout or 120
        self._client = None

        if self.api_key:
            self._init_client()
        else:
            logger.warning("OpenAI API key not found. OpenAIProvider will be unavailable.")

    @property
    def provider_name(self) -> str:
        return "openai"

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            logger.info("OpenAIProvider initialised — model=%s", self.model)
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai>=1.0.0")
            self._client = None

    def generate(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions,
    ) -> LLMResponse:
        """
        Generate a response using the OpenAI API.
        """
        if self._client is None:
            raise RuntimeError("OpenAIProvider client not initialised (missing API key or 'openai' package).")

        start_time = time.perf_counter()
        try:
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
            logger.error("OpenAIProvider generation failed: %s", exc)
            raise RuntimeError(f"OpenAI generation error: {exc}") from exc

    def is_available(self) -> bool:
        """Check if API key is present."""
        return self._client is not None
