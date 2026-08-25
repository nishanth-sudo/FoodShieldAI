# ruff: noqa: E501
"""
Ollama client wrapper for FoodShieldAI.

Provides a unified interface to interact with a locally-running Ollama server.
Ollama exposes an OpenAI-compatible REST API, so the `openai` Python package
is reused as the transport layer — no extra Ollama SDK needed.

Environment variables (all optional, fall back to defaults):
    OLLAMA_BASE_URL   — Ollama server URL  (default: http://localhost:11434/v1)
    OLLAMA_TIMEOUT    — Request timeout in seconds  (default: 120)
    OLLAMA_ENABLED    — Set "false" to disable Ollama and use fallback (default: true)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_TIMEOUT = 120
_DEFAULT_ENABLED = True

# ---------------------------------------------------------------------------
# Supported models — used for validation / documentation
# ---------------------------------------------------------------------------

# Text-only models (for report generation, OCR post-processing, XAI explanation)
TEXT_MODELS = {
    "llama3.1:8b": "Best overall — structured JSON, instruction-following",
    "llama3.2:3b": "Lightweight — fast on CPU, good for short tasks",
    "mistral:7b": "Strong at strict output formats",
    "qwen2.5:7b": "Great JSON adherence, multilingual",
    "qwen2.5:3b": "Lightweight multilingual option",
    "gemma2:9b": "High accuracy, structured reasoning",
    "gemma2:2b": "Smallest viable text model",
    "phi3.5:3.8b": "Efficient, concise explanation generation",
    "phi4:14b": "Best quality, requires 12GB+ VRAM",
}

# Vision-language models (for image augmentation / cross-validation)
VISION_MODELS = {
    "llava:13b": "Strong visual understanding, food image analysis",
    "llava:7b": "Balanced vision model",
    "llava-phi3:3.8b": "Lightweight vision model",
    "minicpm-v:8b": "Excellent food/product image understanding",
    "moondream2": "Ultra-lightweight, basic visual QA",
}


class OllamaClient:
    """
    Thin wrapper around the OpenAI-compatible Ollama REST API.

    Usage::

        client = OllamaClient(model="llama3.1:8b")
        response = client.chat("Generate a food inspection report...")
        print(response)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", str(_DEFAULT_TIMEOUT)))
        self._enabled = os.getenv("OLLAMA_ENABLED", "true").lower() != "false"
        self._client = None

        if self._enabled:
            self._init_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        if not self._enabled or self._client is None:
            return False
        try:
            import httpx

            resp = httpx.get(
                self.base_url.replace("/v1", "/api/tags"),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def chat(self, prompt: str, system: str | None = None) -> str:
        """
        Send a chat completion request to Ollama.

        Args:
            prompt: The user message.
            system: Optional system message prepended to the conversation.

        Returns:
            Model response as a plain string. Empty string on failure.
        """
        if not self._enabled or self._client is None:
            return ""

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Ollama chat request failed: %s", exc)
            return ""

    def chat_json(self, prompt: str, system: str | None = None) -> dict[str, Any] | None:
        """
        Send a chat request and attempt to parse the response as JSON.

        Returns parsed dict or None if parsing fails.
        """
        raw = self.chat(prompt, system=system)
        if not raw:
            return None
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not parse Ollama response as JSON: %s", exc)
            return None

    def chat_vision(self, prompt: str, image_base64: str) -> str:
        """
        Send a vision chat request (requires a vision-capable model like llava).

        Args:
            prompt:        Text question about the image.
            image_base64:  Base64-encoded image bytes.

        Returns:
            Model response as a plain string. Empty string on failure.
        """
        if not self._enabled or self._client is None:
            return ""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Ollama vision request failed: %s", exc)
            return ""

    def list_local_models(self) -> list[str]:
        """Return list of models already pulled in local Ollama instance."""
        try:
            import httpx

            resp = httpx.get(
                self.base_url.replace("/v1", "/api/tags"),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        try:
            from openai import OpenAI  # reuse openai transport for Ollama compat API

            self._client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # required by client lib, ignored by Ollama
                timeout=self.timeout,
            )
            logger.info("OllamaClient initialised — base_url=%s model=%s", self.base_url, self.model)
        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai>=1.0.0"
            )
            self._client = None
