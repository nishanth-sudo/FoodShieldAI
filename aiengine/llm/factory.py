from __future__ import annotations

import logging
from typing import Callable

from aiengine.llm.provider import LLMProvider
from aiengine.domain.exceptions import LLMError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry — maps provider-type string → factory callable
# ---------------------------------------------------------------------------

def _make_ollama(model: str, base_url: str | None) -> LLMProvider:
    from aiengine.llm.ollama_provider import OllamaProvider
    return OllamaProvider(model=model, base_url=base_url)


def _make_vllm(model: str, base_url: str | None) -> LLMProvider:
    from aiengine.llm.vllm_provider import VLLMProvider
    return VLLMProvider(model=model, base_url=base_url)


def _make_openai(model: str, base_url: str | None) -> LLMProvider:
    from aiengine.llm.openai_provider import OpenAIProvider
    # OpenAI provider reads the API key from the environment via OPENAI_API_KEY.
    return OpenAIProvider(model=model)


_PROVIDER_REGISTRY: dict[str, Callable[[str, str | None], LLMProvider]] = {
    "ollama": _make_ollama,
    "vllm": _make_vllm,
    "openai": _make_openai,
}


def create_provider(
    provider_type: str,
    model: str,
    base_url: str | None = None,
) -> LLMProvider | None:
    """
    Instantiate and return an LLMProvider for the given provider type.

    Uses a registry dict to avoid if/elif chains in business logic,
    satisfying Spec §16 (Dependency Inversion / Open-Closed).

    Args:
        provider_type: One of ``"ollama"``, ``"vllm"``, ``"openai"``.
        model:         Model name/identifier to pass to the provider.
        base_url:      Optional base URL override for the provider endpoint.

    Returns:
        An :class:`~aiengine.llm.provider.LLMProvider` instance, or ``None``
        if instantiation fails (e.g. missing optional dependency).

    Raises:
        LLMError: If *provider_type* is not registered.
    """
    factory = _PROVIDER_REGISTRY.get(provider_type.lower())
    if factory is None:
        registered = list(_PROVIDER_REGISTRY.keys())
        raise LLMError(
            f"Unknown LLM provider type '{provider_type}'. "
            f"Registered providers: {registered}"
        )

    try:
        provider = factory(model, base_url)
        logger.info(
            "Created LLM provider — type=%s model=%s base_url=%s",
            provider_type, model, base_url,
        )
        return provider
    except ImportError as exc:
        logger.error(
            "Could not import provider '%s': %s. "
            "Install the required package and retry.",
            provider_type, exc,
        )
        return None
    except Exception as exc:
        logger.error(
            "Failed to create provider '%s': %s", provider_type, exc
        )
        return None
