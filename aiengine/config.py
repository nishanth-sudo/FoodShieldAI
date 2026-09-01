from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderType(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"
    TEMPLATE = "template"


class Settings(BaseSettings):
    """
    Centralized application settings loaded from environment variables / .env file.

    All ``os.getenv()`` calls in business logic should be replaced with a
    reference to the singleton returned by :func:`get_settings`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    app_name: str = "FoodShieldAI"
    environment: Literal["development", "staging", "production"] = "development"

    # LLM (provider-agnostic)
    llm_provider: LLMProviderType = LLMProviderType.OLLAMA
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_timeout: int = 120
    llm_temperature: float = 0.3

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_enabled: bool = True

    # vLLM
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "facebook/opt-125m"
    vllm_timeout: int = 120

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # CV models
    device: str = "cpu"
    models_dir: str = ""
    confidence_threshold: float = 0.5

    # OCR
    ocr_backend: str = "mock"
    ollama_ocr_model: str = "llama3.2:3b"
    ollama_xai_model: str = "llama3.2:3b"

    # VLM
    vlm_enabled: bool = False
    vlm_model: str = "llava:7b"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the application-wide singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
