from __future__ import annotations

class FoodShieldError(Exception):
    """Base exception for all FoodShieldAI domain errors."""
    pass

class ModelError(FoodShieldError):
    """Errors related to ML model loading or inference."""
    pass

class ModelLoadError(ModelError):
    """Raised when a model fails to load from disk."""
    pass

class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    pass

class XAIError(FoodShieldError):
    """Errors related to explainability generation."""
    pass

class ExplanationError(XAIError):
    """Raised when an explanation cannot be generated."""
    pass

class UnsupportedExplainerError(XAIError):
    """Raised when an unsupported XAI method is requested."""
    pass

class LLMError(FoodShieldError):
    """Errors related to LLM provider communication."""
    pass

class LLMConnectionError(LLMError):
    """Raised when the LLM provider is unreachable."""
    pass

class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""
    pass

class LLMResponseError(LLMError):
    """Raised when the LLM returns an invalid or malformed response."""
    pass

class OCRError(FoodShieldError):
    """Errors related to OCR extraction."""
    pass

class ValidationError(FoodShieldError):
    """Raised when input data fails validation."""
    pass
