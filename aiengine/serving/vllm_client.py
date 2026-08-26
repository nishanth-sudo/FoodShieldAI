from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, List, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

class VLLMClient:
    """
    Client for interacting with vLLM servers or any OpenAI-compatible LLM endpoint.
    
    Architecture:
    FoodShieldAI
         │
    ┌────┴────┐
    │         │
    ML       LLM
    Inference Inference
    │         │
    PyTorch   vLLM
    │         │
    │    Qwen/Llama/Mistral
    │         │
    └────┬────┘
         ↓
      FastAPI
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 60.0
    ) -> None:
        """Initialize the VLLM Client."""
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.model = model or os.environ.get("VLLM_MODEL", "qwen-2.5-7b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        if OpenAI is not None:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        else:
            self.client = None
            logger.warning("openai package not found, VLLMClient will not work")
        
        logger.info(f"Initialized VLLMClient with base_url={self.base_url}, model={self.model}")

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a prompt and return a plain text response."""
        if self.client is None:
            logger.error("OpenAI client not initialized")
            return ""
            
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""

    def chat_json(self, prompt: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Send a prompt and return a JSON-parsed response."""
        system_prompt = (system or "") + "\nRespond with valid JSON only."
        response_text = self.chat(prompt, system_prompt)
        if not response_text:
            return None
            
        try:
            # Strip potential markdown blocks
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse text: {response_text}")
            return None

    def chat_structured(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Guided generation using a JSON schema if supported by the backend."""
        # For full vLLM guided decoding, we would pass extra_body={"guided_json": response_schema}
        if self.client is None:
            return {}
            
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if response_schema:
            kwargs["extra_body"] = {"guided_json": response_schema}
            
        try:
            response = self.client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or "{}"
            return json.loads(text)
        except Exception as e:
            logger.error(f"Structured chat error: {e}")
            # Fallback to chat_json
            result = self.chat_json(prompt, system)
            return result or {}

    @property
    def is_available(self) -> bool:
        """Check if the LLM server is accessible."""
        if self.client is None:
            return False
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
            
    def list_models(self) -> List[str]:
        """List served models."""
        if self.client is None:
            return []
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
