from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import json

from aiengine.serving.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified interface for LLM operations in FoodShieldAI.
    Wraps either OllamaClient or VLLMClient based on configuration.
    """
    
    def __init__(self, backend: str = 'auto', model: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the LLM Client.
        
        Args:
            backend: 'vllm', 'ollama', 'openai', or 'auto' (tries vllm, then ollama)
            model: The model to use.
            **kwargs: Additional arguments for the underlying clients.
        """
        self.backend = backend
        self.client = None
        
        if backend in ('auto', 'vllm', 'openai'):
            try:
                vllm = VLLMClient(model=model, **kwargs)
                if vllm.is_available or backend in ('vllm', 'openai'):
                    self.client = vllm
                    logger.info(f"Using VLLMClient backend for LLMClient (model={model})")
            except Exception as e:
                logger.warning(f"Failed to initialize VLLMClient: {e}")
                
        if self.client is None and backend in ('auto', 'ollama'):
            # Fallback logic for ollama would go here
            logger.info(f"Using fallback OllamaClient logic (mocked) for LLMClient (model={model})")
            
        if self.client is None:
            logger.warning("No working LLM backend found. Operations will fail or return mock data.")

    def generate_report(self, inspection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate inspection report from structured CV+XAI results.
        
        Args:
            inspection_data: Dict containing CV findings and XAI evidence.
            
        Returns:
            Structured JSON report.
        """
        if self.client is None:
            return {"status": "error", "message": "LLM client not available"}
            
        prompt = f"Generate an inspection report for the following data:\n{json.dumps(inspection_data, indent=2)}"
        system_prompt = "You are an AI Food Safety Inspector. Analyze the data and provide a structured JSON report."
        
        result = self.client.chat_json(prompt, system=system_prompt)
        return result or {"status": "error", "message": "Failed to generate report"}

    def generate_explanation(self, xai_data: Dict[str, Any]) -> str:
        """
        Generate natural language explanation from XAI evidence.
        
        Args:
            xai_data: Dict containing SHAP, Grad-CAM, or counterfactuals.
            
        Returns:
            String explanation.
        """
        if self.client is None:
            return "LLM client not available to generate explanation."
            
        prompt = f"Explain the following XAI evidence in natural language for a food safety inspector:\n{json.dumps(xai_data, indent=2)}"
        system_prompt = "You are an AI Explainer for computer vision models in food safety. Be clear, concise, and accurate."
        
        return self.client.chat(prompt, system=system_prompt)

    @property
    def is_available(self) -> bool:
        """Check if the underlying LLM client is available."""
        if self.client is None:
            return False
        return getattr(self.client, "is_available", False)
