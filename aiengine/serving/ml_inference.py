from __future__ import annotations
import logging
import psutil
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MLInferenceServer:
    """
    Inference server wrapper for FoodShieldAI computer vision models.
    Provides a clean serving abstraction over the ML pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ML Inference Server.
        
        Args:
            config: Configuration dictionary with model paths, device, etc.
        """
        self.config = config
        self.device = config.get("device", "cpu")
        self.models: Dict[str, Any] = {}
        self.supported_models = [
            "food_classification",
            "spoilage_detection",
            "packaging_defect",
            "contamination_risk",
            "shelf_life"
        ]
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))
        logger.info(f"Initialized MLInferenceServer on {self.device}")

    def load_models(self) -> None:
        """Lazy load all CV models."""
        logger.info("Loading computer vision models...")
        # Simulating model loading
        for model_name in self.supported_models:
            if model_name not in self.models:
                logger.info(f"Loading {model_name} model to {self.device}...")
                self.models[model_name] = f"Loaded {model_name}"
        logger.info("Finished loading all models.")

    def predict(self, image: Any, models: List[str] = None) -> Dict[str, Any]:
        """
        Run inference using the selected models.
        
        Args:
            image: The image to process (e.g., numpy array or PIL Image).
            models: List of model names to run. Defaults to all supported models.
            
        Returns:
            Dictionary containing prediction results from each model.
        """
        if models is None:
            models = self.supported_models
            
        results = {}
        
        def run_model(model_name: str) -> Dict[str, Any]:
            if model_name not in self.models:
                # Lazy load if not already loaded
                logger.info(f"Lazy loading {model_name}...")
                self.models[model_name] = f"Loaded {model_name}"
                
            # Simulate prediction
            return {"status": "success", "prediction": f"mock prediction for {model_name}"}

        # Run predictions in parallel
        futures = {model_name: self.executor.submit(run_model, model_name) for model_name in models}
        
        for model_name, future in futures.items():
            try:
                results[model_name] = future.result()
            except Exception as e:
                logger.error(f"Error running model {model_name}: {e}")
                results[model_name] = {"status": "error", "error": str(e)}
                
        return results

    def health_check(self) -> Dict[str, Any]:
        """Check GPU/model status and memory usage."""
        status = {
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "system_memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent_used": psutil.virtual_memory().percent
            }
        }
        
        if self.device.startswith("cuda"):
            try:
                import torch
                status["gpu"] = {
                    "available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved()
                }
            except ImportError:
                status["gpu"] = {"status": "error", "message": "torch not available"}
                
        return status
