import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.config import settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

class AIEngineService:
    def __init__(self):
        self.base_url = settings.ai_engine_url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_image(self, inspection_id: str, image_url: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/predict",
                    json={"inspection_id": inspection_id, "image_url": image_url}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error in AIEngineService.analyze_image: {e}", exc_info=True)
            return {"error": str(e)}

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking AI engine health: {e}", exc_info=True)
            return False

ai_service = AIEngineService()
