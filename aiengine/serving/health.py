from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: str = ""


@dataclass
class HealthReport:
    overall: Literal["ok", "degraded", "unavailable"]
    components: list[ComponentHealth] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.overall,
            "components": {
                c.name: {"status": c.status, "detail": c.detail}
                for c in self.components
            },
        }


class HealthChecker:
    """
    Checks readiness of all required FoodShieldAI components.
    Used by the /health and /ready endpoints.

    Args:
        check_vllm:   Whether to ping the vLLM server.
        check_ollama: Whether to ping the Ollama server.
        vllm_url:     Base URL for the vLLM service (no path).
        ollama_url:   Base URL for the Ollama service (no path).
    """

    def __init__(
        self,
        check_vllm: bool = False,
        check_ollama: bool = True,
        vllm_url: str = "http://vllm:8000",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.check_vllm = check_vllm
        self.check_ollama = check_ollama
        self.vllm_url = vllm_url
        self.ollama_url = ollama_url

    def check(self) -> HealthReport:
        """Run all configured health checks and return a consolidated report."""
        components: list[ComponentHealth] = []

        # The API process itself is always reachable if this code runs
        components.append(ComponentHealth(name="api", status="ok"))

        if self.check_vllm:
            components.append(self._check_vllm())

        if self.check_ollama:
            components.append(self._check_ollama())

        statuses = [c.status for c in components]
        if all(s == "ok" for s in statuses):
            overall: Literal["ok", "degraded", "unavailable"] = "ok"
        else:
            # Any unavailable/degraded component = degraded overall.
            # Individual LLM providers are non-critical (fallback chain exists).
            overall = "degraded"

        return HealthReport(overall=overall, components=components)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_vllm(self) -> ComponentHealth:
        try:
            import httpx

            resp = httpx.get(f"{self.vllm_url}/health", timeout=3.0)
            if resp.status_code == 200:
                return ComponentHealth(name="vllm", status="ok")
            return ComponentHealth(
                name="vllm",
                status="degraded",
                detail=f"HTTP {resp.status_code}",
            )
        except Exception as exc:
            logger.debug("vLLM health check failed: %s", exc)
            return ComponentHealth(
                name="vllm", status="unavailable", detail=str(exc)
            )

    def _check_ollama(self) -> ComponentHealth:
        try:
            import httpx

            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                return ComponentHealth(name="ollama", status="ok")
            return ComponentHealth(
                name="ollama",
                status="degraded",
                detail=f"HTTP {resp.status_code}",
            )
        except Exception as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return ComponentHealth(
                name="ollama", status="unavailable", detail=str(exc)
            )
