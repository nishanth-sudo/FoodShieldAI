from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from aiengine.llm.provider import ChatMessage, GenerationOptions, LLMProvider
from aiengine.llm.schemas import InspectionReport
from aiengine.llm.prompts import registry as prompt_registry
from aiengine.llm.validator import EvidenceValidator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a food safety inspection expert AI. "
    "Always respond with valid JSON only — no markdown, no explanation outside the JSON block."
)

class LLMReportGenerator:
    """
    Generates food inspection reports using an LLM provider.
    Decoupled from specific LLM implementations via the LLMProvider interface.
    """
    def __init__(
        self,
        provider: LLMProvider | None = None,
        temperature: float = 0.3,
        # Deprecated: model_name was removed when the provider interface was introduced.
        # Accepted here for backward compatibility only; has no effect.
        model_name: str | None = None,
        **_kwargs: object,
    ) -> None:
        if model_name is not None:
            import warnings
            warnings.warn(
                "LLMReportGenerator: 'model_name' is deprecated and has no effect. "
                "Pass a 'provider' (LLMProvider instance) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.provider = provider
        self.temperature = temperature
        self.validator = EvidenceValidator()

    def generate_report(self, inspection_results: dict) -> dict:
        """Generate a structured JSON inspection report from CV model results."""
        prompt = self._build_prompt(inspection_results)

        try:
            if self.provider:
                messages: list[ChatMessage] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                options = GenerationOptions(
                    temperature=self.temperature,
                    response_format="json_object",
                )

                response = self.provider.generate(messages, options)
                result = self._parse_response(response.content, inspection_results)
                if result:
                    # Hallucination Check
                    is_valid, violations = self.validator.validate(result, inspection_results)
                    if not is_valid:
                        logger.error("LLM report failed evidence validation: %s", violations)
                        # We return the template report if the LLM hallucinated
                        return self._template_report(inspection_results)

                    result.setdefault("inspection_date", datetime.now().isoformat())
                    return result

                logger.warning("LLM provider returned empty/invalid JSON — falling back to template")

        except Exception as exc:
            logger.error("LLM report generation failed: %s", exc)

        return self._template_report(inspection_results)

    def generate_summary(self, report: str) -> str:
        """Summarise a report in 2-3 sentences."""
        prompt_ver = prompt_registry.get("report_summary")
        prompt = prompt_ver.template.format(report_text=report)
        try:
            if self.provider:
                messages: list[ChatMessage] = [
                    {"role": "user", "content": prompt},
                ]
                options = GenerationOptions(temperature=self.temperature)
                response = self.provider.generate(messages, options)
                return response.content.strip()

        except Exception:
            pass

        return self._extract_summary(report)

    def _build_prompt(self, results: dict) -> str:
        prompt_ver = prompt_registry.get("inspection_report")
        template = prompt_ver.template

        spoilage = results.get("spoilage", {})
        shelf = results.get("shelf_life", {})
        contamination = results.get("contamination_risks", {})
        food_cls = results.get("food_classification", results)
        defects = results.get("packaging_defects", [])
        ocr = results.get("ocr_data", {})

        defect_lines = (
            "\n".join(
                f"  - {d['defect_type']} (confidence: {d.get('confidence', 0):.2f})"
                for d in (defects if isinstance(defects, list) else [])
            )
            or "  None detected"
        )

        detected_risks_list = []
        if isinstance(contamination, dict):
            for r in contamination.get("detected_risks", []):
                detected_risks_list.append(f"{r['category']} ({r.get('confidence', 0):.2f})")

        return template.format(
            food_type=food_cls.get("food_type", "Unknown"),
            classification_confidence=food_cls.get("confidence_scores", [{"confidence": 0}])[0][
                "confidence"
            ],
            freshness_score=spoilage.get("freshness_score", results.get("freshness_score", 0)),
            is_spoiled=spoilage.get("is_spoiled", False),
            spoilage_severity=spoilage.get("severity_label", "unknown"),
            packaging_defects=defect_lines,
            contamination_risk=contamination.get("overall_risk_level", "unknown")
            if isinstance(contamination, dict)
            else "unknown",
            detected_risks=", ".join(detected_risks_list) or "None",
            shelf_life_days=shelf.get(
                "estimated_days_remaining", results.get("shelf_life_days", "N/A")
            )
            if isinstance(shelf, dict)
            else "N/A",
            freshness_category=shelf.get("freshness_category", "unknown")
            if isinstance(shelf, dict)
            else "unknown",
            ocr_data=json.dumps(ocr.get("parsed", ocr), indent=2)
            if isinstance(ocr, dict)
            else str(ocr),
        )

    def _parse_response(self, response: str, fallback: dict) -> dict | None:
        try:
            json_start = response.index("{")
            json_end = response.rindex("}") + 1
            data = json.loads(response[json_start:json_end])

            # Validate using Pydantic
            validated_report = InspectionReport.model_validate(data)
            return validated_report.model_dump()
        except (ValueError, json.JSONDecodeError, Exception) as exc:
            logger.warning("Could not validate LLM response as InspectionReport: %s", exc)
            return None

    def _template_report(self, results: dict) -> dict:
        spoilage = results.get("spoilage", {})
        shelf = results.get("shelf_life", {})
        contamination = results.get("contamination_risks", {})
        food_type = results.get("food_type", "Unknown")
        freshness = spoilage.get("freshness_score", results.get("freshness_score", 0))
        is_spoiled = spoilage.get("is_spoiled", False)

        risk_flags = []
        if is_spoiled:
            risk_flags.append("Spoilage detected — item is not fit for consumption")
        if freshness < 0.3:
            risk_flags.append("Critically low freshness score")
        defects = results.get("packaging_defects", [])
        if isinstance(defects, list) and len(defects) > 0:
            risk_flags.append(f"{len(defects)} packaging defect(s) found")
        if isinstance(contamination, dict) and contamination.get("overall_risk_level") == "high":
            risk_flags.append("High contamination risk detected")

        verdict = "fail" if risk_flags else "pass"

        recommendations = []
        if is_spoiled:
            recommendations.append("Discard the item immediately due to spoilage")
        if isinstance(defects, list) and defects:
            recommendations.append("Inspect packaging integrity — consider repackaging")
        if 0.3 < freshness < 0.5:
            recommendations.append("Consume within 24 hours")
        recommendations.append("Store at recommended temperature")

        shelf_days = (
            shelf.get("estimated_days_remaining", results.get("shelf_life_days", "N/A"))
            if isinstance(shelf, dict)
            else "N/A"
        )

        return {
            "report_title": f"Food Inspection Report — {food_type}",
            "executive_summary": (
                f"Inspection of {food_type} completed. "
                f"Freshness score: {freshness:.2f}/1.0. "
                f"Estimated shelf life: {shelf_days} days. "
                f"Overall verdict: {verdict}."
            ),
            "detailed_findings": [
                {
                    "area": "Food Classification",
                    "status": "info",
                    "detail": f"Identified as {food_type}",
                },
                {
                    "area": "Freshness",
                    "status": "fail" if is_spoiled else "pass",
                    "detail": f"Score {freshness:.2f}/1.0",
                },
                {
                    "area": "Packaging",
                    "status": "fail" if defects else "pass",
                    "detail": f"{len(defects) if isinstance(defects, list) else 0} defect(s)",
                },
                {
                    "area": "Contamination Risk",
                    "status": contamination.get("overall_risk_level", "pass")
                    if isinstance(contamination, dict)
                    else "pass",
                    "detail": contamination.get("overall_risk_level", "none")
                    if isinstance(contamination, dict)
                    else "none",
                },
                {"area": "Shelf Life", "status": "info", "detail": f"{shelf_days} days remaining"},
            ],
            "risk_flags": risk_flags,
            "recommendations": recommendations,
            "overall_verdict": verdict,
            "inspection_date": datetime.now().isoformat(),
        }

    @staticmethod
    def _extract_summary(report: str) -> str:
        if len(report) < 200:
            return report
        return report[:200] + "..."

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        """
        Deprecated. Infer a provider type string from a model name.

        Kept for backward compatibility with existing tests and tooling.
        New code should use :func:`aiengine.llm.factory.create_provider`
        with an explicit provider type string.

        Returns:
            One of ``"gpt"``, ``"hf"``, ``"local/"`` or a custom prefix.
        """
        import warnings
        warnings.warn(
            "LLMReportGenerator._detect_provider() is deprecated. "
            "Use aiengine.llm.factory.create_provider() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model_name.startswith("ollama/") or model_name == "ollama":
            return "ollama"
        if model_name.startswith("gpt"):
            return "gpt"
        if model_name.startswith("hf/"):
            return "hf"
        if model_name.startswith("local/") or model_name == "local":
            return "local/"
        # Default fallback
        return "local/"
