# ruff: noqa: E501  # long prompt template lines inside triple-quoted strings
import json
import logging
from datetime import datetime

from aiengine.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """You are a food safety inspection expert. Generate a comprehensive inspection report in JSON format based on the following AI analysis results.

## Inspection Results

### Food Classification
- Food type: {food_type}
- Confidence: {classification_confidence}

### Freshness Assessment
- Freshness score: {freshness_score}/1.0
- Spoilage detected: {is_spoiled}
- Spoilage severity: {spoilage_severity}

### Packaging Defects
{packaging_defects}

### Contamination Risks
- Overall risk level: {contamination_risk}
- Detected risks: {detected_risks}

### Shelf Life
- Estimated days remaining: {shelf_life_days}
- Freshness category: {freshness_category}

### OCR Label Data
{ocr_data}

## Output Format
Return a valid JSON object with these fields:
- "report_title": short descriptive title
- "executive_summary": 2-3 sentence overview
- "detailed_findings": array of finding objects with "area", "status", "detail"
- "risk_flags": array of any high-risk items found
- "recommendations": array of actionable recommendation strings
- "overall_verdict": "pass", "conditional_pass", or "fail"
- "inspection_date": current date
"""

SUMMARY_TEMPLATE = """Summarize the following food inspection report in 2-3 sentences focusing on the key findings and verdict:

{report_text}

Summary:"""

SYSTEM_PROMPT = (
    "You are a food safety inspection expert AI. "
    "Always respond with valid JSON only — no markdown, no explanation outside the JSON block."
)

# ---------------------------------------------------------------------------
# Provider prefixes
# ---------------------------------------------------------------------------

_PROVIDER_OLLAMA = "ollama"
_PROVIDER_GPT = "gpt"
_PROVIDER_HF = "hf"
_PROVIDER_LOCAL = "local/"


class LLMReportGenerator:
    """
    Generates food inspection reports using an LLM backend.

    Supported backends (selected via ``model_name`` prefix):
    - ``ollama/<model>``  — Local Ollama server (recommended, free, private)
    - ``gpt-*``           — OpenAI GPT models (requires OPENAI_API_KEY)
    - ``hf/*``            — HuggingFace Inference API
    - ``local/*``         — Template-based fallback (no LLM)

    Examples::

        # Ollama (recommended)
        gen = LLMReportGenerator(model_name="ollama/llama3.1:8b")

        # OpenAI
        gen = LLMReportGenerator(model_name="gpt-4")

        # Offline template fallback
        gen = LLMReportGenerator(model_name="local/template")
    """

    def __init__(self, model_name: str = "ollama/llama3.1:8b", temperature: float = 0.3) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self._client = None
        self._provider: str = self._detect_provider(model_name)

        if self._provider == _PROVIDER_OLLAMA:
            self._init_ollama_client()
        elif self._provider == _PROVIDER_GPT:
            self._init_openai_client()
        elif self._provider == _PROVIDER_HF:
            self._init_huggingface_client()
        # "local" → no client needed; _template_report() is used directly

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self, inspection_results: dict) -> dict:
        """Generate a structured JSON inspection report from CV model results."""
        prompt = self._build_prompt(inspection_results)
        try:
            if self._provider == _PROVIDER_OLLAMA and self._client is not None:
                result = self._client.chat_json(prompt, system=SYSTEM_PROMPT)
                if result:
                    result.setdefault("inspection_date", datetime.now().isoformat())
                    return result
                logger.warning("Ollama returned empty/invalid JSON — falling back to template")

            elif self._provider == _PROVIDER_GPT and self._client is not None:
                raw = self._call_openai(prompt)
                return self._parse_response(raw, inspection_results)

            elif self._provider == _PROVIDER_HF and self._client is not None:
                raw = self._call_huggingface(prompt)
                return self._parse_response(raw, inspection_results)

        except Exception as exc:
            logger.error("LLM report generation failed (%s): %s", self._provider, exc)

        return self._template_report(inspection_results)

    def generate_summary(self, report: str) -> str:
        """Summarise a report in 2-3 sentences."""
        prompt = SUMMARY_TEMPLATE.format(report_text=report)
        try:
            if self._provider == _PROVIDER_OLLAMA and self._client is not None:
                summary = self._client.chat(prompt)
                if summary:
                    return summary.strip()

            elif self._provider == _PROVIDER_GPT and self._client is not None:
                return self._call_openai(prompt).strip()

            elif self._provider == _PROVIDER_HF and self._client is not None:
                return self._call_huggingface(prompt).strip()

        except Exception:
            pass

        return self._extract_summary(report)

    # ------------------------------------------------------------------
    # Provider initialisation
    # ------------------------------------------------------------------

    def _init_ollama_client(self) -> None:
        bare_model = self.model_name.replace("ollama/", "", 1)
        self._client = OllamaClient(
            model=bare_model,
            temperature=self.temperature,
            max_tokens=2000,
        )
        if not self._client.is_available:
            logger.warning(
                "Ollama server not reachable at %s. "
                "Start Ollama with `ollama serve` and pull the model with "
                "`ollama pull %s`. Falling back to template reports.",
                self._client.base_url,
                bare_model,
            )
            self._client = None

    def _init_openai_client(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI()
        except ImportError:
            logger.warning("openai not installed. Install with: pip install openai")
            self._client = None

    def _init_huggingface_client(self) -> None:
        try:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient()
        except ImportError:
            logger.warning(
                "huggingface-hub not installed. Install with: pip install huggingface-hub"
            )
            self._client = None

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _call_openai(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""

    def _call_huggingface(self, prompt: str) -> str:
        response = self._client.text_generation(
            prompt,
            model=self.model_name.replace("hf/", ""),
            temperature=self.temperature,
            max_new_tokens=2000,
        )
        return response

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, results: dict) -> str:
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

        return REPORT_TEMPLATE.format(
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

    # ------------------------------------------------------------------
    # Response parsing / template fallback
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, fallback: dict) -> dict:
        try:
            json_start = response.index("{")
            json_end = response.rindex("}") + 1
            parsed = json.loads(response[json_start:json_end])
            parsed["inspection_date"] = parsed.get("inspection_date", datetime.now().isoformat())
            return parsed
        except (ValueError, json.JSONDecodeError):
            logger.warning("Could not parse LLM response as JSON, using template")
            return self._template_report(fallback)

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
    def _detect_provider(model_name: str) -> str:
        if model_name.startswith(_PROVIDER_OLLAMA + "/"):
            return _PROVIDER_OLLAMA
        if model_name.startswith(_PROVIDER_GPT):
            return _PROVIDER_GPT
        if model_name.startswith(_PROVIDER_HF):
            return _PROVIDER_HF
        return _PROVIDER_LOCAL  # "local/*" or any unrecognised string → template fallback

    @staticmethod
    def _extract_summary(report: str) -> str:
        if len(report) < 200:
            return report
        return report[:200] + "..."
