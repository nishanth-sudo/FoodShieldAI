import json
import logging
from datetime import datetime

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


class LLMReportGenerator:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self._client = None
        self._is_using_local = model_name.startswith("local/")

        if not self._is_using_local:
            self._init_remote_client()

    def _init_remote_client(self):
        if self.model_name.startswith("gpt"):
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                logger.warning("openai not installed. Install with: pip install openai")
                self._client = None
        elif self.model_name.startswith("hf") or self.model_name.startswith("local/"):
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient()
            except ImportError:
                logger.warning("huggingface-hub not installed. Install with: pip install huggingface-hub")
                self._client = None

    def generate_report(self, inspection_results: dict) -> dict:
        if self._client is None or self._is_using_local:
            return self._template_report(inspection_results)

        prompt = self._build_prompt(inspection_results)
        try:
            response = self._call_llm(prompt)
            return self._parse_response(response, inspection_results)
        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            return self._template_report(inspection_results)

    def generate_summary(self, report: str) -> str:
        if self._client is None or self._is_using_local:
            return self._extract_summary(report)

        prompt = SUMMARY_TEMPLATE.format(report_text=report)
        try:
            response = self._call_llm(prompt)
            return response.strip()
        except Exception:
            return self._extract_summary(report)

    def _build_prompt(self, results: dict) -> str:
        spoilage = results.get("spoilage", {})
        shelf = results.get("shelf_life", {})
        contamination = results.get("contamination_risks", {})
        food_cls = results.get("food_classification", results)
        defects = results.get("packaging_defects", [])
        ocr = results.get("ocr_data", {})

        defect_lines = "\n".join(
            f"  - {d['defect_type']} (confidence: {d.get('confidence', 0):.2f})"
            for d in (defects if isinstance(defects, list) else [])
        ) or "  None detected"

        detected_risks_list = []
        if isinstance(contamination, dict):
            for r in contamination.get("detected_risks", []):
                detected_risks_list.append(f"{r['category']} ({r.get('confidence', 0):.2f})")

        return REPORT_TEMPLATE.format(
            food_type=food_cls.get("food_type", "Unknown"),
            classification_confidence=food_cls.get("confidence_scores", [{"confidence": 0}])[0]["confidence"],
            freshness_score=spoilage.get("freshness_score", results.get("freshness_score", 0)),
            is_spoiled=spoilage.get("is_spoiled", False),
            spoilage_severity=spoilage.get("severity_label", "unknown"),
            packaging_defects=defect_lines,
            contamination_risk=contamination.get("overall_risk_level", "unknown") if isinstance(contamination, dict) else "unknown",
            detected_risks=", ".join(detected_risks_list) or "None",
            shelf_life_days=shelf.get("estimated_days_remaining", results.get("shelf_life_days", "N/A")) if isinstance(shelf, dict) else "N/A",
            freshness_category=shelf.get("freshness_category", "unknown") if isinstance(shelf, dict) else "unknown",
            ocr_data=json.dumps(ocr.get("parsed", ocr), indent=2) if isinstance(ocr, dict) else str(ocr),
        )

    def _call_llm(self, prompt: str) -> str:
        if self.model_name.startswith("gpt"):
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        else:
            response = self._client.text_generation(
                prompt,
                model=self.model_name.replace("hf/", ""),
                temperature=self.temperature,
                max_new_tokens=2000,
            )
            return response

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
        if freshness < 0.5 and freshness > 0.3:
            recommendations.append("Consume within 24 hours")
        recommendations.append("Store at recommended temperature")

        shelf_days = shelf.get("estimated_days_remaining", results.get("shelf_life_days", "N/A")) if isinstance(shelf, dict) else "N/A"

        return {
            "report_title": f"Food Inspection Report — {food_type}",
            "executive_summary": (
                f"Inspection of {food_type} completed. "
                f"Freshness score: {freshness:.2f}/1.0. "
                f"Estimated shelf life: {shelf_days} days. "
                f"Overall verdict: {verdict}."
            ),
            "detailed_findings": [
                {"area": "Food Classification", "status": "info", "detail": f"Identified as {food_type}"},
                {"area": "Freshness", "status": "fail" if is_spoiled else "pass", "detail": f"Score {freshness:.2f}/1.0"},
                {"area": "Packaging", "status": "fail" if defects else "pass", "detail": f"{len(defects) if isinstance(defects, list) else 0} defect(s)"},
                {"area": "Contamination Risk", "status": contamination.get("overall_risk_level", "pass") if isinstance(contamination, dict) else "pass", "detail": contamination.get("overall_risk_level", "none") if isinstance(contamination, dict) else "none"},
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
