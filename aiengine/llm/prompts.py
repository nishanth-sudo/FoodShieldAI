from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class PromptVersion:
    """Metadata for a specific version of a prompt."""
    version: str
    template: str
    model_recommendation: str
    provider_recommendation: str

class PromptRegistry:
    """
    Registry for managing prompt versions across the application.
    Prevents hardcoding prompts in business logic.
    """
    def __init__(self) -> None:
        self._prompts: Dict[str, Dict[str, PromptVersion]] = {}

    def register(self, prompt_id: str, version: str, template: str, model: str, provider: str) -> None:
        if prompt_id not in self._prompts:
            self._prompts[prompt_id] = {}
        self._prompts[prompt_id][version] = PromptVersion(
            version=version,
            template=template,
            model_recommendation=model,
            provider_recommendation=provider
        )

    def get(self, prompt_id: str, version: str | None = None) -> PromptVersion:
        if prompt_id not in self._prompts:
            raise KeyError(f"Prompt ID '{prompt_id}' not found in registry.")

        if version:
            if version not in self._prompts[prompt_id]:
                raise KeyError(f"Version '{version}' for prompt '{prompt_id}' not found.")
            return self._prompts[prompt_id][version]

        # Default to the latest version (highest alphanumeric)
        latest_version = sorted(self._prompts[prompt_id].keys())[-1]
        return self._prompts[prompt_id][latest_version]

# Global registry instance
registry = PromptRegistry()

# ---------------------------------------------------------------------------
# inspection_report — v1.0 (legacy, kept for backwards compatibility)
# ---------------------------------------------------------------------------
registry.register(
    prompt_id="inspection_report",
    version="1.0",
    template="""You are a food safety inspection expert. Generate a comprehensive inspection report in JSON format based on the following AI analysis results.

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
""",
    model="llama3.1:8b",
    provider="ollama"
)

# ---------------------------------------------------------------------------
# inspection_report — v2.0 (grounding-safe, per spec §12)
# This is the default version returned by registry.get("inspection_report").
# ---------------------------------------------------------------------------
registry.register(
    prompt_id="inspection_report",
    version="2.0",
    template="""You are a food safety inspection assistant. Your role is to synthesise the AI model outputs supplied below into a structured report. You are NOT the primary classifier — the risk levels and scores below are authoritative deterministic outputs from trained ML models.

## GROUNDING CONSTRAINTS (mandatory)
- IMPORTANT: Use ONLY the supplied evidence below. Do not add information from outside this context.
- Do NOT invent laboratory test results, pathogens, or microbiological findings.
- Do NOT override the deterministic risk classification provided by the ML models.
- Clearly distinguish between model predictions and verified facts in your report.
- Avoid unsupported medical or food-safety claims not evidenced by the data below.
- If a pathogen name appears anywhere in your output, you MUST prefix it with: "not detected / not tested — model predicts elevated risk only".

## Inspection Evidence (authoritative ML model outputs)

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
- "executive_summary": 2-3 sentence overview that distinguishes model predictions from verified facts
- "detailed_findings": array of finding objects with "area", "status", "detail"
- "risk_flags": array of any high-risk items found (prefix any pathogen names as instructed above)
- "recommendations": array of actionable recommendation strings grounded only in the evidence above
- "overall_verdict": "pass", "conditional_pass", or "fail" — must agree with the ML contamination_risk level provided
- "inspection_date": current date
- "evidence_summary": one sentence summarising which evidence fields drove the verdict
""",
    model="llama3.1:8b",
    provider="ollama"
)

# ---------------------------------------------------------------------------
# report_summary — v1.0
# ---------------------------------------------------------------------------
registry.register(
    prompt_id="report_summary",
    version="1.0",
    template="""Summarize the following food inspection report in 2-3 sentences focusing on the key findings and verdict:

{report_text}

Summary:""",
    model="llama3.2:3b",
    provider="ollama"
)
