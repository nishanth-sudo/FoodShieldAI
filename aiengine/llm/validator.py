from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

class EvidenceValidator:
    """
    Validates that the LLM-generated report is grounded in the provided evidence.
    Prevents hallucinations (e.g., claiming pathogens were detected without evidence).
    """
    def __init__(self, forbidden_claims: list[str] | None = None) -> None:
        # Claims that should NEVER appear unless explicitly in the evidence
        self.forbidden_claims = forbidden_claims or [
            "salmonella", "listeria", "e. coli", "pathogen detected",
            "laboratory test", "microbiological analysis"
        ]

    def validate(self, report: dict, evidence: dict) -> tuple[bool, list[str]]:
        """
        Validates the report against the evidence.
        Returns (is_valid, list_of_violations).
        """
        violations = []

        # 1. Check for forbidden claims in the report text
        report_text = str(report).lower()
        for claim in self.forbidden_claims:
            if claim in report_text:
                # Check if the claim is actually supported by evidence
                # In a real system, we would check if the evidence contains these keywords
                if not self._is_supported_by_evidence(claim, evidence):
                    violations.append(f"Unsupported claim detected: '{claim}'")

        # 2. Check if the verdict matches the evidence
        # If spoilage is False, but verdict is 'fail' based on spoilage, that's a violation
        spoilage = evidence.get("spoilage", {})
        is_spoiled = spoilage.get("is_spoiled", False)
        verdict = report.get("overall_verdict", "pass")

        if not is_spoiled and verdict == "fail":
            # This is a potential hallucination if no other risk factors exist
            # We'll just log it as a warning for now
            logger.warning("LLM marked report as 'fail' despite no spoilage evidence.")

        return len(violations) == 0, violations

    def _is_supported_by_evidence(self, claim: str, evidence: dict) -> bool:
        """Checks if a specific claim is present in the provided evidence."""
        evidence_str = str(evidence).lower()
        return claim in evidence_str
