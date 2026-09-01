from __future__ import annotations

from typing import Literal, List, Optional
from pydantic import BaseModel, Field

class Finding(BaseModel):
    """A single finding within the inspection report."""
    area: str = Field(..., description="The area being inspected (e.g., Packaging, Freshness)")
    status: Literal["pass", "fail", "info", "warning"] = Field(..., description="The status of the finding")
    detail: str = Field(..., description="Detailed observation")

class InspectionReport(BaseModel):
    """The complete structured inspection report."""
    report_title: str = Field(..., description="Short descriptive title of the report")
    executive_summary: str = Field(..., description="2-3 sentence overview of the findings")
    detailed_findings: List[Finding] = Field(..., description="List of detailed area-specific findings")
    risk_flags: List[str] = Field(..., description="List of critical risk warnings")
    recommendations: List[str] = Field(..., description="Actionable steps for the user")
    overall_verdict: Literal["pass", "conditional_pass", "fail"] = Field(..., description="Final safety verdict")
    inspection_date: str = Field(..., description="ISO 8601 date of the inspection")
    # Traceability fields — populated by the report generator for audit purposes
    evidence_summary: Optional[str] = Field(
        default=None,
        description="Summary of the input evidence used to generate this report",
    )
    prompt_version: Optional[str] = Field(
        default=None,
        description="Version of the prompt template used (e.g., '2.0')",
    )
    llm_provider: Optional[str] = Field(
        default=None,
        description="Name of the LLM provider that generated this report (e.g., 'ollama')",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Name of the LLM model used (e.g., 'llama3.1:8b')",
    )
