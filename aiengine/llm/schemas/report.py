from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class XAIEvidenceSchema(BaseModel):
    shap_features: List[str] = Field(default_factory=list, description="Top features contributing to the prediction based on SHAP.")
    gradcam_regions: List[str] = Field(default_factory=list, description="Descriptions of regions highlighted by Grad-CAM.")

class CounterfactualSchema(BaseModel):
    original_prediction: str = Field(..., description="The original class predicted by the model.")
    target_prediction: str = Field(..., description="The alternative class for the counterfactual.")
    required_changes: List[str] = Field(..., description="Minimal changes required to flip the prediction.")

class RiskAssessmentSchema(BaseModel):
    level: str = Field(..., description="Risk level: Low, Medium, High, or Critical.")
    factors: List[str] = Field(default_factory=list, description="Factors contributing to this risk level.")

class InspectionReportSchema(BaseModel):
    summary: str = Field(..., description="A brief overview of the findings.")
    findings: List[str] = Field(default_factory=list, description="Detailed list of observations.")
    xai_evidence: Optional[XAIEvidenceSchema] = Field(None, description="Explainable AI evidence supporting the findings.")
    counterfactuals: Optional[List[CounterfactualSchema]] = Field(None, description="Counterfactual scenarios for the predictions.")
    risk_assessment: RiskAssessmentSchema = Field(..., description="Overall risk assessment and contributing factors.")
