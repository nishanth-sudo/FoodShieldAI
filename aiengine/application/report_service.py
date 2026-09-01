from __future__ import annotations

import logging

from aiengine.domain.exceptions import LLMResponseError
from aiengine.llm_report_generator import LLMReportGenerator

logger = logging.getLogger(__name__)


class ReportService:
    """
    Application service that wraps
    :class:`~aiengine.llm_report_generator.LLMReportGenerator`.

    Responsibilities:
    - Attach XAI evidence and grounding metadata to the CV results before
      passing them to the generator so that the LLM prompt is grounded in
      actual model outputs (Spec §12).
    - Fall back to the deterministic template report when the LLM is
      unavailable or returns an invalid response (Spec §10).
    - Raise :class:`~aiengine.domain.exceptions.LLMResponseError` only on
      critical, unrecoverable failures — the generator handles most failures
      gracefully via its own fallback.

    **The LLM must NOT become the authoritative food-safety classifier.**
    The deterministic/model layer remains authoritative for the actual
    prediction; the LLM is used solely for narrating the report (Spec §35).
    """

    def __init__(self, report_generator: LLMReportGenerator) -> None:
        self._generator = report_generator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_inspection_report(
        self,
        cv_results: dict,
        xai_evidence: dict | None = None,
    ) -> dict:
        """
        Generate a structured inspection report from CV model results.

        Attaches ``xai_evidence`` under the ``"xai"`` key before sending
        results to the generator so that the LLM prompt is grounded in
        actual model outputs.

        Args:
            cv_results:   Aggregated CV pipeline results dict.
            xai_evidence: Optional dict containing ``gradcam``, ``shap``,
                          ``counterfactuals``, etc. from ExplanationService.

        Returns:
            A structured report dict as produced by LLMReportGenerator.

        Raises:
            LLMResponseError: If report generation raises an unexpected
                              exception that cannot be handled gracefully.
        """
        enriched = dict(cv_results)
        if xai_evidence:
            enriched["xai"] = xai_evidence

        try:
            return self._generator.generate_report(enriched)
        except LLMResponseError:
            raise
        except Exception as exc:
            logger.error("Unexpected error in report generation: %s", exc)
            # Fall back to template — do not propagate unexpected bare exceptions.
            try:
                return self._generator._template_report(enriched)
            except Exception as template_exc:
                raise LLMResponseError(
                    f"Report generation and template fallback both failed: {template_exc}"
                ) from template_exc

    def generate_summary(self, report_text: str) -> str:
        """
        Summarise a report in 2–3 sentences.

        Delegates to
        :meth:`~aiengine.llm_report_generator.LLMReportGenerator.generate_summary`.
        Falls back to the first 200 characters of ``report_text`` when the
        LLM is unavailable.

        Args:
            report_text: Raw report text to summarise.

        Returns:
            A short summary string.

        Raises:
            LLMResponseError: On unexpected failure.
        """
        try:
            return self._generator.generate_summary(report_text)
        except Exception as exc:
            logger.error("Report summary generation failed: %s", exc)
            raise LLMResponseError(
                f"Summary generation failed: {exc}"
            ) from exc
