"""
LLM Provider Tests — Spec §47

Tests for:
- LLMReportGenerator: template fallback, exception handling, Pydantic validation
- InspectionReport schema: valid/invalid verdicts
- EvidenceValidator: forbidden claim detection
- CircuitBreaker: state transitions
- ResilientLLMProvider: retry and circuit breaker integration
- Provider names: OllamaProvider, VLLMProvider
- PromptRegistry: version resolution, KeyError behaviour
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# LLMReportGenerator tests
# ---------------------------------------------------------------------------

class TestLLMReportGenerator:

    def _minimal_results(self) -> dict:
        return {
            "food_type": "Tomato",
            "freshness_score": 0.7,
            "spoilage": {
                "is_spoiled": False,
                "freshness_score": 0.7,
                "spoilage_score": 0.3,
                "severity_label": "low",
            },
            "packaging_defects": [],
            "contamination_risks": {"overall_risk_level": "low", "detected_risks": []},
            "shelf_life": {
                "estimated_days_remaining": 5,
                "freshness_category": "moderate",
                "confidence": 0.8,
            },
            "food_classification": {
                "food_type": "Tomato",
                "confidence_scores": [{"confidence": 0.9}],
            },
            "ocr_data": {},
        }

    def test_returns_template_when_provider_is_none(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        gen = LLMReportGenerator(provider=None)
        report = gen.generate_report(self._minimal_results())
        assert "overall_verdict" in report
        assert "report_title" in report
        assert "executive_summary" in report

    def test_returns_template_on_llm_exception(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        from aiengine.llm.provider import ChatMessage, GenerationOptions, LLMResponse

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("LLM unavailable")
        mock_provider.provider_name = "mock"

        gen = LLMReportGenerator(provider=mock_provider)
        report = gen.generate_report(self._minimal_results())
        # Must not re-raise; must return a valid template report
        assert "overall_verdict" in report
        assert "report_title" in report

    def test_llm_exception_does_not_reraise(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("boom")
        mock_provider.provider_name = "mock"
        gen = LLMReportGenerator(provider=mock_provider)
        # Should not raise
        try:
            gen.generate_report(self._minimal_results())
        except Exception as exc:
            pytest.fail(f"generate_report raised an unexpected exception: {exc}")

    def test_parse_response_handles_malformed_json(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        gen = LLMReportGenerator(provider=None)
        result = gen._parse_response("this is not json {{{{", {})
        assert result is None

    def test_parse_response_handles_missing_required_fields(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        import json
        gen = LLMReportGenerator(provider=None)
        # Missing all required fields except report_title
        partial = json.dumps({"report_title": "Only Title"})
        result = gen._parse_response(partial, {})
        assert result is None

    def test_parse_response_accepts_valid_report(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        import json
        gen = LLMReportGenerator(provider=None)
        valid = {
            "report_title": "Test Report",
            "executive_summary": "All good.",
            "detailed_findings": [
                {"area": "Freshness", "status": "pass", "detail": "Fresh"}
            ],
            "risk_flags": [],
            "recommendations": ["Store in fridge"],
            "overall_verdict": "pass",
            "inspection_date": datetime.now().isoformat(),
        }
        result = gen._parse_response(json.dumps(valid), {})
        assert result is not None
        assert result["overall_verdict"] == "pass"

    def test_provider_generate_is_called_with_messages(self) -> None:
        from aiengine.llm_report_generator import LLMReportGenerator
        from aiengine.llm.provider import LLMResponse
        import json

        valid_report = {
            "report_title": "Test",
            "executive_summary": "Ok.",
            "detailed_findings": [{"area": "Freshness", "status": "pass", "detail": "Good"}],
            "risk_flags": [],
            "recommendations": ["Store cool"],
            "overall_verdict": "pass",
            "inspection_date": datetime.now().isoformat(),
        }
        mock_provider = MagicMock()
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(valid_report),
            model="test-model",
            provider="mock",
        )
        mock_provider.provider_name = "mock"

        gen = LLMReportGenerator(provider=mock_provider)
        gen.generate_report(self._minimal_results())

        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args
        messages = call_args[0][0]
        assert isinstance(messages, list)
        assert len(messages) >= 1


# ---------------------------------------------------------------------------
# InspectionReport schema tests
# ---------------------------------------------------------------------------

class TestInspectionReportSchema:

    def _valid_report_data(self, verdict: str = "pass") -> dict:
        return {
            "report_title": "Test Report",
            "executive_summary": "Everything looks fine.",
            "detailed_findings": [
                {"area": "Freshness", "status": "pass", "detail": "Fresh"}
            ],
            "risk_flags": [],
            "recommendations": ["Store in fridge"],
            "overall_verdict": verdict,
            "inspection_date": datetime.now().isoformat(),
        }

    def test_rejects_invalid_verdict(self) -> None:
        from aiengine.llm.schemas import InspectionReport
        with pytest.raises(ValidationError):
            InspectionReport(**self._valid_report_data(verdict="unknown_verdict"))

    def test_rejects_empty_string_verdict(self) -> None:
        from aiengine.llm.schemas import InspectionReport
        with pytest.raises(ValidationError):
            InspectionReport(**self._valid_report_data(verdict=""))

    @pytest.mark.parametrize("verdict", ["pass", "conditional_pass", "fail"])
    def test_accepts_valid_verdicts(self, verdict: str) -> None:
        from aiengine.llm.schemas import InspectionReport
        report = InspectionReport(**self._valid_report_data(verdict=verdict))
        assert report.overall_verdict == verdict

    def test_finding_status_rejects_invalid(self) -> None:
        from aiengine.llm.schemas import InspectionReport
        data = self._valid_report_data()
        data["detailed_findings"] = [
            {"area": "Freshness", "status": "unknown_status", "detail": "Bad"}
        ]
        with pytest.raises(ValidationError):
            InspectionReport(**data)

    def test_traceability_fields_are_optional(self) -> None:
        from aiengine.llm.schemas import InspectionReport
        report = InspectionReport(**self._valid_report_data())
        assert report.prompt_version is None
        assert report.llm_provider is None
        assert report.llm_model is None
        assert report.evidence_summary is None

    def test_traceability_fields_can_be_set(self) -> None:
        from aiengine.llm.schemas import InspectionReport
        data = self._valid_report_data()
        data["prompt_version"] = "2.0"
        data["llm_provider"] = "ollama"
        data["llm_model"] = "llama3.1:8b"
        data["evidence_summary"] = "Driven by spoilage score."
        report = InspectionReport(**data)
        assert report.prompt_version == "2.0"
        assert report.llm_provider == "ollama"


# ---------------------------------------------------------------------------
# EvidenceValidator tests
# ---------------------------------------------------------------------------

class TestEvidenceValidator:

    def test_flags_forbidden_pathogen_claim(self) -> None:
        from aiengine.llm.validator import EvidenceValidator
        validator = EvidenceValidator()
        report = {"executive_summary": "Salmonella was detected in this sample."}
        evidence = {"spoilage": {"is_spoiled": True}}
        is_valid, violations = validator.validate(report, evidence)
        assert not is_valid
        assert len(violations) > 0

    def test_flags_listeria_claim(self) -> None:
        from aiengine.llm.validator import EvidenceValidator
        validator = EvidenceValidator()
        report = {"executive_summary": "Listeria contamination found."}
        evidence = {"spoilage": {"is_spoiled": False}}
        is_valid, violations = validator.validate(report, evidence)
        assert not is_valid

    def test_accepts_safe_claim(self) -> None:
        from aiengine.llm.validator import EvidenceValidator
        validator = EvidenceValidator()
        report = {
            "executive_summary": (
                "The model predicts elevated spoilage risk based on visual evidence."
            )
        }
        evidence = {"spoilage": {"is_spoiled": True}}
        is_valid, violations = validator.validate(report, evidence)
        assert is_valid
        assert violations == []

    def test_claim_supported_by_evidence_passes(self) -> None:
        from aiengine.llm.validator import EvidenceValidator
        validator = EvidenceValidator()
        # Claim is "salmonella" and evidence also mentions it → supported
        report = {"summary": "salmonella risk flagged"}
        evidence = {"lab_result": "salmonella detected in batch 42"}
        is_valid, _ = validator.validate(report, evidence)
        assert is_valid

    def test_custom_forbidden_claims(self) -> None:
        from aiengine.llm.validator import EvidenceValidator
        validator = EvidenceValidator(forbidden_claims=["radioactive"])
        report = {"summary": "The food was radioactive."}
        evidence = {}
        is_valid, violations = validator.validate(report, evidence)
        assert not is_valid
        assert any("radioactive" in v for v in violations)


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_initial_state_is_closed(self) -> None:
        from aiengine.llm.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == "CLOSED"
        assert cb.can_execute() is True

    def test_transitions_to_open_after_failure_threshold(self) -> None:
        from aiengine.llm.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.record_failure()
            assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_open_state_can_execute_returns_false(self) -> None:
        from aiengine.llm.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False

    def test_record_success_resets_to_closed(self) -> None:
        from aiengine.llm.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
        cb.record_failure()
        assert cb.state == "OPEN"
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failures == 0

    def test_failure_count_increments(self) -> None:
        from aiengine.llm.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=10)
        for i in range(5):
            cb.record_failure()
        assert cb.failures == 5


# ---------------------------------------------------------------------------
# ResilientLLMProvider tests
# ---------------------------------------------------------------------------

class TestResilientLLMProvider:

    def _make_mock_provider(self, name: str = "mock") -> MagicMock:
        p = MagicMock()
        p.provider_name = name
        return p

    def test_retries_on_failure_then_succeeds(self) -> None:
        from aiengine.llm.resilience import ResilientLLMProvider
        from aiengine.llm.provider import ChatMessage, GenerationOptions, LLMResponse

        mock_provider = self._make_mock_provider()
        success_response = LLMResponse(
            content="ok", model="m", provider="mock"
        )
        # Fail twice, succeed on third
        mock_provider.generate.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            success_response,
        ]

        resilient = ResilientLLMProvider(mock_provider, max_retries=3, backoff_factor=0.0)
        result = resilient.generate(
            [{"role": "user", "content": "hello"}],
            GenerationOptions(),
        )
        assert result.content == "ok"
        assert mock_provider.generate.call_count == 3

    def test_raises_after_all_retries_exhausted(self) -> None:
        from aiengine.llm.resilience import ResilientLLMProvider
        from aiengine.llm.provider import GenerationOptions

        mock_provider = self._make_mock_provider()
        mock_provider.generate.side_effect = RuntimeError("persistent failure")

        resilient = ResilientLLMProvider(mock_provider, max_retries=2, backoff_factor=0.0)
        with pytest.raises(RuntimeError, match="persistent failure"):
            resilient.generate(
                [{"role": "user", "content": "hello"}],
                GenerationOptions(),
            )

    def test_circuit_breaker_opens_after_failures(self) -> None:
        from aiengine.llm.resilience import ResilientLLMProvider, CircuitBreakerOpenError
        from aiengine.llm.provider import GenerationOptions

        mock_provider = self._make_mock_provider()
        mock_provider.generate.side_effect = RuntimeError("fail")

        resilient = ResilientLLMProvider(
            mock_provider,
            max_retries=1,
            backoff_factor=0.0,
        )
        resilient.circuit_breaker.failure_threshold = 1

        # First call — fails and trips circuit breaker
        with pytest.raises(Exception):
            resilient.generate(
                [{"role": "user", "content": "hello"}],
                GenerationOptions(),
            )

        # Circuit breaker should now be open
        assert resilient.circuit_breaker.state == "OPEN"

        # Second call — circuit breaker is open
        with pytest.raises(CircuitBreakerOpenError):
            resilient.generate(
                [{"role": "user", "content": "hello"}],
                GenerationOptions(),
            )


# ---------------------------------------------------------------------------
# Provider name tests
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_provider_name(self) -> None:
        from aiengine.llm.ollama_provider import OllamaProvider
        # Patch _init_client so no network call is made
        with patch.object(OllamaProvider, "_init_client", return_value=None):
            p = OllamaProvider.__new__(OllamaProvider)
            p.model = "llama3.1:8b"
            p.base_url = "http://localhost:11434/v1"
            p.timeout = 120
            p._client = None
        assert p.provider_name == "ollama"


class TestVLLMProvider:
    def test_provider_name(self) -> None:
        from aiengine.llm.vllm_provider import VLLMProvider
        with patch.object(VLLMProvider, "_init_client", return_value=None):
            p = VLLMProvider.__new__(VLLMProvider)
            p.model = "facebook/opt-125m"
            p.base_url = "http://vllm:8000/v1"
            p.timeout = 120
            p._client = None
        assert p.provider_name == "vllm"


# ---------------------------------------------------------------------------
# PromptRegistry tests
# ---------------------------------------------------------------------------

class TestPromptRegistry:

    def _fresh_registry(self):
        from aiengine.llm.prompts import PromptRegistry
        return PromptRegistry()

    def test_get_returns_latest_version_by_default(self) -> None:
        reg = self._fresh_registry()
        reg.register("test_prompt", "1.0", "template v1", "model-a", "ollama")
        reg.register("test_prompt", "1.5", "template v1.5", "model-b", "ollama")
        reg.register("test_prompt", "2.0", "template v2", "model-c", "ollama")
        pv = reg.get("test_prompt")
        assert pv.version == "2.0"

    def test_get_specific_version(self) -> None:
        reg = self._fresh_registry()
        reg.register("test_prompt", "1.0", "template v1", "model-a", "ollama")
        reg.register("test_prompt", "2.0", "template v2", "model-b", "ollama")
        pv = reg.get("test_prompt", version="1.0")
        assert pv.version == "1.0"
        assert pv.template == "template v1"

    def test_get_raises_key_error_for_unknown_prompt_id(self) -> None:
        reg = self._fresh_registry()
        with pytest.raises(KeyError):
            reg.get("nonexistent_prompt")

    def test_get_raises_key_error_for_unknown_version(self) -> None:
        reg = self._fresh_registry()
        reg.register("test_prompt", "1.0", "template", "model", "ollama")
        with pytest.raises(KeyError):
            reg.get("test_prompt", version="9.9")

    def test_global_registry_has_inspection_report_prompt(self) -> None:
        from aiengine.llm.prompts import registry
        pv = registry.get("inspection_report")
        assert pv is not None
        assert pv.version is not None

    def test_global_registry_has_report_summary_prompt(self) -> None:
        from aiengine.llm.prompts import registry
        pv = registry.get("report_summary")
        assert pv is not None

    def test_global_registry_latest_inspection_report_is_v2(self) -> None:
        """After adding v2.0, the default must be the grounding-safe version."""
        from aiengine.llm.prompts import registry
        pv = registry.get("inspection_report")
        assert pv.version == "2.0"

    def test_prompt_version_metadata(self) -> None:
        reg = self._fresh_registry()
        reg.register("my_prompt", "1.0", "Hello {name}", "llama3", "ollama")
        pv = reg.get("my_prompt")
        assert pv.version == "1.0"
        assert pv.template == "Hello {name}"
        assert pv.model_recommendation == "llama3"
        assert pv.provider_recommendation == "ollama"
