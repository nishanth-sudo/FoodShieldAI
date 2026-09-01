# FoodShieldAI — Comprehensive Engineering Remediation & Implementation Specification

## Purpose

This document consolidates the complete technical review of the FoodShieldAI repository and converts the findings into an actionable implementation specification for CLI coding agents.

The objective is to evolve FoodShieldAI from a promising AI application/prototype into a **technically rigorous, explainable, production-oriented ML/AI platform** with:

- Correct XAI implementations
- Genuine vLLM-based LLM serving
- Strong OOP and SOLID design
- Clean/Hexagonal Architecture
- Strict type safety
- Robust exception handling
- Structured domain models
- LLM safety and grounding
- Model/data/prompt versioning
- ML evaluation and regression testing
- Observability and MLOps
- Secure API and file handling
- Production-oriented deployment behavior

The implementation should prioritize **correctness and maintainability over adding technologies for appearance**.

---

# 1. Current Repository Assessment

## Overall assessment

| Area | Current assessment |
|---|---:|
| Architecture | 8/10 |
| OOP | 7/10 |
| SOLID | 6.5/10 |
| Type safety | 6/10 |
| Exception handling | 5.5/10 |
| XAI architecture | 8/10 |
| XAI algorithmic correctness | 5.5/10 |
| Counterfactual XAI | 6/10 |
| LLM integration | 7.5/10 |
| Actual vLLM implementation | 1/10 |
| Graceful degradation | 8.5/10 |
| Test architecture | 7.5/10 |
| Production readiness | ~6/10 |

These ratings are directional engineering assessments, not formal benchmark results.

## Current strengths

The repository already has a strong high-level structure with separate areas for:

- `aiengine`
- `backend`
- `frontend`
- `mlops`
- `infrastructure`
- `tests`

The project also attempts to enforce:

- Clean/Hexagonal Architecture
- SOLID principles
- Ruff
- strict mypy
- pytest
- coverage
- Docker
- CI/CD
- observability
- XAI
- LLM integration

The goal is therefore **not to rewrite the entire project**, but to correct implementation gaps and strengthen the existing architecture.

---

# 2. Critical Finding: XAI Must Be Algorithmically Correct

## 2.1 Grad-CAM

### Current state

The repository contains a real `GradCAMExplainer` implementation using:

- forward hooks
- backward hooks
- activation capture
- gradient capture
- gradient pooling
- weighted activations
- ReLU
- heatmap normalization
- ROI extraction

This is a valid Grad-CAM-style implementation.

### Required changes

There is duplicate Grad-CAM logic in the broader XAI façade.

Do not maintain two separate implementations.

Target architecture:

```text
XAIExplainer
     |
     +-- GradCAMExplainer
     +-- SHAPExplainer
     +-- LIMEExplainer
     +-- CounterfactualExplainer
```

The façade should delegate to dedicated implementations.

### Requirements

- Keep one canonical Grad-CAM implementation.
- Add tests for:
  - output shape
  - normalized range
  - target-layer selection
  - invalid model handling
  - missing gradients
  - CPU execution
  - GPU execution where available
- Ensure hooks are removed after execution.
- Avoid retaining tensors/graphs unnecessarily.
- Use `torch.no_grad()` where gradients are not required.
- Ensure model evaluation mode is handled safely.
- Do not permanently alter the model's training/evaluation state.

---

# 3. Critical Finding: LIME Is Currently Not Actually LIME

## Current problem

The current XAI implementation exposes a LIME method but delegates to Grad-CAM.

This means:

```text
method = "lime"
        |
        v
Grad-CAM
```

That is technically incorrect.

## Required solution

Choose one of two approaches:

### Option A — Implement genuine LIME

Use an established LIME implementation or implement image LIME correctly.

For image classification:

1. Segment the image into interpretable superpixels.
2. Generate perturbed samples.
3. Run the actual prediction model for each perturbed sample.
4. Fit a local surrogate model.
5. Calculate feature/superpixel importance.
6. Produce an explanation mask.
7. Validate the explanation against the model.

### Option B — Remove LIME from the public API

If genuine LIME is not required, remove the `"lime"` option rather than mislabeling Grad-CAM as LIME.

Preferred approach for this project:

**Implement genuine LIME only if there is a clear use case. Otherwise expose Grad-CAM + genuine SHAP and avoid unnecessary algorithm count.**

---

# 4. Critical Finding: Current SHAP Implementation Is Not Genuine SHAP

## Current problem

The current `SHAPExplainer` checks whether the `shap` package exists but the actual explanation logic uses feature perturbation/ablation.

It is closer to:

```text
Feature Ablation / Perturbation Attribution
```

than true SHAP.

It should not be represented as exact SHAP values.

## Required solution

Implement genuine SHAP where appropriate.

### For tabular models

Use the appropriate SHAP explainer:

- `TreeExplainer` for tree-based models
- `LinearExplainer` for linear models
- `KernelExplainer` only when model-specific explainers are inappropriate
- `shap.Explainer` as the high-level API when appropriate

Example architecture:

```text
Trained Model
     |
     v
SHAP Explainer
     |
     +-- base value
     +-- feature contributions
     +-- prediction reconstruction
     +-- global importance
     +-- local explanation
```

### Important correctness requirement

Verify the SHAP additivity relationship where applicable:

```text
base_value + sum(shap_values) ≈ model_prediction
```

with a documented numerical tolerance.

### For image models

Use an appropriate image-capable SHAP approach if image SHAP is genuinely needed.

Do not use a handcrafted approximation and call it SHAP.

---

# 5. Critical Finding: XAI Must Explain the Actual Prediction

## Current problem

The orchestration layer currently constructs a handcrafted risk function using environmental variables such as:

- temperature
- humidity
- storage duration
- spoilage score

and the XAI attribution can effectively explain this manually defined equation rather than the actual trained predictive model.

This creates a serious explainability integrity problem.

## Required architecture

The explanation must target the exact model responsible for the decision.

Correct:

```text
Input
  |
  v
Actual trained model
  |
  +------> Prediction
  |
  +------> XAI explainer
             |
             v
       Model attribution
```

Incorrect:

```text
Input
  |
  v
Actual model
  |
  v
Prediction

Separate handcrafted equation
  |
  v
"SHAP explanation"
```

## Required implementation

Create explicit model interfaces.

Example:

```python
class RiskModel(Protocol):
    def predict(self, features: RiskFeatures) -> RiskPrediction:
        ...

    def predict_proba(self, features: RiskFeatures) -> Probability:
        ...
```

The XAI service must receive the actual model or a callable that invokes the exact production prediction function.

---

# 6. Counterfactual XAI

## Current state

The counterfactual implementation searches feature changes to reduce predicted risk.

This is useful, but currently depends on the same heuristic prediction function.

## Required changes

Counterfactual explanations must operate against the actual trained model.

Implement:

```text
Actual model
    |
    v
Counterfactual generator
    |
    +-- immutable features
    +-- mutable features
    +-- feature constraints
    +-- feasibility rules
    +-- objective function
    |
    v
Minimal actionable changes
```

Each counterfactual must contain:

- original values
- changed values
- changed features
- prediction before
- prediction after
- distance/cost
- constraint violations, if any
- feasibility status

Example:

```json
{
  "feature": "temperature",
  "original": 31.0,
  "counterfactual": 24.0,
  "prediction_before": 0.89,
  "prediction_after": 0.34,
  "cost": 7.0,
  "feasible": true
}
```

Do not generate physically impossible recommendations.

---

# 7. vLLM — Current State and Required Implementation

## Current state

The repository currently has LLM integration primarily through an Ollama-based client and OpenAI-compatible interfaces.

Ollama is not vLLM.

Therefore the project currently has:

**LLM integration: yes**

**Actual vLLM integration: no**

## Required architecture

Introduce a provider abstraction:

```text
                    LLMProvider
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
   OllamaProvider   VLLMProvider   OpenAIProvider
```

The report-generation service must depend on `LLMProvider`, not on Ollama/vLLM directly.

---

# 8. LLM Provider Interface

Create a provider contract.

Example:

```python
class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions,
    ) -> LLMResponse:
        ...
```

Define domain types:

```python
@dataclass(frozen=True)
class GenerationOptions:
    temperature: float
    max_tokens: int
    timeout_seconds: float
    response_format: str | None
```

```python
@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    request_id: str
```

---

# 9. vLLM Provider

Implement a dedicated provider that communicates with a vLLM OpenAI-compatible endpoint.

Architecture:

```text
FoodShieldAI
     |
     v
LLMReportGenerator
     |
     v
LLMProvider
     |
     v
VLLMProvider
     |
     v
OpenAI-compatible vLLM endpoint
     |
     v
Open-source LLM
```

The provider should be configurable through environment variables.

Example conceptual configuration:

```text
LLM_PROVIDER=vllm
LLM_BASE_URL=http://vllm:8000/v1
LLM_MODEL=<configured-model>
```

Do not hardcode model names or URLs.

---

# 10. vLLM Deployment

Provide a development and production-compatible vLLM deployment.

Requirements:

- Docker support
- health endpoint
- configurable model
- configurable tensor parallelism
- configurable GPU memory utilization
- request timeout
- logging
- metrics
- readiness checking

The application should not assume vLLM is always available.

Fallback hierarchy:

```text
vLLM
  |
  +-- unavailable
        |
        v
      Ollama
        |
        +-- unavailable
              |
              v
       deterministic report
```

The deterministic fallback must remain safe and evidence-based.

---

# 11. Do Not Use vLLM for Core Vision Inference

vLLM should serve the language model.

Do not replace:

- PyTorch CV inference
- ONNX inference
- classification models
- spoilage models
- packaging models

with an LLM.

Correct:

```text
CV model -> prediction
XAI -> explanation
vLLM -> natural-language report
```

The LLM should not independently decide food safety.

---

# 12. Ground the LLM

The LLM should receive structured evidence generated by the deterministic/model layers.

Example:

```json
{
  "food_type": "tomato",
  "spoilage_probability": 0.89,
  "shelf_life_days": 2.1,
  "environment": {
    "temperature": 31.2,
    "humidity": 78.0,
    "storage_days": 4
  },
  "xai": {
    "top_factors": [
      "humidity",
      "temperature",
      "storage_duration"
    ]
  },
  "visual_evidence": [
    "discoloration",
    "surface lesion"
  ]
}
```

The prompt must explicitly state:

- use only supplied evidence
- do not invent laboratory test results
- do not invent pathogens
- do not override deterministic risk classification
- clearly distinguish prediction from verified fact
- avoid unsupported medical/food-safety claims

---

# 13. Structured LLM Output

Do not rely only on:

```text
"Return valid JSON."
```

Define Pydantic models.

Example:

```python
class Finding(BaseModel):
    category: str
    description: str
    evidence: list[str]
    confidence: float | None


class InspectionReport(BaseModel):
    report_title: str
    executive_summary: str
    detailed_findings: list[Finding]
    risk_flags: list[str]
    recommendations: list[str]
    overall_verdict: Literal[
        "pass",
        "conditional_pass",
        "fail"
    ]
```

Pipeline:

```text
LLM
 |
 v
Raw JSON
 |
 v
Pydantic validation
 |
 +-- valid --> report
 |
 +-- invalid --> retry / fallback
```

Never blindly trust parsed LLM output.

---

# 14. LLM Hallucination Protection

Implement evidence consistency validation.

Reject or flag statements that introduce unsupported claims.

For example, if the model never performed microbiological testing, the LLM must not claim:

```text
"Salmonella was detected."
```

The report should instead say:

```text
"The model predicts elevated spoilage risk based on the supplied visual and environmental evidence."
```

---

# 15. Prompt Versioning

Treat prompts as production artifacts.

Store:

- prompt ID
- prompt version
- model
- provider
- generation configuration
- timestamp

Example:

```text
inspection_report_prompt:v2.1
```

Include the prompt version in inspection metadata.

---

# 16. LLM Provider Dependency Inversion

`LLMReportGenerator` must not contain large provider-specific conditionals.

Avoid:

```python
if provider == "ollama":
    ...
elif provider == "openai":
    ...
elif provider == "vllm":
    ...
```

inside the business logic.

Use dependency injection:

```text
ReportService
    |
    +--> LLMProvider
              |
              +--> VLLMProvider
              +--> OllamaProvider
              +--> OpenAIProvider
```

This satisfies Dependency Inversion and improves testability.

---

# 17. OOP and SOLID Improvements

The repository already uses classes, but the goal is proper object-oriented design rather than simply class-based code.

## Single Responsibility Principle

The current `AIInferenceOrchestrator` is responsible for too many things:

- preprocessing
- vision inference
- OCR
- XAI
- risk calculations
- LLM report generation
- VLM logic
- concurrency

Split it into:

```text
InspectionService
|
+-- VisionPipeline
+-- OCRService
+-- RiskService
+-- ExplainabilityService
+-- ReportService
```

The orchestrator should coordinate rather than implement everything.

---

# 18. Interface Segregation

Prefer focused interfaces:

```python
class ImageClassifier(Protocol):
    def predict(...): ...


class Explainer(Protocol):
    def explain(...): ...


class OCRProvider(Protocol):
    def extract(...): ...


class LLMProvider(Protocol):
    def generate(...): ...
```

Do not create a giant interface containing unrelated operations.

---

# 19. Domain Models Instead of Generic Dictionaries

Avoid excessive:

```python
dict
list
Any
```

for domain data.

Introduce types such as:

```text
EnvironmentalData
FoodClassification
SpoilageAssessment
PackagingAssessment
ContaminationAssessment
ShelfLifePrediction
OCRResult
XAIExplanation
CounterfactualResult
InspectionResult
InspectionReport
```

Prefer:

```python
def inspect(
    image: Image.Image,
    environmental_data: EnvironmentalData | None,
) -> InspectionResult:
    ...
```

over:

```python
def inspect(image, data) -> dict:
    ...
```

---

# 20. Strict Configuration

Use a single settings object.

Example:

```python
class Settings(BaseSettings):
    app_name: str
    environment: Environment

    database_url: PostgresDsn | None
    redis_url: RedisDsn | None

    llm_provider: LLMProviderType
    llm_model: str
    llm_base_url: AnyHttpUrl

    model_path: Path
    confidence_threshold: float
```

Do not scatter `os.getenv()` throughout business logic.

---

# 21. Exception Hierarchy

Create domain-specific exceptions.

```text
FoodShieldError
|
+-- ModelError
|   +-- ModelLoadError
|   +-- ModelInferenceError
|
+-- XAIError
|   +-- ExplanationError
|   +-- UnsupportedExplainerError
|
+-- LLMError
|   +-- LLMConnectionError
|   +-- LLMTimeoutError
|   +-- LLMResponseError
|
+-- OCRException
|
+-- ValidationError
```

Map them appropriately at the API boundary.

Example:

```text
ValidationError       -> 422
LLMTimeoutError       -> 504
LLMConnectionError    -> 503
ModelInferenceError   -> 500
```

---

# 22. Remove Silent Exception Swallowing

Avoid:

```python
except Exception:
    pass
```

Use specific exceptions.

For expected failures:

```python
except LLMConnectionError as exc:
    logger.warning(
        "LLM unavailable",
        extra={"error": str(exc)}
    )
```

For unexpected failures:

```python
except Exception:
    logger.exception("Unexpected failure")
    raise
```

Do not silently hide failures.

---

# 23. Timeouts

Every external operation must have a bounded timeout.

Include:

- database
- Redis
- object storage
- OCR service
- Ollama
- vLLM
- OpenAI
- HTTP requests

No network call should be allowed to block indefinitely.

---

# 24. Retry Policy

Retries should be selective.

Retry:

- connection reset
- temporary 503
- transient network failures

Do not retry:

- invalid user input
- invalid model schema
- deterministic validation errors

Use bounded retries with exponential backoff.

---

# 25. Circuit Breaker

Implement circuit breaking for LLM providers.

Example:

```text
vLLM healthy
    |
    v
requests accepted

vLLM repeatedly fails
    |
    v
OPEN circuit
    |
    v
use fallback
```

This prevents cascading failures.

---

# 26. Request and Inspection IDs

Every request should have a correlation identifier.

Example:

```text
request_id
inspection_id
```

Include these in:

- API response
- application logs
- model logs
- XAI logs
- LLM logs
- metrics
- database records

This makes end-to-end debugging possible.

---

# 27. Model Versioning

Every prediction must identify the model that generated it.

Store:

```text
model_name
model_version
model_hash
model_framework
model_artifact_uri
```

Example:

```json
{
  "model": "spoilage-detector",
  "version": "1.4.2",
  "hash": "abc123",
  "prediction": 0.87
}
```

This is essential for reproducibility and auditability.

---

# 28. Data Lineage

For each inspection, record:

```text
inspection_id
image_hash
input_data_version
model_version
xai_version
prompt_version
llm_provider
llm_model
generation_configuration
timestamp
```

This allows exact reconstruction of how a result was produced.

---

# 29. Confidence Calibration

Raw model confidence should not automatically be treated as calibrated probability.

Evaluate:

- reliability diagrams
- Expected Calibration Error
- Brier score
- calibration curves

Where appropriate, use:

- temperature scaling
- Platt scaling
- isotonic regression

Expose calibrated confidence separately from raw confidence.

---

# 30. Input Validation

Environmental values must be validated before inference.

Example:

```python
class EnvironmentalData(BaseModel):
    temperature: float = Field(ge=-50, le=100)
    humidity: float = Field(ge=0, le=100)
    storage_days: float = Field(ge=0)
```

Reject impossible values before they reach the model.

---

# 31. Secure Image Upload

Validate:

- file size
- MIME type
- file extension
- image dimensions
- actual image decoding
- malformed images
- decompression bombs

Do not trust the file extension alone.

---

# 32. API Security

Add:

- authentication
- authorization
- rate limiting
- strict CORS
- request validation
- payload size limits
- secure headers

Do not expose internal model paths, exception traces, or secrets in production responses.

---

# 33. Secrets Management

Never commit:

- API keys
- database passwords
- access tokens
- cloud credentials

Use:

- environment variables for development
- Docker/Kubernetes secrets
- a secret manager for production

---

# 34. Prompt Injection Protection

Even though FoodShieldAI is not primarily a chatbot, uploaded OCR text or external text can contain malicious instructions.

Treat all extracted text as **untrusted data**.

Architecture:

```text
OCR / External text
       |
       v
Untrusted evidence
       |
       v
Sanitized structured data
       |
       v
LLM prompt
```

Never allow OCR text to override system instructions.

---

# 35. LLM Safety Boundary

The LLM must not:

- override model predictions
- invent laboratory results
- invent pathogens
- claim certainty beyond evidence
- provide unsupported regulatory claims
- silently change risk classifications

The deterministic/model layer remains authoritative for the actual prediction.

---

# 36. Observability

Track separate timings:

```text
preprocessing_time
classification_time
spoilage_model_time
packaging_model_time
contamination_model_time
shelf_life_time
ocr_time
xai_time
llm_time
serialization_time
total_latency
```

This allows optimization based on actual measurements.

---

# 37. vLLM Metrics

Once vLLM is implemented, monitor:

- request count
- successful requests
- failed requests
- queue time
- time-to-first-token
- generation latency
- input tokens
- output tokens
- tokens/sec
- GPU utilization
- GPU memory
- timeout count
- fallback count

Use Prometheus/Grafana if those components are actually deployed.

---

# 38. Health and Readiness Endpoints

Implement:

```text
GET /health
GET /ready
GET /live
```

Readiness should check required dependencies.

Example:

```text
API              OK
Model            OK
Database         OK
Redis            OK
Object storage   OK
vLLM             OK
```

Do not mark the application ready if mandatory model initialization failed.

---

# 39. Model Loading

Avoid unnecessary memory spikes.

Consider:

- lazy loading
- model caching
- warm-up
- CPU/GPU placement
- quantization
- model sharing

Do not load multiple large models concurrently without measuring GPU memory.

---

# 40. Warm-up

Before marking the service ready:

1. Load model.
2. Run a small inference.
3. Initialize CUDA if required.
4. Initialize tokenizer.
5. Verify output.
6. Mark service ready.

This avoids an unexpectedly slow first request.

---

# 41. Threading and GPU Concurrency

The current orchestrator uses thread pools.

Do not assume more threads automatically mean better performance.

Benchmark:

- sequential inference
- threaded inference
- batched inference
- CPU/GPU execution
- memory usage
- latency
- throughput

GPU models can contend for:

- VRAM
- CUDA streams
- memory bandwidth
- compute

Optimize based on measurements.

---

# 42. Remove Redundant ThreadPoolExecutors

If the class creates an executor in its constructor but the actual method creates another executor locally, eliminate the duplication.

Choose one lifecycle strategy.

Prefer explicit lifecycle management.

---

# 43. Asynchronous Processing

A complete inspection may become too expensive for synchronous HTTP.

Consider:

```text
POST /inspections
       |
       v
202 Accepted
       |
       v
Job Queue
       |
       v
Worker
       |
       v
AI pipeline
       |
       v
Database
```

Then:

```text
GET /inspections/{inspection_id}
```

returns status/results.

Possible future technologies:

- Celery
- Dramatiq
- RQ
- Redis Streams
- Kafka for high-scale event processing

Do not add these until the simpler architecture requires them.

---

# 44. Idempotency

Use an idempotency key or input content hash.

For example:

```text
SHA256(
    image bytes
    +
    environmental data
)
```

This prevents accidental duplicate inspections caused by network retries.

---

# 45. Testing Strategy

The project should have multiple testing layers.

## Unit tests

Test:

- preprocessing
- individual models
- XAI algorithms
- LLM provider adapters
- report validation
- configuration
- exception mapping

## Integration tests

Test:

```text
API -> service
service -> model
service -> XAI
service -> LLM
service -> database
```

## End-to-end tests

Test:

```text
upload image
   |
   v
full inspection
   |
   v
XAI
   |
   v
LLM report
   |
   v
API response
```

---

# 46. XAI-Specific Tests

Do not test only whether the XAI endpoint returns HTTP 200.

Test:

- heatmap dimensions
- numerical range
- target class behavior
- deterministic/reproducible behavior where expected
- attribution stability
- SHAP additivity where applicable
- counterfactual feasibility
- explanation consistency

---

# 47. LLM-Specific Tests

Test:

- provider unavailable
- timeout
- malformed JSON
- schema violation
- unsupported claim
- empty response
- fallback behavior
- prompt version
- provider selection
- vLLM connection
- Ollama connection

---

# 48. ML Regression Testing

Maintain a fixed regression dataset.

For each model version evaluate:

- accuracy
- precision
- recall
- F1
- AUROC
- calibration
- inference latency

Fail CI if important metrics regress beyond configured thresholds.

---

# 49. Separate Training From Inference

Keep production inference free of training logic.

Recommended separation:

```text
ml/
├── training/
├── evaluation/
└── experiments/

aiengine/
├── inference/
├── xai/
└── serving/
```

Production services load versioned artifacts.

They must never retrain models during normal API execution.

---

# 50. CI/CD

The CI pipeline should enforce:

```text
Pull Request
 |
 +-- Ruff lint
 +-- Ruff format check
 +-- mypy strict
 +-- unit tests
 +-- integration tests
 +-- coverage
 +-- dependency audit
 +-- secret scan
 +-- Docker build
 +-- container security scan
 |
 v
Merge
```

Only after successful checks should deployment occur.

---

# 51. Security Tooling

Consider:

- `pip-audit`
- Dependabot
- Bandit
- Trivy
- secret scanning

Keep tooling configuration in the repository.

---

# 52. Documentation Corrections

The repository currently describes many technologies and capabilities.

Do not claim a technology is implemented unless it is actually functional.

Use status markers:

```text
Implemented
Partially implemented
Experimental
Planned
```

This is especially important for:

- vLLM
- SHAP
- LIME
- Kubernetes
- MLflow
- Redis
- PostgreSQL
- S3/MinIO
- Prometheus
- Grafana
- Loki

---

# 53. Repository Naming Corrections

Ensure documentation matches the actual repository.

If the actual folder is:

```text
aiengine/
```

do not document:

```text
ai-engine/
```

unless both really exist.

Replace placeholder clone URLs such as:

```text
https://github.com/your-org/foodshield-ai.git
```

with the real repository URL.

---

# 54. Architecture Documentation

Add an architecture diagram demonstrating dependency direction.

Recommended:

```text
Frontend
   |
   v
FastAPI
   |
   v
Application Services
   |
   v
Domain
   |
   v
Ports / Interfaces
   |
   +----------+----------+----------+
   |          |          |          |
   v          v          v          v
PyTorch     ONNX       vLLM      Database
```

Document the dependency rule:

```text
Infrastructure -> Application -> Domain
```

The domain must not depend on infrastructure.

---

# 55. Recommended Target Architecture

The desired high-level architecture is:

```text
                         Frontend
                            |
                            v
                         FastAPI
                            |
                            v
                  InspectionApplication
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
      VisionPipeline   RiskPipeline   OCRPipeline
             |              |
             v              v
         CV Models      Risk Models
             |              |
             +-------+------+
                     |
                     v
             ExplainabilityService
                 |           |
                 v           v
             Grad-CAM       SHAP
                 |           |
                 +-----+-----+
                       |
                       v
              Counterfactuals
                       |
                       v
               Structured Evidence
                       |
                       v
                 ReportService
                       |
                       v
                  LLMProvider
              /        |        \
             /         |         \
          vLLM       Ollama     OpenAI
             |
             v
       Open-source LLM
```

---

# 56. Suggested Package Structure

Adapt the current repository rather than blindly replacing it.

A target AI-engine structure could be:

```text
aiengine/
├── domain/
│   ├── models/
│   │   ├── inspection.py
│   │   ├── predictions.py
│   │   ├── xai.py
│   │   └── reports.py
│   ├── exceptions.py
│   └── protocols/
│       ├── models.py
│       ├── explainers.py
│       └── llm.py
│
├── application/
│   ├── inspection_service.py
│   ├── explanation_service.py
│   └── report_service.py
│
├── infrastructure/
│   ├── models/
│   ├── xai/
│   ├── llm/
│   │   ├── vllm_provider.py
│   │   ├── ollama_provider.py
│   │   └── openai_provider.py
│   ├── persistence/
│   └── storage/
│
├── serving/
│   ├── dependencies.py
│   └── health.py
│
└── config.py
```

Do not perform a large structural migration in one step. Refactor incrementally and keep tests passing.

---

# 57. Implementation Order

Implement in phases.

## Phase 1 — Correctness

1. Audit all XAI methods.
2. Remove false LIME implementation.
3. Implement genuine SHAP.
4. Connect XAI to actual trained prediction models.
5. Remove duplicate Grad-CAM.
6. Correct counterfactual model dependency.
7. Add XAI tests.

## Phase 2 — LLM/vLLM

8. Define `LLMProvider`.
9. Refactor Ollama behind the interface.
10. Implement `VLLMProvider`.
11. Add vLLM Docker configuration.
12. Add provider selection through configuration.
13. Add timeouts/retries.
14. Add circuit breaker/fallback.
15. Add structured Pydantic report validation.
16. Add prompt versioning.
17. Add hallucination/evidence checks.

## Phase 3 — Architecture

18. Introduce domain models.
19. Introduce domain exceptions.
20. Introduce dependency injection.
21. Break down `AIInferenceOrchestrator`.
22. Centralize configuration.
23. Improve typing.

## Phase 4 — ML Engineering

24. Model version metadata.
25. Data validation.
26. Confidence calibration.
27. Regression dataset.
28. ML metrics.
29. XAI metrics/tests.

## Phase 5 — Production Engineering

30. Request IDs.
31. Inspection IDs.
32. Structured logging.
33. Latency metrics.
34. vLLM metrics.
35. Health/readiness endpoints.
36. Model warm-up.
37. Security controls.
38. Idempotency.
39. Async processing where required.

## Phase 6 — CI/CD

40. Ruff.
41. Format.
42. mypy strict.
43. Unit tests.
44. Integration tests.
45. E2E tests.
46. Coverage.
47. Security scanning.
48. Docker scanning.
49. Model regression checks.

## Phase 7 — Documentation

50. Correct README.
51. Add feature maturity table.
52. Add architecture diagram.
53. Add XAI methodology.
54. Add vLLM setup guide.
55. Add model cards.
56. Add API examples.

---

# 58. Non-Negotiable Engineering Rules

The CLI agent implementing this specification must follow these rules.

### Rule 1

Do not claim SHAP unless actual SHAP methodology/library is being used appropriately.

### Rule 2

Do not claim LIME unless genuine LIME is implemented.

### Rule 3

Do not claim vLLM unless the application can actually communicate with a running vLLM server.

### Rule 4

The LLM must not become the authoritative food-safety classifier.

### Rule 5

XAI must explain the actual model responsible for the prediction.

### Rule 6

Do not silently swallow exceptions.

### Rule 7

Avoid `Any`, untyped dictionaries, and untyped functions where domain types are possible.

### Rule 8

Do not introduce circular dependencies.

### Rule 9

Do not add infrastructure technologies without a functional use case.

### Rule 10

Do not break existing functionality while refactoring.

### Rule 11

Every significant refactor must add/update tests.

### Rule 12

Run linting, typing and tests after each major phase.

### Rule 13

Do not hardcode secrets, model paths, provider URLs, or credentials.

### Rule 14

Do not allow LLM output to override deterministic model predictions.

### Rule 15

Prefer simple, maintainable implementations over unnecessary abstraction.

---

# 59. Definition of Done

FoodShieldAI should not be considered complete until the following are true.

## XAI

- [ ] Grad-CAM is implemented once and tested.
- [ ] LIME is genuine or removed.
- [ ] SHAP is genuine or correctly renamed as perturbation attribution.
- [ ] XAI explains actual trained models.
- [ ] Counterfactuals operate on actual models.
- [ ] XAI results are typed.
- [ ] XAI tests exist.

## vLLM

- [ ] VLLMProvider exists.
- [ ] vLLM server can be started.
- [ ] Application can call vLLM.
- [ ] Model selection is configurable.
- [ ] Health checks exist.
- [ ] Timeout exists.
- [ ] Retry policy exists.
- [ ] Fallback exists.
- [ ] Metrics exist.
- [ ] Integration tests exist.

## OOP/SOLID

- [ ] Provider interfaces exist.
- [ ] Dependency injection is used.
- [ ] Orchestrator responsibilities are reduced.
- [ ] Domain models exist.
- [ ] Domain exceptions exist.
- [ ] No unnecessary provider conditionals exist.
- [ ] Circular dependencies are absent.

## LLM Safety

- [ ] Structured Pydantic output.
- [ ] Evidence-grounded prompting.
- [ ] Prompt versioning.
- [ ] Unsupported-claim prevention.
- [ ] No autonomous safety decisions by LLM.
- [ ] Deterministic fallback.

## ML Engineering

- [ ] Model versions tracked.
- [ ] Model hashes tracked.
- [ ] Regression dataset exists.
- [ ] Evaluation metrics exist.
- [ ] Calibration evaluated.
- [ ] Input validation exists.

## Production Engineering

- [ ] Request IDs.
- [ ] Inspection IDs.
- [ ] Structured logs.
- [ ] Health endpoint.
- [ ] Readiness endpoint.
- [ ] Timeouts.
- [ ] Retries.
- [ ] Circuit breaker.
- [ ] Idempotency.
- [ ] Security controls.

## CI/CD

- [ ] Ruff passes.
- [ ] Formatting passes.
- [ ] Strict mypy passes.
- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] E2E tests pass.
- [ ] Coverage threshold passes.
- [ ] Security scanning passes.
- [ ] Docker image scanning passes.

---

# 60. Final Engineering Objective

The final FoodShieldAI system should demonstrate the following complete chain:

```text
Real-world food image + environmental data
                    |
                    v
             Input validation
                    |
                    v
             Preprocessing
                    |
                    v
             ML/CV inference
                    |
                    +------------------+
                    |                  |
                    v                  v
                Prediction            XAI
                    |                  |
                    |            Grad-CAM / SHAP /
                    |            Counterfactuals
                    |                  |
                    +---------+--------+
                              |
                              v
                     Structured evidence
                              |
                              v
                       Safety decision
                              |
                              v
                       LLM Report Layer
                              |
                         LLMProvider
                       /      |       \
                      v       v        v
                    vLLM   Ollama    OpenAI
                      |
                      v
                Structured report
                      |
                      v
                Pydantic validation
                      |
                      v
                 Safety filtering
                      |
                      v
                 API response
                      |
                      v
             Monitoring / Audit Trail
```

The project should ultimately be able to demonstrate:

**Machine Learning + Explainable AI + LLM Engineering + vLLM Serving + OOP + SOLID + Clean Architecture + MLOps + Observability + Security + Testing.**

The goal is not to maximize the number of technologies. The goal is to make every technology that remains **correctly implemented, testable, observable, and justified by the system's real requirements**.
