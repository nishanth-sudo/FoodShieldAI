# FoodShield AI — High-Level Architecture

## Architecture Philosophy

FoodShield AI follows a **layered, modular, service-oriented architecture** guided by:

- **Clean Architecture** — Separation of concerns with domain at the center
- **Hexagonal Architecture (Ports & Adapters)** — Core logic isolated from external concerns
- **SOLID Principles** — Maintainable, testable, extensible codebase

---

## High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                               │
│  ┌──────────────────┐    ┌──────────────────────────────────┐     │
│  │   Web App        │    │   Mobile App (Future)            │     │
│  │  (React/Next.js) │    │   (React Native / Flutter)       │     │
│  └───────┬──────────┘    └──────────────┬───────────────────┘     │
│          │                               │                         │
│          └───────────────┬───────────────┘                         │
│                          │ HTTPS/TLS                               │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────────────┐
│                    API GATEWAY / LOAD BALANCER                     │
│                    (Nginx / Traefik / ALB)                         │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────────────┐
│                    APPLICATION LAYER (FastAPI)                     │
│                                                                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐    │
│  │  Auth      │ │ Inspection │ │  Report    │ │  Admin     │    │
│  │  Service   │ │  Service   │ │  Service   │ │  Dashboard │    │
│  └─────┬──────┘ └──────┬─────┘ └──────┬─────┘ └──────┬─────┘    │
│        │               │              │               │          │
│  ┌─────┴───────────────┴──────────────┴───────────────┴─────┐    │
│  │                    CORE / DOMAIN LAYER                    │    │
│  │  Entities, Value Objects, Use Cases, Ports, DTOs         │    │
│  └────────────────────────────┬──────────────────────────────┘    │
└───────────────────────────────┼────────────────────────────────────┘
                                │
┌───────────────────────────────┼────────────────────────────────────┐
│                    AI PROCESSING LAYER                             │
│                                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │  Image   │ │   CV     │ │   OCR    │ │  Explainable AI  │    │
│  │  Pre-    │ │  Models  │ │(PaddleOCR│ │  (Grad-CAM /     │    │
│  │  process │ │  (5)     │ │ /Tesseract│ │  LIME / SHAP)    │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘    │
│       │            │            │                 │               │
│       └────────────┴────────────┴─────────────────┘               │
│                          │                                        │
│                 ┌────────┴────────┐                               │
│                 │  LLM Report     │                               │
│                 │  Generator      │                               │
│                 │  (GPT / Llama / │                               │
│                 │   Fine-tuned)   │                               │
│                 └────────┬────────┘                               │
└──────────────────────────┼────────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────────┐
│                    DATA LAYER                                     │
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│  │   PostgreSQL    │  │  Object Storage  │  │    Redis      │   │
│  │  (Users, Insp., │  │  (Images,        │  │  (Cache,      │   │
│  │   Reports, Meta)│  │   Reports, Models)│  │   Sessions)   │   │
│  └─────────────────┘  └──────────────────┘  └───────────────┘   │
└──────────────────────────┼────────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────────┐
│                    MLOps & INFRASTRUCTURE LAYER                   │
│                                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │  Docker  │ │   K8s    │ │ CI/CD    │ │  Monitoring      │    │
│  │          │ │          │ │(GH Actions│ │(Prometheus,      │    │
│  │          │ │          │ │ /Jenkins) │ │  Grafana, Loki)  │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │  DVC     │ │  MLflow  │ │  Model   │ │  Centralized     │    │
│  │(Data Ver)│ │(Exp Track│ │  Registry│ │  Logging (ELK)   │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Layer Breakdown

### 1. Client Layer
- **Web Application**: React/Next.js SPA — image upload, dashboard, reports
- **Mobile Application** (Future): React Native / Flutter
- Communication via HTTPS REST + WebSocket for real-time updates

### 2. API Gateway
- Nginx/Traefik for TLS termination, rate limiting, routing
- Load balancing across multiple backend instances

### 3. Application Layer (FastAPI)
- **Auth Service**: JWT-based authentication, RBAC, OAuth2 support
- **Inspection Service**: Orchestrates the full inspection workflow
- **Report Service**: Manages LLM-generated inspection reports
- **Admin Dashboard**: User management, system monitoring, analytics

### 4. Core / Domain Layer
- Clean Architecture innermost layer — no framework dependencies
- **Entities**: `User`, `Inspection`, `FoodItem`, `Report`, `Model`
- **Value Objects**: `Image`, `ConfidenceScore`, `ShelfLife`, `RiskLevel`
- **Use Cases**: `SubmitInspection`, `GenerateReport`, `TrainModel`
- **Ports**: Interfaces for repositories, AI services, storage

### 5. AI Processing Layer

| Module | Technology | Models |
|--------|-----------|--------|
| Food Classification | CNN / ViT | ResNet, EfficientNet, ViT |
| Spoilage Detection | CNN + Anomaly | Custom trained, Autoencoder |
| Packaging Defect | Object Detection | YOLOv8, DETR |
| Contamination Risk | Multi-label Classifier | Custom ensemble |
| Shelf-Life Prediction | Regression / RNN | Custom model |
| OCR | Scene Text Recognition | PaddleOCR, TrOCR, Tesseract |
| XAI | Visual Explanations | Grad-CAM, LIME, SHAP |
| LLM Reports | Text Generation | GPT-4 / Llama 3 / Fine-tuned |

#### AI Models (5 Core CV Models)

1. **FoodClassifier** — Identifies food type from image
2. **SpoilageDetector** — Binary + severity classification of spoilage
3. **PackagingDefectDetector** — Object detection for dents, tears, leaks
4. **ContaminationRiskAssessor** — Multi-label risk classification
5. **ShelfLifePredictor** — Regression model for remaining days

### 6. Data Layer
- **PostgreSQL**: Users, inspections, reports, metadata — normalized schema
- **Object Storage**: Raw images, processed images, reports, model artifacts
- **Redis**: Session cache, inspection result cache, rate-limiting

### 7. MLOps Layer
- **DVC** — Dataset versioning and pipeline reproducibility
- **MLflow** — Experiment tracking, metrics logging, model registry
- **Model Registry** — Versioned models with staging/production tags
- **Automated Pipelines** — Triggers on new data or schedule

### 8. Infrastructure Layer
- **Docker** — Containerization of all microservices
- **Kubernetes** — Orchestration, auto-scaling, rolling updates
- **CI/CD** — GitHub Actions for test → build → deploy
- **Monitoring** — Prometheus (metrics), Grafana (dashboards)
- **Logging** — Loki / ELK Stack for centralized log aggregation
- **Alerting** — Slack/Email/PagerDuty integration

---

## Data Flow: Inspection Lifecycle

```
1. User Uploads Image
       │
2. Backend Validates & Stores Image
       │
3. Image Preprocessing (resize, normalize, augment)
       │
4. Parallel AI Inference:
   ├── Food Classification → food_type, confidence
   ├── Spoilage Detection → freshness_score, spoilage_regions
   ├── Packaging Defect → defects_bounding_boxes
   ├── Contamination Risk → risk_scores, categories
   └── Shelf-Life Prediction → estimated_days_remaining
       │
5. OCR Extraction → product_name, brand, ingredients, expiry_date
       │
6. XAI Generation → heatmaps, feature importance maps
       │
7. LLM Report Generation → structured human-readable report
       │
8. Persist Results → PostgreSQL + Object Storage
       │
9. Return Response to Client → results + report + visualizations
```

---

## Security Architecture

- **TLS/HTTPS** — All client-server communication encrypted
- **JWT + OAuth2** — Stateless authentication with refresh tokens
- **RBAC** — Role-based access (Consumer, QA, Admin, Inspector)
- **Input Validation** — Pydantic models with strict validation
- **File Sanitization** — Scanned uploads, size limits, format restrictions
- **API Rate Limiting** — Per-user/IP rate limits
- **Audit Logging** — All sensitive operations logged
- **Secrets Management** — HashiCorp Vault / AWS Secrets Manager

---

## Scalability Considerations

- **Horizontal Scaling** — Stateless FastAPI behind load balancer
- **Async Processing** — Celery/Redis queue for long-running AI tasks
- **Model Serving** — Separate GPU-backed inference service (Triton / TorchServe)
- **Database** — Read replicas, connection pooling (PgBouncer)
- **Caching** — Redis for hot inspection results
- **CDN** — CloudFront for static assets and report images

---

## Deployment Topology (Production)

```
                          ┌─────────────┐
                          │   Route 53  │
                          └──────┬──────┘
                                 │
                          ┌──────┴──────┐
                          │  CloudFront │
                          │   (CDN)     │
                          └──────┬──────┘
                                 │
                          ┌──────┴──────┐
                          │    ALB      │
                          └──────┬──────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
        ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
        │  FastAPI    │  │  FastAPI    │  │  FastAPI    │
        │  Pod 1      │  │  Pod 2      │  │  Pod N      │
        └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │       AI Inference       │
                    │    (GPU Node Pool)       │
                    │  ┌──────────────────┐   │
                    │  │ Triton Inference  │   │
                    │  │ Server / TorchServe│  │
                    │  └──────────────────┘   │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
        ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
        │  PostgreSQL │  │    Redis    │  │    MinIO    │
        │  (RDS/Aurora)│  │  (ElastiCache)│ │ (S3 Compat)│
        └─────────────┘  └─────────────┘  └─────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend Framework | FastAPI | Async, auto-docs, Pydantic validation, high performance |
| AI Framework | PyTorch | Research-friendly, production-grade, ONNX export |
| Model Serving | Triton/TorchServe | GPU optimization, batching, versioned models |
| Database | PostgreSQL | ACID compliance, JSONB, full-text search, reliability |
| Object Storage | S3/MinIO | Scalable, cost-effective, any cloud |
| Caching | Redis | Low-latency, pub/sub for real-time updates |
| Task Queue | Celery + Redis | Async AI inference, decoupled processing |
| Containerization | Docker + K8s | Portability, orchestration, auto-scaling |
| Monitoring | Prometheus/Grafana | Industry standard, rich ecosystem |
| MLOps | DVC + MLflow | Established tools, versioning + experiment tracking |

---

*This architecture document is a living artifact and will evolve as the project progresses.*
