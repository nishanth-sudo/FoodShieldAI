import uuid
from dataclasses import dataclass
from datetime import datetime

from backend.core.time import utc_now


@dataclass
class RetrainingJob:
    job_id: str
    status: str
    triggered_at: datetime
    completed_at: datetime | None
    metrics: dict | None
    error: str | None

class RetrainingPipeline:
    def __init__(self) -> None:
        self.jobs: dict[str, RetrainingJob] = {}

    def trigger_retraining(self, reason: str) -> RetrainingJob:
        job_id = str(uuid.uuid4())
        job = RetrainingJob(
            job_id=job_id,
            status="pending",
            triggered_at=utc_now(),
            completed_at=None,
            metrics=None,
            error=None
        )
        self.jobs[job_id] = job
        return job

    def get_job_status(self, job_id: str) -> RetrainingJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[RetrainingJob]:
        return list(self.jobs.values())

retraining_pipeline = RetrainingPipeline()
