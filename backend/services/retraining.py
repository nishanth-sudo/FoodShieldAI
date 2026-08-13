from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import uuid

@dataclass
class RetrainingJob:
    job_id: str
    status: str
    triggered_at: datetime
    completed_at: Optional[datetime]
    metrics: Optional[dict]
    error: Optional[str]

class RetrainingPipeline:
    def __init__(self):
        self.jobs: dict[str, RetrainingJob] = {}

    def trigger_retraining(self, reason: str) -> RetrainingJob:
        job_id = str(uuid.uuid4())
        job = RetrainingJob(
            job_id=job_id,
            status="pending",
            triggered_at=datetime.utcnow(),
            completed_at=None,
            metrics=None,
            error=None
        )
        self.jobs[job_id] = job
        return job

    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[RetrainingJob]:
        return list(self.jobs.values())

retraining_pipeline = RetrainingPipeline()
