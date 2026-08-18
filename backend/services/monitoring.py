import time
from collections import deque
from typing import Any


class MetricsCollector:
    def __init__(self) -> None:
        self.prediction_count = 0
        self.prediction_errors = 0
        self.avg_confidence = 0.0
        self.avg_response_time_ms = 0.0
        self.low_confidence_count = 0
        self.requests_per_minute = deque()

    def record_prediction(
        self, confidence: float, response_time_ms: float, success: bool
    ) -> None:
        if success:
            total = self.prediction_count * self.avg_confidence
            self.prediction_count += 1
            self.avg_confidence = (total + confidence) / self.prediction_count

            total_time = (self.prediction_count - 1) * self.avg_response_time_ms
            self.avg_response_time_ms = (total_time + response_time_ms) / self.prediction_count

            if confidence < 0.5: # threshold example
                self.low_confidence_count += 1
        else:
            self.prediction_errors += 1

    def record_request(self) -> None:
        self.requests_per_minute.append(time.time())

    def get_requests_per_minute(self) -> float:
        now = time.time()
        while self.requests_per_minute and now - self.requests_per_minute[0] > 60:
            self.requests_per_minute.popleft()
        return float(len(self.requests_per_minute))

    def get_stats(self) -> dict[str, Any]:
        return {
            "prediction_count": self.prediction_count,
            "prediction_errors": self.prediction_errors,
            "avg_confidence": self.avg_confidence,
            "avg_response_time_ms": self.avg_response_time_ms,
            "low_confidence_count": self.low_confidence_count,
            "requests_per_minute": self.get_requests_per_minute()
        }

    def reset_stats(self) -> None:
        self.prediction_count = 0
        self.prediction_errors = 0
        self.avg_confidence = 0.0
        self.avg_response_time_ms = 0.0
        self.low_confidence_count = 0
        self.requests_per_minute.clear()

metrics = MetricsCollector()
