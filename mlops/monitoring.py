import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PredictionLogger:
    def __init__(self, log_dir: str = "mlops/monitoring/logs", max_records: int = 10000) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self._buffer: list[dict] = []

    def log(
        self, model_name: str, features: dict,
        prediction: dict, latency_ms: float | None = None,
    ) -> None:
        record = {
            "model_name": model_name,
            "timestamp": time.time(),
            "features": features,
            "prediction": prediction,
            "latency_ms": latency_ms,
        }
        self._buffer.append(record)

        if len(self._buffer) >= 100:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        date_str = time.strftime("%Y%m%d")
        log_file = self.log_dir / f"predictions_{date_str}.jsonl"
        with open(log_file, "a") as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")
        self._buffer.clear()

    def get_recent_predictions(self, model_name: str, n: int = 1000) -> list[dict]:
        records: list[dict] = []
        for log_file in sorted(self.log_dir.glob("predictions_*.jsonl"), reverse=True):
            with open(log_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["model_name"] == model_name:
                        records.append(rec)
                        if len(records) >= n:
                            return records
        return records


class ModelMonitor:
    def __init__(
        self,
        log_dir: str = "mlops/monitoring/logs",
        reference_dir: str = "mlops/monitoring/reference",
    ) -> None:
        self.logger = PredictionLogger(log_dir=log_dir)
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, model_name: str, features: dict, prediction: dict, latency_ms: float | None = None) -> None:
        self.logger.log(model_name, features, prediction, latency_ms)

    def compute_feature_statistics(self, model_name: str, max_samples: int = 5000) -> dict:
        predictions = self.logger.get_recent_predictions(model_name, n=max_samples)
        if not predictions:
            return {}

        feature_values: dict[str, list[float]] = {}
        for pred in predictions:
            for key, value in pred.get("features", {}).items():
                if isinstance(value, (int, float)):
                    feature_values.setdefault(key, []).append(float(value))
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, (int, float)):
                            feature_values.setdefault(f"{key}[{i}]", []).append(float(v))

        stats = {}
        for key, values in feature_values.items():
            arr = np.array(values)
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "n": int(len(values)),
            }
        return stats

    def get_model_performance(self, model_name: str) -> dict:
        predictions = self.logger.get_recent_predictions(model_name, n=1000)
        if not predictions:
            return {}

        latencies = [p.get("latency_ms", 0) for p in predictions if p.get("latency_ms") is not None]
        prediction_counts = len(predictions)

        freshness_scores = []
        spoilage_flags = []
        for p in predictions:
            pred = p.get("prediction", {})
            if isinstance(pred, dict):
                if "freshness_score" in pred:
                    freshness_scores.append(pred["freshness_score"])
                if "is_spoiled" in pred:
                    spoilage_flags.append(pred["is_spoiled"])

        return {
            "model_name": model_name,
            "total_predictions": prediction_counts,
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "avg_freshness_score": float(np.mean(freshness_scores)) if freshness_scores else 0.0,
            "spoilage_rate": float(np.mean(spoilage_flags)) if spoilage_flags else 0.0,
            "time_span_hours": self._compute_time_span(predictions),
        }

    def detect_data_drift(self, model_name: str) -> dict:
        recent_stats = self.compute_feature_statistics(model_name, max_samples=5000)
        reference_path = self.reference_dir / f"{model_name}_reference.json"

        if not reference_path.exists():
            if recent_stats:
                reference_path.write_text(json.dumps(recent_stats, indent=2))
                return {"drift_detected": False, "message": "Reference distribution recorded"}
            return {"drift_detected": False, "message": "Insufficient data"}

        reference_stats = json.loads(reference_path.read_text())
        drift_results: dict[str, dict] = {}

        for feature, ref_stats in reference_stats.items():
            current = recent_stats.get(feature)
            if current is None:
                continue

            ref_mean = ref_stats["mean"]
            ref_std = ref_stats["std"]
            cur_mean = current["mean"]

            effect_size = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
            ps_drift = self._psi_score(ref_stats, current)
            js_drift = self._js_divergence(ref_stats, current)

            drift_detected = effect_size > 0.5 or ps_drift > 0.2

            drift_results[feature] = {
                "drift_detected": bool(drift_detected),
                "effect_size": round(float(effect_size), 4),
                "psi": round(float(ps_drift), 4),
                "js_divergence": round(float(js_drift), 4),
                "reference_mean": round(float(ref_mean), 4),
                "current_mean": round(float(cur_mean), 4),
                "reference_std": round(float(ref_std), 4),
                "current_std": round(float(current.get("std", 0)), 4),
            }

        drifted_features = [k for k, v in drift_results.items() if v["drift_detected"]]
        overall_drift = len(drifted_features) > 0

        return {
            "drift_detected": overall_drift,
            "drifted_features": drifted_features,
            "total_features": len(drift_results),
            "drift_ratio": round(len(drifted_features) / max(len(drift_results), 1), 4),
            "details": drift_results,
        }

    def flush(self) -> None:
        self.logger.flush()

    def _compute_time_span(self, predictions: list[dict]) -> float:
        if not predictions:
            return 0.0
        timestamps = [p.get("timestamp", 0) for p in predictions]
        return (max(timestamps) - min(timestamps)) / 3600.0 if timestamps else 0.0

    @staticmethod
    def _psi_score(reference: dict, current: dict, bins: int = 10) -> float:
        ref_mean = reference.get("mean", 0)
        ref_std = reference.get("std", 1)
        cur_mean = current.get("mean", 0)
        cur_std = current.get("std", 1)

        if ref_std < 1e-8:
            return 0.0

        boundaries = [ref_mean + ref_std * (-3 + 6 * i / bins) for i in range(bins + 1)]

        ref_proportions = np.zeros(bins)
        cur_proportions = np.zeros(bins)

        if "n" in reference and reference["n"] > 0:
            ref_samples = np.random.normal(ref_mean, ref_std, min(reference["n"], 10000))
            cur_samples = np.random.normal(cur_mean, cur_std, min(current.get("n", 10000), 10000))

            for i in range(bins):
                lo, hi = boundaries[i], boundaries[i + 1]
                ref_proportions[i] = np.mean((ref_samples >= lo) & (ref_samples < hi))
                cur_proportions[i] = np.mean((cur_samples >= lo) & (cur_samples < hi))

        ref_proportions = np.clip(ref_proportions, 1e-4, None)
        cur_proportions = np.clip(cur_proportions, 1e-4, None)

        psi = np.sum((cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions))
        return float(psi)

    @staticmethod
    def _js_divergence(reference: dict, current: dict) -> float:
        ref_mean = reference.get("mean", 0)
        ref_std = reference.get("std", 1)
        cur_mean = current.get("mean", 0)
        cur_std = current.get("std", 1)

        var_ref = ref_std ** 2
        var_cur = cur_std ** 2
        mean_diff = cur_mean - ref_mean

        js = 0.5 * (
            np.log(np.sqrt((var_ref + var_cur + mean_diff ** 2) / (2 * var_ref))) +
            np.log(np.sqrt((var_ref + var_cur + mean_diff ** 2) / (2 * var_cur)))
        )
        return float(min(js, 1.0))
