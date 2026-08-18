import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    batch_size: int
    input_size: tuple[int, ...]
    device: str
    inference_time_ms: float
    throughput_fps: float
    peak_memory_mb: float
    model_params: int
    model_size_mb: float


class ModelBenchmark:
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def measure_inference_time(
        self,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> float:
        dummy_input = torch.randn(input_shape, device=self.device)
        with torch.no_grad():
            for _ in range(num_warmup):
                self.model(dummy_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                self.model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)

        return float(np.median(times))

    def measure_peak_memory(
        self,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> float:
        if not torch.cuda.is_available():
            import psutil

            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)
            dummy_input = torch.randn(input_shape, device="cpu")
            with torch.no_grad():
                self.model(dummy_input)
            memory_after = process.memory_info().rss / (1024 * 1024)
            return memory_after - memory_before

        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randn(input_shape, device=self.device)
        with torch.no_grad():
            self.model(dummy_input)
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def estimate_model_size(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def run(
        self,
        batch_size: int = 1,
        input_shape: tuple[int, ...] = (3, 224, 224),
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> BenchmarkResult:
        full_shape = (batch_size,) + input_shape
        inference_time = self.measure_inference_time(full_shape, num_warmup, num_runs)
        throughput = 1000.0 / inference_time * batch_size
        memory = self.measure_peak_memory(full_shape)
        params = self.count_parameters()
        model_size = self.estimate_model_size()

        return BenchmarkResult(
            model_name=self.model.__class__.__name__,
            batch_size=batch_size,
            input_size=full_shape,
            device=self.device,
            inference_time_ms=round(inference_time, 2),
            throughput_fps=round(throughput, 2),
            peak_memory_mb=round(memory, 2),
            model_params=params,
            model_size_mb=round(model_size, 2),
        )

    @staticmethod
    def compare_models(
        models: dict[str, nn.Module],
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        device: str = "cpu",
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            bench = ModelBenchmark(model, device=device)
            result = bench.run(batch_size=input_shape[0], input_shape=input_shape[1:])
            result.model_name = name
            results.append(result)
        return results
