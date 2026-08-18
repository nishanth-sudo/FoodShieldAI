import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

from backend.core.time import utc_now


@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    training_date: datetime
    metrics: dict
    model_path: str
    is_active: bool
    created_at: datetime

class ModelRegistry:
    def __init__(self, registry_path: str = 'models/registry.json') -> None:
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

    def load_registry(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def save_registry(self, data: dict) -> None:
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)

    def register_model(
        self, model_name: str, version: str, model_path: str, metrics: dict
    ) -> ModelVersion:
        registry = self.load_registry()
        if model_name not in registry:
            registry[model_name] = []

        new_version = ModelVersion(
            version_id=version,
            model_name=model_name,
            training_date=utc_now(),
            metrics=metrics,
            model_path=model_path,
            is_active=False,
            created_at=utc_now()
        )
        registry[model_name].append(asdict(new_version))
        self.save_registry(registry)
        return new_version

    def get_active_version(self, model_name: str) -> ModelVersion | None:
        registry = self.load_registry()
        versions = registry.get(model_name, [])
        for v in versions:
            if v.get('is_active'):
                return ModelVersion(**v)
        return None

    def set_active_version(self, model_name: str, version_id: str) -> bool:
        registry = self.load_registry()
        versions = registry.get(model_name, [])
        found = False
        for v in versions:
            if v['version_id'] == version_id:
                v['is_active'] = True
                found = True
            else:
                v['is_active'] = False
        if found:
            self.save_registry(registry)
        return found

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        registry = self.load_registry()
        versions = registry.get(model_name, [])
        return [ModelVersion(**v) for v in versions]

    def get_version(self, model_name: str, version_id: str) -> ModelVersion | None:
        registry = self.load_registry()
        versions = registry.get(model_name, [])
        for v in versions:
            if v['version_id'] == version_id:
                return ModelVersion(**v)
        return None

model_registry = ModelRegistry()
