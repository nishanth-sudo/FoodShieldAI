from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
import hashlib

@dataclass
class ABTestVariant:
    variant_id: str
    model_version: str
    traffic_percentage: float

@dataclass
class ABTest:
    test_id: str
    name: str
    variants: List[ABTestVariant]
    is_active: bool
    created_at: datetime

class ABTestingService:
    def __init__(self):
        self.tests: Dict[str, ABTest] = {}
        self.results: Dict[str, Dict[str, dict]] = {}

    def create_test(self, name: str, variants: List[ABTestVariant]) -> ABTest:
        test_id = name.lower().replace(" ", "_")
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variants,
            is_active=True,
            created_at=datetime.utcnow()
        )
        self.tests[test_id] = test
        self.results[test_id] = {v.variant_id: {"success": 0, "total": 0} for v in variants}
        return test

    def get_active_test(self) -> Optional[ABTest]:
        for test in self.tests.values():
            if test.is_active:
                return test
        return None

    def assign_variant(self, user_id: str) -> Optional[ABTestVariant]:
        test = self.get_active_test()
        if not test or not test.variants:
            return None
        
        # hash user_id to pick variant based on traffic percentage
        h = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        cumulative = 0
        for variant in test.variants:
            cumulative += variant.traffic_percentage
            if h < cumulative:
                return variant
        return test.variants[-1]

    def record_result(self, test_id: str, variant_id: str, success: bool):
        if test_id in self.results and variant_id in self.results[test_id]:
            self.results[test_id][variant_id]["total"] += 1
            if success:
                self.results[test_id][variant_id]["success"] += 1

    def get_results(self, test_id: str) -> dict:
        return self.results.get(test_id, {})

ab_testing = ABTestingService()
