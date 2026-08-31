from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class WorkflowTests(unittest.TestCase):
    def test_default_config_uses_real_tabm_adapter(self):
        config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        command = config["adapter_command"]
        self.assertIn("adapters/tabm_factor_adapter.py", command)
        self.assertNotIn("adapters/mock_adapter.py", command)
        self.assertEqual(config["runtime_config"], "runtime/config.private.yaml")
        adapter = ROOT / "adapters" / "tabm_factor_adapter.py"
        registry = ROOT / "factor_library" / "implementations" / "registry.py"
        compile(adapter.read_text(encoding="utf-8"), str(adapter), "exec")
        compile(registry.read_text(encoding="utf-8"), str(registry), "exec")

    def test_catalog_validator(self):
        completed = subprocess.run([sys.executable, str(SCRIPTS / "validate_catalog.py")], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_mock_adapter_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            request = {
                "schema_version": "1.0.0",
                "experiment_id": "exp-test",
                "created_at": "2026-01-01T00:00:00Z",
                "stage": "smoke",
                "factor_ids": ["factor.power.multiscale-lags"],
                "task": {"horizon_steps": list(range(1, 17))},
                "protected_snapshot": {"combined_sha256": "abc"},
                "leakage_checks": ["availability"],
            }
            request_path = temporary / "request.json"
            output_path = temporary / "result.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "adapters" / "mock_adapter.py"), "--request", str(request_path), "--output", str(output_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(result["protected_snapshot"]["unchanged"])
            self.assertEqual(len(result["metrics"]["by_horizon"]), 16)
            self.assertIn("SYNTHETIC MOCK RESULT", result["notes"][0])


if __name__ == "__main__":
    unittest.main()
