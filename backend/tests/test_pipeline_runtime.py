import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.pipeline_runtime import build_data_quality_block, consume_provider_budget


class PipelineRuntimeTests(unittest.TestCase):
    def test_persistent_provider_ledger_enforces_daily_budget_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ledger_path = temp_path / "provider_budget_ledger.json"

            base_env = {
                "ENABLE_PERSISTENT_PROVIDER_LEDGER": "true",
                "PIPELINE_PROVIDER_LEDGER_PATH": str(ledger_path),
                "PIPELINE_BUDGET_DATE": "2026-03-29",
                "PIPELINE_SCOPE": "runtime_test",
            }

            def consume_for_run(run_id: str, units: int) -> bool:
                run_cache_root = temp_path / "cache" / run_id
                with patch.dict(
                    os.environ,
                    {
                        **base_env,
                        "PIPELINE_RUN_ID": run_id,
                        "PIPELINE_CACHE_ROOT": str(run_cache_root),
                    },
                    clear=False,
                ):
                    return consume_provider_budget(
                        "alpha_vantage",
                        units=units,
                        category="price",
                        daily_limit=25,
                    )

            self.assertTrue(consume_for_run("run-1", 20))
            self.assertTrue(consume_for_run("run-2", 5))

            with patch.dict(
                os.environ,
                {
                    **base_env,
                    "PIPELINE_RUN_ID": "run-3",
                    "PIPELINE_CACHE_ROOT": str(temp_path / "cache" / "run-3"),
                },
                clear=False,
            ):
                self.assertFalse(
                    consume_provider_budget(
                        "alpha_vantage",
                        units=1,
                        category="price",
                        daily_limit=25,
                    )
                )
                quality = build_data_quality_block(scope="runtime_test")

            self.assertEqual("exhausted", quality["provider_status"]["alpha_vantage"]["status"])
            self.assertEqual(25, quality["provider_status"]["alpha_vantage"]["daily"]["used"])
            self.assertEqual(0, quality["provider_status"]["alpha_vantage"]["daily"]["remaining"])

            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(
                25,
                ledger["days"]["2026-03-29"]["providers"]["alpha_vantage"]["used"],
            )


if __name__ == "__main__":
    unittest.main()
