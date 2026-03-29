import unittest
from unittest.mock import patch

from backend.generate_track_record import build_signal_block_report, main


class GenerateTrackRecordTests(unittest.TestCase):
    def test_build_signal_block_report_compares_full_model_against_technical_only(self) -> None:
        entries = []
        realized_values = [0.02, 0.03, -0.01, 0.04, 0.01, -0.02]
        technical_values = [0.10, 0.12, 0.08, 0.14, 0.11, 0.07]
        block_values = [0.03, 0.04, -0.01, 0.05, 0.02, -0.02]
        for realized, technical, block in zip(realized_values, technical_values, block_values):
            entries.append(
                {
                    "realized_5d_excess_return": realized,
                    "diagnostics": {
                        "technical_total": technical,
                        "news_adjustment": block * 0.5,
                        "macro_adjustment": block * 0.3,
                        "sector_adjustment": block * 0.2,
                        "news_signal_input": block * 2.0,
                        "macro_signal_input": block * 1.2,
                        "sector_signal_input": block * 0.8,
                    },
                }
            )

        report = build_signal_block_report(entries)

        self.assertEqual("ok", report["status"])
        self.assertEqual(6, report["sample_count"])
        self.assertIn("technical_only_ic", report)
        self.assertIn("full_model_ic", report)
        self.assertIn("ic_improvement_vs_technical", report)

    @patch("backend.generate_track_record.write_json")
    @patch("backend.generate_track_record.realized_forward_return")
    @patch("backend.generate_track_record.load_sector_map")
    @patch("backend.generate_track_record.load_existing_history_entries")
    def test_main_prefers_stored_sector_over_current_universe_mapping(
        self,
        load_existing_history_entries_mock,
        load_sector_map_mock,
        realized_forward_return_mock,
        write_json_mock,
    ) -> None:
        load_existing_history_entries_mock.return_value = [
            {
                "week_id": "2026-W12",
                "week_start": "2026-03-16",
                "week_end": "2026-03-20",
                "week_label": "Mar 16 - 20, 2026",
                "logged_at": "2026-03-16T12:00:00Z",
                "status": "picked",
                "status_reason": "MSFT cleared the thresholds.",
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "sector": "technology",
                "risk": "low",
                "model_score": 0.20,
                "confidence_score": 0.82,
                "confidence_label": "high",
                "data_as_of": "2026-03-14",
                "model_version": "v3.3",
            }
        ]
        load_sector_map_mock.return_value = {"MSFT": "communication_services"}
        realized_forward_return_mock.side_effect = [
            (0.034, 0.012),
            (0.034, 0.006),
        ]

        main()

        payload = write_json_mock.call_args.args[1]
        self.assertEqual("technology", payload["entries"][0]["sector"])

    @patch("backend.generate_track_record.write_json")
    @patch("backend.generate_track_record.realized_forward_return")
    @patch("backend.generate_track_record.load_sector_map")
    @patch("backend.generate_track_record.load_existing_history_entries")
    def test_main_builds_summary_and_entries(
        self,
        load_existing_history_entries_mock,
        load_sector_map_mock,
        realized_forward_return_mock,
        write_json_mock,
    ) -> None:
        load_existing_history_entries_mock.return_value = [
            {
                "week_id": "2026-W12",
                "week_start": "2026-03-16",
                "week_end": "2026-03-20",
                "week_label": "Mar 16 - 20, 2026",
                "logged_at": "2026-03-16T12:00:00Z",
                "status": "picked",
                "status_reason": "MSFT cleared the thresholds.",
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "risk": "low",
                "model_score": 0.20,
                "confidence_score": 0.82,
                "confidence_label": "high",
                "data_as_of": "2026-03-14",
                "model_version": "v3.3",
            },
            {
                "week_id": "2026-W13",
                "week_start": "2026-03-23",
                "week_end": "2026-03-27",
                "week_label": "Mar 23 - 27, 2026",
                "logged_at": "2026-03-23T12:00:00Z",
                "status": "no_pick",
                "status_reason": "No pick",
                "symbol": None,
                "company_name": None,
                "risk": None,
                "model_score": None,
                "confidence_score": None,
                "confidence_label": None,
                "data_as_of": "2026-03-21",
                "model_version": "v3.3",
            },
        ]
        load_sector_map_mock.return_value = {"MSFT": "technology"}
        realized_forward_return_mock.side_effect = [
            (0.034, 0.012),
            (0.034, 0.006),
        ]

        main()

        payload = write_json_mock.call_args.args[1]
        self.assertEqual(2, payload["summary"]["total_weeks"])
        self.assertEqual(1, payload["summary"]["total_picks"])
        self.assertEqual(1, payload["summary"]["closed_picks"])
        self.assertEqual(1.0, payload["summary"]["win_rate"])
        self.assertEqual("MSFT", payload["entries"][1]["symbol"])
        self.assertEqual("win", payload["entries"][1]["outcome"])


if __name__ == "__main__":
    unittest.main()
