"""HBM sensor exports use synthetic values, never measurements from a GPU."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from powermodeling.analysis import summarize, write_summary
from powermodeling.evaluation import evaluate
from test_analysis import synthetic_trial


def sensor_summary(available=True, workload="hbm"):
    records = [synthetic_trial(workload, throughput=1e12, repeat=i) for i in range(3)]
    summary = summarize(records)
    group = summary["groups"][0]
    sensor = {
        "status": "available" if available else "unavailable", "scope": "memory",
        "source": "memory_power_average_w" if available else None,
        "semantics": "one_second_average" if available else None,
        "power_w": 30.0 if available else None, "energy_j": 240.0 if available else None,
        "idle_power_w": 10.0 if available else None,
        "incremental_power_w": 20.0 if available else None,
        "incremental_energy_j": 160.0 if available else None,
        "pj_per_logical_bit": 3.75 if available else None,
        "incremental_pj_per_logical_bit": 2.5 if available else None,
        "measurement_valid": available, "normalization_valid": available,
        "incremental_valid": available, "observed_repeats": 3,
        "valid_repeats": 3 if available else 0,
        "normalized_repeats": 3 if available else 0,
        "incremental_repeats": 3 if available else 0,
        "issues": [] if available else ["memory_power_unavailable"],
        "incremental_issues": [], "ci95": {"power_w": [30.0, 30.0]} if available else {},
    }
    group["hbm_memory_power"] = sensor if workload == "hbm" else None
    summary["evaluation"] = evaluate(summary, raw_records=records)
    return summary


class HbmMemoryReportingTests(unittest.TestCase):
    def test_sensor_power_energy_and_unit_cost_export_separately_from_board(self):
        summary = sensor_summary()
        point = summary["evaluation"]["components"][0]["points"][0]
        self.assertEqual(point["hbm_memory_power"]["power_w"], 30.0)
        self.assertEqual(point["energies"]["total"], 18.75)
        with tempfile.TemporaryDirectory() as directory:
            files = write_summary(summary, directory)
            with Path(files["hbm_memory_power_csv"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(float(row["power_w"]), 30.0)
            self.assertEqual(float(row["energy_j"]), 240.0)
            self.assertEqual(float(row["pj_per_logical_bit"]), 3.75)
            self.assertEqual(float(row["incremental_pj_per_logical_bit"]), 2.5)
            self.assertEqual(float(row["board_total_pj_per_logical_bit"]), 18.75)
            self.assertEqual(row["source"], "memory_power_average_w")
            self.assertEqual(row["valid_repeats"], "3")
            exported = json.loads(Path(files["evaluation_json"]).read_text())
            self.assertEqual(exported["components"][0]["points"][0]["hbm_memory_power"]["energy_j"], 240.0)
            html = Path(files["evaluation_html"]).read_text()
            self.assertIn('id="hbmMemorySection"', html)
            self.assertIn("HBM memory sensor", html)
            self.assertIn("pJ/logical bit", html)

    def test_unavailable_sensor_stays_visible_without_fabricated_zero(self):
        summary = sensor_summary(available=False)
        self.assertTrue(summary["trials"][0]["valid"])
        with tempfile.TemporaryDirectory() as directory:
            files = write_summary(summary, directory)
            with Path(files["hbm_memory_power_csv"]).open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "unavailable")
            self.assertEqual(row["power_w"], "")
            self.assertEqual(row["energy_j"], "")
            self.assertEqual(row["valid_repeats"], "0")
            self.assertIn("memory_power_unavailable", row["issues"])
            self.assertEqual(float(row["board_total_pj_per_logical_bit"]), 18.75)

    def test_non_hbm_report_has_no_hbm_sensor_export(self):
        with tempfile.TemporaryDirectory() as directory:
            files = write_summary(sensor_summary(workload="tensor"), directory)
            self.assertNotIn("hbm_memory_power_csv", files)


if __name__ == "__main__":
    unittest.main()
