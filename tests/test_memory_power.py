"""Memory-scope sensor fixtures are synthetic, never GPU measurements."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from powermodeling.analysis import analyze_trial, summarize, write_summary
from test_analysis import synthetic_trial


def memory_trial(repeat=0, source="memory_power_instant_w", active=30, idle=10):
    record = synthetic_trial("hbm", throughput=1e12, repeat=repeat)
    for name, phase in record["phases"].items():
        for sample in phase["samples"]:
            sample.update(memory_power_w=active if name == "measure" else idle,
                          memory_power_source=source)
    return record


class MemoryPowerTests(unittest.TestCase):
    def sensor(self, record):
        trial = analyze_trial(record)
        self.assertTrue("hbm_memory_power" in trial, "HBM sensor result missing")
        return trial, trial["hbm_memory_power"]

    def test_memory_energy_and_logical_bit_cost_use_same_completed_epochs(self):
        trial, sensor = self.sensor(memory_trial())
        self.assertEqual(sensor["status"], "available")
        self.assertEqual(sensor["scope"], "memory")
        self.assertEqual(sensor["energy_j"], 240)
        self.assertEqual(sensor["incremental_energy_j"], 160)
        self.assertEqual(sensor["pj_per_logical_bit"], 3.75)
        self.assertEqual(sensor["incremental_pj_per_logical_bit"], 2.5)
        self.assertTrue(sensor["measurement_valid"])
        self.assertTrue(sensor["normalization_valid"])
        self.assertTrue(sensor["incremental_valid"])
        self.assertEqual(trial["total_energy_j"], 1200)
        self.assertEqual(trial["board_power_w"], 150)

    def test_absent_sensor_is_unavailable_without_invalidating_board(self):
        trial, sensor = self.sensor(synthetic_trial("hbm"))
        self.assertTrue(trial["valid"])
        self.assertEqual(sensor["status"], "unavailable")
        self.assertFalse(sensor["measurement_valid"])
        self.assertIsNone(sensor["energy_j"])

    def test_non_hbm_does_not_report_hbm_sensor(self):
        record = memory_trial()
        record["workload"] = "l2"
        trial, sensor = self.sensor(record)
        self.assertIsNone(sensor)

    def test_bad_memory_readings_do_not_invalidate_board(self):
        for defect in ("missing", "negative", "error", "unknown", "mixed"):
            record = memory_trial()
            sample = record["phases"]["measure"]["samples"][20]
            if defect == "missing": sample["memory_power_w"] = None
            elif defect == "negative": sample["memory_power_w"] = -1
            elif defect == "error": sample["errors"] = {"memory_power_instant_w": {"message": "failed"}}
            elif defect == "unknown": sample["memory_power_source"] = "unknown"
            elif defect == "mixed": sample["memory_power_source"] = "memory_power_average_w"
            with self.subTest(defect=defect):
                trial, sensor = self.sensor(record)
                self.assertTrue(trial["valid"], trial["issues"])
                self.assertEqual(sensor["status"], "invalid")
                self.assertFalse(sensor["measurement_valid"])
                self.assertIsNone(sensor["energy_j"])
                self.assertIsNone(sensor["pj_per_logical_bit"])

    def test_source_of_interpolation_boundary_is_qualified(self):
        record = memory_trial()
        record["benchmark"].pop("measure_epochs")
        record["phases"]["measure"]["samples"][8]["memory_power_source"] = "memory_power_average_w"
        trial = analyze_trial(record, {"trim_s": 2.1})
        self.assertIn("hbm_memory_power", trial)
        self.assertFalse(trial["hbm_memory_power"]["measurement_valid"])

    def test_different_idle_source_only_withholds_increment(self):
        record = memory_trial()
        for sample in record["phases"]["idle_pre"]["samples"]:
            sample["memory_power_source"] = "memory_power_average_w"
        _, sensor = self.sensor(record)
        self.assertTrue(sensor["measurement_valid"])
        self.assertTrue(sensor["normalization_valid"])
        self.assertFalse(sensor["incremental_valid"])
        self.assertIsNone(sensor["incremental_energy_j"])

    def test_measured_idle_power_remains_visible_when_board_idle_state_mismatches(self):
        record = memory_trial()
        for sample in record["phases"]["idle_pre"]["samples"]:
            sample["memory_clock_mhz"] = 500
        trial, sensor = self.sensor(record)
        self.assertTrue(trial["valid"])
        self.assertFalse(trial["operational_idle_increment_eligible"])
        self.assertEqual(sensor["idle_power_w"], 10)
        self.assertFalse(sensor["incremental_valid"])
        self.assertIsNone(sensor["incremental_energy_j"])

    def test_memory_idle_drift_withholds_increment_even_with_stable_board_idle(self):
        record = memory_trial()
        for sample in record["phases"]["idle_post"]["samples"]:
            sample["memory_power_w"] = 30
        trial, sensor = self.sensor(record)
        self.assertTrue(trial["operational_idle_increment_eligible"])
        self.assertEqual(sensor["energy_j"], 240)
        self.assertFalse(sensor["incremental_valid"])
        self.assertIsNone(sensor["incremental_energy_j"])
        self.assertIn("memory_idle_baseline_drift", sensor["incremental_issues"])

    def test_old_nvml_field_timestamps_invalidate_sensor_not_board(self):
        record = memory_trial()
        for sample in record["phases"]["measure"]["samples"]:
            sample.update(query_realtime_start_s=1730000000 + sample["t_s"],
                          query_realtime_end_s=1730000000 + sample["t_s"] + .01)
            sample["field_metadata"] = {"memory_power_instant_w": {
                "scope": "memory", "timestamp_clock": "unix_epoch_microseconds",
                "timestamp_us": 1730000006000000}}
        trial, sensor = self.sensor(record)
        self.assertTrue(trial["valid"])
        self.assertFalse(sensor["measurement_valid"])
        self.assertIn("stale_memory_sensor_timestamp", sensor["issues"])

    def test_fresh_average_timestamps_are_compared_in_epoch_clock(self):
        record = memory_trial(source="memory_power_average_w")
        for sample in record["phases"]["measure"]["samples"]:
            epoch = 1730000000 + sample["t_s"]
            sample.update(query_realtime_start_s=epoch, query_realtime_end_s=epoch + .01)
            sample["field_metadata"] = {"memory_power_average_w": {
                "scope": "memory", "timestamp_clock": "unix_epoch_microseconds",
                "timestamp_us": int((epoch - 1) * 1e6)}}
        _, sensor = self.sensor(record)
        self.assertTrue(sensor["measurement_valid"])
        self.assertTrue(sensor.get("freshness_verified"), "Comparable fresh timestamps must be identified")

    def test_legacy_sensor_explicitly_marks_freshness_unverified(self):
        _, sensor = self.sensor(memory_trial())
        self.assertTrue(sensor["measurement_valid"])
        self.assertFalse(sensor.get("freshness_verified", True))
        self.assertEqual(sensor.get("freshness_status"), "unverified")

    def test_legacy_work_count_does_not_normalize_sensor_energy(self):
        record = memory_trial()
        record["benchmark"].pop("measure_epochs")
        _, sensor = self.sensor(record)
        self.assertTrue(sensor["measurement_valid"])
        self.assertFalse(sensor["normalization_valid"])
        self.assertIsNone(sensor["pj_per_logical_bit"])

    def test_average_source_is_supported_without_gpu_name_gate(self):
        _, sensor = self.sensor(memory_trial(source="memory_power_average_w"))
        self.assertEqual(sensor["source"], "memory_power_average_w")
        self.assertEqual(sensor["semantics"], "one_second_average")
        self.assertTrue(sensor["measurement_valid"])

    def test_wrong_scope_metadata_invalidates_only_memory_sensor(self):
        record = memory_trial()
        for sample in record["phases"]["measure"]["samples"]:
            sample["field_metadata"] = {"memory_power_instant_w": {"scope": "gpu"}}
        trial, sensor = self.sensor(record)
        self.assertTrue(trial["valid"])
        self.assertFalse(sensor["measurement_valid"])

    def test_group_retains_sensor_repeat_counts_energy_and_intervals(self):
        records = [memory_trial(i, active=30+i) for i in range(4)]
        group = summarize(records)["groups"][0]
        self.assertIn("hbm_memory_power", group)
        sensor = group["hbm_memory_power"]
        self.assertEqual(sensor["observed_repeats"], 4)
        self.assertEqual(sensor["valid_repeats"], 4)
        self.assertEqual(sensor["normalized_repeats"], 4)
        self.assertEqual(sensor["energy_j"], 252)
        self.assertEqual(sensor["ci95"]["energy_j"], [240, 264])

    def test_group_partial_sensor_availability_is_explicit(self):
        records = [memory_trial(i) for i in range(4)]
        for sample in records[-1]["phases"]["measure"]["samples"]:
            sample["memory_power_w"] = None
        group = summarize(records)["groups"][0]
        self.assertIn("hbm_memory_power", group)
        sensor = group["hbm_memory_power"]
        self.assertEqual(sensor["status"], "partial")
        self.assertEqual(sensor["observed_repeats"], 4)
        self.assertEqual(sensor["valid_repeats"], 3)
        self.assertEqual(sensor["energy_j"], 240)

    def test_group_preserves_measured_idle_without_qualified_increment(self):
        records = [memory_trial(i) for i in range(3)]
        for record in records:
            for sample in record["phases"]["idle_pre"]["samples"]:
                sample["memory_clock_mhz"] = 500
        group = summarize(records)["groups"][0]
        self.assertIn("hbm_memory_power", group)
        sensor = group["hbm_memory_power"]
        self.assertEqual(sensor["idle_power_w"], 10)
        self.assertEqual(sensor["incremental_repeats"], 0)

    def test_group_never_pools_instant_and_average(self):
        records = [memory_trial(i, source="memory_power_instant_w" if i < 2 else "memory_power_average_w") for i in range(4)]
        group = summarize(records)["groups"][0]
        self.assertIn("hbm_memory_power", group)
        sensor = group["hbm_memory_power"]
        self.assertEqual(sensor["status"], "invalid")
        self.assertIsNone(sensor["power_w"])
        self.assertIsNone(sensor["energy_j"])
        self.assertIsNone(group["memory_rail_power_w"])
        self.assertEqual(group["board_power_w"], 150)

    def test_sensor_coverage_includes_observed_board_invalid_repeats(self):
        records = [memory_trial(i) for i in range(2)]
        records[1]["phases"]["measure"]["samples"][20]["power_w"] = None
        summary = summarize(records)
        sensor = summary["groups"][0]["hbm_memory_power"]
        self.assertEqual(sensor["observed_repeats"], 2)
        self.assertEqual(sensor["valid_repeats"], 1)
        self.assertEqual(sensor["status"], "partial")
        self.assertTrue(summary["trials"][1]["hbm_memory_power"]["measurement_valid"])
        self.assertFalse(summary["trials"][1]["hbm_memory_power"]["normalization_valid"])

    def test_all_board_invalid_repeats_are_not_reported_as_unsupported_sensor(self):
        records = [memory_trial(i) for i in range(2)]
        for record in records:
            record["phases"]["measure"]["samples"][20]["power_w"] = None
        sensor = summarize(records)["groups"][0]["hbm_memory_power"]
        self.assertEqual(sensor["observed_repeats"], 2)
        self.assertEqual(sensor["valid_repeats"], 0)
        self.assertEqual(sensor["status"], "invalid")
        self.assertIn("board_invalid_repeats_excluded", sensor["issues"])

    def test_csv_exports_sensor_energy_and_provenance(self):
        summary = summarize([memory_trial(i) for i in range(3)])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_summary(summary, directory)
            with Path(paths["trials_csv"]).open() as stream:
                row = next(csv.DictReader(stream))
            self.assertIn("hbm_memory_power", row)
            self.assertEqual(json.loads(row["hbm_memory_power"])["energy_j"], 240)
            self.assertEqual(float(row["memory_rail_energy_j"]), 240)


if __name__ == "__main__":
    unittest.main()
