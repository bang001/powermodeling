"""Synthetic HBM candidate constraints, never hardware measurements."""

import copy
import unittest

from powermodeling.analysis import analyze_trial, summarize
from powermodeling.cli import parser
from test_analysis import empirical_record


def hbm_record(fraction=0.82, *, blocks=80, power=100, repeat=0, memory=1593, gfx=1110):
    peak = 2 * memory * 1e6 * 5120 / 8
    record = empirical_record("GPU-hbm-bandwidth-synthetic", gfx, memory=memory,
                              workload="hbm", throughput=fraction * peak,
                              blocks=blocks, power=power, repeat=repeat)
    record["cuda_device"] = {"uuid": record["config"]["gpu_uuid"], "memory_bus_width_bits": 5120,
                             "nominal_max_memory_clock_khz": 2619000}
    return record


def candidates():
    return [hbm_record(fraction, blocks=blocks, power=power, repeat=i)
            for fraction, blocks, power in ((0.82, 80, 100), (1.0, 160, 150))
            for i in range(3)]


class HbmBandwidthTests(unittest.TestCase):
    def test_low_energy_82_percent_candidate_is_not_excluded_by_observed_95_percent(self):
        summary = summarize(candidates())
        for key in ("within_clock_best", "cross_clock_best", "within_clock_best_total_energy",
                    "cross_clock_best_total_energy", "verified_target_within_clock_best",
                    "verified_target_cross_clock_best"):
            self.assertEqual(summary[key][0]["config"]["blocks"], 80, key)
        total = next(p for p in summary["empirical_gpu_energy_optima"] if p["objective"] == "total")
        self.assertEqual(total["config"]["blocks"], 80)
        recommendation = next(p for p in summary["evaluation"]["components"][0]["recommendations"]
                              if p["objective"] == "total")
        self.assertEqual(recommendation["group_id"], total["best_group_id"])
        self.assertEqual(summary["selection_policy"]["hbm_bandwidth_fraction"], 0.8)

    def test_current_memory_clock_sets_theory_and_graphics_clock_does_not(self):
        trial = analyze_trial(hbm_record())
        evidence = trial["hbm_bandwidth"]
        self.assertAlmostEqual(evidence["theoretical_bytes_s"], 2039.04e9)
        self.assertAlmostEqual(evidence["fraction_of_theoretical"], 0.82)
        self.assertEqual(evidence["clock_source"], "matching_measure_window_memory_clock_mhz")
        self.assertFalse(evidence["physical_dram_utilization_measured"])
        lower_clock = analyze_trial(hbm_record(memory=1215))
        self.assertAlmostEqual(lower_clock["hbm_bandwidth"]["theoretical_bytes_s"], 1555.2e9)

    def test_missing_one_valid_repeat_bus_metadata_cannot_be_averaged_away(self):
        records = candidates()
        records[0]["cuda_device"].pop("memory_bus_width_bits")
        summary = summarize(records)
        group = next(g for g in summary["groups"] if g["config"]["blocks"] == 80)
        self.assertEqual(group["valid_repeats"], 3)
        self.assertFalse(group["hbm_bandwidth"]["eligible"])
        self.assertIn("missing_or_invalid_memory_bus_width_bits", group["hbm_bandwidth"]["reasons"])
        self.assertEqual(summary["within_clock_best_total_energy"][0]["config"]["blocks"], 160)

    def test_missing_selected_clock_sample_is_not_replaced_by_requested_or_nominal_clock(self):
        record = hbm_record()
        record["phases"]["measure"]["samples"][20].pop("memory_clock_mhz")
        trial = analyze_trial(record)
        self.assertFalse(trial["hbm_bandwidth"]["eligible"])
        self.assertIn("incomplete_measure_window_memory_clock_telemetry", trial["hbm_bandwidth"]["reasons"])

    def test_unknown_and_mig_metadata_keep_raw_energy_but_do_not_qualify(self):
        for change in ("missing_bus", "mig", "tiny_bandwidth"):
            record = hbm_record(fraction=.82)
            if change == "missing_bus": record.pop("cuda_device")
            if change == "mig": record["device"]["mig_mode"] = {"current": 1}
            if change == "tiny_bandwidth":
                record["benchmark"]["logical_bytes"] = 1200
                for epoch in record["benchmark"]["measure_epochs"]: epoch["logical_bytes"] = 100
            trial = analyze_trial(record)
            with self.subTest(change=change):
                self.assertIsNotNone(trial["total_pj_per_logical_bit"])
                self.assertFalse(trial["hbm_bandwidth"]["eligible"])

    def test_logical_rate_above_interface_ceiling_remains_eligible_with_warning(self):
        record = hbm_record(fraction=1.01)
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertTrue(trial["target_verified"])
        self.assertTrue(trial["hbm_bandwidth"]["eligible"])
        self.assertFalse(trial["hbm_bandwidth"]["physical_dram_utilization_measured"])
        warning = "logical_rate_above_interface_ceiling_cache_reuse_or_count_review"
        self.assertIn(warning, trial["hbm_bandwidth"]["warnings"])
        records = [hbm_record(fraction=1.01, repeat=i) for i in range(3)]
        group = summarize(records)["groups"][0]
        self.assertTrue(group["hbm_bandwidth"]["eligible"])
        self.assertIn(warning, group["hbm_bandwidth"]["warnings"])

    def test_fraction_boundary_policy_and_cli_are_separate_from_observed_threshold(self):
        for fraction, expected in ((.799, False), (.8, True)):
            evidence = analyze_trial(hbm_record(fraction))["hbm_bandwidth"]
            self.assertEqual(evidence["eligible"], expected)
        summary = summarize(candidates(), throughput_fraction=.99, hbm_bandwidth_fraction=.85)
        self.assertEqual(summary["within_clock_best_total_energy"][0]["config"]["blocks"], 160)
        self.assertEqual(summary["selection_policy"]["throughput_fraction"], .99)
        self.assertEqual(summary["selection_policy"]["hbm_bandwidth_fraction"], .85)
        for invalid in (0, -1, 1.01, True, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize([], hbm_bandwidth_fraction=invalid)
        args = parser().parse_args(["analyze", "--input", "raw", "--output", "out", "--hbm-bandwidth-fraction", ".85"])
        self.assertEqual(args.hbm_bandwidth_fraction, .85)

    def test_cache_policy_and_index_math_split_contract_and_non_cg_hbm_is_diagnostic(self):
        records = []
        for cache, index in (("cg", "uint64"), ("cg", "uint32"), ("ca", "uint32"), ("cs", "uint32")):
            record = hbm_record()
            record["trial_id"] += cache + index
            record["benchmark"].update(read_cache_policy=cache, memory_read_index_math=index)
            records.append(record)
        trials = [analyze_trial(r) for r in records]
        contracts = [t["experiment_contract"] for t in trials]
        self.assertEqual(len({str(c) for c in contracts}), 4)
        self.assertEqual(len(summarize(records)["groups"]), 4)
        for trial in trials[2:]:
            self.assertFalse(trial["energy_peak_population_eligible"])
            self.assertFalse(trial["hbm_bandwidth"]["eligible"])
            self.assertIn("hbm_read_cache_policy_requires_cg", trial["hbm_bandwidth"]["reasons"])

    def test_non_cg_diagnostics_preserve_complete_group_bandwidth_numbers(self):
        records = [hbm_record(repeat=i) for i in range(3)]
        for record in records:
            record["benchmark"]["read_cache_policy"] = "cs"
            record["experiment_role"] = "diagnostic"
        group = summarize(records)["groups"][0]
        self.assertFalse(group["hbm_bandwidth"]["eligible"])
        self.assertTrue(group["hbm_bandwidth"]["evidence_complete"])
        self.assertAlmostEqual(group["hbm_bandwidth"]["fraction_of_theoretical"], .82)
        self.assertAlmostEqual(group["hbm_bandwidth"]["theoretical_bytes_s"], 2039.04e9)

    def test_unsupported_mig_query_is_distinct_from_unknown_mig_mapping(self):
        record = hbm_record()
        record["device"]["mig_mode"] = None
        record["device"]["errors"] = {"mig_mode": {"code": 3, "type": "NVMLError_NotSupported"}}
        self.assertTrue(analyze_trial(record)["hbm_bandwidth"]["eligible"])
        record["device"]["errors"]["mig_mode"] = {"code": 999, "type": "NVMLError_Unknown"}
        self.assertFalse(analyze_trial(record)["hbm_bandwidth"]["eligible"])

    def test_below_threshold_1110_reference_keeps_valid_energy_comparison(self):
        records = []
        for gfx, fraction, blocks, power in ((1110, .70, 80, 100), (1110, .69, 160, 110),
                                             (1200, .82, 80, 150), (1200, .95, 160, 200)):
            for repeat in range(3):
                record = hbm_record(fraction, gfx=gfx, blocks=blocks, power=power, repeat=repeat)
                if gfx == 1200:
                    record["config"]["clock_selection_reasons"] = ["advertised_default_fixed_pair"]
                records.append(record)
        component = summarize(records)["evaluation"]["components"][0]
        recommendation = next(r for r in component["recommendations"] if r["objective"] == "total")
        comparison = recommendation["exact_1110_comparison"]
        self.assertEqual(comparison["status"], "available")
        self.assertFalse(comparison["reference_hbm_bandwidth"]["eligible"])
        self.assertIn("below_hbm_theoretical_bandwidth_fraction", comparison["reference_hbm_bandwidth"]["reasons"])


if __name__ == "__main__":
    unittest.main()
