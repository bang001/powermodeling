import copy
import json
import itertools
import tempfile
import unittest
from pathlib import Path

from powermodeling.analysis import AnalysisPolicy, analyze_trial, summarize, write_summary

_TRIAL_SERIAL = itertools.count()


def synthetic_trial(workload="tensor", active_power=150.0, throughput=1e12, repeat=0):
    phases = {}
    accumulated = 0.0
    for name, start, end, power in (("idle_pre", 0, 6, 50.0), ("measure", 6, 18, active_power), ("idle_post", 18, 24, 50.0)):
        samples = []
        for index in range(int((end - start) * 4) + 1):
            t = start + index / 4
            samples.append({"t_s": t, "power_w": power, "energy_mj": (accumulated + power * (t - start)) * 1000,
                            "graphics_clock_mhz": 1200, "memory_clock_mhz": 1000, "temperature_c": 50,
                            "throttle_reasons": 0, "compute_processes": [], "graphics_processes": []})
        phases[name] = {"start_s": start, "end_s": end, "samples": samples}
        accumulated += power * (end - start)
    return {"trial_id": f"synthetic-{repeat}-{next(_TRIAL_SERIAL)}", "workload": workload, "status": "complete",
            "config": {"gpu_uuid": "GPU-test", "graphics_clock_mhz": 1200, "memory_clock_mhz": 1000,
                       "blocks": 80, "stride_bytes": 32, "repeat": repeat},
            "benchmark": {"duration_s": 12, "operations": throughput * 12, "logical_bytes": throughput * 12,
                          "measure_epochs": [{"start_s": start, "end_s": start + 1, "operations": throughput,
                                              "logical_bytes": throughput, "counts_exact": True} for start in range(6, 18)]},
            "phases": phases, "device": {"name": "Synthetic, not measured hardware", "power_limit_w": 400}}


def verified_profile(record):
    from test_ncu_validation import pass_fixture
    binding, evidence = pass_fixture(record["workload"])
    record["condition_id"] = binding["condition_id"]
    record["provenance"] = binding["provenance"]
    record["config"].update(binding["config"])
    record["benchmark"].update({key: value for key, value in binding["benchmark"].items()
                                 if key not in ("logical_bytes", "kernel_launches")})
    for phase in record["phases"].values():
        for sample in phase["samples"]:
            sample["memory_clock_mhz"] = 1593
    record["samples"] = [sample for phase in record["phases"].values() for sample in phase["samples"]]
    record["validation"] = {"profiler_evidence": evidence}
    return record


class AnalysisTests(unittest.TestCase):
    def test_counter_units_baseline_energy_and_headline_limit(self):
        trial = analyze_trial(synthetic_trial())
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertEqual(trial["duration_s"], 8)
        self.assertEqual(trial["total_energy_j"], 1200)
        self.assertEqual(trial["idle_energy_j"], 400)
        self.assertEqual(trial["incremental_energy_j"], 800)
        self.assertEqual(trial["pj_per_op"], 100)
        self.assertEqual(trial["idle_fraction_of_measured_power"], 1 / 3)
        self.assertEqual(trial["idle_fraction_of_power_limit"], 0.125)
        self.assertIsNone(trial["memory_rail_power_w"])
        self.assertFalse(trial["target_verified"])

    def test_counter_missing_uses_integrated_power_and_preserves_unavailable(self):
        record = synthetic_trial("l2")
        record["benchmark"].pop("operations")
        for epoch in record["benchmark"]["measure_epochs"]:
            epoch["operations"] = 0
        for phase in record["phases"].values():
            for sample in phase["samples"]:
                sample.pop("energy_mj")
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertEqual(trial["energy_source"], "power_trapezoid")
        self.assertIsNone(trial["pj_per_op"])
        self.assertEqual(trial["pj_per_logical_byte"], 100)

    def test_uint64_counter_delta_does_not_lose_integer_precision(self):
        record = synthetic_trial()
        for phase in record["phases"].values():
            for sample in phase["samples"]:
                sample["energy_mj"] = int(sample["energy_mj"]) + 2**60
        trial = analyze_trial(record)
        self.assertEqual(trial["energy_counter_j"], 1200)
        self.assertEqual(trial["incremental_energy_j"], 800)

    def test_counter_reset_falls_back_and_disagreement_is_invalid(self):
        record = synthetic_trial()
        for sample in record["phases"]["measure"]["samples"]:
            if sample["t_s"] >= 12:
                sample["energy_mj"] -= 1000000
        trial = analyze_trial(record)
        self.assertEqual(trial["energy_source"], "power_trapezoid")
        self.assertIn("energy_counter_reset:measure", trial["warnings"])
        record = synthetic_trial()
        for sample in record["phases"]["measure"]["samples"]:
            sample["energy_mj"] *= 2
        trial = analyze_trial(record)
        self.assertFalse(trial["valid"])
        self.assertIn("energy_counter_disagreement:measure", trial["issues"])

    def test_short_plateau_missing_baseline_and_gap_rejected(self):
        record = synthetic_trial()
        record["phases"]["measure"]["end_s"] = 12
        record["phases"].pop("idle_post")
        trial = analyze_trial(record)
        self.assertFalse(trial["valid"])
        self.assertIn("short_plateau:measure", trial["issues"])
        self.assertIn("missing_idle_baseline", trial["issues"])
        record = synthetic_trial()
        record["phases"]["measure"]["samples"] = [s for s in record["phases"]["measure"]["samples"] if not 10 < s["t_s"] < 14]
        trial = analyze_trial(record)
        self.assertIn("telemetry_gap:measure", trial["issues"])

    def test_clock_throttling_and_interference_rejected(self):
        record = synthetic_trial()
        record["quality"] = {"other_compute_processes": [9999], "mig_active": True}
        for sample in record["phases"]["measure"]["samples"]:
            sample["graphics_clock_mhz"] = 900
            sample["throttle_reasons"] = "0x4"
        trial = analyze_trial(record)
        self.assertIn("active_throttling", trial["issues"])
        self.assertIn("clock_mismatch:graphics_clock_mhz", trial["issues"])
        self.assertIn("other_compute_processes", trial["issues"])
        self.assertIn("mig_active", trial["issues"])

    def test_configured_app_clock_reason_is_not_active_throttling(self):
        record = synthetic_trial()
        for sample in record["phases"]["measure"]["samples"]:
            sample["throttle_reasons"] = 2
        self.assertTrue(analyze_trial(record)["valid"])

    def test_process_filter_mps_and_mig_and_sample_power_limit(self):
        record = synthetic_trial()
        record["quality"] = {"measurement_pid": 123}
        for sample in record["phases"]["measure"]["samples"]:
            sample["compute_processes"] = [{"pid": 123}]
            sample["power_limit_w"] = 300
        self.assertTrue(analyze_trial(record)["valid"])
        self.assertEqual(analyze_trial(record)["power_limit_w"], 300)
        record["phases"]["measure"]["samples"][15]["compute_processes"].append({"pid": 456})
        record["device"]["mig_mode"] = {"current": 1}
        record["provenance"] = {"mps_environment": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"}}
        trial = analyze_trial(record)
        self.assertIn("other_gpu_processes", trial["issues"])
        self.assertIn("mig_active", trial["issues"])
        self.assertIn("mps_environment", trial["issues"])

    def test_gemm_theoretical_peak_uses_actual_sms_and_clock(self):
        record = synthetic_trial("gemm", throughput=200e12)
        record["cuda_device"] = {"compute_capability": [8, 0], "sm_count": 108}
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertAlmostEqual(trial["tensor_peak_tflops_at_achieved_clock"], 265.4208)
        self.assertAlmostEqual(trial["tensor_utilization_vs_dense_clock_peak"], 200 / 265.4208)
        self.assertEqual(trial["tensor_peak_clock_source"],"graphics_clock_mhz")
        self.assertTrue(any("graphics_clock_proxy" in warning for warning in trial["warnings"]))

    def test_tensor_peak_prefers_observed_sm_domain_over_graphics_clock(self):
        record=synthetic_trial("gemm",throughput=200e12)
        record["cuda_device"]={"compute_capability":[8,0],"sm_count":108}
        for sample in record["phases"]["measure"]["samples"]:
            sample["sm_clock_mhz"]=1000
        trial=analyze_trial(record)
        self.assertAlmostEqual(trial["tensor_peak_tflops_at_achieved_clock"],221.184)
        self.assertEqual(trial["tensor_peak_clock_source"],"sm_clock_mhz")
        self.assertEqual(trial["graphics_clock_mhz"],1200)
        self.assertEqual(trial["sm_clock_mhz"],1000)
        self.assertFalse(any("graphics_clock_proxy" in warning for warning in trial["warnings"]))

    def test_legacy_whole_run_host_rate_is_explicitly_qualified_estimate(self):
        record = synthetic_trial()
        record["benchmark"].pop("measure_epochs")
        record["benchmark"]["host_duration_s"] = 12.2
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertAlmostEqual(trial["pj_per_op"], 100 * 12.2 / 12)
        self.assertFalse(trial["count_energy_time_alignment_exact"])
        self.assertEqual(trial["energy_per_work_kind"], "stationary whole-run-rate estimate")
        record["benchmark"]["host_duration_s"] = 15
        self.assertIn("host_device_duration_disagreement", analyze_trial(record)["issues"])

    def test_linear_baseline_drift_interpolates_pre_post(self):
        record = synthetic_trial()
        for phase in record["phases"].values():
            for sample in phase["samples"]:
                sample.pop("energy_mj")
        for sample in record["phases"]["idle_post"]["samples"]:
            sample["power_w"] = 54
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertEqual(trial["idle_power_w"], 52)
        self.assertEqual(trial["incremental_power_w"], 98)

    def test_rail_increment_is_only_available_with_measured_baselines(self):
        record = synthetic_trial("hbm")
        record["validation"] = {"memory_target_verified": True, "profiler_evidence": "synthetic fixture"}
        for name, phase in record["phases"].items():
            for sample in phase["samples"]:
                sample["memory_power_w"] = 30 if name == "measure" else 10
        trial = analyze_trial(record)
        self.assertEqual(trial["memory_rail_incremental_power_w"], 20)
        self.assertEqual(trial["memory_rail_incremental_energy_j"], 160)
        self.assertFalse(trial["target_verified"], "A caller's boolean is not numeric counter evidence")

    def test_repeated_configuration_medians_not_lucky_minima(self):
        records = []
        for index, power in enumerate((151, 150, 149)):
            records.append(synthetic_trial(active_power=power, throughput=100, repeat=index))
        for index in range(3):
            record = synthetic_trial(active_power=90, throughput=80, repeat=index + 3)
            record["config"]["blocks"] = 40
            records.append(record)
        for index, power in enumerate((140, 141, 350)):
            record = synthetic_trial(active_power=power, throughput=96, repeat=index + 6)
            record["config"]["blocks"] = 60
            records.append(record)
        summary = summarize(records)
        self.assertEqual(len(summary["groups"]), 3)
        best = summary["within_clock_best"][0]
        self.assertEqual(best["config"]["blocks"], 60)
        self.assertEqual(best["eligible_groups"], 2)
        self.assertAlmostEqual(best["metric_value"], 91 / 96 * 1e12)
        self.assertEqual(summary["verified_target_within_clock_best"], [])

    def test_total_and_incremental_energy_objectives_can_select_different_groups(self):
        records = []
        for index in range(3):
            records.append(synthetic_trial(active_power=150, throughput=100, repeat=index))
            candidate = synthetic_trial(active_power=145.5, throughput=96, repeat=index + 3)
            candidate["config"]["blocks"] = 60
            records.append(candidate)
        summary = summarize(records)
        self.assertEqual(summary["within_clock_best"][0]["config"]["blocks"], 60)
        self.assertEqual(summary["within_clock_best_total_energy"][0]["config"]["blocks"], 80)
        self.assertEqual(summary["cross_clock_best_total_energy"][0]["metric"], "total_pj_per_op")

    def test_pareto_dominance_retains_low_power_tradeoff_and_excludes_dominated(self):
        records = []
        for blocks, power, throughput in ((80, 150, 100), (60, 160, 96), (40, 90, 80)):
            for index in range(3):
                candidate = synthetic_trial(active_power=power, throughput=throughput, repeat=index)
                candidate["config"]["blocks"] = blocks
                records.append(candidate)
        summary = summarize(records)
        frontier = [result for result in summary["pareto_frontiers"] if result["scope"] == "within_clock"][0]
        self.assertEqual(frontier["candidate_groups"], 3)
        self.assertEqual([point["config"]["blocks"] for point in frontier["frontier"]], [40, 80])

    def test_latency_and_control_are_valid_diagnostics_and_excluded_from_optimization(self):
        records = []
        for index in range(3):
            control = synthetic_trial("control", active_power=70, repeat=index)
            control["benchmark"].update(operations=0, logical_bytes=0)
            for epoch in control["benchmark"]["measure_epochs"]:
                epoch.update(operations=0, logical_bytes=0)
            latency = synthetic_trial("l2_latency", active_power=80, repeat=index + 3)
            latency["benchmark"]["latency_probe"] = {"enabled": True, "per_sm": {"0": {"loads": 1200, "cycles": 12000, "cycles_per_access": 10}}}
            records.extend((control, latency, synthetic_trial(repeat=index + 6)))
        summary = summarize(records)
        for trial in summary["trials"]:
            self.assertTrue(trial["valid"], trial["issues"])
        latency_trial = next(t for t in summary["trials"] if t["workload"] == "l2_latency")
        self.assertTrue(latency_trial["diagnostic_only"])
        self.assertIsNone(latency_trial["pj_per_op"])
        self.assertEqual(latency_trial["latency_diagnostics"]["latency_probe"]["per_sm"]["0"]["cycles_per_access"], 10)
        self.assertEqual(len(summary["within_clock_best"]), 1)
        self.assertEqual(len(summary["within_clock_best_total_energy"]), 1)
        self.assertTrue(all(f["stratum"]["workload"] == "tensor" for f in summary["pareto_frontiers"]))
        self.assertEqual(len(summary["active_control_associations"]), 1)
        self.assertEqual(summary["active_control_associations"][0]["control_groups"][0]["incremental_activation_power_w"], 20)

    def test_uncontrolled_clock_smoke_is_marked_exploratory(self):
        record = synthetic_trial()
        record["config"].update(graphics_clock_mhz=None, memory_clock_mhz=None)
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertFalse(trial["clock_comparison_controlled"])

    def test_idle_downclock_is_valid_operational_baseline_but_not_dynamic_attribution(self):
        record = synthetic_trial("l2")
        record["validation"] = {"memory_target_verified": True}
        record["phases"]["idle_pre"]["samples"] = [dict(s, graphics_clock_mhz=300, temperature_c=40) for s in record["phases"]["idle_pre"]["samples"]]
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertFalse(trial["baseline_clock_matched"])
        self.assertFalse(trial["baseline_temperature_matched"])
        self.assertFalse(trial["dynamic_attribution_eligible"])
        self.assertEqual(trial["active_minus_idle_state_deltas"]["idle_pre"]["graphics_clock_mhz"], 900)
        record = synthetic_trial("l2")
        record["validation"] = {"memory_target_verified": True}
        self.assertFalse(analyze_trial(record)["dynamic_attribution_eligible"], "Manual target flags cannot establish attribution eligibility")

    def test_offset_stride_and_repeat_keys_group_correctly(self):
        a, b, c = [synthetic_trial("l2", repeat=i) for i in range(3)]
        b["config"]["stride_bytes"] = 64
        c["config"]["offset_bytes"] = 1024
        self.assertEqual(len(summarize((a, b, c), min_repeats=1)["groups"]), 3)

    def test_invalid_repeats_do_not_win_and_json_has_no_nan(self):
        records = [synthetic_trial(repeat=i) for i in range(3)]
        bad = synthetic_trial(active_power=1, repeat=4)
        bad["quality"] = {"valid": False}
        records.append(bad)
        summary = summarize(records)
        self.assertEqual(summary["groups"][0]["valid_repeats"], 3)
        self.assertEqual(summary["within_clock_best"][0]["metric_value"], 100)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_summary(summary, directory)
            loaded = json.loads(Path(paths["summary_json"]).read_text())
            self.assertEqual(loaded["groups"][0]["pj_per_op"], 100)
            self.assertIn("memory_rail_power_w", Path(paths["trials_csv"]).read_text())

    def test_policy_is_configurable_and_bad_selection_fraction_rejected(self):
        trial = analyze_trial(synthetic_trial(), AnalysisPolicy(trim_s=1))
        self.assertEqual(trial["duration_s"], 10)
        with self.assertRaises(ValueError):
            summarize([], throughput_fraction=1.1)

    def test_matching_epoch_counts_exclude_boundary_throughput_transients(self):
        record = synthetic_trial(throughput=100e12)
        for epoch in record["benchmark"]["measure_epochs"]:
            if epoch["start_s"] < 8 or epoch["start_s"] >= 16:
                epoch["operations"] = 300e12
        record["benchmark"]["operations"] = sum(epoch["operations"] for epoch in record["benchmark"]["measure_epochs"])
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertEqual(trial["throughput_ops_s"], 100e12)
        self.assertEqual(trial["counted_measure_operations"], 800e12)
        self.assertEqual(trial["pj_per_op"], 1)
        self.assertTrue(trial["count_energy_time_alignment_exact"])

    def test_malformed_or_duplicated_epochs_are_rejected(self):
        record = synthetic_trial()
        record["benchmark"]["measure_epochs"].append(copy.deepcopy(record["benchmark"]["measure_epochs"][0]))
        self.assertIn("overlapping_measure_epochs", analyze_trial(record)["issues"])
        record = synthetic_trial()
        record["benchmark"]["measure_epochs"][4]["operations"] = float("nan")
        self.assertIn("invalid_measure_epoch_count:operations", analyze_trial(record)["issues"])

    def test_numeric_profile_is_recomputed_and_must_match_binary(self):
        from test_ncu_validation import change
        from powermodeling.validation import L2_READ_HITS
        record = verified_profile(synthetic_trial("l2"))
        trial = analyze_trial(record)
        self.assertTrue(trial["target_verified"], trial["ncu_assessment"]["reasons"])
        self.assertTrue(trial["verified_selection_eligible"])
        self.assertTrue(trial["dynamic_attribution_eligible"])
        change(record["validation"]["profiler_evidence"], L2_READ_HITS, 0)
        record["validation"]["memory_target_verified"] = True
        self.assertEqual(analyze_trial(record)["ncu_status"], "fail")
        record = verified_profile(synthetic_trial("l2"))
        record["provenance"]["benchmark_sha256"] = "b" * 64
        self.assertFalse(analyze_trial(record)["target_verified"])

    def test_verified_peak_coverage_reports_unprofiled_faster_conditions(self):
        records = []
        for index in range(3):
            records.append(verified_profile(synthetic_trial("l2", throughput=80, repeat=index)))
            unprofiled = verified_profile(synthetic_trial("l2", throughput=100, repeat=index))
            unprofiled["config"]["blocks"] = 160
            unprofiled["validation"] = {}
            records.append(unprofiled)
        summary = summarize(records)
        self.assertEqual(summary["verified_target_within_clock_best"], [])
        coverage = next(item for item in summary["verified_target_coverage"] if item["scope"] == "within_clock")
        self.assertAlmostEqual(coverage["verified_peak_coverage_fraction"], .8)
        self.assertEqual(coverage["verified_high_throughput_groups"], 0)
        self.assertIn("no_verified_condition", coverage["selection_status"])

    def test_unknown_process_inventory_is_not_empty_inventory(self):
        record = synthetic_trial()
        record["phases"]["idle_pre"]["samples"][10]["graphics_processes"] = None
        self.assertIn("missing_process_inventory:graphics_processes", analyze_trial(record)["issues"])

    def test_seed_and_binary_are_separate_conditions_and_duplicate_ids_not_repeats(self):
        a = synthetic_trial("hbm", repeat=0)
        b = synthetic_trial("hbm", repeat=1)
        a["config"]["seed"], b["config"]["seed"] = 2026, 2027
        self.assertEqual(len(summarize([a, b], min_repeats=1)["groups"]), 2)
        b["config"]["seed"] = 2026
        b["provenance"] = {"benchmark_sha256": "different-binary"}
        self.assertEqual(len(summarize([a, b], min_repeats=1)["groups"]), 2)
        summary = summarize([a, copy.deepcopy(a), copy.deepcopy(a)])
        self.assertEqual(summary["groups"][0]["valid_repeats"], 1)
        self.assertEqual(summary["within_clock_best"], [])
        self.assertEqual(len(summary["duplicate_trial_ids_ignored"]), 2)

    def test_uncontrolled_clocks_cannot_win_fair_comparison(self):
        records = [synthetic_trial(repeat=index) for index in range(3)]
        for record in records:
            record["config"].update(graphics_clock_mhz=None, memory_clock_mhz=None)
        summary = summarize(records)
        self.assertEqual(summary["within_clock_best"], [])
        self.assertEqual(summary["cross_clock_best"], [])
        self.assertEqual(len(summary["uncontrolled_clock_exploratory_best"]), 1)
        records[2]["phases"]["measure"]["samples"] = [dict(sample, graphics_clock_mhz=1250) for sample in records[2]["phases"]["measure"]["samples"]]
        self.assertEqual(len(summarize(records)["groups"]), 2)

    def test_nonfinite_policy_cannot_disable_quality_checks(self):
        with self.assertRaises(ValueError):
            AnalysisPolicy(max_sample_gap_s=float("nan"))

    def test_baseline_matching_requires_complete_stable_idle_clock_samples(self):
        record = verified_profile(synthetic_trial("l2"))
        record["phases"]["idle_pre"]["samples"][10]["graphics_clock_mhz"] = 300
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertFalse(trial["baseline_clock_matched"])
        self.assertFalse(trial["dynamic_attribution_eligible"])

    def test_missing_active_clock_samples_and_negative_idle_sensor_are_rejected(self):
        record = synthetic_trial()
        record["phases"]["measure"]["samples"][12]["graphics_clock_mhz"] = None
        self.assertIn("missing_clock_telemetry:graphics_clock_mhz", analyze_trial(record)["issues"])
        record = synthetic_trial()
        record["phases"]["idle_pre"]["samples"][12]["power_w"] = -1
        self.assertIn("negative_power:idle_pre", analyze_trial(record)["issues"])


if __name__ == "__main__":
    unittest.main()
