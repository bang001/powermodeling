"""Requested payload, measured sector traffic, and experiment-role boundaries.

Synthetic counters are counterexamples to qualification rules, not measurements
of GPU energy or bandwidth. CLI guards need a binary; CUDA checks are opt-in.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from powermodeling.analysis import analyze_trial, summarize
from powermodeling.memory import coalesced_read_geometry
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.runner import capture_trial, validate_plan_execution
from powermodeling.validation import (DRAM_READ, DRAM_WRITE, L1_HITS, L1_REQUESTS,
                                     L1_SECTORS, L2_READ, L2_READ_HITS, L2_WRITE,
                                     assess_profile, validate_evidence)
from test_integration_review import FakeDevice, FakeProcess, FakeSampler
from test_memory_read_contract import DEVICE, memory_record
from test_ncu_validation import change, pass_fixture


ROOT = Path(__file__).resolve().parents[1]
CLOCKS = {"supported_pairs": [{"graphics_mhz": g, "memory_mhz": 1000}
                              for g in (900, 1110, 1200)],
          "default_applications_graphics_mhz": 1200,
          "default_applications_memory_mhz": 1000}
OPTIMUM_KEYS = ("within_clock_best", "cross_clock_best",
               "within_clock_best_total_energy", "cross_clock_best_total_energy",
               "uncontrolled_clock_exploratory_best", "verified_target_within_clock_best",
               "verified_target_cross_clock_best", "empirical_gpu_energy_optima",
               "empirical_gpu_overall_energy_optima", "exploratory_single_geometry_energy_optima",
               "exploratory_single_geometry_overall_energy_optima", "pareto_frontiers")


def coalescing_record(workload="hbm", *, stride=1, offset=0, ratio=1,
                      role="energy_characterization", repeat=0, region_bytes=4096, iterations=3):
    """One scalar load per lane, full warps, one replay, independent counter ratio."""
    record = memory_record(repeat=repeat)
    record["workload"] = workload
    if role is not None:
        record["experiment_role"] = role
    evidence = record["validation"]["profiler_evidence"]
    evidence["workload"] = workload
    total_bytes = region_bytes * 2 if workload == "l1" else 65536
    parameters = {"working_set_bytes": total_bytes, "stride_elements": stride,
                  "offset_bytes": offset, "access": "read", "iterations": iterations}
    record["config"].update(parameters)
    evidence["profile_provenance"]["parameters"].update(parameters)
    for benchmark in (record["benchmark"], evidence["profile_benchmark"]):
        benchmark.update(parameters, iterations_per_launch=iterations, workload=workload, memory_word_bytes=4,
                         stride_words=stride, lane_stride_bytes=stride * 4,
                         l1_bytes_per_block=region_bytes if workload == "l1" else 0)
        for counts in [benchmark, *benchmark.get("measure_epochs", [])]:
            counts["operations"] = counts["admitted_blocks"] * 32 * iterations
            counts["logical_bytes"] = counts["operations"] * 4
    logical = evidence["profile_benchmark"]["logical_bytes"]
    sectors = logical * ratio / 32
    for metric, value in ((L1_SECTORS, sectors), (L1_HITS, sectors if workload == "l1" else 0),
                          (L1_REQUESTS, 2 * iterations),
                          (L2_READ, 0 if workload == "l1" else sectors),
                          (L2_READ_HITS, sectors if workload == "l2" else 0),
                          (L2_WRITE, 0), (DRAM_READ, logical * ratio if workload == "hbm" else 0),
                          (DRAM_WRITE, 0)):
        change(evidence, metric, value)
    identifier = f"coalescing-{workload}-{role}-{stride}-{offset}-{ratio}-{region_bytes}"
    record.update(condition_id=identifier, trial_id=f"{identifier}-r{repeat}")
    evidence["condition_id"] = identifier
    return record


def memory_plan(workload="l2", parameters=None, *, role=None, study="diagnostic"):
    spec = {"workload": workload, "parameters": parameters or {}}
    if role is not None:
        spec["experiment_role"] = role
    config = {"study_design": study, "paired_reference": False, "experiments": [spec]}
    if study == "energy_sweep":
        config["clock_sweep"] = {"graphics_step_mhz": 90, "all_memory_clocks": True}
    else:
        config["clock_pairs"] = [{"graphics_mhz": 1200, "memory_mhz": 1000}]
    return expand_plan(config, DEVICE, CLOCKS)


class MemoryCoalescingTests(unittest.TestCase):
    def test_aligned_scalar_read_has_four_sectors_per_warp_not_four_cache_lines(self):
        for workload in ("l1", "l2", "hbm"):
            with self.subTest(workload=workload):
                record = coalescing_record(workload)
                result = validate_evidence(record, record["validation"]["profiler_evidence"])
                self.assertEqual(result["status"], "pass", result["reasons"])
                coalescing = result["memory_coalescing"]
                self.assertEqual(coalescing["status"], "pass", coalescing)
                self.assertTrue(coalescing["energy_eligible"])
                self.assertEqual(coalescing["observed_sector_bytes_per_logical_read_byte"], 1)
                self.assertEqual(coalescing["observed_sector_efficiency_pct"], 100)
                self.assertEqual(result["kernels"][0]["derived"]["l1_sectors_per_global_read_request"], 4)
                self.assertEqual(coalescing["counter_name"], L1_SECTORS if workload == "l1" else L2_READ)
                self.assertEqual(coalescing["energy_denominator_use"], "forbidden_separate_profiler_run")

    def test_stride_four_is_sixteen_byte_lane_spacing_and_twenty_five_percent_efficiency(self):
        for workload in ("l1", "l2", "hbm"):
            with self.subTest(workload=workload):
                record = coalescing_record(workload, stride=4, ratio=4)
                result = validate_evidence(record, record["validation"]["profiler_evidence"])
                # A valid target path must not be mistaken for efficient payload transfer.
                self.assertEqual(result["status"], "pass", result["reasons"])
                self.assertTrue(result["suitable_verified"])
                coalescing = result["memory_coalescing"]
                self.assertEqual(coalescing["status"], "fail")
                self.assertFalse(coalescing["energy_eligible"])
                self.assertEqual(coalescing["geometry"]["lane_stride_bytes"], 16)
                self.assertEqual(coalescing["requested_sector_efficiency_fraction"], .25)
                self.assertEqual(coalescing["observed_sector_efficiency_pct"], 25)
                self.assertEqual(result["kernels"][0]["derived"]["l1_sectors_per_global_read_request"], 16)
                trial = analyze_trial(record)
                self.assertTrue(trial["valid"], trial["issues"])
                self.assertFalse(trial["verified_selection_eligible"])

    def test_counter_proof_is_required_even_when_declared_addresses_are_coalesced(self):
        record = coalescing_record(ratio=4)
        result = validate_evidence(record, record["validation"]["profiler_evidence"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["memory_coalescing"]["geometry"]["status"], "pass")
        self.assertEqual(result["memory_coalescing"]["status"], "fail")
        self.assertFalse(analyze_trial(record)["verified_selection_eligible"])
        summary = summarize([coalescing_record(ratio=4, repeat=i) for i in range(4)])
        for key in OPTIMUM_KEYS:
            self.assertEqual(summary[key], [], key)

    def test_alignment_is_thirty_two_bytes_and_wrap_regions_must_preserve_it(self):
        for offset, expected_efficiency in ((0, 1), (4, .8), (32, 1), (128, 1)):
            with self.subTest(offset=offset):
                geometry = coalesced_read_geometry({"threads": 32, "working_set_bytes": 4096,
                                                   "stride_elements": 1, "offset_bytes": offset}, "l2")
                self.assertEqual(geometry["requested_sector_efficiency_fraction"], expected_efficiency)
                self.assertEqual(geometry["status"], "fail" if offset == 4 else "pass")
        for region in (4, 124, 132):
            geometry = coalesced_read_geometry({"blocks": 2, "threads": 32,
                                               "working_set_bytes": 2 * region}, "l1")
            self.assertEqual(geometry["status"], "fail", geometry)
        geometry = coalesced_read_geometry({"blocks": 3, "threads": 32,
                                           "working_set_bytes": 3 * 128 + 4}, "l1")
        self.assertEqual(geometry["effective_region_bytes"], 128)
        self.assertEqual(geometry["status"], "pass")
        geometry = coalesced_read_geometry({"blocks": 2, "threads": 32,
                                           "working_set_bytes": 2 * 132, "l1_bytes_per_block": 128}, "l1")
        self.assertEqual(geometry["status"], "fail")

    def test_inclusive_counter_tolerance_is_reported_without_clipping_efficiency(self):
        for ratio, expected in ((.949, "fail"), (.95, "pass"), (1, "pass"), (1.05, "pass"), (1.051, "fail")):
            with self.subTest(ratio=ratio):
                # Integer sector counts put 0.95 exactly at the documented boundary.
                record = coalescing_record(ratio=ratio, iterations=100)
                result = validate_evidence(record, record["validation"]["profiler_evidence"])
                self.assertEqual(result["memory_coalescing"]["status"], expected)
                self.assertAlmostEqual(result["memory_coalescing"]["observed_sector_efficiency_pct"], 100 / ratio)
                self.assertEqual(result["memory_coalescing"]["counter_ratio_bounds"], [.95, 1.05])

    def test_missing_or_wrong_counter_units_never_become_efficient_zero_traffic(self):
        for workload in ("l1", "l2", "hbm"):
            for mode in ("missing", "byte", "missing_unit"):
                with self.subTest(workload=workload, mode=mode):
                    record = coalescing_record(workload)
                    evidence = record["validation"]["profiler_evidence"]
                    counter = L1_SECTORS if workload == "l1" else L2_READ
                    change(evidence, counter, remove=mode == "missing",
                           unit="byte" if mode == "byte" else "" if mode == "missing_unit" else None)
                    coalescing = validate_evidence(record, evidence)["memory_coalescing"]
                    self.assertEqual(coalescing["status"], "inconclusive")
                    self.assertFalse(coalescing["energy_eligible"])
                    self.assertIsNone(coalescing["observed_sector_efficiency_pct"])
        record = coalescing_record()
        evidence = record["validation"]["profiler_evidence"]
        evidence["profile_provenance"]["benchmark_sha256"] = "f" * 64
        coalescing = validate_evidence(record, evidence)["memory_coalescing"]
        self.assertEqual(coalescing["status"], "inconclusive")
        self.assertFalse(coalescing["profile_binding_verified"])
        self.assertFalse(coalescing["energy_eligible"])

    def test_l1_cg_bypass_and_other_memory_directions_do_not_use_read_gate(self):
        record = coalescing_record("l2")
        evidence = record["validation"]["profiler_evidence"]
        change(evidence, L1_SECTORS, 0)
        self.assertEqual(validate_evidence(record, evidence)["memory_coalescing"]["status"], "pass")
        for workload, access in (("l2_latency", "read"), ("hbm", "write"), ("hbm", "copy")):
            _, evidence = pass_fixture(workload, access)
            coalescing = assess_profile(evidence)["memory_coalescing"]
            self.assertFalse(coalescing["applicable"])
            self.assertFalse(coalescing["energy_eligible"])

    def test_diagnostic_and_energy_roles_do_not_pool_or_supply_optima(self):
        records = [coalescing_record(role=role, repeat=i)
                   for role in ("energy_characterization", "diagnostic") for i in range(4)]
        summary = summarize(records)
        self.assertEqual(len(summary["groups"]), 2)
        self.assertEqual({g["experiment_role"] for g in summary["groups"]},
                         {"energy_characterization", "diagnostic"})
        diagnostic = next(g for g in summary["groups"] if g["experiment_role"] == "diagnostic")
        self.assertFalse(diagnostic["verified_selection_eligible"])
        diagnostic_summary = summarize([r for r in records if r["experiment_role"] == "diagnostic"])
        for key in OPTIMUM_KEYS:
            self.assertEqual(diagnostic_summary[key], [], key)
        for component in diagnostic_summary["evaluation"]["components"]:
            self.assertIsNone(component["observed_peak"])
            self.assertTrue(all(r["status"] == "diagnostic_only" for r in component["recommendations"]))
        historical = coalescing_record(role=None)
        for benchmark in (historical["benchmark"], historical["validation"]["profiler_evidence"]["profile_benchmark"]):
            for key in ("memory_word_bytes", "stride_words", "lane_stride_bytes"):
                benchmark.pop(key)
        trial = analyze_trial(historical)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertEqual(trial["experiment_role"], "legacy_unspecified")

    def test_word_alias_normalizes_identically_and_conflicting_input_is_rejected(self):
        old = memory_plan(parameters={"stride_elements": 4}, role="diagnostic")
        new = memory_plan(parameters={"stride_words": 4}, role="diagnostic")
        self.assertEqual(old, new)
        self.assertEqual(new["trials"][0]["parameters"]["stride_elements"], 4)
        command = benchmark_command("bench", new["trials"][0])
        self.assertEqual(command[command.index("--stride-words") + 1], "4")
        self.assertNotIn("--stride-elements", command)
        historical_trial = copy.deepcopy(new["trials"][0])
        historical_trial.pop("experiment_role")
        for profiling in (False, True):
            historical_command = benchmark_command("old-bench", historical_trial, profiling=profiling)
            self.assertEqual(historical_command[historical_command.index("--stride-elements") + 1], "4")
            self.assertNotIn("--stride-words", historical_command)
        for parameters in ({"stride_words": 1, "stride_elements": 1},
                           {"stride_words": 1, "stride_elements": 4}):
            with self.assertRaisesRegex(ValueError, "only one of stride_words or stride_elements"):
                memory_plan(parameters=parameters)
        with self.assertRaisesRegex(ValueError, "only one of stride_words or stride_elements"):
            expand_plan({"study_design": "diagnostic", "clock_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1000}], "experiments": [{"workload": "l2",
                        "parameters": {"stride_words": 1}, "grid": {"stride_elements": [1]}}]}, DEVICE)

    def test_energy_plan_rejects_sparse_or_misaligned_geometry_but_diagnostics_can_measure_it(self):
        cases = (("l2", {"stride_words": 4}), ("l2", {"offset_bytes": 4}),
                 ("l2", {"working_set_bytes": 132}),
                 ("l1", {"blocks": 2, "working_set_bytes": 264}))
        for workload, parameters in cases:
            with self.subTest(workload=workload, parameters=parameters):
                with self.assertRaisesRegex(ValueError, "requires coalesced read geometry"):
                    memory_plan(workload, parameters, role="energy_characterization")
                plan = memory_plan(workload, parameters, role="diagnostic")
                self.assertTrue(all(t["experiment_role"] == "diagnostic" for t in plan["trials"]))
        self.assertTrue(memory_plan(parameters={"offset_bytes": 32}, role="energy_characterization")["trials"])
        self.assertEqual(memory_plan()["trials"][0]["experiment_role"], "diagnostic")
        self.assertEqual(memory_plan(study="energy_sweep")["trials"][0]["experiment_role"], "energy_characterization")
        for role in (True, "energy", "", None):
            config = {"clock_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1000}],
                      "experiments": [{"workload": "l2", "experiment_role": role}]}
            with self.subTest(role=role), self.assertRaises(ValueError):
                expand_plan(config, DEVICE)

    def test_shipped_memory_energy_plans_preserve_aligned_full_warp_payload_on_all_gpus(self):
        for major, sm_count, l2 in ((7, 80, 6 * 1024**2), (8, 108, 40 * 1024**2), (9, 132, 50 * 1024**2)):
            device = {**DEVICE, "compute_capability_major": major, "sm_count": sm_count, "l2_bytes": l2}
            for name in ("memory-read.json", "dvfs.json", "saturation.json"):
                with self.subTest(gpu=major, config=name):
                    plan = expand_plan(json.loads((ROOT / "configs" / name).read_text()), device, CLOCKS)
                    reads = [t for t in plan["trials"] if t["workload"] in ("l1", "l2", "hbm")
                             and t["parameters"]["access"] == "read"]
                    self.assertTrue(reads)
                    for trial in reads:
                        self.assertEqual(trial["experiment_role"], "energy_characterization")
                        self.assertEqual(trial["parameters"]["stride_elements"], 1)
                        geometry = coalesced_read_geometry(trial["parameters"], trial["workload"])
                        self.assertEqual(geometry["status"], "pass", geometry)
                    if name == "memory-read.json":
                        self.assertEqual({row["geometry_conditions"] for row in plan["sweep_dimensions"]["by_workload"]}, {6})
            for name in ("locality.json", "cache-sector-diagnostics.json", "component-diagnostics.json", "memory-read-smoke.json"):
                plan = expand_plan(json.loads((ROOT / "configs" / name).read_text()), device, CLOCKS)
                memory_trials = [t for t in plan["trials"] if t["workload"] in ("l1", "l2", "l2_latency", "hbm")]
                self.assertTrue(memory_trials)
                self.assertTrue(all(t["experiment_role"] == "diagnostic" for t in memory_trials), name)

    def test_capture_keeps_experiment_role_outside_kernel_parameters(self):
        trial = memory_plan(role="diagnostic")["trials"][0]
        self.assertNotIn("experiment_role", trial["parameters"])
        self.assertNotIn("--experiment-role", benchmark_command("bench", trial))
        with patch("powermodeling.telemetry.Sampler", FakeSampler), \
                patch("powermodeling.runner.subprocess.Popen", FakeProcess), \
                patch("powermodeling.runner.platform.platform", return_value="synthetic-platform"):
            record = capture_trial("bench", trial, FakeDevice(uuid=DEVICE["uuid"]), DEVICE)
        self.assertEqual(record["experiment_role"], "diagnostic")

    def test_mutated_energy_plan_is_rejected_before_execution_and_legacy_plan_remains_replayable(self):
        plan = memory_plan(role="energy_characterization")
        plan["trials"][0]["parameters"] = copy.deepcopy(plan["trials"][0]["parameters"])
        plan["trials"][0]["parameters"]["stride_elements"] = 4
        with self.assertRaisesRegex(ValueError, "requires coalesced read geometry"):
            validate_plan_execution(plan)
        plan["trials"][0].pop("experiment_role")
        validate_plan_execution(plan)

    def test_redundant_raw_word_units_cannot_contradict_the_address_geometry(self):
        for field, invalid in (("memory_word_bytes", 16), ("stride_words", 4), ("lane_stride_bytes", 16)):
            record = coalescing_record()
            record["benchmark"][field] = invalid
            geometry = coalesced_read_geometry(record["benchmark"], "hbm")
            self.assertEqual(geometry["status"], "fail", geometry)
            result = validate_evidence(record, record["validation"]["profiler_evidence"])
            self.assertFalse(result["memory_coalescing"]["energy_eligible"])

    def test_valid_replay_cannot_override_inconsistent_geometry_in_the_energy_record(self):
        records = [memory_record(legacy=True, repeat=i) for i in range(4)]
        unchanged = analyze_trial(records[0])
        for record in records:
            # Historical records have no redundant word-unit metadata. The
            # replay remains coherent while the energy-run config contradicts
            # the scalar4B implementation; both geometries must qualify.
            evidence = record["validation"]["profiler_evidence"]
            for benchmark in (record["benchmark"], evidence["profile_benchmark"]):
                self.assertNotIn("memory_word_bytes", benchmark)
            record["config"]["memory_word_bytes"] = 16
            replay = validate_evidence(record, evidence)
            self.assertEqual(replay["status"], "pass", replay["reasons"])
            self.assertTrue(replay["memory_coalescing"]["energy_eligible"])
            trial = analyze_trial(record)
            self.assertTrue(trial["valid"], trial["issues"])
            self.assertEqual(trial["total_pj_per_logical_bit"], unchanged["total_pj_per_logical_bit"])
            self.assertEqual(trial["memory_access_geometry"]["status"], "fail")
            self.assertFalse(trial["memory_coalescing_energy_eligible"])
            self.assertFalse(trial["verified_selection_eligible"])
            self.assertTrue(any("memory_word_bytes" in reason for reason in trial["memory_coalescing"]["reasons"]))
        summary = summarize(records)
        for key in OPTIMUM_KEYS:
            self.assertEqual(summary[key], [], key)
        self.assertTrue(all(group["valid_repeats"] == 4 for group in summary["groups"]))
        self.assertTrue(all(component["observed_peak"] is None
                            for component in summary["evaluation"]["components"]))

    def test_role_is_part_of_the_plan_fingerprint_and_geometry_statistics(self):
        plan = expand_plan({"study_design": "diagnostic", "paired_reference": False,
                            "clock_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1000}],
                            "experiments": [{"workload": "l2", "experiment_role": role}
                                            for role in ("energy_characterization", "diagnostic")]}, DEVICE)
        self.assertEqual(len(plan["trials"]), 8)
        self.assertEqual(len({t["condition_id"] for t in plan["trials"]}), 2)
        self.assertEqual(len({t["trial_id"] for t in plan["trials"]}), 8)
        self.assertEqual(plan["sweep_dimensions"]["geometry_conditions"], 2)
        self.assertEqual(plan["sweep_dimensions"]["by_workload"][0]["geometry_conditions"], 2)
        self.assertEqual(plan["estimated_minimum_seconds"], 8 * (12 + 3 + 2 * 6))

@unittest.skipUnless(os.environ.get("POWERBENCH_CLI_TESTS") == "1", "compiled CLI guards opt-in")
class MemoryStrideCliTests(unittest.TestCase):
    def test_both_word_aliases_cannot_silently_override_one_another(self):
        executable = os.environ.get("POWERBENCH", str(ROOT / "build" / "powerbench"))
        for flags in (("--stride-words", "1", "--stride-elements", "4"),
                      ("--stride-elements", "1", "--stride-words", "1")):
            completed = subprocess.run([executable, "--workload", "l2", *flags],
                                       capture_output=True, text=True, timeout=10)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("choose only one of --stride-words and --stride-elements", completed.stderr)
            self.assertIn("32-bit word units", completed.stderr)
            self.assertFalse(any('"type":"device"' in line or '"type":"result"' in line
                                 for line in completed.stdout.splitlines()))


@unittest.skipUnless(os.environ.get("POWERBENCH_GPU_TESTS") == "1", "actual CUDA checks opt-in")
class MemoryStrideCudaTests(unittest.TestCase):
    def test_word_aliases_emit_equal_counts_addresses_and_explicit_byte_spacing(self):
        executable = os.environ.get("POWERBENCH", str(ROOT / "build" / "powerbench"))
        for stride in (1, 4):
            results = []
            for flag in ("--stride-words", "--stride-elements"):
                command = [executable, "--device", os.environ.get("POWERBENCH_DEVICE", "0"),
                           "--workload", "l2", "--seconds", ".1", "--warmup-seconds", "0",
                           "--idle-seconds", "0", "--blocks", "2", "--threads", "32",
                           "--working-set-bytes", "8192", "--iterations", "7",
                           "--batch-launches", "1", "--fixed-batches", "2", "--warmup-batches", "0",
                           flag, str(stride)]
                completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
                result = next(row for row in map(json.loads, completed.stdout.splitlines()) if row["type"] == "result")
                self.assertEqual(result["memory_word_bytes"], 4)
                self.assertEqual(result["stride_words"], stride)
                self.assertEqual(result["lane_stride_bytes"], 4 * stride)
                self.assertTrue(result["sanity"]["memory_read_sample_matches_reference"])
                self.assertEqual(result["operations"], 2 * 2 * 32 * 7)
                self.assertEqual(result["logical_bytes"], result["operations"] * 4)
                results.append(result)
            self.assertEqual(results[0]["checksum"], results[1]["checksum"])


if __name__ == "__main__":
    unittest.main()
