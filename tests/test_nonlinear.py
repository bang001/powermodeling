"""Synthetic energy records test units and rejection; no measured GPU numbers."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from powermodeling.analysis import analyze_trial, summarize
from powermodeling.nonlinear import (MATH_IMPLEMENTATION, NONLINEAR_WORKLOADS,
                                    ROW_WORKLOADS, RMS_EPSILON, count_issues,
                                    logical_bytes_per_element)
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.profiling import profile_command
from powermodeling.validation import SFU_INSTRUCTIONS, assess_profile, validate_evidence
from test_analysis import synthetic_trial
from test_integration_review import DEVICE
from test_ncu_validation import pass_fixture


def nonlinear_trial(workload="exp", width=128, blocks=80, threads=256, repeat=0):
    per_block = width * 2
    rate = blocks * per_block * 100
    record = synthetic_trial(workload=workload, throughput=rate, repeat=repeat)
    row_width = width if workload in ROW_WORKLOADS else 0
    b = record["benchmark"]
    b.update(elements=rate * 12, logical_bytes=rate * 12 * logical_bytes_per_element(workload),
             row_width=row_width, row_evaluations=rate * 12 / width if row_width else 0,
             blocks=blocks, threads=threads, working_set_bytes=blocks * per_block * 4,
             iterations_per_launch=1, admitted_blocks=blocks * 100 * 12,
             kernel_launches=1200, batch_launches=1, stride_elements=1, offset_bytes=0, access="read",
             input_precision="fp32", math_implementation=MATH_IMPLEMENTATION,
             nonlinear_input_distribution="synthetic bounded FP32 fixture",
             rms_epsilon=RMS_EPSILON if workload == "rmsnorm" else None,
             affine_gamma=workload == "rmsnorm", paired_reference_context_allocated=False,
             operation_unit="complete function output elements, not FLOPs",
             numerical_validation={"checked_values": 24},
             sanity={"numerical_validation_passed": True, "finite_output_sample": True,
                     "requested_sm_coverage_complete": True})
    for epoch in b["measure_epochs"]:
        epoch.update(elements=rate, logical_bytes=rate * logical_bytes_per_element(workload),
                     row_evaluations=rate / width if row_width else 0,
                     admitted_blocks=blocks * 100)
    record["config"].update(blocks=blocks, threads=threads, iterations=1,
                            working_set_bytes=b["working_set_bytes"])
    if row_width:
        record["config"]["row_width"] = row_width
    record["samples"] = [{"phase": name, **sample} for name, phase in record["phases"].items() for sample in phase["samples"]]
    return record


def nonlinear_evidence(record):
    _, evidence = pass_fixture()
    workload = record["workload"]
    record["condition_id"] = evidence["condition_id"]
    record["provenance"] = {"benchmark_sha256": "a" * 64}
    evidence["workload"] = workload
    profile = copy.deepcopy(record["benchmark"])
    per_launch = profile["working_set_bytes"] // 4 * profile["iterations_per_launch"]
    profile.update(kernel_launches=1, admitted_blocks=profile["blocks"], batches=1, batch_launches=1,
                   elements=per_launch, operations=per_launch,
                   logical_bytes=per_launch * logical_bytes_per_element(workload),
                   row_evaluations=per_launch / profile["row_width"] if profile["row_width"] else 0,
                   profile_region=True)
    profile["measure_epochs"] = [{"start_s": 0, "end_s": .1, "counts_exact": True,
                                   **{key: profile[key] for key in ("elements", "operations", "logical_bytes", "row_evaluations",
                                                                   "kernel_launches", "admitted_blocks", "batches")}}]
    evidence["profile_benchmark"] = profile
    parameters = {k: v for k, v in record["config"].items()
                  if k not in ("gpu_uuid", "graphics_clock_mhz", "memory_clock_mhz", "repeat", "stride_bytes")}
    evidence["profile_provenance"]["parameters"] = parameters
    evidence["profile_provenance"]["requested_clocks"]["memory_mhz"] = 1000
    for sample in evidence["profile_context"]["profile_active_nvml_samples"]:
        sample["memory_clock_mhz"] = 1000
    kernel = "row_nonlinear_kernel" if workload in ROW_WORKLOADS else "pointwise_nonlinear_kernel"
    for row in evidence["rows"]:
        row["kernel"] = kernel
    evidence["rows"].append({"id": "0", "kernel": kernel, "metric": SFU_INSTRUCTIONS[0],
                              "unit": "inst", "value": "1024"})
    return evidence


class NonlinearTests(unittest.TestCase):
    def test_all_functions_plan_and_profile_keep_counting_parameters(self):
        root = Path(__file__).resolve().parents[1]
        plan = expand_plan(json.loads((root / "configs/nonlinear-smoke.json").read_text()), DEVICE)
        self.assertEqual(len(plan["trials"]), 20)
        self.assertEqual({t["workload"] for t in plan["trials"]}, NONLINEAR_WORKLOADS)
        for trial in plan["trials"]:
            command = benchmark_command("bench", trial)
            self.assertIn("--working-set-bytes", command)
            self.assertEqual(trial["parameters"]["iterations"], 1)
            profile = profile_command("ncu", "bench", trial)
            self.assertIn("row_nonlinear_kernel", profile[profile.index("--kernel-name") + 1])
            self.assertIn("--profile-region", profile)
            if trial["workload"] in ROW_WORKLOADS:
                self.assertIn("--row-width", profile)
        for workload in NONLINEAR_WORKLOADS:
            orders = [t["treatment_protocol"]["order"] for t in plan["trials"] if t["workload"] == workload]
            self.assertEqual(orders.count("AB"), 2)
            self.assertEqual(orders.count("BA"), 2)

    def test_invalid_shapes_and_memory_ownership_rejected(self):
        invalid = [{"row_width": 0}, {"row_width": 65537}, {"working_set_bytes": 4100},
                   {"stride_elements": 2}, {"access": "copy"}, {"offset_bytes": 4},
                   {"sm_ids": [0]}, {"working_set_bytes": DEVICE["total_memory_bytes"]}]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                expand_plan({"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                             "experiments": [{"workload": "softmax", "parameters": parameters}]}, DEVICE)
        with self.assertRaisesRegex(ValueError, "row_width"):
            expand_plan({"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                         "experiments": [{"workload": "exp", "parameters": {"row_width": 128}}]}, DEVICE)

    def test_energy_is_per_complete_element_and_row_never_flop(self):
        for workload in NONLINEAR_WORKLOADS:
            with self.subTest(workload=workload):
                record = nonlinear_trial(workload)
                self.assertEqual(count_issues(record["benchmark"], workload), [])
                trial = analyze_trial(record)
                self.assertTrue(trial["valid"], trial["issues"])
                self.assertAlmostEqual(trial["total_pj_per_element"], 150 / trial["throughput_elements_s"] * 1e12)
                self.assertAlmostEqual(trial["operational_idle_increment_pj_per_element"], 100 / trial["throughput_elements_s"] * 1e12)
                self.assertEqual(trial["counted_measure_elements"], trial["throughput_elements_s"] * 8)
                self.assertIsNone(trial["total_pj_per_flop"])
                self.assertIsNone(trial["total_pj_per_logical_bit"])
                if workload in ROW_WORKLOADS:
                    self.assertAlmostEqual(trial["total_pj_per_row"], 128 * trial["total_pj_per_element"])
                else:
                    self.assertIsNone(trial["total_pj_per_row"])

    def test_bad_denominators_or_numerics_withhold_element_metric(self):
        for change in (lambda b: b.update(elements=0), lambda b: b.update(logical_bytes=1),
                       lambda b: b["sanity"].update(numerical_validation_passed=False),
                       lambda b: b["measure_epochs"][0].update(elements=1),
                       lambda b: b.pop("measure_epochs"),
                       lambda b: b.update(row_evaluations=3),
                       lambda b: b.update(math_implementation="unreported_fast_math")):
            record = nonlinear_trial("softmax")
            change(record["benchmark"])
            trial = analyze_trial(record)
            self.assertFalse(trial["valid"])
            self.assertIsNone(trial["total_pj_per_element"])

    def test_width_and_footprint_are_separate_comparison_strata(self):
        records = [nonlinear_trial("softmax", width=width, repeat=i)
                   for width in (128, 256) for i in range(4)]
        summary = summarize(records)
        self.assertEqual(len(summary["groups"]), 2)
        best = summary["within_clock_best_total_energy"]
        self.assertEqual(len(best), 2)
        self.assertTrue(all(s["metric"] == "total_pj_per_element" for s in best))
        self.assertEqual({s["stratum"]["nonlinear_contract"]["row_width"] for s in best}, {128, 256})
        import importlib.util
        if importlib.util.find_spec("matplotlib"):
            from powermodeling.reporting import write_plots
            with tempfile.TemporaryDirectory() as directory:
                plots = write_plots(summary, directory)
                self.assertEqual(sum("_total_plot" in key for key in plots), 2)

    def test_numeric_profile_admission_and_missing_sfu(self):
        for workload in NONLINEAR_WORKLOADS:
            with self.subTest(workload=workload):
                record = nonlinear_trial(workload)
                evidence = nonlinear_evidence(record)
                assessment = validate_evidence(record, evidence)
                self.assertEqual(assessment["status"], "pass", assessment["reasons"])
                evidence["rows"] = [r for r in evidence["rows"] if r["metric"] not in SFU_INSTRUCTIONS]
                self.assertEqual(assess_profile(evidence)["status"], "inconclusive")
                evidence["profile_benchmark"]["sanity"]["numerical_validation_passed"] = False
                self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_profile_row_width_binding_cannot_cross_shapes(self):
        record = nonlinear_trial("rmsnorm")
        evidence = nonlinear_evidence(record)
        evidence["profile_benchmark"]["row_width"] *= 2
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")

    def test_paired_contrast_has_element_units_and_preserves_negative_diagnostics(self):
        for reference_power in (80, 200):
            record = nonlinear_trial("silu")
            record["benchmark"]["paired_reference_context_allocated"] = True
            reference_phase = copy.deepcopy(record["phases"]["measure"])
            for sample in reference_phase["samples"]:
                sample["power_w"] = reference_power
                sample.pop("energy_mj", None)
            reference = copy.deepcopy(record["benchmark"])
            reference.update(operations=0, logical_bytes=0)
            for epoch in reference["measure_epochs"]:
                epoch.update(operations=0, logical_bytes=0)
            record["active_reference"] = reference
            record["phases"]["active_reference"] = reference_phase
            for name in ("measure", "idle_post"):
                phase = record["phases"][name]
                phase["start_s"] += 12
                phase["end_s"] += 12
                for sample in phase["samples"]:
                    sample["t_s"] += 12
                    sample.pop("energy_mj", None)
            for epoch in record["benchmark"]["measure_epochs"]:
                epoch["start_s"] += 12
                epoch["end_s"] += 12
            record["treatment_protocol"] = {"kind": "paired_active_reference", "order": "AB",
                "reference_workload": "control", "reference_kind": "issue_loop",
                "same_process": True, "same_allocations": True, "same_clock_policy": True,
                "launch_geometry_matched": True, "phase_order": ["active_reference", "measure"]}
            trial = analyze_trial(record)
            self.assertTrue(trial["valid"], trial["issues"])
            self.assertAlmostEqual(trial["paired_active_reference_pj_per_element"],
                                   (150 - reference_power) / trial["throughput_elements_s"] * 1e12)
            self.assertEqual(trial["paired_active_reference_eligible"], reference_power < 150,
                             trial["paired_active_reference_issues"])

    def test_verified_optimum_requires_geometry_and_complete_peak_coverage(self):
        records = []
        for threads in (128, 256):
            for repeat in range(4):
                record = nonlinear_trial("softmax", threads=threads, repeat=repeat)
                record["validation"] = {"profiler_evidence": nonlinear_evidence(record)}
                records.append(record)
        summary = summarize(records)
        self.assertTrue(summary["empirical_gpu_energy_optima"])
        self.assertTrue(all(point["metric"].endswith("pj_per_element") for point in summary["empirical_gpu_energy_optima"]))
        # An unprofiled faster measured geometry must still set the throughput bar.
        for repeat in range(4):
            record = nonlinear_trial("softmax", threads=512, repeat=repeat)
            record["provenance"] = {"benchmark_sha256": "a" * 64}
            for obj in [record["benchmark"], *record["benchmark"]["measure_epochs"]]:
                for field in ("operations", "elements", "row_evaluations", "logical_bytes", "admitted_blocks"):
                    obj[field] *= 2
            records.append(record)
        self.assertEqual(summarize(records)["empirical_gpu_energy_optima"], [])

    def test_cli_exports_element_metrics_and_labeled_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory) / "synthetic.json", Path(directory) / "report"
            raw.write_text(json.dumps([nonlinear_trial("silu", repeat=i) for i in range(4)]))
            command = [sys.executable, "-m", "powermodeling", "analyze", "--input", str(raw), "--output", str(output)]
            # Plot dependencies are optional in the repository's standard CI.
            import importlib.util
            if importlib.util.find_spec("matplotlib"):
                command.append("--plots")
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["trial_count"], 4)
            summary = json.loads((output / "summary.json").read_text())
            self.assertGreater(summary["groups"][0]["total_pj_per_element"], 0)
            self.assertIn("total_pj_per_element", (output / "trials.csv").read_text().splitlines()[0])
            if "--plots" in command:
                self.assertTrue((output / "silu-total-energy-throughput.png").is_file())


@unittest.skipUnless(os.environ.get("POWERBENCH_GPU_TESTS") == "1", "requires an actual NVIDIA GPU; opt in with POWERBENCH_GPU_TESTS=1")
class NonlinearGpuTests(unittest.TestCase):
    def test_actual_cuda_functions_and_reduction_tails(self):
        for workload in sorted(NONLINEAR_WORKLOADS):
            for width in ((1, 129, 1024) if workload in ROW_WORKLOADS else (131,)):
                with self.subTest(workload=workload, width=width):
                    command = [os.environ.get("POWERBENCH", "build/powerbench"), "--workload", workload,
                               "--grid-mode", "fixed", "--blocks", "2", "--threads", "96", "--iterations", "2",
                               "--working-set-bytes", str(2 * width * 3 * 4),
                               "--seconds", "0.1", "--warmup-seconds", "0", "--idle-seconds", "0",
                               "--batch-launches", "1", "--fixed-batches", "1"]
                    if workload in ROW_WORKLOADS:
                        command += ["--row-width", str(width)]
                    run = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
                    result = next(json.loads(line) for line in run.stdout.splitlines() if json.loads(line)["type"] == "result")
                    self.assertEqual(count_issues(result, workload), [])
                    self.assertGreater(result["numerical_validation"]["checked_values"], 0)
                    self.assertEqual(result["elements"], 2 * width * 3 * 2)
