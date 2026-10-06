"""Q-grid boundary/accounting regressions; synthetic fixtures are not GPU data."""

import copy
import json
import os
import subprocess
import unittest

from powermodeling.analysis import analyze_trial, summarize
from powermodeling.nonlinear import NONLINEAR_WORKLOADS, ROW_WORKLOADS, count_issues, logical_bytes_per_element
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.profiling import profile_command
from powermodeling.validation import SFU_ACTIVITY, validate_evidence
from test_integration_review import DEVICE
from test_nonlinear import nonlinear_evidence, nonlinear_trial


Q_GRID_MATH = "cuda_fp32_q_grid_v2"
Q_GRID_VERSION = "fp32_complete_nonlinear_q_grid_v2"
COMPLETION_SOURCE = "synchronized_completed_launches"


def q_grid_trial(workload="exp", *, elements=97, width=129, threads=96,
                 grid_mode="auto", blocks=None, iterations=3, launches_per_epoch=7, repeat=0):
    """Twelve complete one-second epochs with no padding in the denominator."""
    row_width = width if workload in ROW_WORKLOADS else 0
    if blocks is None:
        blocks = elements // row_width if row_width else (elements + threads - 1) // threads
    record = nonlinear_trial(workload, width=width, blocks=blocks, threads=threads, repeat=repeat)
    benchmark = record["benchmark"]
    per_epoch = launches_per_epoch * elements * iterations
    benchmark.update(
        elements=12 * per_epoch, operations=12 * per_epoch,
        logical_bytes=12 * per_epoch * logical_bytes_per_element(workload),
        row_evaluations=12 * per_epoch // row_width if row_width else 0,
        working_set_bytes=elements * 4, input_elements=elements, grid_mode=grid_mode,
        kernel_launches=12 * launches_per_epoch, admitted_blocks=12 * launches_per_epoch * blocks,
        batches=12 * launches_per_epoch, batch_launches=1,
        iterations_per_launch=iterations, math_implementation=Q_GRID_MATH,
        kernel_implementation_version=Q_GRID_VERSION, block_completion_count_source=COMPLETION_SOURCE,
    )
    for epoch in benchmark["measure_epochs"]:
        epoch.update(
            elements=per_epoch, operations=per_epoch,
            logical_bytes=per_epoch * logical_bytes_per_element(workload),
            row_evaluations=per_epoch // row_width if row_width else 0,
            kernel_launches=launches_per_epoch, admitted_blocks=launches_per_epoch * blocks,
            batches=launches_per_epoch,
        )
    record["config"].update(grid_mode=grid_mode, blocks=blocks, threads=threads,
                            working_set_bytes=elements * 4, iterations=iterations)
    return record


def one_trial(workload, parameters, *, device=DEVICE):
    plan = expand_plan({"paired_reference": False, "study_design": "diagnostic",
                        "clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                        "experiments": [{"workload": workload, "parameters": parameters}]}, device)
    return plan["trials"][0]


class NonlinearQGridTests(unittest.TestCase):
    def test_default_input_size_is_independent_of_gpu_and_thread_count(self):
        for sm_count in (80, 108, 132):
            for threads in (128, 256, 512):
                for workload in sorted(NONLINEAR_WORKLOADS):
                    with self.subTest(sm_count=sm_count, threads=threads, workload=workload):
                        trial = one_trial(workload, {"threads": threads}, device={**DEVICE, "sm_count": sm_count})
                        parameters = trial["parameters"]
                        self.assertEqual(parameters["working_set_bytes"], 1048576 * 4)
                        self.assertEqual(parameters["grid_mode"], "auto")
                        self.assertEqual(parameters["blocks"], 1024 if workload in ROW_WORKLOADS else 1048576 // threads)

    def test_pointwise_auto_grid_covers_partial_ctas_and_routes_enum(self):
        for elements in (1, 95, 96, 97, 205):
            with self.subTest(elements=elements):
                trial = one_trial("exp", {"threads": 96, "working_set_bytes": elements * 4})
                self.assertEqual(trial["parameters"]["blocks"], (elements + 95) // 96)
                for command in (benchmark_command("bench", trial), profile_command("ncu", "bench", trial)):
                    self.assertEqual(command[command.index("--grid-mode") + 1], "auto")
                    self.assertEqual(command[command.index("--working-set-bytes") + 1], str(elements * 4))

    def test_rowwise_auto_grid_counts_whole_rows_and_fixed_grid_accepts_row_tails(self):
        for workload in sorted(ROW_WORKLOADS):
            for width in (1, 129):
                for rows in (1, 3, 5):
                    with self.subTest(workload=workload, width=width, rows=rows):
                        parameters = {"threads": 96, "row_width": width, "working_set_bytes": rows * width * 4}
                        self.assertEqual(one_trial(workload, parameters)["parameters"]["blocks"], rows)
                        fixed = one_trial(workload, {**parameters, "grid_mode": "fixed", "blocks": 2})
                        self.assertEqual(fixed["parameters"]["blocks"], 2)
                        self.assertEqual(fixed["parameters"]["working_set_bytes"], rows * width * 4)

    def test_auto_conflicts_invalid_modes_partial_rows_and_excessive_grids_are_rejected(self):
        invalid = (
            ("exp", {"threads": 96, "working_set_bytes": 97 * 4, "blocks": 3}),
            ("exp", {"grid_mode": "fixed"}),
            ("exp", {"grid_mode": "guess"}),
            ("exp", {"threads": 32, "working_set_bytes": (32 * 1000000 + 1) * 4}),
            ("softmax", {"row_width": 129, "working_set_bytes": 130 * 4}),
            ("softmax", {"row_width": 1, "working_set_bytes": 1000001 * 4}),
        )
        for workload, parameters in invalid:
            with self.subTest(workload=workload, parameters=parameters), self.assertRaises(ValueError):
                one_trial(workload, parameters)
        matching = one_trial("exp", {"threads": 96, "working_set_bytes": 97 * 4, "blocks": 2})
        self.assertEqual(matching["parameters"]["blocks"], 2)

    def test_complete_input_counts_do_not_depend_on_fixed_grid_size_or_padding(self):
        trials = []
        for grid_mode, blocks in (("auto", 2), ("fixed", 1), ("fixed", 3), ("fixed", 8)):
            with self.subTest(grid_mode=grid_mode, blocks=blocks):
                record = q_grid_trial(grid_mode=grid_mode, blocks=blocks)
                self.assertEqual(count_issues(record["benchmark"], "exp"), [])
                trial = analyze_trial(record)
                self.assertTrue(trial["valid"], trial["issues"])
                self.assertEqual(trial["counted_measure_elements"], 8 * 7 * 97 * 3)
                trials.append(trial)
        self.assertEqual(len({trial["total_pj_per_element"] for trial in trials}), 1)

    def test_padded_thread_count_cannot_replace_actual_input_elements(self):
        record = q_grid_trial()
        benchmark = record["benchmark"]
        for obj in (benchmark, *benchmark["measure_epochs"]):
            wrong = obj["kernel_launches"] * benchmark["blocks"] * benchmark["threads"] * benchmark["iterations_per_launch"]
            obj.update(elements=wrong, operations=wrong, logical_bytes=wrong * 8)
        trial = analyze_trial(record)
        self.assertFalse(trial["valid"])
        self.assertTrue(any(issue.startswith("nonlinear_count_disagreement:") for issue in trial["issues"]))
        self.assertIsNone(trial["total_pj_per_element"])

    def test_q_grid_requires_complete_launch_and_epoch_accounting(self):
        changes = (
            lambda b: b.update(kernel_launches=b["kernel_launches"] + 1),
            lambda b: b.update(admitted_blocks=b["admitted_blocks"] + 1),
            lambda b: b["measure_epochs"][4].update(kernel_launches=8),
            lambda b: b["measure_epochs"][4].update(admitted_blocks=15),
            lambda b: b["measure_epochs"][4].pop("kernel_launches"),
            lambda b: b["measure_epochs"][4].pop("admitted_blocks"),
        )
        for change in changes:
            with self.subTest(change=changes.index(change)):
                record = q_grid_trial()
                change(record["benchmark"])
                trial = analyze_trial(record)
                self.assertFalse(trial["valid"])
                self.assertIsNone(trial["total_pj_per_element"])

    def test_present_batch_metadata_cannot_contradict_completed_launches(self):
        changes = (
            lambda b: b.update(batches=1),
            lambda b: b["measure_epochs"][4].update(batches=1),
            lambda b: b.update(batch_launches=2),
            lambda b: b.update(batches=True),
            lambda b: b.update(batches=1.5),
            lambda b: b["measure_epochs"][4].update(batches=True),
            lambda b: b["measure_epochs"][4].update(batches=1.5),
            lambda b: b.update(batch_launches=True),
            lambda b: b.update(batch_launches=0),
            lambda b: b.update(batch_launches=1.5),
            lambda b: b.pop("batch_launches"),
        )
        for change in changes:
            with self.subTest(change=changes.index(change)):
                record = q_grid_trial()
                change(record["benchmark"])
                trial = analyze_trial(record)
                self.assertFalse(trial["valid"])
                self.assertIsNone(trial["total_pj_per_element"])

    def test_consistent_multiple_launch_batches_and_absent_optional_metadata_are_valid(self):
        record = q_grid_trial(launches_per_epoch=6)
        record["benchmark"].update(batch_launches=3, batches=24)
        for epoch in record["benchmark"]["measure_epochs"]:
            epoch["batches"] = 2
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertEqual(trial["counted_measure_elements"], 8 * 6 * 97 * 3)
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                record = nonlinear_trial() if legacy else q_grid_trial()
                record["benchmark"].pop("batches", None)
                record["benchmark"].pop("batch_launches", None)
                for epoch in record["benchmark"]["measure_epochs"]:
                    epoch.pop("batches", None)
                trial = analyze_trial(record)
                self.assertTrue(trial["valid"], trial["issues"])

    def test_q_grid_metadata_and_sampled_numerical_checks_are_required(self):
        changes = (
            lambda b: b.pop("input_elements"),
            lambda b: b.update(input_elements=98),
            lambda b: b.pop("block_completion_count_source"),
            lambda b: b.update(block_completion_count_source="per_cta_atomics"),
            lambda b: b.update(kernel_implementation_version="fp32_complete_nonlinear_v1"),
            lambda b: b.update(grid_mode="guess"),
            lambda b: b["sanity"].update(numerical_validation_passed=False),
            lambda b: b["numerical_validation"].update(checked_values=0),
        )
        for change in changes:
            with self.subTest(change=changes.index(change)):
                record = q_grid_trial()
                change(record["benchmark"])
                trial = analyze_trial(record)
                self.assertFalse(trial["valid"])
                self.assertIsNone(trial["total_pj_per_element"])

    def test_q_grid_row_counts_do_not_require_rows_divisible_by_blocks(self):
        for workload in sorted(ROW_WORKLOADS):
            for width in (1, 129):
                with self.subTest(workload=workload, width=width):
                    record = q_grid_trial(workload, elements=3 * width, width=width, grid_mode="fixed", blocks=2)
                    self.assertEqual(count_issues(record["benchmark"], workload), [])
                    trial = analyze_trial(record)
                    self.assertTrue(trial["valid"], trial["issues"])
                    self.assertEqual(trial["counted_measure_elements"], 8 * 7 * 3 * width * 3)
                    self.assertAlmostEqual(trial["total_pj_per_row"], width * trial["total_pj_per_element"])

    def test_v1_and_v2_same_input_footprint_remain_separate_strata(self):
        records = []
        for repeat in range(4):
            records.append(nonlinear_trial("exp", width=128, blocks=80, repeat=repeat))
            records.append(q_grid_trial(elements=80 * 256, threads=256, iterations=1,
                                        launches_per_epoch=100, repeat=repeat))
        summary = summarize(records)
        self.assertTrue(all(trial["valid"] for trial in summary["trials"]))
        self.assertEqual(len(summary["groups"]), 2)
        self.assertEqual({group["valid_repeats"] for group in summary["groups"]}, {4})
        self.assertEqual(len(summary["evaluation"]["components"]), 2)

    def test_profiler_evidence_cannot_validate_different_q_grid_execution(self):
        for field, wrong in (("grid_mode", "fixed"), ("input_elements", 98),
                             ("block_completion_count_source", "per_cta_atomics")):
            with self.subTest(field=field):
                record = q_grid_trial()
                evidence = nonlinear_evidence(record)
                self.assertEqual(validate_evidence(record, evidence)["status"], "pass")
                evidence["profile_benchmark"][field] = wrong
                assessment = validate_evidence(record, evidence)
                self.assertNotEqual(assessment["status"], "pass")
                self.assertFalse(assessment["suitable_verified"])

    def test_optional_sfu_activity_is_replay_diagnostic_and_missing_stays_unknown(self):
        record = q_grid_trial()
        evidence = nonlinear_evidence(record)
        missing = validate_evidence(record, evidence)
        self.assertEqual(missing["status"], "pass")
        diagnostic = missing["rates_summary"]["sfu_activity_by_kernel"][0]
        self.assertIsNone(diagnostic["active_pct"])
        self.assertIsNone(diagnostic["counter"])
        self.assertEqual(diagnostic["scope"], "unknown")
        evidence["rows"].append({"id": "0", "kernel": "pointwise_nonlinear_kernel",
                                  "metric": SFU_ACTIVITY[0], "unit": "%", "value": "0.1"})
        measured = validate_evidence(record, evidence)
        self.assertEqual(measured["status"], "pass")
        diagnostic = measured["rates_summary"]["sfu_activity_by_kernel"][0]
        self.assertEqual(diagnostic["active_pct"], 0.1)
        self.assertEqual(diagnostic["counter"], SFU_ACTIVITY[0])
        self.assertEqual(diagnostic["scope"], "elapsed")
        self.assertTrue({(check["name"], check["status"]) for check in missing["checks"]}.issubset(
            {(check["name"], check["status"]) for check in measured["checks"]}))

    def test_one_reported_launch_cannot_bind_multiple_nonlinear_kernel_instances(self):
        for conflicting_name in (False, True):
            with self.subTest(conflicting_name=conflicting_name):
                record = q_grid_trial()
                evidence = nonlinear_evidence(record)
                extra_rows = copy.deepcopy(evidence["rows"])
                for row in extra_rows:
                    if conflicting_name:
                        row["kernel"] = "pointwise_nonlinear_kernel_other"
                    else:
                        row["id"] = "1"
                evidence["rows"].extend(extra_rows)
                assessment = validate_evidence(record, evidence)
                self.assertEqual(assessment["status"], "fail")
                self.assertFalse(assessment["suitable_verified"])
                expected_check = ("nonlinear_profile_launch_id_unique_mapping" if conflicting_name
                                  else "nonlinear_profile_observed_launch_count")
                self.assertTrue(any(check["name"] == expected_check and check["status"] == "fail"
                                    for check in assessment["checks"]))


@unittest.skipUnless(os.environ.get("POWERBENCH_GPU_TESTS") == "1",
                     "requires an actual NVIDIA GPU; opt in with POWERBENCH_GPU_TESTS=1")
class NonlinearQGridGpuTests(unittest.TestCase):
    def run_function(self, workload, elements, *, threads=96, width=129, grid_mode="auto", blocks=None):
        command = [os.environ.get("POWERBENCH", "build/powerbench"), "--workload", workload,
                   "--grid-mode", grid_mode, "--threads", str(threads), "--iterations", "2",
                   "--working-set-bytes", str(elements * 4), "--seconds", "0.1",
                   "--warmup-seconds", "0", "--idle-seconds", "0", "--batch-launches", "1",
                   "--fixed-batches", "2"]
        if blocks is not None:
            command += ["--blocks", str(blocks)]
        if workload in ROW_WORKLOADS:
            command += ["--row-width", str(width)]
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
        result = next(event for event in map(json.loads, completed.stdout.splitlines()) if event["type"] == "result")
        self.assertEqual(count_issues(result, workload), [])
        self.assertEqual(result["elements"], 4 * elements)
        self.assertEqual(result["input_elements"], elements)
        self.assertEqual(result["kernel_implementation_version"], Q_GRID_VERSION)
        self.assertEqual(result["block_completion_count_source"], COMPLETION_SOURCE)
        self.assertEqual(result["admitted_blocks"], 2 * result["blocks"])
        self.assertGreater(result["numerical_validation"]["checked_values"], 0)
        self.assertTrue(result["sanity"]["numerical_validation_passed"])
        self.assertTrue(result["sanity"]["finite_output_sample"])
        return result

    def test_actual_pointwise_global_tails_and_over_under_subscribed_fixed_grids(self):
        for workload in sorted(NONLINEAR_WORKLOADS - ROW_WORKLOADS):
            for elements in (1, 95, 96, 97, 205):
                with self.subTest(workload=workload, elements=elements):
                    result = self.run_function(workload, elements)
                    self.assertEqual(result["blocks"], (elements + 95) // 96)
            for blocks in (1, 3, 8):
                with self.subTest(workload=workload, elements=97, blocks=blocks):
                    self.run_function(workload, 97, grid_mode="fixed", blocks=blocks)

    def test_actual_rowwise_complete_row_reductions_and_grid_stride_row_tails(self):
        for workload in sorted(ROW_WORKLOADS):
            for width in (1, 129, 1024):
                for rows in (1, 3, 5):
                    with self.subTest(workload=workload, width=width, rows=rows):
                        result = self.run_function(workload, rows * width, width=width)
                        self.assertEqual(result["blocks"], rows)
                        self.assertEqual(result["row_evaluations"], 4 * rows)
                        self.run_function(workload, rows * width, width=width, grid_mode="fixed", blocks=2)

    def test_actual_cli_rejects_conflicting_auto_grid_and_partial_rows(self):
        invalid = (
            ("exp", ["--threads", "96", "--working-set-bytes", "388", "--blocks", "3"], "Q-derived auto grid"),
            ("exp", ["--grid-mode", "fixed"], "requires explicit positive --blocks"),
            ("exp", ["--grid-mode", "guess"], "--grid-mode must be auto or fixed"),
            ("softmax", ["--row-width", "129", "--working-set-bytes", "520"], "complete rows"),
        )
        for workload, parameters, reason in invalid:
            with self.subTest(workload=workload, parameters=parameters):
                command = [os.environ.get("POWERBENCH", "build/powerbench"), "--workload", workload, *parameters]
                completed = subprocess.run(command, text=True, capture_output=True, timeout=60)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(reason, completed.stderr)
                self.assertFalse(any(json.loads(line).get("type") == "result"
                                     for line in completed.stdout.splitlines() if line.startswith("{")))


if __name__ == "__main__":
    unittest.main()
