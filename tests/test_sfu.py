"""Register SFU accounting/contrast counterexamples, never hardware energy data."""

import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from powermodeling.analysis import analyze_trial, summarize
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.profiling import capture_profile, profile_command
from powermodeling.sfu import SFU_WORKLOADS, count_issues
from powermodeling.validation import SFU_INSTRUCTIONS, validate_evidence
from test_analysis import synthetic_trial
from test_integration_review import DEVICE
from test_ncu_validation import pass_fixture


SFU_VERSION = "sfu_register_recurrence_v1"
CONTROL_VERSION = "sfu_register_control_v1"
SFU_MATH = "ptx_approx_register_v1"
INPUT_POLICY = "feedback_xor_iteration_mantissa_0p5_1_v1"
COUNT_SOURCE = "synchronized_completed_launches"
REFERENCE_KIND = "register_loop_without_sfu"


def sfu_trial(workload="sfu_ex2", *, lanes=97, chains=4, threads=96, iterations=3,
              grid_mode="auto", blocks=None, launches_per_epoch=7,
              reference_launches_per_epoch=11, reference_power=100, power=150,
              order="AB", repeat=0):
    """Exact target/control epochs sharing state but not completed work counts."""
    blocks = math.ceil(lanes / threads) if blocks is None else blocks
    instructions_per_epoch = launches_per_epoch * lanes * chains * iterations
    record = synthetic_trial(workload, active_power=power, throughput=instructions_per_epoch, repeat=repeat)
    benchmark = record["benchmark"]
    benchmark.update(
        workload=workload, blocks=blocks, threads=threads, grid_mode=grid_mode,
        sfu_lanes=lanes, sfu_chains=chains, sfu_primitive=workload.removeprefix("sfu_"),
        sfu_input_policy=INPUT_POLICY, kernel_implementation_version=SFU_VERSION,
        math_implementation=SFU_MATH, block_completion_count_source=COUNT_SOURCE,
        sfu_instructions=12 * instructions_per_epoch, operations=12 * instructions_per_epoch,
        logical_bytes=0, elements=0, kernel_launches=12 * launches_per_epoch,
        admitted_blocks=12 * launches_per_epoch * blocks, batches=12 * launches_per_epoch,
        batch_launches=1, iterations_per_launch=iterations, paired_reference_context_allocated=True,
        numerical_validation={"checked_values": 16, "max_absolute_error": 0,
                              "max_relative_error": 0, "absolute_tolerance": 2e-6,
                              "relative_tolerance": 2e-5, "scope": "synthetic single-step CPU-double fixture"},
        sanity={"numerical_validation_passed": True, "finite_output_sample": True,
                "requested_sm_coverage_complete": True},
    )
    primitive = benchmark["sfu_primitive"]
    benchmark.update(sfu_ptx_opcode=primitive + (".approx.f32" if primitive == "tanh" else ".approx.ftz.f32"),
                     sfu_approximate=True, sfu_flush_to_zero=primitive != "tanh",
                     sfu_exponent_base=2 if primitive in ("ex2", "lg2") else None)
    for epoch in benchmark["measure_epochs"]:
        epoch.update(operations=instructions_per_epoch, sfu_instructions=instructions_per_epoch,
                     logical_bytes=0, elements=0, kernel_launches=launches_per_epoch,
                     admitted_blocks=launches_per_epoch * blocks, batches=launches_per_epoch)
    record["config"].update(blocks=blocks, threads=threads, grid_mode=grid_mode,
                            sfu_lanes=lanes, sfu_chains=chains, iterations=iterations, batch_launches=1)
    record["device"].update(compute_capability_major=8, compute_capability_minor=0)
    record["cuda_device"] = {"compute_capability_major": 8, "compute_capability_minor": 0}
    reference = copy.deepcopy(benchmark)
    reference_slots = reference_launches_per_epoch * lanes * chains * iterations
    reference.update(workload="control", reference_kind=REFERENCE_KIND,
                     kernel_implementation_version=CONTROL_VERSION, operations=0,
                     sfu_instructions=0, reference_loop_slots=12 * reference_slots,
                     kernel_launches=12 * reference_launches_per_epoch,
                     admitted_blocks=12 * reference_launches_per_epoch * blocks,
                     batches=12 * reference_launches_per_epoch)
    reference["sanity"]["numerical_validation_passed"] = None
    reference["numerical_validation"]["checked_values"] = 0
    for epoch in reference["measure_epochs"]:
        epoch.update(operations=0, sfu_instructions=0, reference_loop_slots=reference_slots,
                     kernel_launches=reference_launches_per_epoch,
                     admitted_blocks=reference_launches_per_epoch * blocks,
                     batches=reference_launches_per_epoch)
    reference_phase = copy.deepcopy(record["phases"]["measure"])
    for sample in reference_phase["samples"]:
        sample["power_w"] = reference_power
    record["phases"]["active_reference"] = reference_phase
    record["active_reference"] = reference
    shifted = "measure" if order == "AB" else "active_reference"
    for name in (shifted, "idle_post"):
        phase = record["phases"][name]
        phase["start_s"] += 12
        phase["end_s"] += 12
        for sample in phase["samples"]:
            sample["t_s"] += 12
    shifted_benchmark = benchmark if order == "AB" else reference
    for epoch in shifted_benchmark["measure_epochs"]:
        epoch["start_s"] += 12
        epoch["end_s"] += 12
    for phase in record["phases"].values():
        for sample in phase["samples"]:
            sample.pop("energy_mj", None)
    record["samples"] = [{"phase": name, **sample} for name, phase in record["phases"].items() for sample in phase["samples"]]
    record["treatment_protocol"] = {
        "kind": "paired_active_reference", "reference_kind": REFERENCE_KIND,
        "reference_workload": "control", "order": order,
        "phase_order": ["active_reference", "measure"] if order == "AB" else ["measure", "active_reference"],
        "same_process": True, "same_allocations": True, "same_clock_policy": True,
        "launch_geometry_matched": True,
    }
    return record


def sfu_plan(workload="sfu_ex2", parameters=None, *, device=DEVICE, skip_if_unsupported=False,
             paired_reference=True):
    return expand_plan({
        "study_design": "diagnostic", "paired_reference": paired_reference,
        "clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
        "experiments": [{"workload": workload, "stage": "sfu-test", "parameters": parameters or {},
                         "skip_if_unsupported": skip_if_unsupported}],
    }, device)


def sfu_evidence(record):
    """Synthetic profiler warp counters bound to synthetic reviewed disassembly."""
    from test_sfu_sass import make_sfu_sass_certificate

    _, evidence = pass_fixture()
    workload = record["workload"]
    record["condition_id"] = evidence["condition_id"]
    record["provenance"] = {"benchmark_sha256": "a" * 64}
    profile = copy.deepcopy(record["benchmark"])
    per_launch = profile["sfu_lanes"] * profile["sfu_chains"] * profile["iterations_per_launch"]
    profile.update(kernel_launches=1, admitted_blocks=profile["blocks"], batches=1,
                   batch_launches=1, operations=per_launch, sfu_instructions=per_launch,
                   profile_region=True)
    fields = ("kernel_launches", "admitted_blocks", "batches", "operations", "sfu_instructions", "logical_bytes", "elements")
    profile["measure_epochs"] = [{"start_s": 0, "end_s": .1, "counts_exact": True,
                                   **{field: profile[field] for field in fields}}]
    evidence.update(workload=workload, profile_benchmark=profile)
    evidence["profile_provenance"]["parameters"] = {
        key: value for key, value in record["config"].items()
        if key not in ("gpu_uuid", "graphics_clock_mhz", "memory_clock_mhz", "repeat", "stride_bytes")
    }
    evidence["profile_provenance"]["requested_clocks"]["memory_mhz"] = 1000
    evidence["profile_provenance"]["observed_device_records"][0].update(
        compute_capability_major=8, compute_capability_minor=0)
    for sample in evidence["profile_context"]["profile_active_nvml_samples"]:
        sample["memory_clock_mhz"] = 1000
    op_index = {"sfu_ex2": 0, "sfu_tanh": 1, "sfu_rsqrt": 2, "sfu_rcp": 3,
                "sfu_lg2": 4, "sfu_sqrt": 5}[workload]
    kernel = f"sfu_register_kernel<{op_index}, false, {profile['sfu_chains']}>"
    for row in evidence["rows"]:
        row["kernel"] = kernel
    warp_count = math.ceil(profile["sfu_lanes"] / 32) * profile["sfu_chains"] * profile["iterations_per_launch"]
    evidence["rows"].append({"id": "0", "kernel": kernel, "metric": SFU_INSTRUCTIONS[0],
                              "unit": "inst", "value": str(warp_count)})
    evidence["sfu_sass_evidence"] = make_sfu_sass_certificate(workload, profile["sfu_chains"])
    return evidence


class SfuCountTests(unittest.TestCase):
    def test_scalar_instruction_counts_cover_partial_warps_and_fixed_grid_stride(self):
        for workload in sorted(SFU_WORKLOADS):
            for lanes in (1, 95, 96, 97, 205):
                for chains in (1, 4, 8):
                    with self.subTest(workload=workload, lanes=lanes, chains=chains):
                        record = sfu_trial(workload, lanes=lanes, chains=chains)
                        self.assertEqual(count_issues(record["benchmark"], workload), [])
                        self.assertEqual(count_issues(record["active_reference"], workload, reference=True), [])
                        self.assertEqual(record["benchmark"]["sfu_instructions"], 12 * 7 * lanes * chains * 3)
        for blocks in (1, 3, 8):
            record = sfu_trial(grid_mode="fixed", blocks=blocks)
            self.assertEqual(count_issues(record["benchmark"], record["workload"]), [])
            self.assertEqual(record["benchmark"]["operations"], 12 * 7 * 97 * 4 * 3)

    def test_wrong_instruction_units_padding_source_and_numerical_flags_are_rejected(self):
        changes = (
            lambda b: b.update(sfu_instructions=b["sfu_instructions"] / 32),
            lambda b: b.update(operations=b["kernel_launches"] * b["blocks"] * b["threads"] * 3 * 4),
            lambda b: b.update(logical_bytes=4),
            lambda b: b.update(elements=1),
            lambda b: b.update(sfu_lanes=98),
            lambda b: b.update(sfu_chains=2),
            lambda b: b.pop("kernel_implementation_version"),
            lambda b: b.update(block_completion_count_source="per_cta_atomics"),
            lambda b: b.update(math_implementation="standard_expf"),
            lambda b: b.update(sfu_primitive="rcp"),
            lambda b: b.update(sfu_input_policy="unbounded_repeated_ex2"),
            lambda b: b["sanity"].update(numerical_validation_passed=False),
            lambda b: b["numerical_validation"].update(checked_values=0),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                record = sfu_trial()
                change(record["benchmark"])
                self.assertTrue(count_issues(record["benchmark"], record["workload"]))

    def test_whole_epoch_and_optional_batch_products_must_agree(self):
        changes = (
            lambda b: b.update(kernel_launches=1),
            lambda b: b.update(admitted_blocks=1),
            lambda b: b.update(batches=1),
            lambda b: b.update(batch_launches=True),
            lambda b: b.update(batch_launches=0),
            lambda b: b["measure_epochs"][4].update(batches=1),
            lambda b: b["measure_epochs"][4].update(batches=1.5),
            lambda b: b["measure_epochs"][4].update(kernel_launches=8),
            lambda b: b["measure_epochs"][4].update(admitted_blocks=15),
            lambda b: b["measure_epochs"][4].update(operations=1),
            lambda b: b["measure_epochs"][4].pop("sfu_instructions"),
            lambda b: b["measure_epochs"][4].update(counts_exact=False),
            lambda b: b["measure_epochs"][4].update(sfu_lanes=98),
            lambda b: b["measure_epochs"][4].update(block_completion_count_source="per_cta_atomics"),
        )
        for reference in (False, True):
            for index, change in enumerate(changes):
                with self.subTest(reference=reference, change=index):
                    record = sfu_trial()
                    benchmark = record["active_reference" if reference else "benchmark"]
                    change(benchmark)
                    self.assertTrue(count_issues(benchmark, record["workload"], reference=reference))
        record = sfu_trial()
        for benchmark in (record["benchmark"], record["active_reference"]):
            benchmark.pop("batches")
            benchmark.pop("batch_launches")
            for epoch in benchmark["measure_epochs"]:
                epoch.pop("batches")
        self.assertEqual(count_issues(record["benchmark"], record["workload"]), [])
        self.assertEqual(count_issues(record["active_reference"], record["workload"], reference=True), [])

    def test_register_control_has_slots_but_zero_sfu_and_rejects_old_issue_loop(self):
        record = sfu_trial()
        reference = record["active_reference"]
        self.assertEqual(reference["operations"], 0)
        self.assertEqual(reference["sfu_instructions"], 0)
        self.assertEqual(reference["reference_loop_slots"], 12 * 11 * 97 * 4 * 3)
        for field, wrong in (("reference_kind", "issue_loop"),
                             ("kernel_implementation_version", "integer_issue_loop_v1"),
                             ("sfu_instructions", 1), ("reference_loop_slots", 1)):
            with self.subTest(field=field):
                changed = copy.deepcopy(reference)
                changed[field] = wrong
                self.assertTrue(count_issues(changed, record["workload"], reference=True))


class SfuPlannerTests(unittest.TestCase):
    def test_native_defaults_and_protocol_do_not_depend_on_gpu_sm_count(self):
        for sm_count in (80, 108, 132):
            for workload in sorted(SFU_WORKLOADS):
                with self.subTest(sm_count=sm_count, workload=workload):
                    plan = sfu_plan(workload, device={**DEVICE, "sm_count": sm_count})
                    trial = plan["trials"][0]
                    parameters = trial["parameters"]
                    self.assertEqual(parameters["sfu_lanes"], 262144)
                    self.assertEqual(parameters["sfu_chains"], 4)
                    self.assertEqual(parameters["iterations"], 16384)
                    self.assertEqual(parameters["batch_launches"], 1)
                    self.assertEqual(parameters["blocks"], 1024)
                    self.assertEqual(parameters["threads"], 256)
                    self.assertEqual(trial["treatment_protocol"]["reference_kind"], REFERENCE_KIND)

    def test_auto_tail_fixed_grid_and_profile_route_all_count_parameters(self):
        for lanes in (1, 95, 96, 97, 205):
            for mode, blocks in (("auto", math.ceil(lanes / 96)), ("fixed", 2)):
                with self.subTest(lanes=lanes, mode=mode):
                    trial = sfu_plan(parameters={"sfu_lanes": lanes, "sfu_chains": 8, "threads": 96,
                                                  "grid_mode": mode, "blocks": blocks})["trials"][0]
                    self.assertEqual(trial["parameters"]["blocks"], blocks)
                    for command in (benchmark_command("bench", trial), profile_command("ncu", "bench", trial)):
                        self.assertEqual(command[command.index("--sfu-lanes") + 1], str(lanes))
                        self.assertEqual(command[command.index("--sfu-chains") + 1], "8")
                        self.assertEqual(command[command.index("--grid-mode") + 1], mode)
                        self.assertIn("--paired-reference", command)
                    command = profile_command("ncu", "bench", trial)
                    self.assertIn("sfu_register_kernel", command[command.index("--kernel-name") + 1])
                    self.assertIn("--profile-region", command)

    def test_invalid_parameters_and_unpaired_energy_plans_are_rejected(self):
        invalid = ({"sfu_lanes": 0}, {"sfu_chains": 2}, {"sfu_lanes": 97, "threads": 96, "blocks": 3},
                   {"grid_mode": "fixed"}, {"grid_mode": "guess"}, {"working_set_bytes": 1024},
                   {"offset_bytes": 4}, {"access": "copy"}, {"stride_elements": 2}, {"sm_ids": [0]},
                   {"threads": 32, "sfu_lanes": 32 * 1000000 + 1})
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                sfu_plan(parameters=parameters)
        with self.assertRaises(ValueError):
            sfu_plan(paired_reference=False)

    def test_v100_native_tanh_is_rejected_or_explicitly_skipped_without_emulation(self):
        volta = {**DEVICE, "compute_capability_major": 7, "compute_capability_minor": 0}
        with self.assertRaisesRegex(ValueError, "sm_75"):
            sfu_plan("sfu_tanh", device=volta)
        plan = sfu_plan("sfu_tanh", device=volta, skip_if_unsupported=True)
        self.assertEqual(plan["trials"], [])
        self.assertEqual(plan["unsupported_experiments"][0]["workload"], "sfu_tanh")
        self.assertIn("no emulation", plan["unsupported_experiments"][0]["reason"])
        for workload in sorted(SFU_WORKLOADS - {"sfu_tanh"}):
            self.assertTrue(sfu_plan(workload, device=volta)["trials"])
        unknown = {key: value for key, value in DEVICE.items() if not key.startswith("compute_capability")}
        with self.assertRaises(ValueError):
            sfu_plan(device=unknown)

    def test_shipped_configs_record_v100_native_tanh_skip_and_keep_other_primitives(self):
        root = Path(__file__).resolve().parents[1]
        clocks = {"supported_pairs": [{"graphics_mhz": graphics, "memory_mhz": 1000}
                                       for graphics in (900, 1110, 1200)],
                  "default_applications_graphics_mhz": 1200, "default_applications_memory_mhz": 1000}
        for source in (root / "configs").glob("sfu-register*.json"):
            for major in (7, 8, 9):
                with self.subTest(config=source.name, major=major):
                    device = {**DEVICE, "compute_capability_major": major, "compute_capability_minor": 0}
                    plan = expand_plan(json.loads(source.read_text()), device, clocks)
                    self.assertEqual({trial["workload"] for trial in plan["trials"]},
                                     SFU_WORKLOADS - {"sfu_tanh"} if major == 7 else SFU_WORKLOADS)
                    self.assertEqual([entry["workload"] for entry in plan["unsupported_experiments"]],
                                     ["sfu_tanh"] if major == 7 else [])


class SfuContrastTests(unittest.TestCase):
    def test_primary_uses_matched_register_control_power_while_total_stays_fixed(self):
        trials = [analyze_trial(sfu_trial(reference_power=reference_power)) for reference_power in (90, 120)]
        rate = 7 * 97 * 4 * 3
        for trial, reference_power in zip(trials, (90, 120)):
            self.assertTrue(trial["valid"], trial["issues"])
            self.assertAlmostEqual(trial["total_pj_per_instruction"], 150 / rate * 1e12)
            self.assertAlmostEqual(trial["sfu_reference_delta_pj_per_instruction"], (150 - reference_power) / rate * 1e12)
            self.assertEqual(trial["counted_measure_operations"], 8 * rate)
            self.assertEqual(trial["counted_measure_logical_bytes"], 0)
        self.assertEqual(trials[0]["total_energy_j"], trials[1]["total_energy_j"])
        self.assertEqual(trials[0]["total_pj_per_instruction"], trials[1]["total_pj_per_instruction"])
        self.assertEqual(trials[0]["sfu_reference_delta_pj_per_instruction"], 2 * trials[1]["sfu_reference_delta_pj_per_instruction"])

    def test_unequal_arm_durations_use_mean_power_not_subtracted_arm_energies(self):
        record = sfu_trial()
        reference = record["active_reference"]
        phase = record["phases"]["active_reference"]
        phase["end_s"] += 6
        template = phase["samples"][-1]
        phase["samples"] += [{**template, "t_s": 18 + index / 4} for index in range(1, 25)]
        template_epoch = reference["measure_epochs"][-1]
        reference["measure_epochs"] += [{**template_epoch, "start_s": start, "end_s": start + 1}
                                          for start in range(18, 24)]
        for field in ("kernel_launches", "admitted_blocks", "batches", "reference_loop_slots"):
            reference[field] = sum(epoch[field] for epoch in reference["measure_epochs"])
        reference["duration_s"] = 18
        for name in ("measure", "idle_post"):
            selected = record["phases"][name]
            selected["start_s"] += 6
            selected["end_s"] += 6
            for sample in selected["samples"]:
                sample["t_s"] += 6
        for epoch in record["benchmark"]["measure_epochs"]:
            epoch["start_s"] += 6
            epoch["end_s"] += 6
        record["samples"] = [{"phase": name, **sample} for name, selected in record["phases"].items()
                             for sample in selected["samples"]]
        trial = analyze_trial(record)
        self.assertTrue(trial["sfu_reference_delta_measurement_valid"], trial["sfu_reference_delta_issues"])
        self.assertGreater(trial["active_reference_total_energy_j"], trial["total_energy_j"])
        self.assertGreater(trial["sfu_reference_delta_pj_per_instruction"], 0)
        self.assertAlmostEqual(trial["sfu_reference_delta_pj_per_instruction"], 50 / (7 * 97 * 4 * 3) * 1e12)
        self.assertFalse(trial["sfu_reference_delta_diagnostics"]["equal_work_energy_difference"])

    def test_negative_control_contrast_is_preserved_without_invalidating_target(self):
        trial = analyze_trial(sfu_trial(reference_power=170))
        self.assertTrue(trial["valid"], trial["issues"])
        self.assertLess(trial["sfu_reference_delta_pj_per_instruction"], 0)
        self.assertGreater(trial["total_pj_per_instruction"], 0)
        self.assertTrue(trial["sfu_reference_delta_measurement_valid"])
        self.assertFalse(trial["sfu_reference_delta_positive_optimum_eligible"])

    def test_old_issue_loop_or_wrong_control_counts_cannot_qualify_primary_estimate(self):
        for change in (lambda r: r["treatment_protocol"].update(reference_kind="issue_loop"),
                       lambda r: r["active_reference"].update(reference_loop_slots=1),
                       lambda r: r["active_reference"].update(sfu_chains=8)):
            record = sfu_trial()
            change(record)
            trial = analyze_trial(record)
            self.assertIsNotNone(trial["sfu_reference_delta_pj_per_instruction"])
            self.assertFalse(trial["sfu_reference_delta_measurement_valid"])
            self.assertFalse(trial["paired_active_reference_eligible"])
            self.assertTrue(trial["paired_active_reference_issues"])

    def test_sfu_semantic_contracts_are_not_pooled_across_primitives_q_chains_or_iterations(self):
        variants = ({}, {"workload": "sfu_rcp"}, {"lanes": 98}, {"chains": 8}, {"iterations": 4})
        records = [sfu_trial(**variant, repeat=repeat, order="AB" if repeat % 2 == 0 else "BA")
                   for variant in variants for repeat in range(4)]
        summary = summarize(records)
        self.assertTrue(all(trial["valid"] for trial in summary["trials"]))
        self.assertEqual(len(summary["groups"]), 5)
        self.assertEqual({group["valid_repeats"] for group in summary["groups"]}, {4})
        self.assertEqual(len(summary["evaluation"]["components"]), 5)


class SfuProfilerTests(unittest.TestCase):
    def test_all_native_primitives_and_chain_counts_have_bound_valid_profiles(self):
        for workload in sorted(SFU_WORKLOADS):
            for chains in (1, 4, 8):
                with self.subTest(workload=workload, chains=chains):
                    record = sfu_trial(workload, chains=chains)
                    assessment = validate_evidence(record, sfu_evidence(record))
                    self.assertEqual(assessment["status"], "pass", assessment["reasons"])

    def test_warp_instruction_counter_is_not_the_scalar_energy_denominator(self):
        for lanes in (1, 32, 33, 96, 97):
            with self.subTest(lanes=lanes):
                record = sfu_trial(lanes=lanes)
                evidence = sfu_evidence(record)
                assessment = validate_evidence(record, evidence)
                self.assertEqual(assessment["status"], "pass", assessment["reasons"])
                derived = assessment["kernels"][0]["derived"]
                self.assertEqual(derived["scalar_sfu_instructions"], lanes * 3 * 4)
                self.assertEqual(derived["expected_sfu_warp_instructions"], math.ceil(lanes / 32) * 3 * 4)
                row = next(row for row in evidence["rows"] if row["metric"] == SFU_INSTRUCTIONS[0])
                row["value"] = str(evidence["profile_benchmark"]["sfu_instructions"])
                if lanes > 1:
                    self.assertEqual(validate_evidence(record, evidence)["status"], "fail")

    def test_missing_or_wrong_counter_units_do_not_verify_native_sfu_work(self):
        for unit in (None, "byte", "cycle"):
            with self.subTest(unit=unit):
                record = sfu_trial()
                evidence = sfu_evidence(record)
                next(row for row in evidence["rows"] if row["metric"] == SFU_INSTRUCTIONS[0])["unit"] = unit
                assessment = validate_evidence(record, evidence)
                self.assertNotEqual(assessment["status"], "pass")
                self.assertFalse(assessment["suitable_verified"])
        record = sfu_trial()
        evidence = sfu_evidence(record)
        evidence["rows"] = [row for row in evidence["rows"] if row["metric"] not in SFU_INSTRUCTIONS]
        self.assertNotEqual(validate_evidence(record, evidence)["status"], "pass")

    def test_sass_evidence_and_profile_specialization_must_match_binary_and_native_opcode(self):
        changes = (
            lambda e: e.pop("sfu_sass_evidence"),
            lambda e: e["sfu_sass_evidence"].update(benchmark_sha256="b" * 64),
            lambda e: e["profile_benchmark"].update(sfu_chains=8),
            lambda e: e["profile_benchmark"].update(sfu_lanes=98),
            lambda e: e["profile_benchmark"].update(block_completion_count_source="per_cta_atomics"),
            lambda e: e["sfu_sass_evidence"].update(raw_sass=e["sfu_sass_evidence"]["raw_sass"].replace("MUFU.EX2", "MUFU.RCP")),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                record = sfu_trial()
                evidence = sfu_evidence(record)
                change(evidence)
                assessment = validate_evidence(record, evidence)
                self.assertNotEqual(assessment["status"], "pass")
                self.assertFalse(assessment["suitable_verified"])
        record = sfu_trial()
        evidence = sfu_evidence(record)
        for row in evidence["rows"]:
            row["kernel"] = "sfu_register_kernel<3, false, 4>"
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")

    def test_extra_raw_kernel_instances_cannot_reuse_one_reported_launch(self):
        record = sfu_trial()
        evidence = sfu_evidence(record)
        extra = copy.deepcopy(evidence["rows"])
        for row in extra:
            row["id"] = "1"
        evidence["rows"].extend(extra)
        assessment = validate_evidence(record, evidence)
        self.assertEqual(assessment["status"], "fail")
        self.assertTrue(any(check["name"] == "sfu_profile_observed_launch_count" and check["status"] == "fail"
                            for check in assessment["checks"]))

    def test_profile_capture_keeps_certificate_and_rejects_changed_replay_chain_count(self):
        for second_chains in (4, 8):
            with self.subTest(second_chains=second_chains), tempfile.TemporaryDirectory() as directory:
                binary = Path(directory) / "bench"
                binary.write_bytes(b"synthetic bytes, never executed")
                binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
                record = sfu_trial()
                evidence = sfu_evidence(record)
                certificate = evidence["sfu_sass_evidence"]
                certificate["benchmark_sha256"] = binary_hash
                trial = {"trial_id": "test-sfu-r0", "condition_id": record["condition_id"], "workload": record["workload"],
                         "parameters": evidence["profile_provenance"]["parameters"],
                         "clocks": evidence["profile_provenance"]["requested_clocks"],
                         "seconds": 12, "warmup_seconds": 3, "idle_seconds": 6,
                         "treatment_protocol": {"kind": "paired_active_reference", "order": "AB"}}
                plan = {"device": {"uuid": "GPU-test", "benchmark_sha256": binary_hash, "device_index": 0},
                        "trials": [trial], "sfu_sass_evidence": certificate}
                first = {"type": "result", **evidence["profile_benchmark"], "duration_s": .02}
                second = {**first, "sfu_chains": second_chains, "duration_s": .03}
                events = [{"type": "device", "uuid": "GPU-test", "process_id": 42,
                           "compute_capability_major": 8, "compute_capability_minor": 0},
                          {"type": "phase", "phase": "measure", "event": "start", "host_monotonic_ns": 0},
                          {"type": "phase", "phase": "measure", "event": "end", "host_monotonic_ns": 1000000000},
                          first, second]

                def subprocess_mock(command, **kwargs):
                    if "--version" in command:
                        return SimpleNamespace(stdout="Version 2025.2.1.0", returncode=0, stderr="")
                    if "--query-metrics" in command:
                        return SimpleNamespace(stdout="\n".join(row["metric"] for row in evidence["rows"]), returncode=0, stderr="")
                    stream = io.StringIO()
                    writer = csv.writer(stream)
                    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
                    writer.writerows([row[field] for field in ("id", "kernel", "metric", "unit", "value")]
                                      for row in evidence["rows"])
                    Path(command[command.index("--log-file") + 1]).write_text(stream.getvalue())
                    return SimpleNamespace(stdout="\n".join(json.dumps(event) for event in events), returncode=0, stderr="")

                with patch("powermodeling.profiling.subprocess.run", side_effect=subprocess_mock), \
                        patch("powermodeling.profiling._run_profile_process", side_effect=lambda command, timeout: subprocess_mock(command)):
                    captured = capture_profile(plan, trial["trial_id"], binary, directory)
                manifest = json.loads(Path(captured["evidence"]).read_text())
                self.assertEqual(manifest["sfu_sass_evidence"], certificate)
                self.assertEqual(manifest["profile_provenance"]["deterministic_application_replay"], second_chains == 4)
                self.assertEqual(manifest["assessment"]["status"] == "pass", second_chains == 4)


@unittest.skipUnless(os.environ.get("POWERBENCH_GPU_TESTS") == "1",
                     "requires an actual NVIDIA GPU; opt in with POWERBENCH_GPU_TESTS=1")
class SfuGpuTests(unittest.TestCase):
    def run_kernel(self, workload, lanes, *, chains=4, grid_mode="auto", blocks=None):
        command = [os.environ.get("POWERBENCH", "build/powerbench"), "--workload", workload,
                   "--sfu-lanes", str(lanes), "--sfu-chains", str(chains), "--threads", "96",
                   "--grid-mode", grid_mode, "--iterations", "3", "--batch-launches", "1",
                   "--fixed-batches", "2", "--seconds", "0.1", "--warmup-seconds", "0", "--idle-seconds", "0"]
        if blocks is not None:
            command += ["--blocks", str(blocks)]
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
        events = [json.loads(line) for line in completed.stdout.splitlines()]
        result = next(event for event in events if event["type"] == "result")
        self.assertEqual(count_issues(result, workload), [])
        self.assertEqual(result["sfu_instructions"], 2 * lanes * chains * 3)
        self.assertEqual(result["operations"], result["sfu_instructions"])
        self.assertEqual(result["logical_bytes"], 0)
        self.assertEqual(result["elements"], 0)
        self.assertEqual(result["admitted_blocks"], 2 * result["blocks"])
        self.assertEqual(result["block_completion_count_source"], COUNT_SOURCE)
        self.assertTrue(result["sanity"]["numerical_validation_passed"])
        self.assertEqual(result["numerical_validation"]["checked_values"], 16)
        return events, result

    def test_actual_native_register_loops_cover_lane_tails_chains_and_fixed_grids(self):
        describe = subprocess.run([os.environ.get("POWERBENCH", "build/powerbench"), "--describe"],
                                  text=True, capture_output=True, check=True, timeout=60)
        device = next(event for event in map(json.loads, describe.stdout.splitlines()) if event["type"] == "device")
        cc = 10 * device["compute_capability_major"] + device["compute_capability_minor"]
        for workload in sorted(SFU_WORKLOADS):
            if workload == "sfu_tanh" and cc < 75:
                continue
            for lanes in (1, 95, 96, 97, 205):
                for chains in (1, 4, 8):
                    with self.subTest(workload=workload, lanes=lanes, chains=chains):
                        self.run_kernel(workload, lanes, chains=chains)
            for blocks in (1, 3, 8):
                with self.subTest(workload=workload, blocks=blocks):
                    self.run_kernel(workload, 97, grid_mode="fixed", blocks=blocks)
        if cc < 75:
            completed = subprocess.run([os.environ.get("POWERBENCH", "build/powerbench"), "--workload", "sfu_tanh"],
                                       text=True, capture_output=True, timeout=60)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("native sfu_tanh", completed.stderr)
            self.assertIn(">=7.5", completed.stderr)

    def test_actual_paired_fixed_batches_emit_target_free_control_contract(self):
        # Required paired timing flags are satisfied, but fixed batches exercise
        # JSON/counts only. This is not a telemetry or steady energy measurement.
        command = [os.environ.get("POWERBENCH", "build/powerbench"), "--workload", "sfu_ex2",
                   "--paired-reference", "--reference-order", "AB", "--sfu-lanes", "97",
                   "--sfu-chains", "4", "--threads", "96", "--iterations", "3",
                   "--batch-launches", "1", "--fixed-batches", "2", "--warmup-batches", "1",
                   "--seconds", "10", "--warmup-seconds", "1", "--idle-seconds", "6"]
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
        events = [json.loads(line) for line in completed.stdout.splitlines()]
        target = next(event for event in events if event["type"] == "result")
        control = next(event for event in events if event["type"] == "active_reference_result")
        protocol = next(event for event in events if event["type"] == "treatment_protocol")
        self.assertEqual(count_issues(target, "sfu_ex2"), [])
        self.assertEqual(count_issues(control, "sfu_ex2", reference=True), [])
        self.assertEqual(protocol["reference_kind"], REFERENCE_KIND)
        self.assertEqual(control["kernel_implementation_version"], CONTROL_VERSION)
        self.assertEqual(control["reference_loop_slots"], 2 * 97 * 4 * 3)
        self.assertEqual(control["sfu_instructions"], 0)
        self.assertEqual(control["operations"], 0)
        self.assertEqual(control["logical_bytes"], 0)
        self.assertEqual(control["elements"], 0)
        self.assertEqual(control["kernel_launches"], 2)
        self.assertEqual(control["admitted_blocks"], 2 * control["blocks"])
        self.assertEqual(control["block_completion_count_source"], COUNT_SOURCE)
        self.assertTrue(control["sanity"]["finite_output_sample"])
        self.assertIsNone(control["sanity"]["numerical_validation_passed"])
        self.assertEqual(control["numerical_validation"]["checked_values"], 0)
        self.assertTrue(target["sanity"]["numerical_validation_passed"])
        self.assertEqual(target["sfu_instructions"], control["reference_loop_slots"])


if __name__ == "__main__":
    unittest.main()
