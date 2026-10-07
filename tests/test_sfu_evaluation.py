"""Direct SFU selection counterexamples; all values are synthetic."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from powermodeling.dashboard import write_evaluation
from powermodeling.evaluation import evaluate, experiment_contract, _input_curve_signature
from powermodeling.sfu import COUNT_SOURCE, INPUT_POLICY, KERNEL_IMPLEMENTATION, MATH_IMPLEMENTATION


def sfu_summary(contrast=5.0, interval=None, qs=(1024, 2048, 4096)):
    interval = interval or [contrast - .01, contrast + .01]
    groups = []
    for q in qs:
        for threads in (128, 256):
            contract = {"sfu_primitive": "ex2", "sfu_lanes": q, "sfu_chains": 4,
                "sfu_input_policy": INPUT_POLICY, "iterations_per_launch": 16384,
                "grid_mode": "auto", "math_implementation": MATH_IMPLEMENTATION,
                "kernel_implementation_version": KERNEL_IMPLEMENTATION, "block_completion_count_source": COUNT_SOURCE,
                "sfu_ptx_opcode": "ex2.approx.ftz.f32", "sfu_approximate": True, "sfu_flush_to_zero": True, "sfu_exponent_base": 2}
            config = {"graphics_clock_mhz": 1110, "memory_clock_mhz": 1593, "grid_mode": "auto",
                "threads": threads, "blocks": q // threads, "sfu_lanes": q, "sfu_chains": 4, "iterations": 16384}
            positive = contrast > 0 and interval[0] > 0
            group = {"group_id": f"sfu-synthetic-q{q}-t{threads}", "gpu_uuid": "GPU-SFU-synthetic",
                "gpu_name": "Synthetic SFU fixture, not hardware data", "workload": "sfu_ex2", "benchmark_sha256": "a" * 64,
                "measurement_stratum": {"power_limit_w": 350}, "treatment_design_stratum": {"reference_kind": "register_loop_without_sfu"},
                "config": config, "sfu_contract": contract, "experiment_contract": experiment_contract("sfu_ex2", contract, config),
                "resource_geometry": {"blocks": q // threads, "threads": threads, "sfu_chains": 4},
                "trial_ids": [], "repeats": 4, "valid_repeats": 4,
                "clock_comparison_controlled": True, "count_energy_time_alignment_exact": True,
                "verified_selection_eligible": True, "target_verified": True, "ncu_status": "pass",
                "throughput_sfu_instructions_s": 1e10, "total_pj_per_instruction": 30,
                "operational_idle_increment_pj_per_instruction": 20, "paired_active_reference_pj_per_instruction": contrast,
                "sfu_reference_delta_pj_per_instruction": contrast,
                "operational_idle_increment_eligible": True, "paired_active_reference_eligible": positive,
                "sfu_reference_delta_measurement_valid": True,
                "sfu_reference_delta_positive_optimum_eligible": positive,
                "sfu_reference_delta_ci_crosses_zero": interval[0] <= 0 <= interval[1],
                "ci95": {"throughput_sfu_instructions_s": [9.99e9, 1.001e10], "total_pj_per_instruction": [29.99, 30.01],
                         "operational_idle_increment_pj_per_instruction": [19.99, 20.01],
                         "paired_active_reference_pj_per_instruction": list(interval)}}
            groups.append(group)
    return {"selection_policy": {"min_repeats": 3, "min_geometries": 2, "throughput_fraction": .95, "near_optimum_fraction": .05},
            "groups": groups, "trials": []}


def primary(component):
    return next(r for r in component["recommendations"] if r["objective"] == "paired_active_reference")


class SfuEvaluationTests(unittest.TestCase):
    def test_sass_bound_paired_raw_records_establish_matched_q_evidence(self):
        from powermodeling.analysis import summarize
        from test_sfu import sfu_trial, sfu_evidence
        records = []
        for q in (1024, 2048, 4096):
            for threads in (128, 256):
                for repeat in range(4):
                    record = sfu_trial(lanes=q, threads=threads, launches_per_epoch=8192 // q,
                                       reference_power=100, repeat=repeat, order="AB" if repeat % 2 == 0 else "BA")
                    record["trial_id"] += f"-q{q}-t{threads}"
                    record["validation"] = {"profiler_evidence": sfu_evidence(record)}
                    records.append(record)
        summary = summarize(records)
        self.assertTrue(all(t["valid"] and t["target_verified"] for t in summary["trials"]))
        for component in summary["evaluation"]["components"]:
            self.assertEqual(primary(component)["status"], "provisional_candidate")  # No complete sweep plan.
            self.assertEqual(primary(component)["saturation_evidence"]["status"], "observed_input_size_plateau")
            self.assertEqual(len(component["input_scaling"]["curves"]), 2)
            self.assertAlmostEqual(primary(component)["energy"], 50 / (8192 * 3 * 4) * 1e12)
            self.assertTrue(all(len(curve["rows"]) == 3 for curve in component["input_scaling"]["curves"]))

    def test_raw_paired_records_keep_signed_estimator_through_summary_and_evaluation(self):
        from powermodeling.analysis import summarize
        from test_sfu import sfu_trial
        records = []
        for q in (1024, 2048, 4096):
            for threads in (128, 256):
                for repeat in range(4):
                    record = sfu_trial(lanes=q, threads=threads, launches_per_epoch=8192 // q,
                                       reference_power=200, repeat=repeat, order="AB" if repeat % 2 == 0 else "BA")
                    record["trial_id"] += f"-q{q}-t{threads}"
                    records.append(record)
        summary = summarize(records)
        self.assertTrue(all(t["valid"] for t in summary["trials"]), [t["issues"] for t in summary["trials"]])
        self.assertEqual(len(summary["groups"]), 6)
        self.assertEqual(len(summary["evaluation"]["components"]), 3)
        for component in summary["evaluation"]["components"]:
            self.assertEqual(primary(component)["status"], "no_qualified_candidate")
            for point in component["points"]:
                self.assertAlmostEqual(point["energies"]["paired_active_reference"], -50 / (8192 * 3 * 4) * 1e12)
                self.assertAlmostEqual(point["energies"]["total"], 150 / (8192 * 3 * 4) * 1e12)
                self.assertTrue(point["sfu_reference_delta"]["measurement_valid"])
                self.assertEqual(point["sfu_launch"]["sfu_lanes"], component["stratum"]["experiment_contract"]["sfu_lanes"])

    def test_primary_signed_contrast_is_separate_from_board_diagnostics_and_q(self):
        result = evaluate(sfu_summary())
        self.assertEqual(len(result["components"]), 3)
        for component in result["components"]:
            self.assertEqual(component["primary_objective"], "paired_active_reference")
            self.assertEqual(component["units"]["energy_unit"], "pJ/scalar SFU instruction")
            self.assertEqual(component["units"]["rate"], "throughput_sfu_instructions_s")
            self.assertEqual(primary(component)["energy"], 5)
            self.assertEqual(primary(component)["saturation_evidence"]["status"], "observed_input_size_plateau")
            self.assertFalse(primary(component)["saturation_evidence"]["hardware_saturation_proven"])
            for recommendation in component["recommendations"]:
                if recommendation["objective"] != "paired_active_reference":
                    self.assertEqual(recommendation["status"], "diagnostic_only")
                    self.assertIsNone(recommendation["group_id"])
            self.assertIsNone(component["clocks"][0]["minimum_energy_group_ids"]["total"])
            self.assertEqual({p["sfu_launch"]["sfu_lanes"] for p in component["points"]}, {component["stratum"]["experiment_contract"]["sfu_lanes"]})

    def test_negative_and_zero_crossing_contrasts_remain_visible_without_winner(self):
        for value, interval in ((-5, [-5.01, -4.99]), (.01, [-.01, .03]), (0, [0, 0])):
            with self.subTest(contrast=value):
                component = evaluate(sfu_summary(value, interval))["components"][0]
                self.assertEqual(primary(component)["status"], "no_qualified_candidate")
                self.assertEqual(component["points"][0]["energies"]["paired_active_reference"], value)
                self.assertEqual(component["points"][0]["sfu_reference_delta"]["signed_pj_per_instruction"], value)
                self.assertTrue(component["points"][0]["sfu_reference_delta"]["measurement_valid"])
                self.assertTrue(all(p["energies"]["paired_active_reference"] == value for curve in component["input_scaling"]["curves"] for p in curve["rows"]))

    def test_missing_matched_reference_or_instruction_retention_withholds_candidate(self):
        for missing in ("sfu_reference_delta_positive_optimum_eligible", "verified_selection_eligible"):
            summary = sfu_summary()
            for group in summary["groups"]:
                group[missing] = False
            with self.subTest(missing=missing):
                self.assertTrue(all(primary(c)["status"] == "no_qualified_candidate" for c in evaluate(summary)["components"]))

    def test_fixed_sfu_grid_remains_provisional_even_with_observed_resource_plateau(self):
        summary = sfu_summary()
        prototype = summary["groups"][0]
        summary["groups"] = []
        for blocks in (1, 2, 4):
            group = copy.deepcopy(prototype)
            group["group_id"] += f"-fixed-b{blocks}"
            group["config"].update(grid_mode="fixed", blocks=blocks)
            group["sfu_contract"]["grid_mode"] = "fixed"
            group["experiment_contract"]["grid_mode"] = "fixed"
            group["resource_geometry"]["blocks"] = blocks
            summary["groups"].append(group)
        component = evaluate(summary)["components"][0]
        self.assertEqual(component["clocks"][0]["plateau"]["status"], "observed_plateau")
        self.assertEqual(primary(component)["status"], "provisional_candidate")
        self.assertIn("a Q-derived automatic grid is required; fixed-grid SFU results remain diagnostic or provisional", primary(component)["qualification_limits"])

    def test_per_q_deltas_are_not_replaced_with_cross_q_median(self):
        summary = sfu_summary()
        for group in summary["groups"]:
            value = group["sfu_contract"]["sfu_lanes"] / 1024
            group["paired_active_reference_pj_per_instruction"] = value
            group["sfu_reference_delta_pj_per_instruction"] = value
            group["ci95"]["paired_active_reference_pj_per_instruction"] = [value - .01, value + .01]
        for component in evaluate(summary)["components"]:
            self.assertEqual(primary(component)["energy"], component["stratum"]["experiment_contract"]["sfu_lanes"] / 1024)

    def test_q_curves_require_same_primitive_chains_iterations_input_and_clocks(self):
        group = sfu_summary()["groups"][0]
        signature = _input_curve_signature(group)
        for field, value in (("sfu_chains", 8), ("sfu_primitive", "rcp"), ("sfu_input_policy", "different"),
                             ("iterations_per_launch", 32768), ("sfu_flush_to_zero", False)):
            changed = copy.deepcopy(group)
            changed["sfu_contract"][field] = value
            with self.subTest(field=field):
                self.assertNotEqual(_input_curve_signature(changed), signature)
        for field, value in (("threads", 512), ("graphics_clock_mhz", 1200), ("memory_clock_mhz", 1400)):
            changed = copy.deepcopy(group)
            changed["config"][field] = value
            with self.subTest(field=field):
                self.assertNotEqual(_input_curve_signature(changed), signature)

    def test_unverified_fast_q_remains_in_observed_curve_peak(self):
        summary = sfu_summary(qs=(512, 1024, 2048, 4096))
        for group in summary["groups"]:
            if group["sfu_contract"]["sfu_lanes"] == 512:
                group.update(throughput_sfu_instructions_s=2e10, verified_selection_eligible=False, target_verified=False)
        component = next(c for c in evaluate(summary)["components"] if c["stratum"]["experiment_contract"]["sfu_lanes"] == 2048)
        self.assertEqual(primary(component)["saturation_evidence"]["observed_peak"], 2e10)
        self.assertEqual(primary(component)["saturation_evidence"]["status"], "inconclusive")

    def test_csv_and_html_export_signed_diagnostics_and_sfu_launch_contract(self):
        result = evaluate(sfu_summary(-5))
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation(result, directory)
            with Path(paths["evaluation_csv"]).open() as stream:
                rows = [r for r in csv.DictReader(stream) if r["objective"] == "paired_active_reference"]
            self.assertTrue(all(float(r["energy"]) == -5 for r in rows))
            self.assertEqual({json.loads(r["sfu_launch"])["sfu_lanes"] for r in rows}, {1024, 2048, 4096})
            html = Path(paths["evaluation_html"]).read_text()
            self.assertIn("Signed SFU / control contrast", html)
            self.assertIn("Board total (diagnostic)", html)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plot dependency unavailable")
    def test_standalone_plot_retains_signed_negative_contrast_and_primitive_identity(self):
        from powermodeling.reporting import write_plots
        summary = sfu_summary(-5)
        summary["evaluation"] = evaluate(summary)
        with tempfile.TemporaryDirectory() as directory, patch("powermodeling.reporting._energy_figure") as ordinary, patch("powermodeling.reporting.OBJECTIVES", ("paired_active_reference",)):
            files = write_plots(summary, directory)
            self.assertEqual(ordinary.call_count, 3)  # Negative values survive plotting admission per Q.
            self.assertEqual(len(files), 4)  # Matched thread curves, each once as PNG + SVG.
            svg = next(Path(path) for key, path in files.items() if key.endswith("_svg"))
            ET.parse(svg)
            text = svg.read_text()
            for expected in ("signed SFU/control contrast", "chains=4", "primitive=ex2", "active SFU register lanes", "evidence ineligible"):
                self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
