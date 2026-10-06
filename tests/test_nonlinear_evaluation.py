"""Q-scaling evidence is separate from per-Q energy; no GPU measurements."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from powermodeling.analysis import summarize
from powermodeling.dashboard import write_evaluation
from powermodeling.evaluation import _input_curve_signature, evaluate
from test_nonlinear import nonlinear_evidence
from test_nonlinear_q_grid import q_grid_trial


def q_curve_records(qs=(1024, 2048, 4096), threads=(128, 256)):
    records = []
    for q in qs:
        for t in threads:
            for repeat in range(4):
                record = q_grid_trial(elements=q, threads=t, iterations=1,
                                      launches_per_epoch=8192 // q, repeat=repeat)
                record["trial_id"] += f"-q{q}-t{t}"
                record["validation"] = {"profiler_evidence": nonlinear_evidence(record)}
                records.append(record)
    return records


def component_for(evaluation, q):
    return next(c for c in evaluation["components"] if c["stratum"]["experiment_contract"]["input_elements"] == q)


def total(component):
    return next(r for r in component["recommendations"] if r["objective"] == "total")


class NonlinearInputScalingEvaluationTests(unittest.TestCase):
    def test_different_q_energy_values_remain_separate_through_recommendation(self):
        records = q_curve_records()
        powers = {1024: 100, 2048: 125, 4096: 150}
        for record in records:
            power = powers[record["benchmark"]["input_elements"]]
            for sample in record["phases"]["measure"]["samples"]:
                sample["power_w"] = power
                sample["energy_mj"] = (300 + power * (sample["t_s"] - 6)) * 1000
            for sample in record["phases"]["idle_post"]["samples"]:
                sample["energy_mj"] = (300 + power * 12 + 50 * (sample["t_s"] - 18)) * 1000
            record["samples"] = [{"phase": name, **sample} for name, phase in record["phases"].items() for sample in phase["samples"]]
        evaluation = summarize(records)["evaluation"]
        for q, power in powers.items():
            component = component_for(evaluation, q)
            self.assertAlmostEqual(total(component)["energy"], power / 8192 * 1e12)
            for curve in component["input_scaling"]["curves"]:
                for point in curve["rows"]:
                    self.assertAlmostEqual(point["energies"]["total"], powers[point["input_elements"]] / 8192 * 1e12)

    def test_q_curves_are_diagnostic_and_never_pool_energy_components(self):
        summary = summarize(q_curve_records())
        self.assertEqual(len(summary["groups"]), 6)
        self.assertEqual(len(summary["evaluation"]["components"]), 3)
        for component in summary["evaluation"]["components"]:
            q = component["stratum"]["experiment_contract"]["input_elements"]
            self.assertEqual({p["nonlinear_launch"]["input_elements"] for p in component["points"]}, {q})
            self.assertEqual(component["clocks"][0]["plateau"]["status"], "not_applicable_q_grid")
            self.assertEqual(component["clocks"][0]["verified_geometry_count"], 2)
            self.assertEqual(len(component["input_scaling"]["curves"]), 2)
            for curve in component["input_scaling"]["curves"]:
                self.assertEqual([p["input_elements"] for p in curve["rows"]], [1024, 2048, 4096])
                self.assertEqual(len({p["threads"] for p in curve["rows"]}), 1)
                self.assertTrue(all(p["eligible"]["total"] for p in curve["rows"]))
                self.assertTrue(all(p["block_completion_count_source"] == "synchronized_completed_launches" for p in curve["rows"]))
            rec = total(component)
            self.assertEqual(rec["saturation_evidence"]["status"], "observed_input_size_plateau")
            self.assertFalse(rec["saturation_evidence"]["hardware_saturation_proven"])
            self.assertEqual(rec["status"], "provisional_candidate")  # No complete sweep plan supplied.
            self.assertNotIn("input-size plateau is inconclusive", rec["qualification_limits"])

    def test_selected_q_must_be_in_largest_three_observed_levels(self):
        evaluation = summarize(q_curve_records((512, 1024, 2048, 4096)))["evaluation"]
        rec = total(component_for(evaluation, 512))
        self.assertEqual(rec["saturation_evidence"]["input_elements_levels"], [1024, 2048, 4096])
        self.assertEqual(rec["saturation_evidence"]["status"], "inconclusive")
        self.assertIn("input-size plateau is inconclusive", rec["qualification_limits"])
        self.assertEqual(total(component_for(evaluation, 4096))["saturation_evidence"]["status"], "observed_input_size_plateau")

    def test_unprofiled_largest_q_cannot_be_dropped_to_fabricate_plateau(self):
        records = q_curve_records((512, 1024, 2048, 4096))
        for record in records:
            if record["benchmark"]["input_elements"] == 4096:
                record["validation"] = {}
        rec = total(component_for(summarize(records)["evaluation"], 2048))
        evidence = rec["saturation_evidence"]
        self.assertEqual(evidence["input_elements_levels"], [1024, 2048, 4096])
        self.assertEqual(evidence["status"], "inconclusive")
        self.assertEqual(len(evidence["group_ids"]), 2)

    def test_unverified_fastest_q_keeps_complete_observed_peak(self):
        summary = summarize(q_curve_records((512, 1024, 2048, 4096)))
        for group in summary["groups"]:
            if group["nonlinear_contract"]["input_elements"] == 512:
                group.update(target_verified=False, verified_selection_eligible=False,
                             ncu_status="unprofiled", throughput_elements_s=16384)
        rec = total(component_for(evaluate(summary), 2048))
        evidence = rec["saturation_evidence"]
        self.assertEqual(evidence["observed_peak"], 16384)
        self.assertEqual(evidence["throughputs"], [8192, 8192, 8192])
        self.assertEqual(evidence["status"], "inconclusive")

    def test_peer_uncertainty_is_checked_for_selected_energy_objective(self):
        summary = summarize(q_curve_records())
        for group in summary["groups"]:
            if group["nonlinear_contract"]["input_elements"] == 4096:
                value = group["total_pj_per_element"]
                group["ci95"]["total_pj_per_element"] = [value * .5, value * 1.5]
        component = component_for(evaluate(summary), 2048)
        self.assertEqual(total(component)["saturation_evidence"]["status"], "inconclusive")
        idle = next(r for r in component["recommendations"] if r["objective"] == "operational_idle_increment")
        self.assertEqual(idle["saturation_evidence"]["status"], "observed_input_size_plateau")
        for curve in component["input_scaling"]["curves"]:
            self.assertFalse(curve["rows"][-1]["eligible"]["total"])

    def test_single_geometry_per_q_cannot_qualify_an_energy_candidate(self):
        evaluation = summarize(q_curve_records(threads=(128,)))["evaluation"]
        self.assertTrue(all(total(c)["status"] == "no_qualified_candidate" for c in evaluation["components"]))

    def test_only_q_and_derived_blocks_can_change_within_matched_curve(self):
        original = summarize(q_curve_records())["groups"][0]
        signature = _input_curve_signature(original)
        independent_changes = (
            lambda g: g.update(gpu_uuid="another-gpu"),
            lambda g: g.update(benchmark_sha256="another-binary"),
            lambda g: g["measurement_stratum"].update(driver_version="another-driver"),
            lambda g: g["treatment_design_stratum"].update(kind="another-treatment"),
            lambda g: g["config"].update(iterations=2),
            lambda g: g["config"].update(graphics_clock_mhz=1110),
            lambda g: g["config"].update(memory_clock_mhz=900),
            lambda g: g["resource_geometry"].update(threads=512),
            lambda g: g["nonlinear_contract"].update(row_width=129),
            lambda g: g["nonlinear_contract"].update(nonlinear_input_distribution="another-input"),
            lambda g: g["nonlinear_contract"].update(grid_mode="fixed"),
            lambda g: g["nonlinear_contract"].update(math_implementation="cuda_fp32_standard_streaming_v1"),
        )
        for index, mutate in enumerate(independent_changes):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.subTest(index=index):
                self.assertNotEqual(signature, _input_curve_signature(changed))

    def test_export_preserves_q_launch_metadata_and_cross_q_energy_rows(self):
        evaluation = summarize(q_curve_records())["evaluation"]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation(evaluation, directory)
            exported = json.loads(Path(paths["evaluation_json"]).read_text())
            self.assertEqual(exported, evaluation)
            with Path(paths["evaluation_csv"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({json.loads(r["nonlinear_launch"])["input_elements"] for r in rows}, {1024, 2048, 4096})
            html = Path(paths["evaluation_html"]).read_text()
            self.assertIn('id="inputScalingSection"', html)
            self.assertIn('value="input_rate"', html)
            self.assertIn('value="input_energy"', html)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plot dependency is unavailable")
    def test_standalone_q_figures_keep_matched_points_and_deduplicate_across_components(self):
        from powermodeling.reporting import write_plots
        summary = summarize(q_curve_records())
        with tempfile.TemporaryDirectory() as directory, patch("powermodeling.reporting._energy_figure"), patch("powermodeling.reporting.OBJECTIVES", ("total",)):
            files = write_plots(summary, directory)
            self.assertEqual(len(files), 4)  # Two matched thread curves, PNG + SVG each.
            for curve in summary["evaluation"]["components"][0]["input_scaling"]["curves"]:
                path = Path(files["exp-q-curve-" + curve["curve_id"] + "-total-input-scaling_svg"])
                ET.parse(path)
                svg = path.read_text()
                self.assertIn("Size stability does not prove hardware saturation", svg)
                self.assertIn("Energy remains separate at each Q", svg)
                self.assertIn("row_width=0 (pointwise)", svg)
                self.assertIn("iterations=1", svg)
                for point in curve["rows"]:
                    self.assertIn(point["group_id"][:8], svg)


if __name__ == "__main__":
    unittest.main()
