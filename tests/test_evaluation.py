"""Counterexamples for evaluation decisions; fixtures are never GPU measurements."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from powermodeling.analysis import summarize, write_summary
from powermodeling.evaluation import EvaluationPolicy
from powermodeling.planner import expand_plan, resolve_clock_plan, PARAMETERS
from powermodeling.runner import validate_plan_execution
from test_analysis import empirical_record, synthetic_trial
from test_nonlinear import nonlinear_trial


def study_fixture(step=90):
    prototype = empirical_record("GPU-evaluation-synthetic", 900)
    parameters = {k: v for k, v in prototype["config"].items() if k in PARAMETERS and k != "blocks"}
    plan = expand_plan({"paired_reference": False, "study_design": "energy_sweep", "repeats": 4,
        "clock_sweep": {"graphics_min_mhz": 900, "graphics_step_mhz": step, "all_memory_clocks": True},
        "experiments": [{"workload": "hbm", "parameters": parameters, "grid": {"blocks": [80, 160, 320]}}]},
        {"uuid": "GPU-evaluation-synthetic", "sm_count": 80, "l2_bytes": 1024, "total_memory_bytes": 2**30},
        {"supported_pairs": [{"graphics_mhz": g, "memory_mhz": 1593} for g in (270, 300, 900, 1110, 1200)],
         "default_applications_graphics_mhz": 1200, "default_applications_memory_mhz": 1593})
    records = []
    for trial in plan["trials"]:
        gfx = trial["clocks"]["graphics_mhz"]
        r = empirical_record("GPU-evaluation-synthetic", gfx or 1200, power={900: 110, 1110: 120, 1200: 150}.get(gfx, 150),
                             repeat=trial["repeat"], blocks=trial["parameters"]["blocks"])
        r.update(trial_id=trial["trial_id"], condition_id=trial["condition_id"])
        r["config"].update(trial["parameters"], graphics_clock_mhz=gfx, memory_clock_mhz=trial["clocks"]["memory_mhz"],
                            clock_selection_reasons=trial["clock_selection_reasons"], clock_selection_policy=trial["clock_policy"])
        r["validation"]["profiler_evidence"]["condition_id"] = trial["condition_id"]
        if gfx is None: r["validation"] = {}
        records.append(r)
    return plan, records


def total_recommendation(summary):
    return next(r for r in summary["evaluation"]["components"][0]["recommendations"] if r["objective"] == "total")


class EvaluationTests(unittest.TestCase):
    def test_900_floor_and_variable_intervals_keep_exact_required_anchors(self):
        for step in (60, 90, 120):
            plan, _ = study_fixture(step)
            with self.subTest(step=step):
                validate_plan_execution(plan)
                fixed = {t["clocks"]["graphics_mhz"] for t in plan["trials"]} - {None}
                self.assertEqual(fixed, {900, 1110, 1200})
                self.assertNotIn(270, fixed); self.assertNotIn(300, fixed)

    def test_dense_clock_grid_is_truncated_before_nearest_mapping(self):
        native = list(range(270, 1501, 15))
        for step in (60, 90, 120):
            pairs, coverage = resolve_clock_plan({"study_design": "energy_sweep", "clock_sweep": {
                "all_memory_clocks": True, "graphics_step_mhz": step}}, {
                "supported_pairs": [{"graphics_mhz": g, "memory_mhz": 1593} for g in native],
                "default_applications_graphics_mhz": 1485, "default_applications_memory_mhz": 1593})
            fixed = {p["graphics_mhz"] for p in pairs} - {None}
            self.assertEqual(fixed, set(range(900, 1501, step)) | {1110, 1485, 1500})
            self.assertTrue(all(g >= 900 for g in fixed))
            self.assertEqual(coverage["memory_domains"][0]["evaluation_min_graphics_mhz"], 900)

    def test_factory_default_below_floor_is_the_only_required_low_exception(self):
        pairs, _ = resolve_clock_plan({"study_design": "energy_sweep", "clock_sweep": {"all_memory_clocks": True}}, {
            "supported_pairs": [{"graphics_mhz": g, "memory_mhz": 1593} for g in (270, 300, 900, 1110, 1200)],
            "default_applications_graphics_mhz": 300, "default_applications_memory_mhz": 1593})
        self.assertIn({"graphics_mhz": 300, "memory_mhz": 1593}, pairs)
        self.assertNotIn({"graphics_mhz": 270, "memory_mhz": 1593}, pairs)

    def test_domain_below_floor_is_not_invented_or_forced_into_low_sweep(self):
        pairs, coverage = resolve_clock_plan({"study_design": "energy_sweep", "clock_sweep": {"all_memory_clocks": True}}, {
            "supported_pairs": [{"graphics_mhz": g, "memory_mhz": m} for m, clocks in ((1000, (270, 300)), (1593, (900, 1110, 1200))) for g in clocks],
            "default_applications_graphics_mhz": 1200, "default_applications_memory_mhz": 1593})
        self.assertFalse(any(p["memory_mhz"] == 1000 for p in pairs))
        self.assertEqual(coverage["memory_domains"][0]["evaluation_range_status"], "not_applicable")

    def test_complete_precise_verified_plateau_qualifies_candidate_and_anchor_comparisons(self):
        plan, records = study_fixture()
        summary = summarize(records, plan=plan)
        self.assertEqual(summary["evaluation"]["coverage"]["status"], "complete")
        candidate = total_recommendation(summary)
        self.assertEqual(candidate["status"], "qualified_observed_candidate", candidate)
        self.assertAlmostEqual(candidate["factory_default_comparison"]["energy_reduction_fraction"], 1 - 110/150)
        self.assertAlmostEqual(candidate["exact_1110_comparison"]["energy_reduction_fraction"], 1 - 110/120)
        self.assertEqual(candidate["factory_default_comparison"]["evidence"], "resolved_improvement")
        self.assertTrue(all(c["plateau"]["hardware_saturation_proven"] is False for c in summary["evaluation"]["components"][0]["clocks"]))

    def test_missing_plan_or_missing_trial_cannot_qualify_complete_study(self):
        plan, records = study_fixture()
        for supplied, population in ((None, records), (plan, records[:-1])):
            summary = summarize(population, plan=supplied)
            self.assertEqual(total_recommendation(summary)["status"], "provisional_candidate")
            self.assertNotEqual(summary["evaluation"]["coverage"]["status"], "complete")

    def test_id_only_match_cannot_hide_wrong_workload_parameters(self):
        plan, records = study_fixture()
        records[0]["config"]["iterations"] += 1
        summary = summarize(records, plan=plan)
        self.assertEqual(summary["evaluation"]["coverage"]["status"], "incomplete")
        self.assertIn(records[0]["trial_id"], summary["evaluation"]["coverage"]["by_workload"]["hbm"]["mismatched_trial_ids"])

    def test_plan_id_match_cannot_hide_changed_repeat_or_binary(self):
        for field in ("repeat", "binary"):
            plan, records = study_fixture()
            if field == "repeat": records[0]["repeat"] = 99
            else:
                plan["device"]["benchmark_sha256"] = "f" * 64
            with self.subTest(field=field):
                result = summarize(records, plan=plan)
                self.assertEqual(result["evaluation"]["coverage"]["status"], "incomplete")

    def test_invalid_results_have_no_fabricated_energy_or_recommendation(self):
        r = synthetic_trial("hbm")
        r["benchmark"]["logical_bytes"] = -1
        c = summarize([r])["evaluation"]["components"][0]
        self.assertIsNone(c["observed_peak"])
        self.assertTrue(all(r["group_id"] is None for r in c["recommendations"]))
        self.assertTrue(all(v is None for v in c["points"][0]["energies"].values()))
        self.assertIn("Insufficient valid repeats", c["points"][0]["eligibility_reasons"]["total"])

    def test_unverified_fastest_geometry_does_not_lower_performance_bar(self):
        plan, records = study_fixture()
        for r in records:
            if r["config"]["blocks"] == 320 and r["config"]["graphics_clock_mhz"] is not None:
                r["validation"] = {}
                r["benchmark"]["operations"] *= 2; r["benchmark"]["logical_bytes"] *= 2
                for epoch in r["benchmark"]["measure_epochs"]:
                    epoch["operations"] *= 2; epoch["logical_bytes"] *= 2
        self.assertEqual(total_recommendation(summarize(records, plan=plan))["status"], "no_qualified_candidate")

    def test_three_geometries_at_different_clocks_do_not_establish_plateau(self):
        plan, records = study_fixture()
        records = [r for r in records if r["config"]["blocks"] == {900:80,1110:160,1200:320}.get(r["config"]["graphics_clock_mhz"],80)]
        c = summarize(records, plan=plan)["evaluation"]["components"][0]
        self.assertTrue(all(k["plateau"]["status"] == "inconclusive" for k in c["clocks"]))
        self.assertEqual(total_recommendation(summarize(records, plan=plan))["status"], "no_qualified_candidate")

    def test_large_repeat_variation_is_visible_and_withholds_candidate(self):
        plan, records = study_fixture()
        for r in records:
            factor = (0.5, 1, 1.5, 2)[r["config"]["repeat"]]
            for epoch in r["benchmark"]["measure_epochs"]:
                epoch["logical_bytes"] *= factor; epoch["operations"] *= factor
            r["benchmark"]["logical_bytes"] *= factor; r["benchmark"]["operations"] *= factor
        c = summarize(records, plan=plan)["evaluation"]["components"][0]
        self.assertTrue(any(p["relative_ci_widths"]["total_pj_per_logical_bit"] > .1 for p in c["points"]))
        self.assertEqual(total_recommendation(summarize(records, plan=plan))["status"], "no_qualified_candidate")

    def test_input_footprints_strides_and_toolkits_are_separate_components(self):
        records = [synthetic_trial("l2", repeat=i) for i in range(3)]
        variants = []
        for footprint, stride, toolkit in ((4096,1,12090),(8192,1,12090),(4096,2,12090),(4096,1,13000)):
            for original in records:
                r = copy.deepcopy(original); r["trial_id"] += f"-{footprint}-{stride}-{toolkit}"
                r["config"].update(working_set_bytes=footprint,stride_elements=stride)
                r["cuda_device"] = {"cuda_compile_version":toolkit}; variants.append(r)
        self.assertEqual(len(summarize(variants)["evaluation"]["components"]),4)

    def test_nonlinear_width_units_and_sampled_validation_are_reported(self):
        records = [nonlinear_trial("softmax", width=width, repeat=i) for width in (129,1024) for i in range(3)]
        components = summarize(records)["evaluation"]["components"]
        self.assertEqual(len(components),2)
        for c in components:
            self.assertEqual(c["units"]["energy_unit"],"pJ/element")
            p=c["points"][0]
            self.assertAlmostEqual(p["pj_per_row"]["total"],p["energies"]["total"]*p["row_width"])
            self.assertTrue(c["profiler_evidence"][0]["sampled_numerical_checks"])

    def test_latency_map_retains_sm_offset_clock_and_no_energy_optimum(self):
        records=[]
        for offset in (0,128):
            for i in range(3):
                r=synthetic_trial("l2_latency",repeat=i);r["config"]["offset_bytes"]=offset
                r["benchmark"]["latency_probe"]={"per_sm":{"0":{"cycles_per_access":20+i},"3":{"cycles_per_access":30+i}}}
                records.append(r)
        c=summarize(records)["evaluation"]["components"][0]
        self.assertEqual(len(c["latency_points"]),12)
        self.assertTrue(all(r["group_id"] is None for r in c["recommendations"]))
        self.assertIn("no near/far",c["interpretation"])

    def test_standalone_exports_escape_untrusted_names_and_keep_trace_observed(self):
        _,records=study_fixture();records[0]["device"]["name"]='</script><img src=x onerror=alert(1)>'
        summary=summarize(records)
        with tempfile.TemporaryDirectory() as directory:
            paths=write_summary(summary,directory)
            html=Path(paths['evaluation_html']).read_text()
            self.assertNotIn('</script><img',html)
            self.assertIn('\\u003c/script>',html)
            self.assertIn('energy_ci95',Path(paths['evaluation_csv']).read_text().splitlines()[0])
            exported=json.loads(Path(paths['evaluation_json']).read_text())
            trace=exported['components'][0]['measurement_example']
            self.assertTrue(trace['points'])
            self.assertLessEqual(len(trace['points']),3*96)

    def test_invalid_evaluation_thresholds_are_rejected(self):
        for kwargs in ({"minimum_resource_levels":2},{"plateau_tolerance_fraction":float('nan')},{"maximum_relative_ci_width":True}):
            with self.assertRaises(ValueError):EvaluationPolicy(**kwargs)

    def test_full_sweep_performance_bar_can_select_higher_energy_than_own_clock_minimum(self):
        plan, records = study_fixture()
        for record in records:
            if record["config"]["graphics_clock_mhz"] != 900:
                continue
            record["benchmark"]["operations"] *= 0.8
            record["benchmark"]["logical_bytes"] *= 0.8
            for epoch in record["benchmark"]["measure_epochs"]:
                epoch["operations"] *= 0.8
                epoch["logical_bytes"] *= 0.8
            for name, phase in record["phases"].items():
                for sample in phase["samples"]:
                    sample.pop("energy_mj", None)
                    if name == "measure":
                        sample["power_w"] = 80
        component = summarize(records, plan=plan)["evaluation"]["components"][0]
        diagnostic = next(d for d in component["energy_selection_diagnostics"] if d["objective"] == "total")
        self.assertAlmostEqual(diagnostic["own_clock_candidate_global_throughput_fraction"], 0.8)
        self.assertGreater(diagnostic["global_constraint_candidate_energy"], diagnostic["lowest_own_clock_candidate_energy"])
        self.assertNotEqual(diagnostic["lowest_own_clock_candidate_group_id"], diagnostic["global_constraint_candidate_group_id"])
        self.assertIn("falls below", diagnostic["reason"])

    def test_energy_accounting_preserves_scope_difference_and_unqualified_contrast(self):
        records = []
        for repeat in range(4):
            record = synthetic_trial("hbm", active_power=184, throughput=1e12, repeat=repeat)
            for name, phase in record["phases"].items():
                for sample in phase["samples"]:
                    sample.pop("energy_mj", None)
                    sample["pstate"] = 8 if name.startswith("idle_") else 0
                    if name.startswith("idle_"):
                        sample["power_w"] = 64
            records.append(record)
        summary = summarize(records)
        point = summary["evaluation"]["components"][0]["points"][0]
        self.assertEqual(point["energies"]["total"], 23)
        self.assertEqual(point["energies"]["operational_idle_increment"], 15)
        self.assertFalse(point["objective_eligible"]["operational_idle_increment"])
        diagnostics = point["measurement_diagnostics"]
        self.assertEqual(diagnostics["power_contributions"]["idle_w"], 64)
        self.assertAlmostEqual(diagnostics["scope_factors"]["total_over_idle_increment"], 23 / 15)
        self.assertEqual(diagnostics["objectives"]["total"]["reconstructed_value"], 23)
        with tempfile.TemporaryDirectory() as directory:
            files = write_summary(summary, directory)
            html = Path(files["evaluation_html"]).read_text()
            self.assertIn("Energy accounting and comparison checks", html)
            import csv
            with Path(files["evaluation_csv"]).open() as stream:
                rows = list(csv.DictReader(stream))
            total = next(row for row in rows if row["objective"] == "total")
            exported = json.loads(total["measurement_diagnostics"])
            self.assertEqual(exported["objectives"]["total"]["reported_value"], 23)
            self.assertFalse(exported["scope_factors"]["idle_increment_eligible"])


if __name__ == "__main__":
    unittest.main()
