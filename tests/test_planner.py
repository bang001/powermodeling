"""Plan failures must precede device allocation or clock mutation."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from powermodeling.planner import benchmark_command, expand_plan, numeric_expression, resolve_clock_plan, resolve_clocks


class PlannerTests(unittest.TestCase):
    def device(self):
        return {"sm_count": 108, "l2_bytes": 40 * 1024**2,
                "total_memory_bytes": 40 * 1024**3, "benchmark_sha256": "synthetic-binary"}

    def config(self, workload="tensor", parameters=None):
        return {"clock_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1215}],
                "experiments": [{"workload": workload, "parameters": parameters or {}}]}

    def strict_config(self):
        return {"study_design": "energy_sweep", "clock_sweep": {
            "graphics_step_mhz": 90, "all_memory_clocks": True,
            "include_advertised_default": True, "include_default_policy": True},
            "experiments": [{"workload": "tensor", "grid": {"blocks": [108, 216], "threads": [128, 256]}}]}

    def test_nonfinite_times_and_fractional_repeats_cannot_pass_constraints(self):
        for field, value in (("seconds", float("nan")), ("warmup_seconds", float("inf")),
                             ("idle_seconds", float("nan")), ("repeats", 3.5), ("repeats", True)):
            with self.subTest(field=field):
                config = self.config()
                config[field] = value
                with self.assertRaises(ValueError):
                    expand_plan(config, self.device())

    def test_numeric_expressions_are_integer_and_restricted(self):
        self.assertEqual(numeric_expression("max(l2_bytes/4,4096)", {"l2_bytes": 16384}), 4096)
        for expression in ("1.5", "2**4", "__import__('os')", "[1,2]", "not 0"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                numeric_expression(expression, {})

    def test_duplicate_clock_or_resolved_conditions_cannot_duplicate_trial_ids(self):
        config = self.config()
        config["clock_pairs"] *= 2
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())
        config = self.config()
        config["experiments"][0]["grid"] = {"blocks": [108, "sm_count"]}
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())

    def test_offset_and_write_buffer_allocations_are_included(self):
        config = self.config("l2", {"working_set_bytes": 1024, "offset_bytes": 16 * 1024**3, "access": "write"})
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())
        config = self.config("gemm", {"gemm_m": 131072, "gemm_n": 131072, "gemm_k": 4096})
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())

    def test_word_alignment_and_latency_access_semantics(self):
        for workload, params in (("l1", {"working_set_bytes": 3}), ("l2", {"offset_bytes": 2}),
                                 ("l2_latency", {"access": "copy"}), ("l2_latency", {"stride_elements": 2}),
                                 ("tensor", {"tensor_accumulators": 9})):
            with self.subTest(workload=workload, params=params), self.assertRaises(ValueError):
                expand_plan(self.config(workload, params), self.device())

    def test_sm_ids_are_sparse_identifiers_but_discovered_map_is_enforced(self):
        config = self.config("l2", {"sm_ids": [130]})
        self.assertEqual(expand_plan(config, self.device())["trials"][0]["parameters"]["sm_ids"], [130])
        device = self.device()
        device["discovered_sm_ids"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            expand_plan(config, device)
        with self.assertRaises(ValueError):
            expand_plan(self.config("gemm", {"sm_ids": [0]}), self.device())

    def test_hbm_stride_aliasing_and_default_footprint(self):
        with self.assertRaises(ValueError):
            expand_plan(self.config("hbm", {"working_set_bytes": "l2_bytes*8", "stride_elements": 32}), self.device())
        self.assertEqual(len(expand_plan(self.config("hbm"), self.device())["trials"]), 4)

    def test_profile_command_has_explicit_region_and_one_measured_batch(self):
        trial = expand_plan(self.config(), self.device())["trials"][0]
        command = benchmark_command("bench", trial, profiling=True)
        self.assertIn("--profile-region", command)
        self.assertEqual(command[command.index("--fixed-batches") + 1], "1")

    def test_shuffle_is_reproducible_and_condition_hash_preserves_data_seed(self):
        config = self.config("l2")
        config["experiments"][0]["grid"] = {"seed": [2026, 2027]}
        a, b = expand_plan(config, self.device()), expand_plan(copy.deepcopy(config), self.device())
        self.assertEqual(a["trials"], b["trials"])
        self.assertEqual(len({trial["condition_id"] for trial in a["trials"]}), 2)

    def supported(self):
        return {"supported_pairs": [
            {"graphics_mhz": g, "memory_mhz": m}
            for m, graphics in ((1000, [900, 945, 990, 1035, 1080, 1110, 1170, 1260]),
                                (1500, [930, 1005, 1080, 1155, 1230, 1305]))
            for g in graphics], "default_applications_graphics_mhz": 1170,
            "default_applications_memory_mhz": 1000,
            "applications_graphics_mhz": 990, "applications_memory_mhz": 1000}

    def test_grid_uses_nearest_advertised_pairs_with_endpoints_and_exact_anchor(self):
        pairs, coverage = resolve_clock_plan({"clock_sweep": {"graphics_step_mhz": 90,
            "memory_mhz": [1000]}}, self.supported())
        fixed = [pair["graphics_mhz"] for pair in pairs if pair["memory_mhz"] == 1000]
        self.assertEqual(fixed, [900, 990, 1080, 1110, 1170, 1260])
        self.assertEqual(coverage["memory_domains"][0]["requested_grid_mhz"], [900, 990, 1080, 1170, 1260])
        self.assertEqual(coverage["memory_domains"][0]["actual_gaps_mhz"], [90, 90, 30, 60, 90])
        self.assertEqual(coverage["memory_domains"][0]["required_points"][0]["status"], "included")
        self.assertIn({"graphics_mhz": None, "memory_mhz": None}, pairs)

    def test_unsupported_exact_anchor_has_explicit_reason_and_no_nearest_alias(self):
        pairs, coverage = resolve_clock_plan({"clock_sweep": {"graphics_step_mhz": 90,
            "memory_mhz": [1500]}}, self.supported())
        domain = coverage["memory_domains"][0]
        self.assertNotIn({"graphics_mhz": 1110, "memory_mhz": 1500}, pairs)
        self.assertEqual(domain["required_points"], [{"mhz": 1110, "status": "unavailable",
            "reason": "not_an_advertised_discrete_graphics_clock"}])
        self.assertFalse(coverage["required_anchor_coverage_complete"])
        self.assertTrue(any(point["error_mhz"] != 0 for point in domain["grid_mapping"]))

    def test_anchor_outside_domain_is_never_added_or_used_as_operating_bound(self):
        supported = {"supported_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1500},
                                        {"graphics_mhz": 1350, "memory_mhz": 1500}]}
        pairs, coverage = resolve_clock_plan({}, supported)
        self.assertEqual(coverage["memory_domains"][0]["min_graphics_mhz"], 1200)
        self.assertEqual(coverage["memory_domains"][0]["required_points"][0]["reason"], "outside_advertised_operating_range")
        self.assertNotIn(1110, [pair["graphics_mhz"] for pair in pairs])

    def test_each_memory_domain_has_own_bounds_and_anchor_coverage(self):
        pairs, coverage = resolve_clock_plan({"clock_sweep": {"all_memory_clocks": True}}, self.supported())
        self.assertEqual(coverage["selected_memory_mhz"], [1000, 1500])
        self.assertEqual([(d["min_graphics_mhz"], d["max_graphics_mhz"]) for d in coverage["memory_domains"]],
                         [(900, 1260), (930, 1305)])
        self.assertEqual(len(pairs), len({(p["graphics_mhz"], p["memory_mhz"]) for p in pairs}))

    def test_shipped_dvfs_sweeps_middle_memory_domain_as_well_as_endpoints(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "dvfs.json").read_text())
        supported = self.supported()
        supported["supported_pairs"].extend({"graphics_mhz": g, "memory_mhz": 1250}
                                           for g in [990, 1110, 1200])
        plan = expand_plan(config, self.device(), supported)
        coverage = plan["clock_sweep_coverage"]
        self.assertEqual(coverage["selected_memory_mhz"], [1000, 1250, 1500])
        self.assertEqual(coverage["memory_selection_policy"], "all_supported_memory_domains")
        middle = [trial for trial in plan["trials"] if trial["clocks"]["memory_mhz"] == 1250]
        self.assertTrue(middle)
        self.assertTrue(any(trial["clocks"]["graphics_mhz"] == 1110 for trial in middle))

    def test_default_grid_and_required_aliases_keep_one_physical_trial_condition(self):
        supported = self.supported()
        supported["default_applications_graphics_mhz"] = 1110
        pairs, coverage = resolve_clock_plan({"clock_sweep": {"memory_mhz": [1000]}}, supported)
        row = next(row for row in coverage["clock_conditions"] if row["clocks"] == {"graphics_mhz": 1110, "memory_mhz": 1000})
        self.assertIn("required_exact_1110_mhz", row["selection_reasons"])
        self.assertIn("advertised_default_fixed_pair", row["selection_reasons"])
        self.assertEqual(sum(p["graphics_mhz"] == 1110 for p in pairs), 1)

    def test_unknown_default_does_not_use_current_or_incoming_application_clock(self):
        supported = self.supported()
        supported.pop("default_applications_graphics_mhz")
        supported.pop("default_applications_memory_mhz")
        pairs, coverage = resolve_clock_plan({}, supported)
        self.assertEqual(coverage["advertised_default"]["status"], "unavailable")
        self.assertFalse(coverage["default_policy_reference"]["factory_default_policy_verified"])
        self.assertEqual(coverage["default_policy_reference"]["status"], "included_unverified_factory_default")
        null_row = next(row for row in coverage["clock_conditions"] if row["clocks"]["graphics_mhz"] is None)
        self.assertEqual(null_row["clock_policy"], "incoming_policy_reference")

    def test_default_fixedpair_can_be_separate_from_selected_memory_sweep(self):
        pairs, coverage = resolve_clock_plan({"clock_sweep": {"memory_mhz": [1500]}}, self.supported())
        self.assertIn({"graphics_mhz": 1170, "memory_mhz": 1000}, pairs)
        self.assertEqual(coverage["selected_memory_mhz"], [1500])
        self.assertEqual(len(coverage["memory_domains"]), 1)

    def test_sparse_native_grid_reports_large_gap_without_fabricating_clock(self):
        supported = {"supported_pairs": [{"graphics_mhz": g, "memory_mhz": 1000} for g in [900, 1200, 1500]]}
        pairs, coverage = resolve_clock_plan({}, supported)
        self.assertEqual(coverage["memory_domains"][0]["actual_gaps_mhz"], [300, 300])
        self.assertEqual({p["graphics_mhz"] for p in pairs}, {None, 900, 1200, 1500})

    def test_invalid_sweep_policies_and_unadvertised_explicit_pairs_fail(self):
        cases = [
            {"clock_sweep": {"graphics_step_mhz": 0}},
            {"clock_sweep": {"graphics_step_mhz": True}},
            {"clock_sweep": {"graphics_step_mhz": 90, "graphics_quantiles": [1]}},
            {"clock_sweep": {"memory_mhz": [1000], "memory_quantiles": [1]}},
            {"clock_sweep": {"memory_mhz": [1234]}},
            {"clock_sweep": {"required_graphics_mhz": [1110, 1110]}},
            {"clock_sweep": {"include_default_policy": 1}},
            {"clock_pairs": [{"graphics_mhz": 900, "memory_mhz": 1000}], "clock_sweep": {}},
            {"clock_pairs": [{"graphics_mhz": 1110, "memory_mhz": 1500}]},
        ]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_clocks(config, self.supported())
        with self.assertRaisesRegex(ValueError, "invent operating ranges"):
            resolve_clocks({}, {})

    def test_additional_anchors_cannot_remove_study_mandatory_1110_point(self):
        pairs, coverage = resolve_clock_plan({"clock_sweep": {"memory_mhz": [1000],
            "required_graphics_mhz": [990]}}, self.supported())
        self.assertEqual(coverage["required_graphics_mhz"], [990, 1110])
        self.assertIn({"graphics_mhz": 1110, "memory_mhz": 1000}, pairs)

    def test_plan_records_clock_coverage_sxm_and_alias_provenance(self):
        plan = expand_plan({"clock_sweep": {"memory_mhz": [1000]},
                            "experiments": [{"workload": "tensor"}]}, self.device(), self.supported())
        self.assertEqual(plan["device"]["target_form_factor"], "SXM")
        self.assertEqual(plan["clock_sweep_coverage"]["requested_step_mhz"], 90)
        self.assertTrue(all("clock_policy" in t and "clock_selection_reasons" in t for t in plan["trials"]))
        self.assertEqual(len({t["trial_id"] for t in plan["trials"]}), len(plan["trials"]))

    def test_strict_study_rejects_clock_policy_exceptions(self):
        policies = [
            {"graphics_step_mhz": 60}, {"graphics_quantiles": [1]},
            {"all_memory_clocks": False}, {"memory_quantiles": [1]}, {"memory_mhz": [1000]},
            {"include_advertised_default": False}, {"include_default_policy": False},
        ]
        for changes in policies:
            config = self.strict_config()
            config["clock_sweep"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                expand_plan(config, self.device(), self.supported())
        config = self.config()
        config["study_design"] = "energy_sweep"
        with self.assertRaisesRegex(ValueError, "explicit clock_pairs"):
            expand_plan(config, self.device(), self.supported())

    def test_unknown_default_makes_strict_plan_incomplete_but_preserves_plan_size(self):
        from powermodeling.runner import validate_plan_execution
        supported = self.supported()
        supported.pop("default_applications_graphics_mhz")
        supported.pop("default_applications_memory_mhz")
        plan = expand_plan(self.strict_config(), self.device(), supported)
        self.assertFalse(plan["execution_allowed"])
        self.assertEqual(plan["clock_sweep_coverage"]["requirements_status"], "incomplete")
        self.assertIn("not a substitute", plan["clock_sweep_coverage"]["requirement_reasons"][0])
        self.assertTrue(plan["trials"])
        self.assertGreater(plan["estimated_minimum_seconds"], 0)
        with self.assertRaisesRegex(ValueError, "execution blocked"):
            validate_plan_execution(plan)

    def test_strict_supported_1110_is_included_and_native_unsupported_domain_inapplicable(self):
        from powermodeling.runner import validate_plan_execution
        plan = expand_plan(self.strict_config(), self.device(), self.supported())
        self.assertTrue(plan["execution_allowed"])
        coverage = plan["clock_sweep_coverage"]
        self.assertEqual(coverage["requirements_status"], "complete")
        domains = {domain["memory_mhz"]: domain for domain in coverage["memory_domains"]}
        self.assertEqual(domains[1000]["required_points"][0]["status"], "included")
        self.assertEqual(domains[1500]["required_points"][0]["status"], "not_applicable")
        self.assertTrue(coverage["required_anchor_coverage_complete"])
        validate_plan_execution(plan)

    def test_all_energy_configs_use_full_clock_design_and_dvfs_has_geometry_candidates(self):
        root = Path(__file__).resolve().parents[1]
        supported = self.supported()
        supported["supported_pairs"].extend({"graphics_mhz": g, "memory_mhz": 1250} for g in [990, 1110, 1200])
        supported.update(default_applications_graphics_mhz=1110, default_applications_memory_mhz=1250)
        for name in ("saturation", "dvfs", "locality"):
            config = json.loads((root / "configs" / (name + ".json")).read_text())
            pairs, coverage = resolve_clock_plan(config, supported)
            with self.subTest(config=name):
                self.assertEqual(config["study_design"], "energy_sweep")
                self.assertEqual(coverage["requested_step_mhz"], 90)
                self.assertEqual(coverage["selected_memory_mhz"], [1000, 1250, 1500])
                self.assertEqual(coverage["requirements_status"], "complete")
                self.assertIn({"graphics_mhz": 1110, "memory_mhz": 1250}, pairs)
        config = json.loads((root / "configs" / "dvfs.json").read_text())
        plan = expand_plan(config, self.device(), supported)
        for workload in ("tensor", "gemm", "l1", "l2", "hbm"):
            candidates = {}
            for trial in plan["trials"]:
                if trial["workload"] == workload:
                    pair = (trial["clocks"]["graphics_mhz"], trial["clocks"]["memory_mhz"])
                    geometry = json.dumps(trial["parameters"], sort_keys=True)
                    candidates.setdefault(pair, set()).add(geometry)
            self.assertTrue(all(len(geometries) >= 2 for geometries in candidates.values()), workload)
        dimensions = plan["sweep_dimensions"]
        self.assertEqual(dimensions["clock_geometry_conditions"], dimensions["geometry_conditions"] * dimensions["clock_conditions_per_geometry"])
        self.assertEqual(dimensions["trials"], dimensions["clock_geometry_conditions"] * 4)
        self.assertEqual(sum(row["trials"] for row in dimensions["by_workload"]), len(plan["trials"]))
        smoke = json.loads((root / "configs" / "smoke.json").read_text())
        smoke_plan = expand_plan(smoke, self.device(), supported)
        self.assertEqual(smoke_plan["study_design"], "diagnostic")
        self.assertEqual(smoke_plan["clock_sweep_coverage"]["requirements_status"], "not_required")
        self.assertTrue(all(trial["clocks"] == {"graphics_mhz": None, "memory_mhz": None} for trial in smoke_plan["trials"]))

    def test_cli_plan_exposes_unknown_default_and_run_blocks_before_cuda_probe(self):
        from powermodeling.cli import main
        supported = self.supported()
        supported.pop("default_applications_graphics_mhz")
        supported.pop("default_applications_memory_mhz")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path, device_path, plan_path = (directory / name for name in ("config.json", "device.json", "plan.json"))
            config_path.write_text(json.dumps(self.strict_config()))
            device_path.write_text(json.dumps({"cuda_device": {**self.device(), "compute_capability": "8.0"}, "clocks": supported}))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(["plan", "--config", str(config_path), "--device-json", str(device_path), "--output", str(plan_path)])
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertFalse(summary["execution_allowed"])
            self.assertEqual(summary["requirements_status"], "incomplete")
            self.assertTrue(summary["requirement_reasons"])
            with patch("powermodeling.cli.describe_benchmark") as probe, redirect_stderr(io.StringIO()):
                status = main(["run", "--plan", str(plan_path), "--output", str(directory / "results")])
            self.assertEqual(status, 2)
            probe.assert_not_called()

    def test_paired_orders_are_balanced_by_repeat_without_splitting_condition(self):
        plan = expand_plan(self.config("tensor"), self.device())
        self.assertEqual(len({t["condition_id"] for t in plan["trials"]}), 1)
        orders = [t["treatment_protocol"]["order"] for t in sorted(plan["trials"], key=lambda t: t["repeat"])]
        self.assertNotEqual(orders[0], orders[1])
        self.assertEqual(orders[0], orders[2])
        self.assertEqual(orders.count("AB"), 2)
        self.assertEqual(orders.count("BA"), 2)
        self.assertEqual(plan["estimated_minimum_seconds"], 4*(2*12+3*3+2*6))
        self.assertIn("--paired-reference", benchmark_command("bench", plan["trials"][0]))
        self.assertIn("--paired-reference", benchmark_command("bench", plan["trials"][0], profiling=True))
        self.assertIn("--profile-region", benchmark_command("bench", plan["trials"][0], profiling=True))

    def test_three_repeat_legacy_diagnostic_retains_explicit_order_imbalance(self):
        config = self.config("tensor")
        config["repeats"] = 3
        plan = expand_plan(config, self.device())
        orders = [trial["treatment_protocol"]["order"] for trial in plan["trials"]]
        self.assertEqual(sorted((orders.count("AB"), orders.count("BA"))), [1, 2])
        self.assertTrue(all("imbalance" in trial["treatment_protocol"]["order_balance_note"] for trial in plan["trials"]))

    def test_pairing_defaults_and_opt_out_are_explicit(self):
        for workload in ("control", "l2_latency", "gemm"):
            plan = expand_plan(self.config(workload), self.device())
            self.assertNotIn("treatment_protocol", plan["trials"][0])
        config = self.config("tensor")
        config["paired_reference"] = False
        self.assertNotIn("treatment_protocol", expand_plan(config, self.device())["trials"][0])
        config = self.config("gemm")
        config["experiments"][0]["paired_reference"] = True
        self.assertEqual(expand_plan(config, self.device())["trials"][0]["treatment_protocol"]["reference_matching"], "coarse_unmatched_geometry")


if __name__ == "__main__":
    unittest.main()
