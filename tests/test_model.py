import importlib.util
import unittest

from powermodeling.model import fit_model, predict_power


@unittest.skipUnless(importlib.util.find_spec("numpy"), "model regression requires numpy")
class ModelTests(unittest.TestCase):
    def documented(self, row, index):
        return {"trial_id": f"synthetic-calibration-{index}", "gpu_uuid": "GPU-test",
                "config": {"graphics_clock_mhz": 1200, "memory_clock_mhz": 1000},
                "feature_units": {"x": "synthetic-unit/s", "y": "synthetic-unit/s"},
                "feature_provenance": {"x": "synthetic fixture, not a measured GPU", "y": "synthetic fixture, not a measured GPU"},
                "power_provenance": "synthetic mathematical power, not a measured GPU", **row}

    def rows(self):
        # Multiple activity levels plus independent zero/control establish the
        # intercept without confusing it with an always-active feature.
        return [self.documented({"x": x, "y": y, "incremental_power_w": 5 + 2 * x + 3 * y,
                 "gpu_uuid": "GPU-test", "config": {"graphics_clock_mhz": 1200, "memory_clock_mhz": 1000}}
                , index) for index, (x, y) in enumerate(((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)))]

    def test_recovers_identifiable_rates_and_prohibits_unvalidated_mixed_prediction(self):
        model = fit_model(self.rows(), ["x", "y"])
        self.assertEqual(model["status"], "fitted", model["issues"])
        self.assertAlmostEqual(model["intercept_w"], 5)
        self.assertAlmostEqual(model["coefficients"]["x"], 2)
        self.assertAlmostEqual(model["coefficients"]["y"], 3)
        self.assertAlmostEqual(predict_power(model, {"x": 2, "y": 0}), 9)
        with self.assertRaises(ValueError):
            predict_power(model, {"x": 1, "y": 1})

    def test_heldout_mixed_validation_enables_prediction_and_failure_blocks_it(self):
        rows = self.rows()
        heldout = {**rows[0], "trial_id": "synthetic-heldout", "x": 2, "y": 1, "incremental_power_w": 12, "split": "validation"}
        model = fit_model(rows + [heldout], ["x", "y"])
        self.assertTrue(model["additive_validated"])
        self.assertAlmostEqual(predict_power(model, {"x": 2, "y": 1}), 12)
        heldout["incremental_power_w"] = 40
        model = fit_model(rows + [heldout], ["x", "y"])
        self.assertFalse(model["additive_validated"])
        self.assertIn("mixed_additivity_validation_failed", model["warnings"])

    def test_missing_feature_is_not_silently_zeroed(self):
        rows = self.rows()
        del rows[0]["y"]
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(len(model["skipped_rows"]), 1)
        self.assertEqual(model["calibration_rows"], 6)

    def test_collinear_and_unobserved_features_rejected(self):
        rows = [self.documented({"x": i, "y": i * 2, "incremental_power_w": i * 10}, i) for i in range(10)]
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(model["status"], "rejected")
        self.assertIn("rank_deficient_predictors", model["issues"])
        rows = [self.documented({"x": i, "y": 0, "incremental_power_w": i * 10}, i) for i in range(10)]
        self.assertIn("unobserved_features:y", fit_model(rows, ["x", "y"])["issues"])

    def test_scaling_does_not_confuse_physical_units_with_collinearity(self):
        rows = self.rows()
        for row in rows:
            row["x"] *= 1e12
            row["y"] *= 1e9
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(model["status"], "fitted")
        self.assertAlmostEqual(model["coefficients"]["x"] * 1e12, 2)
        self.assertAlmostEqual(model["coefficients"]["y"] * 1e9, 3)

    def test_separate_clock_and_device_strata_required(self):
        rows = self.rows()
        rows[1]["config"]["graphics_clock_mhz"] = 1000
        self.assertEqual(fit_model(rows, ["x", "y"])["status"], "rejected")
        rows = self.rows()
        rows[1]["gpu_uuid"] = "GPU-other"
        self.assertIn("multiple_devices_require_separate_models", fit_model(rows, ["x", "y"])["issues"])

    def test_insufficient_degrees_of_freedom_and_extrapolation_rejected(self):
        model = fit_model(self.rows()[:3], ["x", "y"])
        self.assertEqual(model["status"], "rejected")
        model = fit_model(self.rows(), ["x", "y"])
        with self.assertRaises(ValueError):
            predict_power(model, {"x": 4, "y": 0})
        with self.assertRaises(ValueError):
            predict_power(model, {"x": 1})

    def test_feature_units_and_measurement_provenance_are_required(self):
        rows = self.rows()
        rows[0].pop("feature_units")
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(model["skipped_rows"][0]["reason"], "explicit_feature_units_and_provenance_required")
        for row in rows:
            row["config"]["graphics_clock_mhz"] = None
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(model["status"], "rejected")

    def test_single_holdout_does_not_validate_other_mixed_vectors(self):
        rows = self.rows()
        heldout = {**rows[0], "trial_id": "synthetic-heldout", "x": 2, "y": 1, "incremental_power_w": 12, "split": "validation"}
        model = fit_model(rows + [heldout], ["x", "y"])
        self.assertTrue(model["additive_validated"])
        with self.assertRaises(ValueError):
            predict_power(model, {"x": 1, "y": 2})
        self.assertIn("single_mixed_holdout_validates_only_that_feature_vector", model["warnings"])

    def test_mixed_hull_is_stricter_than_per_feature_bounding_box(self):
        rows = self.rows()
        holdouts = [{**rows[0], "trial_id": f"synthetic-heldout-{x}", "x": x, "y": x,
                     "incremental_power_w": 5 + 5 * x, "split": "validation"} for x in (1, 2)]
        model = fit_model(rows + holdouts, ["x", "y"])
        self.assertAlmostEqual(predict_power(model, {"x": 1.5, "y": 1.5}), 12.5)
        with self.assertRaises(ValueError):
            predict_power(model, {"x": 1.5, "y": 1.75})

    def test_duplicate_or_unidentified_holdout_cannot_claim_independence(self):
        rows = self.rows()
        duplicate = {**rows[0], "x": 1, "y": 1, "incremental_power_w": 10, "split": "validation"}
        model = fit_model(rows + [duplicate], ["x", "y"])
        self.assertFalse(model["additive_validated"])
        self.assertEqual(model["validation_rows"], 0)
        self.assertIn("duplicate_measurement_identity", model["skipped_rows"][0]["reason"])
        duplicate.pop("trial_id")
        model = fit_model(rows + [duplicate], ["x", "y"])
        self.assertFalse(model["additive_validated"])
        self.assertIn("mixed_holdout_independence_requires_distinct_measured_trial_identities", model["warnings"])

    def test_analysis_flags_do_not_replace_recomputed_numeric_evidence(self):
        rows = self.rows()
        for row in rows:
            row.update(workload="tensor", valid=True, target_verified=True, verified_selection_eligible=True,
                       validation={"assessment": {"status": "pass", "suitable_verified": True}})
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(model["status"], "rejected")
        self.assertIn("analysis_row_requires", model["skipped_rows"][0]["reason"])

    def test_logical_and_physical_bytes_or_noncanonical_units_are_not_mixed(self):
        rows = []
        for index in range(5):
            row = self.documented({"hbm_gbps": index, "incremental_power_w": 5 + index}, index)
            row["feature_units"] = {"hbm_gbps": "GB/s"}
            row["feature_provenance"] = {"hbm_gbps": {"source": "synthetic fixture", "traffic_kind": "physical"}}
            rows.append(row)
        rows[0]["feature_provenance"]["hbm_gbps"]["traffic_kind"] = "logical"
        self.assertIn("mixed_feature_units_or_logical_physical_traffic_definitions", fit_model(rows, ["hbm_gbps"])["issues"])
        rows[0]["feature_units"]["hbm_gbps"] = "GiB/s"
        self.assertIn("incorrect_byte_rate_unit", fit_model(rows, ["hbm_gbps"])["skipped_rows"][0]["reason"])

    def test_nonfinite_validation_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_model(self.rows(), ["x", "y"], max_validation_relative_error=float("nan"))

    def test_different_binary_or_power_cap_cannot_share_a_model(self):
        rows = self.rows()
        rows[0]["benchmark_sha256"] = "different-binary"
        self.assertIn("multiple_power_or_software_strata_require_separate_models", fit_model(rows, ["x", "y"])["issues"])

    def test_failing_isolated_holdout_prevents_mixed_validation_claim(self):
        rows = self.rows()
        mixed = {**rows[0], "trial_id": "mixed-heldout", "x": 1, "y": 1, "incremental_power_w": 10, "split": "validation"}
        isolated = {**rows[0], "trial_id": "isolated-heldout", "x": 2, "y": 0, "incremental_power_w": 40, "split": "validation"}
        self.assertFalse(fit_model(rows + [mixed, isolated], ["x", "y"])["additive_validated"])

    def test_analysis_total_and_idle_models_use_separate_baseline_gates(self):
        from test_analysis import empirical_record
        from powermodeling.analysis import analyze_trial
        rows = []
        for index in range(6):
            record = empirical_record("GPU-baseline-model", 1200, workload="hbm", power=100 + 2 * index)
            record["phases"].pop("idle_post")
            row = analyze_trial(record)
            row.update(model_features={"x": index}, feature_units={"x": "synthetic-unit/s"},
                       feature_provenance={"x": "synthetic mathematical fixture"},
                       power_provenance="synthetic fixture only, not a measured GPU")
            rows.append(row)
        total = fit_model(rows, ["x"], target="board_power_w")
        self.assertEqual(total["status"], "fitted", total["issues"])
        self.assertEqual(total["energy_objective"], "measured total treatment power")
        idle = fit_model(rows, ["x"])
        self.assertEqual(idle["status"], "rejected")
        self.assertTrue(all("invalid_or_missing" in row["reason"] for row in idle["skipped_rows"]))
        for row in rows:
            row["incremental_power_w"] = row["board_power_w"] - 50
        idle = fit_model(rows, ["x"])
        self.assertTrue(all("matched_idle_baseline" in row["reason"] for row in idle["skipped_rows"]))

    def test_paired_model_requires_eligible_signed_contrast_and_both_orders(self):
        from test_analysis import paired_record
        from powermodeling.analysis import analyze_trial
        rows = []
        for index in range(6):
            record = paired_record(power=100 + 2 * index, reference_power=80, order="AB" if index % 2 else "BA")
            row = analyze_trial(record)
            row.update(model_features={"x": index}, feature_units={"x": "synthetic-unit/s"},
                       feature_provenance={"x": "synthetic mathematical fixture"},
                       power_provenance="synthetic fixture only, not a measured GPU")
            rows.append(row)
        model = fit_model(rows, ["x"], target="paired_active_reference_power_w")
        self.assertEqual(model["status"], "fitted", model["issues"])
        self.assertAlmostEqual(model["coefficients"]["x"], 2)
        import copy
        confounded = copy.deepcopy(rows)
        for index, row in enumerate(confounded):
            row["validation_binding"]["condition_id"] = "only-AB" if index < 3 else "only-BA"
            row["validation"]["profiler_evidence"]["condition_id"] = row["validation_binding"]["condition_id"]
            row["paired_active_reference_protocol"]["order"] = "AB" if index < 3 else "BA"
        confounded_model = fit_model(confounded, ["x"], target="paired_active_reference_power_w")
        self.assertIn("paired_reference_model_requires_equal_AB_BA_counts_per_condition_and_partition", confounded_model["issues"])
        unbalanced = fit_model(rows[:-1], ["x"], target="paired_active_reference_power_w")
        self.assertIn("paired_reference_model_requires_equal_valid_AB_and_BA_counts", unbalanced["issues"])
        for row in rows:
            row["paired_active_reference_protocol"]["order"] = "AB"
        model = fit_model(rows, ["x"], target="paired_active_reference_power_w")
        self.assertIn("paired_reference_model_requires_both_AB_and_BA_orders", model["issues"])
        rows[0]["paired_active_reference_eligible"] = False
        model = fit_model(rows, ["x"], target="paired_active_reference_power_w")
        self.assertIn("paired_reference_model_requires", model["skipped_rows"][0]["reason"])

    def test_different_treatment_allocation_contexts_require_separate_models(self):
        rows = self.rows()
        rows[0]["treatment_design_stratum"] = {"kind": "paired_active_reference", "paired_reference_context_allocated": True}
        self.assertIn("multiple_power_or_software_strata_require_separate_models", fit_model(rows, ["x", "y"])["issues"])


if __name__ == "__main__":
    unittest.main()
