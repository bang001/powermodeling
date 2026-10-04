import importlib.util
import unittest

from powermodeling.model import fit_model, predict_power


@unittest.skipUnless(importlib.util.find_spec("numpy"), "model regression requires numpy")
class ModelTests(unittest.TestCase):
    def rows(self):
        # Multiple activity levels plus independent zero/control establish the
        # intercept without confusing it with an always-active feature.
        return [{"x": x, "y": y, "incremental_power_w": 5 + 2 * x + 3 * y,
                 "gpu_uuid": "GPU-test", "config": {"graphics_clock_mhz": 1200, "memory_clock_mhz": 1000}}
                for x, y in ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3))]

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
        heldout = {**rows[0], "x": 2, "y": 1, "incremental_power_w": 12, "split": "validation"}
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
        rows = [{"x": i, "y": i * 2, "incremental_power_w": i * 10} for i in range(10)]
        model = fit_model(rows, ["x", "y"])
        self.assertEqual(model["status"], "rejected")
        self.assertIn("rank_deficient_predictors", model["issues"])
        rows = [{"x": i, "y": 0, "incremental_power_w": i * 10} for i in range(10)]
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


if __name__ == "__main__":
    unittest.main()
