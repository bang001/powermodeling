"""Synthetic regression admission and prediction boundaries; no GPU data."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from powermodeling.analysis import analyze_trial
from powermodeling.cli import main
from powermodeling.model import fit_model, predict_power
import test_analysis


class ModelScopeTests(unittest.TestCase):
    def rows(self):
        return [{
            "trial_id": f"cal-{x}", "gpu_uuid": "GPU-scope-test",
            "config": {"graphics_clock_mhz": 1200, "memory_clock_mhz": 1593},
            "temperature_c": 50, "x": x, "incremental_power_w": 5 + 2 * x,
            "feature_units": {"x": "synthetic-unit/s"},
            "feature_provenance": {"x": "synthetic review fixture"},
            "power_provenance": "synthetic watts, not measured GPU power",
        } for x in range(4)]

    def context(self, **changes):
        return {"gpu_uuid": "GPU-scope-test", "graphics_clock_mhz": 1200,
                "memory_clock_mhz": 1593, "temperature_c": 50, **changes}

    def failed_holdout_model(self):
        rows = self.rows()
        rows.append({**rows[2], "trial_id": "holdout", "split": "validation",
                     "incremental_power_w": 100})
        return fit_model(rows, ["x"])

    def test_failed_isolated_holdout_blocks_prediction_but_retains_calibration(self):
        model = self.failed_holdout_model()
        self.assertEqual(model["status"], "fitted")
        self.assertAlmostEqual(model["coefficients"]["x"], 2)
        self.assertAlmostEqual(model["validation"][0]["relative_error"], .91)
        with self.assertRaisesRegex(ValueError, "holdout"):
            predict_power(model, {"x": 2, **self.context()})
        self.assertEqual(model.get("holdout_validation_status"), "fail")

    def test_passing_isolated_holdout_is_reported(self):
        rows = self.rows()
        rows.append({**rows[2], "trial_id": "holdout", "split": "validation"})
        model = fit_model(rows, ["x"])
        self.assertEqual(model.get("holdout_validation_status"), "pass")
        self.assertAlmostEqual(predict_power(model, {"x": 2, **self.context()}), 9)

    def test_without_holdout_prediction_is_calibration_only(self):
        model = fit_model(self.rows(), ["x"])
        self.assertEqual(model.get("holdout_validation_status"), "not_provided")
        self.assertFalse(model["additive_validated"])
        self.assertAlmostEqual(predict_power(model, {"x": 2, **self.context()}), 9)

    def test_cli_saves_failed_holdout_diagnostics_and_returns_failure(self):
        rows = self.rows()
        rows.append({**rows[2], "trial_id": "holdout", "split": "validation",
                     "incremental_power_w": 100})
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "rows.json", Path(directory) / "model.json"
            source.write_text(json.dumps(rows))
            with redirect_stdout(io.StringIO()):
                code = main(["fit", "--input", str(source), "--features", "x", "--output", str(output)])
            self.assertEqual(code, 2)
            saved = json.loads(output.read_text())
            self.assertEqual(saved["status"], "fitted")
            self.assertEqual(saved["holdout_validation_status"], "fail")
            self.assertAlmostEqual(saved["coefficients"]["x"], 2)

    def test_wide_temperature_span_is_rejected(self):
        rows = self.rows()
        for row, temperature in zip(rows, (30, 40, 60, 80)):
            row["temperature_c"] = temperature
        model = fit_model(rows, ["x"])
        self.assertEqual(model["status"], "rejected")
        self.assertIn("temperature_span_exceeds_model_stratum", model["issues"])

    def test_partly_missing_temperature_is_rejected(self):
        rows = self.rows()
        rows[0].pop("temperature_c")
        model = fit_model(rows, ["x"])
        self.assertEqual(model["status"], "rejected")
        self.assertIn("incomplete_temperature_provenance", model["issues"])

    def test_missing_temperature_retains_unverified_legacy_math_fit(self):
        rows = self.rows()
        for row in rows:
            row.pop("temperature_c")
        model = fit_model(rows, ["x"])
        self.assertEqual(model["status"], "fitted")
        self.assertEqual((model.get("temperature_stratum") or {}).get("status"), "unverified")
        self.assertTrue(any("temperature" in warning for warning in model["warnings"]))

    def test_temperature_gate_uses_measure_extrema_instead_of_only_medians(self):
        rows = []
        for index in range(4):
            record = test_analysis.empirical_record("GPU-scope-test", 1200, workload="tensor",
                                                    power=100 + 2 * index, repeat=index)
            for sample in record["phases"]["measure"]["samples"]:
                sample["temperature_c"] = (45 if sample["t_s"] < 12 else 50) if index == 0 else (50 if sample["t_s"] < 12 else 55)
            row = analyze_trial(record)
            self.assertTrue(row["valid"], row["issues"])
            self.assertTrue(row["verified_selection_eligible"])
            row.update(model_features={"x": index}, feature_units={"x": "synthetic-unit/s"},
                       feature_provenance={"x": "synthetic activity"}, power_provenance="synthetic watts")
            rows.append(row)
        self.assertLessEqual(max(row["temperature_c"] for row in rows) - min(row["temperature_c"] for row in rows), 5)
        model = fit_model(rows, ["x"], target="board_power_w")
        self.assertEqual(model["status"], "rejected")
        self.assertIn("temperature_span_exceeds_model_stratum", model["issues"])

    def test_saved_scope_survives_json_and_accepts_matching_prediction(self):
        rows = self.rows()
        for row in rows:
            row["benchmark_sha256"] = "a" * 64
            row["measurement_stratum"] = {"power_limit_w": 400}
            row["treatment_design_stratum"] = {"kind": "synthetic-calibration"}
        model = json.loads(json.dumps(fit_model(rows, ["x"])))
        scope = model.get("execution_scope") or {}
        self.assertEqual(scope.get("gpu_uuid"), "GPU-scope-test")
        self.assertEqual(scope.get("requested_clock_pairs"), [{"graphics_clock_mhz": 1200, "memory_clock_mhz": 1593}])
        self.assertEqual(scope.get("measurement_stratum"), {"power_limit_w": 400})
        self.assertEqual(scope.get("benchmark_sha256"), "a" * 64)
        self.assertAlmostEqual(predict_power(model, {"x": 2, **self.context()}), 9)

    def test_prediction_requires_device_clock_context(self):
        model = fit_model(self.rows(), ["x"])
        for context in ({}, {"gpu_uuid": "GPU-scope-test"},
                        {"graphics_clock_mhz": 1200, "memory_clock_mhz": 1593}):
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, "context"):
                predict_power(model, {"x": 2, **context})

    def call_with_options(self, model, features, **options):
        try:
            return predict_power(model, features, **options)
        except TypeError as exc:
            self.fail(f"prediction context and diagnostic options must be supported: {exc}")

    def test_context_can_be_supplied_separately_from_activity_features(self):
        model = fit_model(self.rows(), ["x"])
        self.assertAlmostEqual(self.call_with_options(model, {"x": 2}, context=self.context()), 9)

    def test_explicit_unbound_option_retains_diagnostic_math_predictions(self):
        model = fit_model(self.rows(), ["x"])
        self.assertAlmostEqual(self.call_with_options(model, {"x": 2}, allow_unbound_context=True), 9)

    def test_prediction_context_options_require_real_booleans(self):
        model = fit_model(self.rows(), ["x"])
        for flag in ("allow_unbound_context", "allow_extrapolation"):
            for value in ("false", 0, 1, None):
                with self.subTest(flag=flag, value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    self.call_with_options(model, {"x": 2}, context=self.context(), **{flag: value})

    def test_unbound_option_cannot_override_explicit_wrong_gpu(self):
        model = fit_model(self.rows(), ["x"])
        with self.assertRaises(ValueError):
            self.call_with_options(model, {"x": 2}, context=self.context(gpu_uuid="GPU-other"),
                                   allow_unbound_context=True)

    def test_unbound_option_cannot_override_failed_holdout(self):
        with self.assertRaisesRegex(ValueError, "holdout"):
            self.call_with_options(self.failed_holdout_model(), {"x": 2}, allow_unbound_context=True)

    def test_activity_extrapolation_cannot_override_explicit_wrong_clock(self):
        model = fit_model(self.rows(), ["x"])
        with self.assertRaises(ValueError):
            self.call_with_options(model, {"x": 4}, context=self.context(memory_clock_mhz=2600),
                                   allow_extrapolation=True)

    def test_legacy_artifact_requires_refit_or_explicit_diagnostic_option(self):
        model = {"status": "fitted", "features": ["x"], "coefficients": {"x": 2},
                 "intercept_w": 5, "calibration_feature_ranges": {"x": [0, 3]}}
        with self.assertRaisesRegex(ValueError, "scope|context"):
            predict_power(model, {"x": 2, **self.context()})
        self.assertAlmostEqual(self.call_with_options(model, {"x": 2}, allow_unbound_context=True), 9)

    def test_explicit_device_or_clock_mismatch_is_rejected(self):
        model = fit_model(self.rows(), ["x"])
        for changes in ({"gpu_uuid": "GPU-other"}, {"graphics_clock_mhz": 2100},
                        {"memory_clock_mhz": 2600}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                predict_power(model, {"x": 2, **self.context(**changes)})

    def test_prediction_requires_known_temperature_and_respects_five_degree_span(self):
        model = fit_model(self.rows(), ["x"])
        missing = self.context()
        missing.pop("temperature_c")
        with self.assertRaisesRegex(ValueError, "temperature"):
            predict_power(model, {"x": 2, **missing})
        self.assertAlmostEqual(predict_power(model, {"x": 2, **self.context(temperature_c=55)}), 9)
        with self.assertRaisesRegex(ValueError, "temperature"):
            predict_power(model, {"x": 2, **self.context(temperature_c=55.1)})

    def cross_clock_rows(self):
        rows = []
        for graphics, memory in ((1000, 1000), (1000, 1400), (1400, 1000), (1400, 1400)):
            for x in range(3):
                row = copy.deepcopy(self.rows()[x])
                row.update(trial_id=f"{graphics}-{memory}-{x}",
                           config={"graphics_clock_mhz": graphics, "memory_clock_mhz": memory},
                           model_features={"x": x, "graphics_clock_mhz": graphics, "memory_clock_mhz": memory},
                           incremental_power_w=10 + 2 * x + .01 * graphics + .02 * memory)
                row["feature_units"].update(graphics_clock_mhz="MHz", memory_clock_mhz="MHz")
                row["feature_provenance"].update(graphics_clock_mhz="synthetic request", memory_clock_mhz="synthetic request")
                rows.append(row)
        return rows

    def test_cross_clock_features_must_match_requested_clocks(self):
        rows = self.cross_clock_rows()
        for row in rows:
            row["model_features"]["graphics_clock_mhz"] += 1
        model = fit_model(rows, ["x", "graphics_clock_mhz", "memory_clock_mhz"], allow_cross_clock=True)
        self.assertEqual(model["status"], "rejected")

    def test_cross_clock_prediction_uses_frequency_features_and_bound_gpu(self):
        model = fit_model(self.cross_clock_rows(), ["x", "graphics_clock_mhz", "memory_clock_mhz"], allow_cross_clock=True)
        self.assertEqual(model["status"], "fitted", model["issues"])
        self.assertTrue((model.get("execution_scope") or {}).get("cross_clock_model"))
        features = {"x": 1, **self.context(graphics_clock_mhz=1200, memory_clock_mhz=1100)}
        self.assertAlmostEqual(predict_power(model, features), 46)
        features["gpu_uuid"] = "GPU-other"
        with self.assertRaises(ValueError):
            predict_power(model, features)


if __name__ == "__main__":
    unittest.main()
