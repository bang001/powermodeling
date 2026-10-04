"""Identifiability-checked operational power regression.

Predictors must be measured rates with documented meaning. A regression on
logical memory bytes is an empirical workload model, not circuit/rail energy.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_FEATURES = ("tensor_tflops", "l1_gbps", "l2_gbps", "hbm_gbps")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def fit_model(rows: Iterable[Mapping[str, Any]], features: Sequence[str] | None = None,
              target: str = "incremental_power_w", *, include_intercept: bool = True,
              max_condition_number: float = 1e6, max_validation_relative_error: float = 0.10,
              allow_cross_clock: bool = False) -> dict[str, Any]:
    """Fit rates -> incremental device watts, using explicit feature values.

Rows use either feature keys directly or ``model_features``. ``split`` equal
to ``validation``/``test`` reserves a row; otherwise it enters calibration.
Mixed-workload extrapolation stays disabled until independently held-out
mixed measurements pass the validation tolerance. Each feature value must
be supplied explicitly, including measured/known zero; missing is not zero.
"""
    features = tuple(features or DEFAULT_FEATURES)
    result: dict[str, Any] = {
        "status": "rejected", "features": list(features), "target": target,
        "include_intercept": include_intercept, "coefficients": None, "intercept_w": None,
        "additive_validated": False, "issues": [], "warnings": [],
        "interpretation": "Empirical incremental device power per stated measured-rate unit; coefficients do not isolate physical block leakage or switching energy.",
    }
    if not features or len(set(features)) != len(features):
        result["issues"].append("features_must_be_nonempty_and_unique")
        return result
    if max_condition_number <= 1 or max_validation_relative_error <= 0:
        raise ValueError("conditioning and validation thresholds must be positive")
    try:
        import numpy as np
    except ImportError:
        result["issues"].append("numpy_required_for_model_fitting")
        return result
    calibration, validation = [], []
    skipped = []
    for index, row in enumerate(rows):
        source = row.get("model_features") or row
        values = [_number(source.get(feature)) for feature in features]
        y = _number(row.get(target))
        if row.get("valid") is False or any(value is None for value in values) or y is None:
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "invalid_or_missing_explicit_feature_or_target"})
            continue
        if any(value < 0 for value in values):
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "negative_activity_rate"})
            continue
        entry = (values, y, row)
        (validation if row.get("split") in ("validation", "test") else calibration).append(entry)
    result.update(calibration_rows=len(calibration), validation_rows=len(validation), skipped_rows=skipped)
    if not calibration:
        result["issues"].append("no_complete_calibration_rows")
        return result
    strata = set()
    for _, _, row in calibration + validation:
        config = row.get("config") or {}
        strata.add((row.get("gpu_uuid", config.get("gpu_uuid")),
                    config.get("graphics_clock_mhz", row.get("graphics_clock_mhz")),
                    config.get("memory_clock_mhz", row.get("memory_clock_mhz"))))
    gpu_ids = {stratum[0] for stratum in strata if stratum[0] is not None}
    if len(gpu_ids) > 1:
        result["issues"].append("multiple_devices_require_separate_models")
        return result
    if len(strata) > 1 and not allow_cross_clock:
        result["issues"].append("multiple_clock_strata_require_separate_models_or_explicit_frequency_features")
        return result
    if allow_cross_clock and len(strata) > 1:
        result["warnings"].append("cross_clock_model_is_empirical; include frequency and voltage effects explicitly")
    x = np.asarray([entry[0] for entry in calibration], dtype=float)
    y = np.asarray([entry[1] for entry in calibration], dtype=float)
    scales = np.sqrt(np.mean(x * x, axis=0))
    inactive = [feature for feature, scale in zip(features, scales) if scale <= 0]
    if inactive:
        result["issues"].append("unobserved_features:" + ",".join(inactive))
        return result
    normalized = x / scales
    design = np.column_stack((np.ones(len(x)), normalized)) if include_intercept else normalized
    columns = design.shape[1]
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    result.update(rank=rank, parameters=columns, condition_number=condition,
                  conditioning_basis="RMS-scaled predictors; optional unit intercept", feature_scales=dict(zip(features, scales.tolist())))
    if len(x) <= columns:
        result["issues"].append("insufficient_residual_degrees_of_freedom")
    if rank < columns:
        result["issues"].append("rank_deficient_predictors")
    if condition is None or condition > max_condition_number:
        result["issues"].append("ill_conditioned_predictors")
    if result["issues"]:
        return result
    normalized_beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(normalized_beta[0]) if include_intercept else 0.0
    beta = normalized_beta[1:] / scales if include_intercept else normalized_beta / scales
    fitted = intercept + x @ beta
    residuals = y - fitted
    sse = float(residuals @ residuals)
    variance = sse / (len(x) - columns)
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    coefficients_se = standard_errors[1:] / scales if include_intercept else standard_errors / scales
    total_variance = float(((y - y.mean()) ** 2).sum())
    result.update(status="fitted", coefficients=dict(zip(features, beta.tolist())), intercept_w=intercept,
                  coefficient_standard_errors=dict(zip(features, coefficients_se.tolist())),
                  intercept_standard_error_w=float(standard_errors[0]) if include_intercept else None,
                  residual_rmse_w=math.sqrt(sse / len(x)), residual_max_abs_w=float(np.max(np.abs(residuals))),
                  residuals_w=residuals.tolist(), r_squared=1 - sse / total_variance if total_variance > 0 else None,
                  calibration_feature_ranges={feature: [float(x[:, i].min()), float(x[:, i].max())] for i, feature in enumerate(features)},
                  validation_scope="isolated/calibration observations only; mixed additive prediction disabled",
                  feature_units={feature: "W per " + feature for feature in features})
    if any(coefficient < 0 and abs(coefficient) > 2 * error for coefficient, error in zip(beta, coefficients_se)):
        result["warnings"].append("significant_negative_coefficient; attribution may be confounded or model misspecified")
    validation_results = []
    mixed_validation = []
    for values, measured, row in validation:
        predicted = float(intercept + np.asarray(values) @ beta)
        # Near-zero targets use an absolute 1 W floor, recorded explicitly.
        relative_error = abs(predicted - measured) / max(abs(measured), 1.0)
        mixed = sum(value > 0 for value in values) > 1
        entry = {"trial_id": row.get("trial_id"), "measured_power_w": measured, "predicted_power_w": predicted,
                 "relative_error": relative_error, "mixed": mixed,
                 "within_calibration_feature_ranges": all(float(x[:, i].min()) <= value <= float(x[:, i].max()) for i, value in enumerate(values))}
        validation_results.append(entry)
        if mixed:
            mixed_validation.append(entry)
    result["validation"] = validation_results
    result["validation_relative_error_denominator_floor_w"] = 1.0
    result["max_validation_relative_error"] = max_validation_relative_error
    if mixed_validation:
        passed = all(entry["relative_error"] <= max_validation_relative_error and entry["within_calibration_feature_ranges"] for entry in mixed_validation)
        result["additive_validated"] = passed
        result["validation_scope"] = "held-out mixed measurements passed within observed feature ranges" if passed else "held-out mixed measurements failed; mixed additive prediction disabled"
        if not passed:
            result["warnings"].append("mixed_additivity_validation_failed")
    else:
        result["warnings"].append("no_independent_mixed_validation; isolated measurements cannot establish mixed-workload additivity")
    return result


def predict_power(model: Mapping[str, Any], features: Mapping[str, Any], *, allow_extrapolation: bool = False) -> float:
    """Predict only in the validated/calibrated scope; raise outside it."""
    if model.get("status") != "fitted" or not model.get("coefficients"):
        raise ValueError("model was not fitted successfully")
    values = {name: _number(features.get(name)) for name in model["features"]}
    if any(value is None or value < 0 for value in values.values()):
        raise ValueError("supply every explicit nonnegative measured feature rate")
    if sum(value > 0 for value in values.values()) > 1 and not model.get("additive_validated"):
        raise ValueError("mixed-workload additivity has no passing independent validation")
    if not allow_extrapolation:
        for name, value in values.items():
            low, high = model["calibration_feature_ranges"][name]
            if not low <= value <= high:
                raise ValueError(f"{name} lies outside measured calibration range")
    return float(model.get("intercept_w", 0) + sum(model["coefficients"][name] * value for name, value in values.items()))
