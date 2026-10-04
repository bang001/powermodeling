"""Identifiability-checked operational power regression.

Predictors must be measured rates with documented meaning. A regression on
logical memory bytes is an empirical workload model, not circuit/rail energy.
"""
from __future__ import annotations

import math
import json
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_FEATURES = ("tensor_tflops", "l1_gbps", "l2_gbps", "hbm_gbps")
FREQUENCY_FEATURES = {"graphics_clock_mhz", "memory_clock_mhz"}


def _mixed_activity(features, values):
    return sum(value > 0 for feature, value in zip(features, values) if feature not in FREQUENCY_FEATURES) > 1


def _measurement_identity(row):
    power = row.get("power_provenance") or {}
    identity = row.get("trial_id") or row.get("measurement_id") or (power.get("measurement_id") if isinstance(power, Mapping) else None)
    return identity if isinstance(identity, str) and identity else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _within_hull(values, observed, tolerance=1e-7):
    """Conservative convex-hull membership using a simplex-constrained fit.

    A bounding box is insufficient: two held-out mixed points do not validate
    every mixture in the rectangle between their independent feature limits.
    Normalization keeps FLOP and byte units from changing numerical tolerances.
    """
    import numpy as np
    points = np.asarray(observed, dtype=float)
    query = np.asarray(values, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return False
    scale = np.maximum(np.max(np.abs(points), axis=0), np.abs(query))
    scale = np.where(scale > 0, scale, 1)
    points, query = points / scale, query / scale
    matrix = np.vstack((points.T, np.ones(len(points))))
    rhs = np.append(query, 1)
    weights, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if np.min(weights) >= -tolerance and np.linalg.norm(matrix @ weights - rhs) <= tolerance:
        return True
    if len(points) == 1:
        return False
    # Projection onto a simplex constrains both nonnegative weights and sum=1.
    weights = np.full(len(points), 1 / len(points))
    lipschitz = float(np.linalg.norm(points, ord=2) ** 2)
    if lipschitz <= 0:
        return bool(np.linalg.norm(query) <= tolerance)
    for _ in range(10000):
        candidate = weights - points @ (weights @ points - query) / lipschitz
        sorted_values = np.sort(candidate)[::-1]
        cumulative = np.cumsum(sorted_values) - 1
        positive = np.where(sorted_values - cumulative / np.arange(1, len(points) + 1) > 0)[0]
        if not len(positive):
            return False
        index = positive[-1]
        next_weights = np.maximum(candidate - cumulative[index] / (index + 1), 0)
        if np.linalg.norm(next_weights @ points - query) <= tolerance:
            return True
        if np.linalg.norm(next_weights - weights) <= 1e-12:
            return False
        weights = next_weights
    return False


def _provenance_issue(row, features):
    units, sources = row.get("feature_units"), row.get("feature_provenance")
    if not isinstance(units, Mapping) or not isinstance(sources, Mapping):
        return "explicit_feature_units_and_provenance_required"
    for feature in features:
        unit, source = units.get(feature), sources.get(feature)
        if not isinstance(unit, str) or not unit.strip():
            return "missing_feature_unit:" + feature
        if feature == "tensor_tflops" and unit != "TFLOP/s":
            return "incorrect_tensor_unit; use_TFLOP/s"
        if feature in DEFAULT_FEATURES[1:] and unit != "GB/s":
            return "incorrect_byte_rate_unit; use_decimal_GB/s"
        description = source.get("source") if isinstance(source, Mapping) else source
        if not isinstance(description, str) or not description.strip():
            return "missing_feature_provenance:" + feature
        if feature in DEFAULT_FEATURES[1:] and (not isinstance(source, Mapping) or source.get("traffic_kind") not in ("logical", "physical")):
            return "declare_logical_or_physical_traffic_kind:" + feature
    power = row.get("power_provenance")
    description = power.get("source") if isinstance(power, Mapping) else power
    if not isinstance(description, str) or not description.strip():
        return "explicit_power_provenance_required"
    return None


def fit_model(rows: Iterable[Mapping[str, Any]], features: Sequence[str] | None = None,
              target: str = "incremental_power_w", *, include_intercept: bool = True,
              max_condition_number: float = 1e6, max_validation_relative_error: float = 0.10,
              allow_cross_clock: bool = False) -> dict[str, Any]:
    """Fit rates -> incremental device watts, using explicit feature values.

Rows use either feature keys directly or ``model_features``. ``split`` equal
to ``validation``/``test`` reserves a row; otherwise it enters calibration.
    Mixed predictions are restricted to the convex hull of independently
held-out mixed measurements that pass the validation tolerance. Each feature value must
be supplied explicitly, including measured/known zero; missing is not zero.
"""
    features = tuple(features or DEFAULT_FEATURES)
    result: dict[str, Any] = {
        "status": "rejected", "features": list(features), "target": target,
        "include_intercept": include_intercept, "coefficients": None, "intercept_w": None,
        "additive_validated": False, "issues": [], "warnings": [],
        "mixed_validation_feature_points": [],
        "interpretation": "Empirical incremental device power per stated measured-rate unit; coefficients do not isolate physical block leakage or switching energy.",
    }
    if not features or len(set(features)) != len(features):
        result["issues"].append("features_must_be_nonempty_and_unique")
        return result
    if (isinstance(max_condition_number, bool) or isinstance(max_validation_relative_error, bool)
            or not math.isfinite(max_condition_number) or not math.isfinite(max_validation_relative_error)
            or max_condition_number <= 1 or max_validation_relative_error <= 0):
        raise ValueError("conditioning and validation thresholds must be positive")
    try:
        import numpy as np
    except ImportError:
        result["issues"].append("numpy_required_for_model_fitting")
        return result
    calibration, validation = [], []
    skipped, seen_measurements = [], set()
    for index, row in enumerate(rows):
        source = row.get("model_features") or row
        values = [_number(source.get(feature)) for feature in features]
        y = _number(row.get(target))
        if row.get("valid") is False or any(value is None for value in values) or y is None:
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "invalid_or_missing_explicit_feature_or_target"})
            continue
        provenance_issue = _provenance_issue(row, features)
        if provenance_issue:
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": provenance_issue})
            continue
        # Analysis outputs carry an independently recomputed validation. Raw
        # profiler flags cannot make an unverified analysis row fit-eligible.
        analysis_style = any(key in row for key in ("workload", "target_verified", "validation", "phases"))
        if analysis_style:
            from .validation import validate_evidence
            evidence = (row.get("validation") or {}).get("profiler_evidence")
            assessment = validate_evidence(row.get("validation_binding") or row, evidence)
            if row.get("valid") is not True or row.get("verified_selection_eligible") is not True or assessment.get("status") != "pass" or assessment.get("suitable_verified") is not True:
                skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "analysis_row_requires_passing_numeric_profiler_and_matching_work_energy_window"})
                continue
        config = row.get("config") or {}
        uuid = row.get("gpu_uuid", config.get("gpu_uuid"))
        clock_values = [config.get(field, row.get(field)) for field in ("graphics_clock_mhz", "memory_clock_mhz")]
        if not isinstance(uuid, str) or not uuid or any((_number(clock) or 0) <= 0 for clock in clock_values):
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "explicit_device_and_fixed_clock_provenance_required"})
            continue
        if any(value < 0 for value in values):
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "negative_activity_rate"})
            continue
        identity = _measurement_identity(row)
        if identity is not None and identity in seen_measurements:
            skipped.append({"index": index, "trial_id": row.get("trial_id"), "reason": "duplicate_measurement_identity_cannot_establish_independent_calibration_or_holdout"})
            continue
        if identity is not None:
            seen_measurements.add(identity)
        entry = (values, y, row)
        (validation if row.get("split") in ("validation", "test") else calibration).append(entry)
    result.update(calibration_rows=len(calibration), validation_rows=len(validation), skipped_rows=skipped)
    if not calibration:
        result["issues"].append("no_complete_calibration_rows")
        return result
    definitions = {(tuple(row["feature_units"][feature] for feature in features),
                    tuple((row["feature_provenance"][feature].get("traffic_kind") if isinstance(row["feature_provenance"][feature], Mapping) else None) for feature in features))
                   for _, _, row in calibration + validation}
    if len(definitions) > 1:
        result["issues"].append("mixed_feature_units_or_logical_physical_traffic_definitions")
        return result
    result["feature_units"] = dict(calibration[0][2]["feature_units"])
    result["traffic_kind"] = {feature: (calibration[0][2]["feature_provenance"][feature].get("traffic_kind") if isinstance(calibration[0][2]["feature_provenance"][feature], Mapping) else None) for feature in features}
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
    execution_strata = {(row.get("benchmark_sha256"), json.dumps(row.get("measurement_stratum"), sort_keys=True)) for _, _, row in calibration + validation}
    if len(execution_strata) > 1:
        result["issues"].append("multiple_power_or_software_strata_require_separate_models")
        return result
    if len(strata) > 1 and not allow_cross_clock:
        result["issues"].append("multiple_clock_strata_require_separate_models_or_explicit_frequency_features")
        return result
    if allow_cross_clock and len(strata) > 1:
        if not {"graphics_clock_mhz", "memory_clock_mhz"}.issubset(features):
            result["issues"].append("cross_clock_model_requires_explicit_graphics_and_memory_frequency_features")
            return result
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
                  coefficient_units={feature: "W/(" + result["feature_units"][feature] + ")" for feature in features})
    if any(coefficient < 0 and abs(coefficient) > 2 * error for coefficient, error in zip(beta, coefficients_se)):
        result["warnings"].append("significant_negative_coefficient; attribution may be confounded or model misspecified")
    validation_results = []
    mixed_validation = []
    calibration_identity_known = all(_measurement_identity(row) is not None for _, _, row in calibration)
    for values, measured, row in validation:
        predicted = float(intercept + np.asarray(values) @ beta)
        # Near-zero targets use an absolute 1 W floor, recorded explicitly.
        relative_error = abs(predicted - measured) / max(abs(measured), 1.0)
        mixed = _mixed_activity(features, values)
        entry = {"trial_id": row.get("trial_id"), "measured_power_w": measured, "predicted_power_w": predicted,
                 "relative_error": relative_error, "mixed": mixed,
                 "features": dict(zip(features, values)),
                 "independent_measurement_identity": calibration_identity_known and _measurement_identity(row) is not None,
                 "within_calibration_feature_ranges": all(float(x[:, i].min()) <= value <= float(x[:, i].max()) for i, value in enumerate(values))}
        validation_results.append(entry)
        if mixed:
            mixed_validation.append(entry)
    result["validation"] = validation_results
    result["validation_relative_error_denominator_floor_w"] = 1.0
    result["max_validation_relative_error"] = max_validation_relative_error
    if mixed_validation:
        passed = all(entry["relative_error"] <= max_validation_relative_error and entry["within_calibration_feature_ranges"] for entry in validation_results) and all(entry["independent_measurement_identity"] for entry in mixed_validation)
        result["additive_validated"] = passed
        result["mixed_validation_feature_points"] = [[entry["features"][feature] for feature in features] for entry in mixed_validation] if passed else []
        result["validation_scope"] = "convex hull of passing held-out mixed measurements only; additivity beyond those mixtures is unvalidated" if passed else "held-out mixed measurements failed; mixed additive prediction disabled"
        if passed and len(mixed_validation) == 1:
            result["warnings"].append("single_mixed_holdout_validates_only_that_feature_vector")
        if not passed:
            result["warnings"].append("mixed_additivity_validation_failed")
            if any(not entry["independent_measurement_identity"] for entry in mixed_validation):
                result["warnings"].append("mixed_holdout_independence_requires_distinct_measured_trial_identities")
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
    mixed = _mixed_activity(model["features"], [values[name] for name in model["features"]])
    if mixed and not model.get("additive_validated"):
        raise ValueError("mixed-workload additivity has no passing independent validation")
    if mixed and not _within_hull([values[name] for name in model["features"]], model.get("mixed_validation_feature_points", [])):
        raise ValueError("mixed feature vector lies outside the convex hull of passing independent mixed validation")
    if not allow_extrapolation:
        for name, value in values.items():
            low, high = model["calibration_feature_ranges"][name]
            if not low <= value <= high:
                raise ValueError(f"{name} lies outside measured calibration range")
    return float(model.get("intercept_w", 0) + sum(model["coefficients"][name] * value for name, value in values.items()))
