"""Clock-specific logical HBM bandwidth admission, separate from path evidence.

The DDR interface ceiling uses measured memory-clock telemetry and discovered
bus width. Logical energy-run bytes are not simultaneous physical DRAM traffic.
"""

import math
import statistics


DEFAULT_FRACTION = 0.80
FORMULA = "2 * memory_clock_mhz * 1e6 * memory_bus_width_bits / 8"
SCOPE = ("Sustained matching-window logical payload versus the clock-specific DDR interface ceiling; "
         "not simultaneous physical DRAM utilization or isolated memory energy. "
         "Observed-peak ratios and resource plateaus are diagnostics for HBM, not candidate gates.")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def validate_fraction(value):
    value = _number(value)
    if value is None or not 0 < value <= 1:
        raise ValueError("hbm_bandwidth_fraction must be finite and in (0, 1]")
    return value


def _mig_active(record):
    device = record.get("device") or {}
    cuda = record.get("cuda_device") or {}
    mode = device.get("mig_mode", cuda.get("mig_mode"))
    current = mode.get("current") if isinstance(mode, dict) else mode
    return (current in (1, True, "enabled", "Enabled")
            or (record.get("quality") or {}).get("mig_active") is True
            or any(str(value).startswith("MIG-") for value in
                   (cuda.get("uuid"), device.get("uuid"), (record.get("config") or {}).get("gpu_uuid"))))


def trial_bandwidth(record, trial, selected_samples, minimum_fraction=DEFAULT_FRACTION):
    minimum_fraction = validate_fraction(minimum_fraction)
    if trial.get("workload") != "hbm":
        return None
    reasons = []
    cuda = record.get("cuda_device") or {}
    bus = _number(cuda.get("memory_bus_width_bits"))
    if bus is None or bus <= 0 or not bus.is_integer() or bus % 8:
        bus = None
        reasons.append("missing_or_invalid_memory_bus_width_bits")
    samples = list(selected_samples)
    clocks = [_number(sample.get("memory_clock_mhz")) for sample in samples]
    complete_clock = bool(clocks) and all(clock is not None and clock > 0 for clock in clocks)
    clock = statistics.median(clocks) if complete_clock else None
    if not complete_clock:
        reasons.append("incomplete_measure_window_memory_clock_telemetry")
    if cuda.get("uuid") and trial.get("gpu_uuid") and cuda["uuid"] != trial["gpu_uuid"]:
        reasons.append("cuda_device_uuid_mismatch")
    if _mig_active(record):
        reasons.append("mig_memory_bus_mapping_unqualified")
    elif "mig_mode" in (record.get("device") or {}) and (record["device"].get("mig_mode") is None
            or isinstance(record["device"].get("mig_mode"), dict) and record["device"]["mig_mode"].get("current") is None):
        error = ((record["device"].get("errors") or {}).get("mig_mode") or {})
        if error.get("code") != 3 and error.get("type") != "NVMLError_NotSupported":
            reasons.append("unknown_mig_memory_bus_mapping")
    rate = _number(trial.get("throughput_bytes_s"))
    if rate is None or rate <= 0:
        reasons.append("missing_or_invalid_sustained_logical_bandwidth")
    peak = 2 * clock * 1e6 * bus / 8 if clock is not None and bus is not None else None
    if peak is not None and not math.isfinite(peak):
        peak = None
        reasons.append("invalid_theoretical_bandwidth")
    ratio = rate / peak if rate is not None and peak and peak > 0 else None
    evidence_complete = not reasons
    warnings = []
    cache = (trial.get("experiment_contract") or {}).get("read_cache_policy", "cg")
    access = (trial.get("experiment_contract") or {}).get("access", "read")
    if access == "read" and cache != "cg":
        reasons.append("hbm_read_cache_policy_requires_cg")
    if ratio is not None and ratio > 1 + 1e-9:
        warnings.append("logical_rate_above_interface_ceiling_cache_reuse_or_count_review")
    if ratio is not None and ratio < minimum_fraction:
        reasons.append("below_hbm_theoretical_bandwidth_fraction")
    return {"status": "pass" if not reasons else "inconclusive" if not evidence_complete else "fail",
            "eligible": not reasons, "evidence_complete": evidence_complete,
            "minimum_fraction": minimum_fraction, "theoretical_bytes_s": peak,
            "sustained_logical_bytes_s": rate, "fraction_of_theoretical": ratio,
            "achieved_memory_clock_mhz": clock, "memory_bus_width_bits": bus,
            "clock_source": "matching_measure_window_memory_clock_mhz", "formula": FORMULA,
            "physical_dram_utilization_measured": False, "reasons": reasons, "warnings": warnings, "scope": SCOPE}


def group_bandwidth(valid_trials, workload, minimum_fraction=DEFAULT_FRACTION):
    minimum_fraction = validate_fraction(minimum_fraction)
    if workload != "hbm":
        return None
    trials = list(valid_trials)
    rows = [trial.get("hbm_bandwidth") for trial in trials]
    present = [row for row in rows if isinstance(row, dict)]
    complete = bool(trials) and len(present) == len(trials) and all(row.get("evidence_complete") is True for row in present)
    reasons = sorted({reason for row in present for reason in row.get("reasons", [])
                      if reason != "below_hbm_theoretical_bandwidth_fraction"})
    warnings = sorted({warning for row in present for warning in row.get("warnings", [])})
    if len(present) != len(trials) or not trials:
        reasons.append("missing_valid_repeat_hbm_bandwidth_evidence")
    buses = {row.get("memory_bus_width_bits") for row in present}
    if len(buses) > 1:
        complete = False
        reasons.append("inconsistent_valid_repeat_memory_bus_width_bits")
    def median(field):
        values = [_number(row.get(field)) for row in present]
        return statistics.median(values) if complete and all(value is not None for value in values) else None
    ratio = median("fraction_of_theoretical")
    if ratio is not None and ratio < minimum_fraction:
        reasons.append("below_hbm_theoretical_bandwidth_fraction")
    eligible = complete and ratio is not None and not reasons
    return {"status": "pass" if eligible else "inconclusive" if not complete else "fail",
            "eligible": eligible, "evidence_complete": complete, "minimum_fraction": minimum_fraction,
            **{field: median(field) for field in ("theoretical_bytes_s", "sustained_logical_bytes_s",
                "fraction_of_theoretical", "achieved_memory_clock_mhz", "memory_bus_width_bits")},
            "clock_source": "matching_measure_window_memory_clock_mhz", "formula": FORMULA,
            "physical_dram_utilization_measured": False, "reasons": sorted(set(reasons)), "warnings": warnings, "scope": SCOPE,
            "valid_repeats": len(trials), "evidence_repeats": len(present),
            "aggregation": "Median of valid repeat logical bandwidth fractions; every valid repeat needs complete bus/clock evidence. Above-interface logical rates are warnings, not physical-traffic contradictions."}


def throughput_eligible(group, rate_field, observed_peak, observed_fraction):
    """HBM uses its absolute interface gate; other workloads retain observed bars."""
    if group.get("workload") == "hbm":
        return (group.get("hbm_bandwidth") or {}).get("eligible") is True
    rate = _number(group.get(rate_field))
    peak = _number(observed_peak)
    return rate is not None and peak is not None and peak > 0 and rate >= observed_fraction * peak


def threshold_metadata(group, observed_fraction):
    hbm = group.get("workload") == "hbm"
    evidence = group.get("hbm_bandwidth") or {}
    return {"throughput_threshold_basis": "clock_specific_hbm_theoretical_bandwidth" if hbm else "observed_peak",
            "throughput_threshold_fraction": evidence.get("minimum_fraction", DEFAULT_FRACTION) if hbm else observed_fraction,
            "observed_peak_threshold_fraction": observed_fraction,
            "hbm_bandwidth": group.get("hbm_bandwidth")}
