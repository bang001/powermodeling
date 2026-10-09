"""Qualify NVML memory-scope observations independently of device power."""
import math
import statistics

SOURCES = {
    "memory_power_instant_w": ("driver_instantaneous", "NVML_FI_DEV_POWER_INSTANT"),
    "memory_power_average_w": ("one_second_average", "NVML_FI_DEV_POWER_AVERAGE"),
}
NUMERIC_FIELDS = ("power_w", "energy_j", "idle_power_w", "incremental_power_w",
                  "incremental_energy_j", "pj_per_logical_bit", "incremental_pj_per_logical_bit")


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def empty():
    return {"status": "unavailable", "scope": "memory", "source": None,
            "semantics": None, **dict.fromkeys(NUMERIC_FIELDS),
            "measurement_valid": False, "normalization_valid": False,
            "incremental_valid": False, "issues": [], "incremental_issues": [],
            "freshness_verified": False, "freshness_status": "unverified",
            "maximum_sensor_age_s": None, "freshness_maximum_age_s": None}


def phase_sensor(phase, stats, maximum_gap, integrate):
    """Check precisely the samples needed for the declared integration window."""
    result = empty()
    samples = {number(sample.get("t_s")): sample for sample in (phase or {}).get("samples", [])
               if isinstance(sample, dict) and number(sample.get("t_s")) is not None}
    samples = [samples[t] for t in sorted(samples)]
    start, end = number(stats.get("start_s")), number(stats.get("end_s"))
    if start is None or end is None or end <= start:
        result["issues"] = ["invalid_memory_sensor_window"]
        result["status"] = "invalid" if any(s.get("memory_power_w") is not None for s in samples) else "unavailable"
        return result
    covering = [s for s in samples if start <= s["t_s"] <= end]
    if not covering or covering[0]["t_s"] != start:
        before = [s for s in samples if s["t_s"] < start]
        covering = before[-1:] + covering
    if not covering or covering[-1]["t_s"] < end:
        after = [s for s in samples if s["t_s"] > end]
        covering += after[:1]
    if not any(s.get("memory_power_w") is not None for s in covering):
        result["issues"] = ["memory_sensor_unavailable"]
        return result
    issues = []
    if not covering or covering[0]["t_s"] > start or covering[-1]["t_s"] < end:
        issues.append("missing_memory_sensor_coverage")
    values = [number(s.get("memory_power_w")) for s in covering]
    if any(v is None for v in values): issues.append("missing_memory_sensor_coverage")
    if any(v is not None and v < 0 for v in values): issues.append("negative_memory_sensor_power")
    if any(b["t_s"] - a["t_s"] > maximum_gap for a, b in zip(covering, covering[1:])):
        issues.append("memory_sensor_telemetry_gap")
    sources = {s.get("memory_power_source") for s in covering}
    if any(source not in SOURCES for source in sources): issues.append("unknown_memory_sensor_source")
    if len(sources) != 1: issues.append("mixed_memory_sensor_sources")
    source = next(iter(sources)) if len(sources) == 1 else None
    source = source if source in SOURCES else None
    result["source"] = source
    result["semantics"] = SOURCES[source][0] if source else None
    timestamps, ages = [], []
    comparable_timestamps = 0
    # A one-second average can legitimately repeat between updates. This is
    # a stale-reading admission bound, not a claimed sensor refresh cadence.
    maximum_age = max(1.5, maximum_gap)
    result["freshness_maximum_age_s"] = maximum_age
    for sample in covering:
        selected_source = sample.get("memory_power_source")
        errors = sample.get("errors") or {}
        if errors.get("memory_power_w") or selected_source and errors.get(selected_source):
            issues.append("memory_sensor_query_error")
        metadata = (sample.get("field_metadata") or {}).get(selected_source)
        if metadata is not None:
            if not isinstance(metadata, dict) or metadata.get("scope") != "memory":
                issues.append("invalid_memory_sensor_scope")
                continue
            expected = SOURCES.get(selected_source)
            if expected and ("field" in metadata and metadata["field"] != expected[1] or
                             "semantics" in metadata and metadata["semantics"] != expected[0]):
                issues.append("inconsistent_memory_sensor_metadata")
            if "timestamp_us" in metadata:
                timestamp = number(metadata["timestamp_us"])
                if timestamp is None or timestamp < 0:
                    issues.append("invalid_memory_sensor_timestamp")
                else:
                    timestamps.append(timestamp)
                    clock = metadata.get("timestamp_clock")
                    if clock is not None and clock != "unix_epoch_microseconds":
                        issues.append("invalid_memory_sensor_timestamp_clock")
                    query_start = number(sample.get("query_realtime_start_s"))
                    query_end = number(sample.get("query_realtime_end_s"))
                    if clock == "unix_epoch_microseconds" and query_start is not None and query_end is not None:
                        if query_end < query_start:
                            issues.append("invalid_memory_sensor_query_timestamps")
                        else:
                            comparable_timestamps += 1
                            field_time = timestamp / 1e6
                            age = max(0.0, query_start - field_time)
                            ages.append(age)
                            if age > maximum_age:
                                issues.append("stale_memory_sensor_timestamp")
                            if field_time > query_end + .001:
                                issues.append("future_memory_sensor_timestamp")
            if "latency_us" in metadata and (number(metadata["latency_us"]) is None or metadata["latency_us"] < 0):
                issues.append("invalid_memory_sensor_latency")
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        issues.append("memory_sensor_timestamp_regression")
    result["maximum_sensor_age_s"] = max(ages) if ages else None
    result["freshness_status"] = "verified" if comparable_timestamps == len(covering) else "partial" if comparable_timestamps else "unverified"
    result["freshness_verified"] = comparable_timestamps == len(covering) and not issues
    if comparable_timestamps and issues:
        result["freshness_status"] = "invalid"
    energy = integrate(covering, "memory_power_w", start, end)
    if energy is None: issues.append("missing_memory_sensor_coverage")
    result["issues"] = sorted(set(issues))
    result["status"] = "invalid" if issues else "available"
    result["measurement_valid"] = not issues
    if not issues:
        result["energy_j"] = energy
        result["power_w"] = energy / (end - start)
    return result


def trial_sensor(record, result, phases, phase_records, maximum_gap, maximum_idle_drift, integrate, idle_at):
    if record.get("workload") != "hbm":
        return None
    sensor = phase_sensor(phase_records.get("measure"), phases["measure"], maximum_gap, integrate)
    count = number(result.get("counted_measure_logical_bytes"))
    exact = result.get("count_energy_time_alignment_exact") is True
    sensor["normalization_valid"] = bool(sensor["measurement_valid"] and result.get("valid") and exact and count is not None and count > 0)
    if sensor["normalization_valid"]:
        sensor["pj_per_logical_bit"] = sensor["energy_j"] / (count * 8) * 1e12
    pre = phase_sensor(phase_records.get("idle_pre"), phases["idle_pre"], maximum_gap, integrate)
    post = phase_sensor(phase_records.get("idle_post"), phases["idle_post"], maximum_gap, integrate)
    incremental_issues = ["idle_pre:" + issue for issue in pre["issues"]]
    incremental_issues += ["idle_post:" + issue for issue in post["issues"]]
    if sensor["source"] is None or pre["source"] != sensor["source"] or post["source"] != sensor["source"]:
        incremental_issues.append("idle_memory_sensor_source_mismatch")
    if not incremental_issues:
        idle = idle_at({**phases["idle_pre"], "memory_power_w": pre["power_w"]},
                       {**phases["idle_post"], "memory_power_w": post["power_w"]},
                       phases["measure"], "memory_power_w")
        if idle is None:
            incremental_issues.append("missing_idle_memory_sensor_baseline")
        else:
            sensor["idle_power_w"] = idle
            drift = abs(pre["power_w"] - post["power_w"]) / max((pre["power_w"] + post["power_w"]) / 2, 1e-9)
            if drift > maximum_idle_drift:
                incremental_issues.append("memory_idle_baseline_drift")
    if not sensor["measurement_valid"]: incremental_issues.append("invalid_treatment_memory_sensor")
    if result.get("operational_idle_increment_eligible") is not True:
        incremental_issues.append("unqualified_matched_idle_baseline")
    if not incremental_issues:
        sensor["incremental_power_w"] = sensor["power_w"] - sensor["idle_power_w"]
        sensor["incremental_energy_j"] = sensor["energy_j"] - sensor["idle_power_w"] * result["duration_s"]
        sensor["incremental_valid"] = True
        if sensor["normalization_valid"]:
            sensor["incremental_pj_per_logical_bit"] = sensor["incremental_energy_j"] / (count * 8) * 1e12
    sensor["incremental_issues"] = sorted(set(incremental_issues))
    return sensor


def group_sensor(trials, workload, seed, median_ci, observed_trials=None):
    if workload != "hbm":
        return None
    result = empty()
    observed = trials if observed_trials is None else observed_trials
    sensors = [trial["hbm_memory_power"] for trial in trials]
    measured = [s for s in sensors if s["measurement_valid"]]
    normalized = [s for s in measured if s["normalization_valid"]]
    incremental = [s for s in measured if s["incremental_valid"]]
    excluded_invalid = sum(trial.get("valid") is not True for trial in observed)
    excluded_duplicate = len(observed) - len(trials) - excluded_invalid
    result.update(observed_repeats=len(observed), valid_repeats=len(measured),
                  normalized_repeats=len(normalized), incremental_repeats=len(incremental),
                  board_invalid_repeats=excluded_invalid, duplicate_repeats_excluded=excluded_duplicate,
                  ci95=dict.fromkeys(NUMERIC_FIELDS))
    result["issues"] = sorted({issue for s in sensors for issue in s["issues"]})
    if excluded_invalid: result["issues"].append("board_invalid_repeats_excluded")
    if excluded_duplicate: result["issues"].append("duplicate_repeat_indices_excluded")
    result["incremental_issues"] = sorted({issue for s in sensors for issue in s["incremental_issues"]})
    sources = {s["source"] for s in measured}
    if len(sources) > 1:
        result["status"] = "invalid"
        result["issues"] = sorted(set(result["issues"] + ["mixed_memory_sensor_sources_across_repeats"]))
        return result
    if not measured:
        all_board_invalid = bool(observed) and excluded_invalid == len(observed)
        result["status"] = "invalid" if all_board_invalid or any(s["status"] == "invalid" for s in sensors) else "unavailable"
        return result
    result.update(status="available" if len(measured) == len(observed) else "partial",
                  source=measured[0]["source"], semantics=measured[0]["semantics"],
                  measurement_valid=True, normalization_valid=bool(normalized), incremental_valid=bool(incremental))
    result["freshness_verified"] = all(s["freshness_verified"] for s in measured)
    result["freshness_status"] = "verified" if result["freshness_verified"] else "partial" if any(s["freshness_status"] != "unverified" for s in measured) else "unverified"
    result["maximum_sensor_age_s"] = max((s["maximum_sensor_age_s"] for s in measured if s["maximum_sensor_age_s"] is not None), default=None)
    result["freshness_maximum_age_s"] = min(s["freshness_maximum_age_s"] for s in measured)
    for field in NUMERIC_FIELDS:
        population = normalized if field == "pj_per_logical_bit" else incremental if field.startswith("incremental_") else measured
        values = [s[field] for s in population if number(s[field]) is not None]
        result[field] = statistics.median(values) if values else None
        result["ci95"][field] = median_ci(values, seed)
    return result
