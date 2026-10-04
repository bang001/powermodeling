"""Analyze sustained NVML telemetry without pretending it isolates circuit power.

Timestamps in a record must share CLOCK_MONOTONIC's seconds timebase.  Device
power and an optional memory rail are separate observables; idle subtraction is
an operational baseline, not a measurement of transistor leakage.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

_TENSOR_WORKLOADS = {"tensor", "fp16_tensor", "tensor_fp16", "gemm"}
_MEMORY_WORKLOADS = {"l1", "l2", "hbm"}


@dataclass(frozen=True)
class AnalysisPolicy:
    trim_s: float = 2.0
    idle_trim_s: float = 1.0
    min_measure_s: float = 5.0
    min_idle_s: float = 4.0
    min_samples: int = 4
    max_sample_gap_s: float = 2.5
    clock_tolerance_fraction: float = 0.03
    max_clock_drift_fraction: float = 0.03
    max_temperature_drift_c: float = 5.0
    max_idle_drift_fraction: float = 0.10
    counter_disagreement_fraction: float = 0.10


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return statistics.median(numbers) if numbers else None


def _samples(phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    # Last observation wins if a reader emitted duplicate timestamps.
    indexed: dict[float, dict[str, Any]] = {}
    for item in phase.get("samples", []):
        if isinstance(item, dict) and (timestamp := _finite(item.get("t_s"))) is not None:
            indexed[timestamp] = item
    return [indexed[timestamp] for timestamp in sorted(indexed)]


def _interpolate(samples: list[dict[str, Any]], field: str, t: float) -> float | None:
    if not samples or t < float(samples[0]["t_s"]) or t > float(samples[-1]["t_s"]):
        return None
    for i, sample in enumerate(samples):
        current_t = float(sample["t_s"])
        if current_t == t:
            return _finite(sample.get(field))
        if current_t > t:
            prior = samples[i - 1]
            a, b = _finite(prior.get(field)), _finite(sample.get(field))
            if a is None or b is None:
                return None
            ratio = (t - float(prior["t_s"])) / (current_t - float(prior["t_s"]))
            return a + ratio * (b - a)
    return None


def _integrate(samples: list[dict[str, Any]], field: str, start: float, end: float) -> float | None:
    first, last = _interpolate(samples, field, start), _interpolate(samples, field, end)
    if first is None or last is None or end <= start:
        return None
    points = [(start, first)]
    for sample in samples:
        t = float(sample["t_s"])
        if start < t < end:
            value = _finite(sample.get(field))
            if value is None:
                return None
            points.append((t, value))
    points.append((end, last))
    return sum((b[0] - a[0]) * (a[1] + b[1]) * 0.5 for a, b in zip(points, points[1:]))


def _phase_stats(phase: Mapping[str, Any] | None, name: str, policy: AnalysisPolicy) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "issues": [], "warnings": []}
    if not phase:
        result["issues"].append(f"missing_phase:{name}")
        return result
    start, end = _finite(phase.get("start_s")), _finite(phase.get("end_s"))
    trim = policy.idle_trim_s if name.startswith("idle_") else policy.trim_s
    if start is None or end is None or end <= start or end - start <= trim * 2:
        result["issues"].append(f"invalid_phase_window:{name}")
        return result
    start, end = start + trim, end - trim
    duration = end - start
    samples = _samples(phase)
    selected = [s for s in samples if start <= float(s["t_s"]) <= end]
    # Include immediately neighboring observations only for boundary interpolation.
    left = [s for s in samples if float(s["t_s"]) < start]
    right = [s for s in samples if float(s["t_s"]) > end]
    covering = (left[-1:] + selected + right[:1])
    gaps = [float(b["t_s"]) - float(a["t_s"]) for a, b in zip(covering, covering[1:])]
    minimum = policy.min_idle_s if name.startswith("idle_") else policy.min_measure_s
    result.update(start_s=start, end_s=end, duration_s=duration, midpoint_s=(start + end) / 2,
                  samples=len(selected), sample_interval_median_s=_median(gaps),
                  max_sample_gap_s=max(gaps) if gaps else None)
    if duration + 1e-9 < minimum:
        result["issues"].append(f"short_plateau:{name}")
    if len(selected) < policy.min_samples:
        result["issues"].append(f"too_few_samples:{name}")
    if gaps and max(gaps) > policy.max_sample_gap_s:
        result["issues"].append(f"telemetry_gap:{name}")
    energy = _integrate(covering, "power_w", start, end)
    result["integrated_power_energy_j"] = energy
    result["integrated_power_w"] = energy / duration if energy is not None else None
    result["power_w"] = result["integrated_power_w"]
    if energy is None:
        result["issues"].append(f"missing_power_coverage:{name}")
    memory_energy = _integrate(covering, "memory_power_w", start, end)
    result["memory_rail_energy_j"] = memory_energy
    result["memory_power_w"] = memory_energy / duration if memory_energy is not None else None
    # Subtract an integer anchor before converting NVML's cumulative uint64
    # counter to float; subtracting two large floats can lose milli-joules.
    anchor = next((s.get("energy_mj") for s in covering if isinstance(s.get("energy_mj"), int)), 0)
    relative_counter_samples = []
    for sample in covering:
        counter = sample.get("energy_mj")
        relative_counter_samples.append({"t_s": sample["t_s"], "relative_energy_mj": counter - anchor if isinstance(counter, (int, float)) else None})
    start_counter = _interpolate(relative_counter_samples, "relative_energy_mj", start)
    end_counter = _interpolate(relative_counter_samples, "relative_energy_mj", end)
    counter_points = [_finite(s.get("relative_energy_mj")) for s in relative_counter_samples]
    counter_present = [x for x in counter_points if x is not None]
    counter_reset = any(b < a for a, b in zip(counter_present, counter_present[1:]))
    counter_energy = None
    if start_counter is not None and end_counter is not None:
        if counter_reset or end_counter < start_counter:
            result["warnings"].append(f"energy_counter_reset:{name}")
        else:
            counter_energy = (end_counter - start_counter) / 1000.0
    result["energy_counter_j"] = counter_energy
    if counter_energy is not None:
        result["power_w"] = counter_energy / duration
    if counter_energy is not None and energy is not None and energy > 0:
        disagreement = abs(counter_energy - energy) / energy
        result["counter_power_disagreement_fraction"] = disagreement
        if disagreement > policy.counter_disagreement_fraction:
            result["issues"].append(f"energy_counter_disagreement:{name}")
    power_values = [_finite(s.get("power_w")) for s in selected]
    changes = [float(b["t_s"]) for a, b in zip(selected, selected[1:])
               if _finite(a.get("power_w")) != _finite(b.get("power_w"))]
    change_gaps = [b - a for a, b in zip(changes, changes[1:])]
    result["distinct_power_values"] = len(set(v for v in power_values if v is not None))
    result["observed_power_change_interval_median_s"] = _median(change_gaps)
    result["power_update_cadence_note"] = "Changes are not proof of sensor refresh cadence; unchanged values may be steady power."
    for field in ("graphics_clock_mhz", "memory_clock_mhz", "temperature_c"):
        values = [_finite(s.get(field)) for s in selected]
        values = [v for v in values if v is not None]
        result[field] = statistics.median(values) if values else None
        result[f"{field}_min"] = min(values) if values else None
        result[f"{field}_max"] = max(values) if values else None
    result["_selected"] = selected
    return result


def _idle_at(pre: dict[str, Any], post: dict[str, Any], active: dict[str, Any], field: str) -> float | None:
    a, b = _finite(pre.get(field)), _finite(post.get(field))
    t0, t1, t = (_finite(pre.get("midpoint_s")), _finite(post.get("midpoint_s")),
                  _finite(active.get("midpoint_s")))
    if any(v is None for v in (a, b, t0, t1, t)) or t1 <= t0 or not t0 <= t <= t1:
        return None
    return a + (b - a) * (t - t0) / (t1 - t0)


def _throttle_mask(value: Any) -> int | None:
    try:
        if isinstance(value, str):
            return int(value, 0)
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def analyze_trial(record: Mapping[str, Any], policy: AnalysisPolicy | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Calculate operational incremental energy for one sustained trial.

The plateau rate uses the benchmark's measured operation count / benchmark
duration. It assumes a stationary loop, verified separately by clocks and
thermal telemetry; it never scales a whole-run count as if it were counted
inside the trimmed window.
"""
    policy = policy if isinstance(policy, AnalysisPolicy) else AnalysisPolicy(**(policy or {}))
    phase_records = record.get("phases", {})
    if isinstance(phase_records, list):
        phase_records = {p["name"]: p for p in phase_records if isinstance(p, dict) and "name" in p}
    phases = {name: _phase_stats(phase_records.get(name), name, policy)
              for name in ("idle_pre", "measure", "idle_post")}
    if phase_records.get("active_control"):
        phases["active_control"] = _phase_stats(phase_records["active_control"], "active_control", policy)
    measure, pre, post = phases["measure"], phases["idle_pre"], phases["idle_post"]
    issues = [issue for phase in phases.values() for issue in phase["issues"]]
    warnings = [warning for phase in phases.values() for warning in phase["warnings"]]
    config = dict(record.get("config") or {})
    benchmark = record.get("benchmark") or record.get("result") or {}
    device = record.get("device") or {}
    gpu_uuid = config.get("gpu_uuid") or device.get("uuid") or record.get("gpu_uuid")
    for field in ("graphics_clock_mhz", "memory_clock_mhz"):
        requested, actual = _finite(config.get(field)), _finite(measure.get(field))
        if requested is not None and requested > 0:
            if actual is None:
                issues.append(f"missing_clock_telemetry:{field}")
            elif abs(actual - requested) / requested > policy.clock_tolerance_fraction:
                issues.append(f"clock_mismatch:{field}")
        else:
            warnings.append(f"uncontrolled_clock:{field}")
        low, high = _finite(measure.get(f"{field}_min")), _finite(measure.get(f"{field}_max"))
        if actual and low is not None and high is not None and (high - low) / actual > policy.max_clock_drift_fraction:
            issues.append(f"clock_drift:{field}")
    temp_low, temp_high = _finite(measure.get("temperature_c_min")), _finite(measure.get("temperature_c_max"))
    if temp_low is not None and temp_high is not None and temp_high - temp_low > policy.max_temperature_drift_c:
        issues.append("temperature_drift")
    selected = measure.get("_selected", [])
    if selected and any(_finite(sample.get("temperature_c")) is None for sample in selected):
        issues.append("missing_temperature_telemetry")
    if selected and any(_throttle_mask(sample.get("throttle_reasons")) is None for sample in selected):
        issues.append("missing_throttling_telemetry")
    # Idle (bit 0) and an explicitly configured applications clock (bit 1)
    # aren't an active throttling cause. Other nonzero reasons are retained.
    throttle_masks = [_throttle_mask(s.get("throttle_reasons")) for s in selected]
    throttle_masks = [m for m in throttle_masks if m is not None]
    effective_mask = 0
    for mask in throttle_masks:
        effective_mask |= mask & ~0x3
    if effective_mask:
        issues.append("active_throttling")
    quality = record.get("quality") or {}
    if quality.get("valid") is False:
        issues.append("runner_quality_invalid")
    for key in ("interference_detected", "mig_active", "unsafe_short_duration"):
        if quality.get(key) or config.get(key):
            issues.append(key)
    measurement_pid = quality.get("measurement_pid")
    other_pids = set()
    for sample in selected:
        for field in ("compute_processes", "graphics_processes"):
            for process in sample.get(field) or []:
                pid = process.get("pid") if isinstance(process, dict) else process
                if pid is not None and pid != measurement_pid:
                    other_pids.add(pid)
    processes = quality.get("other_compute_processes", record.get("other_compute_processes"))
    if processes:
        issues.append("other_compute_processes")
    if other_pids:
        issues.append("other_gpu_processes")
    mig_mode = device.get("mig_mode") or {}
    if isinstance(mig_mode, dict) and mig_mode.get("current"):
        issues.append("mig_active")
    if (record.get("provenance") or {}).get("mps_environment"):
        issues.append("mps_environment")
    if record.get("status") != "complete" or benchmark.get("error"):
        issues.append("benchmark_failed")
    idle_power = _idle_at(pre, post, measure, "power_w")
    idle_memory_power = _idle_at(pre, post, measure, "memory_power_w")
    pre_power, post_power = _finite(pre.get("power_w")), _finite(post.get("power_w"))
    if pre_power is not None and post_power is not None:
        denominator = max((pre_power + post_power) / 2, 1e-9)
        if abs(pre_power - post_power) / denominator > policy.max_idle_drift_fraction:
            issues.append("idle_baseline_drift")
    duration = _finite(measure.get("duration_s"))
    total = _finite(measure.get("energy_counter_j"))
    energy_source = "energy_counter_delta"
    if total is None:
        total, energy_source = _finite(measure.get("integrated_power_energy_j")), "power_trapezoid"
    board_power = total / duration if total is not None and duration and duration > 0 else None
    idle_energy = idle_power * duration if idle_power is not None and duration else None
    incremental = total - idle_energy if total is not None and idle_energy is not None else None
    if idle_power is None:
        issues.append("missing_idle_baseline")
    if incremental is not None and incremental < 0:
        issues.append("negative_incremental_energy")
    device_duration = _finite(benchmark.get("duration_s"))
    measured_duration = _finite(benchmark.get("host_duration_s")) or device_duration
    if device_duration and measured_duration and abs(device_duration - measured_duration) / measured_duration > 0.05:
        issues.append("host_device_duration_disagreement")
    operations, logical_bytes = _finite(benchmark.get("operations")), _finite(benchmark.get("logical_bytes"))
    ops_rate = operations / measured_duration if operations is not None and measured_duration and measured_duration > 0 else None
    byte_rate = logical_bytes / measured_duration if logical_bytes is not None and measured_duration and measured_duration > 0 else None
    if measured_duration is None or measured_duration <= 0:
        issues.append("missing_benchmark_duration")
    if record.get("workload") in _TENSOR_WORKLOADS and (ops_rate is None or ops_rate <= 0):
        issues.append("missing_operation_count")
    if record.get("workload") in _MEMORY_WORKLOADS and (byte_rate is None or byte_rate <= 0):
        issues.append("missing_logical_byte_count")
    incremental_power = incremental / duration if incremental is not None and duration else None
    rail_energy = _finite(measure.get("memory_rail_energy_j"))
    rail_incremental = rail_energy - idle_memory_power * duration if rail_energy is not None and idle_memory_power is not None and duration else None
    rail_sources = {s.get("memory_power_source") for phase in phases.values() for s in phase.get("_selected", []) if s.get("memory_power_w") is not None and s.get("memory_power_source")}
    if len(rail_sources) > 1:
        warnings.append("memory_rail_source_changed; rail incremental energy withheld")
        rail_incremental = None
    validation = record.get("validation") or {}
    power_limit = (_median(s.get("power_limit_w") for s in selected)
                   or _finite(record.get("power_limit_w", device.get("power_limit_w", config.get("power_limit_w"))))
                   or _finite((device.get("sample") or {}).get("power_limit_w")))
    tensor_peak = None
    if record.get("workload") in _TENSOR_WORKLOADS and measure.get("graphics_clock_mhz") is not None:
        try:
            from .profiles import theoretical_tensor_tflops
            tensor_peak = theoretical_tensor_tflops(record.get("cuda_device") or device, measure["graphics_clock_mhz"])
        except (ValueError, KeyError, TypeError):
            warnings.append("tensor_theoretical_peak_unavailable")
    baseline_deltas = {}
    baseline_clock_matched = True
    baseline_temperature_matched = True
    for name, baseline_phase in (("idle_pre", pre), ("idle_post", post)):
        deltas = {}
        for field in ("graphics_clock_mhz", "memory_clock_mhz"):
            active_clock, idle_clock = _finite(measure.get(field)), _finite(baseline_phase.get(field))
            deltas[field] = active_clock - idle_clock if active_clock is not None and idle_clock is not None else None
            if active_clock is None or idle_clock is None or active_clock <= 0 or abs(active_clock - idle_clock) / active_clock > policy.clock_tolerance_fraction:
                baseline_clock_matched = False
        active_temp, idle_temp = _finite(measure.get("temperature_c")), _finite(baseline_phase.get("temperature_c"))
        deltas["temperature_c"] = active_temp - idle_temp if active_temp is not None and idle_temp is not None else None
        if active_temp is None or idle_temp is None or abs(active_temp - idle_temp) > policy.max_temperature_drift_c:
            baseline_temperature_matched = False
        baseline_deltas[name] = deltas
    if not baseline_clock_matched:
        warnings.append("idle_active_clock_mismatch_or_unavailable; incremental power includes activation and frequency-state differences")
    if not baseline_temperature_matched:
        warnings.append("idle_active_temperature_mismatch_or_unavailable; leakage/thermal drift can confound subtraction")
    target_verified = bool(validation.get("memory_target_verified")) if record.get("workload") in _MEMORY_WORKLOADS | {"l2_latency"} else bool(validation.get("tensor_instructions_verified"))
    result: dict[str, Any] = {
        "trial_id": record.get("trial_id"), "workload": record.get("workload"), "gpu_uuid": gpu_uuid,
        "gpu_name": device.get("name", record.get("gpu_name")), "config": config,
        "valid": not issues, "issues": sorted(set(issues)), "warnings": sorted(set(warnings)),
        "target_verified": target_verified,
        "validation": validation, "duration_s": duration, "benchmark_duration_s": measured_duration,
        "benchmark_device_duration_s": device_duration,
        "throughput_timebase": "benchmark host monotonic duration" if benchmark.get("host_duration_s") else "benchmark duration (host timebase unavailable)",
        "operation_unit": benchmark.get("operation_unit"),
        "diagnostic_only": record.get("workload") not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS,
        "clock_comparison_controlled": all((_finite(config.get(field)) or 0) > 0 for field in ("graphics_clock_mhz", "memory_clock_mhz")),
        "throughput_ops_s": ops_rate, "throughput_bytes_s": byte_rate,
        "device_event_throughput_ops_s": operations / device_duration if operations is not None and device_duration and device_duration > 0 else None,
        "device_event_throughput_bytes_s": logical_bytes / device_duration if logical_bytes is not None and device_duration and device_duration > 0 else None,
        "tensor_peak_tflops_at_achieved_clock": tensor_peak,
        "tensor_utilization_vs_dense_clock_peak": ops_rate / 1e12 / tensor_peak if ops_rate is not None and tensor_peak and tensor_peak > 0 else None,
        "board_power_w": board_power, "idle_power_w": idle_power, "incremental_power_w": incremental_power,
        "total_energy_j": total, "idle_energy_j": idle_energy, "incremental_energy_j": incremental,
        "integrated_power_energy_j": measure.get("integrated_power_energy_j"),
        "energy_counter_j": measure.get("energy_counter_j"), "energy_source": energy_source,
        "pj_per_op": incremental_power / ops_rate * 1e12 if incremental_power is not None and ops_rate and ops_rate > 0 and record.get("workload") in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS else None,
        "pj_per_logical_byte": incremental_power / byte_rate * 1e12 if incremental_power is not None and byte_rate and byte_rate > 0 else None,
        "total_pj_per_op": board_power / ops_rate * 1e12 if board_power is not None and ops_rate and ops_rate > 0 and record.get("workload") in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS else None,
        "total_pj_per_logical_byte": board_power / byte_rate * 1e12 if board_power is not None and byte_rate and byte_rate > 0 else None,
        "memory_rail_power_w": measure.get("memory_power_w"), "memory_rail_idle_power_w": idle_memory_power,
        "memory_rail_energy_j": rail_energy, "memory_rail_incremental_energy_j": rail_incremental,
        "memory_rail_incremental_power_w": rail_incremental / duration if rail_incremental is not None and duration else None,
        "memory_rail_sources": sorted(rail_sources),
        "graphics_clock_mhz": measure.get("graphics_clock_mhz"), "memory_clock_mhz": measure.get("memory_clock_mhz"),
        "temperature_c": measure.get("temperature_c"), "throttle_reasons_mask": effective_mask,
        "other_gpu_process_pids": sorted(other_pids),
        "power_limit_w": power_limit,
        "idle_fraction_of_measured_power": idle_power / board_power if idle_power is not None and board_power and board_power > 0 else None,
        "idle_fraction_of_power_limit": idle_power / power_limit if idle_power is not None and power_limit and power_limit > 0 else None,
        "power_limit_utilization": board_power / power_limit if board_power is not None and power_limit and power_limit > 0 else None,
        "power_scope": (record.get("telemetry") or {}).get("power_scope", device.get("power_scope_note", "NVML reported device power; scope depends on SKU/API")),
        "power_usage_semantics": device.get("power_usage_semantics"),
        "baseline_kind": "powered-idle baseline; includes leakage, clocks, refresh, and device background activity",
        "baseline_clock_matched": baseline_clock_matched,
        "baseline_temperature_matched": baseline_temperature_matched,
        "active_minus_idle_state_deltas": baseline_deltas,
        "dynamic_attribution_eligible": not issues and target_verified and baseline_clock_matched and baseline_temperature_matched,
        "dynamic_attribution_note": "Eligibility indicates matched-baseline/profiler prerequisites only. It does not prove isolation of physical switching power or static leakage.",
        "attribution": "whole-device incremental energy per benchmark operation or logical byte; not isolated block energy",
        "policy": asdict(policy), "phases": {}, "model_features": record.get("model_features"),
    }
    if record.get("workload") == "l2_latency":
        result["latency_diagnostics"] = dict(benchmark)
    for name, phase in phases.items():
        result["phases"][name] = {k: v for k, v in phase.items() if not k.startswith("_")}
    if "active_control" in phases:
        control = phases["active_control"]
        control_idle = _idle_at(pre, post, control, "power_w")
        control_power = _finite(control.get("power_w"))
        control_increment = control_power - control_idle if control_power is not None and control_idle is not None else None
        result["active_control_incremental_power_w"] = control_increment
        result["control_subtracted_power_w"] = incremental_power - control_increment if incremental_power is not None and control_increment is not None else None
        result["control_subtraction_note"] = "Instruction and scheduling differences can bias this counterfactual; it is not pure block power."
    return result


_METRICS = ("board_power_w", "idle_power_w", "incremental_power_w", "throughput_ops_s", "throughput_bytes_s",
            "pj_per_op", "pj_per_logical_byte", "total_pj_per_op", "total_pj_per_logical_byte",
            "memory_rail_power_w", "memory_rail_incremental_power_w", "graphics_clock_mhz", "memory_clock_mhz",
            "temperature_c", "idle_fraction_of_measured_power", "idle_fraction_of_power_limit", "power_limit_utilization",
            "tensor_peak_tflops_at_achieved_clock", "tensor_utilization_vs_dense_clock_peak")
_REPEAT_KEYS = {"repeat", "repeat_id", "repeat_index", "trial_id", "seed", "output_dir", "output_path"}


def _group_key(trial: Mapping[str, Any]) -> str:
    config = {k: v for k, v in trial.get("config", {}).items() if k not in _REPEAT_KEYS}
    return json.dumps({"gpu_uuid": trial.get("gpu_uuid"), "workload": trial.get("workload"), "config": config}, sort_keys=True, separators=(",", ":"))


def _median_ci(values: list[float], seed: int) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(1000))
    return [draws[24], draws[974]]


def _clock_stratum(group: Mapping[str, Any], cross_clock: bool) -> dict[str, Any]:
    key = {"gpu_uuid": group["gpu_uuid"], "workload": group["workload"]}
    if not cross_clock:
        key.update(graphics_clock_mhz=group["config"].get("graphics_clock_mhz") or group.get("graphics_clock_mhz"),
                   memory_clock_mhz=group["config"].get("memory_clock_mhz") or group.get("memory_clock_mhz"),
                   clock_comparison_controlled=group["clock_comparison_controlled"])
    return key


def _selections(groups: list[dict[str, Any]], fraction: float, min_repeats: int, cross_clock: bool,
                verified_only: bool = False, total_energy: bool = False) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["workload"] not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS or group["valid_repeats"] < min_repeats or (verified_only and not group["target_verified"]):
            continue
        tensor = group["workload"] in _TENSOR_WORKLOADS
        metric = ("total_" if total_energy else "") + ("pj_per_op" if tensor else "pj_per_logical_byte")
        throughput = "throughput_ops_s" if tensor else "throughput_bytes_s"
        if group.get(metric) is None or group.get(throughput) is None:
            continue
        key = _clock_stratum(group, cross_clock)
        buckets.setdefault(json.dumps(key, sort_keys=True), []).append(group)
    selections = []
    for key, candidates in sorted(buckets.items()):
        tensor = candidates[0]["workload"] in _TENSOR_WORKLOADS
        metric = ("total_" if total_energy else "") + ("pj_per_op" if tensor else "pj_per_logical_byte")
        throughput = "throughput_ops_s" if tensor else "throughput_bytes_s"
        observed_max = max(g[throughput] for g in candidates)
        eligible = [g for g in candidates if g[throughput] >= fraction * observed_max]
        best = min(eligible, key=lambda g: (g[metric], -g[throughput]))
        selections.append({"stratum": json.loads(key), "best_group_id": best["group_id"], "config": best["config"],
                           "metric": metric, "metric_value": best[metric], "metric_ci95": best["ci95"].get(metric),
                           "throughput": best[throughput], "observed_max_throughput": observed_max,
                           "throughput_fraction_of_observed_max": best[throughput] / observed_max if observed_max else None,
                           "throughput_threshold_fraction": fraction, "eligible_groups": len(eligible),
                           "target_verified": best["target_verified"], "valid_repeats": best["valid_repeats"],
                           "clock_comparison_controlled": best["clock_comparison_controlled"],
                           "selection_scope": "best measured among tested configurations; sweep completeness and physical target attribution are not implied"})
    return selections


def _pareto_frontiers(groups: list[dict[str, Any]], min_repeats: int, cross_clock: bool) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["workload"] not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS or group["valid_repeats"] < min_repeats:
            continue
        throughput = "throughput_ops_s" if group["workload"] in _TENSOR_WORKLOADS else "throughput_bytes_s"
        if group.get("board_power_w") is None or group.get(throughput) is None:
            continue
        buckets.setdefault(json.dumps(_clock_stratum(group, cross_clock), sort_keys=True), []).append(group)
    result = []
    for key, candidates in sorted(buckets.items()):
        throughput = "throughput_ops_s" if candidates[0]["workload"] in _TENSOR_WORKLOADS else "throughput_bytes_s"
        frontier = [group for group in candidates if not any(
            other[throughput] >= group[throughput] and other["board_power_w"] <= group["board_power_w"]
            and (other[throughput] > group[throughput] or other["board_power_w"] < group["board_power_w"])
            for other in candidates)]
        result.append({"stratum": json.loads(key), "scope": "cross_clock" if cross_clock else "within_clock",
                       "objectives": {"maximize": throughput, "minimize": "board_power_w"},
                       "candidate_groups": len(candidates), "frontier": [
                           {"group_id": group["group_id"], "config": group["config"], "throughput": group[throughput],
                            "board_power_w": group["board_power_w"], "incremental_power_w": group["incremental_power_w"],
                            "target_verified": group["target_verified"], "valid_repeats": group["valid_repeats"],
                            "clock_comparison_controlled": group["clock_comparison_controlled"]}
                           for group in sorted(frontier, key=lambda g: (g[throughput], g["board_power_w"]))],
                       "note": "Pareto dominance uses repeat medians; overlapping uncertainty intervals may change ordering."})
    return result


def _associate_controls(groups: list[dict[str, Any]], min_repeats: int) -> list[dict[str, Any]]:
    matched = []
    match_fields = ("graphics_clock_mhz", "memory_clock_mhz", "blocks", "threads", "sm_ids")
    controls = [group for group in groups if group["workload"] == "control" and group["valid_repeats"] >= min_repeats]
    for group in groups:
        if group["workload"] not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS or group["valid_repeats"] < min_repeats:
            continue
        matches = [control for control in controls if control["gpu_uuid"] == group["gpu_uuid"]
                   and all(control["config"].get(field) == group["config"].get(field) for field in match_fields)]
        if matches:
            matched.append({"workload_group_id": group["group_id"], "control_groups": [
                {"group_id": control["group_id"], "board_power_w": control["board_power_w"],
                 "incremental_activation_power_w": control["incremental_power_w"]} for control in matches],
                "matching_fields": ["gpu_uuid", *match_fields],
                "note": "Descriptive activation controls only. Different instruction mixes prevent automatic component-energy subtraction."})
    return matched


def summarize(records: Iterable[Mapping[str, Any]], throughput_fraction: float = 0.95,
              policy: AnalysisPolicy | Mapping[str, Any] | None = None, min_repeats: int = 3) -> dict[str, Any]:
    if not 0 < throughput_fraction <= 1 or min_repeats < 1:
        raise ValueError("throughput_fraction must be in (0, 1], min_repeats must be positive")
    trials = [analyze_trial(record, policy) for record in records]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        buckets.setdefault(_group_key(trial), []).append(trial)
    groups = []
    for key, repeats in sorted(buckets.items()):
        valid = [trial for trial in repeats if trial["valid"]]
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        group = {"group_id": digest, "gpu_uuid": repeats[0]["gpu_uuid"], "workload": repeats[0]["workload"],
                 "config": {k: v for k, v in repeats[0]["config"].items() if k not in _REPEAT_KEYS},
                 "repeats": len(repeats), "valid_repeats": len(valid), "trial_ids": [t["trial_id"] for t in repeats],
                 "target_verified": bool(valid) and all(t["target_verified"] for t in valid),
                 "clock_comparison_controlled": bool(valid) and all(t["clock_comparison_controlled"] for t in valid),
                 "diagnostic_only": repeats[0]["diagnostic_only"],
                 "confidence": "repeat_median" if len(valid) >= min_repeats else "insufficient_valid_repeats", "ci95": {}}
        for metric in _METRICS:
            values = [value for t in valid if (value := _finite(t.get(metric))) is not None]
            group[metric] = statistics.median(values) if values else None
            group["ci95"][metric] = _median_ci(values, int(digest, 16))
        groups.append(group)
    return {"schema_version": 1, "trials": trials, "groups": groups,
            "within_clock_best": _selections(groups, throughput_fraction, min_repeats, False),
            "cross_clock_best": _selections(groups, throughput_fraction, min_repeats, True),
            "within_clock_best_total_energy": _selections(groups, throughput_fraction, min_repeats, False, total_energy=True),
            "cross_clock_best_total_energy": _selections(groups, throughput_fraction, min_repeats, True, total_energy=True),
            "pareto_frontiers": _pareto_frontiers(groups, min_repeats, False) + _pareto_frontiers(groups, min_repeats, True),
            "active_control_associations": _associate_controls(groups, min_repeats),
            "verified_target_within_clock_best": _selections(groups, throughput_fraction, min_repeats, False, True),
            "verified_target_cross_clock_best": _selections(groups, throughput_fraction, min_repeats, True, True),
            "selection_policy": {"throughput_fraction": throughput_fraction, "min_repeats": min_repeats,
                                 "ci_method": "deterministic percentile bootstrap of repeat medians (1000 draws); n=3 intervals are coarse"},
            "notes": ["No fabricated measurement or architecture-specific idle wattage is supplied.",
                      "A device power limit is not measured consumption, and headline FLOP/s is not achieved throughput.",
                      "Idle subtraction cannot separate physical static leakage from refresh/clock/background power.",
                      "L1/L2/HBM logical bytes are not guaranteed physical traffic; profiler evidence is required for target verification."]}


def write_summary(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = destination / "trials.csv"
    fields = ["trial_id", "gpu_uuid", "gpu_name", "workload", "valid", "target_verified", "issues", "warnings", "duration_s", *_METRICS,
              "total_energy_j", "incremental_energy_j", "energy_source", "power_limit_w"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in summary.get("trials", []):
            writer.writerow({field: json.dumps(trial[field]) if isinstance(trial.get(field), (list, dict)) else trial.get(field) for field in fields})
    return {"summary_json": str(json_path), "trials_csv": str(csv_path)}
