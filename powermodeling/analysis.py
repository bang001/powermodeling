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
from dataclasses import asdict, dataclass, replace
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
    near_optimum_fraction: float = 0.05

    def __post_init__(self):
        for field, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Analysis policy {field} must be finite and nonnegative")
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError("min_samples must be an integer >=2")
        if self.min_measure_s <= 0 or self.min_idle_s <= 0 or self.max_sample_gap_s <= 0:
            raise ValueError("Measurement, idle and sample-gap durations must be positive")


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
    if any((value := _finite(s.get("power_w"))) is not None and value < 0 for s in covering):
        result["issues"].append(f"negative_power:{name}")
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
    anchor = next((s.get("energy_mj") for s in covering if type(s.get("energy_mj")) is int), 0)
    relative_counter_samples = []
    for sample in covering:
        counter = sample.get("energy_mj")
        relative_counter_samples.append({"t_s": sample["t_s"], "relative_energy_mj": counter - anchor if isinstance(counter, (int, float)) and not isinstance(counter, bool) and counter >= 0 else None})
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
    for field in ("graphics_clock_mhz", "sm_clock_mhz", "memory_clock_mhz", "temperature_c"):
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


def _measure_epochs(benchmark: Mapping[str, Any], phase: Mapping[str, Any],
                    policy: AnalysisPolicy) -> tuple[list[dict[str, Any]], list[str]]:
    """Choose complete reported work intervals inside the trimmed power window.

    Counts are never fractionally apportioned to a boundary-crossing batch. The
    power integration is subsequently moved to exactly these epoch boundaries.
    """
    source = benchmark.get("measure_epochs", benchmark.get("work_epochs"))
    if source is None:
        return [], []
    if not isinstance(source, list) or not source:
        return [], ["invalid_measure_epochs"]
    start, end = _finite(phase.get("start_s")), _finite(phase.get("end_s"))
    if start is None or end is None:
        return [], ["invalid_measure_epochs_phase_window"]
    parsed = []
    for epoch in source:
        if not isinstance(epoch, dict):
            return [], ["invalid_measure_epoch"]
        a, b = _finite(epoch.get("start_s")), _finite(epoch.get("end_s"))
        if a is None:
            stamp = _finite(epoch.get("host_monotonic_start_ns", epoch.get("start_ns")))
            a = stamp / 1e9 if stamp is not None else None
        if b is None:
            stamp = _finite(epoch.get("host_monotonic_end_ns", epoch.get("end_ns")))
            b = stamp / 1e9 if stamp is not None else None
        if a is None or b is None or b <= a or a < start - 1e-6 or b > end + 1e-6:
            return [], ["invalid_measure_epoch_boundaries"]
        for field in ("operations", "logical_bytes"):
            count = _finite(epoch.get(field))
            if count is None or count < 0:
                return [], [f"invalid_measure_epoch_count:{field}"]
        parsed.append({**epoch, "start_s": a, "end_s": b,
                       "operations": _finite(epoch["operations"]), "logical_bytes": _finite(epoch["logical_bytes"])})
    parsed.sort(key=lambda e: e["start_s"])
    if any(b["start_s"] < a["end_s"] - 1e-6 for a, b in zip(parsed, parsed[1:])):
        return [], ["overlapping_measure_epochs"]
    inner = [e for e in parsed if e["start_s"] >= start + policy.trim_s - 1e-6
             and e["end_s"] <= end - policy.trim_s + 1e-6]
    if not inner or inner[-1]["end_s"] - inner[0]["start_s"] < policy.min_measure_s - 1e-6:
        return [], ["insufficient_complete_measure_epochs"]
    # Whole-run counts are a useful consistency check against lost/duplicated
    # epoch records; they are not used as the trimmed-window denominator.
    for field in ("operations", "logical_bytes"):
        whole = _finite(benchmark.get(field))
        counted = sum(e[field] for e in parsed)
        if whole is not None and abs(counted - whole) > max(abs(whole) * 1e-6, 1.0):
            return [], [f"measure_epoch_total_disagreement:{field}"]
    return inner, []



def _physical_traffic_estimates(result, assessment, byte_rate):
    """Keep replay counters separate from an unprofiled energy denominator.

    Replay can establish path/coalescing diagnostics. It does not count bytes
    in the energy window, so physical pJ/bit is deliberately withheld.
    """
    return {"physical_pj_per_bit": None,
            "status": "unavailable_without_matched_energy_window_hardware_traffic_counts",
            "profile_rates": assessment.get("rates_summary"),
            "note": "Nsight replay busy-time rates and logical/physical ratios are diagnostic; they are not the energy-window physical-bit denominator."}


def _paired_reference_contrast(record, phase_records, result, pre, post, measure, policy):
    """Evaluate an operational paired reference without claiming pure block power."""
    protocol = record.get("treatment_protocol") or (record.get("benchmark") or {}).get("treatment_protocol") or {}
    reference_record = record.get("active_reference") or {}
    reference_phase = phase_records.get("active_reference")
    result.update(paired_active_reference_eligible=False, paired_active_reference_power_w=None,
                  paired_active_reference_energy_j=None, paired_active_reference_pj_per_flop=None,
                  paired_active_reference_pj_per_logical_bit=None, paired_active_reference_issues=[],
                  paired_active_reference_protocol=protocol,
                  paired_active_reference_interpretation="same-process active issue-loop reference contrast; instruction, memory and scheduling differences remain; physical component switching power is not isolated")
    if not reference_phase:
        result["paired_active_reference_issues"] = ["missing_paired_active_reference_phase"]
        return
    reference = _phase_stats(reference_phase, "active_reference", policy)
    reference_epochs, epoch_issues = _measure_epochs(reference_record, reference_phase, policy)
    if reference_epochs:
        aligned = {**reference_phase, "start_s": reference_epochs[0]["start_s"], "end_s": reference_epochs[-1]["end_s"]}
        reference = _phase_stats(aligned, "active_reference", replace(policy, trim_s=0))
    issues = list(reference["issues"]) + epoch_issues
    if not reference_epochs or not all(epoch.get("counts_exact") is True for epoch in reference_epochs):
        issues.append("reference_work_energy_window_unverified")
    # Protocol identity comes from the measured executable, not a caller's
    # generic control-group match. Both arms share context/allocations/clocks.
    for field in ("same_process", "same_allocations", "same_clock_policy", "launch_geometry_matched"):
        if protocol.get(field) is not True:
            issues.append("paired_protocol_unverified:" + field)
    if protocol.get("kind") != "paired_active_reference" or protocol.get("reference_kind") != "issue_loop" or protocol.get("reference_workload") != "control":
        issues.append("paired_reference_definition_unverified")
    order = protocol.get("order")
    if order not in ("AB", "BA"):
        issues.append("paired_protocol_order_unverified")
    if order == "AB" and (_finite(reference_phase.get("end_s")) or 0) > (_finite((phase_records.get("measure") or {}).get("start_s")) or 0):
        issues.append("paired_arm_order_or_overlap_invalid")
    if order == "BA" and (_finite((phase_records.get("measure") or {}).get("end_s")) or 0) > (_finite(reference_phase.get("start_s")) or 0):
        issues.append("paired_arm_order_or_overlap_invalid")
    treatment_benchmark = record.get("benchmark") or {}
    for field in ("blocks", "threads", "batch_launches", "iterations_per_launch"):
        if reference_record.get(field) is None or treatment_benchmark.get(field) is None or reference_record[field] != treatment_benchmark[field]:
            issues.append("paired_geometry_mismatch:" + field)
    state_deltas = {}
    reference_clock_fields = ["graphics_clock_mhz", "memory_clock_mhz"]
    if any(_finite(sample.get("sm_clock_mhz")) is not None for phase in (measure, reference) for sample in phase.get("_selected", [])):
        reference_clock_fields.append("sm_clock_mhz")
    result["paired_reference_clock_domains_compared"] = reference_clock_fields
    result["paired_reference_sm_clock_uses_graphics_proxy"] = "sm_clock_mhz" not in reference_clock_fields
    for field in (*reference_clock_fields, "temperature_c"):
        active, baseline = _finite(measure.get(field)), _finite(reference.get(field))
        state_deltas[field] = active - baseline if active is not None and baseline is not None else None
        tolerance = policy.max_temperature_drift_c if field == "temperature_c" else (active or 0) * policy.clock_tolerance_fraction
        samples = reference.get("_selected", [])
        if active is None or baseline is None or not samples or any((value := _finite(sample.get(field))) is None or abs(value - active) > tolerance for sample in samples):
            issues.append("paired_state_mismatch:" + field)
    for sample in reference.get("_selected", []):
        if (_throttle_mask(sample.get("throttle_reasons")) is None
                or (_throttle_mask(sample.get("throttle_reasons")) or 0) & ~0x3):
            issues.append("reference_throttle_unverified_or_active")
        for field in ("compute_processes", "graphics_processes"):
            inventory = sample.get(field)
            if not isinstance(inventory, list):
                issues.append("reference_process_inventory_unverified:" + field)
            elif any((item.get("pid") if isinstance(item, dict) else item) != (record.get("quality") or {}).get("measurement_pid") for item in inventory):
                issues.append("reference_foreign_gpu_process")
    # A fixed board cap is a state variable even if consumption is below it.
    treatment_caps = [_finite(sample.get("power_limit_w")) for sample in measure.get("_selected", [])]
    reference_caps = [_finite(sample.get("power_limit_w")) for sample in reference.get("_selected", [])]
    all_caps = treatment_caps + reference_caps
    if not all_caps or any(cap is None or cap <= 0 for cap in all_caps):
        issues.append("paired_power_cap_unverified")
    elif max(all_caps) - min(all_caps) > 1e-6:
        issues.append("paired_power_cap_changed")
    for field in ("enforced_power_limit_w", "pstate"):
        active_values = {_finite(sample.get(field)) for sample in measure.get("_selected", [])}
        reference_values = {_finite(sample.get(field)) for sample in reference.get("_selected", [])}
        if active_values != {None} or reference_values != {None}:
            if None in active_values or None in reference_values or len(active_values) != 1 or active_values != reference_values:
                issues.append("paired_state_mismatch:" + field)
    if (reference_record.get("sanity") or {}).get("requested_sm_coverage_complete") is not True:
        issues.append("reference_sm_coverage_unverified")
    reference_power = _finite(reference.get("power_w"))
    treatment_power = _finite(result.get("board_power_w"))
    contrast = treatment_power - reference_power if treatment_power is not None and reference_power is not None else None
    duration = result.get("duration_s")
    if contrast is None:
        issues.append("paired_power_unavailable")
    elif contrast <= 0:
        issues.append("nonpositive_paired_contrast; signed observation retained but cannot minimize an energy objective")
    result.update(paired_active_reference_power_w=contrast,
                  paired_active_reference_energy_j=contrast * duration if contrast is not None and duration else None,
                  paired_active_reference_eligible=result["valid"] and result["count_energy_time_alignment_exact"] and not issues,
                  paired_active_reference_issues=sorted(set(issues)),
                  paired_active_reference_state_deltas=state_deltas,
                  active_reference_power_w=reference_power,
                  active_reference_total_energy_j=reference.get("energy_counter_j") if reference.get("energy_counter_j") is not None else reference.get("integrated_power_energy_j"))
    tensor = record.get("workload") in _TENSOR_WORKLOADS
    memory = record.get("workload") in _MEMORY_WORKLOADS
    ops_rate, byte_rate = result.get("throughput_ops_s"), result.get("throughput_bytes_s")
    result["paired_active_reference_pj_per_flop"] = contrast / ops_rate * 1e12 if tensor and contrast is not None and ops_rate and ops_rate > 0 else None
    result["paired_active_reference_pj_per_logical_bit"] = contrast / byte_rate / 8 * 1e12 if memory and contrast is not None and byte_rate and byte_rate > 0 else None
    treatment_idle, reference_idle = _idle_at(pre, post, measure, "power_w"), _idle_at(pre, post, reference, "power_w")
    result["paired_active_reference_drift_adjusted_power_w"] = (treatment_power - treatment_idle) - (reference_power - reference_idle) if result.get("operational_idle_increment_eligible") and None not in (treatment_power, treatment_idle, reference_power, reference_idle) else None
    result["phases"]["active_reference"] = {key: value for key, value in reference.items() if not key.startswith("_")}


def _treatment_design_stratum(record, benchmark):
    protocol = record.get("treatment_protocol") or benchmark.get("treatment_protocol") or {}
    return {"kind": protocol.get("kind", "unspecified_legacy_protocol"),
            "reference_kind": protocol.get("reference_kind"),
            "paired_reference_context_allocated": benchmark.get("paired_reference_context_allocated", protocol.get("paired_reference_context_allocated"))}

def analyze_trial(record: Mapping[str, Any], policy: AnalysisPolicy | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Calculate operational incremental energy for one sustained trial.

Complete monotonic work epochs align the work denominator and energy integral.
Older records without epochs retain a clearly qualified whole-run rate estimate;
stable clock/thermal telemetry alone cannot prove stationary work throughput.
"""
    policy = policy if isinstance(policy, AnalysisPolicy) else AnalysisPolicy(**(policy or {}))
    benchmark = record.get("benchmark") or record.get("result") or {}
    phase_records = record.get("phases", {})
    if isinstance(phase_records, list):
        phase_records = {p["name"]: p for p in phase_records if isinstance(p, dict) and "name" in p}
    phases = {name: _phase_stats(phase_records.get(name), name, policy)
              for name in ("idle_pre", "measure", "idle_post")}
    epochs, epoch_issues = _measure_epochs(benchmark, phase_records.get("measure") or {}, policy)
    if epochs:
        aligned_phase = {**phase_records["measure"], "start_s": epochs[0]["start_s"], "end_s": epochs[-1]["end_s"]}
        phases["measure"] = _phase_stats(aligned_phase, "measure", replace(policy, trim_s=0))
    if phase_records.get("active_control"):
        phases["active_control"] = _phase_stats(phase_records["active_control"], "active_control", policy)
    measure, pre, post = phases["measure"], phases["idle_pre"], phases["idle_post"]
    # Total treatment energy remains interpretable when an idle/reference arm
    # fails. Baseline failures gate only baseline-dependent contrasts.
    issues = list(measure["issues"])
    issues.extend(epoch_issues)
    baseline_issues = [issue for name in ("idle_pre", "idle_post") for issue in phases[name]["issues"]]
    warnings = [warning for phase in phases.values() for warning in phase["warnings"]]
    config = dict(record.get("config") or {})
    device = record.get("device") or {}
    gpu_uuid = config.get("gpu_uuid") or device.get("uuid") or record.get("gpu_uuid")
    if not record.get("trial_id"):
        issues.append("missing_trial_id")
    treatment_clock_fields = ["graphics_clock_mhz", "memory_clock_mhz"]
    if any(_finite(sample.get("sm_clock_mhz")) is not None for sample in measure.get("_selected", [])):
        treatment_clock_fields.append("sm_clock_mhz")
    for field in treatment_clock_fields:
        requested, actual = _finite(config.get(field)), _finite(measure.get(field))
        if requested is not None and requested > 0:
            if actual is None:
                issues.append(f"missing_clock_telemetry:{field}")
            elif abs(actual - requested) / requested > policy.clock_tolerance_fraction:
                issues.append(f"clock_mismatch:{field}")
        elif field != "sm_clock_mhz":
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
    for field in treatment_clock_fields:
        if selected and any((_finite(sample.get(field)) or 0) <= 0 for sample in selected):
            issues.append(f"missing_clock_telemetry:{field}")
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
    for key in ("interference_detected", "mig_active", "mps_active", "unsafe_short_duration"):
        if quality.get(key) or config.get(key):
            issues.append(key)
    measurement_pid = quality.get("measurement_pid")
    other_pids = set()
    for phase_name, phase in phases.items():
        destination = issues if phase_name == "measure" else baseline_issues
        for sample in phase.get("_selected", []):
            for field in ("compute_processes", "graphics_processes"):
                inventory = sample.get(field)
                if not isinstance(inventory, list):
                    destination.append(f"missing_process_inventory:{field}")
                    continue
                if inventory and measurement_pid is None:
                    destination.append("missing_measurement_pid")
                for process in inventory:
                    pid = process.get("pid") if isinstance(process, dict) else process
                    if pid is not None and pid != measurement_pid:
                        if phase_name == "measure":
                            other_pids.add(pid)
                        else:
                            destination.append("baseline_other_gpu_processes")
            if sample.get("mps_compute_processes"):
                destination.append("mps_active")
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
            baseline_issues.append("idle_baseline_drift")
    duration = _finite(measure.get("duration_s"))
    total = _finite(measure.get("energy_counter_j"))
    energy_source = "energy_counter_delta"
    if total is None:
        total, energy_source = _finite(measure.get("integrated_power_energy_j")), "power_trapezoid"
    board_power = total / duration if total is not None and duration and duration > 0 else None
    idle_energy = idle_power * duration if idle_power is not None and duration else None
    incremental = total - idle_energy if total is not None and idle_energy is not None else None
    if idle_power is None:
        baseline_issues.append("missing_idle_baseline")
    if incremental is not None and incremental < 0:
        baseline_issues.append("negative_incremental_energy")
    device_duration = _finite(benchmark.get("duration_s"))
    measured_duration = _finite(benchmark.get("host_duration_s")) or device_duration
    if device_duration and measured_duration and abs(device_duration - measured_duration) / measured_duration > 0.05:
        issues.append("host_device_duration_disagreement")
    operations, logical_bytes = _finite(benchmark.get("operations")), _finite(benchmark.get("logical_bytes"))
    ops_rate = operations / measured_duration if operations is not None and measured_duration and measured_duration > 0 else None
    byte_rate = logical_bytes / measured_duration if logical_bytes is not None and measured_duration and measured_duration > 0 else None
    count_alignment_exact = bool(epochs) and all(e.get("counts_exact") is True for e in epochs)
    counted_operations = sum(e["operations"] for e in epochs) if epochs else None
    counted_logical_bytes = sum(e["logical_bytes"] for e in epochs) if epochs else None
    if epochs and duration and duration > 0:
        ops_rate = counted_operations / duration
        byte_rate = counted_logical_bytes / duration
    if not count_alignment_exact:
        warnings.append("throughput_time_alignment_unverified; energy per work is a rate estimate assuming stationary throughput")
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
    provided_validation = record.get("validation") or {}
    evidence = provided_validation.get("profiler_evidence") if isinstance(provided_validation, dict) else None
    if isinstance(evidence, dict):
        from .validation import validate_evidence
        assessment = validate_evidence(record, evidence)
    else:
        assessment = {"status": "inconclusive", "suitable_verified": False,
                      "reasons": ["missing_numeric_profiler_evidence"], "checks": [], "kernels": []}
    validation = {"assessment": assessment, "status": assessment["status"],
                  "profiler_evidence": evidence if isinstance(evidence, dict) else None}
    if isinstance(provided_validation, dict) and provided_validation.get("locality"):
        validation["requested_locality"] = provided_validation["locality"]
    power_limit = (_median(s.get("power_limit_w") for s in selected)
                   or _finite(record.get("power_limit_w", device.get("power_limit_w", config.get("power_limit_w"))))
                   or _finite((device.get("sample") or {}).get("power_limit_w")))
    tensor_peak = None
    peak_clock_source = "sm_clock_mhz" if (_finite(measure.get("sm_clock_mhz")) or 0) > 0 else "graphics_clock_mhz"
    peak_clock = _finite(measure.get(peak_clock_source))
    if record.get("workload") in _TENSOR_WORKLOADS and peak_clock is not None and peak_clock > 0:
        try:
            from .profiles import theoretical_tensor_tflops
            tensor_peak = theoretical_tensor_tflops(record.get("cuda_device") or device, peak_clock)
            if peak_clock_source != "sm_clock_mhz":
                warnings.append("tensor_peak_uses_graphics_clock_proxy; measured SM clock unavailable")
        except (ValueError, KeyError, TypeError):
            warnings.append("tensor_theoretical_peak_unavailable")
    baseline_deltas = {}
    baseline_power_cap_matched = True
    baseline_clock_matched = True
    baseline_temperature_matched = True
    baseline_clock_fields = ["graphics_clock_mhz", "memory_clock_mhz"]
    if any(_finite(sample.get("sm_clock_mhz")) is not None for phase in (measure, pre, post) for sample in phase.get("_selected", [])):
        baseline_clock_fields.append("sm_clock_mhz")
    for name, baseline_phase in (("idle_pre", pre), ("idle_post", post)):
        deltas = {}
        for field in baseline_clock_fields:
            active_clock, idle_clock = _finite(measure.get(field)), _finite(baseline_phase.get(field))
            deltas[field] = active_clock - idle_clock if active_clock is not None and idle_clock is not None else None
            if active_clock is None or idle_clock is None or active_clock <= 0 or abs(active_clock - idle_clock) / active_clock > policy.clock_tolerance_fraction:
                baseline_clock_matched = False
            if active_clock is not None and active_clock > 0 and any((clock := _finite(sample.get(field))) is None or abs(clock - active_clock) / active_clock > policy.clock_tolerance_fraction for sample in baseline_phase.get("_selected", [])):
                baseline_clock_matched = False
        active_temp, idle_temp = _finite(measure.get("temperature_c")), _finite(baseline_phase.get("temperature_c"))
        deltas["temperature_c"] = active_temp - idle_temp if active_temp is not None and idle_temp is not None else None
        if active_temp is None or idle_temp is None or abs(active_temp - idle_temp) > policy.max_temperature_drift_c:
            baseline_temperature_matched = False
        if active_temp is not None and any((temp := _finite(sample.get("temperature_c"))) is None or abs(temp - active_temp) > policy.max_temperature_drift_c for sample in baseline_phase.get("_selected", [])):
            baseline_temperature_matched = False
        active_caps = {_finite(sample.get("power_limit_w")) for sample in measure.get("_selected", [])}
        baseline_caps = {_finite(sample.get("power_limit_w")) for sample in baseline_phase.get("_selected", [])}
        if None in active_caps or None in baseline_caps or len(active_caps) != 1 or active_caps != baseline_caps:
            baseline_power_cap_matched = False
        baseline_deltas[name] = deltas
    if not baseline_power_cap_matched:
        warnings.append("idle_active_power_cap_mismatch_or_unavailable; idle contrast is withheld from optimization")
    if not baseline_clock_matched:
        warnings.append("idle_active_clock_mismatch_or_unavailable; incremental power includes activation and frequency-state differences")
    if not baseline_temperature_matched:
        warnings.append("idle_active_temperature_mismatch_or_unavailable; leakage/thermal drift can confound subtraction")
    target_verified = assessment["status"] == "pass" and assessment.get("suitable_verified") is True
    result: dict[str, Any] = {
        "trial_id": record.get("trial_id"), "workload": record.get("workload"), "gpu_uuid": gpu_uuid,
        "repeat_index": record.get("repeat", config.get("repeat", config.get("repeat_index"))),
        "gpu_name": device.get("name", record.get("gpu_name")), "config": config,
        "valid": not issues, "issues": sorted(set(issues)), "warnings": sorted(set(warnings)),
        "baseline_valid": not baseline_issues, "baseline_issues": sorted(set(baseline_issues)),
        "treatment_outcome_kind": "measured whole-device total energy per counted treatment work",
        "target_verified": target_verified,
        "benchmark_sha256": (record.get("provenance") or {}).get("benchmark_sha256"),
        "treatment_design_stratum": _treatment_design_stratum(record, benchmark),
        "measurement_stratum": {"ecc_mode": device.get("ecc_mode"), "mig_mode": device.get("mig_mode"),
                                "driver_version": device.get("driver_version"), "nvml_version": device.get("nvml_version"),
                                "runtime_versions": {key: (record.get("cuda_device") or {}).get(key) for key in ("cuda_runtime_version", "cuda_driver_version", "cuda_compile_version", "cublas_version")},
                                "power_limit_w": power_limit, "power_scope": (record.get("telemetry") or {}).get("power_scope", device.get("power_scope_note")),
                                "power_usage_semantics": device.get("power_usage_semantics")},
        "validation": validation, "duration_s": duration, "benchmark_duration_s": measured_duration,
        "profile_rates_summary": assessment.get("rates_summary"),
        "validation_binding": {"condition_id": record.get("condition_id"), "workload": record.get("workload"),
                               "config": config, "provenance": record.get("provenance"),
                               "benchmark": {key: value for key, value in benchmark.items() if key not in ("measure_epochs", "work_epochs")},
                               "samples": [{"phase": "measure", **{field: sample.get(field) for field in ("graphics_clock_mhz", "memory_clock_mhz")}} for sample in selected]},
        "benchmark_device_duration_s": device_duration,
        "throughput_timebase": "complete monotonic work epochs with energy integrated over matching boundaries" if epochs else "estimated stationary rate from whole benchmark host duration" if benchmark.get("host_duration_s") else "estimated stationary rate from whole benchmark duration; host timebase unavailable",
        "count_energy_time_alignment_exact": count_alignment_exact,
        "counted_measure_operations": counted_operations, "counted_measure_logical_bytes": counted_logical_bytes,
        "complete_measure_epochs": len(epochs),
        "energy_per_work_kind": "counted matching-window energy per work" if count_alignment_exact else "stationary whole-run-rate estimate",
        "profiler_suitability_status": assessment["status"],
        "ncu_status": assessment["status"] if isinstance(evidence, dict) else "unprofiled",
        "ncu_assessment": assessment, "ncu_target_suitability": target_verified,
        "ncu_utilization_status": "diagnostic_only" if assessment.get("rates_summary", {}).get("kernel_duration_s") else "inconclusive",
        "verified_selection_eligible": not issues and target_verified and count_alignment_exact,
        "operation_unit": benchmark.get("operation_unit"),
        "diagnostic_only": record.get("workload") not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS,
        "clock_comparison_controlled": all((_finite(config.get(field)) or 0) > 0 for field in ("graphics_clock_mhz", "memory_clock_mhz")),
        "throughput_ops_s": ops_rate, "throughput_bytes_s": byte_rate,
        "device_event_throughput_ops_s": operations / device_duration if operations is not None and device_duration and device_duration > 0 else None,
        "device_event_throughput_bytes_s": logical_bytes / device_duration if logical_bytes is not None and device_duration and device_duration > 0 else None,
        "tensor_peak_tflops_at_achieved_clock": tensor_peak,
        "tensor_peak_clock_source": peak_clock_source if tensor_peak is not None else None,
        "tensor_utilization_vs_dense_clock_peak": ops_rate / 1e12 / tensor_peak if ops_rate is not None and tensor_peak and tensor_peak > 0 else None,
        "board_power_w": board_power, "idle_power_w": idle_power, "incremental_power_w": incremental_power,
        "operational_idle_increment_power_w": incremental_power,
        "operational_idle_increment_energy_j": incremental,
        "operational_idle_increment_eligible": not issues and not baseline_issues and count_alignment_exact and baseline_clock_matched and baseline_temperature_matched and baseline_power_cap_matched,
        "idle_baseline_estimator": "linear interpolation between pre/post trimmed idle mean power at treatment midpoint; integral exact for the assumed linear drift",
        "idle_increment_interpretation": "operational whole-device contrast against powered idle; activation, clock distribution, refresh and background changes may remain; physical leakage/switching are not isolated",
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
        "graphics_clock_mhz": measure.get("graphics_clock_mhz"), "sm_clock_mhz": measure.get("sm_clock_mhz"), "memory_clock_mhz": measure.get("memory_clock_mhz"),
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
        "baseline_power_scope": "same measured device/API rail; idle is an operational state, not a leakage meter",
        "baseline_temperature_matched": baseline_temperature_matched, "baseline_power_cap_matched": baseline_power_cap_matched,
        "baseline_clock_domains_compared": baseline_clock_fields,
        "baseline_sm_clock_uses_graphics_proxy": "sm_clock_mhz" not in baseline_clock_fields,
        "active_minus_idle_state_deltas": baseline_deltas,
        "dynamic_attribution_eligible": not issues and not baseline_issues and target_verified and count_alignment_exact and baseline_clock_matched and baseline_temperature_matched and baseline_power_cap_matched,
        "dynamic_attribution_note": "Eligibility indicates matched-baseline/profiler prerequisites only. It does not prove isolation of physical switching power or static leakage.",
        "attribution": "whole-device incremental energy per benchmark operation or logical byte; not isolated block energy",
        "policy": asdict(policy), "phases": {}, "model_features": record.get("model_features"),
        "feature_units": record.get("feature_units"), "feature_provenance": record.get("feature_provenance"),
        "power_provenance": record.get("power_provenance"),
    }
    # FLOP is a floating-point operation: one fused multiply-add counts as two.
    # Logical bits count payload requested by the benchmark; cache-sector and
    # DRAM bits are different denominators and are never silently substituted.
    tensor = record.get("workload") in _TENSOR_WORKLOADS
    memory = record.get("workload") in _MEMORY_WORKLOADS
    for prefix, power in (("total_", board_power), ("operational_idle_increment_", incremental_power)):
        result[prefix + "pj_per_flop"] = power / ops_rate * 1e12 if tensor and power is not None and ops_rate and ops_rate > 0 else None
        result[prefix + "pj_per_logical_bit"] = power / byte_rate / 8 * 1e12 if memory and power is not None and byte_rate and byte_rate > 0 else None
    result["flop_count_convention"] = "multiply-add=2 FLOPs; reported dense FP16 tensor operations" if tensor else None
    result["logical_bit_count_convention"] = "8 bits per logical byte; copy counts both read and write payload" if memory else None
    result["physical_traffic_energy"] = _physical_traffic_estimates(result, assessment, byte_rate)
    if record.get("workload") == "l2_latency":
        result["latency_diagnostics"] = dict(benchmark)
    for name, phase in phases.items():
        result["phases"][name] = {k: v for k, v in phase.items() if not k.startswith("_")}
    if "active_control" in phases:
        control = phases["active_control"]
        control_idle = _idle_at(pre, post, control, "power_w")
        control_power = _finite(control.get("power_w"))
        result["active_control_incremental_power_w"] = control_power - control_idle if control_power is not None and control_idle is not None else None
        result["control_subtracted_power_w"] = None
        result["control_subtraction_note"] = "Standalone activation control is descriptive; paired protocol and state checks are required for an active-reference contrast."
    _paired_reference_contrast(record, phase_records, result, pre, post, measure, policy)
    return result


_METRICS = ("board_power_w", "idle_power_w", "incremental_power_w", "throughput_ops_s", "throughput_bytes_s",
            "pj_per_op", "pj_per_logical_byte", "total_pj_per_op", "total_pj_per_logical_byte",
            "memory_rail_power_w", "memory_rail_incremental_power_w", "graphics_clock_mhz", "sm_clock_mhz", "memory_clock_mhz",
            "temperature_c", "idle_fraction_of_measured_power", "idle_fraction_of_power_limit", "power_limit_utilization",
            "tensor_peak_tflops_at_achieved_clock", "tensor_utilization_vs_dense_clock_peak",
            "total_pj_per_flop", "total_pj_per_logical_bit",
            "operational_idle_increment_power_w", "operational_idle_increment_energy_j",
            "operational_idle_increment_pj_per_flop", "operational_idle_increment_pj_per_logical_bit",
            "paired_active_reference_power_w", "paired_active_reference_energy_j",
            "paired_active_reference_pj_per_flop", "paired_active_reference_pj_per_logical_bit")
_REPEAT_KEYS = {"repeat", "repeat_id", "repeat_index", "trial_id", "output_dir", "output_path"}


def _group_key(trial: Mapping[str, Any]) -> str:
    config = {k: v for k, v in trial.get("config", {}).items() if k not in _REPEAT_KEYS}
    # A driver-controlled run has no common requested-frequency stratum. Do
    # not merge repetitions with materially different achieved frequencies.
    achieved = None if trial.get("clock_comparison_controlled") else {
        field: trial.get(field) for field in ("graphics_clock_mhz", "memory_clock_mhz")}
    return json.dumps({"gpu_uuid": trial.get("gpu_uuid"), "workload": trial.get("workload"),
                       "config": config, "uncontrolled_achieved_clocks": achieved,
                       "benchmark_sha256": trial.get("benchmark_sha256"),
                       "measurement_stratum": trial.get("measurement_stratum"),
                       "treatment_design_stratum": trial.get("treatment_design_stratum")}, sort_keys=True, separators=(",", ":"))


def _median_ci(values: list[float], seed: int) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(1000))
    return [draws[24], draws[974]]


def _clock_stratum(group: Mapping[str, Any], cross_clock: bool) -> dict[str, Any]:
    key = {"gpu_uuid": group["gpu_uuid"], "workload": group["workload"],
           "benchmark_sha256": group.get("benchmark_sha256"), "measurement_stratum": group.get("measurement_stratum"),
           "treatment_design_stratum": group.get("treatment_design_stratum"),
           "access": group["config"].get("access", "read") if group["workload"] in _MEMORY_WORKLOADS else None}
    if not cross_clock:
        key.update(graphics_clock_mhz=group["config"].get("graphics_clock_mhz") or group.get("graphics_clock_mhz"),
                   memory_clock_mhz=group["config"].get("memory_clock_mhz") or group.get("memory_clock_mhz"),
                   clock_comparison_controlled=group["clock_comparison_controlled"])
    return key


def _selections(groups: list[dict[str, Any]], fraction: float, min_repeats: int, cross_clock: bool,
                verified_only: bool = False, total_energy: bool = False, exploratory_only: bool = False) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["workload"] not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS or group["valid_repeats"] < min_repeats:
            continue
        if exploratory_only == group["clock_comparison_controlled"]:
            continue
        if verified_only and not group["verified_selection_eligible"]:
            continue
        if not total_energy and not group.get("operational_idle_increment_eligible"):
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
        candidate_max = max(g[throughput] for g in candidates)
        all_observed = [g[throughput] for g in groups if g["workload"] == candidates[0]["workload"]
                        and g["gpu_uuid"] == candidates[0]["gpu_uuid"] and g["valid_repeats"] >= min_repeats
                        and g["clock_comparison_controlled"] == candidates[0]["clock_comparison_controlled"]
                        and _clock_stratum(g, cross_clock) == json.loads(key) and g.get(throughput) is not None]
        all_max = max(all_observed) if all_observed else candidate_max
        # The throughput bar remains the complete observed sweep maximum even
        # when its fastest conditions are unprofiled or fail target admission.
        # An incomplete validated subset cannot lower the high-utilization bar.
        observed_max = all_max if verified_only else candidate_max
        eligible = [g for g in candidates if g[throughput] >= fraction * observed_max]
        if not eligible:
            continue
        best = min(eligible, key=lambda g: (g[metric], -g[throughput]))
        selections.append({"stratum": json.loads(key), "best_group_id": best["group_id"], "config": best["config"],
                           "metric": metric, "metric_value": best[metric], "metric_ci95": best["ci95"].get(metric),
                           "throughput": best[throughput], "observed_max_throughput": observed_max,
                           "throughput_fraction_of_observed_max": best[throughput] / observed_max if observed_max else None,
                           "throughput_threshold_fraction": fraction, "eligible_groups": len(eligible),
                           "target_verified": best["target_verified"], "valid_repeats": best["valid_repeats"],
                           "clock_comparison_controlled": best["clock_comparison_controlled"],
                           "observation_scope": "uncontrolled-clock exploratory" if exploratory_only else "requested fixed-clock comparison",
                           "observed_max_throughput_all_valid_groups": all_max,
                           "verified_peak_coverage_fraction": candidate_max / all_max if verified_only and all_max > 0 else None,
                           "meets_threshold_vs_all_valid_groups": best[throughput] >= fraction * all_max,
                           "selection_scope": "best measured among tested configurations; sweep completeness and physical target attribution are not implied"})
    return selections



def _empirical_energy_optima(groups, fraction, min_repeats, near_fraction, across_memory_domains=False):
    """Choose each GPU's energy optimum using utilization at its own frequency.

    Global-peak throughput constraints answer a different question and remain
    available separately. Every support point below is actually observed; no
    interpolation can establish an optimum at an untested clock.
    """
    populations = [group for group in groups if group["workload"] in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS
                   and group["valid_repeats"] >= min_repeats and group["clock_comparison_controlled"]]
    peaks = {}
    for group in populations:
        throughput = "throughput_ops_s" if group["workload"] in _TENSOR_WORKLOADS else "throughput_bytes_s"
        value = _finite(group.get(throughput))
        if value is None or value <= 0:
            continue
        clock_key = json.dumps(_clock_stratum(group, False), sort_keys=True)
        peaks[clock_key] = max(peaks.get(clock_key, 0), value)
    buckets = {}
    objectives = (("total", "total_", None),
                  ("operational_idle_increment", "operational_idle_increment_", "operational_idle_increment_eligible"),
                  ("paired_active_reference", "paired_active_reference_", "paired_active_reference_eligible"))
    for group in populations:
        if not group["verified_selection_eligible"]:
            continue
        throughput = "throughput_ops_s" if group["workload"] in _TENSOR_WORKLOADS else "throughput_bytes_s"
        clock_peak = peaks.get(json.dumps(_clock_stratum(group, False), sort_keys=True))
        if clock_peak is None or (_finite(group.get(throughput)) or 0) < fraction * clock_peak:
            continue
        tensor = group["workload"] in _TENSOR_WORKLOADS
        for objective, prefix, eligibility in objectives:
            if eligibility and not group.get(eligibility):
                continue
            metric = prefix + ("pj_per_flop" if tensor else "pj_per_logical_bit")
            metric_value = _finite(group.get(metric))
            if metric_value is None or metric_value <= 0:
                continue
            stratum = _clock_stratum(group, True)
            if not across_memory_domains:
                stratum["memory_clock_mhz"] = group["config"].get("memory_clock_mhz")
            stratum["objective"] = objective
            buckets.setdefault(json.dumps(stratum, sort_keys=True), []).append((group, metric, throughput, clock_peak))
    results = []
    for key, candidates in sorted(buckets.items()):
        best = min(candidates, key=lambda item: (item[0][item[1]], -item[0][item[2]], item[0]["group_id"]))
        group, metric, throughput, clock_peak = best
        best_ci = group["ci95"].get(metric)
        def support(candidate):
            point, point_metric, point_throughput, point_peak = candidate
            return {"group_id": point["group_id"], "graphics_clock_mhz": point.get("graphics_clock_mhz"),
                    "memory_clock_mhz": point.get("memory_clock_mhz"),
                    "requested_graphics_clock_mhz": point["config"].get("graphics_clock_mhz"),
                    "requested_memory_clock_mhz": point["config"].get("memory_clock_mhz"),
                    "metric_value": point[point_metric], "metric_ci95": point["ci95"].get(point_metric),
                    "throughput": point[point_throughput], "throughput_fraction_of_own_clock_observed_peak": point[point_throughput] / point_peak,
                    "config": point["config"], "valid_repeats": point["valid_repeats"],
                    "paired_reference_order_counts": point.get("paired_reference_order_counts")}
        near = [candidate for candidate in candidates if candidate[0][metric] <= group[metric] * (1 + near_fraction)]
        overlapping = [candidate for candidate in candidates if best_ci and candidate[0]["ci95"].get(metric)
                       and candidate[0]["ci95"][metric][0] <= best_ci[1] and candidate[0]["ci95"][metric][1] >= best_ci[0]]
        sort_points = lambda points: sorted((support(candidate) for candidate in points), key=lambda point: (point["requested_memory_clock_mhz"], point["requested_graphics_clock_mhz"], point["group_id"]))
        stratum = json.loads(key)
        comparable = [point for point in populations if _clock_stratum(point, True) == {name: value for name, value in stratum.items() if name not in ("objective", "memory_clock_mhz")}
                      and (across_memory_domains or point["config"].get("memory_clock_mhz") == stratum["memory_clock_mhz"])]
        observed_pairs = sorted({(point["config"].get("graphics_clock_mhz"), point["config"].get("memory_clock_mhz")) for point in comparable})
        eligible_pairs = sorted({(point[0]["config"].get("graphics_clock_mhz"), point[0]["config"].get("memory_clock_mhz")) for point in candidates})
        as_pairs = lambda pairs: [{"graphics_clock_mhz": graphics, "memory_clock_mhz": memory} for graphics, memory in pairs]
        results.append({"stratum": json.loads(key), "gpu_name": group.get("gpu_name"), "best_group_id": group["group_id"],
                        "objective": json.loads(key)["objective"], "metric": metric, "metric_value": group[metric], "metric_ci95": best_ci,
                        "config": group["config"], "winning_requested_graphics_clock_mhz": group["config"].get("graphics_clock_mhz"),
                        "winning_requested_memory_clock_mhz": group["config"].get("memory_clock_mhz"),
                        "winning_achieved_graphics_clock_mhz": group.get("graphics_clock_mhz"),
                        "winning_achieved_memory_clock_mhz": group.get("memory_clock_mhz"),
                        "throughput": group[throughput], "own_clock_observed_max_throughput": clock_peak,
                        "throughput_fraction_of_own_clock_observed_peak": group[throughput] / clock_peak,
                        "throughput_threshold_fraction": fraction, "candidate_groups": len(candidates),
                        "observed_valid_frequency_pairs": as_pairs(observed_pairs), "eligible_frequency_pairs": as_pairs(eligible_pairs),
                        "observed_frequency_pairs_without_eligible_candidate": as_pairs(sorted(set(observed_pairs) - set(eligible_pairs))),
                        "selection_scope": "best observed eligible measured point; incomplete geometry, frequency or profiler coverage cannot establish the global hardware optimum",
                        "near_optimum_relative_tolerance": near_fraction,
                        "near_optimum_support_points": sort_points(near),
                        "uncertainty_overlap_support_points": sort_points(overlapping),
                        "all_eligible_support_points": sort_points(candidates),
                        "optimal_interval_kind": "discrete measured support points; unmeasured gaps and continuous intervals are not inferred",
                        "optimization_scope": "across measured graphics and memory frequency domains" if across_memory_domains else "across measured graphics frequencies at one fixed memory domain",
                        "utilization_basis": "complete observed geometry sweep maximum at each frequency pair; no theoretical saturation proof",
                        "energy_scope": "whole-device energy per counted logical payload bit or dense tensor FLOP; physical component energy is not isolated"})
    return results

def _verification_coverage(groups, fraction, min_repeats, cross_clock):
    buckets = {}
    for group in groups:
        if group["workload"] not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS or group["valid_repeats"] < min_repeats or not group["clock_comparison_controlled"]:
            continue
        throughput = "throughput_ops_s" if group["workload"] in _TENSOR_WORKLOADS else "throughput_bytes_s"
        if (group.get(throughput) or 0) <= 0:
            continue
        buckets.setdefault(json.dumps(_clock_stratum(group, cross_clock), sort_keys=True), []).append(group)
    result = []
    for key, candidates in sorted(buckets.items()):
        throughput = "throughput_ops_s" if candidates[0]["workload"] in _TENSOR_WORKLOADS else "throughput_bytes_s"
        peak = max(g[throughput] for g in candidates)
        verified = [g for g in candidates if g["verified_selection_eligible"]]
        verified_peak = max(g[throughput] for g in verified) if verified else None
        eligible = [g for g in verified if g[throughput] >= fraction * peak]
        result.append({"stratum": json.loads(key), "scope": "cross_clock" if cross_clock else "within_clock",
                       "observed_max_throughput_all_valid_groups": peak,
                       "observed_max_throughput_verified_groups": verified_peak,
                       "verified_peak_coverage_fraction": verified_peak / peak if verified_peak is not None else 0,
                       "throughput_threshold_fraction": fraction, "valid_groups": len(candidates),
                       "verified_groups": len(verified), "verified_high_throughput_groups": len(eligible),
                       "selection_status": "eligible_verified_conditions_available" if eligible else "no_verified_condition_reaches_complete_sweep_throughput_threshold",
                       "unverified_or_failed_peak_group_ids": [g["group_id"] for g in candidates if g[throughput] == peak and not g["verified_selection_eligible"]]})
    return result


def _pareto_frontiers(groups: list[dict[str, Any]], min_repeats: int, cross_clock: bool) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["workload"] not in _TENSOR_WORKLOADS | _MEMORY_WORKLOADS or group["valid_repeats"] < min_repeats or not group["clock_comparison_controlled"]:
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
                   and control.get("benchmark_sha256") == group.get("benchmark_sha256")
                   and control.get("measurement_stratum") == group.get("measurement_stratum")
                   and control.get("treatment_design_stratum") == group.get("treatment_design_stratum")
                   and all(control["config"].get(field) == group["config"].get(field) for field in match_fields)]
        if matches:
            matched.append({"workload_group_id": group["group_id"], "control_groups": [
                {"group_id": control["group_id"], "board_power_w": control["board_power_w"],
                 "incremental_activation_power_w": control["incremental_power_w"]} for control in matches],
                "matching_fields": ["gpu_uuid", "benchmark_sha256", "measurement_stratum", "treatment_design_stratum", *match_fields],
                "note": "Descriptive activation controls only. Different instruction mixes prevent automatic component-energy subtraction."})
    return matched


def summarize(records: Iterable[Mapping[str, Any]], throughput_fraction: float = 0.95,
              policy: AnalysisPolicy | Mapping[str, Any] | None = None, min_repeats: int = 3) -> dict[str, Any]:
    if not 0 < throughput_fraction <= 1 or min_repeats < 1:
        raise ValueError("throughput_fraction must be in (0, 1], min_repeats must be positive")
    policy = policy if isinstance(policy, AnalysisPolicy) else AnalysisPolicy(**(policy or {}))
    trials = []
    seen_ids = set()
    duplicate_ids = []
    for record in records:
        trial_id = record.get("trial_id")
        if trial_id is not None and trial_id in seen_ids:
            duplicate_ids.append(trial_id)
            continue
        if trial_id is not None:
            seen_ids.add(trial_id)
        trials.append(analyze_trial(record, policy))
    buckets: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        buckets.setdefault(_group_key(trial), []).append(trial)
    groups = []
    for key, repeats in sorted(buckets.items()):
        valid, seen_repeats, duplicate_repeat_indices = [], set(), []
        for trial in repeats:
            if not trial["valid"]:
                continue
            repeat_index = trial.get("repeat_index")
            if repeat_index is not None and repeat_index in seen_repeats:
                duplicate_repeat_indices.append(repeat_index)
                continue
            if repeat_index is not None:
                seen_repeats.add(repeat_index)
            valid.append(trial)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        group = {"group_id": digest, "gpu_uuid": repeats[0]["gpu_uuid"], "workload": repeats[0]["workload"],
                 "config": {k: v for k, v in repeats[0]["config"].items() if k not in _REPEAT_KEYS},
                 "benchmark_sha256": repeats[0]["benchmark_sha256"], "measurement_stratum": repeats[0]["measurement_stratum"],
                 "treatment_design_stratum": repeats[0]["treatment_design_stratum"],
                 "repeats": len(repeats), "valid_repeats": len(valid), "trial_ids": [t["trial_id"] for t in repeats],
                 "duplicate_repeat_indices_ignored": duplicate_repeat_indices,
                 "target_verified": bool(valid) and all(t["target_verified"] for t in valid),
                 "verified_selection_eligible": bool(valid) and all(t["verified_selection_eligible"] for t in valid),
                 "count_energy_time_alignment_exact": bool(valid) and all(t["count_energy_time_alignment_exact"] for t in valid),
                 "clock_comparison_controlled": bool(valid) and all(t["clock_comparison_controlled"] for t in valid),
                 "diagnostic_only": repeats[0]["diagnostic_only"],
                 "gpu_name": repeats[0]["gpu_name"],
                 "baseline_valid": bool(valid) and all(t["baseline_valid"] for t in valid),
                 "operational_idle_increment_eligible": bool(valid) and all(t["operational_idle_increment_eligible"] for t in valid),
                 "paired_active_reference_eligible": bool(valid) and all(t["paired_active_reference_eligible"] for t in valid),
                 "paired_reference_order_counts": {order: sum(t.get("paired_active_reference_protocol", {}).get("order") == order for t in valid) for order in ("AB", "BA")},
                 "confidence": "repeat_median" if len(valid) >= min_repeats else "insufficient_valid_repeats", "ci95": {}}
        for metric in _METRICS:
            values = [value for t in valid if (value := _finite(t.get(metric))) is not None]
            group[metric] = statistics.median(values) if values else None
            group["ci95"][metric] = _median_ci(values, int(digest, 16))
        statuses = {trial["ncu_status"] for trial in valid}
        group["ncu_status"] = "fail" if "fail" in statuses else "pass" if statuses == {"pass"} else "unprofiled" if statuses == {"unprofiled"} else "inconclusive"
        group["paired_reference_both_orders_observed"] = all(group["paired_reference_order_counts"].values())
        group["paired_reference_counterbalanced"] = (group["paired_reference_both_orders_observed"]
                                                       and group["paired_reference_order_counts"]["AB"] == group["paired_reference_order_counts"]["BA"])
        group["paired_reference_order_count_imbalance"] = group["paired_reference_order_counts"]["AB"] - group["paired_reference_order_counts"]["BA"]
        group["paired_active_reference_eligible"] = group["paired_active_reference_eligible"] and group["paired_reference_counterbalanced"]
        arm_medians = {order: _median(trial.get("paired_active_reference_power_w") for trial in valid if trial.get("paired_active_reference_protocol", {}).get("order") == order) for order in ("AB", "BA")}
        group["paired_reference_order_medians_power_w"] = arm_medians
        group["paired_reference_order_effect_power_w"] = arm_medians["AB"] - arm_medians["BA"] if None not in arm_medians.values() else None
        group["ncu_target_suitability"] = group["target_verified"]
        group["ncu_utilization_status"] = "diagnostic_only" if valid and all(trial["ncu_utilization_status"] == "diagnostic_only" for trial in valid) else "inconclusive"
        groups.append(group)
    return {"schema_version": 2, "trials": trials, "groups": groups, "duplicate_trial_ids_ignored": duplicate_ids,
            "within_clock_best": _selections(groups, throughput_fraction, min_repeats, False),
            "cross_clock_best": _selections(groups, throughput_fraction, min_repeats, True),
            "within_clock_best_total_energy": _selections(groups, throughput_fraction, min_repeats, False, total_energy=True),
            "cross_clock_best_total_energy": _selections(groups, throughput_fraction, min_repeats, True, total_energy=True),
            "uncontrolled_clock_exploratory_best": _selections(groups, throughput_fraction, min_repeats, True, exploratory_only=True),
            "pareto_frontiers": _pareto_frontiers(groups, min_repeats, False) + _pareto_frontiers(groups, min_repeats, True),
            "active_control_associations": _associate_controls(groups, min_repeats),
            "verified_target_within_clock_best": _selections(groups, throughput_fraction, min_repeats, False, True),
            "verified_target_cross_clock_best": _selections(groups, throughput_fraction, min_repeats, True, True),
            "verified_target_coverage": _verification_coverage(groups, throughput_fraction, min_repeats, False) + _verification_coverage(groups, throughput_fraction, min_repeats, True),
            "empirical_gpu_energy_optima": _empirical_energy_optima(groups, throughput_fraction, min_repeats, policy.near_optimum_fraction),
            "empirical_gpu_overall_energy_optima": _empirical_energy_optima(groups, throughput_fraction, min_repeats, policy.near_optimum_fraction, across_memory_domains=True),
            "metric_aliases": {"pj_per_op": "legacy operational idle increment per reported benchmark operation; tensor FLOP uses explicit operational_idle_increment_pj_per_flop",
                               "pj_per_logical_byte": "legacy operational idle increment per logical byte; divide by 8 for logical-bit metric",
                               "incremental_power_w": "operational_idle_increment_power_w; a baseline contrast, not physical switching power"},
            "selection_policy": {"throughput_fraction": throughput_fraction, "min_repeats": min_repeats,
                                 "primary_energy_objective": "whole-device total energy per logical bit or FLOP",
                                 "frequency_utilization_constraint": "own-frequency complete measured geometry peak; global-near-peak optima are reported separately",
                                 "baseline_objectives": "separate matched operational idle increment and paired active-reference contrast; never automatically substituted for total energy",
                                 "ci_method": "deterministic percentile bootstrap of repeat medians (1000 draws); n=3 intervals are coarse"},
            "notes": ["No fabricated measurement or architecture-specific idle wattage is supplied.",
                      "A device power limit is not measured consumption, and headline FLOP/s is not achieved throughput.",
                      "Idle subtraction cannot separate physical static leakage from refresh/clock/background power.",
                      "L1/L2/HBM logical bytes are not guaranteed physical traffic; numeric profiler evidence is recomputed for target verification.",
                      "Verified selections require exact matching work/power windows; old records retain explicitly qualified stationary-rate estimates.",
                      "Uncontrolled default-clock trials are comparative and excluded from controlled-frequency optimum searches.",
                      "Each GPU UUID has its own empirically selected optimum; 1110 MHz is a mandatory supported sweep point, not a preset optimum.",
                      "Near-optimum and uncertainty-overlap sets contain only measured support points; no continuous interval or unmeasured gap is claimed."]}


def write_summary(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = destination / "trials.csv"
    fields = ["trial_id", "gpu_uuid", "gpu_name", "workload", "valid", "target_verified", "ncu_status", "ncu_target_suitability", "ncu_utilization_status", "profiler_suitability_status",
              "verified_selection_eligible", "count_energy_time_alignment_exact", "energy_per_work_kind", "issues", "warnings", "baseline_valid", "baseline_issues", "operational_idle_increment_eligible", "paired_active_reference_eligible", "paired_active_reference_issues", "treatment_design_stratum", "baseline_clock_domains_compared", "baseline_sm_clock_uses_graphics_proxy", "paired_reference_clock_domains_compared", "paired_reference_sm_clock_uses_graphics_proxy", "duration_s", *_METRICS,
              "total_energy_j", "incremental_energy_j", "energy_source", "power_limit_w", "tensor_peak_clock_source"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in summary.get("trials", []):
            writer.writerow({field: json.dumps(trial[field]) if isinstance(trial.get(field), (list, dict)) else trial.get(field) for field in fields})
    return {"summary_json": str(json_path), "trials_csv": str(csv_path)}
