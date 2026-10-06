"""Evaluate measured components, with explicit missing evidence and study scope.

No profiler busy-time rate is substituted for sustained energy-run throughput.
Plateau means an observation over tested resource levels, not hardware saturation.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics

from .nonlinear import NONLINEAR_WORKLOADS

TENSOR = {"tensor", "gemm", "fp16_tensor", "tensor_fp16"}
MEMORY = {"l1", "l2", "hbm"}
OBJECTIVES = ("total", "operational_idle_increment", "paired_active_reference")


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def experiment_contract(workload, benchmark, config):
    def get(key, default=None):
        return benchmark.get(key, config.get(key, default))
    if workload in MEMORY or workload == "l2_latency":
        size = get("working_set_bytes")
        if workload == "l1":
            blocks = get("blocks")
            size = get("l1_bytes_per_block", size / blocks if number(size) is not None and number(blocks) and blocks > 0 else None)
        return {"access": get("access", "read"),
                "kernel_implementation_version": get("kernel_implementation_version"),
                "memory_accesses_per_thread_iteration": get("memory_accesses_per_thread_iteration"),
                "l1_bytes_per_block" if workload == "l1" else "working_set_bytes": size,
                "stride_elements": get("stride_elements", 1),
                **({"blocks": get("blocks"), "threads": get("threads"),
                    "iterations_per_launch": get("iterations_per_launch", config.get("iterations")),
                    "sm_filter_size": len(config.get("sm_ids") or [])} if workload == "l2_latency" else {}),
                **({"offset_bytes": get("offset_bytes", 0), "sm_ids": config.get("sm_ids", [])} if workload != "l2_latency" else {})}
    if workload in NONLINEAR_WORKLOADS:
        return {key: get(key) for key in ("working_set_bytes", "row_width", "input_precision", "math_implementation",
                                          "rms_epsilon", "affine_gamma", "nonlinear_input_distribution",
                                          "input_elements", "grid_mode", "kernel_implementation_version",
                                          "block_completion_count_source")}
    return {"definition": "dense FP16 input / FP32 accumulation", "implementation": workload} if workload in TENSOR else {"definition": "active integer issue control"}


def component_stratum(group):
    return {key: group.get(key) for key in ("gpu_uuid", "workload", "benchmark_sha256", "measurement_stratum",
                                            "treatment_design_stratum", "experiment_contract")}


def units(workload):
    if workload in NONLINEAR_WORKLOADS:
        return {"rate": "throughput_elements_s", "rate_unit": "Gelement/s", "rate_scale": 1e9,
                "energy_suffix": "pj_per_element", "energy_unit": "pJ/element"}
    if workload in TENSOR:
        return {"rate": "throughput_ops_s", "rate_unit": "TFLOP/s", "rate_scale": 1e12,
                "energy_suffix": "pj_per_flop", "energy_unit": "pJ/FLOP"}
    return {"rate": "throughput_bytes_s", "rate_unit": "logical GB/s", "rate_scale": 1e9,
            "energy_suffix": "pj_per_logical_bit", "energy_unit": "pJ/logical bit"}


@dataclass(frozen=True)
class EvaluationPolicy:
    plateau_tolerance_fraction: float = 0.05
    minimum_resource_levels: int = 3
    maximum_relative_ci_width: float = 0.10

    def __post_init__(self):
        if type(self.minimum_resource_levels) is not int or self.minimum_resource_levels < 3:
            raise ValueError("minimum_resource_levels must be an integer >=3")
        for value in (self.plateau_tolerance_fraction, self.maximum_relative_ci_width):
            if number(value) is None or not 0 < value < 1:
                raise ValueError("Evaluation fractions must be finite and in (0,1)")


def _ci_width(group, metric):
    value = number(group.get(metric))
    ci = group.get("ci95", {}).get(metric)
    if value is None or value == 0 or not isinstance(ci, list) or len(ci) != 2 or any(number(v) is None for v in ci):
        return None
    return (ci[1] - ci[0]) / abs(value) if ci[0] <= ci[1] else None


def _capacity(group):
    g = group.get("resource_geometry") or {}
    if group["workload"] == "gemm":
        dimensions = [number(g.get(k)) for k in ("gemm_m", "gemm_n", "gemm_k")]
        return math.prod(dimensions) if all(v is not None and v > 0 for v in dimensions) else None
    blocks, threads = number(g.get("blocks")), number(g.get("threads"))
    if blocks is None or threads is None or blocks <= 0 or threads <= 0:
        return None
    accumulators = number(g.get("tensor_accumulators")) or 1
    return blocks * threads * accumulators if group["workload"] == "tensor" else blocks * threads


def _qualified(group, metric, rate, minimum_repeats, policy):
    width = _ci_width(group, metric)
    rate_width = _ci_width(group, rate)
    return (group.get("valid_repeats", 0) >= minimum_repeats and group.get("verified_selection_eligible") is True
            and number(group.get(metric)) is not None and group[metric] >= 0
            and width is not None and width <= policy.maximum_relative_ci_width
            and rate_width is not None and rate_width <= policy.maximum_relative_ci_width)


def _eligibility_reasons(group, objective, metric, rate, minimum_repeats, policy):
    reasons = []
    if group.get("valid_repeats", 0) < minimum_repeats: reasons.append("Insufficient valid repeats")
    if group.get("target_verified") is not True: reasons.append("Target evidence: " + group.get("ncu_status", "unprofiled"))
    if group.get("count_energy_time_alignment_exact") is not True: reasons.append("Work and energy windows are not exactly aligned")
    if not group.get("clock_comparison_controlled"): reasons.append("Incoming clock policy is outside fixed-clock selection")
    if (number(group.get(rate)) or 0) <= 0: reasons.append("Positive sustained throughput is unavailable")
    for field, label in ((rate, "Throughput"), (metric, "Energy")):
        width = _ci_width(group, field)
        if width is None or width > policy.maximum_relative_ci_width: reasons.append(label + " repeat uncertainty is unavailable or too wide")
    if number(group.get(metric)) is None: reasons.append("Energy value is unavailable")
    elif group[metric] < 0: reasons.append("Negative operational contrast is diagnostic only")
    if objective != "total" and group.get(objective + "_eligible") is not True: reasons.append("Matched baseline or balanced paired reference is unqualified")
    return reasons


def _plateau(groups, rate, fraction, minimum_repeats, policy):
    observed = [g for g in groups if g.get("valid_repeats", 0) >= minimum_repeats and (number(g.get(rate)) or 0) > 0]
    peak = max((g[rate] for g in observed), default=None)
    levels = {}
    for g in observed:
        level = _capacity(g)
        width = _ci_width(g, rate)
        if (level is not None and g.get("verified_selection_eligible") is True
                and width is not None and width <= policy.maximum_relative_ci_width):
            if level not in levels or levels[level][rate] < g[rate]:
                levels[level] = g
    top = sorted(levels)[-policy.minimum_resource_levels:]
    selected = [levels[level] for level in top]
    rates = [g[rate] for g in selected]
    stable = (len(rates) >= policy.minimum_resource_levels and peak is not None
              and min(rates) >= fraction * peak and (max(rates) - min(rates)) / max(rates) <= policy.plateau_tolerance_fraction)
    gemm = bool(groups) and groups[0]["workload"] == "gemm"
    return {"status": ("observed_problem_size_plateau" if gemm else "observed_plateau") if stable else "inconclusive",
            "reason": "highest tested issue-resource levels have similar verified throughput" if stable else
                      "three verified, precise increasing resource levels near the complete observed peak are required",
            "resource_axis": "GEMM m × n × k; problem-size evidence, vendor algorithms can change" if gemm else "blocks × threads × accumulators (Tensor); blocks × threads otherwise",
            "resource_levels": top, "group_ids": [g["group_id"] for g in selected], "throughputs": rates,
            "observed_peak": peak, "hardware_saturation_proven": False}


def _nonlinear_launch(group):
    contract = group.get("nonlinear_contract") or {}
    geometry = group.get("resource_geometry") or {}
    return {"input_elements": contract.get("input_elements"),
            "grid_mode": contract.get("grid_mode"),
            "blocks": geometry.get("blocks"), "threads": geometry.get("threads"),
            "block_completion_count_source": contract.get("block_completion_count_source")}


def _input_curve_signature(group):
    """Compare Q only; never use this signature for energy aggregation."""
    contract = group.get("nonlinear_contract") or {}
    if (group.get("workload") not in NONLINEAR_WORKLOADS or contract.get("grid_mode") != "auto"
            or contract.get("math_implementation") != "cuda_fp32_q_grid_v2"
            or not group.get("clock_comparison_controlled")):
        return None
    q = number(contract.get("input_elements"))
    if q is None or q <= 0 or q != int(q):
        return None
    variable = {"input_elements", "working_set_bytes", "blocks"}
    stratum = component_stratum(group)
    stratum["experiment_contract"] = {k: v for k, v in (group.get("experiment_contract") or {}).items() if k not in variable}
    return canonical({**stratum,
        "nonlinear_contract": {k: v for k, v in contract.items() if k not in variable},
        "config": {k: v for k, v in group.get("config", {}).items() if k not in variable | {
            "clock_selection_policy", "clock_selection_reasons"}},
        "resource_geometry": {k: v for k, v in (group.get("resource_geometry") or {}).items() if k != "blocks"}})


def _input_size_plateau(candidate, peers, objective, metric, rate, fraction, minimum_repeats, policy):
    """A winner's Q curve can show stable throughput, not physical saturation.

    Keep unverified or imprecise valid rates in the observed peak and Q levels,
    so dropping a failed high-Q point cannot fabricate a lower stable plateau.
    """
    observed = [g for g in peers if g.get("valid_repeats", 0) >= minimum_repeats and (number(g.get(rate)) or 0) > 0]
    peak = max((g[rate] for g in observed), default=None)
    top = sorted({g["nonlinear_contract"]["input_elements"] for g in observed})[-policy.minimum_resource_levels:]
    selected = []
    for q in top:
        accepted = [g for g in observed if g["nonlinear_contract"]["input_elements"] == q
                    and _qualified(g, metric, rate, minimum_repeats, policy)
                    and not _eligibility_reasons(g, objective, metric, rate, minimum_repeats, policy)]
        if accepted:
            selected.append(max(accepted, key=lambda g: (g[rate], g["group_id"])))
    rates = [g[rate] for g in selected]
    selected_q = (candidate.get("nonlinear_contract") or {}).get("input_elements")
    spread = (max(rates) - min(rates)) / max(rates) if rates else None
    stable = (len(top) >= policy.minimum_resource_levels and len(selected) == len(top) and selected_q in top
              and peak is not None and min(rates) >= fraction * peak and spread <= policy.plateau_tolerance_fraction)
    return {"status": "observed_input_size_plateau" if stable else "inconclusive",
            "curve_id": hashlib.sha256(_input_curve_signature(candidate).encode()).hexdigest()[:12],
            "reason": "The selected Q lies in the largest tested Q levels with precise, verified near-peak throughput" if stable else
                      f"The selected Q must belong to the {policy.minimum_resource_levels} largest observed Q levels, each qualified and stable near the complete matching-curve peak",
            "resource_axis": "input elements Q at fixed threads, clocks, iterations and function definition",
            "input_elements_levels": top, "selected_input_elements": selected_q,
            "group_ids": [g["group_id"] for g in selected], "throughputs": rates,
            "observed_peak": peak, "relative_throughput_spread": spread,
            "hardware_saturation_proven": False,
            "scope": "Observed input-size stability only; Q-dependent energy values remain separate and SFU or GPU saturation is not established"}


def _input_scaling_rows(groups, curves, rate, suffix, minimum_repeats, policy):
    signatures = sorted({key for group in groups if (key := _input_curve_signature(group)) is not None})
    result = []
    for signature in signatures:
        rows = []
        for group in sorted(curves[signature], key=lambda g: (g["nonlinear_contract"]["input_elements"], g["group_id"])):
            reasons = {o: _eligibility_reasons(group, o, o + "_" + suffix, rate, minimum_repeats, policy) for o in OBJECTIVES}
            rows.append({"group_id": group["group_id"], **_nonlinear_launch(group),
                "requested_graphics_mhz": group["config"].get("graphics_clock_mhz"),
                "requested_memory_mhz": group["config"].get("memory_clock_mhz"),
                "rate": group.get(rate), "rate_ci95": group.get("ci95", {}).get(rate),
                "energies": {o: group.get(o + "_" + suffix) for o in OBJECTIVES},
                "energy_ci95": {o: group.get("ci95", {}).get(o + "_" + suffix) for o in OBJECTIVES},
                "valid_repeats": group.get("valid_repeats", 0), "ncu_status": group.get("ncu_status"),
                "eligible": {o: not reasons[o] and _qualified(group, o + "_" + suffix, rate, minimum_repeats, policy) for o in OBJECTIVES},
                "eligibility_reasons": reasons})
        result.append({"curve_id": hashlib.sha256(signature.encode()).hexdigest()[:12],
                       "signature": json.loads(signature), "rows": rows})
    return {"curves": result,
            "scope": "Each curve holds GPU, binary, measurement/input definition, requested fixed clocks, threads and iterations constant; only Q changes. Energies and recommendations remain separate per Q. Stable throughput is not proof of hardware saturation."}


def _plan_coverage(summary, plan):
    if plan is None:
        return {"status": "unknown", "reason": "No plan supplied; measured data alone cannot establish sweep completeness", "by_workload": {}}
    from .runner import validate_plan_execution, validate_trial_ids
    if not isinstance(plan, dict) or not isinstance(plan.get("trials"), list):
        raise ValueError("Evaluation plan must be an object with a trials list")
    validate_trial_ids(plan["trials"])
    try:
        validate_plan_execution(plan)
        policy_error = None
    except ValueError as exc:
        policy_error = str(exc)
    actual = {t["trial_id"]: t for t in summary["trials"] if t.get("trial_id") is not None}
    planned = plan.get("trials", [])
    expected = {t["trial_id"] for t in planned}
    mismatch, by_workload = [], {}
    for workload in sorted({t["workload"] for t in planned}):
        rows = [t for t in planned if t["workload"] == workload]
        missing, invalid, mismatched = [], [], []
        for row in rows:
            t = actual.get(row["trial_id"])
            if t is None:
                missing.append(row["trial_id"]); continue
            clocks, cfg = row.get("clocks", {}), t.get("config", {})
            uuid = (plan.get("device") or {}).get("uuid")
            expected_order = (row.get("treatment_protocol") or {}).get("order")
            expected_binary = (plan.get("device") or {}).get("benchmark_sha256")
            bound = (t.get("workload") == workload and (uuid is None or t.get("gpu_uuid") == uuid)
                     and t.get("repeat_index") == row.get("repeat")
                     and (expected_binary is None or t.get("benchmark_sha256") == expected_binary)
                     and t.get("condition_id") == row.get("condition_id")
                     and all(cfg.get(field) == clocks.get(key) for field, key in (("graphics_clock_mhz", "graphics_mhz"), ("memory_clock_mhz", "memory_mhz")))
                     and all(cfg.get(k) == v for k, v in row.get("parameters", {}).items())
                     and (expected_order is None or (t.get("paired_active_reference_protocol") or {}).get("order") == expected_order))
            if not bound:
                mismatched.append(row["trial_id"])
            if not t.get("valid"):
                invalid.append(row["trial_id"])
        mismatch.extend(mismatched)
        by_workload[workload] = {"planned_trials": len(rows), "observed_trials": len(rows) - len(missing),
            "missing_trial_ids": missing, "invalid_trial_ids": invalid, "mismatched_trial_ids": mismatched,
            "status": "incomplete" if missing or mismatched else "complete_with_rejections" if invalid else "complete"}
    unexpected = sorted(set(actual) - expected)
    duplicates = summary.get("duplicate_trial_ids_ignored", [])
    return {"status": "incomplete" if policy_error or unexpected or mismatch or duplicates or any(r["missing_trial_ids"] for r in by_workload.values()) else "complete",
            "plan_policy_error": policy_error, "study_design": plan.get("study_design", "legacy_unspecified"),
            "clock_policy": plan.get("clock_sweep_coverage"), "by_workload": by_workload,
            "unexpected_trial_ids": unexpected, "duplicate_trial_ids": duplicates,
            "note": "Coverage binds trial IDs, GPU, workload, clocks, parameters and paired-arm order; target evidence is assessed separately"}


def _match_signature(group):
    config = {k: v for k, v in group["config"].items() if k not in {
        "graphics_clock_mhz", "memory_clock_mhz", "clock_selection_policy", "clock_selection_reasons",
        "seconds", "warmup_seconds", "idle_seconds"}}
    return canonical({"config": config, "resource_geometry": group.get("resource_geometry")})


def _comparison(candidate, groups, objective, rate, metric, minimum_repeats, policy, anchor):
    refs = [g for g in groups if _match_signature(g) == _match_signature(candidate) and
            ("advertised_default_fixed_pair" in g["config"].get("clock_selection_reasons", []) if anchor == "factory_default" else
             g["config"].get("graphics_clock_mhz") == 1110 and g["config"].get("memory_clock_mhz") == candidate["config"].get("memory_clock_mhz"))]
    if len(refs) != 1:
        return {"status": "inconclusive", "reason": "A unique matched geometry/input/seed reference is missing", "anchor": anchor}
    ref = refs[0]
    eligible = objective == "total" or ref.get(objective + "_eligible") is True
    if not eligible or not _qualified(ref, metric, rate, minimum_repeats, policy) or ref[metric] <= 0:
        return {"status": "inconclusive", "reason": "Reference lacks qualified positive energy, repeats, exact counts or profile evidence", "anchor": anchor, "reference_group_id": ref["group_id"]}
    cci, rci = candidate["ci95"][metric], ref["ci95"][metric]
    bounds = [1 - cci[1] / rci[0], 1 - cci[0] / rci[1]] if rci[0] > 0 else None
    same = candidate["group_id"] == ref["group_id"]
    return {"status": "available", "anchor": anchor, "reference_group_id": ref["group_id"],
            "energy_reduction_fraction": 0 if same else 1 - candidate[metric] / ref[metric],
            "throughput_ratio": 1 if same else candidate[rate] / ref[rate],
            "energy_reduction_interval_from_ci_endpoints": [0, 0] if same else bounds,
            "evidence": "same_condition" if same else "resolved_improvement" if bounds and bounds[0] > 0 else "resolved_regression" if bounds and bounds[1] < 0 else "difference_unresolved",
            "uncertainty_note": "CI endpoint envelope is descriptive, not a calibrated ratio confidence interval; sensor systematic error is excluded"}


def _representative_trace(groups, analyzed, raw):
    candidates = sorted(groups, key=lambda g: (-g.get("valid_repeats", 0), g["group_id"]))
    for group in candidates:
        ts = [analyzed[i] for i in group["trial_ids"] if i in analyzed and analyzed[i].get("valid")]
        if not ts: continue
        ts.sort(key=lambda t: t.get("total_energy_j") or 0)
        trial = ts[len(ts) // 2]; record = raw.get(trial["trial_id"])
        if record is None: continue
        phases = record.get("phases") or {}
        if not isinstance(phases, dict): continue
        points = []
        for name, phase in phases.items():
            samples = phase.get("samples", [])
            indices = sorted({round(i * (len(samples) - 1) / min(95, len(samples) - 1)) for i in range(min(96, len(samples)))}) if len(samples) > 1 else list(range(len(samples)))
            points.extend({"phase": name, **{k: number(s.get(k)) for k in ("t_s", "power_w", "graphics_clock_mhz", "sm_clock_mhz", "memory_clock_mhz", "temperature_c")}}
                          for i in indices if isinstance((s := samples[i]), dict))
        return {"trial_id": trial["trial_id"], "group_id": group["group_id"], "points": points,
                "note": "One valid repeat; at most 96 observed samples per phase for display only. Energy uses original samples and exact complete epochs."}
    return None


def evaluate(summary, plan=None, policy=None, raw_records=()):
    policy = policy if isinstance(policy, EvaluationPolicy) else EvaluationPolicy(**(policy or {}))
    selection = summary["selection_policy"]
    minimum_repeats, minimum_geometries, fraction = (selection[k] for k in ("min_repeats", "min_geometries", "throughput_fraction"))
    coverage = _plan_coverage(summary, plan)
    buckets, input_curves = defaultdict(list), defaultdict(list)
    for group in summary.get("groups", []):
        buckets[canonical(component_stratum(group))].append(group)
        if (signature := _input_curve_signature(group)) is not None:
            input_curves[signature].append(group)
    analyzed = {t["trial_id"]: t for t in summary["trials"]}
    raw = {r.get("trial_id"): r for r in raw_records}
    components = []
    for key, groups in sorted(buckets.items()):
        stratum = json.loads(key); workload = stratum["workload"]; u = units(workload); rate = u["rate"]
        component_id = hashlib.sha256(key.encode()).hexdigest()[:12]
        point_rows, clock_buckets, evidence, latency = [], defaultdict(list), [], []
        for group in groups:
            ts = [analyzed[i] for i in group["trial_ids"] if i in analyzed]
            issues = Counter(issue for t in ts for issue in t.get("issues", []))
            cfg = group["config"]
            flags = list(cfg.get("clock_selection_reasons", []))
            if plan is not None:
                default = (plan.get("clock_sweep_coverage") or {}).get("advertised_default") or {}
                if default.get("status") == "included" and (cfg.get("graphics_clock_mhz"), cfg.get("memory_clock_mhz")) == (default.get("graphics_mhz"), default.get("memory_mhz")):
                    flags.append("advertised_default_fixed_pair")
            if cfg.get("graphics_clock_mhz") == 1110: flags.append("required_exact_1110_mhz")
            # Copy tags into local groups for matched anchor comparisons only.
            group = {**group, "config": {**cfg, "clock_selection_reasons": sorted(set(flags))}}
            widths = {m: _ci_width(group, m) for m in [rate, *(o + "_" + u["energy_suffix"] for o in OBJECTIVES)]}
            point_rows.append({"group_id": group["group_id"], "requested_graphics_mhz": cfg.get("graphics_clock_mhz"),
                "requested_memory_mhz": cfg.get("memory_clock_mhz"), "achieved_graphics_mhz": group.get("graphics_clock_mhz"),
                "achieved_sm_mhz": group.get("sm_clock_mhz"), "achieved_memory_mhz": group.get("memory_clock_mhz"),
                "resource_capacity": _capacity(group), "geometry": group.get("resource_geometry"), "anchor_tags": sorted(set(flags)),
                "ncu_status": group.get("ncu_status"), "valid_repeats": group.get("valid_repeats", 0), "rejected_repeats": group["repeats"] - group["valid_repeats"],
                "quality_issues": dict(issues), "relative_ci_widths": widths,
                "rate": group.get(rate), "rate_ci95": group.get("ci95", {}).get(rate),
                "energies": {o: group.get(o + "_" + u["energy_suffix"]) for o in OBJECTIVES},
                "energy_ci95": {o: group.get("ci95", {}).get(o + "_" + u["energy_suffix"]) for o in OBJECTIVES},
                "objective_eligible": {o: o == "total" or group.get(o + "_eligible") is True for o in OBJECTIVES},
                "eligibility_reasons": {o: _eligibility_reasons(group, o, o + "_" + u["energy_suffix"], rate, minimum_repeats, policy) for o in OBJECTIVES},
                "tensor_dense_peak_fraction": group.get("tensor_utilization_vs_dense_clock_peak"),
                "measurement_diagnostics": group.get("measurement_diagnostics"),
                "nonlinear_launch": _nonlinear_launch(group) if workload in NONLINEAR_WORKLOADS else None,
                "row_width": (group.get("nonlinear_contract") or {}).get("row_width"),
                "pj_per_row": {o: group.get(o + "_pj_per_row") for o in OBJECTIVES},
                "config": cfg})
            if group.get("clock_comparison_controlled"):
                clock_buckets[(cfg.get("graphics_clock_mhz"), cfg.get("memory_clock_mhz"))].append(group)
            checks = {}
            for t in ts:
                for check in (t.get("ncu_assessment") or {}).get("checks", []):
                    ck = (check.get("name"), check.get("status"))
                    checks[ck] = check
                for sm, probe in (t.get("latency_probe") or {}).get("per_sm", {}).items():
                    if t.get("valid") and (number(probe.get("cycles_per_access")) or 0) > 0:
                        latency.append({"trial_id": t["trial_id"], "group_id": group["group_id"], "sm_id": sm,
                            "offset_bytes": cfg.get("offset_bytes", 0), "graphics_mhz": cfg.get("graphics_clock_mhz"),
                            "memory_mhz": cfg.get("memory_clock_mhz"), "cycles_per_access": probe["cycles_per_access"]})
            evidence.append({"group_id": group["group_id"], "status": group.get("ncu_status"), "checks": list(checks.values()),
                "sampled_numerical_checks": [t.get("numerical_validation") for t in ts if t.get("numerical_validation")],
                "traffic_amplification": [t["ncu_assessment"]["traffic_amplification"] for t in ts
                                          if (t.get("ncu_assessment") or {}).get("traffic_amplification")],
                "profile_rates": [t["profile_rates_summary"] for t in ts if t.get("profile_rates_summary")]})
        clocks, candidates = [], defaultdict(list)
        for (gfx, mem), rows in sorted(clock_buckets.items()):
            observed = [g for g in rows if g["valid_repeats"] >= minimum_repeats and (number(g.get(rate)) or 0) > 0]
            peak = max((g[rate] for g in observed), default=None)
            verified = [g for g in observed if g.get("verified_selection_eligible")]
            count = len({canonical(g["resource_geometry"]) for g in verified})
            best = {}
            for objective in OBJECTIVES:
                metric = objective + "_" + u["energy_suffix"]
                eligible = [g for g in verified if count >= minimum_geometries and _qualified(g, metric, rate, minimum_repeats, policy)
                            and peak is not None and g[rate] >= fraction * peak
                            and (objective == "total" or g.get(objective + "_eligible") is True)]
                winner = min(eligible, key=lambda g: (g[metric], -g[rate], g["group_id"])) if eligible else None
                best[objective] = winner["group_id"] if winner else None
                candidates[objective].extend(eligible)
            q_grid = all(_input_curve_signature(g) is not None for g in rows)
            plateau = {"status": "not_applicable_q_grid", "reason": "Q determines the grid; selected-candidate evidence uses a matched input-size curve",
                       "hardware_saturation_proven": False} if q_grid else _plateau(rows, rate, fraction, minimum_repeats, policy)
            clocks.append({"graphics_mhz": gfx, "memory_mhz": mem, "observed_peak": peak,
                "verified_peak": max((g[rate] for g in verified), default=None), "verified_geometry_count": count,
                "minimum_energy_group_ids": best, "plateau": plateau})
        observed_peak = max((c["observed_peak"] for c in clocks if c["observed_peak"] is not None), default=None)
        for point in point_rows:
            clock = next((c for c in clocks if (c["graphics_mhz"], c["memory_mhz"]) == (point["requested_graphics_mhz"], point["requested_memory_mhz"])), None)
            extra = []
            if clock and clock["verified_geometry_count"] < minimum_geometries: extra.append("Too few verified execution geometries at this clock")
            if observed_peak and point["rate"] is not None and point["rate"] < fraction * observed_peak: extra.append("Below the all-valid observed throughput threshold")
            for reasons in point["eligibility_reasons"].values(): reasons.extend(extra)
        recommendations = []
        tagged = [g for rows in clock_buckets.values() for g in rows]
        for objective in OBJECTIVES:
            metric = objective + "_" + u["energy_suffix"]
            eligible = [g for g in candidates[objective] if observed_peak is not None and g[rate] >= fraction * observed_peak]
            if not eligible:
                recommendations.append({"objective": objective, "status": "no_qualified_candidate", "group_id": None,
                    "reason": "No candidate meets profile/count/repeat/CI/geometry gates and the full observed throughput threshold"}); continue
            winner = min(eligible, key=lambda g: (g[metric], -g[rate], g["group_id"]))
            clock = next(c for c in clocks if (c["graphics_mhz"], c["memory_mhz"]) == (winner["config"]["graphics_clock_mhz"], winner["config"]["memory_clock_mhz"]))
            reasons = []
            wc = coverage.get("by_workload", {}).get(workload, {})
            if coverage["status"] != "complete" or wc.get("status") not in ("complete", "complete_with_rejections"):
                reasons.append("planned coverage is incomplete or unknown")
            if coverage.get("study_design") != "energy_sweep": reasons.append("a complete energy-sweep plan is required")
            curve = _input_curve_signature(winner)
            plateau = (_input_size_plateau(winner, input_curves[curve], objective, metric, rate, fraction, minimum_repeats, policy)
                       if curve is not None else clock["plateau"])
            if plateau["status"] not in ("observed_plateau", "observed_problem_size_plateau", "observed_input_size_plateau"):
                reasons.append("input-size plateau is inconclusive" if curve is not None else "resource plateau is inconclusive")
            recommendations.append({"objective": objective, "group_id": winner["group_id"],
                "status": "provisional_candidate" if reasons else "qualified_observed_candidate", "qualification_limits": reasons,
                "saturation_evidence": plateau,
                "energy": winner[metric], "energy_ci95": winner["ci95"].get(metric), "throughput_fraction_of_observed_peak": winner[rate] / observed_peak,
                "near_optimum_group_ids": [g["group_id"] for g in eligible if g[metric] <= winner[metric] * (1 + selection.get("near_optimum_fraction", 0.05))],
                "factory_default_comparison": _comparison(winner, tagged, objective, rate, metric, minimum_repeats, policy, "factory_default"),
                "exact_1110_comparison": _comparison(winner, tagged, objective, rate, metric, minimum_repeats, policy, "exact_1110")})
        selection_diagnostics = []
        for objective in OBJECTIVES:
            metric = objective + "_" + u["energy_suffix"]
            own = min(candidates[objective], key=lambda g: (g[metric], -g[rate], g["group_id"]), default=None)
            recommendation = next(r for r in recommendations if r["objective"] == objective)
            energy = recommendation.get("energy")
            selection_diagnostics.append({"objective": objective,
                "lowest_own_clock_candidate_group_id": own["group_id"] if own else None,
                "lowest_own_clock_candidate_energy": own[metric] if own else None,
                "own_clock_candidate_global_throughput_fraction": own[rate] / observed_peak if own and observed_peak else None,
                "global_constraint_candidate_group_id": recommendation.get("group_id"),
                "global_constraint_candidate_energy": energy,
                "global_constraint_energy_over_own_clock_energy": energy / own[metric] if own and own[metric] > 0 and energy is not None else None,
                "reason": "The lowest own-clock candidate falls below the full-sweep throughput threshold" if own and observed_peak and own[rate] < fraction * observed_peak else "The lowest own-clock candidate meets the full-sweep throughput threshold" if own else "No qualified own-clock candidate",
                "scope": "Same count/profile/repeat/CI/geometry gates and own-clock throughput bar; comparing the added full-sweep throughput constraint, not correcting energy or proving saturation"})
        components.append({"component_id": component_id, "stratum": stratum, "gpu_name": groups[0].get("gpu_name"), "units": u,
            "trial_counts": dict(Counter("valid" if analyzed[i].get("valid") else "invalid" for g in groups for i in g["trial_ids"] if i in analyzed)),
            "points": point_rows, "clocks": clocks, "observed_peak": observed_peak, "recommendations": recommendations,
            "energy_selection_diagnostics": selection_diagnostics,
            "input_scaling": _input_scaling_rows(groups, input_curves, rate, u["energy_suffix"], minimum_repeats, policy),
            "profiler_evidence": evidence, "latency_points": latency,
            "correctness_scope": "distributed CPU-double output samples before/after measurement; not exhaustive" if workload in NONLINEAR_WORKLOADS else
                                 "finite output samples and issued-operation accounting; full mathematical reference not established" if workload in TENSOR else
                                 "issued addresses/counts and path evidence; no physical partition identity or exhaustive bytewise output proof",
            "measurement_example": _representative_trace(groups, analyzed, raw),
            "interpretation": "dependent-load latency map; no near/far or energy optimum" if workload == "l2_latency" else
                              "active issue-control reference; no component energy optimum" if workload == "control" else
                              "measured whole-device energy of the stated implementation; component rail energy is not isolated"})
    return {"schema_version": 1, "policy": {**asdict(policy), "throughput_fraction": fraction, "minimum_repeats": minimum_repeats,
              "minimum_geometries": minimum_geometries}, "coverage": coverage, "components": components,
            "missing_workloads": sorted(set(coverage.get("by_workload", {})) - {c["stratum"]["workload"] for c in components}),
            "limits": ["All optima are restricted to observed supported clocks and tested configurations.",
                       "Bootstrap intervals measure repeat variation, not sensor calibration accuracy.",
                       "Profiler counter rates use replay busy time and are never energy-run sustained throughput.",
                       "Compare the same energy objective and denominator. Total/idle ratios and replay traffic amplification diagnose differences; they do not rescale the measured energy.",
                       "A resource plateau is empirical evidence; it does not prove hardware saturation or pure component energy.",
                       "Nonlinear auto-grid candidates use matching Q-scaling evidence; input-size stability does not prove SFU or GPU saturation, and energy is never pooled across Q.",
                       "Factory default means the advertised fixed clock pair; incoming driver policy is a separate reference."]}
