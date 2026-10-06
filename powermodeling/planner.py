"""Expand explicit, reproducible sweeps with a restricted numeric expression grammar."""

import ast
import hashlib
import itertools
import json
import math
import random

from .profiles import declare_sxm
from .nonlinear import NONLINEAR_WORKLOADS, ROW_WORKLOADS

WORKLOADS = {"tensor", "gemm", "l1", "l2", "l2_latency", "hbm", "control"} | NONLINEAR_WORKLOADS
PARAMETERS = {"blocks", "threads", "working_set_bytes", "stride_elements", "iterations",
              "tensor_accumulators", "access", "sm_ids", "seed", "offset_bytes",
              "gemm_m", "gemm_n", "gemm_k", "batch_launches", "row_width"}


def numeric_expression(value, names):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = value
    elif isinstance(value, str):
        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.Name) and node.id in names:
                return names[node.id]
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Sub): return a - b
                if isinstance(node.op, ast.Mult): return a * b
                if isinstance(node.op, ast.Div): return a / b
                return a // b
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max") and not node.keywords and node.args:
                return (min if node.func.id == "min" else max)(visit(arg) for arg in node.args)
            raise ValueError(f"Disallowed experiment expression: {value!r}")
        result = visit(ast.parse(value, mode="eval").body)
    else:
        raise ValueError(f"Expected numeric value/expression, got {value!r}")
    if not math.isfinite(result) or result < 0 or result > 2**63-1:
        raise ValueError(f"Expression outside supported range: {value!r}")
    if result != int(result):
        raise ValueError(f"Parameter expression must evaluate to an integer, got {value!r}")
    return int(result)


def _quantiles(values, quantiles):
    if any(type(x) is not int or x <= 0 for x in values):
        raise ValueError("Supported clock MHz values must be positive integers")
    values = sorted(set(values))
    if not values:
        raise ValueError("Supported clocks unavailable; supply verified explicit clock_pairs")
    if not isinstance(quantiles, list) or not quantiles or any(isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1 for q in quantiles):
        raise ValueError("Clock quantiles must lie in [0,1]")
    return sorted({values[round(q * (len(values)-1))] for q in quantiles})


def _supported_domains(supported):
    by_mem = {}
    raw = supported.get("supported_pairs", supported.get("pairs", []))
    if not isinstance(raw, list):
        raise ValueError("Supported clock pairs must be a list")
    for pair in raw:
        if not isinstance(pair, dict):
            raise ValueError("Supported clock pair must be an object")
        memory = pair.get("memory_mhz", pair.get("memory_clock_mhz"))
        graphics = pair.get("graphics_mhz", pair.get("graphics_clock_mhz"))
        if any(type(value) is not int or value <= 0 for value in (memory, graphics)):
            raise ValueError("Advertised graphics/memory clock MHz values must be positive integers")
        by_mem.setdefault(memory, set()).add(graphics)
    return {memory: sorted(graphics) for memory, graphics in sorted(by_mem.items())}


def _clock_key(pair):
    return pair["graphics_mhz"], pair["memory_mhz"]


def _study_design(config):
    design = config.get("study_design", "legacy_unspecified")
    if design not in ("energy_sweep", "diagnostic", "legacy_unspecified"):
        raise ValueError("study_design must be energy_sweep, diagnostic or legacy_unspecified")
    return design


def _strict_clock_policy(config):
    """The declared energy study cannot silently become a partial clock probe."""
    if _study_design(config) != "energy_sweep":
        return
    if "clock_pairs" in config:
        raise ValueError("energy_sweep requires a discovered interval clock grid, not explicit clock_pairs")
    sweep = config.get("clock_sweep", {})
    if not isinstance(sweep, dict):
        raise ValueError("clock_sweep must be an object")
    if ("graphics_quantiles" in sweep or type(sweep.get("graphics_step_mhz", 90)) is not int
            or sweep.get("graphics_step_mhz", 90) <= 0):
        raise ValueError("energy_sweep requires a positive graphics_step_mhz; graphics quantile exceptions are not allowed")
    if sweep.get("all_memory_clocks") is not True or "memory_mhz" in sweep or "memory_quantiles" in sweep:
        raise ValueError("energy_sweep requires all_memory_clocks=true without a memory-domain subset")
    if sweep.get("include_advertised_default", True) is not True:
        raise ValueError("energy_sweep cannot omit the advertised default fixed pair")
    if sweep.get("include_default_policy", True) is not True:
        raise ValueError("energy_sweep cannot omit the separate incoming-policy reference")


def _requirements_coverage(coverage, strict):
    """Unknown default is unresolved; unsupported exact anchors are inapplicable."""
    if not strict:
        return {"requirements_status": "not_required", "execution_allowed": True,
                "requirement_checks": [], "requirement_reasons": []}
    checks = [
        {"requirement": "graphics_interval_grid", "status": "satisfied", "reason": "requested interval grid maps to advertised graphics clocks within the evaluation range",
         "step_mhz": coverage["requested_step_mhz"], "minimum_mhz": coverage["graphics_min_mhz"]},
        {"requirement": "all_memory_domains", "status": "satisfied", "reason": "every advertised memory-clock domain is selected",
         "memory_mhz": coverage["selected_memory_mhz"],
         "no_supported_graphics_in_evaluation_range_memory_mhz": [d["memory_mhz"] for d in coverage["memory_domains"] if d["evaluation_range_status"] == "not_applicable"]},
    ]
    default = coverage["advertised_default"]
    if default["status"] == "included":
        checks.append({"requirement": "advertised_default_fixed_pair", "status": "satisfied",
                       "reason": "exact advertised default pair included", "clocks": {
                           "graphics_mhz": default["graphics_mhz"], "memory_mhz": default["memory_mhz"]}})
    else:
        checks.append({"requirement": "advertised_default_fixed_pair", "status": "unresolved",
                       "reason": "advertised default pair is unavailable; incoming-policy reference is not a substitute"})
    checks.append({"requirement": "incoming_policy_reference", "status": "satisfied",
                   "reason": "separate no-mutation reference included; factory-default policy is not asserted"})
    supported = []
    unsupported = []
    for domain in coverage["memory_domains"]:
        point = next(point for point in domain["required_points"] if point["mhz"] == 1110)
        (supported if point["status"] == "included" else unsupported).append(domain["memory_mhz"])
    checks.append({"requirement": "exact1110_when_supported", "status": "satisfied" if supported else "not_applicable",
                   "reason": "exact1110MHz included in every supporting memory domain" if supported else "no advertised memory domain supports exact1110MHz",
                   "included_memory_mhz": supported, "not_applicable_memory_mhz": unsupported})
    reasons = [check["reason"] for check in checks if check["status"] == "unresolved"]
    return {"requirements_status": "incomplete" if reasons else "complete",
            "execution_allowed": not reasons, "requirement_checks": checks, "requirement_reasons": reasons}


def resolve_clock_plan(config, supported):
    """Return real supported pairs plus explicit grid/default/anchor coverage.

    Nearest supported values are used only for approximate grid points. Required
    points (1110 MHz by default) must be exact or are recorded as unavailable.
    The advertised default pair and the incoming-policy reference are separate:
    the latter never changes driver policy or asserts it is a factory default.
    """
    if not isinstance(config, dict) or not isinstance(supported, dict):
        raise ValueError("Experiment config and supported clock discovery must be objects")
    _strict_clock_policy(config)
    strict = _study_design(config) == "energy_sweep"
    if "clock_pairs" in config and "clock_sweep" in config:
        raise ValueError("Choose explicit clock_pairs or a discovered clock_sweep, not both")
    by_mem = _supported_domains(supported)
    if "clock_pairs" in config:
        pairs = config["clock_pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("clock_pairs cannot be empty")
        normalized = []
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("Explicit clock pair must be an object")
            graphics, memory = pair.get("graphics_mhz"), pair.get("memory_mhz")
            if (graphics is None) != (memory is None):
                raise ValueError("Specify both graphics and memory clock, or neither")
            if graphics is not None and (type(graphics) is not int or graphics <= 0 or type(memory) is not int or memory <= 0):
                raise ValueError("Clock MHz values must be positive integers")
            if graphics is not None and by_mem and graphics not in by_mem.get(memory, []):
                raise ValueError(f"Explicit clock pair {graphics}/{memory} MHz is not advertised by this GPU")
            normalized.append({"graphics_mhz": graphics, "memory_mhz": memory})
        if len({(p["graphics_mhz"], p["memory_mhz"]) for p in normalized}) != len(normalized):
            raise ValueError("clock_pairs contains duplicates")
        coverage = {"strategy": "explicit_pairs", "requested_step_mhz": None,
                    "supported_pair_validation": "validated" if by_mem else "deferred_to_runtime_discovery",
                    "selected_memory_mhz": sorted({p["memory_mhz"] for p in normalized if p["memory_mhz"] is not None}),
                    "memory_domains": [], "clock_conditions": [
                        {"clocks": pair, "clock_policy": "fixed_explicit" if pair["graphics_mhz"] is not None else "incoming_policy_reference",
                         "selection_reasons": ["explicit_configuration"]} for pair in normalized],
                    "note": "Explicit pairs are not advertised as a complete frequency sweep."}
        coverage.update(_requirements_coverage(coverage, strict=False))
        return normalized, coverage
    sweep = config.get("clock_sweep", {})
    if not isinstance(sweep, dict):
        raise ValueError("clock_sweep must be an object")
    if not by_mem:
        raise ValueError("Supported clocks unavailable; cannot invent operating ranges for a frequency sweep")
    memory_options = [key for key in ("memory_mhz", "all_memory_clocks", "memory_quantiles") if key in sweep]
    if len(memory_options) > 1:
        raise ValueError("Choose exactly one memory selection policy: memory_mhz, all_memory_clocks or memory_quantiles")
    if "memory_mhz" in sweep:
        memories = sweep["memory_mhz"]
        if not isinstance(memories, list) or not memories or any(type(m) is not int or m not in by_mem for m in memories) or len(set(memories)) != len(memories):
            raise ValueError("memory_mhz must list unique advertised memory-domain clocks")
        memories = sorted(memories)
        memory_policy = "explicit_supported_memory_domains"
    elif "all_memory_clocks" in sweep:
        if sweep["all_memory_clocks"] is not True:
            raise ValueError("all_memory_clocks must be true when supplied")
        memories = sorted(by_mem)
        memory_policy = "all_supported_memory_domains"
    else:
        memories = _quantiles(list(by_mem), sweep.get("memory_quantiles", [1.0]))
        memory_policy = "supported_memory_domain_quantiles"
    if "graphics_step_mhz" in sweep and "graphics_quantiles" in sweep:
        raise ValueError("Choose graphics_step_mhz or graphics_quantiles, not both")
    grid = "graphics_quantiles" not in sweep
    step = sweep.get("graphics_step_mhz", 90) if grid else None
    if grid and (type(step) is not int or step <= 0):
        raise ValueError("graphics_step_mhz must be a positive integer")
    floor = sweep.get("graphics_min_mhz", 900)
    if type(floor) is not int or floor <= 0:
        raise ValueError("graphics_min_mhz must be a positive integer")
    required = sweep.get("required_graphics_mhz", [1110])
    if not isinstance(required, list) or any(type(g) is not int or g <= 0 for g in required) or len(set(required)) != len(required):
        raise ValueError("required_graphics_mhz must contain unique positive integer exact anchors")
    # 1110 MHz is mandatory for this study wherever the exact pair exists.
    # Additional anchors can be requested without accidentally removing it.
    required = sorted(set(required) | {1110})
    for key in ("include_default_policy", "include_advertised_default"):
        if key in sweep and type(sweep[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    coverage = {"strategy": "nearest_supported_graphics_grid" if grid else "geometry_graphics_quantiles",
                "requested_step_mhz": step, "graphics_min_mhz": floor, "memory_selection_policy": memory_policy,
                "supported_memory_mhz": sorted(by_mem), "selected_memory_mhz": memories,
                "required_graphics_mhz": required, "memory_domains": [], "clock_conditions": [],
                "note": "Only advertised pairs are fixed. Grid anchors use nearest supported clocks; required points are exact. An observed optimum is GPU-specific and restricted to measured configurations."}
    conditions = {}

    def add(graphics, memory, reason, policy="fixed_supported_sweep"):
        pair = {"graphics_mhz": graphics, "memory_mhz": memory}
        key = _clock_key(pair)
        if key not in conditions:
            conditions[key] = {"clocks": pair, "clock_policy": policy, "selection_reasons": []}
        if reason not in conditions[key]["selection_reasons"]:
            conditions[key]["selection_reasons"].append(reason)

    for memory in memories:
        graphics = by_mem[memory]
        low, high = graphics[0], graphics[-1]
        eligible = [g for g in graphics if g >= floor]
        grid_start = max(floor, low)
        if grid and (high - grid_start) // step + 1 > 100000:
            raise ValueError("Requested graphics grid exceeds100000 points; check discovery or increase step")
        targets = list(range(grid_start, high + 1, step)) if grid and eligible else []
        mapping = []
        if grid:
            for target in targets:
                nearest = min(eligible, key=lambda g: (abs(g - target), g))
                mapping.append({"requested_mhz": target, "selected_mhz": nearest, "error_mhz": nearest - target})
                add(nearest, memory, "graphics_grid")
        else:
            for g in (_quantiles(eligible, sweep["graphics_quantiles"]) if eligible else []):
                add(g, memory, "geometry_graphics_quantile")
        # Operating endpoints of each *selected* memory domain are explicit.
        for g in ([eligible[0], high] if eligible else []):
            add(g, memory, "evaluation_range_endpoint")
        points = []
        for anchor in required:
            status = "included" if anchor in graphics else "not_applicable" if strict else "unavailable"
            reason = ("exact_supported_pair" if status == "included" else
                      "outside_advertised_operating_range" if anchor < low or anchor > high else
                      "not_an_advertised_discrete_graphics_clock")
            points.append({"mhz": anchor, "status": status, "reason": reason})
            if status == "included":
                add(anchor, memory, f"required_exact_{anchor}_mhz")
        selected = sorted(g for g, m in conditions if m == memory)
        coverage["memory_domains"].append({"memory_mhz": memory, "supported_graphics_mhz": graphics,
            "min_graphics_mhz": low, "max_graphics_mhz": high, "requested_grid_mhz": targets,
            "evaluation_min_graphics_mhz": eligible[0] if eligible else None,
            "evaluation_max_graphics_mhz": high if eligible else None,
            "evaluation_range_status": "included" if eligible else "not_applicable",
            "excluded_below_floor_graphics_mhz": [g for g in graphics if g < floor],
            "below_floor_scope": "excluded from the routine grid; mandatory default/exact anchors can still be included",
            "grid_mapping": mapping, "selected_graphics_mhz": selected,
            "actual_gaps_mhz": [b - a for a, b in zip(selected, selected[1:])],
            "required_points": points,
            "gap_note": "Native supported-clock gaps and added exact/default anchors can change spacing from the requested step."})
    default_g = supported.get("default_applications_graphics_mhz")
    default_m = supported.get("default_applications_memory_mhz")
    default_requested = sweep.get("include_advertised_default", True)
    valid_default = (type(default_g) is int and type(default_m) is int
                     and default_g in by_mem.get(default_m, []))
    coverage["advertised_default"] = {"requested": default_requested,
        "graphics_mhz": default_g, "memory_mhz": default_m,
        "source": "nvmlDeviceGetDefaultApplicationsClock",
        "status": "included" if default_requested and valid_default else "not_requested" if not default_requested else "unavailable",
        "reason": "exact_advertised_default_pair" if valid_default else "default_pair_unknown_or_not_in_supported_pair_list"}
    if default_requested and valid_default:
        add(default_g, default_m, "advertised_default_fixed_pair")
    policy_requested = sweep.get("include_default_policy", True)
    coverage["default_policy_reference"] = {
        "requested": policy_requested, "status": "included_unverified_factory_default" if policy_requested else "not_requested",
        "clock_policy": "incoming_policy_reference", "requested_default_alias": True,
        "factory_default_policy_verified": False,
        "incoming_applications_graphics_mhz": supported.get("applications_graphics_mhz"),
        "incoming_applications_memory_mhz": supported.get("applications_memory_mhz"),
        "note": "No mutation: preserve the incoming driver policy, including unknown locked ranges. This reference is not claimed to be factory-default or fixed-frequency."}
    if policy_requested:
        add(None, None, "default_policy_reference", "incoming_policy_reference")
    # Recompute selections after default insertion; default may be in an
    # additional memory domain and is then a distinct reference, not a sweep.
    for domain in coverage["memory_domains"]:
        selected = sorted(g for g, m in conditions if m == domain["memory_mhz"])
        domain["selected_graphics_mhz"] = selected
        domain["actual_gaps_mhz"] = [b - a for a, b in zip(selected, selected[1:])]
    ordered = sorted(conditions.values(), key=lambda row: (
        row["clocks"]["memory_mhz"] is None, row["clocks"]["memory_mhz"] or 0,
        row["clocks"]["graphics_mhz"] or 0))
    coverage["clock_conditions"] = ordered
    coverage["required_anchor_coverage_complete"] = all(point["status"] in ("included", "not_applicable")
        for domain in coverage["memory_domains"] for point in domain["required_points"])
    coverage.update(_requirements_coverage(coverage, strict))
    return [row["clocks"] for row in ordered], coverage


def resolve_clocks(config, supported):
    """Backwards-compatible pair list; full coverage lives in expanded plans."""
    return resolve_clock_plan(config, supported)[0]


def expand_plan(config, device, supported_clocks=None, stage=None):
    device = declare_sxm(device, config.get("target_form_factor", "SXM"))
    seconds = float(config.get("seconds", 12))
    warmup = float(config.get("warmup_seconds", 3))
    idle = float(config.get("idle_seconds", 6))
    repeats = config.get("repeats", 4)
    interval = float(config.get("sample_interval_s", 0.05))
    if (any(not math.isfinite(value) for value in (seconds, warmup, idle, interval))
            or type(repeats) is not int or seconds < 10 or warmup < 1 or idle < 6
            or repeats < 3 or not 0.005 <= interval <= 1):
        raise ValueError("Energy experiments require seconds>=10, warmup>=1, idle>=6, repeats>=3 and sample interval 0.005..1s")
    clocks, clock_coverage = resolve_clock_plan(config, supported_clocks or {})
    clock_annotations = {_clock_key(row["clocks"]): row for row in clock_coverage["clock_conditions"]}
    names = {"sm_count": int(device["sm_count"]), "l2_bytes": int(device["l2_bytes"]),
             "total_memory_bytes": int(device["total_memory_bytes"])}
    if any(value <= 0 for value in names.values()):
        raise ValueError("Discovered device capacities and SM count must be positive")
    trials = []
    conditions = set()
    geometry_keys = {}
    paired_default = config.get("paired_reference", True)
    if type(paired_default) is not bool:
        raise ValueError("paired_reference must be boolean")
    for spec in config["experiments"]:
        if stage and spec.get("stage", "saturation") != stage: continue
        workload = spec["workload"]
        if workload not in WORKLOADS: raise ValueError(f"Unknown workload {workload}")
        paired = spec.get("paired_reference", paired_default if workload in {"tensor", "l1", "l2", "hbm"} | NONLINEAR_WORKLOADS else False)
        if type(paired) is not bool:
            raise ValueError("Experiment paired_reference must be boolean")
        if paired and workload not in {"tensor", "l1", "l2", "hbm", "gemm"} | NONLINEAR_WORKLOADS:
            raise ValueError("Paired issue-loop reference is available only for treatment workloads; latency/control remain diagnostic")
        grid = spec.get("grid", {})
        if set(grid) - PARAMETERS: raise ValueError(f"Unknown parameters: {set(grid)-PARAMETERS}")
        if any(not isinstance(v, list) or not v for v in grid.values()):
            raise ValueError("Each grid parameter must be a nonempty list")
        keys = sorted(grid, key=lambda k: (k not in ("blocks", "threads"), k))
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(spec.get("parameters", {}))
            params.update(dict(zip(keys, values)))
            if set(params)-PARAMETERS: raise ValueError(f"Unknown parameters: {set(params)-PARAMETERS}")
            resolved, env = {}, dict(names)
            for key in sorted(params, key=lambda k: (k not in ("blocks", "threads"), k)):
                value = params[key]
                resolved[key] = value if key in ("access", "sm_ids") else numeric_expression(value, env)
                env[key] = resolved[key]
            blocks, threads = resolved.get("blocks", names["sm_count"]*2), resolved.get("threads", 256)
            if blocks < 1 or threads < 32 or threads > 1024 or threads % 32:
                raise ValueError("blocks>=1 and threads must be a multiple of warp size32 within32..1024")
            if blocks > 1000000 or blocks > 2**31-1:
                raise ValueError("blocks exceeds the benchmark practical limit of1000000")
            access = resolved.get("access", "read")
            if access not in ("read", "write", "copy"):
                raise ValueError("access must be read, write or copy")
            if workload in ("l1", "l2_latency") and access != "read":
                raise ValueError("L1 experiment supports read only; stores are not equivalent L1 accesses")
            if resolved.get("stride_elements", 1) < 1 or resolved.get("iterations", 1) < 1:
                raise ValueError("stride_elements and iterations must be positive")
            if resolved.get("stride_elements", 1) > 2**32 or resolved.get("iterations", 1) > 2**32:
                raise ValueError("stride_elements and iterations must be <=2^32")
            if not 1 <= resolved.get("tensor_accumulators", 4) <= 8:
                raise ValueError("tensor_accumulators must be within1..8")
            if not 1 <= resolved.get("batch_launches", 1) <= 65536:
                raise ValueError("batch_launches must be within1..65536")
            if workload == "l2_latency" and resolved.get("stride_elements", 1) != 1:
                raise ValueError("Dependent latency probes require stride_elements=1")
            if "sm_ids" in resolved:
                ids = resolved["sm_ids"]
                if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < 4096 for i in ids) or len(set(ids)) != len(ids):
                    raise ValueError("sm_ids must be a nonempty unique list of integers within0..4095")
                if workload == "gemm" or workload in NONLINEAR_WORKLOADS:
                    raise ValueError("GEMM/nonlinear full-output workloads cannot honor sm_ids")
                # SM IDs are sparse hardware identifiers on some SKUs. They
                # cannot be inferred merely from the enabled SM count.
                known_ids = device.get("discovered_sm_ids")
                if known_ids is not None and any(i not in known_ids for i in ids):
                    raise ValueError("sm_ids includes an ID absent from the discovered hardware map")
            if "row_width" in resolved and workload not in ROW_WORKLOADS:
                raise ValueError("row_width applies only to RMSNorm and Softmax")
            width = resolved.get("row_width", 1024) if workload in ROW_WORKLOADS else 0
            if workload in ROW_WORKLOADS and not 1 <= width <= 65536:
                raise ValueError("row_width must be within 1..65536")
            if workload in NONLINEAR_WORKLOADS: default_ws = blocks * (width or threads * 128) * 4
            elif workload == "l1": default_ws = blocks * 16 * 1024
            elif workload in ("l2", "l2_latency"): default_ws = max(4, names["l2_bytes"] // 8 * 4)
            else: default_ws = max(512 * 1024 * 1024, names["l2_bytes"] * 8)
            ws = resolved.get("working_set_bytes", default_ws)
            offset = resolved.get("offset_bytes", 0)
            if ws < 4 or ws % 4 or offset % 4:
                raise ValueError("working_set_bytes and offset_bytes must be word-aligned (4bytes), with a nonempty working set")
            if workload in NONLINEAR_WORKLOADS:
                if access != "read" or offset != 0 or resolved.get("stride_elements", 1) != 1:
                    raise ValueError("Nonlinear kernels require access=read, offset_bytes=0 and stride_elements=1; input and output are distinct")
                if ws % (blocks * (width or 1) * 4):
                    raise ValueError("Nonlinear working_set_bytes must contain equal complete block slices/rows")
                allocation = ws * 2 + width * 4 + blocks * threads * 8 + 256 * 1024
                if allocation > names["total_memory_bytes"] * 0.7:
                    raise ValueError("Nonlinear input/output/gamma and reference allocations exceed70% of device memory")
                resolved.update(working_set_bytes=ws, iterations=resolved.get("iterations", 1))
                if width: resolved["row_width"] = width
            if workload == "l1" and ws < blocks * 4:
                raise ValueError("L1 working set must include at least one 4byte word per block")
            if workload == "l2_latency" and ws // 4 > 2**32 - 1:
                raise ValueError("Latency probe exceeds its32bit node index range")
            if workload in ("l1", "l2", "l2_latency", "hbm"):
                multiplier = 1 if access == "read" else 2
                allocation = (ws + offset) * multiplier + blocks * threads * 4
                if allocation > names["total_memory_bytes"] * 0.7:
                    raise ValueError("Allocated working buffers including offset exceed70% of device memory")
            if workload == "gemm":
                m, n, k = (resolved.get(f"gemm_{axis}", 4096) for axis in ("m", "n", "k"))
                if min(m, n, k) <= 0 or max(m, n, k) > 2**31 - 1:
                    raise ValueError("GEMM dimensions must be positive signed32bit integers")
                if 2 * m * k + 2 * k * n + 4 * m * n > names["total_memory_bytes"] * 0.7:
                    raise ValueError("GEMM A/B/C allocations exceed70% of device memory")
            if workload == "hbm" and ws < names["l2_bytes"]*4:
                raise ValueError("HBM requested footprint must be at least4x discovered L2; verify physical traffic separately")
            if workload == "hbm":
                stride=resolved.get("stride_elements",1)
                word_count=ws//4
                # Upper bound for a complete stride cycle. Finite batches and SM
                # admission can touch less; physical DRAM counters remain required.
                reachable_sector_bytes=min(ws,(word_count//math.gcd(word_count,stride))*32)
                if reachable_sector_bytes < names["l2_bytes"]*4:
                    raise ValueError("HBM stride aliases a potentially cache-resident footprint; enlarge working_set_bytes")
            geometry_keys.setdefault(workload, set()).add(json.dumps({
                "stage": spec.get("stage", "saturation"), "parameters": resolved, "paired_reference": paired}, sort_keys=True))
            for pair in clocks:
                clock_annotation = clock_annotations[_clock_key(pair)]
                condition = {"workload": workload, "stage": spec.get("stage", "saturation"),
                             "parameters": resolved, "clocks": pair,
                             "clock_policy": clock_annotation["clock_policy"],
                             "clock_selection_reasons": clock_annotation["selection_reasons"],
                             "seconds": seconds, "warmup_seconds": warmup, "idle_seconds": idle}
                if paired:
                    condition["treatment_protocol"] = {
                        "kind": "paired_active_reference", "reference_workload": "control",
                        "reference_kind": "issue_loop",
                        "reference_matching": "coarse_unmatched_geometry" if workload == "gemm" else "launch_geometry_matched",
                        "order_strategy": "condition_seeded_alternation_by_repeat",
                        "note": "Reference is an operational treatment comparator, not an isolated static-power or component-energy measurement.",
                    }
                fingerprint={**condition,"benchmark_sha256":device.get("benchmark_sha256")}
                condition_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
                if condition_id in conditions:
                    raise ValueError("Duplicate resolved experiment condition; remove duplicate grid values/specifications")
                conditions.add(condition_id)
                first_reference = bool(int(condition_id, 16) % 2)
                for repeat in range(repeats):
                    trial = {**condition, "condition_id": condition_id, "repeat": repeat,
                             "trial_id": f"{condition_id}-r{repeat}"}
                    if paired:
                        reference_first = first_reference != bool(repeat % 2)
                        trial["treatment_protocol"] = {**condition["treatment_protocol"],
                            "order": "AB" if reference_first else "BA",
                            "phase_order": ["active_reference", "measure"] if reference_first else ["measure", "active_reference"],
                            "order_balance_note": "Odd repeat counts have a one-trial order imbalance; both orders are measured when repeats>=3."}
                    trials.append(trial)
    rng = random.Random(config.get("randomization_seed", 2026))
    rng.shuffle(trials)
    estimated_minimum = sum((2*seconds+3*warmup+2*idle)
        if trial.get("treatment_protocol", {}).get("kind") == "paired_active_reference"
        else seconds+warmup+2*idle for trial in trials)
    by_workload = [{"workload": workload, "geometry_conditions": len(keys),
                    "clock_conditions_per_geometry": len(clocks), "clock_geometry_conditions": len(keys)*len(clocks),
                    "repeats": repeats, "trials": len(keys)*len(clocks)*repeats}
                   for workload, keys in sorted(geometry_keys.items())]
    return {"schema_version":1, "device":device, "trials":trials,
            "study_design": _study_design(config), "execution_allowed": clock_coverage["execution_allowed"],
            "clock_sweep_coverage": clock_coverage,
            "clock_sweep_policy": {"graphics_step_mhz": clock_coverage.get("requested_step_mhz"),
                                   "graphics_min_mhz": clock_coverage.get("graphics_min_mhz", 900)},
            "sweep_dimensions": {"geometry_conditions": sum(len(keys) for keys in geometry_keys.values()),
                "clock_conditions_per_geometry": len(clocks), "clock_geometry_conditions": len(conditions),
                "repeats": repeats, "trials": len(trials), "by_workload": by_workload,
                "runtime_note": "Every workload/geometry is crossed with every deduplicated clock condition and repeated. Estimated runtime includes both paired arms and all planned repetitions, excluding preparation, settling and overruns."},
            "sample_interval_s":interval, "randomization_seed": config.get("randomization_seed",2026),
            "estimated_minimum_seconds": estimated_minimum,
            "notes":["Runtime estimate excludes allocation, preparation, clock settling and batch overruns.",
                     "Near/far labels require independent empirical locality evidence; offsets are exploratory."]}


def benchmark_command(executable, trial, device_index=0, profiling=False):
    command = [str(executable), "--device", str(device_index), "--workload", trial["workload"],
               "--seconds", str(0.1 if profiling else trial["seconds"]),
               "--warmup-seconds", str(0 if profiling else trial["warmup_seconds"]),
               "--idle-seconds", str(0 if profiling else trial["idle_seconds"])]
    for key, value in trial["parameters"].items():
        if profiling and key == "batch_launches": continue
        if key == "sm_ids" and isinstance(value, list): value = ",".join(str(v) for v in value)
        command += ["--"+key.replace("_", "-"), str(value)]
    if profiling:
        command += ["--batch-launches","1","--warmup-batches","1","--fixed-batches","1", "--profile-region"]
    if trial.get("treatment_protocol", {}).get("kind") == "paired_active_reference":
        order = trial["treatment_protocol"]["order"]
        if order not in ("AB", "BA"):
            raise ValueError("Paired treatment order must be AB or BA")
        # Under --profile-region CUDA retains the same paired allocations but
        # suppresses reference-arm execution. Counter evidence stays target-only.
        command += ["--paired-reference", "--reference-order", order]
    return command
