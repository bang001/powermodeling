"""Expand explicit, reproducible sweeps with a restricted numeric expression grammar."""

import ast
import hashlib
import itertools
import json
import math
import random

WORKLOADS = {"tensor", "gemm", "l1", "l2", "l2_latency", "hbm", "control"}
PARAMETERS = {"blocks", "threads", "working_set_bytes", "stride_elements", "iterations",
              "tensor_accumulators", "access", "sm_ids", "seed", "offset_bytes",
              "gemm_m", "gemm_n", "gemm_k", "batch_launches"}


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


def resolve_clocks(config, supported):
    if "clock_pairs" in config:
        pairs = config["clock_pairs"]
        if not pairs:
            raise ValueError("clock_pairs cannot be empty")
        normalized = []
        for pair in pairs:
            graphics, memory = pair.get("graphics_mhz"), pair.get("memory_mhz")
            if (graphics is None) != (memory is None):
                raise ValueError("Specify both graphics and memory clock, or neither")
            if graphics is not None and (type(graphics) is not int or graphics <= 0 or type(memory) is not int or memory <= 0):
                raise ValueError("Clock MHz values must be positive integers")
            normalized.append({"graphics_mhz": graphics, "memory_mhz": memory})
        if len({(p["graphics_mhz"], p["memory_mhz"]) for p in normalized}) != len(normalized):
            raise ValueError("clock_pairs contains duplicates")
        return normalized
    sweep = config.get("clock_sweep", {})
    raw = supported.get("supported_pairs", supported.get("pairs", []))
    by_mem = {}
    for pair in raw:
        memory = pair.get("memory_mhz", pair.get("memory_clock_mhz"))
        graphics = pair.get("graphics_mhz", pair.get("graphics_clock_mhz"))
        if memory is not None and graphics is not None:
            by_mem.setdefault(memory, []).append(graphics)
    memories = _quantiles(by_mem, sweep.get("memory_quantiles", [1.0]))
    return [{"graphics_mhz": g, "memory_mhz": m} for m in memories
            for g in _quantiles(by_mem[m], sweep.get("graphics_quantiles", [0.25, 0.6, 1.0]))]


def expand_plan(config, device, supported_clocks=None, stage=None):
    seconds = float(config.get("seconds", 12))
    warmup = float(config.get("warmup_seconds", 3))
    idle = float(config.get("idle_seconds", 6))
    repeats = config.get("repeats", 3)
    interval = float(config.get("sample_interval_s", 0.05))
    if (any(not math.isfinite(value) for value in (seconds, warmup, idle, interval))
            or type(repeats) is not int or seconds < 10 or warmup < 1 or idle < 6
            or repeats < 3 or not 0.005 <= interval <= 1):
        raise ValueError("Energy experiments require seconds>=10, warmup>=1, idle>=6, repeats>=3 and sample interval 0.005..1s")
    clocks = resolve_clocks(config, supported_clocks or {})
    names = {"sm_count": int(device["sm_count"]), "l2_bytes": int(device["l2_bytes"]),
             "total_memory_bytes": int(device["total_memory_bytes"])}
    if any(value <= 0 for value in names.values()):
        raise ValueError("Discovered device capacities and SM count must be positive")
    trials = []
    conditions = set()
    for spec in config["experiments"]:
        if stage and spec.get("stage", "saturation") != stage: continue
        workload = spec["workload"]
        if workload not in WORKLOADS: raise ValueError(f"Unknown workload {workload}")
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
                if workload == "gemm":
                    raise ValueError("cuBLAS GEMM cannot honor sm_ids")
                # SM IDs are sparse hardware identifiers on some SKUs. They
                # cannot be inferred merely from the enabled SM count.
                known_ids = device.get("discovered_sm_ids")
                if known_ids is not None and any(i not in known_ids for i in ids):
                    raise ValueError("sm_ids includes an ID absent from the discovered hardware map")
            if workload == "l1": default_ws = blocks * 16 * 1024
            elif workload in ("l2", "l2_latency"): default_ws = max(4, names["l2_bytes"] // 8 * 4)
            else: default_ws = max(512 * 1024 * 1024, names["l2_bytes"] * 8)
            ws = resolved.get("working_set_bytes", default_ws)
            offset = resolved.get("offset_bytes", 0)
            if ws < 4 or ws % 4 or offset % 4:
                raise ValueError("working_set_bytes and offset_bytes must be word-aligned (4bytes), with a nonempty working set")
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
            for pair in clocks:
                condition = {"workload": workload, "stage": spec.get("stage", "saturation"),
                             "parameters": resolved, "clocks": pair,
                             "seconds": seconds, "warmup_seconds": warmup, "idle_seconds": idle}
                fingerprint={**condition,"benchmark_sha256":device.get("benchmark_sha256")}
                condition_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
                if condition_id in conditions:
                    raise ValueError("Duplicate resolved experiment condition; remove duplicate grid values/specifications")
                conditions.add(condition_id)
                for repeat in range(repeats):
                    trials.append({**condition, "condition_id": condition_id, "repeat": repeat,
                                   "trial_id": f"{condition_id}-r{repeat}"})
    rng = random.Random(config.get("randomization_seed", 2026))
    rng.shuffle(trials)
    return {"schema_version":1, "device":device, "trials":trials,
            "sample_interval_s":interval, "randomization_seed": config.get("randomization_seed",2026),
            "estimated_minimum_seconds": sum(seconds+warmup+2*idle for _ in trials),
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
    return command
