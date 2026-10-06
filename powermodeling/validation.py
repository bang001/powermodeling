"""Conservative, numeric Nsight Compute admission policies.

Thresholds are explicit project policies, not NVIDIA architectural guarantees.
Missing/unsupported counters are unknown. Profiler replay is never energy data.
"""
from dataclasses import asdict, dataclass
import math
import statistics
from .nonlinear import NONLINEAR_WORKLOADS, ROW_WORKLOADS, count_issues


@dataclass(frozen=True)
class ValidationPolicy:
    l1_min_hit_pct: float = 95.0
    l2_min_hit_pct: float = 95.0
    bypass_max_l1_hit_pct: float = 5.0
    cache_max_downstream_bytes_per_logical_byte: float = 0.10
    hbm_max_l2_read_hit_pct: float = 20.0
    hbm_min_dram_bytes_per_logical_byte: float = 0.75
    hbm_min_dram_bytes_per_l2_byte: float = 0.75
    max_sector_inflation: float = 8.25
    min_sector_bytes_per_logical_byte: float = 0.90
    max_dram_bytes_per_l2_byte: float = 1.25
    max_clock_error_fraction: float = 0.03
    max_clock_drift_fraction: float = 0.03
    tensor_min_active_pct: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid validation policy {name}")
        for name in ("l1_min_hit_pct", "l2_min_hit_pct", "bypass_max_l1_hit_pct", "hbm_max_l2_read_hit_pct", "tensor_min_active_pct"):
            if getattr(self, name) > 100:
                raise ValueError(f"Percentage policy {name} must be in 0..100")
        if self.max_sector_inflation < self.min_sector_bytes_per_logical_byte:
            raise ValueError("Sector inflation maximum must exceed minimum")


L1_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
L1_HITS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"
L1_MISSES = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
L1_REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
L2_READ = "lts__t_sectors_srcunit_tex_op_read.sum"
L2_READ_HITS = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
L2_WRITE = "lts__t_sectors_srcunit_tex_op_write.sum"
LOCAL_LOAD = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"
LOCAL_STORE = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
DURATION = "gpu__time_duration.sum"
SM_HZ = "sm__cycles_elapsed.avg.per_second"
TENSOR_INSTRUCTIONS = ("sm__inst_executed_pipe_tensor.sum", "smsp__inst_executed_pipe_tensor.sum")
TENSOR_ACTIVITY = ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
SFU_INSTRUCTIONS = ("smsp__inst_executed_pipe_sfu.sum", "sm__inst_executed_pipe_sfu.sum")


def _number(value):
    if isinstance(value, bool): return None
    try: value = float(value)
    except (TypeError, ValueError): return None
    return value if math.isfinite(value) else None


def _policy(policy):
    return ValidationPolicy(**policy) if isinstance(policy, dict) else policy or ValidationPolicy()


def _check(checks, name, value, predicate, requirement):
    status = "inconclusive" if value is None else "pass" if predicate(value) else "fail"
    checks.append({"name": name, "status": status, "value": value, "requirement": requirement})
    return status


def _status(checks):
    return "fail" if any(c["status"] == "fail" for c in checks) else "inconclusive" if not checks or any(c["status"] == "inconclusive" for c in checks) else "pass"


def _ratio(a, b, scale=1):
    return a / b * scale if a is not None and b is not None and b > 0 else None


def _metrics(rows):
    # Parse raw values again; a hand-edited numeric_value cannot bypass checking.
    from .profiling import normalize_metric_value
    values, errors = {}, []
    for row in rows:
        value, unit, error = normalize_metric_value(row.get("value"), row.get("unit"))
        metric = row.get("metric")
        if metric in values:
            errors.append(f"duplicate_counter:{metric}")
            values[metric] = None
            continue
        expected = "sector" if metric in (L1_SECTORS, L1_HITS, L1_MISSES, L2_READ, L2_READ_HITS, L2_WRITE, LOCAL_LOAD, LOCAL_STORE) else "request" if metric == L1_REQUESTS else "inst" if metric in TENSOR_INSTRUCTIONS else "second" if metric == DURATION else "cycle/second" if metric == SM_HZ else "byte" if metric in (DRAM_READ, DRAM_WRITE) else "%" if str(metric).endswith((".pct", "pct_of_peak_sustained_active", "pct_of_peak_sustained_elapsed")) else None
        if expected and unit != expected:
            errors.append(f"counter_unit_mismatch:{metric}:{unit}")
            value = None
        if error: errors.append(f"{metric}:{error}")
        values[metric] = value
    return values, errors


def _kernel_assessment(rows, evidence, policy):
    metrics, errors = _metrics(rows)
    checks, derived = [], {}
    _check(checks, "kernel_launch_id", rows[0].get("id"), lambda v: str(v).isdigit(), "explicit numeric kernel launch ID; ratios are never averaged across dissimilar kernels")
    for metric, value in metrics.items():
        if value is not None:
            _check(checks, "nonnegative_" + str(metric), value, lambda v: v >= 0, "nonnegative hardware counters")
    for error in errors:
        checks.append({"name": error, "status": "inconclusive", "value": None, "requirement": "finite, unambiguous raw counter value"})
    for metric in (LOCAL_LOAD, LOCAL_STORE):
        _check(checks, metric, metrics.get(metric), lambda v: v == 0, "zero local-memory sectors; spilled/local traffic invalidates component isolation")
    duration = metrics.get(DURATION)
    _check(checks, "profile_kernel_duration_s", duration, lambda v: v > 0, "positive normalized kernel duration")
    workload = evidence.get("workload")
    tensor = workload in ("tensor", "gemm")
    if tensor:
        inst = next((metrics.get(m) for m in TENSOR_INSTRUCTIONS if metrics.get(m) is not None), None)
        activity = next((metrics.get(m) for m in TENSOR_ACTIVITY if metrics.get(m) is not None), None)
        path = inst if inst is not None else activity
        if workload == "tensor":
            _check(checks, "tensor_path_activity", path, lambda v: v > 0, "positive tensor instructions or active tensor pipeline cycles")
        if activity is not None:
            _check(checks, "tensor_active_pct", activity, lambda v: policy.tensor_min_active_pct <= v <= 100, f"{policy.tensor_min_active_pct}..100%; path admission alone does not prove peak utilization")
        derived.update(tensor_instructions=inst, tensor_active_pct=activity,
                       tensor_utilization_scope="elapsed" if metrics.get(TENSOR_ACTIVITY[0]) is not None else "active" if activity is not None else "unknown")
        if workload == "gemm": derived["component_scope"] = "GEMM includes memory and auxiliary work; tensor path verified, pure tensor energy not isolated"
    elif workload in NONLINEAR_WORKLOADS:
        result = evidence.get("profile_benchmark") or {}
        inst = next((metrics.get(m) for m in SFU_INSTRUCTIONS if metrics.get(m) is not None), None)
        _check(checks, "nonlinear_sfu_path", inst, lambda v: v > 0, "positive SFU pipeline instructions; implementation admission, not pure SFU energy")
        _check(checks, "single_nonlinear_profile_launch", _number(result.get("kernel_launches")), lambda v: v == 1, "exactly one measured nonlinear kernel")
        name = "row_nonlinear_kernel" if workload in ROW_WORKLOADS else "pointwise_nonlinear_kernel"
        _check(checks, "nonlinear_kernel_name", rows[0].get("kernel"), lambda v: name in v, "expected complete function implementation kernel")
        _check(checks, "nonlinear_profile_count_contract", count_issues(result, workload), lambda v: not v, "consistent exact element/row counts and sampled CPU-reference correctness")
        derived.update(sfu_instructions=inst, elements=result.get("elements"), row_width=result.get("row_width"),
                       component_scope="whole FP32 nonlinear function includes memory/reduction; SFU instructions do not define its energy denominator")
    elif workload in ("l1", "l2", "l2_latency", "hbm"):
        result = evidence.get("profile_benchmark") or {}
        logical = _number(result.get("logical_bytes"))
        launches = _number(result.get("kernel_launches"))
        if launches != 1:
            logical = None
        _check(checks, "single_profile_workload_launch", launches, lambda v: v == 1, "exactly one measured microkernel for logical/physical traffic comparison")
        _check(checks, "profile_logical_bytes", logical, lambda v: v > 0, "positive logical payload from profiled application result")
        access = result.get("access") or (evidence.get("profile_provenance") or {}).get("parameters", {}).get("access", "read")
        logical_read = logical / 2 if logical is not None and access == "copy" else logical if access == "read" else 0.0
        logical_write = logical / 2 if logical is not None and access == "copy" else logical if access == "write" else 0.0
        sectors = metrics.get(L1_SECTORS)
        hits = metrics.get(L1_HITS)
        if hits is None and sectors is not None and metrics.get(L1_MISSES) is not None:
            hits = sectors - metrics[L1_MISSES]
        l1_hit = _ratio(hits, sectors, 100)
        if sectors == 0 and hits == 0 and (metrics.get(L1_REQUESTS) or 0) > 0 and (metrics.get(L2_READ) or 0) > 0:
            l1_hit = 0.0
            derived["l1_bypass_evidence"] = "zero L1 lookup sectors/hits with positive global-load requests and downstream L2 read sectors"
        l2_read_bytes = metrics[L2_READ] * 32 if metrics.get(L2_READ) is not None else None
        l2_write_bytes = metrics[L2_WRITE] * 32 if metrics.get(L2_WRITE) is not None else None
        l2_hit = _ratio(metrics.get(L2_READ_HITS), metrics.get(L2_READ), 100)
        dram_read, dram_write = metrics.get(DRAM_READ), metrics.get(DRAM_WRITE)
        dram_total = dram_read + dram_write if dram_read is not None and dram_write is not None else None
        l2_total = l2_read_bytes + l2_write_bytes if l2_read_bytes is not None and l2_write_bytes is not None else None
        derived.update(logical_bytes=logical, logical_read_bytes=logical_read, logical_write_bytes=logical_write,
                       dram_read_bytes=dram_read, dram_write_bytes=dram_write, dram_bytes=dram_total,
                       l1_global_read_hit_pct=l1_hit, l2_tex_read_hit_pct=l2_hit,
                       l2_read_bytes=l2_read_bytes, l2_write_bytes=l2_write_bytes,
                       l1_sectors_per_global_read_request=_ratio(sectors, metrics.get(L1_REQUESTS)),
                       dram_bytes_per_logical_byte=_ratio(dram_total, logical),
                       dram_bytes_per_l2_byte=_ratio(dram_total, l2_total),
                       dram_throughput_bytes_s=_ratio(dram_total, duration),
                       l2_sector_inflation=_ratio(l2_total, logical),
                       l1_read_sector_inflation=_ratio(sectors * 32 if sectors is not None else None, logical_read))
        for name, value in (("l1_hit_counter_consistent", l1_hit), ("l2_hit_counter_consistent", l2_hit)):
            if value is not None: _check(checks, name, value, lambda v: 0 <= v <= 100, "hit counts within 0..100%; inconsistent replay counts rejected")
        if workload == "l1":
            _check(checks, "l1_read_hit_pct", l1_hit, lambda v: v >= policy.l1_min_hit_pct, f">={policy.l1_min_hit_pct}% operation-specific global read hits; broad L1 hit rate is diagnostic only")
            _check(checks, "l1_global_read_requests", metrics.get(L1_REQUESTS), lambda v: v > 0, "positive global-load requests")
            for name, value in (("l1_dram_leakage_ratio", _ratio(dram_total, logical)), ("l1_l2_leakage_ratio", _ratio(l2_total, logical))):
                _check(checks, name, value, lambda v: 0 <= v <= policy.cache_max_downstream_bytes_per_logical_byte, f"<={policy.cache_max_downstream_bytes_per_logical_byte} downstream bytes per logical byte")
            inflation = derived["l1_read_sector_inflation"]
        elif workload in ("l2", "l2_latency"):
            if access != "read":
                checks.append({"name": "l2_store_residency_unsupported", "status": "inconclusive", "value": access, "requirement": "read hit/miss policy does not establish write or copy L2 residency"})
            _check(checks, "l2_read_hit_pct", l2_hit, lambda v: v >= policy.l2_min_hit_pct, f">={policy.l2_min_hit_pct}% TEX-origin read sector hits")
            _check(checks, "l1_bypass_hit_pct", l1_hit, lambda v: v <= policy.bypass_max_l1_hit_pct, f"<={policy.bypass_max_l1_hit_pct}% global read L1 hits")
            _check(checks, "l2_dram_leakage_ratio", _ratio(dram_total, logical), lambda v: 0 <= v <= policy.cache_max_downstream_bytes_per_logical_byte, f"<={policy.cache_max_downstream_bytes_per_logical_byte} DRAM bytes per logical byte")
            inflation = derived["l2_sector_inflation"]
        else:
            if access not in ("read", "write", "copy"):
                checks.append({"name": "hbm_access", "status": "inconclusive", "value": access, "requirement": "read, write or copy"})
            if access in ("read", "copy"):
                _check(checks, "l1_bypass_hit_pct", l1_hit, lambda v: v <= policy.bypass_max_l1_hit_pct, f"<={policy.bypass_max_l1_hit_pct}% global read L1 hits")
                _check(checks, "hbm_l2_read_hit_pct", l2_hit, lambda v: v <= policy.hbm_max_l2_read_hit_pct, f"<={policy.hbm_max_l2_read_hit_pct}% L2 read hits")
                _check(checks, "hbm_read_payload_ratio", _ratio(dram_read, logical_read), lambda v: v >= policy.hbm_min_dram_bytes_per_logical_byte, f">={policy.hbm_min_dram_bytes_per_logical_byte} physical DRAM read bytes per logical read byte")
            if access in ("write", "copy"):
                _check(checks, "hbm_write_payload_ratio", _ratio(dram_write, logical_write), lambda v: v >= policy.hbm_min_dram_bytes_per_logical_byte, f">={policy.hbm_min_dram_bytes_per_logical_byte} physical DRAM write bytes per logical write byte; short unflushed writes may fail")
            _check(checks, "hbm_physical_dram_dominance", derived["dram_bytes_per_l2_byte"], lambda v: policy.hbm_min_dram_bytes_per_l2_byte <= v <= policy.max_dram_bytes_per_l2_byte, f"{policy.hbm_min_dram_bytes_per_l2_byte}..{policy.max_dram_bytes_per_l2_byte} DRAM/L2-requested byte ratio")
            if access in ("read", "write"):
                wrong_direction = dram_write if access == "read" else dram_read
                _check(checks, "hbm_unrequested_direction_ratio", _ratio(wrong_direction, logical), lambda v: v <= policy.cache_max_downstream_bytes_per_logical_byte, f"<={policy.cache_max_downstream_bytes_per_logical_byte} unexpected-direction DRAM bytes per logical byte")
            _check(checks, "hbm_physical_throughput", derived["dram_throughput_bytes_s"], lambda v: v > 0, "positive physical DRAM byte/s; saturation is determined by throughput sweep, not this admission check")
            inflation = derived["l2_sector_inflation"]
        # Pointer chasing requests one 4B value per 32B sector by design.
        minimum = policy.min_sector_bytes_per_logical_byte
        _check(checks, "sector_inflation", inflation, lambda v: minimum <= v <= policy.max_sector_inflation, f"{minimum}..{policy.max_sector_inflation} sector bytes/logical bytes; record coalescing costs explicitly")
    else:
        checks.append({"name": "workload_target", "status": "inconclusive", "value": workload, "requirement": "tensor, gemm, l1, l2, l2_latency or hbm target"})
    return {"id": rows[0].get("id"), "kernel": rows[0].get("kernel"), "status": _status(checks), "checks": checks, "metrics": metrics, "derived": derived}


def assess_profile(evidence, policy=None):
    """Recompute target admission from raw rows; ignore hand-written flags."""
    policy = _policy(policy if policy is not None else evidence.get("validation_policy"))
    groups = {}
    for row in evidence.get("rows") or []:
        if isinstance(row, dict): groups.setdefault((row.get("id"), row.get("kernel")), []).append(row)
    kernels = [_kernel_assessment(rows, evidence, policy) for rows in groups.values()]
    checks = [check for kernel in kernels for check in kernel["checks"]]
    if evidence.get("workload") == "gemm":
        paths = [k["derived"].get("tensor_instructions") if k["derived"].get("tensor_instructions") is not None else k["derived"].get("tensor_active_pct") for k in kernels]
        paths = [v for v in paths if v is not None]
        _check(checks, "gemm_contains_tensor_arithmetic", max(paths) if paths else None, lambda v: v > 0, "at least one measured cuBLAS kernel uses tensor instructions/activity; auxiliary kernels retain their own spill/duration checks")
    if not kernels: checks = [{"name": "counter_rows", "status": "inconclusive", "value": None, "requirement": "nonempty raw numeric Nsight counter evidence"}]
    duration = sum(k["metrics"].get(DURATION) or 0 for k in kernels)
    rate_summary = {"scope": "summed profiled-kernel counters/summed kernel duration; profiler busy time differs from energy experiment wall time", "kernel_count": len(kernels), "kernel_duration_s": duration or None}
    for key, metric, scale in (("dram_read_bytes_s", DRAM_READ, 1), ("dram_write_bytes_s", DRAM_WRITE, 1), ("l1_read_sector_bytes_s", L1_SECTORS, 32), ("l2_read_sector_bytes_s", L2_READ, 32), ("l2_write_sector_bytes_s", L2_WRITE, 32)):
        values = [k["metrics"].get(metric) for k in kernels]
        rate_summary[key] = sum(values) * scale / duration if values and all(v is not None for v in values) and duration > 0 else None
    rate_summary["tensor_activity_by_kernel"] = [{"id": k["id"], "kernel": k["kernel"], "active_pct": k["derived"].get("tensor_active_pct"), "scope": k["derived"].get("tensor_utilization_scope"), "instructions": k["derived"].get("tensor_instructions")} for k in kernels]
    return {"status": _status(checks), "checks": checks, "kernels": kernels, "rates_summary": rate_summary, "policy": asdict(policy),
            "reasons": [c["name"] + ": " + c["requirement"] for c in checks if c["status"] != "pass"],
            "scope": "path/residency admission for whole-device incremental-energy experiments; does not isolate rail or block energy"}


def _clock_values(evidence, assessment, field):
    context = evidence.get("profile_context") or {}
    samples = context.get("profile_active_nvml_samples") or []
    values = [_number(sample.get(field)) for sample in samples if isinstance(sample, dict)]
    values = [v for v in values if v is not None and v > 0]
    if field == "graphics_clock_mhz":
        frequencies = [_number(k["metrics"].get(SM_HZ)) for k in assessment["kernels"]]
        frequencies = [v / 1e6 for v in frequencies if v is not None and v > 0]
        if frequencies: values = frequencies
    return values


def validate_evidence(record, evidence, policy=None):
    """Bind raw counter admission to exact binary/GPU/parameters/actual clocks."""
    if not isinstance(evidence, dict): evidence = {}
    policy = _policy(policy if policy is not None else evidence.get("validation_policy"))
    result = assess_profile(evidence, policy)
    checks = result["checks"]
    config = record.get("config") or {}
    provenance = evidence.get("profile_provenance") or {}
    context = evidence.get("profile_context") or {}
    _check(checks, "profile_session_status", evidence.get("profile_session_status"), lambda v: v == "complete", "successfully completed dedicated profiling session")
    clock_control = evidence.get("clock_control") or context.get("clock_control") or {}
    _check(checks, "profile_clock_policy_applied", clock_control.get("applied"), lambda v: v is True, "owned controlled clock policy successfully applied/read back")
    _check(checks, "profile_clock_policy_restored", clock_control.get("restored"), lambda v: v is True and not clock_control.get("restore_errors"), "exact previous clock policy restored without error")
    _check(checks, "profile_nvml_gpu_uuid", context.get("gpu_uuid"), lambda v: v == config.get("gpu_uuid"), "profiler NVML device UUID matches measured CUDA GPU")
    worker_ids = [d.get("process_id") for d in provenance.get("observed_device_records") or [] if isinstance(d, dict)]
    worker_ids = [pid for pid in worker_ids if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0]
    observed_worker_uuids = [d.get("uuid") for d in provenance.get("observed_device_records") or [] if isinstance(d, dict)]
    _check(checks, "profile_worker_gpu_uuids", observed_worker_uuids if observed_worker_uuids else None, lambda v: all(uuid == config.get("gpu_uuid") for uuid in v), "every replay worker reports the measured GPU UUID")
    _check(checks, "profile_worker_process_ids", len(worker_ids) if worker_ids else None, lambda v: v > 0, "actual CUDA worker PIDs emitted by each application replay pass")
    profile_samples = context.get("profile_active_nvml_samples") or []
    _check(checks, "profile_active_inventory_samples", len(profile_samples) if profile_samples else None, lambda v: v > 0, "NVML process inventories sampled during profiled measured regions")
    from .runner import _sample_exclusivity
    inventory = _sample_exclusivity(profile_samples, worker_ids)
    _check(checks, "profile_process_inventory_available", None if inventory["process_inventory_unknown"] else True, bool, "compute and graphics process inventories available; unsupported MPS inventory is optional")
    _check(checks, "profile_process_exclusive", len(inventory["other_compute_processes"]) if worker_ids else None, lambda v: v == 0, "no foreign compute, graphics or MPS process during profiling")
    result["profile_exclusivity"] = inventory
    _check(checks, "deterministic_application_replay", provenance.get("deterministic_application_replay"), lambda v: v is True, "identical counter-relevant workload metadata/payload across application replay passes")
    _check(checks, "profile_application_measured_region", (evidence.get("profile_benchmark") or {}).get("profile_region"), lambda v: v is True, "cudaProfilerStart/Stop encloses measured workload only")
    identity = (("condition_id", evidence.get("condition_id"), record.get("condition_id")),
                ("workload", evidence.get("workload"), record.get("workload")),
                ("gpu_uuid", provenance.get("observed_gpu_uuid"), config.get("gpu_uuid")),
                ("benchmark_sha256", provenance.get("benchmark_sha256"), (record.get("provenance") or {}).get("benchmark_sha256")))
    for name, actual, expected in identity:
        _check(checks, "matching_" + name, actual if actual and expected else None, lambda v, e=expected: v == e, "exact profiled/measured " + name)
    parameters = provenance.get("parameters")
    if not isinstance(parameters, dict):
        _check(checks, "profile_parameters", None, bool, "exact requested parameter dictionary")
    else:
        for name, expected in parameters.items():
            _check(checks, "matching_parameter_" + name, config.get(name), lambda v, e=expected: v == e, "exact requested workload parameter")
    benchmark = evidence.get("profile_benchmark") or {}
    measured_benchmark = record.get("benchmark") or {}
    effective_names = ("blocks", "threads", "iterations_per_launch", "working_set_bytes", "stride_elements", "offset_bytes", "tensor_accumulators", "gemm_m", "gemm_n", "gemm_k", "access", "l1_bytes_per_block", "paired_reference_context_allocated")
    if record.get("workload") in NONLINEAR_WORKLOADS:
        effective_names += ("row_width", "math_implementation", "input_precision", "nonlinear_input_distribution")
        if record.get("workload") == "rmsnorm": effective_names += ("rms_epsilon", "affine_gamma")
    for name in effective_names:
        if name in benchmark or name in measured_benchmark:
            _check(checks, "matching_effective_" + name, benchmark.get(name), lambda v, e=measured_benchmark.get(name): e is not None and v == e, "exact effective profile/energy workload parameter")
    clocks = provenance.get("requested_clocks") or {}
    actual_clocks = {}
    required_effective = ("blocks", "threads", "iterations_per_launch", "tensor_accumulators") if record.get("workload") == "tensor" else ("gemm_m", "gemm_n", "gemm_k", "iterations_per_launch") if record.get("workload") == "gemm" else ("blocks", "threads", "iterations_per_launch", "working_set_bytes", "stride_elements", "offset_bytes", "access")
    if record.get("workload") in NONLINEAR_WORKLOADS:
        required_effective += ("row_width", "math_implementation", "input_precision", "nonlinear_input_distribution")
    for name in required_effective:
        if name not in benchmark or name not in measured_benchmark:
            _check(checks, "missing_effective_" + name, None, bool, "effective profile and energy kernel parameters required")
    active_samples = [s for s in record.get("samples") or [] if isinstance(s, dict) and s.get("phase") == "measure"]
    # Runner samples are phase-associated through explicit monotonic intervals.
    if not active_samples:
        phases = record.get("phases") or []
        if isinstance(phases, dict):
            phase = phases.get("measure") or {}
            intervals = [(phase.get("start_s"), phase.get("end_s"))]
        else:
            intervals = [(p.get("start_s"), p.get("end_s")) for p in phases if p.get("name") == "measure"] if isinstance(phases, list) else []
        active_samples = [s for s in record.get("samples") or [] if any(a is not None and b is not None and a <= s.get("t_s", -1) <= b for a, b in intervals)]
    for field, key in (("graphics_clock_mhz", "graphics_mhz"), ("memory_clock_mhz", "memory_mhz")):
        requested = _number(config.get(field))
        profile_requested = _number(clocks.get(key))
        _check(checks, "matching_requested_" + field, profile_requested if requested is not None else None, lambda v, e=requested: v == e and v > 0, "matching explicit positive controlled frequency; uncontrolled DVFS profiles remain unverified")
        values = _clock_values(evidence, result, field)
        median = statistics.median(values) if values else None
        actual_clocks[field] = {"median": median, "minimum": min(values) if values else None, "maximum": max(values) if values else None, "samples": len(values)}
        _check(checks, "profile_actual_" + field, median if requested is not None else None, lambda v, e=requested: e is not None and abs(v - e) / e <= policy.max_clock_error_fraction, f"actual profiled frequency within {policy.max_clock_error_fraction:.1%} of requested frequency")
        spread = (max(values) - min(values)) / median if values and median else None
        _check(checks, "profile_stability_" + field, spread, lambda v: v <= policy.max_clock_drift_fraction, f"profile frequency spread <= {policy.max_clock_drift_fraction:.1%}")
        measured = [_number(s.get(field)) for s in active_samples]
        measured = [v for v in measured if v is not None and v > 0]
        measured_median = statistics.median(measured) if measured else None
        _check(checks, "matched_actual_" + field, measured_median if median is not None else None, lambda v, e=median: e is not None and abs(v - e) / e <= policy.max_clock_error_fraction, f"actual energy/profile frequency agreement within {policy.max_clock_error_fraction:.1%}")
    result.update(status=_status(checks), suitable_verified=_status(checks) == "pass", profile_actual_clocks=actual_clocks,
                  reasons=[c["name"] + ": " + c["requirement"] for c in checks if c["status"] != "pass"])
    return result
