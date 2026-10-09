"""Conservative, numeric Nsight Compute admission policies.

Thresholds are explicit project policies, not NVIDIA architectural guarantees.
Missing/unsupported counters are unknown. Profiler replay is never energy data.
"""
from dataclasses import asdict, dataclass
import math
import statistics
import re
from .nonlinear import NONLINEAR_WORKLOADS, ROW_WORKLOADS, Q_GRID_MATH_IMPLEMENTATION, Q_GRID_IMPLEMENTATION_VERSION, count_issues
from .memory import (VERSIONS as MEMORY_IMPLEMENTATION_VERSIONS,
                     coalesced_read_geometry, count_issues as memory_count_issues)
from .sfu import SFU_WORKLOADS, CONTRACT_FIELDS as SFU_CONTRACT_FIELDS, count_issues as sfu_count_issues


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
    coalesced_read_counter_ratio_tolerance: float = 0.05
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
        if self.coalesced_read_counter_ratio_tolerance >= 1:
            raise ValueError("Coalesced-read counter ratio tolerance must be below1")


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
SFU_ACTIVITY = ("sm__pipe_sfu_cycles_active.avg.pct_of_peak_sustained_elapsed",
                "smsp__pipe_sfu_cycles_active.avg.pct_of_peak_sustained_elapsed",
                "sm__pipe_sfu_cycles_active.avg.pct_of_peak_sustained_active",
                "smsp__pipe_sfu_cycles_active.avg.pct_of_peak_sustained_active")


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


def _traffic_amplification(derived):
    """Compare replay traffic with that same launch's logical payload only."""
    fields = {key: derived.get(source) for key, source in (
        ("logical_bytes", "logical_bytes"), ("logical_read_bytes", "logical_read_bytes"),
        ("logical_write_bytes", "logical_write_bytes"),
        ("l1_read_sector_bytes", "l1_read_sector_bytes"),
        ("l2_read_sector_bytes", "l2_read_bytes"), ("l2_write_sector_bytes", "l2_write_bytes"),
        ("l2_sector_bytes", "l2_sector_bytes"),
        ("dram_read_bytes", "dram_read_bytes"), ("dram_write_bytes", "dram_write_bytes"),
        ("dram_bytes", "dram_bytes"))}
    for key, numerator, denominator in (
        ("l1_read_sector_bytes_per_logical_read_byte", "l1_read_sector_bytes", "logical_read_bytes"),
        ("l2_read_sector_bytes_per_logical_read_byte", "l2_read_sector_bytes", "logical_read_bytes"),
        ("l2_write_sector_bytes_per_logical_write_byte", "l2_write_sector_bytes", "logical_write_bytes"),
        ("l2_sector_bytes_per_logical_byte", "l2_sector_bytes", "logical_bytes"),
        ("dram_read_bytes_per_logical_read_byte", "dram_read_bytes", "logical_read_bytes"),
        ("dram_write_bytes_per_logical_write_byte", "dram_write_bytes", "logical_write_bytes"),
        ("dram_bytes_per_logical_byte", "dram_bytes", "logical_bytes")):
        fields[key] = _ratio(fields[numerator], fields[denominator])
    return fields


def _memory_coalescing(evidence, traffic, policy, target_path_status):
    """Keep energy-read coalescing distinct from broad path admission.

    L2 TEX-origin sectors are downstream of the .cg load path. Their byte ratio
    is a qualified replay diagnostic, not a frontend coalescing or HBM bandwidth
    measurement. No replay-derived efficiency changes an energy denominator.
    """
    workload = evidence.get("workload")
    geometry = coalesced_read_geometry(evidence.get("profile_benchmark") or {}, workload,
                                      (evidence.get("profile_provenance") or {}).get("parameters"))
    applicable = geometry["applicable"]
    counter = L1_SECTORS if workload == "l1" else L2_READ if workload in ("l2", "hbm") else None
    ratio_key = "l1_read_sector_bytes_per_logical_read_byte" if workload == "l1" else "l2_read_sector_bytes_per_logical_read_byte"
    bound = traffic.get("logical_payload_binding") == "one_observed_and_reported_microkernel"
    ratio = _number(traffic.get(ratio_key)) if applicable and bound else None
    efficiency = 1 / ratio if ratio is not None and ratio > 0 else None
    tolerance = policy.coalesced_read_counter_ratio_tolerance
    bounds = [1 - tolerance, 1 + tolerance]
    failures = list(geometry["reasons"]) if geometry["status"] == "fail" else []
    unknown = list(geometry["reasons"]) if geometry["status"] == "inconclusive" else []
    if applicable:
        if not bound:
            unknown.append("coalescing_profile_payload_unbound")
        if ratio is None:
            unknown.append("coalescing_sector_ratio_unknown")
        elif not bounds[0] <= ratio <= bounds[1]:
            failures.append("coalescing_sector_ratio_outside_unit_tolerance")
    status = "fail" if failures else "inconclusive" if unknown or not applicable else "pass"
    return {"applicable": applicable, "status": status, "energy_eligible": False,
            "reasons": sorted(set(failures + unknown)), "geometry": geometry,
            "expected_sector_bytes_per_logical_read_byte": 1.0 if applicable else None,
            "expected_sector_efficiency_fraction": 1.0 if applicable else None,
            "requested_sector_efficiency_fraction": geometry["requested_sector_efficiency_fraction"],
            "observed_sector_bytes_per_logical_read_byte": ratio,
            "observed_sector_efficiency_fraction": efficiency,
            "observed_sector_efficiency_pct": efficiency * 100 if efficiency is not None else None,
            "counter_name": counter if applicable else None,
            "counter_scope": "L1 global-read lookup sectors from one replay launch" if workload == "l1" else
                             "TEX-origin L2 read sectors downstream of the .cg path; not frontend request efficiency or measured DRAM traffic" if workload in ("l2", "hbm") else "not_applicable",
            "logical_payload_binding": traffic.get("logical_payload_binding"),
            "relative_counter_ratio_tolerance": tolerance, "counter_ratio_bounds": bounds,
            "target_path_status": target_path_status, "profile_binding_verified": None,
            "energy_denominator_use": "forbidden_separate_profiler_run",
            "interpretation": "Requested scalar read geometry and replay sector ratio qualify coalesced energy candidates; target path admission, saturation and physical memory energy remain separate. Efficiencies are not clipped at100%."}


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
        expected = "sector" if metric in (L1_SECTORS, L1_HITS, L1_MISSES, L2_READ, L2_READ_HITS, L2_WRITE, LOCAL_LOAD, LOCAL_STORE) else "request" if metric == L1_REQUESTS else "inst" if metric in TENSOR_INSTRUCTIONS + SFU_INSTRUCTIONS else "second" if metric == DURATION else "cycle/second" if metric == SM_HZ else "byte" if metric in (DRAM_READ, DRAM_WRITE) else "%" if str(metric).endswith((".pct", "pct_of_peak_sustained_active", "pct_of_peak_sustained_elapsed")) else None
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
    elif workload in SFU_WORKLOADS:
        from .sfu_sass import validate_certificate
        result = evidence.get("profile_benchmark") or {}
        count_problems = sfu_count_issues(result, workload)
        _check(checks, "sfu_profile_count_contract", count_problems, lambda v: not v,
               "exact completed scalar-lane SFU instruction counts and one-step primitive correctness")
        _check(checks, "single_sfu_profile_launch", _number(result.get("kernel_launches")), lambda v: v == 1,
               "exactly one measured register SFU target kernel")
        kernel_name = rows[0].get("kernel") or ""
        specialization = re.search(r"sfu_register_kernel\s*<\s*(\d+)\s*,\s*(false|true|0|1)\s*,\s*(\d+)\s*>", kernel_name)
        if not specialization:
            specialization = re.search(r"sfu_register_kernelILi(\d+)ELb([01])ELi(\d+)E", kernel_name)
        op_index = {"sfu_ex2": 0, "sfu_tanh": 1, "sfu_rsqrt": 2, "sfu_rcp": 3, "sfu_lg2": 4, "sfu_sqrt": 5}[workload]
        identity = [int(specialization[1]), specialization[2] in ("true", "1"), int(specialization[3])] if specialization else None
        _check(checks, "sfu_kernel_specialization", identity,
               lambda v: v == [op_index, False, result.get("sfu_chains")],
               "native opcode, treatment specialization and independent chain count match")
        expected_warp_instructions = None
        if not count_problems:
            expected_warp_instructions = (result["kernel_launches"] * math.ceil(result["sfu_lanes"] / 32)
                                          * result["iterations_per_launch"] * result["sfu_chains"])
        instruction_counts = [metrics[name] for name in SFU_INSTRUCTIONS if metrics.get(name) is not None]
        _check(checks, "sfu_warp_instruction_count", instruction_counts if instruction_counts and expected_warp_instructions is not None else None,
               lambda values: all(math.isclose(v, expected_warp_instructions, rel_tol=1e-6, abs_tol=1e-6) for v in values),
               "SFU warp instructions equal launches × ceil(active lanes/32) × iterations × chains; never use this warp count as the scalar energy denominator")
        provenance = evidence.get("profile_provenance") or {}
        devices = provenance.get("observed_device_records") or []
        capabilities = []
        for device in devices:
            cc = device.get("cc")
            if cc is None and type(device.get("compute_capability_major")) is int and type(device.get("compute_capability_minor", 0)) is int:
                cc = str(device["compute_capability_major"]) + "." + str(device.get("compute_capability_minor", 0))
            capabilities.append(cc)
        certificate = evidence.get("sfu_sass_evidence")
        certificate_issues = None
        if certificate is not None and capabilities and all(cc is not None for cc in capabilities):
            certificate_issues = sorted({issue for cc in capabilities for issue in validate_certificate(
                certificate, provenance.get("benchmark_sha256"), cc, workload, result.get("sfu_chains"))})
        _check(checks, "sfu_register_sass_contract", certificate_issues, lambda v: not v,
               "recomputed binary/architecture/opcode/chain-bound SASS proves retained native instructions, zero hot-loop memory, no spills and target-free control")
        activity_counter = next((name for name in SFU_ACTIVITY if metrics.get(name) is not None), None)
        derived.update(sfu_instructions=instruction_counts[0] if instruction_counts else None,
                       sfu_instruction_counter_unit="warp instructions, not scalar lane operations",
                       expected_sfu_warp_instructions=expected_warp_instructions,
                       scalar_sfu_instructions=result.get("sfu_instructions"),
                       sfu_active_pct=metrics.get(activity_counter), sfu_activity_counter=activity_counter,
                       sfu_activity_scope="elapsed" if activity_counter and activity_counter.endswith("_elapsed") else "active" if activity_counter else "unknown",
                       component_scope="register-resident native SFU primitive; paired board-power contrast is not isolated physical SFU rail energy")
    elif workload in NONLINEAR_WORKLOADS:
        result = evidence.get("profile_benchmark") or {}
        inst = next((metrics.get(m) for m in SFU_INSTRUCTIONS if metrics.get(m) is not None), None)
        _check(checks, "nonlinear_sfu_path", inst, lambda v: v > 0, "positive SFU pipeline instructions; implementation admission, not pure SFU energy")
        _check(checks, "single_nonlinear_profile_launch", _number(result.get("kernel_launches")), lambda v: v == 1, "exactly one measured nonlinear kernel")
        name = "row_nonlinear_kernel" if workload in ROW_WORKLOADS else "pointwise_nonlinear_kernel"
        _check(checks, "nonlinear_kernel_name", rows[0].get("kernel"), lambda v: name in v, "expected complete function implementation kernel")
        _check(checks, "nonlinear_profile_count_contract", count_issues(result, workload), lambda v: not v, "consistent exact element/row counts and sampled CPU-reference correctness")
        activity_counter = next((name for name in SFU_ACTIVITY if metrics.get(name) is not None), None)
        derived.update(sfu_instructions=inst,
                       sfu_active_pct=metrics.get(activity_counter), sfu_activity_counter=activity_counter,
                       sfu_activity_scope="elapsed" if activity_counter and activity_counter.endswith("_elapsed") else "active" if activity_counter else "unknown",
                       sfu_activity_interpretation="optional advertised profiler pipeline activity; separate replay run, not a pure-function energy denominator or proof of hardware saturation",
                       elements=result.get("elements"), row_width=result.get("row_width"),
                       component_scope="whole FP32 nonlinear function includes memory/reduction; SFU instructions do not define its energy denominator")
    elif workload in ("l1", "l2", "l2_latency", "hbm"):
        result = evidence.get("profile_benchmark") or {}
        if result.get("kernel_implementation_version") in MEMORY_IMPLEMENTATION_VERSIONS:
            _check(checks, "profile_memory_count_contract", memory_count_issues(result, workload), lambda v: not v,
                   "versioned scalar memory operations and bytes match admission/thread/iteration counts")
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
                       l1_read_sector_bytes=sectors * 32 if sectors is not None else None,
                       l2_sector_bytes=l2_total,
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
        derived["traffic_amplification"] = _traffic_amplification(derived)
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
    if evidence.get("workload") in {"l1", "l2", "l2_latency", "hbm"} | NONLINEAR_WORKLOADS | SFU_WORKLOADS:
        # Raw launch IDs bind the one profiled custom launch to its payload.
        # An extra otherwise-valid kernel must not silently reuse that payload.
        ids = [int(str(kernel["id"])) for kernel in kernels if str(kernel.get("id")).isdecimal()]
        ids_known = bool(kernels) and len(ids) == len(kernels)
        prefix = "sfu" if evidence.get("workload") in SFU_WORKLOADS else "nonlinear" if evidence.get("workload") in NONLINEAR_WORKLOADS else "memory"
        _check(checks, prefix + "_profile_launch_id_unique_mapping",
               len(ids) == len(set(ids)) if ids_known else None, bool,
               "each numeric profile launch ID maps to exactly one kernel name")
        reported = _number((evidence.get("profile_benchmark") or {}).get("kernel_launches"))
        reported = reported if reported is not None and reported > 0 and reported.is_integer() else None
        _check(checks, prefix + "_profile_observed_launch_count",
               len(set(ids)) if ids_known and reported is not None else None,
               lambda value: value == reported,
               "number of distinct observed kernel launch IDs matches reported positive integer kernel_launches")
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
    rate_summary["sfu_activity_by_kernel"] = [{"id": k["id"], "kernel": k["kernel"],
        "active_pct": k["derived"].get("sfu_active_pct"), "counter": k["derived"].get("sfu_activity_counter"),
        "scope": k["derived"].get("sfu_activity_scope", "unknown"), "instructions": k["derived"].get("sfu_instructions")} for k in kernels if "sfu_instructions" in k["derived"]]
    traffic_kernels = [kernel for kernel in kernels if "traffic_amplification" in kernel["derived"]]
    # The application payload is attributable only to one observed microkernel.
    # Multiple observed kernels with one reported launch cannot reuse that total.
    bound = (len(kernels) == len(traffic_kernels) == 1
             and str(traffic_kernels[0].get("id")).isdecimal()
             and _number((evidence.get("profile_benchmark") or {}).get("kernel_launches")) == 1
             and (traffic_kernels[0]["derived"].get("logical_bytes") or 0) > 0)
    traffic = {**(traffic_kernels[0]["derived"]["traffic_amplification"] if bound else _traffic_amplification({})),
               "scope": "same profiler replay launch counters/logical payload; separate from the energy experiment",
               "logical_payload_binding": "one_observed_and_reported_microkernel" if bound else "unknown_missing_or_multiple_profile_kernels",
               "energy_denominator_use": "forbidden_separate_profiler_run",
               "interpretation": "A ratio above one can reflect sector overfetch/coalescing or other traffic. Path admission does not imply unit traffic amplification or saturation.",
               "by_kernel": [{"id": kernel["id"], "kernel": kernel["kernel"], **kernel["derived"]["traffic_amplification"]} for kernel in traffic_kernels]}
    rate_summary["traffic_amplification"] = traffic
    return {"status": _status(checks), "checks": checks, "kernels": kernels, "rates_summary": rate_summary,
            "traffic_amplification": traffic, "policy": asdict(policy),
            "memory_coalescing": _memory_coalescing(evidence, traffic, policy, _status(checks)),
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
    path_check_count = len(checks)
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
    effective_names = ("blocks", "threads", "iterations_per_launch", "working_set_bytes", "stride_elements", "offset_bytes", "tensor_accumulators", "gemm_m", "gemm_n", "gemm_k", "access", "l1_bytes_per_block", "paired_reference_context_allocated", "kernel_implementation_version", "memory_accesses_per_thread_iteration")
    if record.get("workload") in ("l1", "l2", "l2_latency", "hbm"):
        effective_names += ("memory_word_bytes", "stride_words", "lane_stride_bytes")
    scalar_read = record.get("workload") in ("l1", "l2", "hbm") and measured_benchmark.get("access", config.get("access", "read")) == "read"
    if scalar_read:
        effective_names += ("read_cache_policy", "memory_read_index_math")
    if record.get("workload") in NONLINEAR_WORKLOADS:
        effective_names += ("row_width", "math_implementation", "input_precision", "nonlinear_input_distribution",
                            "grid_mode", "input_elements", "block_completion_count_source")
        if record.get("workload") == "rmsnorm": effective_names += ("rms_epsilon", "affine_gamma")
    if record.get("workload") in SFU_WORKLOADS:
        effective_names += SFU_CONTRACT_FIELDS
    for name in effective_names:
        if record.get("workload") in SFU_WORKLOADS and name == "sfu_exponent_base" and name in benchmark and name in measured_benchmark and benchmark[name] is None and measured_benchmark[name] is None:
            # Non-exponential native primitives explicitly have no exponent base;
            # the SFU count/math contract independently checks this definition.
            continue
        if name in ("kernel_implementation_version", "memory_accesses_per_thread_iteration") and benchmark.get(name) is None and measured_benchmark.get(name) is None:
            continue
        if name in benchmark or name in measured_benchmark:
            _check(checks, "matching_effective_" + name, benchmark.get(name), lambda v, e=measured_benchmark.get(name): e is not None and v == e, "exact effective profile/energy workload parameter")
    if scalar_read:
        for name, allowed in (("read_cache_policy", {"ca"} if record["workload"] == "l1" else {"cg"} if record["workload"] == "l2" else {"ca", "cg", "cs"}),
                              ("memory_read_index_math", {"uint32", "uint64"})):
            if name in benchmark or name in measured_benchmark:
                _check(checks, "valid_effective_" + name, benchmark.get(name), lambda v, a=allowed: isinstance(v, str) and v in a, "canonical scalar-read implementation metadata")
        requested_cache = config.get("read_cache_policy")
        if requested_cache is not None:
            expected_cache = ("ca" if record["workload"] == "l1" else "cg") if requested_cache == "auto" else requested_cache
            actual_cache = benchmark.get("read_cache_policy")
            if ("read_cache_policy" not in benchmark and "read_cache_policy" not in measured_benchmark
                    and benchmark.get("kernel_implementation_version") in (None, "scalar_single_stream_read_v2")
                    and measured_benchmark.get("kernel_implementation_version") in (None, "scalar_single_stream_read_v2")):
                # Old scalar-read binaries have fixed .ca/.cg modifiers and no
                # metadata field. Preserve that known default, never invent a
                # legacy .ca/.cs HBM diagnostic from absent metadata.
                actual_cache = "ca" if record["workload"] == "l1" else "cg"
            _check(checks, "matching_requested_read_cache_policy", actual_cache,
                   lambda v: v == expected_cache, "effective modifier matches requested read cache policy")
        for name in ("read_cache_policy", "memory_read_index_math"):
            if name in provenance:
                _check(checks, "matching_variant_provenance_" + name, provenance[name],
                       lambda v, n=name: benchmark.get(n) is not None and v == benchmark[n],
                       "profile variant provenance matches effective worker metadata")
        if benchmark.get("kernel_implementation_version") == "scalar_single_stream_read_v3" and "memory_read_index_math" in benchmark:
            working_set, iterations, blocks = (_number(benchmark.get(name)) for name in ("working_set_bytes", "iterations_per_launch", "blocks"))
            expected_index = None
            dimensions = (working_set, iterations, blocks) if record["workload"] == "l1" else (working_set, iterations)
            if all(value is not None and value > 0 and value.is_integer() for value in dimensions):
                words = int(working_set) // 4
                if record["workload"] == "l1":
                    words //= int(blocks)
                expected_index = "uint32" if words <= 2**32 - 1 and iterations <= 2**32 - 1 else "uint64"
            _check(checks, "consistent_memory_read_index_math", benchmark.get("memory_read_index_math") if expected_index else None,
                   lambda v: v == expected_index, "index variant matches effective region words and iteration bounds")
    clocks = provenance.get("requested_clocks") or {}
    actual_clocks = {}
    required_effective = ("blocks", "threads", "iterations_per_launch", "tensor_accumulators") if record.get("workload") == "tensor" else ("gemm_m", "gemm_n", "gemm_k", "iterations_per_launch") if record.get("workload") == "gemm" else ("blocks", "threads", "iterations_per_launch", "working_set_bytes", "stride_elements", "offset_bytes", "access")
    if record.get("workload") in SFU_WORKLOADS:
        required_effective = ("blocks", "threads", "paired_reference_context_allocated", *SFU_CONTRACT_FIELDS)
    if scalar_read and "scalar_single_stream_read_v3" in (benchmark.get("kernel_implementation_version"), measured_benchmark.get("kernel_implementation_version")):
        required_effective += ("read_cache_policy", "memory_read_index_math")
    if record.get("workload") in NONLINEAR_WORKLOADS:
        required_effective += ("row_width", "math_implementation", "input_precision", "nonlinear_input_distribution")
        if (measured_benchmark.get("math_implementation") == Q_GRID_MATH_IMPLEMENTATION
                or measured_benchmark.get("kernel_implementation_version") == Q_GRID_IMPLEMENTATION_VERSION):
            required_effective += ("grid_mode", "input_elements", "kernel_implementation_version", "block_completion_count_source")
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
    coalescing = result["memory_coalescing"]
    payload_binding_checks = [check for check in checks[:path_check_count] if check["name"] in (
        "profile_memory_count_contract", "single_profile_workload_launch", "profile_logical_bytes",
        "memory_profile_launch_id_unique_mapping", "memory_profile_observed_launch_count")]
    binding_verified = _status(checks[path_check_count:] + payload_binding_checks) == "pass"
    if coalescing["applicable"] and not binding_verified:
        coalescing["reasons"] = sorted(set(coalescing["reasons"] + ["coalescing_profile_binding_unverified"]))
        # An unrelated/mismatched replay cannot qualify or disqualify the
        # measured run's coalescing. Preserve raw ratios with the binding flag.
        coalescing["status"] = "fail" if coalescing["geometry"]["status"] == "fail" else "inconclusive"
    coalescing.update(target_path_status=result["status"], profile_binding_verified=binding_verified,
                      energy_eligible=coalescing["status"] == "pass" and result["suitable_verified"])
    return result
