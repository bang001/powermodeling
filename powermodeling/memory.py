"""Count contracts for explicitly versioned scalar memory implementations.

Legacy records retain their reported denominators: an absent version is never
interpreted as the new single-stream implementation or divided by four.
"""
import math

SINGLE_STREAM_READ = "scalar_single_stream_read_v3"
SINGLE_STREAM_READ_VERSIONS = {SINGLE_STREAM_READ, "scalar_single_stream_read_v2"}
FOUR_STREAM_WRITE_COPY = "scalar_four_stream_write_copy_v1"
VERSIONS = SINGLE_STREAM_READ_VERSIONS | {FOUR_STREAM_WRITE_COPY}
READ_WORKLOADS = {"l1", "l2", "hbm"}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def coalesced_read_geometry(benchmark, workload, config=None):
    """Check sector-aligned scalar reads without claiming observed traffic.

    CUDA allocation bases are aligned; the byte offset and every L1 CTA slice
    must preserve 32-byte sector alignment. A 128-byte aligned base is not
    necessary. The requested efficiency describes one unwrapped full warp at
    the first region base, not actual cache/DRAM traffic or wraparound reuse.
    """
    benchmark, config = benchmark or {}, config or {}

    def get(key, default=None):
        return benchmark[key] if key in benchmark else config.get(key, default)

    access = get("access", "read")
    applicable = workload in READ_WORKLOADS and access == "read"
    result = {
        "applicable": applicable, "status": "inconclusive", "energy_eligible": False,
        "reasons": [], "word_bytes": 4, "sector_bytes": 32,
        "expected_sector_efficiency_fraction": 1.0 if applicable else None,
        "requested_sector_efficiency_fraction": None,
        "requested_sector_efficiency_scope": "first full warp scalar read at the declared region base, without wrapping or duplicate addresses; observed counters are required",
        "effective_region_bytes": None, "region_source": None,
        "base_alignment_scope": "CUDA allocation base plus offset; each L1 CTA adds its slice size; sector alignment does not require 128-byte alignment",
    }
    if not applicable:
        result["reasons"] = ["coalesced_read_energy_not_applicable"]
        return result
    failures, unknown = [], []

    def integer(key, default=None, minimum=0):
        value = get(key, default)
        number = _number(value)
        if value is None:
            unknown.append("missing_geometry:" + key)
        elif number is None or number < minimum or not number.is_integer():
            failures.append("invalid_geometry:" + key)
        else:
            return int(value)
        return None

    stride = integer("stride_elements", 1, 1)
    offset = integer("offset_bytes", 0)
    threads = integer("threads", minimum=1)
    working_set = integer("working_set_bytes", minimum=4)
    result.update(stride_elements=stride, stride_words=stride,
                  lane_stride_bytes=stride * 4 if stride is not None else None,
                  offset_bytes=offset, threads=threads)
    for key, expected in (("memory_word_bytes", 4), ("stride_words", stride),
                          ("lane_stride_bytes", stride * 4 if stride is not None else None)):
        if key in benchmark or key in config:
            declared = _number(get(key))
            if declared is None or declared <= 0 or not declared.is_integer():
                failures.append("invalid_geometry:" + key)
            elif expected is not None and declared != expected:
                failures.append("inconsistent_word_stride_metadata:" + key)
    if stride is not None and stride != 1:
        failures.append("nonunit_word_stride")
    if offset is not None and offset % 32:
        failures.append("offset_not_sector_aligned")
    if threads is not None and (threads < 32 or threads > 1024 or threads % 32):
        failures.append("threads_not_complete_supported_warps")
    if working_set is not None and working_set % 4:
        failures.append("working_set_not_word_aligned")

    region = working_set
    result["region_source"] = "working_set_bytes"
    if workload == "l1":
        blocks = integer("blocks", minimum=1)
        derived = working_set // 4 // blocks * 4 if working_set is not None and blocks else None
        reported = get("l1_bytes_per_block")
        if reported is not None:
            region = integer("l1_bytes_per_block", minimum=4)
            result["region_source"] = "l1_bytes_per_block"
            if derived is not None and region is not None and region != derived:
                failures.append("l1_slice_disagrees_with_working_set_and_blocks")
        else:
            region = derived
            result["region_source"] = "floor(working_set_words/blocks)*4"
    result["effective_region_bytes"] = region
    if region is None:
        unknown.append("effective_region_unknown")
    elif region < 128:
        failures.append("region_smaller_than_full_warp_payload")
    elif region % 32:
        failures.append("region_not_sector_aligned")

    if stride is not None and offset is not None:
        sectors = {(offset % 32 + lane * stride * 4) // 32 for lane in range(32)}
        result["requested_sector_efficiency_fraction"] = 128 / (len(sectors) * 32)
    result["reasons"] = sorted(set(failures + unknown))
    result["status"] = "fail" if failures else "inconclusive" if unknown else "pass"
    result["energy_eligible"] = result["status"] == "pass"
    return result


def count_issues(benchmark, workload):
    if workload not in ("l1", "l2", "hbm") or benchmark.get("kernel_implementation_version") not in VERSIONS:
        return []
    version, access = benchmark["kernel_implementation_version"], benchmark.get("access")
    read = version in SINGLE_STREAM_READ_VERSIONS
    factor = 1 if read else 4
    issues = []
    if access not in (("read",) if read else ("write", "copy")):
        issues.append("memory_contract_access_implementation_mismatch")
    if _number(benchmark.get("memory_accesses_per_thread_iteration")) != factor:
        issues.append("memory_contract_accesses_per_iteration_mismatch")
    threads, iterations = (_number(benchmark.get(key)) for key in ("threads", "iterations_per_launch"))
    if any(value is None or value <= 0 or not value.is_integer() for value in (threads, iterations)):
        issues.append("memory_contract_invalid_execution_parameters")
        return issues
    records = [("total", benchmark)]
    epochs = benchmark.get("measure_epochs", benchmark.get("work_epochs"))
    if isinstance(epochs, list):
        records.extend(("epoch_" + str(index), epoch) for index, epoch in enumerate(epochs) if isinstance(epoch, dict))
    for label, record in records:
        admitted = _number(record.get("admitted_blocks"))
        if admitted is None or admitted < 0 or not admitted.is_integer():
            issues.append("memory_contract_missing_or_invalid_admissions:" + label)
            continue
        operations = admitted * threads * iterations * factor
        logical_bytes = operations * 4 * (2 if access == "copy" else 1)
        for field, expected in (("operations", operations), ("logical_bytes", logical_bytes)):
            actual = _number(record.get(field))
            if actual is None or actual < 0 or abs(actual - expected) > max(1.0, abs(expected) * 1e-6):
                issues.append("memory_contract_count_disagreement:" + label + ":" + field)
    return issues
