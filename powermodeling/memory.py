"""Count contracts for explicitly versioned scalar memory implementations.

Legacy records retain their reported denominators: an absent version is never
interpreted as the new single-stream implementation or divided by four.
"""
import math

SINGLE_STREAM_READ = "scalar_single_stream_read_v2"
FOUR_STREAM_WRITE_COPY = "scalar_four_stream_write_copy_v1"
VERSIONS = {SINGLE_STREAM_READ, FOUR_STREAM_WRITE_COPY}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def count_issues(benchmark, workload):
    if workload not in ("l1", "l2", "hbm") or benchmark.get("kernel_implementation_version") not in VERSIONS:
        return []
    version, access = benchmark["kernel_implementation_version"], benchmark.get("access")
    read = version == SINGLE_STREAM_READ
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
