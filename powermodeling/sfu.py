"""Exact scalar instruction contracts for bounded register SFU recurrence loops.

These are approximate PTX primitive applications per active lane, not full
functions, FLOPs, warp instructions or isolated physical SFU energy events.
"""
import math

SFU_WORKLOADS = {"sfu_ex2", "sfu_lg2", "sfu_rcp", "sfu_rsqrt", "sfu_sqrt", "sfu_tanh"}
KERNEL_IMPLEMENTATION = "sfu_register_recurrence_v1"
CONTROL_IMPLEMENTATION = "sfu_register_control_v1"
MATH_IMPLEMENTATION = "ptx_approx_register_v1"
INPUT_POLICY = "feedback_xor_iteration_mantissa_0p5_1_v1"
REFERENCE_KIND = "register_loop_without_sfu"
COUNT_SOURCE = "synchronized_completed_launches"
PTX_OPCODES = {primitive: primitive + ".approx.ftz.f32" for primitive in ("ex2", "lg2", "rcp", "rsqrt", "sqrt")}
PTX_OPCODES["tanh"] = "tanh.approx.f32"
CONTRACT_FIELDS = ("sfu_lanes", "sfu_chains", "sfu_primitive", "sfu_input_policy",
                   "iterations_per_launch", "grid_mode", "math_implementation",
                   "kernel_implementation_version", "block_completion_count_source", "sfu_ptx_opcode",
                   "sfu_approximate", "sfu_flush_to_zero", "sfu_exponent_base")


def count_issues(benchmark, workload, *, reference=False):
    """Reject contradictory target/control counters; there is no legacy fallback."""
    if workload not in SFU_WORKLOADS:
        return []
    issues = []
    prefix = "sfu_reference_" if reference else "sfu_"

    def number(key, obj=benchmark):
        value = obj.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or value != int(value)):
            issues.append(prefix + "invalid_count:" + key)
            return None
        return value

    def equal(name, actual, expected):
        if actual is None or expected is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
            issues.append(prefix + "count_disagreement:" + name)

    expected_version = CONTROL_IMPLEMENTATION if reference else KERNEL_IMPLEMENTATION
    if benchmark.get("kernel_implementation_version") != expected_version:
        issues.append(prefix + "unsupported_implementation")
    if not reference and benchmark.get("math_implementation") != MATH_IMPLEMENTATION:
        issues.append(prefix + "unsupported_math_contract")
    if benchmark.get("sfu_input_policy") != INPUT_POLICY or benchmark.get("sfu_primitive") != workload.removeprefix("sfu_"):
        issues.append(prefix + "unsupported_primitive_or_input_policy")
    primitive = workload.removeprefix("sfu_")
    if (benchmark.get("sfu_ptx_opcode") != PTX_OPCODES[primitive]
            or benchmark.get("sfu_approximate") is not True
            or benchmark.get("sfu_flush_to_zero") is not (primitive != "tanh")
            or benchmark.get("sfu_exponent_base") != (2 if primitive in ("ex2", "lg2") else None)):
        issues.append(prefix + "unsupported_ptx_definition")
    if benchmark.get("block_completion_count_source") != COUNT_SOURCE:
        issues.append(prefix + "unsupported_count_source")
    if reference and (benchmark.get("workload") != "control" or benchmark.get("reference_kind") != REFERENCE_KIND):
        issues.append(prefix + "unsupported_reference_definition")
    lanes, chains, iterations, blocks, threads, launches = (number(key) for key in (
        "sfu_lanes", "sfu_chains", "iterations_per_launch", "blocks", "threads", "kernel_launches"))
    if (not lanes or chains not in (1, 4, 8) or not iterations or iterations > 2**32
            or not blocks or blocks > 1000000 or not launches
            or threads is None or not 32 <= threads <= 1024 or threads % 32):
        issues.append(prefix + "invalid_execution_parameters")
    mode = benchmark.get("grid_mode")
    if mode not in ("auto", "fixed"):
        issues.append(prefix + "invalid_grid_mode")
    elif mode == "auto" and lanes and threads:
        equal("auto_grid_blocks", blocks, math.ceil(lanes / threads))
    batch_launches = number("batch_launches") if "batch_launches" in benchmark else None
    if "batch_launches" in benchmark and (not batch_launches or batch_launches > 65536):
        issues.append(prefix + "invalid_batch_launches")

    def check_counts(obj, label):
        count = number("kernel_launches", obj)
        slots = count * lanes * iterations * chains if None not in (count, lanes, iterations, chains) else None
        equal(label + "completed_ctas", number("admitted_blocks", obj), count * blocks if count is not None and blocks is not None else None)
        equal(label + "instructions", number("sfu_instructions", obj), 0 if reference else slots)
        equal(label + "operations", number("operations", obj), 0 if reference else slots)
        if reference:
            equal(label + "reference_loop_slots", number("reference_loop_slots", obj), slots)
        equal(label + "logical_bytes", number("logical_bytes", obj), 0)
        equal(label + "elements", number("elements", obj), 0)
        if "batches" in obj:
            batches = number("batches", obj)
            equal(label + "batch_kernel_launches", count, batches * batch_launches if batches is not None and batch_launches is not None else None)

    check_counts(benchmark, "total_")
    epochs = benchmark.get("measure_epochs")
    if not isinstance(epochs, list) or not epochs:
        issues.append(prefix + "missing_measure_epochs")
    else:
        for epoch in epochs:
            if not isinstance(epoch, dict):
                issues.append(prefix + "invalid_epoch")
                continue
            for field in CONTRACT_FIELDS + ("batch_launches",):
                if field in epoch and epoch[field] != benchmark.get(field):
                    issues.append(prefix + "epoch_contract_disagreement:" + field)
            check_counts(epoch, "epoch_")
            if epoch.get("counts_exact") is not True:
                issues.append(prefix + "inexact_epoch")
        for field in ("kernel_launches", "sfu_instructions", "operations", "reference_loop_slots") if reference else ("kernel_launches", "sfu_instructions", "operations"):
            values = [number(field, epoch) for epoch in epochs if isinstance(epoch, dict)]
            equal("epoch_total_" + field, sum(values) if all(value is not None for value in values) else None, number(field))
    sanity = benchmark.get("sanity") or {}
    if sanity.get("finite_output_sample") is False:
        issues.append(prefix + "output_sanity_failed")
    if not reference:
        if sanity.get("numerical_validation_passed") is not True:
            issues.append("sfu_numerical_validation_failed_or_missing")
        checked = (benchmark.get("numerical_validation") or {}).get("checked_values")
        if isinstance(checked, bool) or not isinstance(checked, (int, float)) or not math.isfinite(checked) or checked <= 0:
            issues.append("sfu_missing_one_step_reference_checks")
    return sorted(set(issues))
