"""Counting contract for complete FP32 nonlinear function applications.

An element is one output value, including the full row reduction for RMSNorm
and stable Softmax. This denominator is neither a FLOP nor an SFU instruction.
"""
import math

NONLINEAR_WORKLOADS = {"exp", "tanh", "silu", "rmsnorm", "softmax"}
ROW_WORKLOADS = {"rmsnorm", "softmax"}
MATH_IMPLEMENTATION = "cuda_fp32_standard_streaming_v1"
RMS_EPSILON = 1e-5


def logical_bytes_per_element(workload):
    # RMSNorm: x twice, gamma once, y once. Softmax: x three times, y once.
    return 16 if workload in ROW_WORKLOADS else 8


def count_issues(benchmark, workload):
    """Reject missing/contradictory denominators rather than invent a rate."""
    if workload not in NONLINEAR_WORKLOADS:
        return []
    issues = []

    def number(key, obj=benchmark):
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value != int(value):
            issues.append("invalid_nonlinear_count:" + key)
            return None
        return value

    def equal(name, actual, expected):
        if actual is None or expected is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
            issues.append("nonlinear_count_disagreement:" + name)

    elements, operations = number("elements"), number("operations")
    if elements is not None and elements <= 0:
        issues.append("missing_nonlinear_elements")
    equal("operations_are_elements", operations, elements)
    equal("logical_bytes", number("logical_bytes"), elements * logical_bytes_per_element(workload) if elements is not None else None)
    width = number("row_width")
    if workload in ROW_WORKLOADS:
        if width is None or not 1 <= width <= 65536 or width != int(width):
            issues.append("invalid_nonlinear_row_width")
        equal("row_evaluations", number("row_evaluations"), elements / width if elements is not None and width else None)
    else:
        equal("row_width", width, 0)
        equal("row_evaluations", number("row_evaluations"), 0)
    blocks, ws, iterations, admitted = (number(key) for key in ("blocks", "working_set_bytes", "iterations_per_launch", "admitted_blocks"))
    if blocks and ws and iterations and admitted:
        equal("admission_elements", elements, admitted * (ws / 4 / blocks) * iterations)
        if ws % (4 * blocks * (width if workload in ROW_WORKLOADS and width else 1)):
            issues.append("incomplete_nonlinear_block_slice_or_row")
    else:
        issues.append("missing_nonlinear_execution_counts")
    if benchmark.get("math_implementation") != MATH_IMPLEMENTATION or benchmark.get("input_precision") != "fp32":
        issues.append("unsupported_nonlinear_math_contract")
    if workload == "rmsnorm" and (benchmark.get("rms_epsilon") != RMS_EPSILON or benchmark.get("affine_gamma") is not True):
        issues.append("unsupported_rmsnorm_definition")
    if (benchmark.get("sanity") or {}).get("numerical_validation_passed") is not True:
        issues.append("nonlinear_numerical_validation_failed_or_missing")
    checked = (benchmark.get("numerical_validation") or {}).get("checked_values")
    if isinstance(checked, bool) or not isinstance(checked, (int, float)) or not math.isfinite(checked) or checked <= 0:
        issues.append("missing_nonlinear_numerical_reference_checks")
    epochs = benchmark.get("measure_epochs")
    if not isinstance(epochs, list) or not epochs:
        issues.append("missing_nonlinear_measure_epochs")
    else:
        for epoch in epochs:
            if not isinstance(epoch, dict):
                issues.append("invalid_nonlinear_epoch")
                continue
            e = number("elements", epoch)
            equal("epoch_operations_are_elements", number("operations", epoch), e)
            equal("epoch_logical_bytes", number("logical_bytes", epoch), e * logical_bytes_per_element(workload) if e is not None else None)
            equal("epoch_rows", number("row_evaluations", epoch), e / width if e is not None and width and workload in ROW_WORKLOADS else 0)
            if epoch.get("counts_exact") is not True:
                issues.append("inexact_nonlinear_measure_epoch")
        equal("epoch_total_elements", sum(epoch.get("elements", 0) for epoch in epochs if isinstance(epoch, dict) and isinstance(epoch.get("elements"), (int, float))), elements)
    return sorted(set(issues))
