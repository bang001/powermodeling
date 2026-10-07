"""Recompute register-SFU loop evidence from disassembly, without a GPU.

Certificates bind a reviewed binary hash to raw SASS and resource output. They
are reproducible build evidence, not signatures or runtime utilization proofs.
Only the selected architecture, primitive, and chain count are admitted by
``validate_certificate``; a stored ``status`` flag is never trusted.
"""
from __future__ import annotations

from collections import Counter
import re


SCHEMA_VERSION = 1
KIND = "sfu_register_sass"
OPERATIONS = {
    0: ("sfu_ex2", "EX2"), 1: ("sfu_tanh", "TANH"),
    2: ("sfu_rsqrt", "RSQ"), 3: ("sfu_rcp", "RCP"),
    4: ("sfu_lg2", "LG2"), 5: ("sfu_sqrt", "SQRT"),
}
CHAINS = {1, 4, 8}
FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
ARCH = re.compile(r"EF_CUDA_SM(\d+)\b")
SPECIALIZATION = re.compile(r"sfu_register_kernelILi(\d+)ELb([01])ELi(\d+)EE")
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;", re.MULTILINE)
BRANCH = re.compile(r"\bBRA(?:\.[A-Z]+)*\s+(?:!?P\d+,\s*)?(0x[0-9a-fA-F]+)\b")
MUFU = re.compile(r"\bMUFU\.([A-Z0-9]+)\b")
MEMORY = re.compile(r"\b(?:LDG|STG|LDS|STS|LDL|STL|LDC|ULDC|LD|ST|ATOM|ATOMS|RED|REDUX|CCTL|CCTLL|CPASYNC|UTMA|LDGSTS|TEX|TLD|SULD|SUST|SUATOM|SURED)(?:\.|\s)")
LOCAL = re.compile(r"\b(?:LDL|STL)(?:\.|\s)")
FLOAT_ALU = re.compile(r"\b(?:FADD|FMUL|FFMA|FSETP|FCHK|RRO|DADD|DMUL|DFMA)(?:\.|\s)")
CONTROL_FLOW = re.compile(r"\b(?:BRA|BRX|JMP|JMX|CALL|RET|EXIT|TRAP)(?:\.|\s|$)")
RESOURCE = re.compile(r"\b(REG|STACK|SHARED|LOCAL):(\d+)")
HOT_OPCODES = {"MUFU", "LOP3", "ULOP3", "LOP", "IADD3", "UIADD3", "IADD", "UIADD",
               "VIADD", "IMAD", "UIMAD", "ISETP", "UISETP", "BRA", "SHFL", "SHF", "USHF",
               "MOV", "UMOV", "NOP"}


def _opcode(instruction):
    words = instruction.split()
    return words[1] if words[0].startswith("@") else words[0]


def normalize_cc(cc) -> int | None:
    if isinstance(cc, bool):
        return None
    text = str(cc)
    if re.fullmatch(r"\d+\.\d", text):
        return int(text.replace(".", ""))
    if re.fullmatch(r"(?:sm_)?\d+", text):
        return int(text.removeprefix("sm_"))
    return None


def _resources(text: str, architectures: set[int]):
    rows, errors = {}, []
    arch = next(iter(architectures)) if len(architectures) == 1 else None
    name = None
    for line in text.splitlines():
        found_arch = re.search(r"\barch\s*=\s*sm_(\d+)\b", line)
        if found_arch:
            arch = int(found_arch.group(1))
            name = None
        found_function = re.match(r"\s*Function\s+(\S+):\s*$", line)
        if found_function:
            name = found_function.group(1)
        values = dict(RESOURCE.findall(line))
        if values and name and "sfu_register_kernel" in name:
            key = (arch, name)
            if key in rows:
                errors.append("duplicate_sfu_resource_row")
            rows[key] = {key.lower(): int(value) for key, value in values.items()}
            name = None
    return rows, errors


def inspect_disassembly(raw_sass: str, raw_resources: str) -> dict:
    """Inspect all SFU specializations; retain detailed failures in each row.

    The current contract is one non-unrolled recurrence loop inside an optional
    Q grid-stride loop. The recurrence has one +1 induction step, one back edge,
    exactly ``chains`` unpredicated native MUFU instructions (zero in control),
    and no memory, calls, or floating arithmetic expansion. The single global
    store occurs after the recurrence, once per processed logical lane.
    """
    if not isinstance(raw_sass, str) or not isinstance(raw_resources, str):
        return {"status": "fail", "issues": ["missing_raw_sfu_disassembly"], "kernels": []}
    functions = list(FUNCTION.finditer(raw_sass))
    architectures = {int(m.group(1)) for m in ARCH.finditer(raw_sass)}
    resources, issues = _resources(raw_resources, architectures)
    rows = []
    for index, match in enumerate(functions):
        name = match.group(1)
        if "sfu_register_kernel" not in name:
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(raw_sass)
        body = raw_sass[match.end():end]
        arch_match, spec = ARCH.search(body), SPECIALIZATION.search(name)
        if not arch_match or not spec:
            issues.append("unrecognized_sfu_kernel_specialization")
            continue
        cc = int(arch_match.group(1))
        operation, control, chains = int(spec.group(1)), spec.group(2) == "1", int(spec.group(3))
        workload, expected_opcode = OPERATIONS.get(operation, ("unknown", "unknown"))
        problems = []
        if operation not in OPERATIONS or chains not in CHAINS or (control and operation != 0):
            problems.append("unsupported_sfu_specialization")
        instructions = [(int(m.group(1), 16), m.group(2)) for m in INSTRUCTION.finditer(body)]
        if len({address for address, _ in instructions}) != len(instructions):
            problems.append("duplicate_sfu_instruction_address")
        row = {"workload": workload, "control": control, "chains": chains,
               "cc": cc, "architecture": f"sm_{cc}", "kernel": name,
               "target_opcode": expected_opcode, "loop_unroll_factor": 1}
        resource = resources.get((cc, name), {})
        row["resources"] = resource
        row["no_spills"] = (all(k in resource for k in ("reg", "stack", "local", "shared"))
                            and resource["reg"] > 0 and resource["stack"] == resource["local"] == 0
                            and not any(LOCAL.search(i) for _, i in instructions))
        unsupported_tanh = operation == 1 and cc < 75 and not control
        if unsupported_tanh:
            trapped = any(re.search(r"\b(?:TRAP|BPT)(?:\.|\s|$)", i) for _, i in instructions)
            row.update(status="unsupported" if trapped and not problems else "fail",
                       issues=problems + ([] if trapped else ["unsupported_tanh_missing_trap"]),
                       target_instruction_count_per_iteration=0,
                       no_hot_loop_memory=None, control_target_instructions=None)
            rows.append(row)
            continue
        loops = []
        for address, instruction in instructions:
            branch = BRANCH.search(instruction)
            if branch and int(branch.group(1), 16) < address:
                start = int(branch.group(1), 16)
                loops.append((start, address))
        inner = [(start, end) for start, end in loops
                 if not any(start <= a and b <= end and (a, b) != (start, end) for a, b in loops)]
        if len(inner) != 1:
            problems.append("expected_one_sfu_recurrence_loop")
        hot = [(a, i) for a, i in instructions if inner and inner[0][0] <= a <= inner[0][1]] if len(inner) == 1 else []
        mu = [(a, i, MUFU.search(i).group(1)) for a, i in hot if MUFU.search(i)]
        expected_count = 0 if control else chains
        if len(mu) != expected_count or any(op != expected_opcode for _, _, op in mu):
            problems.append("sfu_instruction_count_or_opcode_mismatch")
        if any(i.lstrip().startswith("@") for _, i, _ in mu):
            problems.append("predicated_sfu_instruction")
        increments = [i for _, i in hot if re.search(r"\b(?:IADD3|UIADD3|IADD|UIADD|VIADD)(?:\.[A-Z]+)*\s", i)
                      and re.search(r",\s*(?:0x1|1),\s*(?:RZ|URZ)\b", i)]
        if len(increments) != 1:
            problems.append("unverified_sfu_loop_increment")
        if hot and (not hot[-1][1].lstrip().startswith("@") or not BRANCH.search(hot[-1][1])):
            problems.append("unverified_sfu_loop_back_edge")
        if any(CONTROL_FLOW.search(i) for _, i in hot[:-1]):
            problems.append("additional_sfu_hot_loop_control_flow")
        row["no_hot_loop_memory"] = bool(hot) and not any(MEMORY.search(i) for _, i in hot)
        if not row["no_hot_loop_memory"]:
            problems.append("sfu_hot_loop_memory_or_missing_loop")
        if any(FLOAT_ALU.search(i) for _, i in hot):
            problems.append("sfu_hot_loop_floating_expansion")
        if any(_opcode(i).split(".")[0] not in HOT_OPCODES for _, i in hot):
            problems.append("unreviewed_sfu_hot_loop_opcode")
        if not row["no_spills"]:
            problems.append("sfu_spills_or_missing_resources")
        if any(re.search(r"\b(?:LDG|LDS|STS|LDL|STL|ATOM|ATOMS|RED|REDUX)(?:\.|\s)", i) for _, i in instructions):
            problems.append("unexpected_sfu_kernel_memory_or_atomic")
        if any("SMID" in i for _, i in instructions):
            problems.append("unexpected_sfu_smid_telemetry")
        stores = [(a, i) for a, i in instructions if re.search(r"\bSTG(?:\.|\s)", i)]
        if len(stores) != 1 or (inner and stores and stores[0][0] <= inner[0][1]):
            problems.append("expected_single_sfu_sink_after_recurrence")
        if any(MUFU.search(i) for a, i in instructions if (a, i) not in hot):
            problems.append("sfu_instruction_outside_recurrence")
        row.update(status="fail" if problems else "pass", issues=sorted(set(problems)),
                   target_instruction_count_per_iteration=len(mu),
                   control_target_instructions=len(mu) if control else None,
                   global_sink_store_count=len(stores), loop_instruction_count=len(hot),
                   common_opcode_counts=dict(Counter(_opcode(i) for _, i in hot if not MUFU.search(i))),
                   constant_parameter_operand_count=sum("c[" in i for _, i in hot),
                   memory_check_scope="no global/shared/local/texture/surface/atomic/cache or explicit constant load instructions; immutable parameter operands in instruction constant-bank fields are reported separately",
                   loop_start=inner[0][0] if len(inner) == 1 else None,
                   loop_end=inner[0][1] if len(inner) == 1 else None,
                   loop_instructions=[{"address": a, "instruction": i} for a, i in hot])
        rows.append(row)
    keys = Counter((r["cc"], r["workload"], r["control"], r["chains"]) for r in rows)
    if any(count > 1 for count in keys.values()):
        issues.append("duplicate_sfu_kernel_specialization")
    for row in rows:
        if row["control"] or row["status"] != "pass":
            continue
        controls = [control for control in rows if control["control"] and control["cc"] == row["cc"]
                    and control["chains"] == row["chains"] and control["workload"] == "sfu_ex2"]
        if len(controls) != 1 or controls[0]["status"] != "pass":
            row["issues"].append("missing_or_unverified_sfu_common_control")
        elif row["common_opcode_counts"] != controls[0]["common_opcode_counts"]:
            row["issues"].append("sfu_target_control_common_instruction_mismatch")
        if row["issues"]:
            row["status"] = "fail"
    if not rows:
        issues.append("missing_sfu_kernels")
    return {"status": "fail" if issues or any(r["status"] == "fail" for r in rows) else "pass",
            "issues": sorted(set(issues)), "architectures": sorted(architectures), "kernels": rows}


def make_certificate(raw_sass: str, raw_resources: str, benchmark_sha256: str) -> dict:
    result = inspect_disassembly(raw_sass, raw_resources)
    return {"schema_version": SCHEMA_VERSION, "kind": KIND, "benchmark_sha256": benchmark_sha256,
            "raw_sass": raw_sass, "raw_resources": raw_resources, **result,
            "scope": "Static native instruction/count and register-loop checks for the bound binary; kernel parameters may use constant operands; not a runtime bandwidth, utilization, physical SFU energy, or numerical-accuracy proof"}


def extract_sfu_disassembly(raw_sass: str, raw_resources: str, selection=None) -> tuple[str, str]:
    """Keep SFU function/architecture headers and instruction text only.

    Address/instruction text remains verbatim; encoding comment columns and
    unrelated kernels are omitted. Optional selection is (cc, workload, chains)
    and includes its common control. Validate source evidence before narrowing.
    """
    functions = list(FUNCTION.finditer(raw_sass))
    architectures = {int(m.group(1)) for m in ARCH.finditer(raw_sass)}
    resource_rows, resource_issues = _resources(raw_resources, architectures)
    if resource_issues:
        raise ValueError(", ".join(resource_issues))
    code, resource_text = [], []
    for index, match in enumerate(functions):
        name = match.group(1)
        if "sfu_register_kernel" not in name:
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(raw_sass)
        body = raw_sass[match.end():end]
        arch, spec = ARCH.search(body), SPECIALIZATION.search(name)
        if not arch or not spec:
            raise ValueError("unrecognized_sfu_kernel_specialization")
        cc, operation, control, chains = int(arch.group(1)), int(spec.group(1)), spec.group(2) == "1", int(spec.group(3))
        if selection:
            wanted_cc, wanted_workload, wanted_chains = selection
            workload = OPERATIONS.get(operation, ("unknown",))[0]
            if cc != wanted_cc or chains != wanted_chains or workload != ("sfu_ex2" if control else wanted_workload):
                continue
        instructions = [f"/*{m.group(1)}*/ {m.group(2)};" for m in INSTRUCTION.finditer(body)]
        code.append(f'Function : {name}\n.headerflags @"EF_CUDA_SM{cc}"\n' + "\n".join(instructions))
        resource = resource_rows.get((cc, name), {})
        resource_text.append(f"arch = sm_{cc}\n Function {name}:\n " +
                             " ".join(f"{key.upper()}:{value}" for key, value in resource.items()))
    return "\n".join(code), "\n".join(resource_text)


def select_certificate(certificate, cc, workload, chains) -> dict:
    """Return a compact, recomputed target/control certificate for one trial."""
    issues = validate_certificate(certificate, certificate.get("benchmark_sha256") if isinstance(certificate, dict) else None,
                                  cc, workload, chains)
    if issues:
        raise ValueError(", ".join(issues))
    sass, resources = extract_sfu_disassembly(certificate["raw_sass"], certificate["raw_resources"],
                                             (normalize_cc(cc), workload, chains))
    return make_certificate(sass, resources, certificate["benchmark_sha256"])


def validate_certificate(certificate, benchmark_sha256, cc, workload, chains) -> list[str]:
    """Return issues, recomputing raw target and control evidence for one trial."""
    if not isinstance(certificate, dict):
        return ["missing_sfu_sass_evidence"]
    issues = []
    if type(certificate.get("schema_version")) is not int or certificate.get("schema_version") != SCHEMA_VERSION or certificate.get("kind") != KIND:
        issues.append("unsupported_sfu_sass_evidence_schema")
    if not isinstance(benchmark_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", benchmark_sha256):
        issues.append("missing_sfu_benchmark_sha256")
    if certificate.get("benchmark_sha256") != benchmark_sha256:
        issues.append("sfu_sass_binary_hash_mismatch")
    capability = normalize_cc(cc)
    if capability is None or workload not in {v[0] for v in OPERATIONS.values()} or type(chains) is not int or chains not in CHAINS:
        return sorted(set(issues + ["invalid_sfu_sass_selection"]))
    result = inspect_disassembly(certificate.get("raw_sass"), certificate.get("raw_resources"))
    issues.extend(result["issues"])
    selected = {}
    for control in (False, True):
        name = "sfu_ex2" if control else workload
        rows = [r for r in result["kernels"] if r["cc"] == capability and r["workload"] == name
                and r["control"] == control and r["chains"] == chains]
        label = "control" if control else "target"
        if len(rows) != 1:
            issues.append("missing_or_ambiguous_sfu_sass_" + label)
        elif rows[0]["status"] != "pass":
            issues.append("sfu_sass_" + label + "_not_verified")
            issues.extend(rows[0]["issues"])
        else:
            selected[control] = rows[0]
    if len(selected) == 2 and selected[False]["common_opcode_counts"] != selected[True]["common_opcode_counts"]:
        issues.append("sfu_target_control_common_instruction_mismatch")
    return sorted(set(issues))
