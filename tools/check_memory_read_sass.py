#!/usr/bin/env python3
"""Check compiled scalar-read loops without a GPU.

Examples::

    python tools/check_memory_read_sass.py --binary build/powerbench
    python tools/check_memory_read_sass.py --sass-file powerbench.sass

The binary mode requires cuobjdump and its matching nvdisasm on PATH; an
explicit cuobjdump path can be supplied. This detects the regression where
ptxas removes unused intermediate loads despite C++ ``asm volatile``: every
backward branch in each simple_memory_kernel must enclose a global load.
Both L1 and L2/HBM specializations must be present for every SASS architecture.

It also rejects local-memory accesses and recognizable XOR instructions in
these loops. This is a structural regression check, not a proof of dynamic
load counts, cache residency, coalescing, bandwidth, or measured energy. In
particular, this does not establish dynamic traffic. Version3 additionally
matches recognized loop induction steps against scalar load counts and limits
uint32-loop instruction overhead. Retain the runtime NCU traffic/path checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;", re.MULTILINE)
ARCHITECTURE = re.compile(r"EF_CUDA_SM(\d+[a-z]?)\b")
BRANCH = re.compile(r"\bBRA(?:\.[A-Z]+)*\s+(0x[0-9a-fA-F]+)\b")
LOAD = re.compile(r"\bLDG(?:\.[A-Z0-9]+)*\s")
LOCAL = re.compile(r"\b(?:LDL|STL)(?:\.[A-Z0-9]+)*\s")
XOR = re.compile(r"\bXOR(?:\.[A-Z0-9]+)*\s|\bLOP3\.LUT\b.*?,\s*0x(?:96|3c|66|5a),", re.IGNORECASE)
INDUCTION = re.compile(r"\bU?IADD3\s+(U?R\d+),\s*(?:U?P\d+,\s*)?\1,\s*(-?0x[0-9a-fA-F]+),\s*U?RZ\b")
VECTOR_INDUCTION = re.compile(r"\bVIADD\s+(R\d+),\s*\1,\s*(0x[0-9a-fA-F]+)\b")


def check_sass(sass: str) -> tuple[list[str], list[str]]:
    """Return human-readable checks and failures for cuobjdump SASS text."""
    checks: list[str] = []
    failures: list[str] = []
    functions = list(FUNCTION.finditer(sass))
    seen: dict[str, set[str]] = {}
    versioned = False
    for index, match in enumerate(functions):
        name = match.group(1)
        if "simple_memory_kernel" not in name:
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(sass)
        body = sass[match.end():end]
        arch_match = ARCHITECTURE.search(body)
        specialization = re.search(r"simple_memory_kernelILb([01])E", name)
        if not arch_match or not specialization:
            failures.append(f"Cannot identify architecture/specialization: {name}")
            continue
        arch = "sm_" + arch_match.group(1)
        variant = "L1" if specialization.group(1) == "1" else "L2/HBM"
        contract = re.search(r"simple_memory_kernelILb([01])ELi([012])ELb([01])E", name)
        narrow = False
        if contract:
            versioned = True
            policy = ("ca", "cg", "cs")[int(contract.group(2))]
            narrow = contract.group(3) == "1"
            variant += " " + policy + (" uint32" if narrow else " uint64")
        seen.setdefault(arch, set()).add(variant)
        label = f"{arch} {variant}"
        instructions = [(int(m.group(1), 16), m.group(2)) for m in INSTRUCTION.finditer(body)]
        loops = []
        for address, instruction in instructions:
            branch = BRANCH.search(instruction)
            if branch and int(branch.group(1), 16) < address:
                start = int(branch.group(1), 16)
                loops.append((start, address, [(a, i) for a, i in instructions if start <= a <= address]))
        if not loops:
            failures.append(f"{label}: no backward loop recognized; inspect disassembly manually")
            continue
        for start, end_address, loop in loops:
            loop_label = f"{label} loop 0x{start:x}..0x{end_address:x}"
            loads = sum(bool(LOAD.search(instruction)) for _, instruction in loop)
            checks.append(f"{loop_label}: {loads} static global load(s), {len(loop)} instructions")
            if not loads:
                failures.append(f"{loop_label}: no global load inside backward loop (possible load elimination)")
            if narrow and loads and len(loop) / loads > 10:
                failures.append(f"{loop_label}: {len(loop) / loads:.2f} static instructions/load exceeds uint32 ceiling10; inspect address/loop arithmetic")
            if contract:
                cache_marker = ".STRONG.GPU" if policy == "cg" else ".EF" if policy == "cs" else ".STRONG.CTA" if arch == "sm_70" else ".STRONG.SM"
                if any(cache_marker not in ins.split()[0] for _, ins in loop if LOAD.search(ins)):
                    failures.append(f"{loop_label}: scalar load cache policy opcode does not match {policy} ({cache_marker})")
                steps = []
                for _, ins in loop:
                    match = INDUCTION.search(ins) or VECTOR_INDUCTION.search(ins)
                    if match:
                        step = int(match.group(2), 16)
                        if step >= 2**31: step -= 2**32
                        if abs(step) in (1, 4, 16): steps.append(abs(step))
                if steps != [loads]:
                    failures.append(f"{loop_label}: scalar load count {loads} does not match recognized iteration induction steps {steps}; inspect disassembly")
            for address, instruction in loop:
                if LOCAL.search(instruction):
                    failures.append(f"{loop_label}: local-memory access at 0x{address:x}: {instruction}")
                if XOR.search(instruction):
                    failures.append(f"{loop_label}: XOR at 0x{address:x}: {instruction}")
    if not seen:
        failures.append("No recognizable simple_memory_kernel SASS specializations found")
    architectures = {"sm_" + match.group(1) for match in ARCHITECTURE.finditer(sass)}
    for arch in sorted(architectures):
        variants = seen.get(arch, set())
        expected = ({"L1 ca uint32", "L1 ca uint64"} |
                    {f"L2/HBM {policy} {mode}" for policy in ("ca", "cg", "cs") for mode in ("uint32", "uint64")}) if versioned else {"L1", "L2/HBM"}
        missing = expected - variants
        if missing:
            failures.append(f"{arch}: missing specialization(s): {', '.join(sorted(missing))}")
    return checks, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--binary", type=Path, help="CUDA executable/fatbin to disassemble")
    source.add_argument("--sass-file", type=Path, help="Existing cuobjdump --dump-sass output")
    parser.add_argument("--cuobjdump", default="cuobjdump", help="cuobjdump executable (default: PATH)")
    args = parser.parse_args()
    try:
        if args.sass_file:
            sass = args.sass_file.read_text(encoding="utf-8")
        else:
            result = subprocess.run(
                [args.cuobjdump, "--dump-sass", str(args.binary)],
                capture_output=True, text=True, check=True,
            )
            sass = result.stdout
    except (OSError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"ERROR: cuobjdump failed ({error.returncode}): {(error.stderr or '').strip()}", file=sys.stderr)
        return 2
    checks, failures = check_sass(sass)
    for check in checks:
        print(check)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("PASS: scalar-read loop structure; dynamic counts and NCU path validation remain separate checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
