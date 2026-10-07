#!/usr/bin/env python3
"""Disassemble a benchmark and save reproducible native-SFU loop evidence.

Example:
  python tools/check_sfu_sass.py --binary build/powerbench --output sfu-sass.json

Use a CUDA 12 nvdisasm when the binary includes Volta/sm70. CUDA 13 nvdisasm
does not support that architecture. No GPU or driver is needed. Certificate
validation checks raw SASS again; the stored pass flag is only a summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from powermodeling.sfu_sass import CHAINS, OPERATIONS, extract_sfu_disassembly, make_certificate


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable(value):
    located = shutil.which(value)
    if located is None:
        raise OSError(f"Executable not found: {value}")
    return str(Path(located).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--nvdisasm", help="Explicit nvdisasm executable; otherwise cuobjdump resolves it")
    parser.add_argument("--workdir", type=Path, help="Parent directory for temporary tool files")
    args = parser.parse_args()
    try:
        binary = args.binary.resolve(strict=True)
        cuobjdump = _executable(args.cuobjdump)
        before_hash = _sha256(binary)
        environment = dict(os.environ)
        if args.workdir:
            args.workdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sfu-sass-", dir=args.workdir) as work:
            if args.nvdisasm:
                nvdisasm = _executable(args.nvdisasm)
                (Path(work) / "nvdisasm").symlink_to(nvdisasm)
                environment["PATH"] = work + os.pathsep + environment.get("PATH", "")
            outputs = {}
            for key, flag in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
                result = subprocess.run([cuobjdump, flag, str(binary)], check=True, capture_output=True,
                                        text=True, env=environment, cwd=work)
                outputs[key] = result.stdout
            after_hash = _sha256(binary)
            if before_hash != after_hash:
                raise OSError("Benchmark changed while being disassembled; rebuild, then run the check again")
            sass, resources = extract_sfu_disassembly(outputs["sass"], outputs["resources"])
            certificate = make_certificate(sass, resources, before_hash)
            certificate["tooling"] = {"cuobjdump": cuobjdump, "nvdisasm": args.nvdisasm,
                                      "binary_name": binary.name}
        # The repository binary is expected to include all supported primitive
        # and chain variants. Unsupported Volta TANH must remain an explicit trap.
        available = {(r["cc"], r["workload"], r["control"], r["chains"]) for r in certificate["kernels"]}
        for cc in certificate["architectures"]:
            for chains in sorted(CHAINS):
                expected = [(cc, name, False, chains) for name, _ in OPERATIONS.values()]
                expected.append((cc, "sfu_ex2", True, chains))
                for key in expected:
                    if key not in available:
                        certificate["issues"].append("missing_suite_specialization:" + ":".join(map(str, key)))
        if certificate["issues"]:
            certificate["status"] = "fail"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"ERROR: cuobjdump failed ({error.returncode}): {(error.stderr or '').strip()}", file=sys.stderr)
        return 2
    print(f"{certificate['status'].upper()}: {len(certificate['kernels'])} SFU kernel specializations; {args.output}")
    for issue in certificate["issues"]:
        print(f"FAIL: {issue}", file=sys.stderr)
    for row in certificate["kernels"]:
        if row["status"] == "fail":
            print(f"FAIL: {row['architecture']} {row['workload']} chains={row['chains']} control={row['control']}: "
                  + ", ".join(row["issues"]), file=sys.stderr)
    return 0 if certificate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
