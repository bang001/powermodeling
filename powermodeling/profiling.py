"""Separate Nsight Compute validation. Never mix replay runs with power results."""

import csv
import io
import json
from pathlib import Path
import subprocess

from .planner import benchmark_command
from .runner import atomic_json

# Availability differs between architectures and Nsight Compute releases.
METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_sector_hit_rate.pct",
           "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
           "l1tex__t_sector_hit_rate.pct", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
           "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]


def query_metrics(ncu="ncu"):
    result = subprocess.run([ncu, "--query-metrics", "--query-metrics-mode", "all"],
                            check=True,capture_output=True,text=True,timeout=60)
    return result.stdout


def profile_command(ncu, executable, trial, device_index=0, available=None):
    metrics = METRICS if available is None else [m for m in METRICS if m in available]
    if not metrics: raise ValueError("None of the validation counters are available")
    # Exclude deterministic initialization and admission-probe kernels. Kernel names
    # are checked against the checked-in CUDA code. cuBLAS names vary by release.
    kernel = "regex:.*(memory_kernel|tensor_kernel|latency_kernel|control_kernel).*"
    command = [ncu,"--csv","--page","raw","--target-processes","all",
               "--clock-control","none","--cache-control","none","--replay-mode","application",
               "--metrics",",".join(metrics)]
    if trial["workload"] != "gemm": command += ["--kernel-name", kernel,"--launch-skip","1","--launch-count","1"]
    return command + benchmark_command(executable,trial,device_index,profiling=True)


def parse_ncu_csv(text):
    """Preserve per-kernel rows; do not average ratios across dissimilar kernels."""
    lines = text.splitlines()
    start = next((i for i,l in enumerate(lines) if "Metric Name" in l and "Metric Value" in l),None)
    if start is None: raise ValueError("Nsight CSV missing Metric Name/Metric Value header")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows=[]
    for row in reader:
        if row.get("Metric Name") and row.get("Metric Value"):
            rows.append({"id":row.get("ID"),"kernel":row.get("Kernel Name"),
                         "metric":row["Metric Name"],"unit":row.get("Metric Unit"),
                         "value":row["Metric Value"]})
    if not rows: raise ValueError("Nsight CSV contained no counter values")
    return rows


def capture_profile(plan, trial_id, executable, output_dir, ncu="ncu", extra_metrics=()):
    matches=[t for t in plan["trials"] if t["trial_id"]==trial_id]
    if len(matches)!=1: raise ValueError("trial_id must identify exactly one planned trial")
    trial=matches[0]
    available=query_metrics(ncu)
    command=profile_command(ncu,executable,trial,plan["device"].get("device_index",0),available)
    # User may request architecture-specific L2 fabric counters after discovery.
    if extra_metrics:
        if any(m not in available for m in extra_metrics):
            raise ValueError("An extra metric is not advertised by this Nsight installation")
        i=command.index("--metrics")+1
        command[i] += ","+",".join(extra_metrics)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(command,capture_output=True,text=True,timeout=600)
    raw=output/(trial_id+".ncu.csv")
    raw.write_text(result.stdout,encoding="utf-8")
    (output/(trial_id+".ncu.stderr.txt")).write_text(result.stderr,encoding="utf-8")
    if result.returncode: raise RuntimeError(f"Nsight Compute failed; evidence saved under {output}")
    evidence={"schema_version":1,"condition_id":trial["condition_id"],
              "gpu_uuid":plan["device"]["uuid"],"command":command,
              "profile_requested_clocks":trial["clocks"],"profile_clocks_controlled":False,
              "workload":trial["workload"],"rows":parse_ncu_csv(result.stdout),
              "memory_target_verified":False,"locality":"unclassified",
              "notes":["Profiler replay changes execution; none of these readings are energy measurements.",
                       "No clock mutation during profile. Confirm actual clocks and hit rates before attaching verification.",
                       "Only validated L2 fabric counters and empirical SM/address maps support locality labels."]}
    path=output/(trial_id+".evidence.json")
    atomic_json(path,evidence)
    return {"evidence":str(path.resolve()),"raw_csv":str(raw.resolve())}


def attach_verification(record,evidence):
    """Attach an explicitly reviewed evidence manifest with exact condition identity.

    A manifest needs verification_notes and profile_clocks_verified=true. Neither
    footprint nor a high logical bandwidth is sufficient proof of cache residency.
    """
    if evidence.get("condition_id")!=record.get("condition_id") or evidence.get("gpu_uuid")!=record["config"]["gpu_uuid"]:
        raise ValueError("Profiler evidence condition/GPU does not match measured trial")
    if evidence.get("workload")!=record["workload"]:
        raise ValueError("Profiler workload differs from measured trial")
    verification_key="tensor_instructions_verified" if record["workload"] in ("tensor","gemm") else "memory_target_verified"
    if not evidence.get(verification_key) or not evidence.get("verification_notes") or not evidence.get("profile_clocks_verified"):
        raise ValueError("Evidence must explicitly verify target/actual clocks and explain counter-based verification")
    if not evidence.get("rows"): raise ValueError("Counter evidence cannot be empty")
    locality=evidence.get("locality","unclassified")
    if locality not in ("unclassified","local-heavy","remote-heavy","mixed"):
        raise ValueError("Use empirical locality labels local-heavy/remote-heavy/mixed/unclassified")
    if locality!="unclassified" and not evidence.get("locality_mapping_evidence"):
        raise ValueError("Locality labels require independently validated SM/address/fabric mapping evidence")
    updated=dict(record)
    updated["validation"]={verification_key:True,"locality":locality,
                           "profiler_evidence":evidence}
    return updated
