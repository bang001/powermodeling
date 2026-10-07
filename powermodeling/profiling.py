"""Separate Nsight Compute evidence, with raw numeric target admission."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

from .planner import benchmark_command
from .runner import atomic_json
from .sfu import SFU_WORKLOADS, CONTRACT_FIELDS as SFU_CONTRACT_FIELDS
from .validation import (DURATION, DRAM_READ, DRAM_WRITE, L1_HITS, L1_MISSES,
                         L1_REQUESTS, L1_SECTORS, L2_READ, L2_READ_HITS, L2_WRITE,
                         LOCAL_LOAD, LOCAL_STORE, SM_HZ, TENSOR_ACTIVITY,
                         TENSOR_INSTRUCTIONS, SFU_INSTRUCTIONS, SFU_ACTIVITY, assess_profile, validate_evidence)

# Exact operation-specific counters are preferred. Architecture/release discovery
# determines support; absent counters are retained as unknown in admission.
METRICS = list(dict.fromkeys([DRAM_READ, DRAM_WRITE, DURATION, SM_HZ,
    "dram__cycles_elapsed.avg.per_second", L2_READ, L2_READ_HITS, L2_WRITE,
    L1_SECTORS, L1_HITS, L1_MISSES, L1_REQUESTS, LOCAL_LOAD, LOCAL_STORE,
    *TENSOR_INSTRUCTIONS, *TENSOR_ACTIVITY, *SFU_INSTRUCTIONS, *SFU_ACTIVITY,
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct"]))


def _environment():
    # Never inherit a locale-specific decimal separator from the host shell.
    return {**os.environ, "LC_ALL": "C", "LANG": "C"}


def available_metric_names(text):
    if isinstance(text, (set, list, tuple)): return set(text)
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*__[a-zA-Z0-9_.]+", text or ""))


def query_metrics(ncu="ncu", device_index=0):
    result = subprocess.run([ncu, "--devices", str(device_index), "--query-metrics", "--query-metrics-mode", "all", "--csv"],
                            check=True, capture_output=True, text=True, timeout=60, env=_environment())
    return result.stdout


def query_version(ncu="ncu"):
    result = subprocess.run([ncu, "--version"], check=True, capture_output=True, text=True, timeout=60, env=_environment())
    return result.stdout.strip()


def _version_release(text):
    match = re.search(r"(?:Version|version)\s+(20\d{2})\.(\d+)(?:\.(\d+))?", text or "")
    return tuple(int(x or 0) for x in match.groups()) if match else None


def _volta_device(device):
    capability = device.get("compute_capability")
    return (device.get("compute_capability_major") == 7 and device.get("compute_capability_minor", 0) == 0) or capability in ([7, 0], (7, 0)) or device.get("cc") == "7.0"


def profile_command(ncu, executable, trial, device_index=0, available=None, log_file=None, report_path=None):
    advertised = available_metric_names(available) if available is not None else set(METRICS)
    metrics = [m for m in METRICS if m in advertised]
    if not metrics: raise ValueError("None of the validation counters are available")
    command = [ncu, "--devices", str(device_index), "--csv", "--print-units", "base", "--page", "raw", "--target-processes", "all",
               "--clock-control", "none", "--cache-control", "none", "--replay-mode", "application",
               "--profile-from-start", "off", "--metrics", ",".join(metrics)]
    if log_file is not None: command += ["--log-file", str(log_file)]
    if report_path is not None: command += ["--export", str(report_path), "--force-overwrite"]
    if trial["workload"] != "gemm":
        command += ["--kernel-name", "regex:.*(memory_kernel|tensor_kernel|latency_kernel|control_kernel|pointwise_nonlinear_kernel|row_nonlinear_kernel|sfu_register_kernel).*", "--launch-count", "1"]
    return command + benchmark_command(executable, trial, device_index, profiling=True)


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_SPACED = re.compile(r"^[+-]?\d{1,3}(?: \d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def normalize_metric_value(value, unit):
    """Return (finite base-unit number, canonical unit, error), never zero-fill.

    Nsight uses SI (1000) K/M/G byte prefixes. KiB/MiB are explicitly binary.
    Ambiguous comma-decimal strings, instance lists, N/A and infinities are unknown.
    """
    if isinstance(value, bool) or value is None: return None, None, "missing_or_non_numeric_counter"
    raw = str(value).strip().replace("\u00a0", " ").replace("\u202f", " ")
    if raw.endswith("%") and str(unit).strip() in ("%", "percent"): raw = raw[:-1].strip()
    if _THOUSANDS.fullmatch(raw): raw = raw.replace(",", "")
    elif _SPACED.fullmatch(raw): raw = raw.replace(" ", "")
    if not _NUMBER.fullmatch(raw): return None, None, "missing_or_ambiguous_counter"
    number = float(raw)
    if not math.isfinite(number): return None, None, "non_finite_counter"
    raw_unit = str(unit or "").strip().replace("µ", "u").replace("μ", "u")
    aliases = {"": (1, "count"), "%": (1, "%"), "percent": (1, "%"),
               "byte": (1, "byte"), "bytes": (1, "byte"), "B": (1, "byte"),
               "Kbyte": (1e3, "byte"), "Mbyte": (1e6, "byte"), "Gbyte": (1e9, "byte"),
               "KB": (1e3, "byte"), "MB": (1e6, "byte"), "GB": (1e9, "byte"),
               "KiB": (1024, "byte"), "MiB": (1024**2, "byte"), "GiB": (1024**3, "byte"),
               "second": (1, "second"), "s": (1, "second"), "msecond": (1e-3, "second"), "ms": (1e-3, "second"),
               "usecond": (1e-6, "second"), "us": (1e-6, "second"), "microsecond": (1e-6, "second"),
               "nsecond": (1e-9, "second"), "ns": (1e-9, "second"), "nanosecond": (1e-9, "second"),
               "cycle/second": (1, "cycle/second"), "cycle/s": (1, "cycle/second"),
               "cycle/nsecond": (1e9, "cycle/second"), "cycle/usecond": (1e6, "cycle/second"),
               "Hz": (1, "cycle/second"), "KHz": (1e3, "cycle/second"), "MHz": (1e6, "cycle/second"), "GHz": (1e9, "cycle/second"),
               "sector": (1, "sector"), "request": (1, "request"), "inst": (1, "inst"),
               "instruction": (1, "inst"), "cycle": (1, "cycle"), "count": (1, "count")}
    if raw_unit.endswith(("/second", "/s")) and raw_unit not in aliases:
        suffix = "/second" if raw_unit.endswith("/second") else "/s"
        base = raw_unit[:-len(suffix)]
        if base in aliases:
            factor, canonical = aliases[base]
            normalized = number * factor
            return (normalized, canonical + "/second", None) if math.isfinite(normalized) else (None, canonical + "/second", "non_finite_counter")
    if raw_unit not in aliases: return None, raw_unit, "unsupported_counter_unit"
    factor, canonical = aliases[raw_unit]
    normalized = number * factor
    return (normalized, canonical, None) if math.isfinite(normalized) else (None, canonical, "non_finite_counter")


def parse_ncu_csv(text):
    """Preserve per-launch raw values, normalize units, and expose unknowns."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Metric Name" in line and "Metric Value" in line), None)
    if start is None: raise ValueError("Nsight CSV missing Metric Name/Metric Value header")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows = []
    for row in reader:
        metric, value = row.get("Metric Name"), row.get("Metric Value")
        if not metric or metric == "Metric Name": continue
        numeric, canonical, error = normalize_metric_value(value, row.get("Metric Unit"))
        rows.append({"id": row.get("ID"), "kernel": row.get("Kernel Name"), "device": row.get("Device"),
                     "metric": metric, "unit": row.get("Metric Unit"), "value": value,
                     "numeric_value": numeric, "normalized_unit": canonical, "parse_error": error})
    if not rows: raise ValueError("Nsight CSV contained no counter values")
    return rows


def _binary_hash(executable):
    path = Path(executable)
    if not path.is_file(): path = Path(shutil.which(str(executable)) or str(executable))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _profile_events(text):
    events = []
    for line in text.splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(event, dict): events.append(event)
    return events


def _profile_context(context, events):
    context = dict(context or {})
    provider = context.pop("nvml_sample_provider", None)
    samples = list(provider()) if callable(provider) else context.pop("nvml_samples", [])
    intervals, start = [], None
    for event in events:
        if event.get("type") != "phase" or event.get("phase") != "measure": continue
        stamp = event.get("host_monotonic_ns")
        if not isinstance(stamp, (int, float)): continue
        if event.get("event") == "start": start = stamp / 1e9
        elif event.get("event") == "end" and start is not None:
            intervals.append({"start_s": start, "end_s": stamp / 1e9})
            start = None
    context["measure_intervals"] = intervals
    if samples:
        context["profile_active_nvml_samples"] = [s for s in samples if isinstance(s, dict) and any(p["start_s"] <= s.get("t_s", -1) <= p["end_s"] for p in intervals)]
    return context


def _output_text(value):
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace")
    return value or ""


def _run_profile_process(command, timeout=600):
    """Own one process group; stop its descendants before restoring clocks.

    The cleanup target is solely the group created by start_new_session=True.
    On interruption/timeout partial stdout/stderr are carried on the exception.
    Reaping is bounded even if a native profiler call cannot complete promptly.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True, env=_environment())
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as exc:
        stdout, stderr = _output_text(getattr(exc, "output", None)), _output_text(getattr(exc, "stderr", None))
        cleanup = {"owned_process_group": process.pid, "group_stop_requested": False,
                   "leader_reaped": False, "errors": []}
        try:
            os.killpg(process.pid, signal.SIGKILL)
            cleanup["group_stop_requested"] = True
        except ProcessLookupError:
            # The owned group is already gone; no unrelated PID is targeted.
            cleanup["group_stop_requested"] = True
        except BaseException as stop_error:
            cleanup["errors"].append("process_group_stop: " + repr(stop_error))
            # Never fall back to a broad GPU/process search or kill arbitrary PIDs.
        try:
            recovered_stdout, recovered_stderr = process.communicate(timeout=10)
            stdout = _output_text(recovered_stdout) or stdout
            stderr = _output_text(recovered_stderr) or stderr
            cleanup["leader_reaped"] = process.poll() is not None
        except BaseException as reap_error:
            stdout = _output_text(getattr(reap_error, "output", None)) or stdout
            stderr = _output_text(getattr(reap_error, "stderr", None)) or stderr
            cleanup["errors"].append("process_group_reap: " + repr(reap_error))
            try:
                process.wait(timeout=2)
                cleanup["leader_reaped"] = process.poll() is not None
            except BaseException as wait_error:
                cleanup["errors"].append("process_group_wait: " + repr(wait_error))
        exc.profile_capture = {"stdout": stdout, "stderr": stderr,
                               "returncode": process.poll(), "cleanup": cleanup,
                               "failure_type": type(exc).__name__}
        raise


def _persist_profile_failure(output, trial, plan, command, binary_hash, profile_context, policy, raw, exc):
    """Keep reviewable raw data and a failed manifest, including cancellation."""
    details = getattr(exc, "profile_capture", {})
    stdout, stderr = details.get("stdout", ""), details.get("stderr", "")
    trial_id = trial["trial_id"]
    (output / (trial_id + ".application.jsonl")).write_text(stdout, encoding="utf-8")
    (output / (trial_id + ".ncu.stderr.txt")).write_text(stderr, encoding="utf-8")
    events = _profile_events(stdout)
    devices = [event for event in events if event.get("type") == "device"]
    rows, csv_error = [], None
    if raw.is_file():
        try: rows = parse_ncu_csv(raw.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error: csv_error = str(error)
    failure = {"schema_version": 2, "condition_id": trial["condition_id"], "trial_id": trial_id,
               "gpu_uuid": plan["device"]["uuid"], "workload": trial["workload"], "command": command,
               "profile_session_status": "failed", "profile_session_error": repr(exc),
               "profile_capture_failure": details, "raw_csv_parse_error": csv_error,
               "profile_context": _profile_context(profile_context, events), "profile_benchmark": {},
               "profile_provenance": {"benchmark_sha256": binary_hash, "parameters": trial["parameters"],
                   "requested_clocks": trial["clocks"], "observed_device_records": devices,
                   "deterministic_application_replay": False},
               "rows": rows, "locality": "unclassified",
               "notes": ["Failed or interrupted profiler capture; partial data cannot be admitted as verified evidence."]}
    failure["assessment"] = assess_profile(failure, policy)
    failure["validation_policy"] = failure["assessment"]["policy"]
    path = output / (trial_id + ".evidence.json")
    atomic_json(path, failure)
    exc.evidence_path = str(path.resolve())
    return path


def capture_profile(plan, trial_id, executable, output_dir, ncu="ncu", extra_metrics=(), profile_context=None, policy=None, export_report=True):
    matches = [t for t in plan["trials"] if t["trial_id"] == trial_id]
    if len(matches) != 1: raise ValueError("trial_id must identify exactly one planned trial")
    trial = matches[0]
    binary_hash = _binary_hash(executable)
    if binary_hash != plan["device"].get("benchmark_sha256"):
        raise ValueError("Profiler executable differs from planned benchmark; regenerate plan")
    ordinal = plan["device"].get("device_index", plan["device"].get("cuda_ordinal", 0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        version_text = query_version(ncu)
        (output / (trial_id + ".ncu.version.txt")).write_text(version_text, encoding="utf-8")
        version_release = _version_release(version_text)
        if _volta_device(plan["device"]) and version_release is not None and version_release >= (2025, 3, 0):
            raise ValueError("Nsight Compute 2025.3 and later removed Volta/GV100 support; profile V100, A100 and H100 consistently with Nsight Compute 2025.2.x (pass its executable via --ncu)")
        available = available_metric_names(query_metrics(ncu, ordinal))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        diagnostic = str(exc) + "\n" + str(getattr(exc, "stdout", "") or "") + "\n" + str(getattr(exc, "stderr", "") or "")
        (output / (trial_id + ".ncu.discovery.error.txt")).write_text(diagnostic, encoding="utf-8")
        raise RuntimeError(f"Nsight version/metric discovery failed for CUDA device {ordinal}; diagnostics saved under {output}. V100 requires a Volta-capable Nsight release such as 2025.2.x") from exc
    raw = output / (trial_id + ".ncu.csv")
    report = output / (trial_id + ".ncu-rep") if export_report else None
    # A previous log must not masquerade as a successful fresh capture.
    if raw.exists(): raw.unlink()
    command = profile_command(ncu, executable, trial, ordinal, available, raw.resolve(), report.resolve() if report else None)
    if extra_metrics:
        if any(m not in available for m in extra_metrics):
            raise ValueError("An extra metric is not advertised by this selected Nsight device")
        index = command.index("--metrics") + 1
        command[index] = ",".join(dict.fromkeys(command[index].split(",") + list(extra_metrics)))
    try:
        result = _run_profile_process(command, timeout=600)
    except BaseException as exc:
        _persist_profile_failure(output, trial, plan, command, binary_hash, profile_context, policy, raw, exc)
        raise
    (output / (trial_id + ".application.jsonl")).write_text(result.stdout, encoding="utf-8")
    (output / (trial_id + ".ncu.stderr.txt")).write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        exc = RuntimeError(f"Nsight Compute failed; evidence saved under {output}")
        exc.profile_capture = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        _persist_profile_failure(output, trial, plan, command, binary_hash, profile_context, policy, raw, exc)
        raise exc
    if _binary_hash(executable) != binary_hash:
        raise ValueError("Benchmark executable changed during profiling; evidence cannot bind to the plan")
    if not raw.is_file(): raise RuntimeError("Nsight did not create requested raw CSV log; profiler stdout is never treated as counter CSV")
    events = _profile_events(result.stdout)
    benchmarks = [e for e in events if e.get("type") == "result"]
    devices = [e for e in events if e.get("type") == "device"]
    uuids = {d.get("uuid") for d in devices}
    observed_uuid = next(iter(uuids)) if len(uuids) == 1 else None
    if observed_uuid and observed_uuid != plan["device"]["uuid"]:
        raise ValueError("Profiled application CUDA UUID differs from plan")
    # Application replay emits one result per pass. Require exact deterministic
    # counter-relevant metadata/payload, allowing elapsed timing to differ.
    determinism_fields = ("workload", "access", "blocks", "threads", "admitted_blocks", "iterations_per_launch", "working_set_bytes", "stride_elements", "offset_bytes", "tensor_accumulators", "gemm_m", "gemm_n", "gemm_k", "logical_bytes", "operations", "kernel_launches", "paired_reference_context_allocated", "row_width", "elements", "row_evaluations", "math_implementation", "input_precision", "rms_epsilon", "affine_gamma", "nonlinear_input_distribution", "kernel_implementation_version", "memory_accesses_per_thread_iteration", "grid_mode", "input_elements", "block_completion_count_source")
    if trial["workload"] in SFU_WORKLOADS:
        determinism_fields += (*SFU_CONTRACT_FIELDS, "sfu_instructions")
    deterministic = bool(benchmarks) and all(all(item.get(k) == benchmarks[0].get(k) for k in determinism_fields) for item in benchmarks)
    context = _profile_context(profile_context, events)
    evidence = {"schema_version": 2, "condition_id": trial["condition_id"], "trial_id": trial_id,
                "gpu_uuid": plan["device"]["uuid"], "command": command, "workload": trial["workload"],
                "ncu_version": version_text, "ncu_release": list(version_release) if version_release else None,
                "profile_requested_clocks": trial["clocks"], "profile_context": context,
                "profile_benchmark": benchmarks[0] if deterministic else {},
                "profile_provenance": {"observed_gpu_uuid": observed_uuid, "benchmark_sha256": binary_hash,
                    "parameters": trial["parameters"], "requested_clocks": trial["clocks"],
                    "ncu_version": version_text, "ncu_release": list(version_release) if version_release else None,
                    "cuda_ordinal": ordinal, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "deterministic_application_replay": deterministic, "application_pass_results": len(benchmarks),
                    "profile_region": "cudaProfilerStart/Stop measured phase only",
                    "observed_device_records": devices},
                "available_metrics": sorted(available), "requested_metrics": command[command.index("--metrics") + 1].split(","),
                "unsupported_metrics": [m for m in METRICS if m not in available],
                "rows": parse_ncu_csv(raw.read_text(encoding="utf-8")), "locality": "unclassified",
                "notes": ["Profiler replay readings are not energy measurements.",
                          "Admission thresholds are configurable project policies, not architecture guarantees.",
                          "Broad aggregate hit rates do not prove operation-specific cache residency.",
                          "Locality requires independent empirical SM/address maps and validated fabric counters."]}
    if trial["workload"] in SFU_WORKLOADS:
        certificate = plan.get("sfu_sass_evidence")
        evidence["sfu_sass_evidence"] = certificate
        if certificate is not None and devices and benchmarks:
            from .sfu_sass import select_certificate
            device = devices[0]
            cc = device.get("cc")
            if cc is None and type(device.get("compute_capability_major")) is int:
                cc = str(device["compute_capability_major"]) + "." + str(device.get("compute_capability_minor", 0))
            try:
                # Keep only the raw target/control proof needed by this profile;
                # a full multi-architecture certificate need not be copied per repeat.
                evidence["sfu_sass_evidence"] = select_certificate(certificate, cc, trial["workload"], benchmarks[0].get("sfu_chains"))
            except ValueError as error:
                # Preserve invalid evidence for an explicit failed assessment.
                evidence["sfu_sass_selection_error"] = str(error)
    evidence["assessment"] = assess_profile(evidence, policy)
    evidence["validation_policy"] = evidence["assessment"]["policy"]
    path = output / (trial_id + ".evidence.json")
    atomic_json(path, evidence)
    return {"evidence": str(path.resolve()), "raw_csv": str(raw.resolve()),
            "native_report": str(report.resolve()) if report else None, "assessment": evidence["assessment"]}


def attach_verification(record, evidence, policy=None):
    """Always recompute; manual booleans and stored assessments cannot verify."""
    if not isinstance(evidence, dict): raise ValueError("Profiler evidence must be a numeric manifest")
    if evidence.get("condition_id") != record.get("condition_id") or evidence.get("gpu_uuid") != (record.get("config") or {}).get("gpu_uuid"):
        raise ValueError("Profiler evidence condition/GPU does not match measured trial")
    if evidence.get("workload") != record.get("workload"):
        raise ValueError("Profiler workload differs from measured trial")
    locality = evidence.get("locality", "unclassified")
    if locality not in ("unclassified", "local-heavy", "remote-heavy", "mixed"):
        raise ValueError("Use empirical locality labels local-heavy/remote-heavy/mixed/unclassified")
    # The automatic assessor never assigns a topology label from address offsets
    # or broad hits. Such evidence remains independently reviewable metadata.
    if locality != "unclassified" and not evidence.get("locality_mapping_evidence"):
        raise ValueError("Locality labels require independently validated SM/address/fabric mapping evidence")
    assessment = validate_evidence(record, evidence, policy)
    verified = assessment["suitable_verified"]
    updated = dict(record)
    updated["validation"] = {"memory_target_verified": verified and record["workload"] in ("l1", "l2", "l2_latency", "hbm"),
                             "tensor_instructions_verified": verified and record["workload"] in ("tensor", "gemm"),
                             "sfu_instructions_verified": verified and record["workload"] in SFU_WORKLOADS,
                             "suitable_verified": verified, "status": assessment["status"], "assessment": assessment,
                             "locality": locality if verified else "unclassified", "profiler_evidence": evidence}
    override = evidence.get("manual_override")
    if override:
        if not isinstance(override, dict) or not override.get("reason"):
            raise ValueError("Manual override must include an explicit reason")
        updated["validation"]["manual_override"] = override
        updated["validation"]["manual_override_note"] = "Human review metadata cannot promote automatic verified admission"
    return updated
