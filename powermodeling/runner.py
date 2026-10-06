"""Capture sustained benchmark phases and raw telemetry without profiling interference."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import shutil
import signal
import time

from .planner import benchmark_command


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def validate_trial_ids(records):
    """Require unique, portable filename identifiers before writing trial data."""
    seen = set()
    for index, record in enumerate(records):
        identifier = record.get("trial_id") if isinstance(record, dict) else None
        if (not isinstance(identifier, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", identifier) is None):
            raise ValueError(f"Unsafe or missing trial_id at record {index}: {identifier!r}")
        if identifier in seen:
            raise ValueError(f"Duplicate trial_id: {identifier}")
        seen.add(identifier)


def validate_plan_execution(plan):
    """Reject an incomplete required-clock study before GPU or file mutation.

    Diagnostic and legacy plans keep their explicit scope. A full energy study
    must retain every planned clock condition for each workload geometry; use
    the run limit for a partial diagnostic instead of deleting required arms.
    """
    coverage = plan.get("clock_sweep_coverage") or {}
    strict = plan.get("study_design") == "energy_sweep"
    reasons = coverage.get("requirement_reasons") or []
    if plan.get("execution_allowed") is False or (strict and plan.get("execution_allowed") is not True):
        detail = "; ".join(str(reason) for reason in reasons) or "required clock coverage is unverified"
        raise ValueError("Study clock requirements are incomplete; execution blocked: " + detail)
    if not strict:
        return
    if coverage.get("requirements_status") != "complete":
        raise ValueError("Study clock requirements must be complete before execution")
    step = coverage.get("requested_step_mhz")
    floor = coverage.get("graphics_min_mhz", 900)
    if (coverage.get("strategy") != "nearest_supported_graphics_grid"
            or type(step) is not int or step <= 0 or type(floor) is not int or floor <= 0):
        raise ValueError("Energy study requires a positive interval and minimum MHz for the supported graphics sweep")
    declared_policy = plan.get("clock_sweep_policy") or {}
    if declared_policy != {"graphics_step_mhz": step, "graphics_min_mhz": floor}:
        raise ValueError("Clock coverage differs from the declared interval/minimum policy; regenerate the plan")
    supported_memory = coverage.get("supported_memory_mhz") or []
    selected_memory = coverage.get("selected_memory_mhz") or []
    domains = coverage.get("memory_domains") or []
    if (not supported_memory or set(supported_memory) != set(selected_memory)
            or set(supported_memory) != {domain.get("memory_mhz") for domain in domains}):
        raise ValueError("Energy study must cover every advertised memory clock domain")
    default = coverage.get("advertised_default") or {}
    if default.get("status") != "included":
        raise ValueError("Required advertised default clock pair is missing or unverified")
    default_pair = (default.get("graphics_mhz"), default.get("memory_mhz"))
    if any(type(value) is not int or value <= 0 for value in default_pair):
        raise ValueError("Required advertised default clock pair is invalid")
    declared = set()
    for condition in coverage.get("clock_conditions") or []:
        pair = condition.get("clocks") or {}
        graphics, memory = pair.get("graphics_mhz"), pair.get("memory_mhz")
        if (graphics is None) != (memory is None) or (graphics is not None and
                any(type(value) is not int or value <= 0 for value in (graphics, memory))):
            raise ValueError("Invalid clock condition in study coverage")
        declared.add((graphics, memory))
    if default_pair not in declared or (None, None) not in declared:
        raise ValueError("Study coverage must include advertised default and incoming-policy reference")
    native_pairs = []
    for domain in domains:
        selected_graphics = {graphics for graphics, memory in declared if memory == domain.get("memory_mhz")}
        supported_graphics = domain.get("supported_graphics_mhz") or []
        eligible = [g for g in supported_graphics if g >= floor]
        if (not supported_graphics or (eligible and not selected_graphics)
                or not selected_graphics.issubset(set(supported_graphics))):
            raise ValueError("Study clock conditions must use supported pairs in every memory domain")
        if 1110 in domain.get("supported_graphics_mhz", []) and (1110, domain.get("memory_mhz")) not in declared:
            raise ValueError("Study coverage omitted a supported exact 1110 MHz condition")
        native_pairs.extend({"graphics_mhz": graphics, "memory_mhz": domain.get("memory_mhz")}
                            for graphics in supported_graphics)
    # Recompute the grid from advertised native pairs. Selected conditions and
    # plotting metadata can be edited together with trials; their agreement
    # alone cannot establish that the required sweep remains complete.
    from .planner import resolve_clock_plan
    pairs, rebuilt = resolve_clock_plan({"study_design": "energy_sweep", "clock_sweep": {
        "graphics_step_mhz": step, "graphics_min_mhz": floor, "all_memory_clocks": True,
        "required_graphics_mhz": coverage.get("required_graphics_mhz", [1110]),
        "include_advertised_default": True, "include_default_policy": True}}, {
        "supported_pairs": native_pairs,
        "default_applications_graphics_mhz": default_pair[0],
        "default_applications_memory_mhz": default_pair[1]})
    if rebuilt["execution_allowed"] is not True:
        raise ValueError("Required advertised default clock pair is not in the discovered supported clock domains")
    expected = {(pair["graphics_mhz"], pair["memory_mhz"]) for pair in pairs}
    if declared != expected:
        raise ValueError("Study clock coverage does not retain the required interval grid and evaluation endpoints; regenerate the full plan")
    by_geometry = {}
    for trial in plan.get("trials") or []:
        protocol = trial.get("treatment_protocol") or {}
        design = {key: value for key, value in protocol.items()
                  if key not in ("order", "phase_order", "order_balance_note")}
        signature = json.dumps({"workload": trial.get("workload"), "stage": trial.get("stage"),
                                "parameters": trial.get("parameters"), "treatment_design": design}, sort_keys=True)
        pair = trial.get("clocks") or {}
        by_geometry.setdefault(signature, set()).add((pair.get("graphics_mhz"), pair.get("memory_mhz")))
    if not by_geometry or any(actual != expected for actual in by_geometry.values()):
        raise ValueError("Study trials do not retain every required clock condition for each workload geometry; regenerate the full plan")


def describe_benchmark(executable, device_index=0):
    result = subprocess.run([str(executable), "--device", str(device_index), "--describe"],
                            capture_output=True, text=True, check=True, timeout=60)
    records = [json.loads(line, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
               for line in result.stdout.splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Benchmark describe must emit JSON objects")
    devices = [r for r in records if r.get("type") == "device"]
    if len(devices) != 1: raise ValueError("Benchmark describe did not emit exactly one device")
    device = devices[0]
    # Accept either common spelling; persist one stable schema downstream.
    device["l2_bytes"] = device.get("l2_bytes", device.get("l2_cache_bytes"))
    device["total_memory_bytes"] = device.get("total_memory_bytes", device.get("global_memory_bytes"))
    if not device.get("uuid"): raise ValueError("CUDA describe must provide GPU UUID")
    device["benchmark_sha256"] = binary_sha256(executable)
    return device


def _reject_json_constant(value):
    raise ValueError(f"Non-finite JSON constant {value} is not valid evidence")


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number {value} is not valid evidence")
    return number


def binary_sha256(executable):
    binary = Path(executable)
    if not binary.is_file():
        binary = Path(shutil.which(str(executable)) or str(executable))
    with binary.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_records(events, samples):
    phases = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Benchmark events must be JSON objects")
        if event.get("type") != "phase": continue
        name, kind = event.get("phase"), event.get("event")
        stamp = event.get("host_monotonic_ns")
        if (not isinstance(name, str) or not name or kind not in ("start", "end")
                or not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0):
            raise ValueError("Malformed benchmark phase event")
        phase = phases.setdefault(name, {})
        key = "start_s" if kind == "start" else "end_s"
        if key in phase: raise ValueError(f"Duplicate {name} {kind} event")
        phase[key] = stamp/1e9
    for phase in phases.values():
        start, end = phase.get("start_s"), phase.get("end_s")
        if start is None or end is None or end <= start:
            raise ValueError("Benchmark phase must have increasing start/end timestamps")
        if any(not isinstance(s, dict) or not isinstance(s.get("t_s"), (int, float))
               or not math.isfinite(s["t_s"]) for s in samples):
            raise ValueError("Telemetry sample must have a finite monotonic timestamp")
        inside = [i for i, s in enumerate(samples) if start <= s["t_s"] <= end]
        # Retain adjacent samples to interpolate boundaries; no extrapolated energy.
        if inside:
            lo, hi = max(0,inside[0]-1), min(len(samples),inside[-1]+2)
            phase["samples"] = samples[lo:hi]
        else: phase["samples"] = []
    return phases


_MPS_ENVIRONMENT_KEYS = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                         "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING")


def _mps_environment():
    return {key: os.environ[key] for key in _MPS_ENVIRONMENT_KEYS if key in os.environ}


def _sample_exclusivity(samples, allowed_pids=()):
    allowed = set(allowed_pids)
    foreign, unknown, mps_active = set(), set(), False
    for sample in samples:
        for field in ("compute_processes", "graphics_processes", "mps_compute_processes"):
            processes = sample.get(field)
            if processes is None and field == "mps_compute_processes":
                # Optional old-driver capability; physical compute inventory
                # still sees a foreign MPS server context when it is active.
                continue
            if not isinstance(processes, list):
                unknown.add(field)
                continue
            if field == "mps_compute_processes" and processes:
                mps_active = True
            for process in processes:
                pid = process.get("pid") if isinstance(process, dict) else process
                if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                    unknown.add(field)
                elif pid not in allowed:
                    foreign.add(pid)
    return {"other_compute_processes": sorted(foreign),
            "process_inventory_unknown": sorted(unknown),
            "mps_active": mps_active,
            "interference_detected": bool(foreign),
            "exclusivity_verified": not foreign and not unknown and not mps_active}


def assess_exclusive_device(device, allowed_pids=()):
    """Fail before GPU mutation if dedicated physical-device ownership is unproved.

    Returns the read-only NVML metadata/snapshot for persisted provenance. An
    advisory lock only coordinates this program, so inventories are required
    separately before every trial and again throughout measurement.
    """
    metadata = device.metadata()
    if not isinstance(metadata, dict) or metadata.get("uuid") != device.uuid or not str(device.uuid).startswith("GPU-"):
        raise RuntimeError("Physical GPU UUID could not be verified for telemetry")
    mig = metadata.get("mig_mode")
    if isinstance(mig, dict) and (mig.get("current") == 1 or mig.get("pending") == 1):
        raise RuntimeError("Disable current/pending MIG for physical-device power experiments")
    if "mig_mode" in metadata and mig is None:
        error = (metadata.get("errors") or {}).get("mig_mode") or {}
        if error.get("code") != 3 and error.get("type") not in ("NVMLError_NotSupported",):
            raise RuntimeError("MIG status unavailable; physical-device ownership cannot be verified")
    snapshot = metadata.get("sample")
    if not isinstance(snapshot, dict):
        raise RuntimeError("GPU process snapshot is unavailable")
    quality = _sample_exclusivity([snapshot], allowed_pids)
    if quality["process_inventory_unknown"]:
        raise RuntimeError(f"GPU process inventory unavailable: {quality['process_inventory_unknown']}; exclusivity cannot be verified")
    if quality["other_compute_processes"]:
        raise RuntimeError(f"Selected GPU has existing compute/graphics/MPS contexts: {quality['other_compute_processes']}")
    if quality["mps_active"]:
        raise RuntimeError("Selected GPU has active CUDA MPS contexts; dedicated physical-device ownership is required")
    if _mps_environment():
        raise RuntimeError("CUDA MPS environment is active; use a dedicated physical-device environment")
    if snapshot.get("fatal_errors"):
        raise RuntimeError(f"Fatal NVML device error: {snapshot['fatal_errors']}")
    metadata["exclusivity_assessment"] = quality
    return metadata


@contextmanager
def exclusive_device_lock(uuid, output_dir):
    """Linux advisory lock prevents this tool launching competing local experiments."""
    import fcntl
    path = Path("/tmp") / ("powermodeling-"+uuid.replace("/", "_")+".lock")
    with path.open("a+") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("Another powermodeling run owns this GPU") from exc
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


def _terminate_process_group(process):
    """Stop the benchmark and descendants before restoring the clock policy."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (AttributeError, PermissionError):
        process.kill()


def _trial_record(trial, cuda_device, command):
    return {
        "schema_version": 1, "trial_id": trial["trial_id"], "condition_id": trial["condition_id"],
        "repeat": trial["repeat"], "workload": trial["workload"], "status": "failed",
        "config": {**trial["parameters"], "gpu_uuid": cuda_device["uuid"],
                   "graphics_clock_mhz": trial["clocks"]["graphics_mhz"],
                   "memory_clock_mhz": trial["clocks"]["memory_mhz"],
                   "seconds": trial["seconds"], "warmup_seconds": trial["warmup_seconds"],
                   "idle_seconds": trial["idle_seconds"], "stage": trial["stage"],
                   "clock_selection_policy": trial.get("clock_policy", "fixed_explicit" if trial["clocks"]["graphics_mhz"] is not None else "incoming_policy_reference"),
                   "clock_selection_reasons": list(trial.get("clock_selection_reasons", [])),
                   "target_form_factor": cuda_device.get("target_form_factor"),
                   "form_factor_validation": cuda_device.get("form_factor_validation")},
        "benchmark": {}, "active_reference": {}, "treatment_protocol": {},
        "planned_treatment_protocol": trial.get("treatment_protocol"),
        "phases": {}, "samples": [], "events": [], "device": {},
        "cuda_device": cuda_device, "telemetry": {},
        "validation": {"memory_target_verified": False, "locality": "unclassified", "profiler_evidence": None},
        "quality": {"valid": False, "benchmark_exit_code": None, "runner_errors": [],
                    "benchmark_stderr": "", "benchmark_stdout": "", "measurement_pid": None},
        "provenance": {"utc": datetime.now(timezone.utc).isoformat(), "command": command,
                       "benchmark_sha256": cuda_device.get("benchmark_sha256"), "python": sys.version,
                       "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "mps_environment": _mps_environment(),
                       "declared_platform": {"target_form_factor": cuda_device.get("target_form_factor"),
                                             "form_factor_validation": cuda_device.get("form_factor_validation")}},
    }


def capture_trial(executable, trial, device, cuda_device, sample_interval_s=0.05):
    """Always preserve raw output/samples on capture failures, including interruption.

    Ordinary errors return a failed trial so run_plan can persist it and stop.
    KeyboardInterrupt/SystemExit keep cancellation semantics and carry the
    same record on ``exc.record`` for the outer clock context to save.
    """
    from .telemetry import Sampler
    command = benchmark_command(executable, trial, cuda_device.get("device_index", 0))
    record = _trial_record(trial, cuda_device, command)
    record["config"]["sample_interval_s"] = sample_interval_s
    quality = record["quality"]
    sampler, process, interrupted = None, None, None
    paired_requested = (trial.get("treatment_protocol") or {}).get("kind") == "paired_active_reference"
    arm_count = 2 if paired_requested else 1
    warmup_count = 3 if paired_requested else 1
    timeout = max(120, 5 * (arm_count * trial["seconds"] + warmup_count * trial["warmup_seconds"] + 2 * trial["idle_seconds"]))
    try:
        record["device"] = device.metadata()
        record["telemetry"] = record["device"].get("capabilities", {})
        sampler = Sampler(device, interval_s=sample_interval_s)
        sampler.start()
        quality["process_start_monotonic_s"] = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        quality["measurement_pid"] = process.pid
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            quality["runner_errors"].append(f"Benchmark exceeded wall-clock timeout {timeout}s")
            _terminate_process_group(process)
            stdout, stderr = process.communicate(timeout=10)
        quality["benchmark_stdout"], quality["benchmark_stderr"] = stdout, stderr
    except BaseException as exc:
        quality["runner_errors"].append(f"Capture failure: {type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            interrupted = exc
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    _terminate_process_group(process)
                    stdout, stderr = process.communicate(timeout=10)
                    quality["benchmark_stdout"], quality["benchmark_stderr"] = stdout, stderr
                quality["benchmark_exit_code"] = process.returncode
            except BaseException as exc:
                quality["runner_errors"].append(f"Benchmark shutdown failure: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception):
                    interrupted = exc
        if sampler is not None:
            try:
                record["samples"] = sampler.stop()
            except BaseException as exc:
                record["samples"] = list(getattr(sampler, "samples", []))
                quality["runner_errors"].append(f"Telemetry failure: {type(exc).__name__}: {exc}")
                quality["sampler_error"] = getattr(sampler, "error", None)
                quality["sampler_stopped"] = not bool(getattr(sampler, "_attached", False))
                if not isinstance(exc, Exception):
                    interrupted = exc
    for line in quality["benchmark_stdout"].splitlines():
        try:
            event = json.loads(line, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
            if not isinstance(event, dict):
                raise ValueError("Benchmark output must be JSON objects")
            record["events"].append(event)
        except (ValueError, TypeError) as exc:
            quality["runner_errors"].append(f"Invalid benchmark output: {exc}: {line[:200]}")
    results = [event for event in record["events"] if event.get("type") == "result"]
    if len(results) == 1:
        record["benchmark"] = results[0]
    else:
        quality["runner_errors"].append("Benchmark did not emit exactly one result")
    protocols = [event for event in record["events"] if event.get("type") == "treatment_protocol"]
    references = [event for event in record["events"] if event.get("type") == "active_reference_result"]
    if len(protocols) == 1:
        record["treatment_protocol"] = protocols[0]
    elif paired_requested or protocols:
        quality["runner_errors"].append("Benchmark did not emit exactly one treatment protocol")
    actual_protocol = record["treatment_protocol"]
    paired_actual = actual_protocol.get("kind") == "paired_active_reference"
    if paired_requested:
        planned = trial["treatment_protocol"]
        if not paired_actual or actual_protocol.get("order") != planned.get("order"):
            quality["runner_errors"].append("Actual treatment/reference protocol differed from the planned randomized order")
    if paired_actual and not paired_requested:
        quality["runner_errors"].append("Unexpected paired protocol outside the planned treatment/reference design")
    if paired_actual:
        if len(references) == 1:
            record["active_reference"] = references[0]
        else:
            quality["runner_errors"].append("Paired trial did not emit exactly one active-reference result")
    elif references:
        quality["runner_errors"].append("Unexpected active-reference result outside a paired protocol")
    devices = [event for event in record["events"] if event.get("type") == "device"]
    if len(devices) != 1 or devices[0].get("uuid") != cuda_device["uuid"] or cuda_device["uuid"] != device.uuid:
        quality["runner_errors"].append("Actual benchmark CUDA UUID did not match selected physical NVML GPU")
    try:
        record["phases"] = _phase_records(record["events"], record["samples"])
        if paired_actual:
            order = actual_protocol.get("order")
            if order not in ("AB", "BA"):
                raise ValueError("Paired protocol order must be AB or BA")
            arm_order = ["active_reference", "measure"] if order == "AB" else ["measure", "active_reference"]
            if actual_protocol.get("phase_order") != arm_order:
                raise ValueError("Paired protocol phase_order disagrees with AB/BA order")
            required_names = ["idle_pre"]
            for arm in arm_order:
                required_names.extend(["warmup_reference" if arm == "active_reference" else "warmup_treatment", arm])
            required_names.append("idle_post")
            for arm in arm_order:
                preparation = record["phases"]["warmup_reference" if arm == "active_reference" else "warmup_treatment"]
                if preparation["end_s"] > record["phases"][arm]["start_s"]:
                    raise ValueError("Arm warmup overlaps or follows its measurement")
                if preparation["start_s"] < record["phases"]["idle_pre"]["end_s"]:
                    raise ValueError("Arm warmup must occur after the initial idle interval")
            for key in ("same_process", "same_allocations", "same_clock_policy"):
                if actual_protocol.get(key) is not True:
                    raise ValueError(f"Paired protocol missing {key} assurance")
            reference = record["active_reference"]
            if reference.get("workload") != "control" or reference.get("reference_kind") != "issue_loop":
                raise ValueError("Paired reference must be the declared issue-loop control")
            if any(reference.get(key) != record["benchmark"].get(key) for key in ("blocks", "threads", "batch_launches", "iterations_per_launch")):
                raise ValueError("Paired reference launch geometry/batching/iterations differ from treatment")
            if reference.get("sanity", {}).get("requested_sm_coverage_complete") is not True:
                raise ValueError("Paired reference did not cover every requested SM")
            if actual_protocol.get("launch_geometry_matched") is not (trial["workload"] != "gemm"):
                raise ValueError("Paired launch-geometry assurance disagrees with workload")
        else:
            required_names = ["idle_pre", "measure", "idle_post"]
        required = [record["phases"][name] for name in required_names]
        if any(left["end_s"] > right["start_s"] for left, right in zip(required, required[1:])):
            raise ValueError("Benchmark idle/measure/reference phases overlap or are out of order")
    except (ValueError, KeyError, TypeError) as exc:
        quality["runner_errors"].append(f"Invalid benchmark phases: {exc}")
    quality.update(_sample_exclusivity(record["samples"], (quality["measurement_pid"],)))
    if not quality["exclusivity_verified"]:
        quality["runner_errors"].append("GPU exclusivity was not verified throughout capture")
    if not record["samples"]:
        quality["runner_errors"].append("No telemetry samples captured")
    if _mps_environment():
        quality["runner_errors"].append("CUDA MPS environment was active during measurement")
    quality["valid"] = quality["benchmark_exit_code"] == 0 and not quality["runner_errors"]
    record["status"] = "complete" if quality["valid"] else "failed"
    if interrupted is not None:
        interrupted.record = record
        raise interrupted
    return record


def run_plan(plan, executable, output_dir, cuda_device, apply_clocks=False,
             clock_method="applications", locked_restore=None, resume=False, limit=None):
    validate_plan_execution(plan)
    from .telemetry import NvmlDevice
    from .clocks import clock_context
    validate_trial_ids(plan["trials"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plan["device"]["uuid"] != cuda_device["uuid"]:
        raise ValueError("Plan GPU UUID differs from selected CUDA device; regenerate the plan")
    if plan["device"].get("benchmark_sha256") != cuda_device.get("benchmark_sha256"):
        raise ValueError("Benchmark binary differs from plan; regenerate the plan and use a new output directory")
    # Carry user-declared platform metadata while retaining freshly observed
    # CUDA UUID/name/capacities. SXM is a module form factor, not an HBM version.
    cuda_device = {**cuda_device, **{key: plan["device"][key] for key in
                   ("target_form_factor", "form_factor_validation") if key in plan["device"]}}
    trials = plan["trials"][:limit] if limit else plan["trials"]
    if not apply_clocks and any(any(value is not None for value in t["clocks"].values()) for t in trials):
        raise ValueError("Plan requests clock changes; pass --apply-clocks on an exclusively owned GPU")
    if clock_method not in ("applications", "locked"):
        raise ValueError("Clock method must be applications or locked")
    completed = failed = skipped = 0
    with exclusive_device_lock(cuda_device["uuid"],output_dir):
        device = NvmlDevice(uuid=cuda_device["uuid"])
        try:
            assess_exclusive_device(device)
            atomic_json(output_dir/"plan.json",plan)
            for i,trial in enumerate(trials):
                pair = trial["clocks"]
                controlled = any(pair.get(key) is not None for key in ("graphics_mhz", "memory_mhz"))
                policy = {"clock_method": clock_method if controlled else "uncontrolled",
                          "locked_restore": locked_restore if controlled and clock_method == "locked" else None}
                path = output_dir/"trials"/(trial["trial_id"]+".json")
                if path.exists():
                    old = json.loads(path.read_text())
                    if old.get("condition_id") != trial["condition_id"]:
                        raise ValueError("Existing trial fingerprint differs; choose a new output directory")
                    if (old.get("provenance") or {}).get("benchmark_sha256") != cuda_device.get("benchmark_sha256"):
                        raise ValueError("Existing trial was measured with a different binary; choose a new output directory")
                    if (old.get("config") or {}).get("gpu_uuid") != cuda_device["uuid"]:
                        raise ValueError("Existing trial GPU UUID differs; choose a new output directory")
                    previous_policy = {key: (old.get("config") or {}).get(key) for key in policy}
                    # JSON normalizes prior locked tuples to lists.
                    if json.dumps(previous_policy, sort_keys=True) != json.dumps(policy, sort_keys=True):
                        raise ValueError("Existing trial clock method/restoration policy differs; choose a new output directory")
                    if (old.get("config") or {}).get("sample_interval_s") != plan["sample_interval_s"]:
                        raise ValueError("Existing trial telemetry interval differs; choose a new output directory")
                    if resume and old.get("status") == "complete":
                        skipped += 1
                        continue
                    if not resume: raise FileExistsError(f"Trial exists: {path}; use --resume or a new output")
                print(f"[{i+1}/{len(trials)}] {trial['trial_id']} {trial['workload']} {trial['clocks']}",flush=True)
                record = None
                clock_record = None
                try:
                    # External processes may have arrived since the prior trial.
                    ownership = assess_exclusive_device(device)
                    expected_binary = cuda_device.get("benchmark_sha256")
                    if expected_binary is not None and binary_sha256(executable) != expected_binary:
                        raise ValueError("Benchmark binary changed since planning; refusing the trial")
                    with clock_context(device, graphics_mhz=pair["graphics_mhz"],memory_mhz=pair["memory_mhz"],
                                       method=clock_method,allow_mutation=apply_clocks,locked_restore=locked_restore) as clock_record:
                        record = capture_trial(executable,trial,device,cuda_device,plan["sample_interval_s"])
                        record["clock_policy"]=clock_record
                        record["config"].update(policy, clocks_controlled=controlled,
                                                gpu_uuid=cuda_device["uuid"], sample_interval_s=plan["sample_interval_s"])
                        record.setdefault("provenance", {})["benchmark_sha256"] = cuda_device.get("benchmark_sha256")
                        record["quality"]["pretrial_exclusivity"] = ownership["exclusivity_assessment"]
                        if record["samples"]:
                            record["quality"].update(_sample_exclusivity(record["samples"], (record["quality"]["measurement_pid"],)))
                            if not record["quality"]["exclusivity_verified"]:
                                record["status"] = "failed"
                                record["quality"]["valid"] = False
                    atomic_json(path,record)
                except BaseException as exc:
                    record = record or getattr(exc, "record", None) or _trial_record(trial, cuda_device, benchmark_command(executable, trial, cuda_device.get("device_index", 0)))
                    record["status"] = "failed"
                    record["quality"]["valid"] = False
                    record["config"].update(policy, clocks_controlled=controlled,
                                            gpu_uuid=cuda_device["uuid"], sample_interval_s=plan["sample_interval_s"])
                    policy_record = clock_record if clock_record is not None else getattr(exc, "clock_record", None)
                    if policy_record is not None:
                        record["clock_policy"] = policy_record
                    record["quality"]["runner_errors"].append(f"Clock/context failure: {type(exc).__name__}: {exc}")
                    atomic_json(path,record)
                    raise
                if record["status"] == "complete": completed += 1
                else:
                    failed += 1
                    print(record["quality"]["benchmark_stderr"], file=sys.stderr)
                    # Continuing after kernel errors can invalidate every later sample.
                    raise RuntimeError(f"Benchmark failed; raw evidence saved at {path}")
        finally:
            device.close()
    return {"completed":completed,"skipped":skipped,"failed":failed,"output":str(output_dir.resolve())}


def load_trials(input_path):
    path = Path(input_path)
    if path.is_file():
        value = json.loads(path.read_text())
        if isinstance(value,list): return value
        if "trials" in value and isinstance(value["trials"],list): return value["trials"]
        return [value]
    files = sorted((path/"trials").glob("*.json")) if (path/"trials").is_dir() else sorted(path.glob("*.json"))
    records = [json.loads(p.read_text()) for p in files]
    return [r for r in records if "phases" in r and "benchmark" in r]
