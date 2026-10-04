"""Capture sustained benchmark phases and raw telemetry without profiling interference."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import shutil
import time

from .planner import benchmark_command


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def describe_benchmark(executable, device_index=0):
    result = subprocess.run([str(executable), "--device", str(device_index), "--describe"],
                            capture_output=True, text=True, check=True, timeout=60)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    devices = [r for r in records if r.get("type") == "device"]
    if len(devices) != 1: raise ValueError("Benchmark describe did not emit exactly one device")
    device = devices[0]
    # Accept either common spelling; persist one stable schema downstream.
    device["l2_bytes"] = device.get("l2_bytes", device.get("l2_cache_bytes"))
    device["total_memory_bytes"] = device.get("total_memory_bytes", device.get("global_memory_bytes"))
    if not device.get("uuid"): raise ValueError("CUDA describe must provide GPU UUID")
    binary=Path(executable)
    if not binary.is_file(): binary=Path(shutil.which(str(executable)) or str(executable))
    with binary.open("rb") as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    device["benchmark_sha256"]=digest.hexdigest()
    return device


def _phase_records(events, samples):
    phases = {}
    for event in events:
        if event.get("type") != "phase": continue
        name, kind = event.get("phase"), event.get("event")
        stamp = event.get("host_monotonic_ns")
        if not name or kind not in ("start", "end") or stamp is None:
            raise ValueError("Malformed benchmark phase event")
        phase = phases.setdefault(name, {})
        key = "start_s" if kind == "start" else "end_s"
        if key in phase: raise ValueError(f"Duplicate {name} {kind} event")
        phase[key] = stamp/1e9
    for phase in phases.values():
        start, end = phase.get("start_s"), phase.get("end_s")
        if start is None or end is None or end <= start:
            phase["samples"] = []
            continue
        inside = [i for i, s in enumerate(samples) if start <= s["t_s"] <= end]
        # Retain adjacent samples to interpolate boundaries; no extrapolated energy.
        if inside:
            lo, hi = max(0,inside[0]-1), min(len(samples),inside[-1]+2)
            phase["samples"] = samples[lo:hi]
        else: phase["samples"] = []
    return phases


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


def capture_trial(executable, trial, device, cuda_device, sample_interval_s=0.05):
    from .telemetry import Sampler
    command = benchmark_command(executable, trial, cuda_device.get("device_index", 0))
    metadata = device.metadata()
    sampler = Sampler(device, interval_s=sample_interval_s)
    sampler.start()
    events, parse_errors = [], []
    process = None
    # Bound wall-clock time even when a CUDA call hangs; do not wait forever in readline.
    timeout = max(120, 5*(trial["seconds"]+trial["warmup_seconds"]+2*trial["idle_seconds"]))
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try: stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            parse_errors.append(f"Benchmark exceeded wall-clock timeout {timeout}s")
        for line in stdout.splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: parse_errors.append(f"Non-JSON benchmark output: {line[:200]}")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        samples = sampler.stop()
    results = [e for e in events if e.get("type") == "result"]
    success = process is not None and process.returncode == 0 and len(results) == 1 and not parse_errors
    benchmark = results[0] if len(results) == 1 else {}
    config = {**trial["parameters"], "gpu_uuid":cuda_device["uuid"],
              "graphics_clock_mhz":trial["clocks"]["graphics_mhz"],
              "memory_clock_mhz":trial["clocks"]["memory_mhz"],
              "seconds":trial["seconds"], "warmup_seconds":trial["warmup_seconds"],
              "idle_seconds":trial["idle_seconds"], "stage":trial["stage"]}
    return {"schema_version":1, "trial_id":trial["trial_id"], "condition_id":trial["condition_id"],
            "repeat":trial["repeat"], "workload":trial["workload"], "config":config,
            "status":"complete" if success else "failed", "benchmark":benchmark,
            "phases":_phase_records(events,samples), "samples":samples,
            "device":metadata, "cuda_device":cuda_device,
            "telemetry":metadata.get("capabilities", {}), "events":events,
            "validation":{"memory_target_verified":False,
                          "locality":"unclassified", "profiler_evidence":None},
            "quality":{"benchmark_exit_code":process.returncode if process else None,
                       "runner_errors":parse_errors, "benchmark_stderr":stderr if process else "",
                       "measurement_pid":process.pid if process else None},
            "provenance":{"utc":datetime.now(timezone.utc).isoformat(), "command":command,
                          "benchmark_sha256":cuda_device.get("benchmark_sha256"),
                          "python":sys.version,"platform":platform.platform(),
                          "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "mps_environment":{k:os.environ[k] for k in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") if k in os.environ}}}


def run_plan(plan, executable, output_dir, cuda_device, apply_clocks=False,
             clock_method="applications", locked_restore=None, resume=False, limit=None):
    from .telemetry import NvmlDevice
    from .clocks import clock_context
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plan["device"]["uuid"] != cuda_device["uuid"]:
        raise ValueError("Plan GPU UUID differs from selected CUDA device; regenerate the plan")
    if plan["device"].get("benchmark_sha256") != cuda_device.get("benchmark_sha256"):
        raise ValueError("Benchmark binary differs from plan; regenerate the plan and use a new output directory")
    trials = plan["trials"][:limit] if limit else plan["trials"]
    if not apply_clocks and any(t["clocks"]["graphics_mhz"] is not None for t in trials):
        raise ValueError("Plan requests clock changes; pass --apply-clocks on an exclusively owned GPU")
    completed = failed = skipped = 0
    with exclusive_device_lock(cuda_device["uuid"],output_dir):
        device = NvmlDevice(uuid=cuda_device["uuid"])
        try:
            initial=device.metadata()
            mig=initial.get("mig_mode")
            if isinstance(mig,dict) and mig.get("current")==1:
                raise RuntimeError("Disable MIG for physical-device power experiments before running")
            snapshot=initial.get("sample",{})
            if snapshot.get("compute_processes") is None:
                raise RuntimeError("Compute process inventory unavailable; GPU exclusivity cannot be checked")
            if snapshot.get("compute_processes") or snapshot.get("graphics_processes"):
                raise RuntimeError("Selected GPU has existing compute/graphics contexts; use an idle dedicated GPU")
            atomic_json(output_dir/"plan.json",plan)
            for i,trial in enumerate(trials):
                path = output_dir/"trials"/(trial["trial_id"]+".json")
                if path.exists():
                    old = json.loads(path.read_text())
                    if old.get("condition_id") != trial["condition_id"]:
                        raise ValueError("Existing trial fingerprint differs; choose a new output directory")
                    if (old.get("provenance") or {}).get("benchmark_sha256") != cuda_device.get("benchmark_sha256"):
                        raise ValueError("Existing trial was measured with a different binary; choose a new output directory")
                    if resume and old.get("status") == "complete":
                        skipped += 1
                        continue
                    if not resume: raise FileExistsError(f"Trial exists: {path}; use --resume or a new output")
                print(f"[{i+1}/{len(trials)}] {trial['trial_id']} {trial['workload']} {trial['clocks']}",flush=True)
                pair = trial["clocks"]
                record=None
                try:
                    with clock_context(device, graphics_mhz=pair["graphics_mhz"],memory_mhz=pair["memory_mhz"],
                                       method=clock_method,allow_mutation=apply_clocks,locked_restore=locked_restore) as clock_record:
                        record = capture_trial(executable,trial,device,cuda_device,plan["sample_interval_s"])
                        record["clock_policy"]=clock_record
                        record["config"]["clock_method"] = clock_method if pair["graphics_mhz"] is not None else "uncontrolled"
                        record["config"]["clocks_controlled"] = pair["graphics_mhz"] is not None
                        record["config"]["locked_restore"] = locked_restore
                        own={record["quality"]["measurement_pid"],os.getpid()}
                        foreign=set()
                        for sample in record["samples"]:
                            for field in ("compute_processes","graphics_processes"):
                                for process in sample.get(field) or []:
                                    pid=process.get("pid") if isinstance(process,dict) else process
                                    if pid not in own: foreign.add(pid)
                        record["quality"]["other_compute_processes"]=sorted(foreign)
                        record["quality"]["interference_detected"]=bool(foreign)
                    atomic_json(path,record)
                except BaseException as exc:
                    if record is not None:
                        record["status"]="failed"
                        record["quality"]["runner_errors"].append(f"Clock/context failure: {exc}")
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
