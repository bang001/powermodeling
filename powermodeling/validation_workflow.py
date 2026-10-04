"""Validate each measured condition once and preserve the complete energy population."""

import json
from pathlib import Path

from .runner import atomic_json, load_trials, validate_trial_ids


def validate_run(plan, input_path, output_dir, profiles_dir, executable, cuda_device,
                 ncu="ncu", extra_metrics=(), policy=None, apply_clocks=False,
                 clock_method="applications", locked_restore=None, limit_conditions=None):
    from .profile_session import capture_profile_session
    from .profiling import attach_verification

    records=load_trials(input_path)
    if not records:
        raise ValueError("No measured trial records found")
    validate_trial_ids(records)
    validate_trial_ids(plan["trials"])
    source=Path(input_path).resolve()
    output=Path(output_dir).resolve()
    if source==output:
        raise ValueError("Choose a separate output directory to preserve original energy records")
    if (output/"trials").exists() and any((output/"trials").glob("*.json")):
        raise FileExistsError("Validation output already contains trials; choose a new output directory")
    planned={trial["trial_id"]:trial for trial in plan["trials"]}
    conditions={}
    for record in records:
        if record.get("status") != "complete": continue
        trial=planned.get(record["trial_id"])
        if not trial or trial["condition_id"] != record.get("condition_id"):
            raise ValueError("A measured trial is absent from or differs from the supplied plan")
        if record["config"]["gpu_uuid"] != cuda_device["uuid"]:
            raise ValueError("Measured GPU UUID differs from selected profiling device")
        if (record.get("provenance") or {}).get("benchmark_sha256") != cuda_device.get("benchmark_sha256"):
            raise ValueError("Measured binary differs from profiling binary; counter evidence cannot be transferred")
        conditions.setdefault(trial["condition_id"],trial)
    selected=list(conditions.values())
    if limit_conditions is not None:
        if limit_conditions<=0: raise ValueError("limit_conditions must be positive")
        selected=selected[:limit_conditions]
    if not selected:
        raise ValueError("No completed measured conditions are eligible for profiling")
    report={"schema_version":1,"status":"in_progress","conditions_total":len(conditions),
            "conditions_requested":len(selected),"conditions":[],"energy_records_preserved":len(records)}
    output.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        for record in records:
            atomic_json(output/"trials"/(record["trial_id"]+".json"),record)
        atomic_json(output/"validation-run.json",report)
    checkpoint()
    for index,trial in enumerate(selected):
        print(f"[ncu {index+1}/{len(selected)}] condition {trial['condition_id']} {trial['workload']}",flush=True)
        try:
            result=capture_profile_session(plan,trial["trial_id"],executable,profiles_dir,cuda_device,
                                           ncu,extra_metrics,policy,apply_clocks,clock_method,locked_restore)
            evidence=json.loads(Path(result["evidence"]).read_text(encoding="utf-8"))
            records=[attach_verification(record,evidence,policy)
                     if record.get("condition_id")==trial["condition_id"] else record for record in records]
            assessed=[record["validation"] for record in records if record.get("condition_id")==trial["condition_id"]]
            report["conditions"].append({"condition_id":trial["condition_id"],"trial_id":trial["trial_id"],
                                         "evidence":result["evidence"],"assessments":assessed})
            checkpoint()
        except BaseException as exc:
            report["status"]="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
            report["error"]={"condition_id":trial["condition_id"],"message":str(exc)}
            checkpoint()
            raise
    report["status"]="complete" if len(selected)==len(conditions) else "partial"
    checkpoint()
    return {"output":str(output),"validation_report":str(output/"validation-run.json"),
            "profiled_conditions":len(selected),"total_conditions":len(conditions),
            "energy_records_preserved":len(records),"status":report["status"]}
