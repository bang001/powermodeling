"""Run counter validation under the same owned clock policy as energy trials."""

from pathlib import Path

from .runner import atomic_json, exclusive_device_lock, validate_trial_ids


def capture_profile_session(plan, trial_id, executable, output_dir, cuda_device,
                            ncu="ncu", extra_metrics=(), policy=None,
                            apply_clocks=False, clock_method="applications", locked_restore=None):
    from .clocks import clock_context
    from .profiling import capture_profile
    from .telemetry import NvmlDevice, Sampler
    from .runner import assess_exclusive_device
    import json

    validate_trial_ids(plan["trials"])
    matches=[trial for trial in plan["trials"] if trial["trial_id"]==trial_id]
    if len(matches)!=1:
        raise ValueError("trial_id must identify exactly one planned trial")
    trial=matches[0]
    if plan["device"]["uuid"] != cuda_device["uuid"]:
        raise ValueError("Profile GPU UUID differs from plan")
    if plan["device"].get("benchmark_sha256") != cuda_device.get("benchmark_sha256"):
        raise ValueError("Profile benchmark binary differs from plan; regenerate the plan")
    pair=trial["clocks"]
    if not apply_clocks and any(value is not None for value in pair.values()):
        raise ValueError("Profile plan requests fixed clocks; pass --apply-clocks on an exclusively owned GPU")
    plan={**plan,"device":dict(cuda_device)}
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    expected_evidence=output/(trial_id+".evidence.json")
    if expected_evidence.exists():
        raise FileExistsError("Profile evidence already exists; choose a new profiles directory")
    with exclusive_device_lock(cuda_device["uuid"], output):
        device=NvmlDevice(uuid=cuda_device["uuid"])
        sampler=None
        evidence_path=None
        clock_record=None
        primary_error=None
        def save_failure(exc, cleanup_errors=None):
            failure={"schema_version":1,"trial_id":trial_id,"condition_id":trial["condition_id"],
                     "profile_session_status":"failed","profile_session_error":str(exc),
                     "clock_control":getattr(exc,"clock_record",None) or clock_record,
                     "cleanup_errors":cleanup_errors or []}
            atomic_json(output/(trial_id+".session-failure.json"),failure)
            if evidence_path is not None and evidence_path.is_file():
                evidence=json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence.update({k:v for k,v in failure.items() if k not in ("schema_version","trial_id","condition_id")})
                atomic_json(evidence_path,evidence)
        try:
            assess_exclusive_device(device)
            with clock_context(device,graphics_mhz=pair["graphics_mhz"],memory_mhz=pair["memory_mhz"],
                               method=clock_method,allow_mutation=apply_clocks,locked_restore=locked_restore) as clock_record:
                sampler=Sampler(device,interval_s=0.02)
                sampler.start()
                context={"clock_control":clock_record,"nvml_sample_provider":lambda:sampler.samples,
                         "gpu_uuid":device.uuid,"selected_cuda_device":dict(cuda_device)}
                result=capture_profile(plan,trial_id,executable,output,ncu,extra_metrics,
                                       profile_context=context,policy=policy)
                evidence_path=Path(result["evidence"])
                sampler.stop()
                sampler=None
            # Record restoration after context exit, just as the energy runner does.
            evidence=json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["clock_control"]=clock_record
            evidence["profile_session_status"]="complete"
            atomic_json(evidence_path,evidence)
            return result
        except BaseException as exc:
            primary_error=exc
            if evidence_path is None and getattr(exc,"evidence_path",None):
                evidence_path=Path(exc.evidence_path)
            save_failure(exc)
            raise
        finally:
            cleanup_errors=[]
            if sampler is not None:
                try: sampler.stop()
                except BaseException as exc: cleanup_errors.append(str(exc))
            try: device.close()
            except BaseException as exc: cleanup_errors.append(str(exc))
            if cleanup_errors:
                error=primary_error or RuntimeError("Profile cleanup failed: "+"; ".join(cleanup_errors))
                save_failure(error,cleanup_errors)
                if primary_error is None: raise error
