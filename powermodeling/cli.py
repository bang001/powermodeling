"""CLI with hardware-free planning and explicit clock experiment ownership."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .planner import expand_plan, benchmark_command
from .profiles import architecture
from .runner import atomic_json, describe_benchmark, load_trials, run_plan, validate_plan_execution


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parser():
    p=argparse.ArgumentParser(description="Sustained V100/A100/H100 throughput and operational incremental energy")
    commands=p.add_subparsers(dest="command",required=True)
    def hardware(c):
        c.add_argument("--bench",default="build/powerbench")
        c.add_argument("--device",type=int,default=0,help="CUDA ordinal, mapped to NVML by exact UUID")
    discover=commands.add_parser("discover",help="Read-only CUDA/NVML capabilities and supported clock pairs")
    hardware(discover)
    discover.add_argument("--output",help="Save discovery JSON")
    plan=commands.add_parser("plan",help="Expand geometry/clock grids without modifying device policy")
    hardware(plan)
    plan.add_argument("--config",required=True)
    plan.add_argument("--device-json",help="Offline discovery JSON or CUDA device JSON")
    plan.add_argument("--clocks-json",help="Offline supported clock discovery JSON")
    plan.add_argument("--stage")
    plan.add_argument("--sfu-sass-evidence",help="Register SFU SASS certificate from tools/check_sfu_sass.py; retained for binary-bound validation")
    plan.add_argument("--output",required=True)
    run=commands.add_parser("run",help="Run a saved plan; raw telemetry is written after every trial")
    hardware(run)
    run.add_argument("--plan",required=True)
    run.add_argument("--output",required=True)
    run.add_argument("--apply-clocks",action="store_true",help="Apply the saved clock sweep; requires exclusive ownership of the GPU")
    run.add_argument("--clock-method",choices=("applications","locked"),default="applications")
    policy=run.add_mutually_exclusive_group()
    policy.add_argument("--locked-restore",help="JSON file giving known preexisting graphics/memory ranges or null for unlocked")
    policy.add_argument("--clock-reset-on-exit",action="store_true",help="Declare graphics AND memory clocks were previously unlocked; reset these policies on exit")
    run.add_argument("--resume",action="store_true")
    run.add_argument("--limit",type=int,help="Run at most N trials; incomplete repeat groups will not be selected")
    analyze=commands.add_parser("analyze",help="Integrate telemetry and select efficient conditions near observed throughput saturation")
    analyze.add_argument("--input",required=True)
    analyze.add_argument("--output",required=True)
    analyze.add_argument("--throughput-fraction",type=float,default=0.95)
    analyze.add_argument("--hbm-bandwidth-fraction",type=float,default=0.80,
                         help="HBM-only minimum sustained logical bandwidth / theory from measured memory clock and CUDA bus width; observed-peak fraction is diagnostic for HBM")
    analyze.add_argument("--plan",help="Original plan for missing-trial and required-clock evaluation; defaults to INPUT/plan.json when present")
    analyze.add_argument("--evaluation-policy",help="JSON overrides for plateau tolerance, minimum resource levels and maximum relative CI width")
    analyze.add_argument("--plots",action="store_true",help="Write energy/throughput and clock plots (requires matplotlib)")
    fit=commands.add_parser("fit",help="Fit an identifiable empirical incremental-power model from prepared feature rows")
    fit.add_argument("--input",required=True)
    fit.add_argument("--features",required=True,help="Comma-separated measured rate features")
    fit.add_argument("--target",default="incremental_power_w")
    fit.add_argument("--output",required=True)
    profile=commands.add_parser("profile",help="Separate short Nsight Compute run; counter validation is not an energy measurement")
    hardware(profile)
    profile.add_argument("--plan",required=True)
    profile.add_argument("--trial-id",required=True)
    profile.add_argument("--ncu",default="ncu")
    profile.add_argument("--extra-metrics",default="",help="Additional discovered L2 fabric metrics, comma-separated")
    profile.add_argument("--output",required=True)
    profile.add_argument("--print-command",action="store_true")
    profile.add_argument("--policy",help="JSON file overriding documented counter-assessment thresholds")
    profile.add_argument("--apply-clocks",action="store_true",help="Use the planned clock pair during counter validation")
    profile.add_argument("--clock-method",choices=("applications","locked"),default="applications")
    profile_restore=profile.add_mutually_exclusive_group()
    profile_restore.add_argument("--locked-restore",help="Known preexisting graphics/memory locked ranges JSON")
    profile_restore.add_argument("--clock-reset-on-exit",action="store_true",help="Declare both clock domains were unlocked before profiling")
    evaluate=commands.add_parser("evaluate-profile",help="Compute pass/fail/inconclusive from raw ncu evidence without GPU access")
    evaluate.add_argument("--evidence",required=True)
    evaluate.add_argument("--policy",help="JSON overrides for documented counter thresholds")
    evaluate.add_argument("--output",required=True)
    batch=commands.add_parser("validate-run",help="Run ncu once per measured condition and attach computed judgments to every repeat")
    hardware(batch)
    batch.add_argument("--plan",required=True)
    batch.add_argument("--input",required=True)
    batch.add_argument("--output",required=True)
    batch.add_argument("--profiles-dir",required=True)
    batch.add_argument("--ncu",default="ncu")
    batch.add_argument("--extra-metrics",default="")
    batch.add_argument("--policy")
    batch.add_argument("--apply-clocks",action="store_true")
    batch.add_argument("--clock-method",choices=("applications","locked"),default="applications")
    batch_restore=batch.add_mutually_exclusive_group()
    batch_restore.add_argument("--locked-restore")
    batch_restore.add_argument("--clock-reset-on-exit",action="store_true")
    batch.add_argument("--limit-conditions",type=int,help="Profile first N unique conditions; output explicitly records partial coverage")
    verify=commands.add_parser("attach-verification",help="Attach reviewed counter/locality evidence to matching measured trials")
    verify.add_argument("--input",required=True)
    verify.add_argument("--evidence",required=True)
    verify.add_argument("--output",required=True,help="New directory; original raw measurements stay intact")
    verify.add_argument("--policy",help="JSON overrides for documented counter thresholds")
    return p


def _discover(args):
    from .telemetry import NvmlDevice
    from .clocks import discover
    cuda=describe_benchmark(args.bench,args.device)
    cuda["device_index"]=args.device
    cuda["architecture_profile"]=architecture(cuda)
    device=NvmlDevice(uuid=cuda["uuid"])
    try: return {"schema_version":1,"cuda_device":cuda,"nvml":device.metadata(),"clocks":discover(device)}
    finally: device.close()


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=="discover":
            result=_discover(args)
            if args.output: atomic_json(args.output,result)
        elif args.command=="plan":
            if args.device_json:
                info=read_json(args.device_json)
                cuda=info.get("cuda_device",info)
                clocks=read_json(args.clocks_json) if args.clocks_json else info.get("clocks",{})
            else:
                info=_discover(args)
                cuda,clocks=info["cuda_device"],info["clocks"]
            architecture(cuda)
            result=expand_plan(read_json(args.config),cuda,clocks,args.stage)
            if args.sfu_sass_evidence:
                certificate = read_json(args.sfu_sass_evidence)
                if certificate.get("kind") != "sfu_register_sass" or certificate.get("benchmark_sha256") != cuda.get("benchmark_sha256"):
                    raise ValueError("SFU SASS certificate must bind to this discovered benchmark binary")
                result["sfu_sass_evidence"] = certificate
            atomic_json(args.output,result)
            result={"output":str(Path(args.output).resolve()),"trials":len(result["trials"]),
                    "estimated_minimum_hours":result["estimated_minimum_seconds"]/3600,
                    "study_design": result["study_design"], "execution_allowed": result["execution_allowed"],
                    "requirements_status": result["clock_sweep_coverage"]["requirements_status"],
                    "unsupported_experiments": result["unsupported_experiments"],
                    "requirement_reasons": result["clock_sweep_coverage"]["requirement_reasons"],
                    "sweep_dimensions": result["sweep_dimensions"]}
        elif args.command=="run":
            if args.limit is not None and args.limit<=0: raise ValueError("--limit must be positive")
            restore=(read_json(args.locked_restore) if args.locked_restore else
                     {"graphics":None,"memory":None} if args.clock_reset_on_exit else None)
            plan=read_json(args.plan)
            validate_plan_execution(plan)
            cuda=describe_benchmark(args.bench,args.device)
            cuda["device_index"]=args.device
            result=run_plan(plan,args.bench,args.output,cuda,args.apply_clocks,
                            args.clock_method,restore,args.resume,args.limit)
        elif args.command=="analyze":
            from .analysis import summarize, write_summary
            records=load_trials(args.input)
            if not records: raise ValueError("No raw measured trial records found")
            plan_path=Path(args.plan) if args.plan else Path(args.input)/"plan.json"
            evaluation_plan=read_json(plan_path) if args.plan or plan_path.is_file() else None
            result=summarize(records,throughput_fraction=args.throughput_fraction,plan=evaluation_plan,
                             hbm_bandwidth_fraction=args.hbm_bandwidth_fraction,
                             evaluation_policy=read_json(args.evaluation_policy) if args.evaluation_policy else None)
            paths=write_summary(result,args.output)
            if args.plots:
                from .reporting import write_plots
                paths.update(write_plots(result,args.output))
            result={"files":paths,"trial_count":len(records)}
        elif args.command=="fit":
            from .model import fit_model
            rows=read_json(args.input)
            result=fit_model(rows,features=[s.strip() for s in args.features.split(",") if s.strip()],target=args.target)
            atomic_json(args.output,result)
        elif args.command=="profile":
            from .profiling import profile_command
            plan=read_json(args.plan)
            if args.print_command:
                trial=next((t for t in plan["trials"] if t["trial_id"]==args.trial_id),None)
                if not trial: raise ValueError("Unknown trial ID")
                result={"command":profile_command(args.ncu,args.bench,trial,args.device)}
            else:
                selected=describe_benchmark(args.bench,args.device)
                if selected["uuid"]!=plan["device"]["uuid"]: raise ValueError("Profile device UUID differs from plan")
                from .profile_session import capture_profile_session
                selected["device_index"]=args.device
                restore=(read_json(args.locked_restore) if args.locked_restore else
                         {"graphics":None,"memory":None} if args.clock_reset_on_exit else None)
                result=capture_profile_session(plan,args.trial_id,args.bench,args.output,selected,args.ncu,
                                       tuple(x.strip() for x in args.extra_metrics.split(",") if x.strip()),
                                       read_json(args.policy) if args.policy else None,
                                       args.apply_clocks,args.clock_method,restore)
        elif args.command=="evaluate-profile":
            from .validation import assess_profile
            result=assess_profile(read_json(args.evidence),read_json(args.policy) if args.policy else None)
            atomic_json(args.output,result)
        elif args.command=="validate-run":
            from .validation_workflow import validate_run
            selected=describe_benchmark(args.bench,args.device)
            selected["device_index"]=args.device
            restore=(read_json(args.locked_restore) if args.locked_restore else
                     {"graphics":None,"memory":None} if args.clock_reset_on_exit else None)
            result=validate_run(read_json(args.plan),args.input,args.output,args.profiles_dir,args.bench,selected,
                                args.ncu,tuple(x.strip() for x in args.extra_metrics.split(",") if x.strip()),
                                read_json(args.policy) if args.policy else None,args.apply_clocks,
                                args.clock_method,restore,args.limit_conditions)
        elif args.command=="attach-verification":
            from .profiling import attach_verification
            evidence=read_json(args.evidence)
            records=load_trials(args.input)
            from .runner import validate_trial_ids
            validate_trial_ids(records)
            matching=[r for r in records if r.get("condition_id")==evidence.get("condition_id")]
            if not matching: raise ValueError("No measured trials match evidence condition_id")
            output=Path(args.output)
            # Preserve unmatched measurements so assessment does not silently
            # change the population used to estimate maximum observed throughput.
            updated=[]
            for record in records:
                updated.append(attach_verification(record,evidence,read_json(args.policy) if args.policy else None)
                               if record.get("condition_id")==evidence.get("condition_id") else record)
            for record in updated:
                path=output/"trials"/(record["trial_id"]+".json")
                if path.exists(): raise FileExistsError(f"Output already exists: {path}")
            for record in updated:
                atomic_json(output/"trials"/(record["trial_id"]+".json"),record)
            result={"assessed_trials":len(matching),"preserved_trials":len(records)-len(matching),"output":str(output.resolve())}
        print(json.dumps(result,indent=2,allow_nan=False))
        if args.command=="fit" and result.get("status")=="rejected": return 2
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"powermodeling: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__": raise SystemExit(main())
