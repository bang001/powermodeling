"""Workflow mocks test orchestration only; none of these are GPU readings."""

from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from powermodeling.profile_session import capture_profile_session
from powermodeling.validation_workflow import validate_run


DEVICE={"uuid":"GPU-workflow-fixture","benchmark_sha256":"synthetic-binary","device_index":2}


def fixture_plan():
    return {"device":dict(DEVICE),"trials":[
        {"trial_id":f"c1-r{r}","condition_id":"c1","workload":"l2","parameters":{},
         "clocks":{"graphics_mhz":None,"memory_mhz":None}} for r in range(3)]}


class Device:
    closed=False
    def __init__(self,uuid): self.uuid=uuid
    def close(self): self.closed=True


class Sampler:
    def __init__(self,*args,**kwargs): self.samples=[{"t_s":1.0}]
    def start(self): return self
    def stop(self): return self.samples


@contextmanager
def fake_clocks(*args,**kwargs):
    record={"restored":False}
    try: yield record
    finally: record["restored"]=True


class ProfileWorkflowTests(unittest.TestCase):
    def test_profile_records_restoration_and_dynamic_sample_provider(self):
        def capture(plan,trial_id,executable,output,ncu,metrics,profile_context,policy):
            self.assertEqual(profile_context["selected_cuda_device"],DEVICE)
            self.assertEqual(profile_context["nvml_sample_provider"](),[{"t_s":1.0}])
            evidence=Path(output)/"evidence.json"
            evidence.write_text(json.dumps({"schema_version":1}))
            return {"evidence":str(evidence)}
        with tempfile.TemporaryDirectory() as temp, \
             patch("powermodeling.telemetry.NvmlDevice",Device), \
             patch("powermodeling.telemetry.Sampler",Sampler), \
             patch("powermodeling.runner.assess_exclusive_device",return_value={}), \
             patch("powermodeling.clocks.clock_context",fake_clocks), \
             patch("powermodeling.profiling.capture_profile",side_effect=capture):
            result=capture_profile_session(fixture_plan(),"c1-r0","bench",temp,DEVICE)
            evidence=json.loads(Path(result["evidence"]).read_text())
            self.assertTrue(evidence["clock_control"]["restored"])
            self.assertEqual(evidence["profile_session_status"],"complete")

    def test_profile_rejects_wrong_binary_and_missing_clock_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"binary"):
                capture_profile_session(fixture_plan(),"c1-r0","bench",temp,{**DEVICE,"benchmark_sha256":"changed"})
            plan=fixture_plan()
            plan["trials"][0]["clocks"]={"graphics_mhz":1000,"memory_mhz":1200}
            with self.assertRaisesRegex(ValueError,"apply-clocks"):
                capture_profile_session(plan,"c1-r0","bench",temp,DEVICE)

    def test_clock_restore_failure_invalidates_saved_profile(self):
        @contextmanager
        def failing_clocks(*args,**kwargs):
            record={"restored":None}
            yield record
            record["restored"]=False
            error=RuntimeError("restoration failed")
            error.clock_record=record
            raise error
        def capture(plan,trial_id,executable,output,*args,**kwargs):
            path=Path(output)/(trial_id+".evidence.json")
            path.write_text('{}')
            return {"evidence":str(path)}
        with tempfile.TemporaryDirectory() as temp, \
             patch("powermodeling.telemetry.NvmlDevice",Device), \
             patch("powermodeling.telemetry.Sampler",Sampler), \
             patch("powermodeling.runner.assess_exclusive_device",return_value={}), \
             patch("powermodeling.clocks.clock_context",failing_clocks), \
             patch("powermodeling.profiling.capture_profile",side_effect=capture):
            with self.assertRaisesRegex(RuntimeError,"restoration failed"):
                capture_profile_session(fixture_plan(),"c1-r0","bench",temp,DEVICE)
            evidence=json.loads((Path(temp)/"c1-r0.evidence.json").read_text())
            self.assertEqual(evidence["profile_session_status"],"failed")
            self.assertFalse(evidence["clock_control"]["restored"])
            self.assertTrue((Path(temp)/"c1-r0.session-failure.json").is_file())

    def test_cleanup_preserves_primary_error_and_saves_failure(self):
        class FailingSampler(Sampler):
            def stop(self): raise RuntimeError("sampler cleanup failed")
        with tempfile.TemporaryDirectory() as temp, \
             patch("powermodeling.telemetry.NvmlDevice",Device), \
             patch("powermodeling.telemetry.Sampler",FailingSampler), \
             patch("powermodeling.runner.assess_exclusive_device",return_value={}), \
             patch("powermodeling.clocks.clock_context",fake_clocks), \
             patch("powermodeling.profiling.capture_profile",side_effect=RuntimeError("primary ncu failure")):
            with self.assertRaisesRegex(RuntimeError,"primary ncu failure"):
                capture_profile_session(fixture_plan(),"c1-r0","bench",temp,DEVICE)
            failure=json.loads((Path(temp)/"c1-r0.session-failure.json").read_text())
            self.assertEqual(failure["cleanup_errors"],["sampler cleanup failed"])
            self.assertTrue(failure["clock_control"]["restored"])

    def test_existing_profile_and_duplicate_records_rejected_before_gpu_use(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch("powermodeling.telemetry.NvmlDevice",side_effect=AssertionError("must not initialize GPU")):
            root=Path(temp)
            (root/"c1-r0.evidence.json").write_text('{}')
            with self.assertRaises(FileExistsError):
                capture_profile_session(fixture_plan(),"c1-r0","bench",root,DEVICE)
            (root/"raw.json").write_text(json.dumps([{"trial_id":"same"},{"trial_id":"same"}]))
            with self.assertRaisesRegex(ValueError,"uplicate"):
                validate_run(fixture_plan(),root/"raw.json",root/"out",root/"profiles","bench",DEVICE)

    def test_partial_ncu_failure_evidence_receives_clock_restoration(self):
        def capture(plan,trial_id,executable,output,*args,**kwargs):
            evidence=Path(output)/(trial_id+".failed.evidence.json")
            evidence.write_text(json.dumps({"partial_stdout":"partial application output"}))
            error=RuntimeError("ncu interrupted")
            error.evidence_path=str(evidence)
            raise error
        with tempfile.TemporaryDirectory() as temp, \
             patch("powermodeling.telemetry.NvmlDevice",Device), \
             patch("powermodeling.telemetry.Sampler",Sampler), \
             patch("powermodeling.runner.assess_exclusive_device",return_value={}), \
             patch("powermodeling.clocks.clock_context",fake_clocks), \
             patch("powermodeling.profiling.capture_profile",side_effect=capture):
            with self.assertRaisesRegex(RuntimeError,"ncu interrupted"):
                capture_profile_session(fixture_plan(),"c1-r0","bench",temp,DEVICE)
            evidence=json.loads((Path(temp)/"c1-r0.failed.evidence.json").read_text())
            self.assertEqual(evidence["partial_stdout"],"partial application output")
            self.assertTrue(evidence["clock_control"]["restored"])
            self.assertEqual(evidence["profile_session_status"],"failed")

    def test_batch_profiles_each_condition_once_preserves_other_records(self):
        plan=fixture_plan()
        records=[{"trial_id":t["trial_id"],"condition_id":"c1","workload":"l2","status":"complete",
                  "config":{"gpu_uuid":DEVICE["uuid"]},"provenance":{"benchmark_sha256":DEVICE["benchmark_sha256"]},
                  "phases":{},"benchmark":{}} for t in plan["trials"]]
        records.append({**records[0],"trial_id":"failed","condition_id":"failed","status":"failed"})
        def attach(record,evidence,policy): return {**record,"validation":{"status":"inconclusive"}}
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp)
            source=root/"raw.json"
            source.write_text(json.dumps(records))
            evidence=root/"evidence.json"
            evidence.write_text('{}')
            with patch("powermodeling.profile_session.capture_profile_session",return_value={"evidence":str(evidence)}) as capture, \
                 patch("powermodeling.profiling.attach_verification",side_effect=attach):
                result=validate_run(plan,source,root/"validated",root/"profiles","bench",DEVICE)
            self.assertEqual(capture.call_count,1)
            self.assertEqual(result["energy_records_preserved"],4)
            self.assertEqual(len(list((root/"validated"/"trials").glob('*.json'))),4)
            failed=json.loads((root/"validated"/"trials"/"failed.json").read_text())
            self.assertNotIn("validation",failed)


if __name__=="__main__": unittest.main()
