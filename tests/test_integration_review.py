"""CPU-only integration checks for planning, capture, resume and CLI routing."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from powermodeling.cli import main
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.profiling import parse_ncu_csv, profile_command, attach_verification
from powermodeling.runner import capture_trial, load_trials, run_plan


DEVICE = {
    "uuid": "GPU-integration-fixture", "device_index": 3,
    "compute_capability_major": 9, "compute_capability_minor": 0,
    "sm_count": 132, "l2_bytes": 50 * 1024**2,
    "total_memory_bytes": 80 * 1024**3,
}
CLOCKS = {"supported_pairs": [
    {"graphics_mhz": graphics, "memory_mhz": memory}
    for graphics in (600, 1200, 1500) for memory in (1000, 1500)
]}


def small_plan(controlled=False):
    return expand_plan({
        "clock_pairs": [{"graphics_mhz": 1200 if controlled else None,
                         "memory_mhz": 1000 if controlled else None}],
        "experiments": [{"workload": "tensor", "parameters": {
            "blocks": "sm_count * 2", "threads": 256, "tensor_accumulators": 4,
        }}],
    }, dict(DEVICE))


class FakeDevice:
    closed = False

    def __init__(self, **kwargs):
        self.uuid = kwargs["uuid"]

    def metadata(self):
        return {"uuid": self.uuid, "sample": {
            "compute_processes": [], "graphics_processes": []},
            "capabilities": {"power_scope": "fixture"}}

    def close(self):
        self.closed = True


class FakeSampler:
    def __init__(self, device, interval_s):
        self.device, self.interval_s = device, interval_s

    def start(self):
        return self

    def stop(self):
        return [{"t_s": index / 2, "power_w": 50.0,
                 "compute_processes":[],"graphics_processes":[]} for index in range(49)]


class FakeProcess:
    pid, returncode = 42, 0

    def __init__(self, command, **kwargs):
        self.command = command

    def communicate(self, timeout):
        events = [{"type":"device",**DEVICE}]
        for name, start, end in (("idle_pre", 0, 6), ("measure", 6, 18), ("idle_post", 18, 24)):
            for kind, timestamp in (("start", start), ("end", end)):
                events.append({"type": "phase", "phase": name, "event": kind,
                               "host_monotonic_ns": timestamp * 10**9})
        events.append({"type": "result", "duration_s": 12,
                       "host_duration_s": 12, "operations": 12e12})
        return "\n".join(json.dumps(value) for value in events), ""

    def poll(self):
        return self.returncode


class IntegrationReviewTests(unittest.TestCase):
    def test_all_shipped_configs_expand_for_each_architecture(self):
        root = Path(__file__).resolve().parents[1]
        for sm, l2 in ((80, 6 * 1024**2), (108, 40 * 1024**2), (132, 50 * 1024**2)):
            device = {**DEVICE, "sm_count": sm, "l2_bytes": l2}
            for source in (root / "configs").glob("*.json"):
                with self.subTest(sm=sm, config=source.name):
                    plan = expand_plan(json.loads(source.read_text()), device, CLOCKS)
                    self.assertGreater(len(plan["trials"]), 0)
                    self.assertEqual(len({t["trial_id"] for t in plan["trials"]}), len(plan["trials"]))
                    self.assertTrue(all(t["seconds"] >= 10 for t in plan["trials"]))
                    self.assertEqual(plan, expand_plan(json.loads(source.read_text()), device, CLOCKS))

    def test_expression_dependencies_and_sm_filter_argument(self):
        plan = expand_plan({"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                            "experiments": [{"workload": "l1", "parameters": {
                                "blocks": "sm_count * 2", "working_set_bytes": "blocks * 16384",
                                "sm_ids": [0, 2]}}]}, dict(DEVICE))
        trial = plan["trials"][0]
        self.assertEqual(trial["parameters"]["working_set_bytes"], 264 * 16384)
        command = benchmark_command("bench", trial, device_index=3)
        self.assertEqual(command[command.index("--device") + 1], "3")
        self.assertEqual(command[command.index("--sm-ids") + 1], "0,2")

    def test_capture_uses_uuid_and_worker_phase_timestamps(self):
        trial = small_plan()["trials"][0]
        with patch("powermodeling.telemetry.Sampler", FakeSampler), \
                patch("powermodeling.runner.subprocess.Popen", FakeProcess), \
                patch("powermodeling.runner.platform.platform", return_value="fixture-platform"):
            record = capture_trial("bench", trial, FakeDevice(uuid=DEVICE["uuid"]), dict(DEVICE))
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["quality"]["measurement_pid"], 42)
        self.assertEqual(record["phases"]["measure"]["start_s"], 6)
        self.assertEqual(record["phases"]["measure"]["end_s"], 18)
        self.assertEqual(record["config"]["gpu_uuid"], DEVICE["uuid"])
        self.assertFalse(record["validation"]["memory_target_verified"])
        self.assertEqual(record["provenance"]["command"][2], "3")

    def test_run_writes_completed_record_after_restoration_and_resumes(self):
        plan = small_plan()

        @contextmanager
        def clocks(device, **kwargs):
            info = {"restored": None}
            try:
                yield info
            finally:
                info["restored"] = True

        def capture(executable, trial, device, cuda_device, interval):
            return {"trial_id": trial["trial_id"], "condition_id": trial["condition_id"],
                    "status": "complete", "workload": "tensor", "config": {},
                    "phases": {}, "benchmark": {}, "samples": [],
                    "quality": {"measurement_pid": 42, "runner_errors": [], "benchmark_stderr": ""}}

        with tempfile.TemporaryDirectory() as directory, \
                patch("powermodeling.telemetry.NvmlDevice", FakeDevice), \
                patch("powermodeling.clocks.clock_context", clocks), \
                patch("powermodeling.runner.capture_trial", side_effect=capture) as mock_capture, \
                redirect_stdout(io.StringIO()):
            result = run_plan(plan, "bench", directory, dict(DEVICE), limit=1)
            self.assertEqual(result["completed"], 1)
            record = load_trials(directory)[0]
            self.assertTrue(record["clock_policy"]["restored"])
            result = run_plan(plan, "bench", directory, dict(DEVICE), resume=True, limit=1)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(mock_capture.call_count, 1)

    def test_clock_plan_requires_explicit_application_and_uuid_match(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "apply-clocks"):
                run_plan(small_plan(True), "bench", directory, dict(DEVICE))
            with self.assertRaisesRegex(ValueError, "UUID"):
                run_plan(small_plan(), "bench", directory, {**DEVICE, "uuid": "GPU-other"})

    def test_ncu_rows_preserve_kernel_and_counter_identity(self):
        source = ('==PROF== Connected\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                  '"0","memory_kernel","dram__bytes_read.sum","byte","1,024"\n'
                  '"1","memory_kernel","lts__t_sector_hit_rate.pct","%","99.5"\n')
        rows = parse_ncu_csv(source)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["value"], "1,024")
        self.assertEqual(rows[1]["id"], "1")
        command = profile_command("ncu", "bench", small_plan()["trials"][0], 3)
        self.assertEqual(command[command.index("--clock-control") + 1], "none")
        self.assertEqual(command[command.index("--cache-control") + 1], "none")

    def test_verification_rejects_wrong_gpu_condition_and_unsupported_locality(self):
        record = {"condition_id": "c", "workload": "l2", "config": {"gpu_uuid": DEVICE["uuid"]}}
        evidence = {"condition_id": "c", "gpu_uuid": DEVICE["uuid"], "workload": "l2",
                    "memory_target_verified": True, "verification_notes": "Synthetic test, not measured evidence",
                    "profile_clocks_verified": True, "rows": [{"metric": "counter", "value": "1"}]}
        assessment = attach_verification(record, evidence)["validation"]
        self.assertFalse(assessment["memory_target_verified"])
        self.assertEqual(assessment["status"], "inconclusive")
        for modified in ({"gpu_uuid": "GPU-other"}, {"condition_id": "other"}, {"locality": "near"}):
            with self.subTest(modified=modified), self.assertRaises(ValueError):
                attach_verification(record, {**evidence, **modified})

    def test_cli_offline_plan_and_error_route_do_not_initialize_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            device = Path(directory) / "device.json"
            config = Path(directory) / "config.json"
            output = Path(directory) / "plan.json"
            device.write_text(json.dumps(DEVICE))
            config.write_text(json.dumps({"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                                          "experiments": [{"workload": "tensor"}]}))
            with patch("powermodeling.cli._discover", side_effect=AssertionError("offline planning queried GPU")), redirect_stdout(io.StringIO()):
                status = main(["plan", "--config", str(config), "--device-json", str(device), "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(len(json.loads(output.read_text())["trials"]), 3)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["analyze", "--input", str(Path(directory) / "absent"), "--output", directory]), 2)

    def test_cli_rejected_fit_is_saved_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source=Path(directory)/"rows.json"
            output=Path(directory)/"model.json"
            source.write_text('[]')
            self.assertEqual(main(["fit","--input",str(source),"--features","x","--output",str(output)]),2)
            self.assertEqual(json.loads(output.read_text())["status"],"rejected")


if __name__ == "__main__":
    unittest.main()
