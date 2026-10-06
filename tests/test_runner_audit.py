"""Regression checks for rejected evidence and reversible experiment ownership."""

from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from powermodeling.runner import assess_exclusive_device, capture_trial, load_trials, run_plan, validate_plan_execution, validate_trial_ids


DEVICE = {"uuid": "GPU-runner-audit", "device_index": 0, "benchmark_sha256": None}
TRIAL = {"trial_id": "condition-r0", "condition_id": "condition", "repeat": 0,
         "workload": "tensor", "parameters": {}, "stage": "saturation",
         "seconds": 12, "warmup_seconds": 3, "idle_seconds": 6,
         "clocks": {"graphics_mhz": None, "memory_mhz": None}}


class Device:
    def __init__(self, uuid=DEVICE["uuid"], **kwargs):
        self.uuid = uuid
        self.closed = False
        self.meta = {"uuid": uuid, "mig_mode": {"current": 0, "pending": 0},
                     "sample": {"compute_processes": [], "graphics_processes": []}, "capabilities": {}}

    def metadata(self):
        return self.meta

    def close(self):
        self.closed = True


class Sampler:
    def __init__(self, device, interval_s):
        self.samples = [{"t_s": index / 2, "power_w": 50,
                         "compute_processes": [], "graphics_processes": []} for index in range(49)]

    def start(self):
        return self

    def stop(self):
        return self.samples


def output(uuid=DEVICE["uuid"]):
    events = [{"type": "device", "uuid": uuid}]
    for name, begin, end in (("idle_pre", 0, 6), ("measure", 6, 18), ("idle_post", 18, 24)):
        for event, stamp in (("start", begin), ("end", end)):
            events.append({"type": "phase", "phase": name, "event": event, "host_monotonic_ns": stamp * 10**9})
    events.append({"type": "result", "duration_s": 12, "host_duration_s": 12, "operations": 12e12})
    return "\n".join(json.dumps(event) for event in events)


class Process:
    pid, returncode = 12345, 0
    stdout_text = output()

    def __init__(self, command, **kwargs):
        self.kwargs = kwargs

    def communicate(self, timeout):
        return self.stdout_text, "fixture stderr"

    def poll(self):
        return self.returncode


@contextmanager
def no_clock_mutation(device, **kwargs):
    record = {"restored": None}
    try:
        yield record
    finally:
        record["restored"] = True


class RunnerAuditTests(unittest.TestCase):
    def dense_strict_plan(self):
        from powermodeling.planner import expand_plan
        supported = {"supported_pairs": [
            {"graphics_mhz": graphics, "memory_mhz": 1000}
            for graphics in (900, 990, 1080, 1110, 1170, 1260)],
            "default_applications_graphics_mhz": 900, "default_applications_memory_mhz": 1000}
        device = {**DEVICE, "sm_count": 80, "l2_bytes": 6*1024**2,
                  "total_memory_bytes": 16*1024**3}
        config = {"study_design": "energy_sweep", "clock_sweep": {
            "graphics_step_mhz": 90, "all_memory_clocks": True},
            "experiments": [{"workload": "tensor", "parameters": {"blocks": 80, "threads": 256}}]}
        return expand_plan(config, device, supported)

    def strict_plan(self):
        pairs = [{"graphics_mhz": g, "memory_mhz": 1000 if g is not None else None}
                 for g in (900, 1110, None)]
        return {"study_design": "energy_sweep", "execution_allowed": True,
                "clock_sweep_policy": {"graphics_step_mhz": 90, "graphics_min_mhz": 900},
                "device": DEVICE, "sample_interval_s": .05,
                "clock_sweep_coverage": {
                    "requirements_status": "complete",
                    "strategy": "nearest_supported_graphics_grid", "requested_step_mhz": 90,
                    "supported_memory_mhz": [1000], "selected_memory_mhz": [1000],
                    "advertised_default": {"status": "included", "graphics_mhz": 900, "memory_mhz": 1000},
                    "clock_conditions": [{"clocks": pair} for pair in pairs],
                    "memory_domains": [{"memory_mhz": 1000, "supported_graphics_mhz": [900, 1110]}]},
                "trials": [{**TRIAL, "trial_id": f"geometry-{blocks}-clock-{i}",
                            "parameters": {"blocks": blocks}, "clocks": pair}
                           for blocks in (80, 160) for i, pair in enumerate(pairs)]}

    def test_unknown_required_default_blocks_before_device_and_output_mutation(self):
        plan = self.strict_plan()
        plan["execution_allowed"] = False
        plan["clock_sweep_coverage"].update(requirements_status="incomplete",
            requirement_reasons=["advertised_default_unavailable"])
        with tempfile.TemporaryDirectory() as directory, patch("powermodeling.telemetry.NvmlDevice") as device:
            destination = Path(directory) / "uncreated"
            with self.assertRaisesRegex(ValueError, "advertised_default_unavailable"):
                run_plan(plan, "fixture", destination, DEVICE, apply_clocks=True)
            device.assert_not_called()
            self.assertFalse(destination.exists())

    def test_required_clock_must_be_retained_for_every_workload_geometry(self):
        plan = self.strict_plan()
        validate_plan_execution(plan)
        # A default in another geometry cannot hide the missing comparison arm.
        plan["trials"] = [trial for trial in plan["trials"]
                          if not (trial["parameters"]["blocks"] == 160 and trial["clocks"]["graphics_mhz"] == 900)]
        with self.assertRaisesRegex(ValueError, "each workload geometry"):
            validate_plan_execution(plan)

    def test_strict_plan_cannot_promote_incomplete_or_missing_1110_coverage(self):
        plan = self.strict_plan()
        plan["clock_sweep_coverage"]["requirements_status"] = "incomplete"
        with self.assertRaisesRegex(ValueError, "must be complete"):
            validate_plan_execution(plan)
        plan = self.strict_plan()
        plan["clock_sweep_coverage"]["clock_conditions"] = [condition for condition in plan["clock_sweep_coverage"]["clock_conditions"]
                                                           if condition["clocks"]["graphics_mhz"] != 1110]
        with self.assertRaisesRegex(ValueError, "supported exact 1110"):
            validate_plan_execution(plan)

    def test_complete_label_cannot_hide_wrong_step_or_missing_memory_domain(self):
        plan = self.strict_plan()
        plan["clock_sweep_coverage"]["requested_step_mhz"] = 180
        with self.assertRaisesRegex(ValueError, "declared interval"):
            validate_plan_execution(plan)
        plan = self.strict_plan()
        plan["clock_sweep_coverage"]["supported_memory_mhz"].append(1500)
        with self.assertRaisesRegex(ValueError, "every advertised memory"):
            validate_plan_execution(plan)

    def test_coordinated_grid_point_deletion_cannot_hide_incomplete_sweep(self):
        plan = self.dense_strict_plan()
        validate_plan_execution(plan)
        coverage = plan["clock_sweep_coverage"]
        coverage["clock_conditions"] = [condition for condition in coverage["clock_conditions"]
                                        if condition["clocks"]["graphics_mhz"] != 990]
        domain = coverage["memory_domains"][0]
        domain["selected_graphics_mhz"].remove(990)
        domain["grid_mapping"] = [point for point in domain["grid_mapping"] if point["selected_mhz"] != 990]
        domain["requested_grid_mhz"].remove(990)
        domain["actual_gaps_mhz"] = [b-a for a, b in zip(domain["selected_graphics_mhz"], domain["selected_graphics_mhz"][1:])]
        plan["trials"] = [trial for trial in plan["trials"] if trial["clocks"]["graphics_mhz"] != 990]
        with self.assertRaisesRegex(ValueError, "interval grid"):
            validate_plan_execution(plan)

    def test_complementary_paired_and_unpaired_clock_subsets_do_not_complete_each_other(self):
        plan = self.dense_strict_plan()
        validate_plan_execution(plan)
        for trial in plan["trials"]:
            if trial["clocks"]["graphics_mhz"] in (900, 990, 1080):
                trial.pop("treatment_protocol")
        with self.assertRaisesRegex(ValueError, "each workload geometry"):
            validate_plan_execution(plan)

    def test_order_randomization_stays_one_design_but_distinct_reference_designs_do_not(self):
        plan = self.dense_strict_plan()
        self.assertEqual({trial["treatment_protocol"]["order"] for trial in plan["trials"]}, {"AB", "BA"})
        validate_plan_execution(plan)
        # A differently matched reference cannot supply missing frequency arms.
        for trial in plan["trials"]:
            if trial["clocks"]["graphics_mhz"] in (900, 990, 1080):
                trial["treatment_protocol"]["reference_matching"] = "coarse_unmatched_geometry"
        with self.assertRaisesRegex(ValueError, "each workload geometry"):
            validate_plan_execution(plan)

    def test_runtime_grid_rebuild_uses_planner_ties_endpoints_and_exact_anchor(self):
        from powermodeling.planner import expand_plan
        device = {**DEVICE, "sm_count": 80, "l2_bytes": 6*1024**2,
                  "total_memory_bytes": 16*1024**3}
        supported = {"supported_pairs": [{"graphics_mhz": g, "memory_mhz": 1000}
                                        for g in (900, 945, 1035, 1110, 1140, 1275)],
                     "default_applications_graphics_mhz": 1140, "default_applications_memory_mhz": 1000}
        config = {"study_design": "energy_sweep", "clock_sweep": {
            "graphics_step_mhz": 90, "all_memory_clocks": True},
            "experiments": [{"workload": "tensor"}]}
        plan = expand_plan(config, device, supported)
        selected = {condition["clocks"]["graphics_mhz"]
                    for condition in plan["clock_sweep_coverage"]["clock_conditions"]}
        self.assertIn(945, selected)  # 990 MHz grid target ties945/1035; lower wins.
        self.assertIn(1275, selected)  # Non-grid maximum is still mandatory.
        self.assertIn(1110, selected)
        validate_plan_execution(plan)
        for missing in (945, 1275):
            incomplete = copy.deepcopy(plan)
            incomplete["clock_sweep_coverage"]["clock_conditions"] = [
                condition for condition in incomplete["clock_sweep_coverage"]["clock_conditions"]
                if condition["clocks"]["graphics_mhz"] != missing]
            incomplete["trials"] = [trial for trial in incomplete["trials"]
                                    if trial["clocks"]["graphics_mhz"] != missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "interval grid"):
                validate_plan_execution(incomplete)
    def test_unsafe_and_duplicate_trial_ids_fail_before_device_or_output_mutation(self):
        for identifier in (None, "", ".", "..", "../escaped", "folder/trial", "folder\\trial", "bad\nname"):
            with self.subTest(identifier=identifier), tempfile.TemporaryDirectory() as directory, \
                    patch("powermodeling.telemetry.NvmlDevice") as device:
                plan = {"device": DEVICE, "trials": [{**TRIAL, "trial_id": identifier}], "sample_interval_s": 0.05}
                destination = Path(directory) / "uncreated"
                with self.assertRaisesRegex(ValueError, "trial_id"):
                    run_plan(plan, "fixture", destination, DEVICE)
                device.assert_not_called()
                self.assertFalse(destination.exists())
        with self.assertRaisesRegex(ValueError, "Duplicate trial_id"):
            validate_trial_ids([TRIAL, dict(TRIAL)])
        validate_trial_ids([{"trial_id": "a0123456789bcdef-r0"}, {"trial_id": "fixture-trial_1.2"}])

    def capture(self, process=Process, sampler=Sampler, trial=TRIAL):
        with patch("powermodeling.telemetry.Sampler", sampler), \
                patch("powermodeling.runner.subprocess.Popen", process), \
                patch("powermodeling.runner.platform.platform", return_value="fixture-platform"):
            return capture_trial("fixture", trial, Device(), DEVICE)

    def paired_output(self, order="AB", malformed=None):
        arm_order = ["active_reference", "measure"] if order == "AB" else ["measure", "active_reference"]
        protocol = {"type": "treatment_protocol", "kind": "paired_active_reference", "order": order,
                    "phase_order": arm_order, "same_process": True, "same_allocations": True,
                    "same_clock_policy": True, "launch_geometry_matched": True,
                    "reference_workload": "control", "reference_kind": "issue_loop"}
        geometry = {"blocks": 80, "threads": 256, "batch_launches": 16, "iterations_per_launch": 1024}
        events = [{"type": "device", "uuid": DEVICE["uuid"]}, protocol]
        windows, cursor = [("idle_pre", 0, 6)], 6
        for arm in arm_order:
            windows.append(("warmup_reference" if arm == "active_reference" else "warmup_treatment", cursor, cursor + 3))
            cursor += 3
            windows.append((arm, cursor, cursor + 12))
            cursor += 12
        windows.append(("idle_post", cursor, cursor + 6))
        for name, begin, end in windows:
            for event, stamp in (("start", begin), ("end", end)):
                if malformed == "missing_warmup" and name == "warmup_reference": continue
                events.append({"type": "phase", "phase": name, "event": event, "host_monotonic_ns": stamp * 10**9})
        events.append({"type": "result", **geometry, "duration_s": 12, "host_duration_s": 12, "operations": 12e12})
        reference = {"type": "active_reference_result", "workload": "control", "reference_kind": "issue_loop",
                     **geometry, "duration_s": 12, "host_duration_s": 12, "operations": 0, "logical_bytes": 0,
                     "sanity": {"requested_sm_coverage_complete": True}, "measure_epochs": []}
        if malformed == "geometry": reference["threads"] = 128
        if malformed != "missing_reference": events.append(reference)
        if malformed == "duplicate_reference": events.append(dict(reference))
        if malformed == "wrong_order": protocol["order"] = "BA" if order == "AB" else "AB"
        if malformed == "unmatched_allocations": protocol["same_allocations"] = False
        return "\n".join(json.dumps(event) for event in events)

    def test_paired_capture_preserves_each_arm_and_randomized_order(self):
        class PairedSampler(Sampler):
            def __init__(self, device, interval_s):
                self.samples = [{"t_s": index / 2, "power_w": 50,
                                 "compute_processes": [], "graphics_processes": []} for index in range(85)]
        for order in ("AB", "BA"):
            class PairedProcess(Process):
                stdout_text = self.paired_output(order)
            trial = {**TRIAL, "treatment_protocol": {"kind": "paired_active_reference", "order": order},
                     "clock_policy": "fixed_supported_sweep", "clock_selection_reasons": ["grid_90mhz"]}
            with self.subTest(order=order):
                record = self.capture(process=PairedProcess, sampler=PairedSampler, trial=trial)
                self.assertEqual(record["status"], "complete", record["quality"]["runner_errors"])
                self.assertEqual(record["treatment_protocol"]["order"], order)
                self.assertEqual(record["active_reference"]["operations"], 0)
                self.assertEqual(record["active_reference"]["reference_kind"], "issue_loop")
                self.assertEqual(record["config"]["clock_selection_policy"], "fixed_supported_sweep")
                self.assertIn("active_reference", record["phases"])

    def test_missing_or_mismatched_paired_arm_cannot_pass(self):
        trial = {**TRIAL, "treatment_protocol": {"kind": "paired_active_reference", "order": "AB"}}
        for malformed in ("missing_reference", "duplicate_reference", "geometry", "missing_warmup", "wrong_order", "unmatched_allocations"):
            class InvalidPair(Process):
                stdout_text = self.paired_output(malformed=malformed)
            with self.subTest(malformed=malformed):
                record = self.capture(process=InvalidPair, trial=trial)
                self.assertEqual(record["status"], "failed")
                self.assertTrue(record["quality"]["runner_errors"])
                self.assertIn('"type": "treatment_protocol"', record["quality"]["benchmark_stdout"])
        class UnplannedPair(Process):
            stdout_text = self.paired_output()
        self.assertEqual(self.capture(process=UnplannedPair)["status"], "failed")

    def test_null_graphics_inventory_and_malformed_pid_reject_exclusivity(self):
        for inventory in (None, "unavailable", [{"pid": None}], [{"pid": True}]):
            with self.subTest(inventory=inventory):
                device = Device()
                device.meta["sample"]["graphics_processes"] = inventory
                with self.assertRaisesRegex(RuntimeError, "inventory unavailable"):
                    assess_exclusive_device(device)

    def test_foreign_graphics_and_mps_contexts_reject_exclusivity(self):
        for field in ("graphics_processes", "mps_compute_processes"):
            with self.subTest(field=field):
                device = Device()
                device.meta["sample"][field] = [{"pid": 42}]
                with self.assertRaisesRegex(RuntimeError, "existing"):
                    assess_exclusive_device(device)
                if field == "graphics_processes":
                    self.assertTrue(assess_exclusive_device(device, allowed_pids=(42,))["exclusivity_assessment"]["exclusivity_verified"])
                else:
                    with self.assertRaisesRegex(RuntimeError, "MPS contexts"):
                        assess_exclusive_device(device, allowed_pids=(42,))

    def test_pending_mig_and_mps_environment_fail_before_mutation(self):
        device = Device()
        device.meta["mig_mode"]["pending"] = 1
        with self.assertRaisesRegex(RuntimeError, "MIG"):
            assess_exclusive_device(device)
        with patch.dict(os.environ, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100"}):
            with self.assertRaisesRegex(RuntimeError, "MPS environment"):
                assess_exclusive_device(Device())

    def test_sampler_error_preserves_raw_samples_and_worker_output(self):
        class FailingSampler(Sampler):
            error = {"type": "RuntimeError", "message": "sensor failed"}
            def stop(self):
                raise RuntimeError("sensor failed")
        record = self.capture(sampler=FailingSampler)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(record["samples"]), 49)
        self.assertIn('"type": "result"', record["quality"]["benchmark_stdout"])
        self.assertEqual(record["quality"]["sampler_error"]["message"], "sensor failed")

    def test_array_nan_malformed_phase_and_wrong_gpu_cannot_pass(self):
        for invalid in ("[]", '{"type":"result","duration_s":NaN}',
                        '{"type":"result","duration_s":1e999}',
                        '{"type":"phase","phase":"measure","event":"start","host_monotonic_ns":null}'):
            class InvalidProcess(Process):
                stdout_text = output() + "\n" + invalid
            with self.subTest(invalid=invalid):
                record = self.capture(process=InvalidProcess)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(len(record["samples"]), 49)
                self.assertTrue(record["quality"]["runner_errors"])
        class WrongGpu(Process):
            stdout_text = output("GPU-other")
        self.assertEqual(self.capture(process=WrongGpu)["status"], "failed")

    def test_unknown_measurement_inventory_retains_metrics_but_cannot_pass(self):
        class UnknownSampler(Sampler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.samples[20]["graphics_processes"] = None
        record = self.capture(sampler=UnknownSampler)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["quality"]["process_inventory_unknown"], ["graphics_processes"])
        self.assertFalse(record["quality"]["valid"])

    def test_keyboard_interrupt_stops_worker_preserves_record_and_restores_clocks(self):
        class InterruptedProcess(Process):
            returncode = None
            calls = 0
            def communicate(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt("cancelled")
                self.returncode = -9
                return self.stdout_text, "stopped"
        def terminate(process):
            process.returncode = None  # communicate harvests output and final exit
        plan = {"device": DEVICE, "trials": [TRIAL], "sample_interval_s": 0.05}
        with tempfile.TemporaryDirectory() as directory, \
                patch("powermodeling.telemetry.NvmlDevice", Device), \
                patch("powermodeling.telemetry.Sampler", Sampler), \
                patch("powermodeling.runner.subprocess.Popen", InterruptedProcess), \
                patch("powermodeling.runner.platform.platform", return_value="fixture-platform"), \
                patch("powermodeling.runner._terminate_process_group", side_effect=terminate) as stop, \
                patch("powermodeling.clocks.clock_context", no_clock_mutation), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_plan(plan, "fixture", directory, DEVICE)
            record = load_trials(directory)[0]
            self.assertEqual(record["status"], "failed")
            self.assertTrue(record["clock_policy"]["restored"])
            self.assertEqual(len(record["samples"]), 49)
            self.assertIn('"type": "result"', record["quality"]["benchmark_stdout"])
            stop.assert_called_once()

    def test_resume_rejects_clock_method_and_restore_policy_change(self):
        controlled = {**TRIAL, "clocks": {"graphics_mhz": 900, "memory_mhz": 1215}}
        plan = {"device": DEVICE, "trials": [controlled], "sample_interval_s": 0.05}
        record = {"condition_id": "condition", "status": "complete", "provenance": {"benchmark_sha256": None},
                  "config": {"gpu_uuid": DEVICE["uuid"], "sample_interval_s": 0.05,
                             "clock_method": "locked", "locked_restore": {"graphics": None, "memory": None}}}
        with tempfile.TemporaryDirectory() as directory, patch("powermodeling.telemetry.NvmlDevice", Device):
            path = Path(directory) / "trials" / "condition-r0.json"
            path.parent.mkdir()
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "restoration policy differs"):
                run_plan(plan, "fixture", directory, DEVICE, apply_clocks=True, clock_method="applications", resume=True)
            with self.assertRaisesRegex(ValueError, "restoration policy differs"):
                run_plan(plan, "fixture", directory, DEVICE, apply_clocks=True, clock_method="locked",
                         locked_restore={"graphics": [900, 1200], "memory": None}, resume=True)

    def test_failed_clock_setup_still_saves_failed_trial(self):
        @contextmanager
        def failing_clock(device, **kwargs):
            error = RuntimeError("clock setup failed")
            error.clock_record = {"restored": True, "applied": False}
            raise error
            yield  # make a contextmanager, never entered
        plan = {"device": DEVICE, "trials": [TRIAL], "sample_interval_s": 0.05}
        with tempfile.TemporaryDirectory() as directory, \
                patch("powermodeling.telemetry.NvmlDevice", Device), \
                patch("powermodeling.clocks.clock_context", failing_clock), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "clock setup failed"):
                run_plan(plan, "fixture", directory, DEVICE)
            record = load_trials(directory)[0]
            self.assertEqual(record["status"], "failed")
            self.assertTrue(record["clock_policy"]["restored"])
            self.assertIn("clock setup failed", record["quality"]["runner_errors"][0])


if __name__ == "__main__":
    unittest.main()
