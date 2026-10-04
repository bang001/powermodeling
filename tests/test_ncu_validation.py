"""Synthetic counters validate admission logic; they are not GPU measurements."""
import copy
import csv
import hashlib
import io
import json
import signal
import subprocess
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from powermodeling.profiling import (attach_verification, available_metric_names,
    capture_profile, normalize_metric_value, parse_ncu_csv, profile_command,
    _run_profile_process,
    query_metrics)
from powermodeling.validation import *


def pass_fixture(workload="l1", access="read"):
    """A deterministic synthetic record/evidence pair for analyzer tests."""
    logical = 100000.0
    result = {"workload": workload, "access": access, "logical_bytes": logical,
              "kernel_launches": 1, "blocks": 80, "threads": 256,
              "iterations_per_launch": 1024, "working_set_bytes": 65536,
              "stride_elements": 1, "offset_bytes": 0, "tensor_accumulators": 4,
              "gemm_m": 1024, "gemm_n": 1024, "gemm_k": 1024,
              "l1_bytes_per_block": 819, "profile_region": True}
    params = {"blocks": 80, "threads": 256, "iterations": 1024, "access": access}
    samples = [{"t_s": .5, "graphics_clock_mhz": 1200, "memory_clock_mhz": 1593, "compute_processes": [{"pid": 42}], "graphics_processes": [], "mps_compute_processes": []}]
    record = {"condition_id": "test-condition", "workload": workload,
              "config": {**params, "gpu_uuid": "GPU-test", "graphics_clock_mhz": 1200, "memory_clock_mhz": 1593},
              "provenance": {"benchmark_sha256": "a" * 64}, "benchmark": copy.deepcopy(result),
              "phases": {"measure": {"start_s": 0, "end_s": 1}}, "samples": copy.deepcopy(samples)}
    values = {LOCAL_LOAD: (0, "sector"), LOCAL_STORE: (0, "sector"), DURATION: (1000, "usecond"), SM_HZ: (1200000000, "cycle/second"),
              L1_REQUESTS: (782, "request"), L1_SECTORS: (3125, "sector"), L1_HITS: (3100 if workload == "l1" else 0, "sector"),
              L2_READ: (0 if workload == "l1" else 3125, "sector"), L2_READ_HITS: (3100 if workload in ("l2", "l2_latency") else 0, "sector"),
              L2_WRITE: (0, "sector"), DRAM_READ: (100000 if workload == "hbm" else 0, "byte"), DRAM_WRITE: (0, "byte"),
              TENSOR_INSTRUCTIONS[0]: (1024, "inst"), TENSOR_ACTIVITY[0]: (90, "%")}
    if workload == "hbm" and access == "write":
        values[L2_READ] = (0, "sector")
        values[L2_WRITE] = (3125, "sector")
        values[DRAM_READ] = (0, "byte")
        values[DRAM_WRITE] = (100000, "byte")
    if workload == "hbm" and access == "copy":
        values[L2_READ] = (1562.5, "sector")
        values[L2_WRITE] = (1562.5, "sector")
        values[DRAM_READ] = (50000, "byte")
        values[DRAM_WRITE] = (50000, "byte")
    evidence = {"condition_id": record["condition_id"], "gpu_uuid": "GPU-test", "workload": workload,
                "profile_benchmark": copy.deepcopy(result),
                "profile_session_status": "complete", "clock_control": {"applied": True, "restored": True, "restore_errors": {}},
                "profile_context": {"gpu_uuid": "GPU-test", "profile_active_nvml_samples": copy.deepcopy(samples)},
                "profile_provenance": {"observed_gpu_uuid": "GPU-test", "benchmark_sha256": "a" * 64, "parameters": params,
                    "requested_clocks": {"graphics_mhz": 1200, "memory_mhz": 1593}, "deterministic_application_replay": True, "observed_device_records": [{"uuid": "GPU-test", "process_id": 42}]},
                "rows": [{"id": "0", "kernel": "synthetic_kernel", "metric": m, "unit": unit, "value": str(v)} for m, (v, unit) in values.items()]}
    return record, evidence


def change(evidence, metric, value=None, unit=None, remove=False):
    rows = evidence["rows"]
    for row in rows:
        if row["metric"] == metric:
            if remove: rows.remove(row)
            else:
                if value is not None: row["value"] = str(value)
                if unit is not None: row["unit"] = unit
            return
    raise AssertionError(metric)


class NcuParsingTests(unittest.TestCase):
    def test_si_binary_duration_rate_and_thousands(self):
        cases = [("1,024", "byte", 1024, "byte"), ("1.5", "Mbyte", 1500000, "byte"),
                 ("2", "MiB", 2097152, "byte"), ("2500", "nsecond", .0000025, "second"),
                 ("2.5", "usecond", .0000025, "second"), ("1.2", "cycle/nsecond", 1.2e9, "cycle/second"),
                 ("2", "Gbyte/second", 2e9, "byte/second"), ("1\u202f024", "byte", 1024, "byte"),
                 ("95%", "%", 95, "%")]
        for raw, unit, expected, canonical in cases:
            with self.subTest(raw=raw, unit=unit):
                actual, normalized, error = normalize_metric_value(raw, unit)
                self.assertAlmostEqual(actual, expected)
                self.assertEqual(normalized, canonical)
                self.assertIsNone(error)

    def test_ambiguous_unknown_and_instance_values_are_never_zero(self):
        for value in ("1,2", "1,234,56", "N/A", "nan", "inf", "20; 40", "", None, True):
            with self.subTest(value=value):
                result, _, error = normalize_metric_value(value, "byte")
                self.assertIsNone(result)
                self.assertTrue(error)
        self.assertIsNone(normalize_metric_value("1", "unknownunit")[0])

    def test_csv_preserves_raw_and_unknowns(self):
        rows = parse_ncu_csv('==PROF==\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","k","dram__bytes_read.sum","byte","1,024"\n"0","k","dram__bytes_write.sum","byte","N/A"\n')
        self.assertEqual(rows[0]["value"], "1,024")
        self.assertEqual(rows[0]["numeric_value"], 1024)
        self.assertIsNone(rows[1]["numeric_value"])

    def test_metric_discovery_is_exact_and_selected_device(self):
        self.assertEqual(available_metric_names('"dram__bytes_read.sum.per_second","dram__bytes_write.sum"'), {"dram__bytes_read.sum.per_second", "dram__bytes_write.sum"})
        with patch("powermodeling.profiling.subprocess.run") as run:
            run.return_value.stdout = "metrics"
            query_metrics("ncu", 3)
            args, kwargs = run.call_args
            self.assertEqual(args[0][args[0].index("--devices") + 1], "3")
            self.assertEqual(kwargs["env"]["LC_ALL"], "C")

    def test_command_measure_region_base_units_and_separate_outputs(self):
        trial = {"workload": "l1", "parameters": {}, "seconds": 12, "warmup_seconds": 3, "idle_seconds": 6}
        command = profile_command("ncu", "bench", trial, 3, {DRAM_READ}, "/tmp/raw.csv", "/tmp/report")
        self.assertEqual(command[command.index("--metrics") + 1], DRAM_READ)
        self.assertEqual(command[command.index("--print-units") + 1], "base")
        self.assertEqual(command[command.index("--profile-from-start") + 1], "off")
        self.assertNotIn("--launch-skip", command)
        self.assertIn("--profile-region", command)
        self.assertEqual(command[command.index("--log-file") + 1], "/tmp/raw.csv")


class TargetAdmissionTests(unittest.TestCase):
    def test_supported_targets_pass_numeric_admission(self):
        for workload in ("l1", "l2", "l2_latency", "hbm", "tensor", "gemm"):
            with self.subTest(workload=workload):
                record, evidence = pass_fixture(workload)
                result = validate_evidence(record, evidence)
                self.assertEqual(result["status"], "pass", result["reasons"])
                self.assertTrue(result["suitable_verified"])

    def test_hbm_read_write_copy_directions(self):
        for access in ("read", "write", "copy"):
            record, evidence = pass_fixture("hbm", access)
            self.assertEqual(validate_evidence(record, evidence)["status"], "pass")
        _, evidence = pass_fixture("hbm", "write")
        change(evidence, DRAM_WRITE, 1)
        self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_manual_flags_stored_assessment_and_numeric_value_ignored(self):
        record, evidence = pass_fixture()
        change(evidence, L1_HITS, 1)
        for row in evidence["rows"]: row["numeric_value"] = 999999
        evidence.update(memory_target_verified=True, profile_clocks_verified=True, assessment={"status": "pass"}, manual_override={"reason": "Synthetic reviewer override"})
        result = attach_verification(record, evidence)["validation"]
        self.assertFalse(result["memory_target_verified"])
        self.assertEqual(result["status"], "fail")
        self.assertIn("manual_override", result)

    def test_broad_hit_rate_alone_cannot_verify_l1(self):
        _, evidence = pass_fixture()
        change(evidence, L1_HITS, remove=True)
        evidence["rows"].append({"id": "0", "kernel": "synthetic_kernel", "metric": "l1tex__t_sector_hit_rate.pct", "unit": "%", "value": "99"})
        self.assertEqual(assess_profile(evidence)["status"], "inconclusive")

    def test_missing_spill_counter_not_assumed_zero(self):
        _, evidence = pass_fixture("tensor")
        change(evidence, LOCAL_LOAD, remove=True)
        self.assertEqual(assess_profile(evidence)["status"], "inconclusive")

    def test_spills_reject_every_workload(self):
        for workload in ("l1", "l2", "hbm", "tensor", "gemm"):
            _, evidence = pass_fixture(workload)
            change(evidence, LOCAL_STORE, 1)
            self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_downstream_cache_traffic_rejects(self):
        for workload in ("l1", "l2"):
            _, evidence = pass_fixture(workload)
            change(evidence, DRAM_READ, 50000)
            self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_hbm_cache_dominated_or_wrong_direction_rejected(self):
        _, evidence = pass_fixture("hbm")
        change(evidence, L2_READ_HITS, 3000)
        self.assertEqual(assess_profile(evidence)["status"], "fail")
        _, evidence = pass_fixture("hbm")
        change(evidence, DRAM_WRITE, 50000)
        self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_tensor_positive_path_does_not_claim_peak(self):
        _, evidence = pass_fixture("tensor")
        change(evidence, TENSOR_ACTIVITY[0], 1)
        result = assess_profile(evidence)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["kernels"][0]["derived"]["tensor_active_pct"], 1)
        self.assertNotIn("peak_utilized", result)
        self.assertEqual(assess_profile(evidence, {"tensor_min_active_pct": 80})["status"], "fail")

    def test_duplicate_or_malformed_counter_inconclusive(self):
        _, evidence = pass_fixture()
        evidence["rows"].append(copy.deepcopy(evidence["rows"][0]))
        self.assertEqual(assess_profile(evidence)["status"], "inconclusive")
        _, evidence = pass_fixture()
        change(evidence, L1_HITS, "1,2")
        self.assertEqual(assess_profile(evidence)["status"], "inconclusive")

    def test_counter_unit_mismatch_not_silent(self):
        _, evidence = pass_fixture()
        change(evidence, L1_HITS, unit="byte")
        result = assess_profile(evidence)
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(any("counter_unit_mismatch" in r for r in result["reasons"]))

    def test_sector_inflation_and_physical_rates(self):
        _, evidence = pass_fixture("hbm")
        result = assess_profile(evidence)
        self.assertEqual(result["rates_summary"]["dram_read_bytes_s"], 100000000)
        self.assertEqual(result["kernels"][0]["derived"]["dram_bytes_per_logical_byte"], 1)
        change(evidence, L2_READ, 30000)
        self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_no_cross_kernel_ratio_average(self):
        _, evidence = pass_fixture()
        second = copy.deepcopy(evidence["rows"])
        for row in second:
            row["id"] = "1"
            if row["metric"] == L1_HITS: row["value"] = "1"
        evidence["rows"].extend(second)
        result = assess_profile(evidence)
        self.assertEqual(len(result["kernels"]), 2)
        self.assertEqual(result["status"], "fail")

    def test_gemm_auxiliary_kernel_permitted_but_spill_checked(self):
        _, evidence = pass_fixture("gemm")
        auxiliary = copy.deepcopy(evidence["rows"])
        for row in auxiliary:
            row["id"] = "1"
            row["kernel"] = "epilogue"
            if row["metric"] in TENSOR_INSTRUCTIONS + TENSOR_ACTIVITY: row["value"] = "0"
        evidence["rows"].extend(auxiliary)
        self.assertEqual(assess_profile(evidence)["status"], "pass")
        for row in auxiliary:
            if row["metric"] == LOCAL_STORE: row["value"] = "1"
        self.assertEqual(assess_profile(evidence)["status"], "fail")

    def test_zero_l1_lookup_sectors_prove_cg_bypass_with_requests_and_l2(self):
        record, evidence = pass_fixture("l2")
        change(evidence, L1_SECTORS, 0)
        self.assertEqual(validate_evidence(record, evidence)["status"], "pass")
        self.assertTrue(assess_profile(evidence)["kernels"][0]["derived"]["l1_bypass_evidence"])
        change(evidence, L1_REQUESTS, 0)
        self.assertEqual(assess_profile(evidence)["status"], "inconclusive")

    def test_l2_write_residency_remains_inconclusive(self):
        _, evidence = pass_fixture("l2", "write")
        self.assertNotEqual(assess_profile(evidence)["status"], "pass")


class ProvenanceAdmissionTests(unittest.TestCase):
    def test_hash_uuid_params_effective_defaults_and_replay_bind(self):
        for mutation in (lambda r, e: e["profile_provenance"].update(benchmark_sha256="b" * 64),
                         lambda r, e: e["profile_provenance"].update(observed_gpu_uuid="GPU-other"),
                         lambda r, e: r["config"].update(threads=128),
                         lambda r, e: e["profile_benchmark"].update(iterations_per_launch=1),
                         lambda r, e: e["profile_provenance"].update(deterministic_application_replay=False)):
            record, evidence = pass_fixture()
            mutation(record, evidence)
            self.assertEqual(validate_evidence(record, evidence)["status"], "fail")

    def test_missing_provenance_is_inconclusive(self):
        record, evidence = pass_fixture()
        del evidence["profile_provenance"]["benchmark_sha256"]
        self.assertEqual(validate_evidence(record, evidence)["status"], "inconclusive")

    def test_actual_clocks_require_both_domains_and_measure_match(self):
        record, evidence = pass_fixture()
        evidence["profile_context"]["profile_active_nvml_samples"] = []
        self.assertEqual(validate_evidence(record, evidence)["status"], "inconclusive")
        record, evidence = pass_fixture()
        change(evidence, SM_HZ, 900e6)
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")
        record, evidence = pass_fixture()
        record["samples"][0]["memory_clock_mhz"] = 1000
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")

    def test_uncontrolled_clocks_never_verified(self):
        record, evidence = pass_fixture()
        record["config"]["graphics_clock_mhz"] = None
        evidence["profile_provenance"]["requested_clocks"]["graphics_mhz"] = None
        self.assertEqual(validate_evidence(record, evidence)["status"], "inconclusive")

    def test_locality_cannot_be_inferred_from_offsets(self):
        record, evidence = pass_fixture()
        evidence["locality"] = "local-heavy"
        with self.assertRaisesRegex(ValueError, "mapping evidence"):
            attach_verification(record, evidence)
        evidence["locality"] = "unclassified"
        self.assertEqual(attach_verification(record, evidence)["validation"]["locality"], "unclassified")

    def test_failed_session_and_clock_restoration_reject(self):
        record, evidence = pass_fixture()
        evidence["profile_session_status"] = "failed"
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")
        record, evidence = pass_fixture()
        evidence["clock_control"]["restored"] = False
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")

    def test_foreign_or_unknown_profile_process_inventory(self):
        record, evidence = pass_fixture()
        evidence["profile_context"]["profile_active_nvml_samples"][0]["graphics_processes"] = [{"pid": 999}]
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")
        record, evidence = pass_fixture()
        del evidence["profile_context"]["profile_active_nvml_samples"][0]["compute_processes"]
        self.assertEqual(validate_evidence(record, evidence)["status"], "inconclusive")
        record, evidence = pass_fixture()
        del evidence["profile_provenance"]["observed_device_records"]
        self.assertEqual(validate_evidence(record, evidence)["status"], "inconclusive")

    def test_explicit_configured_policy_persists_across_fresh_assessment(self):
        record, evidence = pass_fixture()
        change(evidence, L1_HITS, 2900)
        self.assertEqual(validate_evidence(record, evidence)["status"], "fail")
        evidence["validation_policy"] = {"l1_min_hit_pct": 90}
        self.assertEqual(validate_evidence(record, evidence)["status"], "pass")
        self.assertEqual(validate_evidence(record, evidence, {"l1_min_hit_pct": 99})["status"], "fail")

    def test_invalid_policy_rejected(self):
        for policy in ({"l1_min_hit_pct": 101}, {"max_clock_error_fraction": -1}, {"max_sector_inflation": float("nan")}):
            with self.assertRaises(ValueError): assess_profile(pass_fixture()[1], policy)


class ProfileCaptureTests(unittest.TestCase):
    def _setup(self, directory):
        binary = Path(directory) / "bench"
        binary.write_bytes(b"synthetic executable bytes; never executed")
        record, evidence = pass_fixture()
        trial = {"trial_id": "test-condition-r0", "condition_id": record["condition_id"], "workload": "l1", "parameters": evidence["profile_provenance"]["parameters"], "clocks": evidence["profile_provenance"]["requested_clocks"], "seconds": 12, "warmup_seconds": 3, "idle_seconds": 6}
        plan = {"device": {"uuid": "GPU-test", "benchmark_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "device_index": 0}, "trials": [trial]}
        return binary, plan, trial, evidence

    def test_capture_csv_separate_from_json_and_filters_active_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            binary, plan, trial, evidence = self._setup(directory)
            events = [{"type": "device", "uuid": "GPU-test", "process_id": 42}, {"type": "phase", "phase": "measure", "event": "start", "host_monotonic_ns": 0}, {"type": "phase", "phase": "measure", "event": "end", "host_monotonic_ns": 1000000000}, {"type": "result", **evidence["profile_benchmark"]}]
            def subprocess_mock(command, **kwargs):
                if "--version" in command: return SimpleNamespace(stdout="Version 2025.2.1.0", returncode=0, stderr="")
                if "--query-metrics" in command:
                    return SimpleNamespace(stdout="\n".join(r["metric"] for r in evidence["rows"]), returncode=0, stderr="")
                stream = io.StringIO()
                writer = csv.writer(stream)
                writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
                for row in evidence["rows"]: writer.writerow([row[k] for k in ("id", "kernel", "metric", "unit", "value")])
                Path(command[command.index("--log-file") + 1]).write_text(stream.getvalue())
                return SimpleNamespace(stdout="\n".join(json.dumps(e) for e in events), returncode=0, stderr="")
            with patch("powermodeling.profiling.subprocess.run", side_effect=subprocess_mock), patch("powermodeling.profiling._run_profile_process", side_effect=lambda command, timeout: subprocess_mock(command)):
                result = capture_profile(plan, trial["trial_id"], binary, directory, profile_context={"nvml_sample_provider": lambda: [{"t_s": .5, "memory_clock_mhz": 1593}, {"t_s": 2, "memory_clock_mhz": 200}], "gpu_uuid": "GPU-test"})
            manifest = json.loads(Path(result["evidence"]).read_text())
            self.assertEqual(manifest["profile_provenance"]["observed_gpu_uuid"], "GPU-test")
            self.assertEqual(manifest["profile_context"]["profile_active_nvml_samples"], [{"t_s": .5, "memory_clock_mhz": 1593}])
            self.assertNotIn("nvml_sample_provider", manifest["profile_context"])
            self.assertEqual(manifest["assessment"]["status"], "pass")
            self.assertTrue(Path(directory, trial["trial_id"] + ".application.jsonl").read_text().startswith('{"type": "device"'))

    def test_known_unsupported_volta_profiler_rejected_before_metric_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            binary, plan, trial, evidence = self._setup(directory)
            plan["device"].update(compute_capability_major=7, compute_capability_minor=0)
            with patch("powermodeling.profiling.subprocess.run", return_value=SimpleNamespace(stdout="Version 2025.3.0.0", returncode=0, stderr="")) as run:
                with self.assertRaisesRegex(ValueError, "removed Volta"):
                    capture_profile(plan, trial["trial_id"], binary, directory)
                self.assertEqual(run.call_count, 1)
            self.assertTrue(Path(directory, trial["trial_id"] + ".ncu.version.txt").is_file())

    def test_previous_csv_cannot_be_reused_as_fresh_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            binary, plan, trial, evidence = self._setup(directory)
            Path(directory, trial["trial_id"] + ".ncu.csv").write_text("stale profiler rows")
            with patch("powermodeling.profiling.subprocess.run", return_value=SimpleNamespace(stdout=DRAM_READ, returncode=0, stderr="")), patch("powermodeling.profiling._run_profile_process", return_value=SimpleNamespace(stdout=DRAM_READ, returncode=0, stderr="")):
                with self.assertRaisesRegex(RuntimeError, "requested raw CSV log"):
                    capture_profile(plan, trial["trial_id"], binary, directory)

    def test_timeout_and_interrupt_stop_only_owned_group_and_preserve_partial_output(self):
        for failure in (subprocess.TimeoutExpired(["ncu"], 600, output=b"partial json", stderr=b"partial warning"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                process = MagicMock()
                process.pid = 4321
                process.returncode = -9
                process.poll.return_value = -9
                process.communicate.side_effect = [failure, ("partial json\ntail", "partial warning")]
                with patch("powermodeling.profiling.subprocess.Popen", return_value=process) as popen, patch("powermodeling.profiling.os.killpg") as kill_group:
                    with self.assertRaises(type(failure)) as caught:
                        _run_profile_process(["ncu", "bench"], timeout=600)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                kill_group.assert_called_once_with(4321, signal.SIGKILL)
                self.assertEqual(process.communicate.call_args_list[1].kwargs["timeout"], 10)
                self.assertEqual(caught.exception.profile_capture["stdout"], "partial json\ntail")
                self.assertEqual(caught.exception.profile_capture["stderr"], "partial warning")
                self.assertTrue(caught.exception.profile_capture["cleanup"]["leader_reaped"])

    def test_timeout_capture_keeps_raw_application_stderr_and_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            binary, plan, trial, evidence = self._setup(directory)
            process = MagicMock()
            process.pid = 4321
            process.returncode = -9
            process.poll.return_value = -9
            def communicate(timeout):
                if timeout == 600:
                    Path(directory, trial["trial_id"] + ".ncu.csv").write_text('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","k","dram__bytes_read.sum","byte","1024"\n')
                    raise subprocess.TimeoutExpired(["ncu"], timeout, output=b'{"type":"device","uuid":"GPU-test"}', stderr=b"profiler partial stderr")
                return '{"type":"device","uuid":"GPU-test"}', "profiler partial stderr"
            process.communicate.side_effect = communicate
            def discovery(command, **kwargs):
                return SimpleNamespace(stdout="Version 2025.2.1.0" if "--version" in command else DRAM_READ, returncode=0, stderr="")
            with patch("powermodeling.profiling.subprocess.run", side_effect=discovery), patch("powermodeling.profiling.subprocess.Popen", return_value=process), patch("powermodeling.profiling.os.killpg") as kill_group:
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    capture_profile(plan, trial["trial_id"], binary, directory)
            kill_group.assert_called_once_with(4321, signal.SIGKILL)
            manifest = json.loads(Path(caught.exception.evidence_path).read_text())
            self.assertEqual(manifest["profile_session_status"], "failed")
            self.assertEqual(manifest["rows"][0]["metric"], DRAM_READ)
            self.assertEqual(manifest["profile_capture_failure"]["stdout"], '{"type":"device","uuid":"GPU-test"}')
            self.assertEqual(Path(directory, trial["trial_id"] + ".ncu.stderr.txt").read_text(), "profiler partial stderr")
            self.assertEqual(Path(directory, trial["trial_id"] + ".application.jsonl").read_text(), '{"type":"device","uuid":"GPU-test"}')

    def test_interruption_capture_preserves_failure_manifest_before_reraising(self):
        with tempfile.TemporaryDirectory() as directory:
            binary, plan, trial, evidence = self._setup(directory)
            interrupted = KeyboardInterrupt()
            interrupted.profile_capture = {"stdout": "partial application", "stderr": "partial stderr", "cleanup": {"leader_reaped": True}}
            with patch("powermodeling.profiling.subprocess.run", return_value=SimpleNamespace(stdout=DRAM_READ, returncode=0, stderr="")), patch("powermodeling.profiling._run_profile_process", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    capture_profile(plan, trial["trial_id"], binary, directory)
            manifest = json.loads(Path(caught.exception.evidence_path).read_text())
            self.assertEqual(manifest["profile_session_status"], "failed")
            self.assertEqual(manifest["profile_capture_failure"]["stderr"], "partial stderr")

    def test_changed_binary_fails_before_profiler_or_clock_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            binary, plan, trial, evidence = self._setup(directory)
            binary.write_bytes(b"changed executable")
            with patch("powermodeling.profiling.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "differs from planned"):
                    capture_profile(plan, trial["trial_id"], binary, directory)
                run.assert_not_called()


if __name__ == "__main__": unittest.main()
