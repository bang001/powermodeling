"""Cache-policy plans and replay binding use synthetic, hardware-free records."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from powermodeling.planner import benchmark_command, expand_plan, validate_experiment_geometry
from powermodeling.profiling import capture_profile, profile_command
from powermodeling.validation import DRAM_READ, validate_evidence
from test_memory_read_contract import memory_record


class ReadCachePlanTests(unittest.TestCase):
    def plan(self, workload="hbm", policy=None, role=None, access="read"):
        parameters = {"access": access}
        if policy is not None:
            parameters["read_cache_policy"] = policy
        experiment = {"workload": workload, "parameters": parameters}
        if role is not None:
            experiment["experiment_role"] = role
        return expand_plan({"clock_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1215}],
                            "experiments": [experiment]},
                           {"sm_count": 108, "l2_bytes": 40 * 1024**2,
                            "total_memory_bytes": 40 * 1024**3})

    def test_default_and_auto_resolve_to_workload_cache_policy(self):
        for workload, expected in (("l1", "ca"), ("l2", "cg"), ("hbm", "cg")):
            for requested in (None, "auto"):
                with self.subTest(workload=workload, requested=requested):
                    trial = self.plan(workload, requested)["trials"][0]
                    self.assertEqual(trial["parameters"].get("read_cache_policy"), expected)
                    for profiling in (False, True):
                        command = benchmark_command("bench", trial, profiling=profiling)
                        self.assertEqual(command[command.index("--read-cache-policy") + 1], expected)

    def test_hbm_ca_cs_require_explicit_diagnostic_role(self):
        for policy in ("ca", "cs"):
            for role in (None, "energy_characterization"):
                with self.subTest(policy=policy, role=role), self.assertRaisesRegex(ValueError, "diagnostic"):
                    self.plan(policy=policy, role=role)
            trial = self.plan(policy=policy, role="diagnostic")["trials"][0]
            self.assertEqual(trial["parameters"]["read_cache_policy"], policy)
            self.assertEqual(trial["experiment_role"], "diagnostic")

    def test_wrong_cache_policy_or_workload_is_rejected(self):
        for workload, policy, access in (("l1", "cg", "read"), ("l2", "ca", "read"),
                                         ("l2_latency", "cg", "read"), ("tensor", "ca", "read"),
                                         ("hbm", "cg", "write"), ("hbm", "cg", "copy"),
                                         ("hbm", "unknown", "read")):
            with self.subTest(workload=workload, policy=policy, access=access), self.assertRaises(ValueError):
                self.plan(workload, policy, "diagnostic", access)

    def test_edited_unlabelled_plan_cannot_run_hbm_cache_diagnostic(self):
        for role in (None, "energy_characterization"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "diagnostic"):
                validate_experiment_geometry("hbm", {"read_cache_policy": "cs"}, role, {})

    def test_old_plan_omits_new_worker_flag(self):
        trial = {"workload": "hbm", "parameters": {}, "seconds": 12,
                 "warmup_seconds": 3, "idle_seconds": 6}
        self.assertNotIn("--read-cache-policy", benchmark_command("legacy-bench", trial))

    def test_hbm_diagnostic_preset_preserves_geometry_and_supported_fixed_anchors(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/hbm-cache-policy-diagnostics.json").read_text())
        supported = {"supported_pairs": [{"graphics_mhz": g, "memory_mhz": m}
                    for m, graphics in ((1000, [900, 1080, 1110]), (1500, [900, 1110, 1410])) for g in graphics],
                     "default_applications_graphics_mhz": 1080, "default_applications_memory_mhz": 1000}
        plan = expand_plan(config, {"sm_count": 108, "l2_bytes": 40 * 1024**2,
                                   "total_memory_bytes": 40 * 1024**3}, supported)
        self.assertEqual(len(plan["trials"]), 27)
        self.assertEqual({(t["clocks"]["graphics_mhz"], t["clocks"]["memory_mhz"]) for t in plan["trials"]},
                         {(1080, 1000), (1110, 1500), (1410, 1500)})
        self.assertEqual({t["parameters"]["read_cache_policy"] for t in plan["trials"]}, {"cg", "ca", "cs"})
        geometries = {json.dumps({k: v for k, v in t["parameters"].items() if k != "read_cache_policy"}, sort_keys=True)
                      for t in plan["trials"]}
        self.assertEqual(len(geometries), 1)
        for trial in plan["trials"]:
            self.assertEqual(trial["experiment_role"], "diagnostic")
            self.assertNotIn("treatment_protocol", trial)
            for profiling in (False, True):
                command = benchmark_command("bench", trial, profiling=profiling)
                self.assertEqual(command[command.index("--read-cache-policy") + 1], trial["parameters"]["read_cache_policy"])


class ReadCacheBindingTests(unittest.TestCase):
    def record(self, legacy=False):
        record = memory_record()
        record["workload"] = "hbm"
        record["validation"]["profiler_evidence"]["workload"] = "hbm"
        for benchmark in (record["benchmark"], record["validation"]["profiler_evidence"]["profile_benchmark"]):
            benchmark["workload"] = "hbm"
            if not legacy:
                benchmark.update(kernel_implementation_version="scalar_single_stream_read_v3",
                                 read_cache_policy="cg", memory_read_index_math="uint32")
        return record

    def test_matching_v3_metadata_and_legacy_without_new_fields_pass(self):
        for record in (self.record(), self.record(legacy=True)):
            result = validate_evidence(record, record["validation"]["profiler_evidence"])
            self.assertEqual(result["status"], "pass", result["reasons"])

    def test_cache_policy_and_index_variant_mismatches_fail_binding(self):
        for key, value in (("read_cache_policy", "ca"), ("memory_read_index_math", "uint64")):
            record = self.record()
            record["validation"]["profiler_evidence"]["profile_benchmark"][key] = value
            result = validate_evidence(record, record["validation"]["profiler_evidence"])
            with self.subTest(key=key):
                self.assertEqual(result["status"], "fail")
                self.assertIn("matching_effective_" + key, [check["name"] for check in result["checks"] if check["status"] == "fail"])

    def test_legacy_default_cache_can_bind_new_explicit_default_plan(self):
        record = self.record(legacy=True)
        record["config"]["read_cache_policy"] = "cg"
        evidence = record["validation"]["profiler_evidence"]
        evidence["profile_provenance"]["parameters"]["read_cache_policy"] = "cg"
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "pass", result["reasons"])

    def test_new_implementation_cannot_drop_both_binding_fields(self):
        for key in ("read_cache_policy", "memory_read_index_math"):
            record = self.record()
            for benchmark in (record["benchmark"], record["validation"]["profiler_evidence"]["profile_benchmark"]):
                benchmark.pop(key)
            result = validate_evidence(record, record["validation"]["profiler_evidence"])
            with self.subTest(key=key):
                self.assertEqual(result["status"], "inconclusive")
                self.assertIn("missing_effective_" + key, [check["name"] for check in result["checks"] if check["status"] == "inconclusive"])

    def test_matching_effective_cache_cannot_contradict_requested_policy(self):
        record = self.record()
        record["config"]["read_cache_policy"] = "cg"
        record["validation"]["profiler_evidence"]["profile_provenance"]["parameters"]["read_cache_policy"] = "cg"
        for benchmark in (record["benchmark"], record["validation"]["profiler_evidence"]["profile_benchmark"]):
            benchmark["read_cache_policy"] = "ca"
        result = validate_evidence(record, record["validation"]["profiler_evidence"])
        self.assertEqual(result["status"], "fail")

    def test_v3_matching_index_variant_must_fit_effective_region(self):
        record = self.record()
        for benchmark in (record["benchmark"], record["validation"]["profiler_evidence"]["profile_benchmark"]):
            benchmark["memory_read_index_math"] = "uint64"
        result = validate_evidence(record, record["validation"]["profiler_evidence"])
        self.assertEqual(result["status"], "fail")

    def test_v3_uint64_fallback_for_region_above_uint32_range_passes(self):
        record = self.record()
        for benchmark in (record["benchmark"], record["validation"]["profiler_evidence"]["profile_benchmark"]):
            benchmark.update(memory_read_index_math="uint64", working_set_bytes=4 * 2**32)
        result = validate_evidence(record, record["validation"]["profiler_evidence"])
        self.assertEqual(result["status"], "pass", result["reasons"])

    def test_v3_index_check_rejects_fractional_l1_geometry_without_crashing(self):
        for field, fractional in (("blocks", .5), ("working_set_bytes", 65536.5),
                                  ("iterations_per_launch", .5)):
            with self.subTest(field=field):
                record = self.record()
                record["workload"] = "l1"
                evidence = record["validation"]["profiler_evidence"]
                evidence["workload"] = "l1"
                for benchmark in (record["benchmark"], evidence["profile_benchmark"]):
                    benchmark.update(workload="l1", read_cache_policy="ca", **{field: fractional})
                result = validate_evidence(record, evidence)
                self.assertNotEqual(result["status"], "pass")
                check = next(c for c in result["checks"] if c["name"] == "consistent_memory_read_index_math")
                self.assertEqual(check["status"], "inconclusive")

    def test_unknown_new_metadata_values_cannot_bind_even_when_equal(self):
        for key, value in (("read_cache_policy", "auto"), ("memory_read_index_math", "float32")):
            record = self.record()
            for benchmark in (record["benchmark"], record["validation"]["profiler_evidence"]["profile_benchmark"]):
                benchmark[key] = value
            result = validate_evidence(record, record["validation"]["profiler_evidence"])
            with self.subTest(key=key):
                self.assertEqual(result["status"], "fail")

    def test_variant_provenance_cannot_contradict_effective_kernel(self):
        for key, other in (("read_cache_policy", "ca"), ("memory_read_index_math", "uint64")):
            record = self.record()
            record["validation"]["profiler_evidence"]["profile_provenance"][key] = other
            result = validate_evidence(record, record["validation"]["profiler_evidence"])
            with self.subTest(key=key):
                self.assertEqual(result["status"], "fail")


class ReadCacheCaptureTests(unittest.TestCase):
    def test_optional_exact_issue_counters_are_requested_only_if_discovered(self):
        trial = {"workload": "hbm", "parameters": {}, "seconds": 12,
                 "warmup_seconds": 3, "idle_seconds": 6}
        metric = "smsp__sass_thread_inst_executed_op_global_ld_pred_on.sum"
        advertised = {DRAM_READ, metric, "smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_second"}
        command = profile_command("ncu", "bench", trial, available=advertised)
        requested = command[command.index("--metrics") + 1].split(",")
        self.assertIn(metric, requested)
        self.assertNotIn("smsp__sass_thread_inst_executed_op_integer_pred_on.sum", requested)

    def test_capture_preserves_variant_provenance_and_rejects_replay_changes(self):
        for key, other in ((None, None), ("read_cache_policy", "ca"), ("memory_read_index_math", "uint64")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                record = ReadCacheBindingTests().record()
                evidence = record["validation"]["profiler_evidence"]
                binary = Path(directory) / "bench"
                binary.write_bytes(b"synthetic binary, not executed")
                trial = {"trial_id": "synthetic-r0", "condition_id": record["condition_id"],
                         "workload": "hbm", "parameters": evidence["profile_provenance"]["parameters"],
                         "clocks": evidence["profile_provenance"]["requested_clocks"],
                         "seconds": 12, "warmup_seconds": 3, "idle_seconds": 6}
                plan = {"device": {"uuid": record["config"]["gpu_uuid"],
                                    "benchmark_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
                        "trials": [trial]}
                first = {"type": "result", **evidence["profile_benchmark"]}
                second = {**first, **({key: other} if key else {})}
                events = [{"type": "device", "uuid": record["config"]["gpu_uuid"]}, first, second]
                def capture(command, timeout):
                    stream = io.StringIO()
                    writer = csv.writer(stream)
                    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
                    for row in evidence["rows"]:
                        writer.writerow([row[field] for field in ("id", "kernel", "metric", "unit", "value")])
                    Path(command[command.index("--log-file") + 1]).write_text(stream.getvalue())
                    return SimpleNamespace(stdout="\n".join(json.dumps(e) for e in events), stderr="", returncode=0)
                with patch("powermodeling.profiling.query_version", return_value="Version 2025.2.1.0"), \
                     patch("powermodeling.profiling.query_metrics", return_value={row["metric"] for row in evidence["rows"]}), \
                     patch("powermodeling.profiling._run_profile_process", side_effect=capture):
                    paths = capture_profile(plan, trial["trial_id"], binary, directory)
                manifest = json.loads(Path(paths["evidence"]).read_text())
                self.assertEqual(manifest["profile_provenance"]["deterministic_application_replay"], key is None)
                if key is None:
                    self.assertEqual(manifest["profile_provenance"].get("read_cache_policy"), "cg")
                    self.assertEqual(manifest["profile_provenance"].get("memory_read_index_math"), "uint32")
                else:
                    self.assertEqual(manifest["profile_benchmark"], {})


if __name__ == "__main__":
    unittest.main()
