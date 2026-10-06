"""Synthetic accounting contracts and opt-in CUDA address/count checks.

CPU fixtures are not energy measurements. CUDA tests exercise actual output
and small-buffer wrap boundaries; they do not establish cache residency.
"""

import json
import os
import subprocess
import unittest

from powermodeling.analysis import analyze_trial, summarize
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.validation import (DRAM_READ, DRAM_WRITE, L1_HITS, L1_REQUESTS,
                                     L1_SECTORS, L2_READ, L2_READ_HITS, L2_WRITE,
                                     validate_evidence)
from test_analysis import empirical_record
from test_ncu_validation import change, pass_fixture


READ_VERSION = "scalar_single_stream_read_v2"
WRITE_VERSION = "scalar_four_stream_write_copy_v1"
DEVICE = {"uuid": "GPU-memory-contract-synthetic", "device_index": 0,
          "compute_capability_major": 9, "compute_capability_minor": 0,
          "sm_count": 132, "l2_bytes": 50 * 1024**2,
          "total_memory_bytes": 80 * 1024**3}


def memory_record(access="read", *, legacy=False, repeat=0):
    """Twelve exact synthetic epochs, each completing ten two-block launches."""
    factor = 4 if legacy or access != "read" else 1
    record = empirical_record("GPU-memory-contract-synthetic", 1200, access=access,
                              blocks=2, repeat=repeat)
    benchmark = record["benchmark"]
    record["config"].update(threads=32, iterations=3)
    record["condition_id"] = f"synthetic-{access}-{'legacy' if legacy else 'current'}"
    benchmark.update(blocks=2, threads=32, iterations_per_launch=3,
                     admitted_blocks=240, kernel_launches=120)
    if not legacy:
        benchmark.update(kernel_implementation_version=READ_VERSION if access == "read" else WRITE_VERSION,
                         memory_accesses_per_thread_iteration=factor)
    for epoch in benchmark["measure_epochs"]:
        epoch.update(admitted_blocks=20, kernel_launches=10,
                     operations=20 * 32 * 3 * factor,
                     logical_bytes=20 * 32 * 3 * factor * 4 * (2 if access == "copy" else 1))
    for field in ("operations", "logical_bytes"):
        benchmark[field] = sum(epoch[field] for epoch in benchmark["measure_epochs"])
    evidence = record["validation"]["profiler_evidence"]
    evidence["condition_id"] = record["condition_id"]
    evidence["profile_provenance"]["parameters"].update(threads=32, iterations=3)
    profile = evidence["profile_benchmark"]
    profile.update(blocks=2, threads=32, iterations_per_launch=3,
                   admitted_blocks=2, kernel_launches=1, operations=2 * 32 * 3 * factor,
                   logical_bytes=2 * 32 * 3 * factor * 4 * (2 if access == "copy" else 1))
    if not legacy:
        profile.update(kernel_implementation_version=benchmark["kernel_implementation_version"],
                       memory_accesses_per_thread_iteration=factor)
    read_bytes = profile["logical_bytes"] / 2 if access == "copy" else profile["logical_bytes"] if access == "read" else 0
    write_bytes = profile["logical_bytes"] / 2 if access == "copy" else profile["logical_bytes"] if access == "write" else 0
    for metric, value in ((L1_REQUESTS, 2 * 3 * factor if read_bytes else 0),
                          (L1_SECTORS, read_bytes / 32), (L1_HITS, 0),
                          (L2_READ, read_bytes / 32), (L2_READ_HITS, 0),
                          (L2_WRITE, write_bytes / 32), (DRAM_READ, read_bytes),
                          (DRAM_WRITE, write_bytes)):
        change(evidence, metric, value)
    return record


class MemoryReadContractTests(unittest.TestCase):
    def test_old_four_load_records_keep_reported_counts_and_energy(self):
        old, new = memory_record(legacy=True), memory_record()
        old_trial, new_trial = analyze_trial(old), analyze_trial(new)
        self.assertTrue(old_trial["valid"], old_trial["issues"])
        self.assertTrue(new_trial["valid"], new_trial["issues"])
        self.assertEqual(old_trial["total_energy_j"], new_trial["total_energy_j"])
        self.assertEqual(old_trial["counted_measure_logical_bytes"], 245760)
        self.assertEqual(new_trial["counted_measure_logical_bytes"], 61440)
        self.assertEqual(new_trial["total_pj_per_logical_bit"], 4 * old_trial["total_pj_per_logical_bit"])
        self.assertNotIn("memory_accesses_per_thread_iteration", old["benchmark"])
        self.assertIsNone(old_trial["kernel_implementation_version"])

    def test_old_and_new_implementations_are_not_repeated_measurements_of_one_group(self):
        records = [memory_record(legacy=legacy, repeat=repeat)
                   for legacy in (True, False) for repeat in range(4)]
        # Use the same synthetic hash to check explicit implementation metadata,
        # independently of the real-world binary-hash separation.
        summary = summarize(records)
        self.assertEqual(len(summary["groups"]), 2)
        self.assertEqual(sorted(group["valid_repeats"] for group in summary["groups"]), [4, 4])
        self.assertEqual(len(summary["evaluation"]["components"]), 2)
        self.assertEqual({group["kernel_implementation_version"] for group in summary["groups"]},
                         {None, READ_VERSION})

    def test_read_one_access_and_write_copy_four_access_payloads_are_distinct(self):
        trials = {access: analyze_trial(memory_record(access)) for access in ("read", "write", "copy")}
        for trial in trials.values():
            self.assertTrue(trial["valid"], trial["issues"])
            self.assertTrue(trial["count_energy_time_alignment_exact"])
        self.assertEqual(trials["write"]["counted_measure_operations"],
                         4 * trials["read"]["counted_measure_operations"])
        self.assertEqual(trials["copy"]["counted_measure_operations"],
                         trials["write"]["counted_measure_operations"])
        self.assertEqual(trials["copy"]["counted_measure_logical_bytes"],
                         2 * trials["write"]["counted_measure_logical_bytes"])

    def test_new_fourfold_count_error_is_rejected_even_if_epoch_totals_match(self):
        record = memory_record()
        for field in ("operations", "logical_bytes"):
            record["benchmark"][field] *= 4
            for epoch in record["benchmark"]["measure_epochs"]:
                epoch[field] *= 4
        trial = analyze_trial(record)
        self.assertFalse(trial["valid"])
        self.assertTrue(any(issue.startswith("memory_contract_") for issue in trial["issues"]))

    def test_new_read_needs_admission_counts_while_old_records_are_not_reinterpreted(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                record = memory_record(legacy=legacy)
                record["benchmark"]["measure_epochs"][3].pop("admitted_blocks")
                trial = analyze_trial(record)
                self.assertEqual(trial["valid"], legacy, trial["issues"])
                if not legacy:
                    self.assertIn("memory_contract_missing_or_invalid_admissions:epoch_3", trial["issues"])

    def test_complete_records_with_failed_output_checks_are_invalid(self):
        failures = (
            ("sanity", {"finite_output_sample": False}, "benchmark_output_sanity_failed"),
            ("sanity", {"memory_read_sample_matches_reference": False}, "memory_read_reference_mismatch"),
            ("memory_read_validation", {"status": "fail", "checked_values": 1, "mismatched_values": 0},
             "memory_read_reference_mismatch"),
            ("memory_read_validation", {"status": "pass", "checked_values": 1, "mismatched_values": 1},
             "memory_read_reference_mismatch"),
        )
        for field, evidence, expected_issue in failures:
            with self.subTest(field=field, evidence=evidence):
                record = memory_record()
                self.assertEqual(record["status"], "complete")
                record["benchmark"][field] = evidence
                trial = analyze_trial(record)
                self.assertFalse(trial["valid"])
                self.assertIn(expected_issue, trial["issues"])
                self.assertNotIn("benchmark_failed", trial["issues"])

    def test_missing_or_unchecked_read_reference_keeps_existing_records_compatible(self):
        for status in (None, "not_checked_sm_filtered", "not_checked_iteration_budget"):
            with self.subTest(status=status):
                record = memory_record()
                if status is not None:
                    record["benchmark"]["sanity"] = {"finite_output_sample": True,
                                                         "memory_read_sample_matches_reference": None}
                    record["benchmark"]["memory_read_validation"] = {
                        "status": status, "checked_values": 0, "mismatched_values": 0,
                    }
                trial = analyze_trial(record)
                self.assertTrue(trial["valid"], trial["issues"])
                self.assertNotIn("memory_read_reference_mismatch", trial["issues"])
                diagnostics = trial["measurement_diagnostics"]["execution"]
                self.assertEqual(diagnostics["memory_read_validation"],
                                 record["benchmark"].get("memory_read_validation"))

    def test_execution_diagnostics_preserve_read_checksum_and_reference_evidence(self):
        record = memory_record()
        benchmark = record["benchmark"]
        benchmark["memory_read_checksum_scope"] = "sum32_of_all_reads_per_thread"
        benchmark["memory_read_validation"] = {
            "status": "pass", "checked_values": 6, "mismatched_values": 0,
            "scope": "sampled_threads_sum32_of_all_iteration_reads",
        }
        benchmark["sanity"] = {"finite_output_sample": True,
                               "memory_read_sample_matches_reference": True}
        trial = analyze_trial(record)
        self.assertTrue(trial["valid"], trial["issues"])
        diagnostics = trial["measurement_diagnostics"]["execution"]
        self.assertEqual(diagnostics["memory_read_checksum_scope"], benchmark["memory_read_checksum_scope"])
        self.assertEqual(diagnostics["memory_read_validation"], benchmark["memory_read_validation"])

    def test_profile_cannot_validate_a_different_load_factor_or_implementation(self):
        for field, wrong in (("memory_accesses_per_thread_iteration", 4),
                             ("kernel_implementation_version", WRITE_VERSION)):
            with self.subTest(field=field):
                record = memory_record()
                evidence = record["validation"]["profiler_evidence"]
                self.assertEqual(validate_evidence(record, evidence)["status"], "pass")
                evidence["profile_benchmark"][field] = wrong
                result = validate_evidence(record, evidence)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["suitable_verified"])

    def test_missing_new_profile_metadata_cannot_validate_current_read(self):
        record = memory_record()
        evidence = record["validation"]["profiler_evidence"]
        for field in ("memory_accesses_per_thread_iteration", "kernel_implementation_version"):
            evidence["profile_benchmark"].pop(field)
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["suitable_verified"])

    def test_one_retained_load_cannot_verify_three_reported_read_iterations(self):
        record = memory_record()
        evidence = record["validation"]["profiler_evidence"]
        # Model the compiler retaining only one of three claimed warp loads.
        # Output/address checks alone would not expose this missing work.
        for metric in (L1_REQUESTS, L1_SECTORS, L2_READ, DRAM_READ):
            row = next(row for row in evidence["rows"] if row["metric"] == metric)
            change(evidence, metric, float(row["value"]) / 3)
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["suitable_verified"])
        self.assertTrue(any(check["name"] == "sector_inflation" and check["status"] == "fail"
                            for check in result["checks"]))

    def test_tensor_null_memory_metadata_does_not_create_memory_checks(self):
        record, evidence = pass_fixture("tensor")
        for field in ("memory_accesses_per_thread_iteration", "kernel_implementation_version"):
            record["benchmark"][field] = None
            evidence["profile_benchmark"][field] = None
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "pass", result["reasons"])
        self.assertFalse(any(check["name"] == "profile_memory_count_contract" for check in result["checks"]))

    def test_implicit_read_iterations_change_and_explicit_iterations_are_literal(self):
        for workload, access in (("l1", "read"), ("l2", "read"), ("hbm", "read"),
                                 ("l2", "write"), ("hbm", "copy")):
            for explicit in (False, True):
                with self.subTest(workload=workload, access=access, explicit=explicit):
                    parameters = {"access": access}
                    if explicit:
                        parameters["iterations"] = 128
                    plan = expand_plan({"study_design": "diagnostic", "paired_reference": False,
                                        "clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                                        "experiments": [{"workload": workload, "parameters": parameters}]}, DEVICE)
                    trial = plan["trials"][0]
                    expected = 128 if explicit else 4096 if access == "read" else 1024
                    self.assertEqual(trial["parameters"]["iterations"], expected)
                    command = benchmark_command("bench", trial)
                    self.assertEqual(command[command.index("--iterations") + 1], str(expected))


def initialized_word(index, seed):
    """Reference data generator, evaluated on the host with bounded integers."""
    mask = (1 << 64) - 1
    value = ((index ^ seed) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 32) - 1)


@unittest.skipUnless(os.environ.get("POWERBENCH_GPU_TESTS") == "1",
                     "requires an actual NVIDIA GPU; opt in with POWERBENCH_GPU_TESTS=1")
class MemoryReadGpuTests(unittest.TestCase):
    def run_kernel(self, workload, access, words, iterations, stride=1, offset=0):
        command = [os.environ.get("POWERBENCH", "build/powerbench"), "--workload", workload,
                   "--access", access, "--blocks", "2", "--threads", "32",
                   "--working-set-bytes", str(words * 4), "--stride-elements", str(stride),
                   "--offset-bytes", str(offset), "--seed", "7", "--seconds", "0.1",
                   "--warmup-seconds", "0", "--idle-seconds", "0", "--batch-launches", "1",
                   "--fixed-batches", "2"]
        if iterations is not None:
            command += ["--iterations", str(iterations)]
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
        return next(event for event in map(json.loads, completed.stdout.splitlines()) if event["type"] == "result")

    def test_actual_read_sums_wrap_small_nonpower_of_two_and_strided_buffers(self):
        cases = (("l2", 1, 1, 1, 0), ("l2", 10, 7, 3, 4),
                 ("l2", 2048, 2, 1, 32), ("l1", 2, 7, 1, 4),
                 ("l1", 6, 5, 2, 4), ("l1", 4096, 1, 32, 0))
        for workload, words, iterations, stride, offset in cases:
            with self.subTest(workload=workload, words=words, iterations=iterations, stride=stride, offset=offset):
                result = self.run_kernel(workload, "read", words, iterations, stride, offset)
                self.assertEqual(result["kernel_implementation_version"], READ_VERSION)
                self.assertEqual(result["memory_accesses_per_thread_iteration"], 1)
                self.assertEqual(result["admitted_blocks"], 4)
                self.assertEqual(result["operations"], 4 * 32 * iterations)
                self.assertEqual(result["logical_bytes"], result["operations"] * 4)
                self.assertEqual(sum(epoch["operations"] for epoch in result["measure_epochs"]), result["operations"])
                self.assertEqual(sum(epoch["logical_bytes"] for epoch in result["measure_epochs"]), result["logical_bytes"])
                footprint, digest = set(), 0
                region = words // 2 if workload == "l1" else words
                lanes = 32 if workload == "l1" else 64
                for thread in range(64):
                    local = thread % 32 if workload == "l1" else thread
                    start = (thread // 32) * region if workload == "l1" else 0
                    addresses = [start + ((local + iteration * lanes) * stride) % region
                                 for iteration in range(iterations)]
                    footprint.update(addresses)
                    thread_sum = sum(initialized_word(address + offset // 4, 7) for address in addresses) & ((1 << 32) - 1)
                    digest = (digest * 1315423911 + thread_sum) & ((1 << 64) - 1)
                self.assertEqual(result["finite_launch_reachable_bytes_upper_bound"], len(footprint) * 4)
                self.assertTrue(result["finite_launch_reachable_bytes_exact"])
                self.assertEqual(result["checksum_kind"], "sum32_scalar_reads_per_thread_sample_hash")
                self.assertEqual(result["memory_read_checksum_scope"], "sum32_of_all_reads_per_thread")
                self.assertEqual(result["checksum"], digest)
                self.assertEqual(result["memory_read_validation"]["status"], "pass")
                self.assertGreater(result["memory_read_validation"]["checked_values"], 0)
                self.assertEqual(result["memory_read_validation"]["mismatched_values"], 0)
                self.assertTrue(result["sanity"]["memory_read_sample_matches_reference"])

    def test_actual_write_copy_keep_four_access_counts_and_read_default_is_4096(self):
        for access in ("read", "write", "copy"):
            with self.subTest(access=access):
                result = self.run_kernel("l2", access, 1024, None)
                expected_iterations = 4096 if access == "read" else 1024
                factor = 1 if access == "read" else 4
                self.assertEqual(result["iterations_per_launch"], expected_iterations)
                self.assertEqual(result["memory_accesses_per_thread_iteration"], factor)
                self.assertEqual(result["operations"], 4 * 32 * expected_iterations * factor)
                self.assertEqual(result["logical_bytes"], result["operations"] * (8 if access == "copy" else 4))
                self.assertEqual(result["finite_launch_reachable_bytes_upper_bound"], 4096)
                if access == "read":
                    self.assertEqual(result["memory_read_validation"]["status"], "pass")
                    self.assertTrue(result["sanity"]["memory_read_sample_matches_reference"])


if __name__ == "__main__":
    unittest.main()
