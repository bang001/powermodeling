"""Missing profile clocks must stay unknown instead of certifying a subset."""
import copy
import unittest

from powermodeling.validation import SM_HZ, validate_evidence
import test_ncu_validation


class ProfileClockCoverageTests(unittest.TestCase):
    def test_missing_profile_memory_clock_cannot_certify_one_successful_sample(self):
        record, evidence = test_ncu_validation.pass_fixture("l2")
        samples = evidence["profile_context"]["profile_active_nvml_samples"]
        for index in range(10):
            sample = copy.deepcopy(samples[0])
            sample.update(t_s=.51 + index * .02, memory_clock_mhz=None)
            samples.append(sample)
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["suitable_verified"])
        clock = result["profile_actual_clocks"]["memory_clock_mhz"]
        self.assertEqual(clock["raw_samples"], 11)
        self.assertEqual(clock["valid_samples"], 1)

    def test_clock_query_error_invalidates_even_a_present_numeric_value(self):
        record, evidence = test_ncu_validation.pass_fixture("l2")
        sample = evidence["profile_context"]["profile_active_nvml_samples"][0]
        sample["errors"] = {"memory_clock_mhz": {"type": "NVMLError_Unknown", "code": 999}}
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["suitable_verified"])

    def test_clock_error_key_with_empty_payload_does_not_certify_a_reading(self):
        for payload in ({}, None, False, 0):
            with self.subTest(payload=payload):
                record, evidence = test_ncu_validation.pass_fixture("l2")
                sample = evidence["profile_context"]["profile_active_nvml_samples"][0]
                sample["errors"] = {"memory_clock_mhz": payload}
                result = validate_evidence(record, evidence)
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["suitable_verified"])

    def test_malformed_error_container_is_inconclusive_without_crashing(self):
        for container in ([], ["query failed"], None, "query failed"):
            with self.subTest(container=container):
                record, evidence = test_ncu_validation.pass_fixture("l2")
                sample = evidence["profile_context"]["profile_active_nvml_samples"][0]
                sample["errors"] = container
                try:
                    result = validate_evidence(record, evidence)
                except (AttributeError, TypeError) as exc:
                    self.fail(f"malformed clock errors must return inconclusive: {exc}")
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["suitable_verified"])

    def test_nonpositive_or_nonfinite_profile_clock_is_not_silently_removed(self):
        for value in (0, -1, float("nan"), True):
            with self.subTest(value=value):
                record, evidence = test_ncu_validation.pass_fixture("l2")
                sample = copy.deepcopy(evidence["profile_context"]["profile_active_nvml_samples"][0])
                sample["memory_clock_mhz"] = value
                evidence["profile_context"]["profile_active_nvml_samples"].append(sample)
                result = validate_evidence(record, evidence)
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["suitable_verified"])

    def test_energy_clock_coverage_cannot_be_certified_from_a_subset(self):
        record, evidence = test_ncu_validation.pass_fixture("l2")
        missing = copy.deepcopy(record["samples"][0])
        missing.update(t_s=.75, memory_clock_mhz=None)
        record["samples"].append(missing)
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["suitable_verified"])

    def test_partial_kernel_frequencies_do_not_replace_complete_nvml_graphics_clock(self):
        record, evidence = test_ncu_validation.pass_fixture("gemm")
        second = copy.deepcopy(evidence["rows"])
        for row in second:
            row["id"] = "1"
        second = [row for row in second if row["metric"] != SM_HZ]
        evidence["rows"].extend(second)
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["suitable_verified"])
        clock = result["profile_actual_clocks"]["graphics_clock_mhz"]
        self.assertEqual(clock["kernel_samples"], 2)
        self.assertEqual(clock["valid_kernel_samples"], 1)

    def test_complete_kernel_frequency_can_replace_missing_nvml_graphics_clock(self):
        record, evidence = test_ncu_validation.pass_fixture("tensor")
        evidence["profile_context"]["profile_active_nvml_samples"][0]["graphics_clock_mhz"] = None
        result = validate_evidence(record, evidence)
        self.assertEqual(result["status"], "pass", result["reasons"])
        self.assertTrue(result["suitable_verified"])


if __name__ == "__main__":
    unittest.main()
