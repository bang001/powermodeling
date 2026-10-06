"""Independent accounting examples, not expected physical GPU measurements."""
import copy
import unittest

from powermodeling.analysis import analyze_trial, summarize
from test_analysis import paired_record, synthetic_trial


def powers(record, idle):
    # Use integration so changing the baseline cannot leave stale fake counters.
    for name, phase in record["phases"].items():
        for sample in phase["samples"]:
            sample.pop("energy_mj", None)
            if name.startswith("idle_"):
                sample["power_w"] = idle
    return record


class MeasurementDiagnosticsTests(unittest.TestCase):
    def test_h100_23_vs_15_can_be_scope_change_with_identical_work(self):
        record = powers(synthetic_trial("hbm", active_power=184, throughput=1e12), idle=64)
        record["benchmark"]["operations"] /= 4
        for epoch in record["benchmark"]["measure_epochs"]:
            epoch["operations"] /= 4
            epoch["counter_readback_ns"] = 10_000_000
        trial = analyze_trial(record)
        diagnostics = trial["measurement_diagnostics"]
        self.assertEqual(trial["total_pj_per_logical_bit"], 23)
        self.assertEqual(trial["operational_idle_increment_pj_per_logical_bit"], 15)
        self.assertAlmostEqual(diagnostics["scope_factors"]["total_over_idle_increment"], 23 / 15)
        self.assertEqual(diagnostics["denominator"]["measured_count"], 8e12 * 8)
        self.assertEqual(diagnostics["denominator"]["unit"], "logical bit")
        for objective, expected in (("total", 23), ("operational_idle_increment", 15)):
            actual = diagnostics["objectives"][objective]
            self.assertEqual(actual["reported_value"], expected)
            self.assertEqual(actual["reconstructed_value"], expected)
            self.assertEqual(actual["reconstruction_difference"], 0)
        units = diagnostics["unit_conventions"]
        self.assertEqual(units["aliases"]["pj_per_logical_byte"]["value_multiplier"], 8)
        self.assertEqual(units["logical_bits_per_reported_memory_operation"], 32)
        self.assertEqual(trial["pj_per_logical_byte"], 8 * 15)
        self.assertAlmostEqual(diagnostics["timing"]["counter_readback_fraction"], 0.01)
        self.assertEqual(diagnostics["timing"]["inter_epoch_gap_s"], 0)

    def test_tensor_flop_and_fma_provenance_does_not_rescale_measurement(self):
        trial = analyze_trial(powers(synthetic_trial("tensor", active_power=23, throughput=1e12), idle=8))
        diagnostics = trial["measurement_diagnostics"]
        self.assertEqual(diagnostics["denominator"]["unit"], "FLOP")
        self.assertEqual(diagnostics["unit_conventions"]["flops_per_fma"], 2)
        self.assertEqual(diagnostics["unit_conventions"]["aliases"]["pj_per_op"]["value_multiplier"], 1)
        self.assertEqual(trial["total_pj_per_flop"], 23)
        self.assertEqual(trial["operational_idle_increment_pj_per_flop"], 15)

    def test_legacy_rate_estimate_never_claims_exact_count_or_busy_time(self):
        record = synthetic_trial("hbm")
        record["benchmark"].pop("measure_epochs")
        record["benchmark"]["host_duration_s"] = 12.2
        record["benchmark"]["duration_scope"] = "CUDA event elapsed experiment window including gaps; not summed kernel busy time"
        trial = analyze_trial(record)
        diagnostics = trial["measurement_diagnostics"]
        self.assertFalse(diagnostics["denominator"]["exact_matching_window"])
        self.assertIsNone(diagnostics["denominator"]["measured_count"])
        self.assertIn("stationary", diagnostics["denominator"]["kind"])
        self.assertIsNone(diagnostics["timing"]["kernel_busy_time_s"])
        self.assertIn("not summed kernel busy time", diagnostics["timing"]["event_duration_scope"])
        self.assertAlmostEqual(diagnostics["objectives"]["total"]["reconstructed_value"], trial["total_pj_per_logical_bit"])

    def test_counter_disagreement_and_negative_contrast_stay_visible(self):
        record = synthetic_trial("tensor", active_power=25)
        for phase in record["phases"].values():
            for sample in phase["samples"]:
                sample["energy_mj"] *= 2
        trial = analyze_trial(record)
        diagnostics = trial["measurement_diagnostics"]
        self.assertEqual(diagnostics["telemetry_crosscheck"]["counter_over_integrated"], 2)
        self.assertEqual(diagnostics["telemetry_crosscheck"]["disagreement_fraction"], 1)
        self.assertFalse(diagnostics["quality"]["valid"])
        self.assertFalse(diagnostics["objectives"]["total"]["eligible"])
        self.assertIn("energy_counter_disagreement:measure", diagnostics["quality"]["issues"])
        negative = analyze_trial(synthetic_trial("tensor", active_power=25))["measurement_diagnostics"]
        self.assertLess(negative["objectives"]["operational_idle_increment"]["reported_value"], 0)
        self.assertIsNone(negative["scope_factors"]["total_over_idle_increment"])
        self.assertFalse(negative["scope_factors"]["idle_increment_eligible"])

    def test_paired_reference_fraction_uses_its_own_arm_power(self):
        trial = analyze_trial(paired_record(power=150, reference_power=80))
        diagnostics = trial["measurement_diagnostics"]
        self.assertAlmostEqual(diagnostics["power_contributions"]["paired_reference_fraction"], 80 / 150)
        self.assertAlmostEqual(diagnostics["scope_factors"]["total_over_paired_reference"], 150 / 70)
        self.assertEqual(diagnostics["objectives"]["paired_active_reference"]["power_w"], 70)
        self.assertAlmostEqual(diagnostics["objectives"]["paired_active_reference"]["reconstructed_value"], trial["paired_active_reference_pj_per_logical_bit"])

    def test_inter_epoch_gaps_are_counted_in_energy_window(self):
        record = synthetic_trial()
        # Remove complete one-second work interval, retaining a real idle gap.
        removed = record["benchmark"]["measure_epochs"].pop(5)
        record["benchmark"]["operations"] -= removed["operations"]
        record["benchmark"]["logical_bytes"] -= removed["logical_bytes"]
        trial = analyze_trial(record)
        timing = trial["measurement_diagnostics"]["timing"]
        self.assertEqual(timing["complete_epoch_span_s"], 8)
        self.assertEqual(timing["summed_epoch_duration_s"], 7)
        self.assertEqual(timing["inter_epoch_gap_s"], 1)
        self.assertEqual(trial["total_energy_j"], 1200)
        self.assertAlmostEqual(trial["total_pj_per_flop"], 1200 / 7)

    def test_group_reports_median_of_repeat_ratios_not_ratio_of_medians(self):
        records = []
        for index, (power, idle, rate) in enumerate(((100, 50, 1e12), (200, 100, 10e12), (300, 20, 1e12))):
            record = powers(synthetic_trial("tensor", active_power=power, throughput=rate, repeat=index), idle)
            if index == 2:
                for sample in record["phases"]["idle_post"]["samples"]:
                    sample["power_w"] = idle * 2
            records.append(record)
        group = summarize(records)["groups"][0]
        diagnostics = group["measurement_diagnostics"]
        self.assertEqual(diagnostics["valid_repeats"], 3)
        self.assertEqual(diagnostics["objectives"]["total"]["reported_value"], 100)
        self.assertEqual(diagnostics["objectives"]["total"]["reconstructed_value"], 100)
        self.assertEqual(diagnostics["scope_factors"]["total_over_idle_increment"], 2)
        self.assertIn("medians of per-repeat ratios", diagnostics["aggregation"])
        self.assertFalse(diagnostics["quality"]["baseline_valid"])
        self.assertTrue(diagnostics["quality"]["baseline_issues"])

    def test_no_valid_repeats_do_not_fabricate_diagnostics(self):
        record = copy.deepcopy(synthetic_trial())
        record["status"] = "failed"
        diagnostics = summarize([record])["groups"][0]["measurement_diagnostics"]
        self.assertEqual(diagnostics["valid_repeats"], 0)
        self.assertNotIn("objectives", diagnostics)

    def test_profiler_rate_is_a_separate_diagnostic_never_energy_denominator(self):
        record = paired_record(power=150, reference_power=80)
        evidence = record["validation"]["profiler_evidence"]
        trial = analyze_trial(record)
        crosscheck = trial["measurement_diagnostics"]["throughput_crosscheck"]
        expected_count = evidence["profile_benchmark"]["logical_bytes"] * 8
        self.assertEqual(crosscheck["profiler_replay_count"], expected_count)
        self.assertAlmostEqual(crosscheck["profiler_replay_rate_per_s"], expected_count / crosscheck["profiler_replay_summed_kernel_duration_s"])
        self.assertFalse(crosscheck["same_energy_run"])
        self.assertEqual(trial["measurement_diagnostics"]["denominator"]["measured_count"], trial["counted_measure_logical_bytes"] * 8)

    def test_resource_bounds_and_implementation_versions_survive_without_saturation_claim(self):
        records = []
        for version in ("baseline", "resource-diagnostics-1"):
            record = synthetic_trial()
            record["benchmark"]["kernel_implementation_version"] = version
            record["benchmark"]["kernel_resources"] = {"occupancy_upper_bound_fraction": 1.0, "registers_per_thread": 64}
            records.append(record)
        summary = summarize(records)
        self.assertEqual(len(summary["groups"]), 2)
        for group in summary["groups"]:
            execution = group["measurement_diagnostics"]["execution"]
            self.assertEqual(execution["kernel_resources"]["occupancy_upper_bound_fraction"], 1.0)
            self.assertEqual(execution["implementation_version"], group["kernel_implementation_version"])
            self.assertIn("do not establish", execution["saturation_status"])

    def test_group_paired_eligibility_requires_balanced_orders(self):
        records = [paired_record(order="AB") for _ in range(3)]
        for index, record in enumerate(records):
            record["config"]["repeat"] = index
        group = summarize(records)["groups"][0]
        self.assertFalse(group["paired_active_reference_eligible"])
        self.assertFalse(group["measurement_diagnostics"]["objectives"]["paired_active_reference"]["eligible"])
        self.assertFalse(group["measurement_diagnostics"]["scope_factors"]["paired_reference_eligible"])

    def test_optional_resource_metadata_is_safe_across_legacy_repeats(self):
        records = [synthetic_trial(repeat=index) for index in range(3)]
        records[1]["benchmark"]["kernel_resources"] = {"occupancy_upper_bound_fraction": 0.5}
        records[1]["benchmark"]["execution_diagnostics"] = {"notes": [{"kind": "bound"}]}
        group = summarize(records)["groups"][0]
        execution = group["measurement_diagnostics"]["execution"]
        self.assertEqual(execution["kernel_resources"]["occupancy_upper_bound_fraction"], 0.5)
        self.assertEqual(execution["execution_diagnostics"]["notes"], [{"kind": "bound"}])


if __name__ == "__main__":
    unittest.main()
