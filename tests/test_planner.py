"""Plan failures must precede device allocation or clock mutation."""
import copy
import unittest

from powermodeling.planner import benchmark_command, expand_plan, numeric_expression, resolve_clocks


class PlannerTests(unittest.TestCase):
    def device(self):
        return {"sm_count": 108, "l2_bytes": 40 * 1024**2,
                "total_memory_bytes": 40 * 1024**3, "benchmark_sha256": "synthetic-binary"}

    def config(self, workload="tensor", parameters=None):
        return {"clock_pairs": [{"graphics_mhz": 1200, "memory_mhz": 1215}],
                "experiments": [{"workload": workload, "parameters": parameters or {}}]}

    def test_nonfinite_times_and_fractional_repeats_cannot_pass_constraints(self):
        for field, value in (("seconds", float("nan")), ("warmup_seconds", float("inf")),
                             ("idle_seconds", float("nan")), ("repeats", 3.5), ("repeats", True)):
            with self.subTest(field=field):
                config = self.config()
                config[field] = value
                with self.assertRaises(ValueError):
                    expand_plan(config, self.device())

    def test_numeric_expressions_are_integer_and_restricted(self):
        self.assertEqual(numeric_expression("max(l2_bytes/4,4096)", {"l2_bytes": 16384}), 4096)
        for expression in ("1.5", "2**4", "__import__('os')", "[1,2]", "not 0"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                numeric_expression(expression, {})

    def test_duplicate_clock_or_resolved_conditions_cannot_duplicate_trial_ids(self):
        config = self.config()
        config["clock_pairs"] *= 2
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())
        config = self.config()
        config["experiments"][0]["grid"] = {"blocks": [108, "sm_count"]}
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())

    def test_offset_and_write_buffer_allocations_are_included(self):
        config = self.config("l2", {"working_set_bytes": 1024, "offset_bytes": 16 * 1024**3, "access": "write"})
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())
        config = self.config("gemm", {"gemm_m": 131072, "gemm_n": 131072, "gemm_k": 4096})
        with self.assertRaises(ValueError):
            expand_plan(config, self.device())

    def test_word_alignment_and_latency_access_semantics(self):
        for workload, params in (("l1", {"working_set_bytes": 3}), ("l2", {"offset_bytes": 2}),
                                 ("l2_latency", {"access": "copy"}), ("l2_latency", {"stride_elements": 2}),
                                 ("tensor", {"tensor_accumulators": 9})):
            with self.subTest(workload=workload, params=params), self.assertRaises(ValueError):
                expand_plan(self.config(workload, params), self.device())

    def test_sm_ids_are_sparse_identifiers_but_discovered_map_is_enforced(self):
        config = self.config("l2", {"sm_ids": [130]})
        self.assertEqual(expand_plan(config, self.device())["trials"][0]["parameters"]["sm_ids"], [130])
        device = self.device()
        device["discovered_sm_ids"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            expand_plan(config, device)
        with self.assertRaises(ValueError):
            expand_plan(self.config("gemm", {"sm_ids": [0]}), self.device())

    def test_hbm_stride_aliasing_and_default_footprint(self):
        with self.assertRaises(ValueError):
            expand_plan(self.config("hbm", {"working_set_bytes": "l2_bytes*8", "stride_elements": 32}), self.device())
        self.assertEqual(len(expand_plan(self.config("hbm"), self.device())["trials"]), 3)

    def test_profile_command_has_explicit_region_and_one_measured_batch(self):
        trial = expand_plan(self.config(), self.device())["trials"][0]
        command = benchmark_command("bench", trial, profiling=True)
        self.assertIn("--profile-region", command)
        self.assertEqual(command[command.index("--fixed-batches") + 1], "1")

    def test_shuffle_is_reproducible_and_condition_hash_preserves_data_seed(self):
        config = self.config("l2")
        config["experiments"][0]["grid"] = {"seed": [2026, 2027]}
        a, b = expand_plan(config, self.device()), expand_plan(copy.deepcopy(config), self.device())
        self.assertEqual(a["trials"], b["trials"])
        self.assertEqual(len({trial["condition_id"] for trial in a["trials"]}), 2)


if __name__ == "__main__":
    unittest.main()
