"""Canonical nonlinear plans mean register SFU; historical streaming stays explicit."""

import json
import os
from pathlib import Path
import subprocess
import unittest

from powermodeling.analysis import summarize
from powermodeling.nonlinear import NONLINEAR_WORKLOADS
from powermodeling.planner import benchmark_command, expand_plan
from powermodeling.sfu import SFU_WORKLOADS
from test_integration_review import DEVICE
from test_nonlinear import nonlinear_trial
from test_nonlinear_q_grid import q_grid_trial
from test_sfu import REFERENCE_KIND, sfu_trial


ROOT = Path(__file__).resolve().parents[1]
CLOCKS = {"supported_pairs": [{"graphics_mhz": graphics, "memory_mhz": 1000}
                               for graphics in (900, 1110, 1200)],
          "default_applications_graphics_mhz": 1200, "default_applications_memory_mhz": 1000}
CANONICAL = (("nonlinear.json", "sfu-register.json"),
             ("nonlinear-smoke.json", "sfu-register-smoke.json"),
             ("nonlinear-amortization.json", "sfu-register-amortization.json"))


class NonlinearMigrationTests(unittest.TestCase):
    def test_canonical_configs_and_compatibility_aliases_are_same_register_sfu_study(self):
        forbidden = {"working_set_bytes", "row_width", "nonlinear_mode"}
        for canonical, alias in CANONICAL:
            for major in (7, 8, 9):
                with self.subTest(config=canonical, major=major):
                    device = {**DEVICE, "compute_capability_major": major, "compute_capability_minor": 0}
                    config = json.loads((ROOT / "configs" / canonical).read_text())
                    plan = expand_plan(config, device, CLOCKS)
                    compatibility = expand_plan(json.loads((ROOT / "configs" / alias).read_text()), device, CLOCKS)
                    self.assertEqual(plan["trials"], compatibility["trials"])
                    expected = SFU_WORKLOADS - {"sfu_tanh"} if major == 7 else SFU_WORKLOADS
                    self.assertEqual({trial["workload"] for trial in plan["trials"]}, expected)
                    self.assertEqual([entry["workload"] for entry in plan["unsupported_experiments"]],
                                     ["sfu_tanh"] if major == 7 else [])
                    for trial in plan["trials"]:
                        self.assertFalse(forbidden & trial["parameters"].keys())
                        self.assertEqual(trial["treatment_protocol"]["reference_kind"], REFERENCE_KIND)
                        self.assertIn("--paired-reference", benchmark_command("bench", trial))
                    self.assertTrue(all(experiment["workload"] not in NONLINEAR_WORKLOADS
                                        for experiment in config["experiments"]))

    def test_canonical_smoke_expands_to_six_or_five_native_ops_with_balanced_reference(self):
        config = json.loads((ROOT / "configs/nonlinear-smoke.json").read_text())
        for major, count in ((7, 20), (8, 24), (9, 24)):
            with self.subTest(major=major):
                plan = expand_plan(config, {**DEVICE, "compute_capability_major": major,
                                            "compute_capability_minor": 0}, CLOCKS)
                self.assertEqual(len(plan["trials"]), count)
                for workload in {trial["workload"] for trial in plan["trials"]}:
                    orders = [trial["treatment_protocol"]["order"] for trial in plan["trials"]
                              if trial["workload"] == workload]
                    self.assertEqual(orders.count("AB"), 2)
                    self.assertEqual(orders.count("BA"), 2)

    def test_streaming_plans_require_explicit_mode_and_sfu_rejects_that_mode(self):
        for workload in sorted(NONLINEAR_WORKLOADS):
            for mode in (None, "auto", "sfu"):
                with self.subTest(workload=workload, mode=mode):
                    parameters = {} if mode is None else {"nonlinear_mode": mode}
                    config = {"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                              "experiments": [{"workload": workload, "parameters": parameters}]}
                    with self.assertRaisesRegex(ValueError, "nonlinear_mode='streaming'"):
                        expand_plan(config, DEVICE)
            config["experiments"][0]["parameters"] = {"nonlinear_mode": "streaming"}
            trial = expand_plan(config, DEVICE)["trials"][0]
            command = benchmark_command("bench", trial)
            self.assertEqual(command[command.index("--nonlinear-mode") + 1], "streaming")
        with self.assertRaisesRegex(ValueError, "only to legacy streaming"):
            expand_plan({"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                         "experiments": [{"workload": "sfu_ex2", "parameters": {"nonlinear_mode": "streaming"}}]}, DEVICE)

    def test_archived_streaming_configs_keep_complete_function_contract_explicit(self):
        for filename in ("nonlinear-streaming.json", "nonlinear-streaming-smoke.json"):
            with self.subTest(config=filename):
                config = json.loads((ROOT / "configs/legacy" / filename).read_text())
                plan = expand_plan(config, DEVICE, CLOCKS)
                self.assertEqual({trial["workload"] for trial in plan["trials"]}, NONLINEAR_WORKLOADS)
                self.assertTrue(all(trial["parameters"]["nonlinear_mode"] == "streaming" for trial in plan["trials"]))
                self.assertTrue(all(trial["parameters"]["working_set_bytes"] > 0 for trial in plan["trials"]))
                self.assertTrue(all(trial["treatment_protocol"]["reference_kind"] == "issue_loop" for trial in plan["trials"]))

    def test_stale_stage_names_fail_loudly_and_valid_stages_remain_selectable(self):
        for canonical, _ in CANONICAL:
            config = json.loads((ROOT / "configs" / canonical).read_text())
            for stage in ("pointwise", "rowwise", "nonlinear-smoke", "misspelled"):
                with self.subTest(config=canonical, stage=stage), self.assertRaisesRegex(ValueError, "Unknown stage.*available stages"):
                    expand_plan(config, DEVICE, CLOCKS, stage=stage)
            stage = config["experiments"][0]["stage"]
            self.assertTrue(expand_plan(config, DEVICE, CLOCKS, stage=stage)["trials"])

    def test_matching_stage_can_explicitly_skip_unsupported_native_tanh(self):
        config = {"clock_pairs": [{"graphics_mhz": None, "memory_mhz": None}],
                  "experiments": [{"workload": "sfu_tanh", "stage": "native-tanh",
                                   "skip_if_unsupported": True}]}
        volta = {**DEVICE, "compute_capability_major": 7, "compute_capability_minor": 0}
        plan = expand_plan(config, volta, stage="native-tanh")
        self.assertEqual(plan["trials"], [])
        self.assertEqual(plan["unsupported_experiments"][0]["workload"], "sfu_tanh")
        self.assertEqual(plan["unsupported_experiments"][0]["stage"], "native-tanh")
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            expand_plan(config, volta, stage="misspelled")

    def test_historical_streaming_without_mode_still_analyzes_separately_from_sfu(self):
        records = []
        for repeat in range(4):
            records += [nonlinear_trial("exp", repeat=repeat), q_grid_trial(repeat=repeat),
                        sfu_trial(repeat=repeat, order="AB" if repeat % 2 == 0 else "BA")]
        self.assertTrue(all("nonlinear_mode" not in record["config"] for record in records))
        summary = summarize(records)
        self.assertTrue(all(trial["valid"] for trial in summary["trials"]))
        self.assertEqual(len(summary["groups"]), 3)
        self.assertEqual({group["valid_repeats"] for group in summary["groups"]}, {4})
        self.assertEqual(len(summary["evaluation"]["components"]), 3)
        for trial in summary["trials"]:
            if trial["workload"] == "exp":
                self.assertIsNotNone(trial["total_pj_per_element"])
                self.assertIsNone(trial.get("total_pj_per_instruction"))
            else:
                self.assertIsNotNone(trial["sfu_reference_delta_pj_per_instruction"])
                self.assertIsNone(trial["total_pj_per_element"])


@unittest.skipUnless(os.environ.get("POWERBENCH_CLI_TESTS") == "1",
                     "requires a freshly compiled powerbench; opt in with POWERBENCH_CLI_TESTS=1, no GPU needed")
class NonlinearCliGuardTests(unittest.TestCase):
    def test_streaming_guard_rejections_happen_before_cuda_driver_calls(self):
        binary = os.environ.get("POWERBENCH", "build/powerbench")
        cases = ((["--workload", "exp"], "--nonlinear-mode streaming"),
                 (["--workload", "softmax", "--nonlinear-mode", "sfu"], "--nonlinear-mode streaming"),
                 (["--workload", "sfu_ex2", "--nonlinear-mode", "streaming"], "applies only to complete"))
        for parameters, reason in cases:
            with self.subTest(parameters=parameters):
                completed = subprocess.run([binary, *parameters], text=True, capture_output=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(reason, completed.stderr)
                self.assertFalse(any(json.loads(line).get("type") in ("device", "result")
                                     for line in completed.stdout.splitlines() if line.startswith("{")))


if __name__ == "__main__":
    unittest.main()
