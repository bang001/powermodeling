import unittest

from powermodeling.clocks import ClockError, ClockRestoreError, clock_context, discover


class ClockDevice:
    uuid = "GPU-test"

    def __init__(self):
        self.nvml = self
        self.application_graphics = 1200
        self.application_memory = 1215
        self.events = []
        self.fail_gpu_set = False
        self.fail_restore = False

    def call(self, function, *args):
        return getattr(self, function)(*args)

    def nvmlDeviceGetClockInfo(self, clock):
        return 210 if clock == 0 else 405

    def nvmlDeviceGetApplicationsClock(self, clock):
        return self.application_graphics if clock == 0 else self.application_memory

    nvmlDeviceGetDefaultApplicationsClock = nvmlDeviceGetApplicationsClock

    def nvmlDeviceGetSupportedMemoryClocks(self):
        return [1215, 1593]

    def nvmlDeviceGetSupportedGraphicsClocks(self, memory):
        return [900, 1200, 1410]

    def nvmlDeviceSetApplicationsClocks(self, memory, graphics):
        if self.fail_restore and graphics == 1200:
            raise RuntimeError("restore permission denied")
        self.events.append(("applications", memory, graphics))
        self.application_memory = memory
        self.application_graphics = graphics

    def nvmlDeviceSetGpuLockedClocks(self, minimum, maximum):
        if self.fail_gpu_set:
            raise RuntimeError("GPU lock failed")
        self.events.append(("lock_graphics", minimum, maximum))

    def nvmlDeviceSetMemoryLockedClocks(self, minimum, maximum):
        self.events.append(("lock_memory", minimum, maximum))

    def nvmlDeviceResetGpuLockedClocks(self):
        self.events.append(("reset_graphics",))

    def nvmlDeviceResetMemoryLockedClocks(self):
        self.events.append(("reset_memory",))


class ClockTests(unittest.TestCase):
    def test_discovery_and_default_context_are_read_only(self):
        device = ClockDevice()
        self.assertEqual(len(discover(device)["supported_pairs"]), 6)
        with clock_context(device) as record:
            self.assertEqual(record["method"], "uncontrolled")
        self.assertEqual(device.events, [])

    def test_mutation_and_unadvertised_pair_require_explicit_valid_request(self):
        device = ClockDevice()
        with self.assertRaises(ClockError):
            with clock_context(device, graphics_mhz=900):
                pass
        with self.assertRaises(ClockError):
            with clock_context(device, graphics_mhz=1234, memory_mhz=1215, allow_mutation=True):
                pass
        self.assertEqual(device.events, [])

    def test_applications_restore_prior_pair_not_default_or_idle_clocks(self):
        device = ClockDevice()
        with self.assertRaisesRegex(ValueError, "workload failed"):
            with clock_context(device, graphics_mhz=900, memory_mhz=1593, allow_mutation=True) as record:
                self.assertEqual(device.application_graphics, 900)
                raise ValueError("workload failed")
        self.assertEqual(device.events, [("applications", 1593, 900), ("applications", 1215, 1200)])
        self.assertTrue(record["restored"])

    def test_locked_method_refuses_unknown_preexisting_policy_before_any_write(self):
        device = ClockDevice()
        with self.assertRaisesRegex(ClockError, "unknown"):
            with clock_context(device, graphics_mhz=900, memory_mhz=1215, method="locked", allow_mutation=True):
                pass
        self.assertEqual(device.events, [])

    def test_locked_exact_prior_range_and_owned_unlocked_reset(self):
        device = ClockDevice()
        with clock_context(device, graphics_mhz=900, memory_mhz=1215, method="locked", allow_mutation=True,
                           locked_restore={"graphics": [1200, 1410], "memory": None}) as record:
            self.assertTrue(record["applied"])
        self.assertEqual(device.events, [("lock_memory", 1215, 1215), ("lock_graphics", 900, 900),
                                         ("lock_graphics", 1200, 1410), ("reset_memory",)])

    def test_partial_clock_setup_failure_rolls_back_first_domain(self):
        device = ClockDevice()
        device.fail_gpu_set = True
        with self.assertRaisesRegex(RuntimeError, "GPU lock failed"):
            with clock_context(device, graphics_mhz=900, memory_mhz=1215, method="locked", allow_mutation=True,
                               locked_restore={"graphics": None, "memory": None}):
                pass
        self.assertEqual(device.events, [("lock_memory", 1215, 1215), ("reset_graphics",), ("reset_memory",)])

    def test_error_after_driver_mutation_still_restores_prior_application_pair(self):
        device = ClockDevice()
        original_set = device.nvmlDeviceSetApplicationsClocks
        def mutate_then_fail(memory, graphics):
            original_set(memory, graphics)
            if graphics == 900:
                raise RuntimeError("set failed after mutation")
        device.nvmlDeviceSetApplicationsClocks = mutate_then_fail
        with self.assertRaisesRegex(RuntimeError, "after mutation") as caught:
            with clock_context(device, graphics_mhz=900, memory_mhz=1593, allow_mutation=True):
                pass
        self.assertEqual((device.application_graphics, device.application_memory), (1200, 1215))
        self.assertTrue(caught.exception.clock_record["restored"])

    def test_partial_locked_failure_restores_only_requested_domains(self):
        device = ClockDevice()
        device.fail_gpu_set = True
        with self.assertRaisesRegex(RuntimeError, "GPU lock failed"):
            with clock_context(device, graphics_mhz=900, method="locked", allow_mutation=True,
                               locked_restore={"graphics": None, "memory": [1215, 1593]}):
                pass
        self.assertEqual(device.events, [("reset_graphics",)])

    def test_restoration_failure_is_fatal_and_recorded(self):
        device = ClockDevice()
        with self.assertRaises(ClockRestoreError):
            with clock_context(device, graphics_mhz=900, allow_mutation=True) as record:
                device.fail_restore = True
        self.assertFalse(record["restored"])
        self.assertIn("applications", record["restore_errors"])


if __name__ == "__main__":
    unittest.main()
