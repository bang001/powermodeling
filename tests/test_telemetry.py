import unittest
import threading
import time
from types import SimpleNamespace

from powermodeling.telemetry import NvmlDevice, Sampler, TelemetryError


class FakeNvml:
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_TEMPERATURE_GPU = 0
    NVML_POWER_SCOPE_GPU = 0
    NVML_POWER_SCOPE_MEMORY = 2
    NVML_FI_DEV_POWER_INSTANT = 186
    NVML_FI_DEV_POWER_AVERAGE = 185
    NVML_VALUE_TYPE_UNSIGNED_INT = 1
    NVML_SUCCESS = 0
    nvmlClocksThrottleReasonSwPowerCap = 4

    def __init__(self, cc=(9, 0), memory_supported=False):
        self.cc = cc
        self.memory_supported = memory_supported
        self.energy = 123456
        self.events = []

    def nvmlInit(self):
        self.events.append("init")

    def nvmlShutdown(self):
        self.events.append("shutdown")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.events.append(("uuid_lookup", uuid))
        return "handle"

    def nvmlDeviceGetHandleByIndex(self, index):
        self.events.append(("index_lookup", index))
        return "handle"

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-12345678-abcd-aaaa-bbbb-000000000001"

    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA H100 SXM"

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return self.cc

    def nvmlDeviceGetPowerUsage(self, handle):
        return 220000

    def nvmlDeviceGetFieldValues(self, handle, fields):
        result = []
        for field_id, scope in fields:
            unsupported = scope == self.NVML_POWER_SCOPE_MEMORY and not self.memory_supported
            result.append(SimpleNamespace(
                nvmlReturn=3 if unsupported else 0, valueType=1,
                value=SimpleNamespace(uiVal=42000 if scope == 2 else 200000),
                timestamp=987654321, latencyUsec=20,
            ))
        return result

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        return self.energy

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 1590 if clock != 2 else 2619

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 63

    def nvmlDeviceGetPerformanceState(self, handle):
        return 0

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle):
        return 4

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=100, memory=80)

    def nvmlDeviceGetPowerManagementLimit(self, handle):
        return 700000

    nvmlDeviceGetEnforcedPowerLimit = nvmlDeviceGetPowerManagementLimit

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=111, usedGpuMemory=None)]

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle):
        return []


class TelemetryTests(unittest.TestCase):
    def test_uuid_selection_is_exact_and_mig_rejected(self):
        fake = FakeNvml()
        uuid = fake.nvmlDeviceGetUUID(None)
        with NvmlDevice(uuid=uuid, nvml_module=fake) as device:
            self.assertEqual(device.uuid, uuid)
            self.assertIn(("uuid_lookup", uuid), fake.events)
            self.assertFalse(any(isinstance(v, tuple) and v[0] == "index_lookup" for v in fake.events))
        with self.assertRaises(TelemetryError):
            NvmlDevice(uuid="MIG-abc", nvml_module=fake)
        with self.assertRaises(TelemetryError):
            NvmlDevice(uuid="GPU-different", nvml_module=fake)

    def test_unsupported_h100_memory_rail_is_null_with_error(self):
        with NvmlDevice(nvml_module=FakeNvml()) as device:
            sample = device.read_sample()
            self.assertEqual(sample["power_w"], 220)
            self.assertEqual(sample["power_instant_w"], 200)
            self.assertIsNone(sample["memory_power_w"])
            self.assertIn("memory_power_instant_w", sample["errors"])
            self.assertFalse(device.capabilities["memory_power_instant_w"]["available"])
            self.assertEqual(sample["throttle_reason_names"], ["SwPowerCap"])
            self.assertIsNone(sample["compute_processes"][0]["used_gpu_memory_bytes"])

    def test_field_scope_timestamp_and_memory_source_preserved(self):
        with NvmlDevice(nvml_module=FakeNvml(memory_supported=True)) as device:
            sample = device.read_sample()
            self.assertEqual(sample["memory_power_w"], 42)
            self.assertEqual(sample["memory_power_source"], "memory_power_instant_w")
            meta = sample["field_metadata"]["memory_power_instant_w"]
            self.assertEqual(meta["scope"], "memory")
            self.assertEqual(meta["timestamp_us"], 987654321)
            self.assertEqual(meta["timestamp_clock"], "unix_epoch_microseconds")
            self.assertLessEqual(sample["query_realtime_start_s"], sample["query_realtime_end_s"])
            self.assertLessEqual(sample["query_start_s"], sample["t_s"])
            self.assertLessEqual(sample["t_s"], sample["query_end_s"])

    def test_h100_average_only_memory_power_preserves_sensor_semantics(self):
        fake = FakeNvml(memory_supported=True)
        original = fake.nvmlDeviceGetFieldValues
        timestamp_us = 1730000000123456

        def average_only(handle, fields):
            values = original(handle, fields)
            for value, (field_id, scope) in zip(values, fields):
                value.fieldId = field_id
                value.scopeId = scope
                if scope == fake.NVML_POWER_SCOPE_MEMORY:
                    if field_id == fake.NVML_FI_DEV_POWER_INSTANT:
                        value.nvmlReturn = 3
                    else:
                        value.value.uiVal = 43125
                        value.timestamp = timestamp_us
                        value.latencyUsec = 35
            return values

        fake.nvmlDeviceGetFieldValues = average_only
        with NvmlDevice(nvml_module=fake) as device:
            sample = device.read_sample()
            self.assertEqual(sample["memory_power_average_w"], 43.125)
            self.assertIsNone(sample["memory_power_instant_w"])
            self.assertEqual(sample["memory_power_w"], 43.125)
            self.assertEqual(sample["memory_power_source"], "memory_power_average_w")
            self.assertIn("error code 3", sample["errors"]["memory_power_instant_w"]["message"])
            self.assertFalse(device.capabilities["memory_power_instant_w"]["available"])
            self.assertTrue(device.capabilities["memory_power_average_w"]["available"])
            meta = sample["field_metadata"]["memory_power_average_w"]
            self.assertEqual(meta["field"], "NVML_FI_DEV_POWER_AVERAGE")
            self.assertEqual(meta["scope"], "memory")
            self.assertEqual(meta["scope_id"], fake.NVML_POWER_SCOPE_MEMORY)
            self.assertEqual(meta["timestamp_us"], timestamp_us)
            self.assertEqual(meta["timestamp_clock"], "unix_epoch_microseconds")
            self.assertEqual(meta["latency_us"], 35)
            self.assertEqual(meta["semantics"], "one_second_average")

    def test_malformed_memory_power_metadata_cannot_select_rejected_value(self):
        for field_id, key in ((186, "memory_power_instant_w"),
                              (185, "memory_power_average_w")):
            for attribute in ("timestamp", "latencyUsec"):
                with self.subTest(field=key, attribute=attribute):
                    fake = FakeNvml(memory_supported=True)
                    original = fake.nvmlDeviceGetFieldValues

                    def malformed(handle, fields):
                        values = original(handle, fields)
                        for value, (requested_id, scope) in zip(values, fields):
                            if scope == fake.NVML_POWER_SCOPE_MEMORY:
                                if requested_id == field_id:
                                    setattr(value, attribute, "invalid sensor metadata")
                                else:
                                    value.nvmlReturn = 3
                        return values

                    fake.nvmlDeviceGetFieldValues = malformed
                    with NvmlDevice(nvml_module=fake) as device:
                        sample = device.read_sample()
                        self.assertIsNone(sample[key])
                        self.assertIsNone(sample["memory_power_w"])
                        self.assertIsNone(sample["memory_power_source"])
                        self.assertNotIn(key, sample["field_metadata"])
                        self.assertEqual(sample["errors"][key]["type"], "ValueError")
                        self.assertFalse(device.capabilities[key]["available"])
                        self.assertFalse(device.capabilities["memory_power_w"]["available"])

    def test_failed_power_api_does_not_emit_zero(self):
        fake = FakeNvml()
        def unavailable(handle):
            raise RuntimeError("sensor not supported")
        fake.nvmlDeviceGetPowerUsage = unavailable
        with NvmlDevice(nvml_module=fake) as device:
            sample = device.sample()
            self.assertIsNone(sample["power_w"])
            self.assertEqual(sample["errors"]["power_w"]["message"], "sensor not supported")

    def test_counter_decrease_and_integer_precision(self):
        fake = FakeNvml()
        fake.energy = (1 << 54) + 1
        with NvmlDevice(nvml_module=fake) as device:
            self.assertEqual(device.read_sample()["energy_mj"], (1 << 54) + 1)
            fake.energy = 10
            sample = device.read_sample()
            self.assertTrue(sample["energy_counter_decreased"])
            self.assertIn("energy_counter", sample["errors"])

    def test_ga100_exception_and_hopper_average_labeled(self):
        for cc, expected in [((7, 0), "instantaneous"), ((8, 0), "instantaneous"), ((8, 6), "one_second_average"), ((9, 0), "one_second_average")]:
            with self.subTest(cc=cc), NvmlDevice(nvml_module=FakeNvml(cc)) as device:
                self.assertTrue(device.metadata()["power_usage_semantics"].startswith(expected))

    def test_shared_backend_not_shutdown_while_other_device_live(self):
        fake = FakeNvml()
        first = NvmlDevice(nvml_module=fake)
        second = NvmlDevice(nvml_module=fake)
        first.close()
        self.assertNotIn("shutdown", fake.events)
        self.assertEqual(second.read_sample()["power_w"], 220)
        second.close()
        self.assertEqual(fake.events.count("init"), 1)
        self.assertEqual(fake.events.count("shutdown"), 1)

    def test_sampler_brackets_and_device_close_protection(self):
        device = NvmlDevice(nvml_module=FakeNvml())
        sampler = Sampler(device, interval_s=0.002).start()
        with self.assertRaises(TelemetryError):
            device.close()
        samples = sampler.stop()
        self.assertGreaterEqual(len(samples), 2)
        self.assertEqual([s["t_s"] for s in samples], sorted(s["t_s"] for s in samples))
        self.assertEqual(sampler.stop(), samples)
        device.close()

    def test_mismatched_nvml_field_identity_cannot_be_used_as_memory_power(self):
        fake = FakeNvml(memory_supported=True)
        original = fake.nvmlDeviceGetFieldValues
        def wrong_scope(handle, fields):
            values = original(handle, fields)
            for value, (field_id, scope) in zip(values, fields):
                value.fieldId = field_id
                value.scopeId = fake.NVML_POWER_SCOPE_GPU
            return values
        fake.nvmlDeviceGetFieldValues = wrong_scope
        with NvmlDevice(nvml_module=fake) as device:
            sample = device.read_sample()
            self.assertIsNone(sample["memory_power_w"])
            self.assertIn("wrong power scope", sample["errors"]["memory_power_instant_w"]["message"])

    def test_energy_counter_rejects_fractional_and_negative_values(self):
        for value in (-1, 1.5, True):
            fake = FakeNvml()
            fake.energy = value
            with self.subTest(value=value), NvmlDevice(nvml_module=fake) as device:
                sample = device.read_sample()
                self.assertIsNone(sample["energy_mj"])
                self.assertIn("energy_mj", sample["errors"])

    def test_fatal_gpu_lost_is_preserved_and_sampler_does_not_continue(self):
        class GpuLost(RuntimeError):
            value = 15
        fake = FakeNvml()
        def lost(handle):
            raise GpuLost("GPU lost")
        fake.nvmlDeviceGetPowerUsage = lost
        device = NvmlDevice(nvml_module=fake)
        sampler = Sampler(device, interval_s=0.001)
        with self.assertRaisesRegex(TelemetryError, "Initial NVML sample failed"):
            sampler.start()
        with self.assertRaisesRegex(TelemetryError, "NVML sampler failed"):
            sampler.stop()
        self.assertEqual(len(sampler.samples), 1)
        self.assertIn("power_w", sampler.samples[0]["fatal_errors"])
        self.assertEqual(device._active_samplers, 0)
        device.close()

    def test_failed_final_bracket_is_bounded_worker_error_and_releases_device(self):
        device = NvmlDevice(nvml_module=FakeNvml())
        original = device.read_sample
        count = 0
        def fail_second():
            nonlocal count
            count += 1
            if count > 1:
                raise RuntimeError("final query failed")
            return original()
        device.read_sample = fail_second
        sampler = Sampler(device, interval_s=1).start()
        with self.assertRaisesRegex(TelemetryError, "final query failed"):
            sampler.stop()
        self.assertEqual(len(sampler.samples), 1)
        self.assertEqual(device._active_samplers, 0)
        device.close()

    def test_stuck_native_query_cannot_block_stop_or_unsafe_close(self):
        fake = FakeNvml()
        entered, release = threading.Event(), threading.Event()
        original = fake.nvmlDeviceGetPowerUsage
        queries = 0
        def block_second(handle):
            nonlocal queries
            queries += 1
            if queries == 2:
                entered.set()
                release.wait(timeout=2)
            return original(handle)
        fake.nvmlDeviceGetPowerUsage = block_second
        device = NvmlDevice(nvml_module=fake)
        sampler = Sampler(device, interval_s=0.001).start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            started = time.monotonic()
            with self.assertRaisesRegex(TelemetryError, "did not stop"):
                sampler.stop(timeout_s=0.002)
            with self.assertRaisesRegex(TelemetryError, "Stop all telemetry samplers"):
                device.close()
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertNotIn("shutdown", fake.events)
        finally:
            release.set()
            sampler.stop(timeout_s=1)
            device.close()


if __name__ == "__main__":
    unittest.main()
