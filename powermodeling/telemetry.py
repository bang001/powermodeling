"""Capability-probed NVML telemetry; importing this module never initializes NVML.

``power_w`` preserves ``nvmlDeviceGetPowerUsage`` device/circuitry scope. The
explicit power fields retain their requested scopes and sensor timestamps.
Unsupported values are null with an error, never a fabricated zero. Query
midpoints are host CLOCK_MONOTONIC times, not the sensor's acquisition time.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from typing import Any


class TelemetryError(RuntimeError):
    """An operation needed to identify or monitor the selected GPU failed."""


_lifecycle_lock = threading.Lock()
_sessions: dict[int, int] = {}


def _error(exc: Exception) -> dict[str, Any]:
    result = {"type": type(exc).__name__, "message": str(exc)}
    if getattr(exc, "value", None) is not None:
        result["code"] = int(exc.value)
    return result


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid nonnegative telemetry value: {value!r}")
    return result


def _integer(value: Any) -> int:
    """Validate counters without converting their potentially large integers to float."""
    result = int(value)
    if isinstance(value, bool) or result < 0 or result != value:
        raise ValueError(f"Invalid nonnegative integer telemetry value: {value!r}")
    return result


class NvmlDevice:
    """Select a physical NVML device by exact UUID (preferred) or NVML index.

    An NVML index is not a CUDA ordinal: CUDA_VISIBLE_DEVICES may reorder them.
    The runner must compare the benchmark's CUDA UUID to ``device.uuid``. MIG
    UUIDs are rejected because board power cannot be attributed to one slice.
    ``nvml_module`` is a dependency-injection hook for hardware-free tests.
    """

    def __init__(self, index: int = 0, uuid: str | None = None, nvml_module: Any = None):
        if uuid is not None and not uuid.startswith("GPU-"):
            raise TelemetryError("Use a complete physical GPU UUID beginning GPU-; MIG UUIDs are unsupported")
        try:
            self.nvml = nvml_module or importlib.import_module("pynvml")
        except ImportError as exc:
            raise TelemetryError("Install the optional GPU dependency: pip install nvidia-ml-py") from exc
        self._lock = threading.RLock()
        self._closed = False
        self._active_samplers = 0
        self._last_energy_mj: int | None = None
        self.capabilities: dict[str, Any] = {}
        with _lifecycle_lock:
            key = id(self.nvml)
            if _sessions.get(key, 0) == 0:
                self.nvml.nvmlInit()
            _sessions[key] = _sessions.get(key, 0) + 1
        try:
            self.handle = (self.nvml.nvmlDeviceGetHandleByUUID(uuid)
                           if uuid is not None else self.nvml.nvmlDeviceGetHandleByIndex(index))
            self.uuid = _text(self.nvml.nvmlDeviceGetUUID(self.handle))
            if not self.uuid.startswith("GPU-"):
                raise TelemetryError("Physical GPU handle required; cannot use a MIG device for board telemetry")
            if uuid is not None and self.uuid.lower() != uuid.lower():
                raise TelemetryError(f"GPU UUID mismatch: requested {uuid}, resolved {self.uuid}")
            self.index = index if uuid is None else None
        except Exception:
            self.close()
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise TelemetryError("NVML device is closed")

    def call(self, name: str, *args: Any) -> Any:
        """Call a backend function under the per-device telemetry lock."""
        if not self._lock.acquire(timeout=1):
            raise TelemetryError("Another NVML query is still running; refusing concurrent driver access")
        try:
            self._check_open()
            return getattr(self.nvml, name)(self.handle, *args)
        finally:
            self._lock.release()

    def _query(self, result: dict, key: str, function: str, *args: Any,
               transform=None) -> Any:
        try:
            value = self.call(function, *args)
            value = transform(value) if transform else value
            result[key] = value
            self.capabilities[key] = {"available": True, "api": function}
            return value
        except Exception as exc:
            result[key] = None
            result["errors"][key] = _error(exc)
            self.capabilities[key] = {"available": False, "api": function, "error": _error(exc)}
            return None

    def _power_fields(self, result: dict) -> None:
        result["memory_power_source"] = None
        specs = [
            ("power_instant_w", "NVML_FI_DEV_POWER_INSTANT", "NVML_POWER_SCOPE_GPU", "gpu"),
            ("power_average_w", "NVML_FI_DEV_POWER_AVERAGE", "NVML_POWER_SCOPE_GPU", "gpu"),
            ("memory_power_instant_w", "NVML_FI_DEV_POWER_INSTANT", "NVML_POWER_SCOPE_MEMORY", "memory"),
            ("memory_power_average_w", "NVML_FI_DEV_POWER_AVERAGE", "NVML_POWER_SCOPE_MEMORY", "memory"),
        ]
        pending = []
        for key, field_name, scope_name, scope_label in specs:
            result[key] = None
            try:
                # Missing scope constants cannot be replaced with guessed IDs.
                pending.append((key, field_name, scope_label,
                                getattr(self.nvml, field_name), getattr(self.nvml, scope_name)))
            except AttributeError as exc:
                result["errors"][key] = _error(exc)
                self.capabilities[key] = {"available": False, "error": _error(exc)}
        if not pending:
            result["memory_power_w"] = None
            return
        try:
            fields = self.call("nvmlDeviceGetFieldValues", [(p[3], p[4]) for p in pending])
            if len(fields) != len(pending):
                raise TelemetryError("NVML field query returned the wrong number of values")
        except Exception as exc:
            for key, field_name, scope_label, _, _ in pending:
                result["errors"][key] = _error(exc)
                self.capabilities[key] = {"available": False, "field": field_name,
                                          "scope": scope_label, "error": _error(exc)}
            result["memory_power_w"] = None
            return
        for (key, field_name, scope_label, _, scope_id), field in zip(pending, fields):
            try:
                code = int(field.nvmlReturn)
                if code != int(getattr(self.nvml, "NVML_SUCCESS", 0)):
                    error_class = getattr(self.nvml, "NVMLError", None)
                    if error_class is not None:
                        raise error_class(code)
                    raise TelemetryError(f"NVML field returned error code {code}")
                if hasattr(field, "fieldId") and int(field.fieldId) != getattr(self.nvml, field_name):
                    raise TelemetryError("NVML returned a field with the wrong field ID")
                if hasattr(field, "scopeId") and int(field.scopeId) != scope_id:
                    raise TelemetryError("NVML returned a field with the wrong power scope")
                value_type = int(field.valueType)
                member = {
                    getattr(self.nvml, "NVML_VALUE_TYPE_DOUBLE", 0): "dVal",
                    getattr(self.nvml, "NVML_VALUE_TYPE_UNSIGNED_INT", 1): "uiVal",
                    getattr(self.nvml, "NVML_VALUE_TYPE_UNSIGNED_LONG", 2): "ulVal",
                    getattr(self.nvml, "NVML_VALUE_TYPE_UNSIGNED_LONG_LONG", 3): "ullVal",
                    getattr(self.nvml, "NVML_VALUE_TYPE_SIGNED_LONG_LONG", 4): "sllVal",
                }.get(value_type)
                if member is None:
                    raise TelemetryError(f"Unsupported NVML value type {value_type}")
                result[key] = _number(getattr(field.value, member)) / 1000.0
                result["field_metadata"][key] = {
                    "field": field_name, "scope": scope_label, "scope_id": scope_id,
                    "timestamp_us": int(field.timestamp),
                    "timestamp_clock": "unix_epoch_microseconds",
                    "latency_us": int(field.latencyUsec),
                    "semantics": "one_second_average" if "average" in key else "driver_instantaneous",
                }
                self.capabilities[key] = {"available": True, "field": field_name, "scope": scope_label}
            except Exception as exc:
                result["errors"][key] = _error(exc)
                self.capabilities[key] = {"available": False, "field": field_name,
                                          "scope": scope_label, "error": _error(exc)}
        # Prefer memory instant when available. Record the actual source; a GPU
        # architecture name never implies that a memory rail is readable.
        source = ("memory_power_instant_w" if result["memory_power_instant_w"] is not None
                  else "memory_power_average_w" if result["memory_power_average_w"] is not None else None)
        result["memory_power_w"] = result[source] if source else None
        result["memory_power_source"] = source
        self.capabilities["memory_power_w"] = {"available": source is not None, "source": source, "scope": "memory"}

    def read_sample(self) -> dict[str, Any]:
        with self._lock:
            self._check_open()
            start = time.monotonic()
            realtime_start = time.time()
            result: dict[str, Any] = {"uuid": self.uuid, "errors": {}, "field_metadata": {}}
            self._query(result, "power_w", "nvmlDeviceGetPowerUsage", transform=lambda v: _number(v) / 1000)
            self._power_fields(result)
            energy = self._query(result, "energy_mj", "nvmlDeviceGetTotalEnergyConsumption", transform=self._energy)
            result["energy_counter_decreased"] = False
            if energy is not None:
                if self._last_energy_mj is not None and energy < self._last_energy_mj:
                    result["energy_counter_decreased"] = True
                    result["errors"]["energy_counter"] = {"type": "CounterDecrease", "message": "Energy counter decreased: driver reset or counter rollover; do not subtract across this boundary"}
                self._last_energy_mj = energy
            graphics = getattr(self.nvml, "NVML_CLOCK_GRAPHICS", 0)
            sm_clock = getattr(self.nvml, "NVML_CLOCK_SM", 1)
            memory = getattr(self.nvml, "NVML_CLOCK_MEM", 2)
            temperature = getattr(self.nvml, "NVML_TEMPERATURE_GPU", 0)
            self._query(result, "graphics_clock_mhz", "nvmlDeviceGetClockInfo", graphics, transform=_integer)
            self._query(result, "sm_clock_mhz", "nvmlDeviceGetClockInfo", sm_clock, transform=_integer)
            self._query(result, "memory_clock_mhz", "nvmlDeviceGetClockInfo", memory, transform=_integer)
            self._query(result, "temperature_c", "nvmlDeviceGetTemperature", temperature, transform=int)
            self._query(result, "pstate", "nvmlDeviceGetPerformanceState", transform=int)
            throttle_api = ("nvmlDeviceGetCurrentClocksEventReasons"
                            if hasattr(self.nvml, "nvmlDeviceGetCurrentClocksEventReasons")
                            else "nvmlDeviceGetCurrentClocksThrottleReasons")
            throttle = self._query(result, "throttle_reasons", throttle_api, transform=int)
            result["throttle_reason_names"] = self._throttle_names(throttle) if throttle is not None else None
            util = self._query(result, "utilization", "nvmlDeviceGetUtilizationRates",
                               transform=lambda v: {"gpu_percent": int(v.gpu), "memory_percent": int(v.memory)})
            result["gpu_utilization_percent"] = util["gpu_percent"] if util else None
            result["memory_utilization_percent"] = util["memory_percent"] if util else None
            self._query(result, "power_limit_w", "nvmlDeviceGetPowerManagementLimit", transform=lambda v: _number(v) / 1000)
            self._query(result, "enforced_power_limit_w", "nvmlDeviceGetEnforcedPowerLimit", transform=lambda v: _number(v) / 1000)
            self._query(result, "compute_processes", "nvmlDeviceGetComputeRunningProcesses", transform=self._processes)
            self._query(result, "graphics_processes", "nvmlDeviceGetGraphicsRunningProcesses", transform=self._processes)
            self._query(result, "mps_compute_processes", "nvmlDeviceGetMPSComputeRunningProcesses", transform=self._processes)
            end = time.monotonic()
            result.update(t_s=(start + end) / 2, query_start_s=start,
                          query_end_s=end, query_duration_s=end - start,
                          query_realtime_start_s=realtime_start,
                          query_realtime_end_s=time.time())
            fatal_codes = {getattr(self.nvml, name, default) for name, default in (
                ("NVML_ERROR_UNINITIALIZED", 1), ("NVML_ERROR_GPU_IS_LOST", 15),
                ("NVML_ERROR_RESET_REQUIRED", 16), ("NVML_ERROR_LIB_RM_VERSION_MISMATCH", 18))}
            result["fatal_errors"] = {key: value for key, value in result["errors"].items()
                                      if value.get("code") in fatal_codes}
            return result

    sample = read_sample

    @staticmethod
    def _energy(value: Any) -> int:
        return _integer(value)

    def _processes(self, processes: Any) -> list[dict[str, Any]]:
        unavailable = getattr(self.nvml, "NVML_VALUE_NOT_AVAILABLE", (1 << 64) - 1)
        result = []
        for process in processes:
            used = getattr(process, "usedGpuMemory", None)
            pid = _integer(process.pid)
            if pid == 0:
                raise ValueError("NVML process inventory must contain positive process IDs")
            result.append({"pid": pid, "used_gpu_memory_bytes": None if used in (None, unavailable) else _integer(used)})
        return result

    def _throttle_names(self, value: int) -> list[str]:
        names = []
        seen_bits = set()
        for prefix in ("nvmlClocksEventReason", "nvmlClocksThrottleReason"):
            for name in sorted(dir(self.nvml)):
                if name.startswith(prefix) and not name.endswith(("All", "None")):
                    bit = getattr(self.nvml, name)
                    if isinstance(bit, int) and bit > 0 and bit & (bit - 1) == 0 and value & bit and bit not in seen_bits:
                        names.append(name[len(prefix):])
                        seen_bits.add(bit)
        return names

    def metadata(self) -> dict[str, Any]:
        """Read metadata and probe monitoring capabilities without changing policy."""
        result: dict[str, Any] = {"uuid": self.uuid, "nvml_index": self.index, "errors": {}}
        self._query(result, "name", "nvmlDeviceGetName", transform=_text)
        self._query(result, "compute_capability", "nvmlDeviceGetCudaComputeCapability", transform=lambda v: list(v))
        self._query(result, "memory", "nvmlDeviceGetMemoryInfo",
                    transform=lambda v: {"total_bytes": int(v.total), "used_bytes": int(v.used), "free_bytes": int(v.free)})
        self._query(result, "pci_bus_id", "nvmlDeviceGetPciInfo", transform=lambda v: _text(v.busId))
        self._query(result, "mig_mode", "nvmlDeviceGetMigMode", transform=lambda v: {"current": int(v[0]), "pending": int(v[1])})
        self._query(result, "persistence_mode", "nvmlDeviceGetPersistenceMode", transform=int)
        self._query(result, "ecc_mode", "nvmlDeviceGetEccMode", transform=lambda v: {"current": int(v[0]), "pending": int(v[1])})
        self._query(result, "power_limit_constraints_w", "nvmlDeviceGetPowerManagementLimitConstraints", transform=lambda v: [v[0] / 1000, v[1] / 1000])
        self._query(result, "default_power_limit_w", "nvmlDeviceGetPowerManagementDefaultLimit", transform=lambda v: v / 1000)
        for key, api in (("driver_version", "nvmlSystemGetDriverVersion"), ("nvml_version", "nvmlSystemGetNVMLVersion")):
            try:
                result[key] = _text(getattr(self.nvml, api)())
            except Exception as exc:
                result[key] = None
                result["errors"][key] = _error(exc)
        cc = result["compute_capability"]
        result["power_usage_semantics"] = (
            "instantaneous_device_and_associated_circuitry" if cc and (cc[0] < 8 or cc == [8, 0])
            else "one_second_average_device_and_associated_circuitry" if cc
            else "unknown_inspect_driver_and_architecture")
        result["energy_counter_units"] = "mJ_since_driver_reload"
        result["power_scope_note"] = "Do not equate device/circuitry or GPU scope to an isolated core rail, or subtract a memory rail without validating scope inclusion on this SKU. No module-scope fallback is performed."
        result["sample"] = self.read_sample()
        result["capabilities"] = dict(self.capabilities)
        result["capabilities"].update({
            "power_scope": "nvmlDeviceGetPowerUsage: device and associated circuitry (e.g. memory); driver/SKU scope must be validated",
            "power_usage_semantics": result["power_usage_semantics"],
            "energy_counter_units": result["energy_counter_units"],
            "memory_power_source": result["sample"].get("memory_power_source"),
        })
        return result

    def close(self) -> None:
        # Do not block on a stuck native NVML query while trying to close a
        # still-active sampler. Native calls cannot be safely killed in Python.
        if self._active_samplers:
            raise TelemetryError("Stop all telemetry samplers before closing the device")
        if not self._lock.acquire(timeout=1):
            raise TelemetryError("NVML query is still running; device cannot be closed safely")
        try:
            if self._closed:
                return
            if self._active_samplers:
                raise TelemetryError("Stop all telemetry samplers before closing the device")
            self._closed = True
            with _lifecycle_lock:
                key = id(self.nvml)
                count = _sessions.get(key, 0)
                if count <= 1:
                    _sessions.pop(key, None)
                    self.nvml.nvmlShutdown()
                else:
                    _sessions[key] = count - 1
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Sampler:
    """Monotonic host sampling with explicit start/end bracket readings.

    A small polling interval does not increase the NVML sensor update rate.
    Query durations and sensor timestamps permit detection of stale readings
    and monitoring overhead. Samples can be written as JSON Lines unchanged.
    """

    def __init__(self, device: NvmlDevice, interval_s: float = 0.02):
        if not math.isfinite(interval_s) or interval_s < 0.001:
            raise ValueError("Telemetry interval must be finite and at least 0.001 seconds")
        self.device = device
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._samples: list[dict] = []
        self._samples_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._attached = False
        self._started = threading.Event()
        self.error: dict | None = None

    @property
    def samples(self) -> list[dict]:
        with self._samples_lock:
            return list(self._samples)

    def _append(self, sample: dict) -> None:
        if not isinstance(sample, dict) or not math.isfinite(float(sample.get("t_s", float("nan")))):
            raise TelemetryError("NVML sampler received a malformed or non-finite sample timestamp")
        with self._samples_lock:
            self._samples.append(sample)
        if sample.get("fatal_errors"):
            raise TelemetryError(f"Fatal NVML device error: {sample['fatal_errors']}")

    def start(self):
        if self._thread is not None:
            raise TelemetryError("Sampler has already been started")
        with self.device._lock:
            self.device._check_open()
            self.device._active_samplers += 1
            self._attached = True
        try:
            self._thread = threading.Thread(target=self._run, name="nvml-telemetry", daemon=True)
            self._thread.start()
        except Exception:
            with self.device._lock:
                self.device._active_samplers -= 1
                self._attached = False
            raise
        if not self._started.wait(timeout=10):
            self._stop.set()
            raise TelemetryError("Initial NVML query did not finish; stop further experiments")
        if self.error:
            raise TelemetryError(f"Initial NVML sample failed: {self.error}")
        return self

    def _run(self) -> None:
        try:
            self._append(self.device.read_sample())
            self._started.set()
            deadline = time.monotonic() + self.interval_s
            while not self._stop.wait(max(0.0, deadline - time.monotonic())):
                self._append(self.device.read_sample())
                now = time.monotonic()
                # Skip missed ticks instead of spinning if a query is slow.
                deadline += self.interval_s
                if deadline <= now:
                    deadline = now + self.interval_s
            # Final bracketing query stays on the worker so stop() can time out
            # even when the driver's final query hangs.
            self._append(self.device.read_sample())
        except BaseException as exc:
            self.error = _error(exc)
        finally:
            self._started.set()
            if self._attached:
                with self.device._lock:
                    self.device._active_samplers -= 1
                    self._attached = False

    def stop(self, timeout_s: float = 10) -> list[dict]:
        if self._thread is None:
            raise TelemetryError("Sampler has not been started")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Sampler shutdown timeout must be finite and positive")
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TelemetryError("NVML sampler did not stop; device must remain open until its query finishes")
        if self.error:
            raise TelemetryError(f"NVML sampler failed: {self.error}")
        return self.samples

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
