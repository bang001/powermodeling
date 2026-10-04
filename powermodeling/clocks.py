"""Read-only clock discovery and explicitly owned, reversible clock experiments.

Application clocks can be read and restored exactly (where that deprecated API
is still supported). NVML has no portable getter for preexisting locked ranges.
The locked method therefore requires an explicit restoration policy, never
silently resets another user's locks. No power limit or persistence mutation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .telemetry import NvmlDevice, _error


class ClockError(RuntimeError):
    pass


class ClockRestoreError(ClockError):
    """Restoration failed; measurements and subsequent sweeps must be stopped."""


def _mhz(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClockError(f"{label} must be a positive integer in MHz")
    return value


def _clock_types(device: NvmlDevice) -> tuple[int, int]:
    return (getattr(device.nvml, "NVML_CLOCK_GRAPHICS", 0),
            getattr(device.nvml, "NVML_CLOCK_MEM", 2))


def discover(device: NvmlDevice) -> dict[str, Any]:
    """Enumerate advertised graphics/memory pairs without changing GPU state."""
    result: dict[str, Any] = {
        "uuid": device.uuid, "supported_pairs": [], "errors": {},
        "locked_clock_prior_policy": "not_portably_queryable",
        "applications_api_note": "deprecated; scheduled for removal in CUDA 14",
    }
    graphics_type, memory_type = _clock_types(device)
    queries = [
        ("current_graphics_mhz", "nvmlDeviceGetClockInfo", graphics_type),
        ("current_memory_mhz", "nvmlDeviceGetClockInfo", memory_type),
        ("applications_graphics_mhz", "nvmlDeviceGetApplicationsClock", graphics_type),
        ("applications_memory_mhz", "nvmlDeviceGetApplicationsClock", memory_type),
        ("default_applications_graphics_mhz", "nvmlDeviceGetDefaultApplicationsClock", graphics_type),
        ("default_applications_memory_mhz", "nvmlDeviceGetDefaultApplicationsClock", memory_type),
    ]
    for key, function, clock_type in queries:
        try:
            result[key] = int(device.call(function, clock_type))
        except Exception as exc:
            result[key] = None
            result["errors"][key] = _error(exc)
    try:
        memories = sorted({int(v) for v in device.call("nvmlDeviceGetSupportedMemoryClocks")})
        result["supported_memory_mhz"] = memories
    except Exception as exc:
        memories = []
        result["supported_memory_mhz"] = None
        result["errors"]["supported_memory_mhz"] = _error(exc)
    for memory in memories:
        try:
            graphics = sorted({int(v) for v in device.call("nvmlDeviceGetSupportedGraphicsClocks", memory)})
            result["supported_pairs"].extend({"graphics_mhz": g, "memory_mhz": memory} for g in graphics)
        except Exception as exc:
            result["errors"][f"graphics_for_memory_{memory}"] = _error(exc)
    result["applications_restorable"] = (
        result["applications_graphics_mhz"] is not None
        and result["applications_memory_mhz"] is not None
        and hasattr(device.nvml, "nvmlDeviceSetApplicationsClocks")
    )
    result["locked_api_symbols_present"] = {
        "graphics": hasattr(device.nvml, "nvmlDeviceSetGpuLockedClocks"),
        "memory": hasattr(device.nvml, "nvmlDeviceSetMemoryLockedClocks"),
    }
    return result


def _validate_pair(discovery: dict, graphics: int, memory: int) -> None:
    pairs = discovery["supported_pairs"]
    if not pairs:
        raise ClockError("Supported clock pairs could not be discovered; refusing an unvalidated clock request")
    if {"graphics_mhz": graphics, "memory_mhz": memory} not in pairs:
        raise ClockError(f"Clock pair graphics={graphics}, memory={memory} MHz is not advertised by this GPU")


@contextmanager
def clock_context(
    device: NvmlDevice,
    graphics_mhz: int | None = None,
    memory_mhz: int | None = None,
    method: str = "applications",
    allow_mutation: bool = False,
    locked_restore: dict[str, list[int] | tuple[int, int] | None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Apply requested clocks and restore only the policy modified by this run.

    With no requested clocks this is read-only and labels the domain as
    uncontrolled. ``allow_mutation`` must be explicitly enabled for requests.

    For ``locked`` pass previous policies for every changed domain, e.g.
    ``{'graphics': [900, 1200], 'memory': None}``. A range restores exactly;
    None is the caller's assertion that that domain was previously unlocked,
    and authorizes a reset on exit. Missing/unknown policies fail before writes.
    Restoration is attempted after partial setup failure and exceptions too.
    Actual measured clocks and throttle reasons remain necessary validation.
    """
    if method not in ("applications", "locked"):
        raise ClockError("Clock method must be applications or locked")
    if graphics_mhz is not None:
        graphics_mhz = _mhz(graphics_mhz, "graphics_mhz")
    if memory_mhz is not None:
        memory_mhz = _mhz(memory_mhz, "memory_mhz")
    discovery = discover(device)
    record = {
        "method": method if graphics_mhz is not None or memory_mhz is not None else "uncontrolled",
        "requested_graphics_mhz": graphics_mhz, "requested_memory_mhz": memory_mhz,
        "before": discovery, "applied": False, "restored": None,
    }
    if graphics_mhz is None and memory_mhz is None:
        yield record
        return
    if not allow_mutation:
        raise ClockError("Clock mutation requires explicit allow_mutation=True")
    changed: list[str] = []
    previous: dict[str, Any] = {}
    if method == "applications":
        if not discovery["applications_restorable"]:
            raise ClockError("Application clocks are not readable/restorable on this device; choose locked with an explicit prior policy")
        previous = {"graphics": discovery["applications_graphics_mhz"], "memory": discovery["applications_memory_mhz"]}
        graphics_mhz = previous["graphics"] if graphics_mhz is None else graphics_mhz
        memory_mhz = previous["memory"] if memory_mhz is None else memory_mhz
        _validate_pair(discovery, graphics_mhz, memory_mhz)
    else:
        requested = {"graphics": graphics_mhz, "memory": memory_mhz}
        for domain, value in requested.items():
            if value is None:
                continue
            if locked_restore is None or domain not in locked_restore:
                raise ClockError(f"Previous {domain} locked-clock policy is unknown; supply an exact prior range or explicitly assert unlocked")
            prior = locked_restore[domain]
            if prior is not None:
                if not isinstance(prior, (list, tuple)) or len(prior) != 2:
                    raise ClockError(f"Previous {domain} policy must be [min,max] MHz or None for known unlocked")
                prior = [_mhz(prior[0], "prior minimum"), _mhz(prior[1], "prior maximum")]
                if prior[0] > prior[1]:
                    raise ClockError("Prior locked-clock minimum must not exceed maximum")
            previous[domain] = prior
            set_api = "nvmlDeviceSetGpuLockedClocks" if domain == "graphics" else "nvmlDeviceSetMemoryLockedClocks"
            restore_api = ("nvmlDeviceResetGpuLockedClocks" if domain == "graphics" else "nvmlDeviceResetMemoryLockedClocks") if prior is None else set_api
            if not hasattr(device.nvml, set_api) or not hasattr(device.nvml, restore_api):
                raise ClockError(f"Required {domain} clock set/restore API is absent")
        # Validate each requested domain against the advertised supported clocks.
        # A locked GPU range is independent of the current idle memory clock.
        if memory_mhz is not None and memory_mhz not in (discovery["supported_memory_mhz"] or []):
            raise ClockError(f"Memory clock {memory_mhz} MHz is not advertised")
        if graphics_mhz is not None:
            valid_graphics = {pair["graphics_mhz"] for pair in discovery["supported_pairs"]
                              if memory_mhz is None or pair["memory_mhz"] == memory_mhz}
            if graphics_mhz not in valid_graphics:
                raise ClockError(f"Graphics clock {graphics_mhz} MHz is not advertised for the requested memory domain")
    record["effective_requested_graphics_mhz"] = graphics_mhz
    record["effective_requested_memory_mhz"] = memory_mhz
    record["previous_policy"] = previous
    try:
        if method == "applications":
            if graphics_mhz != previous["graphics"] or memory_mhz != previous["memory"]:
                device.call("nvmlDeviceSetApplicationsClocks", memory_mhz, graphics_mhz)
                changed.append("applications")
            # Read back policy; hardware under load may still throttle.
            graphics_type, memory_type = _clock_types(device)
            got_g = int(device.call("nvmlDeviceGetApplicationsClock", graphics_type))
            got_m = int(device.call("nvmlDeviceGetApplicationsClock", memory_type))
            if (got_g, got_m) != (graphics_mhz, memory_mhz):
                raise ClockError(f"Application clock readback differs: got {got_g}/{got_m} MHz")
        else:
            if memory_mhz is not None:
                device.call("nvmlDeviceSetMemoryLockedClocks", memory_mhz, memory_mhz)
                changed.append("memory")
            if graphics_mhz is not None:
                device.call("nvmlDeviceSetGpuLockedClocks", graphics_mhz, graphics_mhz)
                changed.append("graphics")
        record["applied"] = True
        yield record
    finally:
        failures = {}
        for domain in reversed(changed):
            try:
                if domain == "applications":
                    device.call("nvmlDeviceSetApplicationsClocks", previous["memory"], previous["graphics"])
                    graphics_type, memory_type = _clock_types(device)
                    if (int(device.call("nvmlDeviceGetApplicationsClock", graphics_type)),
                        int(device.call("nvmlDeviceGetApplicationsClock", memory_type))) != (previous["graphics"], previous["memory"]):
                        raise ClockRestoreError("Application-clock restoration readback differs from the prior policy")
                else:
                    prior = previous[domain]
                    if prior is None:
                        api = "nvmlDeviceResetGpuLockedClocks" if domain == "graphics" else "nvmlDeviceResetMemoryLockedClocks"
                        device.call(api)
                    else:
                        api = "nvmlDeviceSetGpuLockedClocks" if domain == "graphics" else "nvmlDeviceSetMemoryLockedClocks"
                        device.call(api, *prior)
            except Exception as exc:
                failures[domain] = _error(exc)
        record["restored"] = not bool(failures)
        record["restore_errors"] = failures
        if failures:
            raise ClockRestoreError(f"Clock policy restoration failed; stop further runs: {failures}")
