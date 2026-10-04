"""Architecture facts; actual device discovery always overrides capacity/SM counts.

Dense FP16 Tensor operations count each multiply-add as two FLOPs, FP32 accumulation.
These constants describe theoretical issue capacity, not guaranteed achieved throughput.
"""

ARCHITECTURES = {
    (7, 0): {"architecture": "Volta", "family": "V100", "dense_tensor_flops_per_sm_clock": 1024},
    (8, 0): {"architecture": "Ampere GA100", "family": "A100", "dense_tensor_flops_per_sm_clock": 2048},
    (9, 0): {"architecture": "Hopper", "family": "H100", "dense_tensor_flops_per_sm_clock": 4096},
}


def architecture(device):
    cc = device.get("compute_capability")
    if isinstance(cc, str):
        cc = tuple(int(x) for x in cc.split("."))
    elif isinstance(cc, (list, tuple)):
        cc = tuple(cc)
    else:
        cc = (device.get("cc_major", device.get("compute_capability_major")),
              device.get("cc_minor", device.get("compute_capability_minor")))
    if cc not in ARCHITECTURES:
        raise ValueError(f"Unsupported compute capability {cc}; expected V100 7.0/A100 8.0/H100 9.0")
    return dict(ARCHITECTURES[cc])


def theoretical_tensor_tflops(device, achieved_graphics_mhz):
    """Clock-specific dense ceiling, never power-limit divided by operations."""
    info = architecture(device)
    return (info["dense_tensor_flops_per_sm_clock"] * device["sm_count"]
            * achieved_graphics_mhz * 1e6 / 1e12)
