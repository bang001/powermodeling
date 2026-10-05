"""Architecture facts; actual device discovery always overrides capacity/SM counts.

Dense FP16 Tensor operations count each multiply-add as two FLOPs, FP32 accumulation.
These constants describe theoretical issue capacity, not guaranteed achieved throughput.
"""

import re

ARCHITECTURES = {
    (7, 0): {"architecture": "Volta", "family": "V100", "dense_tensor_flops_per_sm_clock": 1024},
    (8, 0): {"architecture": "Ampere GA100", "family": "A100", "dense_tensor_flops_per_sm_clock": 2048},
    (9, 0): {"architecture": "Hopper", "family": "H100", "dense_tensor_flops_per_sm_clock": 4096},
}


def _named_family(device):
    """Check identifiable product names without inventing a SKU from capacity."""
    name = str(device.get("name", device.get("gpu_name", ""))).upper()
    # SXM/HBM, capacity suffixes and chip-code names are intentionally absent.
    # Clearly identifiable other GPU products cannot share an H100/A100 label
    # merely because their compute capability happens to be the same.
    products = re.findall(r"\b[APHVTBL][0-9]{1,4}[A-Z]?\b", name)
    supported = {"V100", "A100", "H100"}
    conflicts = sorted({product for product in products if product not in supported})
    if conflicts or any(word in name for word in ("GEFORCE", "QUADRO", "TITAN", "INSTINCT")) or re.search(r"\bRTX", name):
        raise ValueError(f"GPU product name {name!r} is outside V100/A100/H100 experiment scope")
    named = sorted(set(products) & supported)
    if len(named) > 1:
        raise ValueError(f"GPU product name {name!r} contains conflicting GPU families")
    return named[0] if named else None


def declare_sxm(device, target_form_factor="SXM"):
    """Record the user's module type without inferring it from HBM technology.

    Some driver names (notably ``H100 80GB HBM3``) do not expose module type.
    A user declaration is retained in that case, with its lack of independent
    confirmation explicit. A conflicting product name or discovery field fails
    planning rather than quietly benchmarking a PCIe/NVL device as SXM.
    Capacity, enabled SMs, HBM generation and power limits remain discovered.
    """
    if not isinstance(target_form_factor, str) or target_form_factor.upper() != "SXM":
        raise ValueError("This experiment targets SXM modules; target_form_factor must be SXM")
    result = dict(device)
    named_family = _named_family(device)
    name = str(device.get("name", device.get("gpu_name", "")))
    normalized_name = name.upper().replace("-", " ")
    detected = device.get("detected_form_factor", device.get("form_factor"))
    if detected is not None and not str(detected).upper().startswith("SXM"):
        raise ValueError(f"Discovered form factor {detected!r} conflicts with SXM experiment")
    if "PCIE" in normalized_name.replace(" ", "") or "NVL" in normalized_name.split():
        raise ValueError(f"GPU product name {name!r} conflicts with SXM experiment")
    confirmed = detected is not None or "SXM" in normalized_name
    result["target_form_factor"] = "SXM"
    result["form_factor_validation"] = {
        "declared": "SXM", "declaration_source": "experiment_target",
        "status": "confirmed_by_discovery" if confirmed else "declared_not_independently_verified",
        "discovered_form_factor": detected, "discovered_product_name": name or None,
        "note": "HBM is memory technology, not a GPU module form factor; capacity and power are not inferred from SXM.",
    }
    result["gpu_family_validation"] = {
        "named_family": named_family,
        "status": "supported_family_name" if named_family else "name_unverified",
        "note": "Generic product names do not independently establish V100/A100/H100; compute-capability inference is recorded separately.",
    }
    if any(key in device for key in ("compute_capability", "cc_major", "compute_capability_major")):
        info = architecture(device)
        result["gpu_family_validation"] = info["gpu_family_validation"]
    return result


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
    result = dict(ARCHITECTURES[cc])
    named_family = _named_family(device)
    if named_family is not None and named_family != result["family"]:
        raise ValueError(f"GPU named {named_family} conflicts with discovered compute capability {cc}")
    result["gpu_family_validation"] = {
        "named_family": named_family, "compute_capability_inferred_family": result["family"],
        "status": "name_matches_compute_capability" if named_family else "compute_capability_inferred_name_unverified",
    }
    return result


def theoretical_tensor_tflops(device, achieved_graphics_mhz):
    """Clock-specific dense ceiling, never power-limit divided by operations."""
    info = architecture(device)
    return (info["dense_tensor_flops_per_sm_clock"] * device["sm_count"]
            * achieved_graphics_mhz * 1e6 / 1e12)
