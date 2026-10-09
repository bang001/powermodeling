"""Export measured component figures with intervals, anchors and missing cells."""

from collections import defaultdict
from pathlib import Path
import math
import statistics
import textwrap

from .evaluation import OBJECTIVES
from .sfu import SFU_WORKLOADS

STYLES = {"pass": ("#177c54", "o"), "fail": ("#bf4343", "x"),
          "inconclusive": ("#b48425", "s"), "unprofiled": ("#3e77b5", "^")}


def write_plots(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    files, exported_input_curves = {}, set()
    components = (summary.get("evaluation") or {}).get("components", [])
    counts = {c["stratum"]["workload"]: sum(x["stratum"]["workload"] == c["stratum"]["workload"] for x in components) for c in components}
    for c in components:
        workload = c["stratum"]["workload"]
        suffix = "-" + c["component_id"] if counts[workload] > 1 else ""
        name = workload + suffix
        if workload == "hbm":
            domains = {p.get("requested_memory_mhz") for p in c["points"]}
            for memory in sorted(domains, key=lambda value: (value is None, value or 0)):
                subset = sorted((p for p in c["points"] if p.get("requested_memory_mhz") == memory),
                                key=lambda p: (p.get("requested_graphics_mhz") is None,
                                               p.get("requested_graphics_mhz") or 0, p["group_id"]))
                sensor_name = name + (f"-mem-{memory}" if len(domains) > 1 else "")
                for start in range(0, len(subset), 8):
                    page_name = sensor_name + (f"-page-{start // 8 + 1:02d}" if len(subset) > 8 else "")
                    _memory_sensor_figure(c, subset[start:start + 8], page_name, output, files, plt)
        if workload in ("control", "l2_latency"):
            _diagnostic(c, name, output, files, plt, np)
            continue
        for curve in (c.get("input_scaling") or {}).get("curves", []):
            if curve["curve_id"] in exported_input_curves:
                continue
            exported_input_curves.add(curve["curve_id"])
            for objective in OBJECTIVES:
                if any(p["energies"][objective] is not None for p in curve["rows"]):
                    _input_size_figure(c, curve, objective, summary["selection_policy"]["min_repeats"], output, files, plt)
        for objective in OBJECTIVES:
            points = [p for p in c["points"] if p["valid_repeats"] >= summary["selection_policy"]["min_repeats"]
                      and p["rate"] is not None and p["energies"][objective] is not None
                      and p["objective_measurement_valid"][objective]]
            if points:
                domains = sorted({p["requested_memory_mhz"] for p in points if p["requested_memory_mhz"] is not None})
                if len(domains) <= 1:
                    _energy_figure(c, points, objective, name, output, files, plt, np)
                else:
                    for memory in domains + ([None] if any(p["requested_memory_mhz"] is None for p in points) else []):
                        subset = [p for p in points if p["requested_memory_mhz"] == memory]
                        panel = {**c, "clocks": [k for k in c["clocks"] if k["memory_mhz"] == memory],
                                 "points": [p for p in c["points"] if p["requested_memory_mhz"] == memory]}
                        _energy_figure(panel, subset, objective, name + f"-mem-{memory}", output, files, plt, np)
    return files


def _save(fig, stem, output, files, plt):
    for extension in ("png", "svg"):
        path = output / (stem + "." + extension)
        fig.savefig(path, dpi=160)
        files[stem + "_" + extension] = str(path.resolve())
    plt.close(fig)


def _memory_sensor_figure(component, points, name, output, files, plt):
    """Export the measured memory scope independently of board objectives."""
    def value(point, field, validity):
        sensor = point.get("hbm_memory_power") or {}
        number = sensor.get(field)
        if (sensor.get("measurement_valid") and sensor.get(validity)
                and sensor.get("status") in ("available", "partial")
                and isinstance(number, (int, float)) and not isinstance(number, bool)
                and math.isfinite(number)):
            return number
        return None

    if not any(value(p, "power_w", "measurement_valid") is not None for p in points):
        return
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    panels = (
        ("power_w", "measurement_valid", "Memory-scope power (W)", "Memory sensor: measured power"),
        ("energy_j", "measurement_valid", "Memory-scope energy (J)", "Memory sensor: integrated energy"),
        ("pj_per_logical_bit", "normalization_valid", "pJ/logical bit", "Memory sensor: logical payload normalization"),
        ("incremental_power_w", "incremental_valid", "Memory-scope increment (W)", "Matched idle increment: power"),
        ("incremental_energy_j", "incremental_valid", "Memory-scope increment (J)", "Matched idle increment: energy"),
        ("incremental_pj_per_logical_bit", "incremental_valid", "Incremental pJ/logical bit", "Matched idle increment: logical payload"))
    labels, captions = [], []
    for point in points:
        sensor = point.get("hbm_memory_power") or {}
        geometry = point.get("geometry") or {}
        labels.append(f"{point['group_id'][:12]}\nSM={point.get('requested_graphics_mhz')} / mem={point.get('requested_memory_mhz')}\n"
                      f"B={geometry.get('blocks')} T={geometry.get('threads')}")
        captions.append(f"{point['group_id'][:12]}: {sensor.get('status', 'unavailable')}; "
                        f"{sensor.get('valid_repeats', 0)}/{sensor.get('observed_repeats', 0)} sensor repeats; "
                        f"{sensor.get('normalized_repeats', 0)} normalized; {sensor.get('incremental_repeats', 0)} matched idle; "
                        f"freshness {sensor.get('freshness_status', 'unverified')}")
    for ax, (field, validity, ylabel, title) in zip(axes.flat, panels):
        observed = [(index, p, value(p, field, validity)) for index, p in enumerate(points)]
        observed = [(index, p, number) for index, p, number in observed if number is not None
                    and (field != "incremental_pj_per_logical_bit" or (p.get("hbm_memory_power") or {}).get("normalization_valid"))]
        if observed:
            ax.scatter([index for index, _, _ in observed], [number for _, _, number in observed], color="#3e77b5")
            for index, point, _ in observed:
                interval = (point["hbm_memory_power"].get("ci95") or {}).get(field)
                if (isinstance(interval, (list, tuple)) and len(interval) == 2
                        and all(isinstance(n, (int, float)) and math.isfinite(n) for n in interval)):
                    ax.plot([index, index], interval, color="#3e77b5", alpha=.5, linewidth=1)
        else:
            ax.text(.5, .5, "N/A: no qualified sensor values", ha="center", transform=ax.transAxes)
        ax.set_xticks(range(len(points)), labels, rotation=35, ha="right", fontsize=7)
        ax.set(xlabel="Measured condition / requested clocks (MHz) / launch geometry", ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
    sources = sorted({str((p.get("hbm_memory_power") or {}).get("source")) for p in points
                      if (p.get("hbm_memory_power") or {}).get("source")})
    semantics = sorted({str((p.get("hbm_memory_power") or {}).get("semantics")) for p in points
                        if (p.get("hbm_memory_power") or {}).get("semantics")})
    fig.suptitle(f"{name}: separate HBM memory sensor — {component.get('gpu_name')} / {component['stratum']['gpu_uuid']}\n"
                 f"Source: {'; '.join(sources)}. {'; '.join(semantics)}\n"
                 "Repeat medians / bootstrap 95% intervals. Missing or invalid measurements are blank. "
                 "Board total remains a separate objective; sensor values are not added to board energy.\n"
                 + "\n".join(captions), fontsize=9)
    _save(fig, name + "-memory-sensor", output, files, plt)


def _scatter(ax, points, x, y, ci=None):
    for status, (color, marker) in STYLES.items():
        subset = [p for p in points if p["ncu_status"] == status and x(p) is not None and y(p) is not None]
        if not subset: continue
        ax.scatter([x(p) for p in subset], [y(p) for p in subset], color=color, marker=marker, label=status, alpha=.8)
        for p in subset:
            interval = ci(p) if ci else None
            if interval:
                ax.plot([x(p), x(p)], interval, color=color, alpha=.5, linewidth=1)
        for tag, edge in (("advertised_default_fixed_pair", "#8424ae"), ("required_exact_1110_mhz", "#17263c")):
            selected = [p for p in subset if tag in p["anchor_tags"]]
            if selected:
                ax.scatter([x(p) for p in selected], [y(p) for p in selected], s=100, facecolors="none", edgecolors=edge, linewidths=1.4)
    ax.grid(alpha=.2)


def _input_size_figure(component, curve, objective, minimum_repeats, output, files, plt):
    """Export one matched Q curve, without pooling energies across input sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    units = component["units"]
    sfu = component["stratum"]["workload"] in SFU_WORKLOADS
    points = [{**p, "anchor_tags": []} for p in curve["rows"]]
    x = lambda p: p["input_elements"] / 1e6
    rate = lambda p: p["rate"] / units["rate_scale"] if p["rate"] is not None else None
    energy = lambda p: p["energies"][objective] if p["objective_measurement_valid"][objective] else None
    rci = lambda p: [v / units["rate_scale"] for v in p["rate_ci95"]] if p["rate_ci95"] else None
    eci = lambda p: p["energy_ci95"][objective]
    _scatter(axes[0], points, x, rate, rci)
    _scatter(axes[1], points, x, energy, eci)
    if sfu and objective == "paired_active_reference":
        axes[1].axhline(0, color="#68758c", linewidth=.8, label="zero contrast")
    observed = [p["rate"] for p in points if p["valid_repeats"] >= minimum_repeats and p["rate"] is not None and p["rate"] > 0]
    if observed:
        axes[0].axhline(max(observed) / units["rate_scale"], linestyle=":", linewidth=1,
                       color="#68758c", label="all-valid curve peak")
    for ax, y, ylabel, title in ((axes[0], rate, units["rate_unit"], "Observed input-size throughput"),
                                (axes[1], energy, units["energy_unit"], "Energy remains separate at each Q")):
        rejected = [p for p in points if not p["eligible"][objective] and y(p) is not None]
        if rejected:
            ax.scatter([x(p) for p in rejected], [y(p) for p in rejected], s=110,
                       facecolors="none", edgecolors="#17263c", label="evidence ineligible")
        if len(points) <= 12:
            q_min, q_max = min(p["input_elements"] for p in points), max(p["input_elements"] for p in points)
            for point in points:
                if y(point) is not None:
                    ax.annotate(point["group_id"][:8] + f" / B={point['blocks']}", (x(point), y(point)),
                                xytext=(0, 8), textcoords="offset points", fontsize=7,
                                ha="left" if point["input_elements"] == q_min else "right" if point["input_elements"] == q_max else "center")
        ax.set(xlabel="Q (million active SFU register lanes)" if sfu else "Input Q (million elements)", ylabel=ylabel, title=title)
        if ax.collections:
            ax.legend(fontsize=7)
    first = points[0]
    signature = curve["signature"]
    contract = signature.get("sfu_contract" if sfu else "nonlinear_contract") or {}
    width = contract.get("row_width")
    iterations = signature["config"].get("iterations")
    details = f"chains={contract.get('sfu_chains')}; primitive={contract.get('sfu_primitive')}" if sfu else f"row_width={width}" + (" (pointwise)" if width == 0 else "")
    objective_label = ("signed SFU/control contrast" if objective == "paired_active_reference" else "board diagnostic: " + objective.replace("_", " ")) if sfu else objective.replace("_", " ")
    fig.suptitle(f"{component['stratum']['workload']} Q curve {curve['curve_id']}: {objective_label}\n"
                 f"{component.get('gpu_name')} / {component['stratum']['gpu_uuid']}; "
                 f"SM/memory={first['requested_graphics_mhz']}/{first['requested_memory_mhz']} MHz\n"
                 f"threads={first['threads']}; iterations={iterations}; {details}; grid=auto\n"
                 "Repeat medians / bootstrap 95% intervals; matched definition and iterations. Size stability does not prove hardware saturation.", fontsize=9)
    _save(fig, component["stratum"]["workload"] + "-q-curve-" + curve["curve_id"] + "-" + objective + "-input-scaling", output, files, plt)


def _energy_figure(c, points, objective, name, output, files, plt, np):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    u = c["units"]; rate = lambda p: p["rate"] / u["rate_scale"]
    energy = lambda p: p["energies"][objective]
    clock = lambda p: p["requested_graphics_mhz"]
    rci = lambda p: [v / u["rate_scale"] for v in p["rate_ci95"]] if p["rate_ci95"] else None
    eci = lambda p: p["energy_ci95"][objective]
    _scatter(axes[0, 0], points, clock, rate, rci)
    _scatter(axes[0, 1], points, clock, energy, eci)
    _scatter(axes[0, 2], points, rate, energy, eci)
    sfu = c["stratum"]["workload"] in SFU_WORKLOADS
    if sfu and objective == "paired_active_reference":
        axes[0, 1].axhline(0, color="#68758c", linewidth=.8)
        axes[0, 2].axhline(0, color="#68758c", linewidth=.8)
    clock_pairs = sorted({(p["requested_graphics_mhz"], p["requested_memory_mhz"]) for p in points if p["requested_graphics_mhz"] is not None})
    for index, pair in enumerate(clock_pairs):
        color = plt.cm.viridis(index / max(1, len(clock_pairs) - 1))
        selected = [p for p in points if (p["requested_graphics_mhz"], p["requested_memory_mhz"]) == pair and p["resource_capacity"] is not None]
        axes[1, 0].scatter([p["resource_capacity"] for p in selected], [rate(p) for p in selected], color=color,
                           label=f"{pair[0]} MHz", s=22)
        for p in selected:
            ci = rci(p)
            if ci: axes[1, 0].plot([p["resource_capacity"]] * 2, ci, color=color, alpha=.5, linewidth=1)
    axes[1, 0].grid(alpha=.2)
    rec = next(r for r in c["recommendations"] if r["objective"] == objective)
    winner = next((p for p in points if p["group_id"] == rec.get("group_id")), None)
    if winner:
        axes[0, 2].scatter([rate(winner)], [energy(winner)], marker="*", s=220, color="#1c2840", label="candidate")
    hbm = c["stratum"]["workload"] == "hbm"
    if c["observed_peak"] is not None:
        axes[0, 0].axhline(c["observed_peak"] / u["rate_scale"], color="#68758c", linestyle=":", linewidth=1,
                          label="observed peak (diagnostic only)" if hbm else "energy-population observed peak")
    if hbm:
        thresholds = [(p, p.get("hbm_bandwidth") or {}) for p in points]
        thresholds = [(p, b) for p, b in thresholds if b.get("theoretical_bytes_s") is not None
                      and b.get("minimum_fraction") is not None and clock(p) is not None]
        if thresholds:
            axes[0, 0].scatter([clock(p) for p, b in thresholds],
                              [b["minimum_fraction"] * b["theoretical_bytes_s"] / u["rate_scale"] for p, b in thresholds],
                              marker="_", color="#98551e", s=120, label="required fraction of theoretical HBM bandwidth")
    for ax, xlabel, ylabel, title in (
        (axes[0, 0], "Requested SM / graphics MHz", u["rate_unit"], "Sustained throughput; every geometry"),
        (axes[0, 1], "Requested SM / graphics MHz", u["energy_unit"], "Energy and repeat uncertainty"),
        (axes[0, 2], u["rate_unit"], u["energy_unit"], "Measured energy / throughput tradeoff"),
        (axes[1, 0], "GEMM m × n × k" if c["stratum"]["workload"] == "gemm" else
         "Launched grid threads; not simultaneous residency" if c["stratum"]["experiment_contract"].get("grid_mode") == "auto" else
         "blocks × threads (× accumulators for Tensor)", u["rate_unit"], "Per-clock resource / size response; colour = SM MHz")):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if ax.collections: ax.legend(fontsize=7, ncol=2 if ax is axes[1, 0] else 1)
    graphics = sorted({k["graphics_mhz"] for k in c["clocks"]})
    memories = sorted({k["memory_mhz"] for k in c["clocks"]})
    if graphics and memories:
        grid = np.full((len(memories), len(graphics)), np.nan)
        lookup = {p["group_id"]: p for p in c["points"]}
        for row in c["clocks"]:
            best = lookup.get(row["minimum_energy_group_ids"].get(objective))
            if best is not None:
                grid[memories.index(row["memory_mhz"]), graphics.index(row["graphics_mhz"])] = best["energies"][objective]
        if np.isfinite(grid).any():
            image = axes[1, 1].imshow(np.ma.masked_invalid(grid), aspect="auto", cmap="viridis_r")
            fig.colorbar(image, ax=axes[1, 1], label=u["energy_unit"])
        else: axes[1, 1].text(.5, .5, "No qualified energy cells", ha="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_xticks(range(len(graphics)), graphics, rotation=60, fontsize=7)
        axes[1, 1].set_yticks(range(len(memories)), memories, fontsize=8)
    axes[1, 1].set(title="HBM bandwidth-qualified clock minima; missing evidence is blank" if hbm else "Own-clock energy minima; missing evidence is blank",
                   xlabel="Requested SM MHz (discrete measured cells)", ylabel="Requested memory MHz")
    statuses = list(STYLES)
    valid = [sum(p["valid_repeats"] for p in c["points"] if p["ncu_status"] == s) for s in statuses]
    rejected = [sum(p["rejected_repeats"] for p in c["points"] if p["ncu_status"] == s) for s in statuses]
    axes[1, 2].bar(statuses, valid, label="valid", color="#6886a7")
    axes[1, 2].bar(statuses, rejected, bottom=valid, label="rejected", color="#c77373")
    axes[1, 2].set(title="Trial quality by NCU status", ylabel="Trial count")
    axes[1, 2].legend(fontsize=8)
    contract = textwrap.fill(str(c["stratum"]["experiment_contract"]), width=150)
    objective_label = ("signed SFU/control contrast; unqualified values are diagnostic" if objective == "paired_active_reference" else "board diagnostic: " + objective.replace("_", " ")) if sfu else objective.replace("_", " ")
    role = c["stratum"].get("experiment_role", "legacy_unspecified")
    admission_note = " NCU status is target-path evidence; coalescing eligibility is separate. Diagnostic observations do not establish energy winners." if any(p.get("memory_coalescing") for p in points) else ""
    if hbm:
        fractions = sorted({b["minimum_fraction"] for p in points if (b := p.get("hbm_bandwidth") or {}).get("minimum_fraction") is not None})
        threshold_label = "/".join(f"{fraction:.0%}" for fraction in fractions) or "configured fraction"
        admission_note += (f"\nHBM admission: logical sustained bandwidth ≥ {threshold_label} of actual-clock / bus-width theory."
                           "\nObserved peak and plateau are diagnostic; no extra 95% gate; physical DRAM utilization is not measured.")
    fig.suptitle(f"{name}: {objective_label} — {c['stratum']['gpu_uuid']}\n{contract}\n"
                 f"Role: {role}. Repeat medians / bootstrap 95% intervals. Factory default: purple ring; exact 1110: black ring. {rec['status']}.\n{admission_note}", fontsize=10)
    _save(fig, name + "-" + objective + "-energy-throughput", output, files, plt)
    files[name + "_" + objective + "_plot"] = files[name + "-" + objective + "-energy-throughput_png"]


def _diagnostic(c, name, output, files, plt, np):
    if c["stratum"]["workload"] == "l2_latency" and c["latency_points"]:
        partitions = defaultdict(list)
        for p in c["latency_points"]:
            partitions[(p["graphics_mhz"], p["memory_mhz"])].append(p)
        for clocks, points in sorted(partitions.items(), key=lambda row: str(row[0])):
            sms = sorted({p["sm_id"] for p in points}, key=int)
            offsets = sorted({p["offset_bytes"] for p in points})
            grid = np.full((len(sms), len(offsets)), np.nan)
            for i, sm in enumerate(sms):
                for j, offset in enumerate(offsets):
                    samples = [p["cycles_per_access"] for p in points if p["sm_id"] == sm and p["offset_bytes"] == offset]
                    if len(samples) >= 3: grid[i, j] = statistics.median(samples)
            fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
            if np.isfinite(grid).any():
                image = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap="viridis")
                fig.colorbar(image, ax=ax, label="Dependent cycles/access")
            ax.set_xticks(range(len(offsets)), offsets, rotation=45)
            ax.set_yticks(range(len(sms)), sms)
            ax.set(xlabel="Observed address offset (B)", ylabel="Observed SM ID", title=f"{name}: {clocks} MHz\nRepeat medians; ≥3 probes/cell; missing cells blank; no near/far label")
            _save(fig, name + f"-{clocks[0]}-{clocks[1]}-latency", output, files, plt)
    trace = c.get("measurement_example")
    if trace and trace["points"]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True)
        phases = defaultdict(list)
        for p in trace["points"]:
            if p.get("t_s") is not None: phases[p["phase"]].append(p)
        start = min((p["t_s"] for rows in phases.values() for p in rows), default=0)
        for phase, rows in phases.items():
            for ax, field in ((axes[0], "power_w"), (axes[1], "temperature_c")):
                values = [p for p in rows if p.get(field) is not None]
                ax.scatter([p["t_s"] - start for p in values], [p[field] for p in values], s=8, label=phase)
        axes[0].set(ylabel="Observed power (W)", title=f"{name}: one valid repeat {trace['trial_id']}")
        axes[1].set(xlabel="Elapsed monotonic time (s)", ylabel="Temperature (°C)")
        for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
        _save(fig, name + "-measurement-trace", output, files, plt)
