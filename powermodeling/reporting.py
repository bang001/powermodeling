"""Export measured component figures with intervals, anchors and missing cells."""

from collections import defaultdict
from pathlib import Path
import statistics

from .evaluation import OBJECTIVES

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
                      and p["rate"] is not None and p["energies"][objective] is not None and p["objective_eligible"][objective]]
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
    points = [{**p, "anchor_tags": []} for p in curve["rows"]]
    x = lambda p: p["input_elements"] / 1e6
    rate = lambda p: p["rate"] / units["rate_scale"] if p["rate"] is not None else None
    energy = lambda p: p["energies"][objective]
    rci = lambda p: [v / units["rate_scale"] for v in p["rate_ci95"]] if p["rate_ci95"] else None
    eci = lambda p: p["energy_ci95"][objective]
    _scatter(axes[0], points, x, rate, rci)
    _scatter(axes[1], points, x, energy, eci)
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
        ax.set(xlabel="Input Q (million elements)", ylabel=ylabel, title=title)
        if ax.collections:
            ax.legend(fontsize=7)
    first = points[0]
    signature = curve["signature"]
    width = signature["nonlinear_contract"].get("row_width")
    iterations = signature["config"].get("iterations")
    fig.suptitle(f"{component['stratum']['workload']} Q curve {curve['curve_id']}: {objective.replace('_', ' ')}\n"
                 f"{component.get('gpu_name')} / {component['stratum']['gpu_uuid']}; "
                 f"SM/memory={first['requested_graphics_mhz']}/{first['requested_memory_mhz']} MHz\n"
                 f"threads={first['threads']}; iterations={iterations}; row_width={width}" + (" (pointwise)" if width == 0 else "") + "; grid=auto\n"
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
    if c["observed_peak"] is not None:
        axes[0, 0].axhline(c["observed_peak"] / u["rate_scale"], color="#68758c", linestyle=":", linewidth=1, label="all-valid observed peak")
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
    axes[1, 1].set(title="Own-clock energy minima; missing evidence is blank", xlabel="Requested SM MHz (discrete measured cells)", ylabel="Requested memory MHz")
    statuses = list(STYLES)
    valid = [sum(p["valid_repeats"] for p in c["points"] if p["ncu_status"] == s) for s in statuses]
    rejected = [sum(p["rejected_repeats"] for p in c["points"] if p["ncu_status"] == s) for s in statuses]
    axes[1, 2].bar(statuses, valid, label="valid", color="#6886a7")
    axes[1, 2].bar(statuses, rejected, bottom=valid, label="rejected", color="#c77373")
    axes[1, 2].set(title="Trial quality by NCU status", ylabel="Trial count")
    axes[1, 2].legend(fontsize=8)
    contract = str(c["stratum"]["experiment_contract"])
    fig.suptitle(f"{name}: {objective.replace('_', ' ')} — {c['stratum']['gpu_uuid']}\n{contract[:180]}\n"
                 f"Repeat medians / bootstrap 95% intervals. Factory default: purple ring; exact 1110: black ring. {rec['status']}.", fontsize=10)
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
