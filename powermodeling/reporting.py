"""Export charts only from measured/analyzed groups; never generate fake GPU curves."""

from pathlib import Path


def write_plots(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    files={}
    workloads=sorted({g["workload"] for g in summary.get("groups",[])}-{"control","l2_latency"})
    objectives = (
        ("total", "Whole-device total energy", None),
        ("operational_idle_increment", "Matched idle operational increment", "operational_idle_increment_eligible"),
        ("paired_active_reference", "Paired active-reference operational contrast", "paired_active_reference_eligible"),
    )
    for workload in workloads:
        for objective, title, eligibility in objectives:
            _write_objective_plot(summary, output, files, workload, objective, title, eligibility, plt)
    return files


def _write_objective_plot(summary, output, files, workload, objective, title, eligibility, plt):
    groups=[g for g in summary["groups"] if g["workload"]==workload and g["valid_repeats"]>=3]
    tensor=workload in ("tensor","gemm")
    rate="throughput_ops_s" if tensor else "throughput_bytes_s"
    energy=objective + ("_pj_per_flop" if tensor else "_pj_per_logical_bit")
    groups=[g for g in groups if g.get(rate) is not None and g.get(energy) is not None
            and (eligibility is None or g.get(eligibility))]
    if not groups: return
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    styles={"pass":("#15966c","o"),"fail":("#c84444","x"),
            "inconclusive":("#818894","s"),"unprofiled":("#447dcc","^")}
    for uuid in sorted({g["gpu_uuid"] for g in groups},key=lambda value:value or ""):
        for status,(color,marker) in styles.items():
            selected=[g for g in groups if g["gpu_uuid"]==uuid and g.get("ncu_status","unprofiled")==status]
            if not selected: continue
            scale=1e12 if tensor else 1e9
            label=f"{(uuid or 'unknown')[:14]}: {status}"
            axes[0].scatter([g[rate]/scale for g in selected],[g[energy] for g in selected],
                            label=label,color=color,marker=marker,alpha=.8)
            clocked=[g for g in selected if g.get("graphics_clock_mhz") is not None]
            if clocked:
                axes[1].scatter([g["graphics_clock_mhz"] for g in clocked],
                                [g[rate]/scale for g in clocked],label=label,color=color,marker=marker,alpha=.8)
    axes[0].set_xlabel("Measured sustained throughput (TFLOP/s)" if tensor else "Logical throughput (GB/s)")
    axes[0].set_ylabel("Energy (pJ/FLOP)" if tensor else "Energy (pJ/logical bit)")
    axes[1].set_xlabel("Achieved graphics clock (MHz)")
    axes[1].set_ylabel("Measured TFLOP/s" if tensor else "Logical GB/s")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{workload}: {title}\nRepeat medians; whole-device scope; NCU target admission shown by color")
    path=output/(workload+"-"+objective+"-energy-throughput.png")
    fig.savefig(path,dpi=160)
    plt.close(fig)
    files[workload+"_"+objective+"_plot"]=str(path.resolve())
