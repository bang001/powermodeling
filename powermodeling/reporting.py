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
    for workload in workloads:
        groups=[g for g in summary["groups"] if g["workload"]==workload and g["valid_repeats"]>=3]
        tensor=workload in ("tensor","gemm")
        rate="throughput_ops_s" if tensor else "throughput_bytes_s"
        energy="pj_per_op" if tensor else "pj_per_logical_byte"
        groups=[g for g in groups if g.get(rate) is not None and g.get(energy) is not None]
        if not groups: continue
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
        axes[0].set_ylabel("Incremental energy (pJ/FLOP)" if tensor else "Incremental energy (pJ/logical byte)")
        axes[1].set_xlabel("Achieved graphics clock (MHz)")
        axes[1].set_ylabel("Measured TFLOP/s" if tensor else "Logical GB/s")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{workload}: repeat medians; device incremental energy; ncu target admission shown by color")
        path=output/(workload+"-energy-throughput.png")
        fig.savefig(path,dpi=160)
        plt.close(fig)
        files[workload+"_plot"]=str(path.resolve())
    return files
