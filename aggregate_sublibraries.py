#!/usr/bin/env python
"""Combine N per-sublibrary filtered AnnData objects into one aggregated
AnnData, rerun CLR/PCA/PHATE jointly on the pooled raw counts, and surface the
saturation curve of whichever sublibrary had the best overall barcode/ADT-tag
recovery rate.

Usage:
    python aggregate_sublibraries.py --sublibrary-dir outs/Sublibrary_3 \\
        --sublibrary-dir outs/Sublibrary_4 --sublibrary-dir outs/Sublibrary_6 \\
        --sublibrary-dir outs/Sublibrary_7 --output-dir outs/combined
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from signal_lib import analysis, qc_plots


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sublibrary-dir", action="append", required=True, dest="sublibrary_dirs",
                    help="Output dir from process_sublibrary.py; repeatable.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_sublibrary_files(sub_dir: Path):
    h5ad_candidates = list(sub_dir.glob("*_filtered.h5ad"))
    stats_candidates = list(sub_dir.glob("*_qc_stats.json"))
    if not h5ad_candidates or not stats_candidates:
        raise FileNotFoundError(f"Expected *_filtered.h5ad and *_qc_stats.json in {sub_dir}")
    return h5ad_candidates[0], stats_candidates[0]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "combined")
    fig_prefix = str(fig_dir / "combined")

    adatas = []
    all_stats = []
    for sub_dir in args.sublibrary_dirs:
        sub_dir = Path(sub_dir)
        h5ad_path, stats_path = _find_sublibrary_files(sub_dir)
        log(f"Loading {h5ad_path}")
        a = sc.read_h5ad(h5ad_path)
        a.X = a.layers["counts"].copy()  # reset to raw counts; CLR/PCA/PHATE reran jointly below
        for key in ("clr", ):
            if key in a.layers:
                del a.layers[key]
        for key in ("X_pca", "X_phate"):
            if key in a.obsm:
                del a.obsm[key]
        adatas.append(a)
        with open(stats_path) as fh:
            stats = json.load(fh)
        stats["_source_dir"] = str(sub_dir)
        all_stats.append(stats)

    log(f"Concatenating {len(adatas)} sublibraries (obs_names already sublibrary-prefixed)")
    combined = ad.concat(adatas, join="outer", label="sublibrary_source",
                          keys=[s["sublibrary"] for s in all_stats], index_unique=None)
    combined.var = adatas[0].var
    n_dup = combined.obs_names.duplicated().sum()
    if n_dup:
        raise ValueError(f"{n_dup} duplicate cell barcodes across sublibraries after prefixing -- investigate")
    log(f"Combined: {combined.n_obs:,} cells x {combined.n_vars} ADTs from {len(adatas)} sublibraries")

    log("Rerunning CLR normalization + PCA + PHATE jointly on pooled data")
    t0 = time.time()
    combined = analysis.run_clr_pca_phate(combined)
    log(f"CLR/PCA/PHATE done in {time.time()-t0:.1f}s")

    qc_plots.plot_well_heatmaps(combined, fig_prefix, "combined")
    qc_plots.plot_clr_distributions(combined, fig_prefix)
    qc_plots.plot_pca_suite(combined, fig_prefix)
    qc_plots.plot_phate_suite(combined, fig_prefix)
    qc_plots.plot_violin_all_adts(combined, fig_prefix)
    qc_plots.plot_adt_correlation(combined, fig_prefix)
    qc_plots.plot_cells_per_sample(combined, fig_prefix)
    qc_plots.plot_qc_by_sample(combined, fig_prefix)
    qc_plots.plot_adt_heatmap_by_sample(combined, fig_prefix)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, sub in zip([s["sublibrary"] for s in all_stats], adatas):
        vals = np.log1p(sub.obs["total_counts"])
        ax.hist(vals, bins=40, histtype="step", linewidth=1.5, label=name, density=True)
    ax.set_xlabel("log1p(Total UMI Counts)"); ax.set_ylabel("Density")
    ax.set_title("Total UMI Counts per Cell -- by Sublibrary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    qc_plots._save(fig, fig_prefix, "umi_by_sublibrary")

    combined.write(f"{prefix}_filtered.h5ad")
    log(f"Combined AnnData -> {prefix}_filtered.h5ad")

    best = max(all_stats, key=lambda s: s["overall_valid_rate"])
    log(f"Best-quality sublibrary by overall valid-read rate: "
        f"{best['sublibrary']} ({100*best['overall_valid_rate']:.1f}%)")
    best_sat_csv = Path(best["_source_dir"]) / f"{best['sublibrary']}_saturation.csv"
    best_sat_png = Path(best["_source_dir"]) / "figures" / f"{best['sublibrary']}_saturation_curve.png"
    if best_sat_csv.exists():
        shutil.copy(best_sat_csv, out_dir / f"saturation_{best['sublibrary']}.csv")
    if best_sat_png.exists():
        shutil.copy(best_sat_png, fig_dir / f"saturation_curve_{best['sublibrary']}_BEST.png")

    combined_stats = {
        "n_sublibraries": len(all_stats),
        "sublibraries": [s["sublibrary"] for s in all_stats],
        "best_quality_sublibrary": best["sublibrary"],
        "best_quality_overall_valid_rate": best["overall_valid_rate"],
        "per_sublibrary_overall_valid_rate": {s["sublibrary"]: s["overall_valid_rate"] for s in all_stats},
        "total_cells_combined": int(combined.n_obs),
        "n_adts": int(combined.n_vars),
        "total_reads_all_sublibraries": sum(s["processing_stats"]["total_reads"] for s in all_stats),
        "total_valid_reads_all_sublibraries": sum(s["processing_stats"]["valid_all"] for s in all_stats),
        "median_umis_per_cell_combined": float(np.median(combined.obs["total_counts"])) if combined.n_obs else 0.0,
        "median_adts_per_cell_combined": float(np.median(combined.obs["n_adts_detected"])) if combined.n_obs else 0.0,
        "cells_per_condition_combined": {k: int(v) for k, v in combined.obs["condition"].value_counts().items()},
    }
    with open(f"{prefix}_qc_stats.json", "w") as fh:
        json.dump(combined_stats, fh, indent=2)
    log(f"Combined QC stats -> {prefix}_qc_stats.json")
    log("Done: aggregation")


if __name__ == "__main__":
    main()
