#!/usr/bin/env python
"""Process one ADT sublibrary: FASTQ pair -> raw + knee-filtered AnnData,
full QC/composition plots, and a per-cell UMI/ADT downsampling (saturation)
curve. Run once per sublibrary before aggregate_sublibraries.py.

Usage:
    python process_sublibrary.py --r1 R1.fastq.gz --r2 R2.fastq.gz \\
        --name Sublibrary_3 --output-dir outs/Sublibrary_3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from signal_lib import (analysis, bc_correct, config, hot_loop, knee, matrix_build,
                         qc_plots, sample_layout, saturation, whitelists)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r1", required=True, help="R1 FASTQ (ADT tag read), .fastq.gz")
    p.add_argument("--r2", required=True, help="R2 FASTQ (cell barcode + UMI read), .fastq.gz")
    p.add_argument("--name", required=True, help="Sublibrary name, e.g. Sublibrary_3")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-adts-floor", type=int, default=config.DEFAULT_MIN_ADTS_FLOOR)
    p.add_argument("--max-umi-per-cell", type=float, default=None)
    p.add_argument("--min-umi-per-cell", type=float, default=None,
                    help="Fixed UMI floor overriding automatic knee detection "
                         "(e.g. for a permissive first-pass cut ahead of joint RNA+ADT filtering). "
                         "The knee is still computed and shown on the QC plots for reference.")
    p.add_argument("--rt-layout-csv", default=str(sample_layout.DEFAULT_RT_LAYOUT_CSV),
                    help="RT-well -> experimental condition layout CSV.")
    p.add_argument("--debug-reads", type=int, default=None,
                    help="Only process this many reads (for smoke-testing).")
    p.add_argument("--skip-analysis", action="store_true",
                    help="Skip CLR/PCA/PHATE + saturation curve (matrix + core QC only).")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / args.name)
    fig_prefix = str(fig_dir / args.name)

    log(f"Loading whitelists for {args.name}")
    adt_wl = whitelists.load_adt_whitelist()
    rt_wl = whitelists.load_rt_whitelist()
    lig_wl = whitelists.load_ligation_whitelist()

    log("Running barcode-correction census (Parse-matching combo-count threshold)")
    combo_counts, count_thresh = bc_correct.run_census(args.r2, rt_wl, lig_wl)
    log(f"Census: {len(combo_counts):,} distinct exact combos, count_thresh={count_thresh}")

    log(f"Scanning FASTQ pair (debug_reads={args.debug_reads})")
    t0 = time.time()
    cell_id, adt_idx, umi, stats, reads_per_rt = hot_loop.scan_sublibrary(
        args.r1, args.r2, adt_wl, rt_wl, lig_wl, combo_counts, count_thresh,
        debug_reads=args.debug_reads,
    )
    dt = time.time() - t0
    log(f"Scanned {stats['total_reads']:,} reads in {dt:.1f}s "
        f"({stats['total_reads']/max(dt,1e-9):,.0f} reads/sec); {len(cell_id):,} valid reads")

    log("Collapsing UMIs (vectorized exact + directional 1bp collapse)")
    from signal_lib.umi_dedup import collapse_umis
    cell_out, adt_out, counts = collapse_umis(cell_id, adt_idx, umi)
    log(f"{len(cell_out):,} distinct (cell,ADT) pairs; {counts.sum():,} molecules "
        f"({100*(1 - counts.sum()/max(len(cell_id),1)):.1f}% UMI duplication)")

    log("Building raw sparse AnnData")
    raw_adata = matrix_build.build_raw_anndata(
        cell_out, adt_out, counts, adt_wl, rt_wl, lig_wl, stats, reads_per_rt, args.name
    )
    log(f"Annotating sample identity from {args.rt_layout_csv}")
    rt_layout_df = sample_layout.load_rt_layout(args.rt_layout_csv)
    sample_layout.annotate_sample_identity(raw_adata, rt_layout_df)

    raw_adata.write(f"{prefix}_raw.h5ad")
    log(f"Raw: {raw_adata.n_obs:,} candidate barcodes x {raw_adata.n_vars} ADTs -> {prefix}_raw.h5ad")

    qc_plots.plot_barcode_funnel(stats, fig_prefix)
    qc_plots.plot_reads_per_rt(reads_per_rt, rt_wl, fig_prefix)

    if args.min_umi_per_cell is not None:
        log(f"Cell calling: fixed permissive floor of {args.min_umi_per_cell:.0f} UMIs/cell "
            f"(min_adts_floor={args.min_adts_floor}) -- knee still computed for reference")
    else:
        log("Automatic knee-point cell calling")
    mask, knee_result, umi_threshold_used = knee.call_cells(
        raw_adata, args.min_adts_floor, args.max_umi_per_cell, args.min_umi_per_cell
    )
    log(f"Knee: rank={knee_result.knee_rank:,}, knee UMI={knee_result.knee_umi:.0f}; "
        f"applied UMI threshold={umi_threshold_used:.0f}; {mask.sum():,} cells called")

    qc_plots.plot_knee(knee_result, fig_prefix, umi_threshold_used)
    qc_plots.plot_qc_distributions(raw_adata, fig_prefix, "prefilter", knee_result, umi_threshold_used)
    qc_plots.plot_well_heatmaps(raw_adata, fig_prefix, "prefilter")

    filtered = raw_adata[mask].copy()
    qc_plots.plot_qc_distributions(filtered, fig_prefix, "postfilter")
    qc_plots.plot_well_heatmaps(filtered, fig_prefix, "postfilter")
    qc_plots.plot_cells_per_sample(filtered, fig_prefix)
    qc_plots.plot_qc_by_sample(filtered, fig_prefix)

    saturation_df = None
    if not args.skip_analysis:
        log("Computing UMI + ADTs-detected downsampling (saturation) curve")
        called_ids = filtered.obs["cell_id_packed"].to_numpy()
        saturation_df = saturation.compute_saturation_curve(cell_id, adt_idx, umi, called_ids)
        saturation_df.to_csv(f"{prefix}_saturation.csv", index=False)
        qc_plots.plot_saturation_curve(saturation_df, fig_prefix)

        log("Running CLR normalization + PCA + PHATE")
        t0 = time.time()
        filtered = analysis.run_clr_pca_phate(filtered)
        log(f"CLR/PCA/PHATE done in {time.time()-t0:.1f}s")

        qc_plots.plot_clr_distributions(filtered, fig_prefix)
        qc_plots.plot_pca_suite(filtered, fig_prefix)
        qc_plots.plot_phate_suite(filtered, fig_prefix)
        qc_plots.plot_violin_all_adts(filtered, fig_prefix)
        qc_plots.plot_adt_correlation(filtered, fig_prefix)
        qc_plots.plot_adt_heatmap_by_sample(filtered, fig_prefix)

    filtered.write(f"{prefix}_filtered.h5ad")
    log(f"Filtered: {filtered.n_obs:,} cells x {filtered.n_vars} ADTs -> {prefix}_filtered.h5ad")

    qc_stats = {
        "sublibrary": args.name,
        "processing_stats": stats,
        "matrix_stats": {
            "n_candidate_barcodes": int(raw_adata.n_obs),
            "n_adts": int(raw_adata.n_vars),
            "n_cells_called": int(filtered.n_obs),
            "total_umis_called_cells": float(filtered.obs["total_counts"].sum()),
            "median_umis_per_cell": float(np.median(filtered.obs["total_counts"])) if filtered.n_obs else 0.0,
            "median_adts_per_cell": float(np.median(filtered.obs["n_adts_detected"])) if filtered.n_obs else 0.0,
            "umi_duplication_rate": float(1 - counts.sum() / max(len(cell_id), 1)),
        },
        "knee": {
            "knee_rank": knee_result.knee_rank,
            "knee_umi": knee_result.knee_umi,
            "umi_threshold_used": umi_threshold_used,
        },
        "well_stats": {
            "n_rt_wells": int(filtered.obs["RT_well"].nunique()) if filtered.n_obs else 0,
            "n_r2_wells": int(filtered.obs["R2_well"].nunique()) if filtered.n_obs else 0,
            "n_r3_wells": int(filtered.obs["R3_well"].nunique()) if filtered.n_obs else 0,
        },
        "cells_per_condition": {k: int(v) for k, v in filtered.obs["condition"].value_counts().items()} if filtered.n_obs else {},
        "overall_valid_rate": stats["valid_all"] / max(stats["total_reads"], 1),
    }
    with open(f"{prefix}_qc_stats.json", "w") as fh:
        json.dump(qc_stats, fh, indent=2)
    log(f"QC stats -> {prefix}_qc_stats.json")
    log(f"Done: {args.name}")


if __name__ == "__main__":
    main()
