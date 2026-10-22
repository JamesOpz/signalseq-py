"""Process one ADT sublibrary: FASTQ pair -> raw + filtered AnnData, QC plots and
a per-cell downsampling (saturation) curve.

Stage order matches the original pipeline exactly, because several outputs depend
on it: the census runs before the scan, cell calling before the post-filter
plots, and the saturation curve uses the called-cell set from the full-depth call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import knee, matrix_build, provenance, qc_plots, sample_layout, saturation
from .bc_correct import BarcodeCorrector, run_census
from .config import Config
from .encoding import Encoder
from .hot_loop import scan_sublibrary
from .umi_dedup import collapse_umis
from .well_merge import WellMerge
from .whitelists import load_all


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_sublibrary(cfg: Config, name: str, debug_reads: int | None = None) -> Path:
    """Run the full per-sublibrary pipeline. Returns its output directory."""
    sub = cfg.sublibrary(name)
    out_dir = cfg.sublibrary_output_dir(name)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = cfg.outputs
    plots_on = bool(outputs["figures"]) and bool(cfg.qc["plots"]["enabled"])
    if plots_on:
        fig_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(out_dir / name)
    plotter = qc_plots.Plotter(str(fig_dir / name), cfg.qc["plots"], cfg.read_structure)

    provenance.write(cfg, out_dir, command=f"signalseq adt run --sublibrary {name}",
                     extra={"sublibrary": name, "debug_reads": debug_reads})

    rs = cfg.read_structure

    log(f"Loading whitelists for {name}")
    panel, whitelists = load_all(cfg)
    encoder = Encoder.build(
        [whitelists[el.name].n for el in rs.rounds], panel.n, rs.umi.length
    )
    log(f"Encoding: {encoder.describe()}")
    merge = WellMerge.build(rs, whitelists, cfg.whitelists, encoder)
    if merge is not None:
        log(f"Merging barcodes that share a well (per-primer columns: "
            f"{', '.join(merge.type_labels) or 'none'})")

    barcode_read = rs.rounds[0].read
    barcode_path = sub.read_path(barcode_read)

    log("Running barcode-correction census (Parse-matching combo-count threshold)")
    cb_cfg = cfg.matching["cell_barcode"]
    census = run_census(barcode_path, rs, whitelists, cb_cfg["census"],
                        cfg.performance["gzip_binaries"])
    log(f"Census: {census.n_distinct_combos:,} distinct exact combos, "
        f"count_thresh={census.count_threshold}")
    corrector = BarcodeCorrector.build(rs, whitelists, census, cb_cfg)

    log(f"Scanning FASTQ pair (debug_reads={debug_reads})")
    t0 = time.time()
    cell_id, adt_idx, umi, stats, reads_per_round = scan_sublibrary(
        sub.r1, sub.r2, rs, panel, whitelists, corrector, encoder,
        chunk_rows=int(cfg.performance["chunk_rows"]),
        gzip_binaries=cfg.performance["gzip_binaries"],
        max_reads=debug_reads,
    )
    dt = time.time() - t0
    log(f"Scanned {stats['total_reads']:,} reads in {dt:.1f}s "
        f"({stats['total_reads']/max(dt,1e-9):,.0f} reads/sec); {len(cell_id):,} valid reads")

    log("Collapsing UMIs (vectorized exact + mismatch-tolerant collapse)")
    umi_cfg = cfg.matching["umi"]
    cell_out, adt_out, counts, pair_reads = collapse_umis(
        cell_id, adt_idx, umi, encoder, umi_cfg, return_reads=True
    )
    log(f"{len(cell_out):,} distinct (cell,ADT) pairs; {counts.sum():,} molecules "
        f"({100*(1 - counts.sum()/max(len(cell_id),1)):.1f}% UMI duplication)")

    log("Building raw sparse AnnData")
    raw_adata = matrix_build.build_raw_anndata(
        cell_out, adt_out, counts, pair_reads, rs, panel, whitelists, encoder,
        stats, reads_per_round, name, parse_index=sub.parse_index, merge=merge,
    )

    log(f"Annotating sample identity from {cfg.layout.csv.name}")
    ann_cfg = dict(cfg.sample_annotation)
    layout_df = sample_layout.load_rt_layout(cfg.layout, ann_cfg)
    join_round = str(ann_cfg.get("join_on_round") or rs.cell_barcode_order[0])
    join_column = f"{rs[join_round].label}_well"
    sample_layout.annotate_sample_identity(raw_adata, layout_df, join_column, ann_cfg)

    if outputs["raw_anndata"]:
        raw_adata.write(f"{prefix}_raw.h5ad")
        log(f"Raw: {raw_adata.n_obs:,} candidate barcodes x {raw_adata.n_vars} ADTs "
            f"-> {prefix}_raw.h5ad")

    if plots_on:
        plotter.barcode_funnel(stats)
        stype_round = _round_with_stype(rs, whitelists) or join_round
        plotter.reads_per_round(
            reads_per_round[stype_round], whitelists[stype_round], rs[stype_round].label
        )

    filt_cfg = cfg.filtering
    if filt_cfg["min_umi_per_cell"] is not None:
        log(f"Cell calling: fixed floor of {filt_cfg['min_umi_per_cell']:.0f} UMIs/cell "
            f"(min_adts_detected={filt_cfg['min_adts_detected']}) -- knee still computed "
            f"for reference")
    else:
        log("Automatic knee-point cell calling")
    mask, knee_result, umi_threshold_used = knee.call_cells(raw_adata, filt_cfg)
    log(f"Knee: rank={knee_result.knee_rank:,}, knee UMI={knee_result.knee_umi:.0f}; "
        f"applied UMI threshold={umi_threshold_used:.0f}; {mask.sum():,} cells called")

    round_labels = [el.label for el in rs.rounds]
    if plots_on:
        plotter.knee(knee_result, umi_threshold_used)
        plotter.qc_distributions(raw_adata, "prefilter", knee_result, umi_threshold_used)
        plotter.well_heatmaps(raw_adata, "prefilter", round_labels)

    filtered = raw_adata[mask].copy()
    if plots_on:
        plotter.qc_distributions(filtered, "postfilter")
        plotter.well_heatmaps(filtered, "postfilter", round_labels)
        plotter.cells_per_sample(filtered)
        plotter.qc_by_sample(filtered)

    saturation_df = None
    if cfg.qc["saturation"]["enabled"]:
        log("Computing UMI + ADTs-detected downsampling (saturation) curve")
        called_ids = filtered.obs["cell_id_packed"].to_numpy()
        saturation_df = saturation.compute_saturation_curve(
            cell_id, adt_idx, umi, called_ids, encoder, umi_cfg,
            fractions=cfg.qc["saturation"]["fractions"],
            seed=int(cfg.qc["saturation"]["seed"]),
            merge=merge,
        )
        if outputs["saturation_table"]:
            saturation_df.to_csv(f"{prefix}_saturation.csv", index=False)
        if plots_on:
            plotter.saturation_curve(saturation_df)

    analysis_ran = False
    if cfg.qc["analysis"]["enabled"]:
        from . import analysis
        try:
            # Checked before the expensive work, so a doomed run says why up front
            # instead of failing later inside a solver.
            analysis.check_analysis_feasible(filtered, cfg.qc["analysis"])
        except analysis.InsufficientCellsError as exc:
            log(f"WARNING: skipping CLR/PCA/PHATE.\n{exc}")
            log("Counts, core QC and the remaining plots are still written.")
        else:
            log("Running CLR normalization + PCA + PHATE")
            t0 = time.time()
            filtered = analysis.run_clr_pca_phate(filtered, cfg.qc["analysis"], cfg.threads)
            log(f"CLR/PCA/PHATE done in {time.time()-t0:.1f}s")
            analysis_ran = True

    if analysis_ran:
        if plots_on:
            plotter.clr_distributions(filtered)
            plotter.pca_suite(filtered)
            plotter.phate_suite(filtered)
            plotter.violin_all_adts(filtered)
            plotter.adt_correlation(filtered)
            plotter.adt_heatmap_by_sample(filtered)

    if outputs["filtered_anndata"]:
        filtered.write(f"{prefix}_filtered.h5ad")
        log(f"Filtered: {filtered.n_obs:,} cells x {filtered.n_vars} ADTs "
            f"-> {prefix}_filtered.h5ad")

    if outputs["qc_stats"]:
        qc_stats = _build_qc_stats(name, stats, raw_adata, filtered, counts, cell_id,
                                   knee_result, umi_threshold_used, round_labels)
        with open(f"{prefix}_qc_stats.json", "w") as fh:
            json.dump(qc_stats, fh, indent=2)
        log(f"QC stats -> {prefix}_qc_stats.json")

    log(f"Done: {name}")
    return out_dir


def _round_with_stype(rs, whitelists) -> str | None:
    """First barcode round whose whitelist carries a sample-type column."""
    for el in rs.rounds:
        if whitelists[el.name].stype is not None:
            return el.name
    return None


def _build_qc_stats(name, stats, raw_adata, filtered, counts, cell_id,
                    knee_result, umi_threshold_used, round_labels) -> dict:
    well_stats = {
        f"n_{label.lower()}_wells": int(filtered.obs[f"{label}_well"].nunique()) if filtered.n_obs else 0
        for label in round_labels
    }
    return {
        "sublibrary": name,
        "processing_stats": stats,
        "matrix_stats": {
            "n_candidate_barcodes": int(raw_adata.n_obs),
            "n_adts": int(raw_adata.n_vars),
            "n_cells_called": int(filtered.n_obs),
            "total_umis_called_cells": float(filtered.obs["total_counts"].sum()),
            "median_umis_per_cell": float(np.median(filtered.obs["total_counts"])) if filtered.n_obs else 0.0,
            "median_adts_per_cell": float(np.median(filtered.obs["n_adts_detected"])) if filtered.n_obs else 0.0,
            "median_reads_per_cell": float(np.median(filtered.obs["total_reads"])) if filtered.n_obs else 0.0,
            "umis_called_cells_by_primer": {
                col.removeprefix("total_counts_"): float(filtered.obs[col].sum())
                for col in filtered.obs.columns if col.startswith("total_counts_")
            },
            "umi_duplication_rate": float(1 - counts.sum() / max(len(cell_id), 1)),
        },
        "knee": {
            "knee_rank": knee_result.knee_rank,
            "knee_umi": knee_result.knee_umi,
            "umi_threshold_used": umi_threshold_used,
        },
        "well_stats": well_stats,
        "cells_per_condition": (
            {k: int(v) for k, v in filtered.obs["condition"].value_counts().items()}
            if filtered.n_obs else {}
        ),
        "overall_valid_rate": stats["valid_all"] / max(stats["total_reads"], 1),
    }
