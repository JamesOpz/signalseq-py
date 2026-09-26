"""Combine per-sublibrary filtered AnnData objects into one aggregated object.

CLR/PCA/PHATE is rerun jointly on the pooled raw counts rather than reusing the
per-sublibrary embeddings, so cells from different sublibraries are directly
comparable. The saturation curve of whichever sublibrary had the best overall
barcode/ADT recovery is surfaced alongside it.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from . import provenance, qc_plots
from .config import Config
from .readstructure import ConfigError


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_sublibrary_files(sub_dir: Path) -> tuple[Path, Path]:
    h5ad = sorted(sub_dir.glob("*_filtered.h5ad"))
    stats = sorted(sub_dir.glob("*_qc_stats.json"))
    if not h5ad or not stats:
        raise ConfigError(
            f"Expected *_filtered.h5ad and *_qc_stats.json in {sub_dir}. "
            f"Run `signalseq adt run --sublibrary <name>` for it first."
        )
    return h5ad[0], stats[0]


def aggregate(cfg: Config, sublibrary_dirs: list[Path] | None = None) -> Path:
    """Aggregate the given sublibrary output dirs (default: all in the config)."""
    if sublibrary_dirs is None:
        sublibrary_dirs = [cfg.sublibrary_output_dir(s.name) for s in cfg.sublibraries]
    if not sublibrary_dirs:
        raise ConfigError("Nothing to aggregate: no sublibrary directories given")

    out_dir = cfg.combined_output_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = cfg.outputs
    agg_outputs = outputs["aggregate"]
    plots_on = bool(outputs["figures"]) and bool(cfg.qc["plots"]["enabled"])
    if plots_on:
        fig_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(out_dir / "combined")
    plotter = qc_plots.Plotter(str(fig_dir / "combined"), cfg.qc["plots"], cfg.read_structure)

    provenance.write(cfg, out_dir, command="signalseq adt aggregate",
                     extra={"sublibrary_dirs": [str(d) for d in sublibrary_dirs]})

    adatas = []
    all_stats = []
    for sub_dir in sublibrary_dirs:
        sub_dir = Path(sub_dir)
        h5ad_path, stats_path = _find_sublibrary_files(sub_dir)
        log(f"Loading {h5ad_path}")
        a = sc.read_h5ad(h5ad_path)
        # Reset to raw counts; CLR/PCA/PHATE are rerun jointly below.
        a.X = a.layers["counts"].copy()
        for key in ("clr",):
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
        raise ConfigError(
            f"{n_dup} duplicate cell barcodes across sublibraries after prefixing. "
            f"Check that read_structure.cell_barcode.name_template includes {{sublibrary}}."
        )
    log(f"Combined: {combined.n_obs:,} cells x {combined.n_vars} ADTs "
        f"from {len(adatas)} sublibraries")

    analysis_ran = False
    if cfg.qc["analysis"]["enabled"]:
        from . import analysis
        try:
            analysis.check_analysis_feasible(combined, cfg.qc["analysis"])
        except analysis.InsufficientCellsError as exc:
            log(f"WARNING: skipping CLR/PCA/PHATE.\n{exc}")
        else:
            log("Rerunning CLR normalization + PCA + PHATE jointly on pooled data")
            t0 = time.time()
            combined = analysis.run_clr_pca_phate(combined, cfg.qc["analysis"], cfg.threads)
            log(f"CLR/PCA/PHATE done in {time.time()-t0:.1f}s")
            analysis_ran = True

    if plots_on:
        round_labels = [el.label for el in cfg.read_structure.rounds]
        plotter.well_heatmaps(combined, "combined", round_labels)
        if analysis_ran:
            plotter.clr_distributions(combined)
            plotter.pca_suite(combined)
            plotter.phate_suite(combined)
            plotter.violin_all_adts(combined)
            plotter.adt_correlation(combined)
        plotter.cells_per_sample(combined)
        plotter.qc_by_sample(combined)
        if analysis_ran:
            plotter.adt_heatmap_by_sample(combined)
        plotter.umi_by_sublibrary(adatas, [s["sublibrary"] for s in all_stats])

    if agg_outputs["combined_anndata"]:
        combined.write(f"{prefix}_filtered.h5ad")
        log(f"Combined AnnData -> {prefix}_filtered.h5ad")

    best = max(all_stats, key=lambda s: s["overall_valid_rate"])
    log(f"Best-quality sublibrary by overall valid-read rate: "
        f"{best['sublibrary']} ({100*best['overall_valid_rate']:.1f}%)")

    if agg_outputs["copy_best_saturation"]:
        src_dir = Path(best["_source_dir"])
        best_csv = src_dir / f"{best['sublibrary']}_saturation.csv"
        best_png = src_dir / "figures" / f"{best['sublibrary']}_saturation_curve.png"
        if best_csv.exists():
            shutil.copy(best_csv, out_dir / f"saturation_{best['sublibrary']}.csv")
        if best_png.exists() and plots_on:
            shutil.copy(best_png, fig_dir / f"saturation_curve_{best['sublibrary']}_BEST.png")

    if agg_outputs["combined_qc_stats"]:
        combined_stats = {
            "n_sublibraries": len(all_stats),
            "sublibraries": [s["sublibrary"] for s in all_stats],
            "best_quality_sublibrary": best["sublibrary"],
            "best_quality_overall_valid_rate": best["overall_valid_rate"],
            "per_sublibrary_overall_valid_rate": {
                s["sublibrary"]: s["overall_valid_rate"] for s in all_stats
            },
            "total_cells_combined": int(combined.n_obs),
            "n_adts": int(combined.n_vars),
            "total_reads_all_sublibraries": sum(
                s["processing_stats"]["total_reads"] for s in all_stats
            ),
            "total_valid_reads_all_sublibraries": sum(
                s["processing_stats"]["valid_all"] for s in all_stats
            ),
            "median_umis_per_cell_combined": (
                float(np.median(combined.obs["total_counts"])) if combined.n_obs else 0.0
            ),
            "median_adts_per_cell_combined": (
                float(np.median(combined.obs["n_adts_detected"])) if combined.n_obs else 0.0
            ),
            "cells_per_condition_combined": {
                k: int(v) for k, v in combined.obs["condition"].value_counts().items()
            },
        }
        with open(f"{prefix}_qc_stats.json", "w") as fh:
            json.dump(combined_stats, fh, indent=2)
        log(f"Combined QC stats -> {prefix}_qc_stats.json")

    log("Done: aggregation")
    return out_dir
