"""QC and composition plots.

Ported from the pilot notebooks into a `Plotter` that carries the config-driven
geometry (plate shape, DPI, grid width, colour percentiles, category orders and
palettes), so the same code draws both the per-sublibrary and the aggregated
figures and none of the layout is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .readstructure import ReadStructure


@dataclass
class Plotter:
    """Draws every figure for one output prefix.

    `prefix` is a path stem: files land at `<prefix>_<suffix>.png`.
    """

    prefix: str
    cfg: Mapping
    read_structure: ReadStructure | None = None

    # -- setup -------------------------------------------------------------
    def __post_init__(self):
        style = self.cfg.get("style")
        if style:
            try:
                plt.style.use(style)
            except Exception:
                # A missing style should degrade the look, not fail the run.
                pass

    @property
    def plate_rows(self) -> list:
        return list(self.cfg.get("plate_rows") or [])

    @property
    def plate_cols(self) -> list:
        return list(self.cfg.get("plate_cols") or [])

    @property
    def grid_columns(self) -> int:
        return int(self.cfg.get("grid_columns", 4))

    @property
    def percentiles(self) -> tuple[float, float]:
        lo, hi = self.cfg.get("color_percentiles", [5, 95])
        return float(lo), float(hi)

    def _save(self, fig, suffix: str) -> Path:
        path = Path(f"{self.prefix}_{suffix}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=int(self.cfg.get("dpi", 150)), bbox_inches="tight")
        plt.close(fig)
        return path

    # -- shared primitives -------------------------------------------------
    @staticmethod
    def _smoothed_hist(data, ax, color="steelblue", label=None, xlabel=None, title=None,
                       log_scale=False, threshold_lines=None):
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data) & ~np.isinf(data)]
        if len(data) < 2:
            return
        if log_scale:
            data = np.log1p(data)
        kde = gaussian_kde(data, bw_method="scott")
        x_range = np.linspace(data.min(), data.max(), 200)
        density = kde(x_range)
        ax.fill_between(x_range, density, alpha=0.3, color=color)
        ax.plot(x_range, density, color=color, linewidth=2, label=label)
        if threshold_lines:
            for val, c, lbl in threshold_lines:
                v = np.log1p(val) if log_scale else val
                ax.axvline(v, color=c, linestyle="--", linewidth=1.5, label=lbl)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        if title:
            ax.set_title(title)
        if label or threshold_lines:
            ax.legend(fontsize=8)

    def plate_heatmap(self, well_counts, ax, title, cmap="viridis", annot=True):
        rows, cols = self.plate_rows, self.plate_cols
        plate_matrix = np.zeros((len(rows), len(cols)))
        counts = well_counts.to_dict() if hasattr(well_counts, "to_dict") else dict(well_counts)
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                plate_matrix[i, j] = counts.get(f"{row}{col}", 0)

        im = ax.imshow(plate_matrix, cmap=cmap, aspect="equal")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
        if annot:
            vmax = plate_matrix.max()
            for i in range(len(rows)):
                for j in range(len(cols)):
                    val = int(plate_matrix[i, j])
                    if val > 0:
                        color = "white" if plate_matrix[i, j] > vmax * 0.5 else "black"
                        ax.text(j, i, f"{val}", ha="center", va="center", fontsize=7,
                                color=color, fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        return im

    def _dense(self, adata):
        return adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    # -- funnel / knee -----------------------------------------------------
    def barcode_funnel(self, stats: dict):
        fig, ax = plt.subplots(figsize=(7, 5))
        stages = ["Total\nReads", "Valid\nADT", "Valid cell\nbarcode"]
        counts = [stats["total_reads"], stats["valid_adt"], stats["valid_all"]]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stages)))
        bars = ax.bar(stages, counts, color=colors)
        ax.set_ylabel("Number of Reads")
        ax.set_title("Barcode Matching Funnel")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        for bar, count in zip(bars, counts):
            pct = 100 * count / max(stats["total_reads"], 1)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        return self._save(fig, "funnel")

    def knee(self, knee_result, umi_threshold_used: float | None = None):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(knee_result.ranks, knee_result.sorted_counts, "b-", linewidth=1)
        ax.axvline(knee_result.knee_rank, color="red", linestyle="--",
                   label=f"Knee: rank {knee_result.knee_rank:,}, {knee_result.knee_umi:.0f} UMIs")
        if umi_threshold_used is not None and umi_threshold_used != knee_result.knee_umi:
            ax.axhline(umi_threshold_used, color="darkorange", linestyle=":",
                       label=f"Applied threshold: {umi_threshold_used:.0f} UMIs")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Barcode Rank"); ax.set_ylabel("UMI Counts")
        ax.set_title("Knee Plot (Automatic Cell Calling)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "knee")

    # -- distributions -----------------------------------------------------
    def qc_distributions(self, adata, stage: str, knee_result=None,
                         umi_threshold_used: float | None = None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        thresholds = None
        if knee_result is not None or umi_threshold_used is not None:
            thresholds = []
            if umi_threshold_used is not None:
                thresholds.append((umi_threshold_used, "red", f"Applied: {umi_threshold_used:.0f}"))
            if knee_result is not None and knee_result.knee_umi != umi_threshold_used:
                thresholds.append((knee_result.knee_umi, "gray", f"Knee: {knee_result.knee_umi:.0f}"))
        self._smoothed_hist(adata.obs["total_counts"], axes[0, 0], color="steelblue",
                            xlabel="log1p(Total UMI Counts)", title="Total UMI Counts per Cell",
                            log_scale=True, threshold_lines=thresholds)
        self._smoothed_hist(adata.obs["n_adts_detected"], axes[0, 1], color="coral",
                            xlabel="Number of ADTs Detected", title="ADTs Detected per Cell")

        ax = axes[0, 2]
        ax.scatter(adata.obs["total_counts"], adata.obs["n_adts_detected"],
                   alpha=0.3, s=5, c="mediumpurple")
        ax.set_xscale("log")
        ax.set_xlabel("Total UMI Counts"); ax.set_ylabel("Number of ADTs Detected")
        ax.set_title("UMI vs ADTs Detected")

        adt_order = adata.var["total_counts"].sort_values(ascending=True).index
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(adt_order)))

        ax = axes[1, 0]
        ax.barh(range(len(adt_order)), adata.var.loc[adt_order, "total_counts"], color=colors)
        ax.set_yticks(range(len(adt_order))); ax.set_yticklabels(adt_order, fontsize=8)
        ax.set_xlabel("Total UMI Counts"); ax.set_title("Total Counts per ADT")

        ax = axes[1, 1]
        ax.barh(range(len(adt_order)), adata.var.loc[adt_order, "detection_rate"] * 100, color=colors)
        ax.set_yticks(range(len(adt_order))); ax.set_yticklabels(adt_order, fontsize=8)
        ax.set_xlabel("Detection Rate (%)"); ax.set_title("Detection Rate per ADT")

        ax = axes[1, 2]
        colors_adt = plt.cm.tab20(np.linspace(0, 1, adata.n_vars))
        X = self._dense(adata)
        for i, name in enumerate(adata.var_names):
            vals = X[:, i]
            vals = vals[vals > 0]
            if len(vals) > 10:
                log_vals = np.log1p(vals)
                try:
                    kde = gaussian_kde(log_vals, bw_method="scott")
                    xr = np.linspace(0, log_vals.max(), 100)
                    ax.plot(xr, kde(xr), color=colors_adt[i], linewidth=1.5, label=name, alpha=0.7)
                except Exception:
                    pass
        ax.set_xlabel("log1p(UMI Counts)"); ax.set_ylabel("Density")
        ax.set_title("Per-ADT Count Distributions")
        ax.legend(fontsize=6, loc="upper right", ncol=2)

        fig.tight_layout()
        return self._save(fig, f"qc_{stage}")

    def well_heatmaps(self, adata, stage: str, round_labels: Sequence[str]):
        """One plate heatmap per cell-barcode round."""
        cmaps = list(self.cfg.get("well_heatmap_cmaps") or ["Blues", "Greens", "Oranges"])
        n = len(round_labels)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        axes = np.atleast_1d(axes)
        for i, label in enumerate(round_labels):
            col = f"{label}_well"
            im = self.plate_heatmap(adata.obs[col].value_counts(), axes[i],
                                    f"Cells per {label} Well ({stage})",
                                    cmap=cmaps[i % len(cmaps)])
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="Cell Count")
        fig.tight_layout()
        return self._save(fig, f"wells_{stage}")

    def reads_per_round(self, reads_per_round: np.ndarray, whitelist, label: str):
        fig, ax = plt.subplots(figsize=(18, 6))
        stype_colors = dict(self.cfg.get("stype_colors") or {})
        if whitelist.stype is not None and stype_colors:
            colors = [stype_colors.get(s, "gray") for s in whitelist.stype]
        else:
            colors = "steelblue"
        ax.bar(range(whitelist.n), reads_per_round, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(whitelist.n))
        ax.set_xticklabels(whitelist.wells, rotation=90, fontsize=7)
        ax.set_xlabel(f"{label} Barcode (Well ID)"); ax.set_ylabel("Number of Reads")
        ax.set_title(f"Reads per {label} Barcode (n={whitelist.n} barcodes)")
        if whitelist.stype is not None and stype_colors:
            legend = [Patch(facecolor=c, label=s) for s, c in stype_colors.items()]
            ax.legend(handles=legend, loc="upper right")
        fig.tight_layout()
        return self._save(fig, f"reads_per_{label.lower()}_barcode")

    def clr_distributions(self, adata):
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_adt = plt.cm.tab20(np.linspace(0, 1, adata.n_vars))
        X = self._dense(adata)
        for i, name in enumerate(adata.var_names):
            vals = X[:, i]
            if len(vals) > 10:
                try:
                    kde = gaussian_kde(vals, bw_method="scott")
                    xr = np.linspace(vals.min(), vals.max(), 100)
                    ax.plot(xr, kde(xr), color=colors_adt[i], linewidth=1.5, label=name, alpha=0.7)
                except Exception:
                    pass
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("CLR-normalized Expression"); ax.set_ylabel("Density")
        ax.set_title("CLR-normalized ADT Distributions")
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        fig.tight_layout()
        return self._save(fig, "clr_normalized")

    # -- embeddings --------------------------------------------------------
    def pca_suite(self, adata):
        import scanpy as sc
        paths = []

        plt.figure(figsize=(6, 4))
        sc.pl.pca_variance_ratio(adata, n_pcs=adata.obsm["X_pca"].shape[1],
                                 log=False, show=False, save=False)
        plt.title("PCA Variance Explained")
        plt.tight_layout()
        paths.append(self._save(plt.gcf(), "pca_variance"))

        plt.subplots(1, 2, figsize=(10, 4))
        sc.pl.pca(adata, color=["total_counts", "n_adts_detected"], ax=None, ncols=2,
                  cmap="viridis", show=False, save=False)
        paths.append(self._save(plt.gcf(), "pca_scatter"))

        paths.append(self._per_adt_grid(adata, adata.obsm["X_pca"], "PC1", "PC2",
                                        "PCA - ADT Expression (CLR normalized)", "pca_per_adt"))
        return paths

    def phate_suite(self, adata):
        paths = []
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc_ = axes[0].scatter(adata.obsm["X_phate"][:, 0], adata.obsm["X_phate"][:, 1],
                              c=np.log1p(adata.obs["total_counts"]), cmap="viridis", s=10, alpha=0.7)
        plt.colorbar(sc_, ax=axes[0], label="log1p(Total Counts)")
        axes[0].set_xlabel("PHATE1"); axes[0].set_ylabel("PHATE2")
        axes[0].set_title("PHATE - Total Counts")

        sc_ = axes[1].scatter(adata.obsm["X_phate"][:, 0], adata.obsm["X_phate"][:, 1],
                              c=adata.obs["n_adts_detected"], cmap="plasma", s=10, alpha=0.7)
        plt.colorbar(sc_, ax=axes[1], label="ADTs Detected")
        axes[1].set_xlabel("PHATE1"); axes[1].set_ylabel("PHATE2")
        axes[1].set_title("PHATE - ADTs Detected")
        fig.tight_layout()
        paths.append(self._save(fig, "phate_qc"))

        paths.append(self._per_adt_grid(adata, adata.obsm["X_phate"], "PHATE1", "PHATE2",
                                        "PHATE - ADT Expression (CLR normalized)", "phate_per_adt"))
        return paths

    def _per_adt_grid(self, adata, embedding, xlabel, ylabel, suptitle, suffix):
        n_adts = adata.n_vars
        n_cols = self.grid_columns
        n_rows = int(np.ceil(n_adts / n_cols))
        lo, hi = self.percentiles
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        X = self._dense(adata)
        for i, name in enumerate(adata.var_names):
            ax = axes[i]
            vals = X[:, i]
            sc_ = ax.scatter(embedding[:, 0], embedding[:, 1], c=vals, cmap="RdYlBu_r",
                             s=8, alpha=0.7, vmin=np.percentile(vals, lo), vmax=np.percentile(vals, hi))
            plt.colorbar(sc_, ax=ax, shrink=0.8)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(name, fontsize=10)
        for i in range(n_adts, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle(suptitle, fontsize=14, y=1.02)
        fig.tight_layout()
        return self._save(fig, suffix)

    def violin_all_adts(self, adata):
        import scanpy as sc
        fig, ax = plt.subplots(figsize=(14, 5))
        sc.pl.violin(adata, keys=list(adata.var_names), rotation=45, ax=ax, show=False)
        ax.set_title("CLR-normalized ADT Expression"); ax.set_ylabel("CLR Expression")
        fig.tight_layout()
        return self._save(fig, "violin_all_adts")

    def adt_correlation(self, adata):
        X = self._dense(adata)
        corr = np.corrcoef(X.T)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(adata.n_vars)); ax.set_yticks(range(adata.n_vars))
        ax.set_xticklabels(adata.var_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(adata.var_names, fontsize=9)
        ax.set_title("ADT-ADT Correlation (CLR normalized)")
        for i in range(adata.n_vars):
            for j in range(adata.n_vars):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
        fig.tight_layout()
        return self._save(fig, "adt_correlation")

    # -- per-condition composition ----------------------------------------
    def _treatment_rank(self) -> dict:
        order = list(self.cfg.get("treatment_order") or [])
        return {name: i for i, name in enumerate(order)}

    def _culture_colors(self) -> dict:
        return dict(self.cfg.get("culture_colors") or {})

    def _ordered_conditions(self, adata):
        """Conditions present, ordered by (culture_type, treatment, timepoint)."""
        df = adata.obs[["condition", "culture_type", "treatment", "timepoint_hrs"]] \
            .drop_duplicates("condition")
        df = df.astype({"condition": str, "culture_type": str, "treatment": str})
        df = df.assign(_torder=df["treatment"].map(self._treatment_rank()).fillna(9))
        df = df.sort_values(["culture_type", "_torder", "timepoint_hrs"])
        return df["condition"].tolist(), df["culture_type"].tolist()

    def cells_per_sample(self, adata):
        order, cultures = self._ordered_conditions(adata)
        counts = adata.obs["condition"].value_counts().reindex(order).fillna(0)
        palette = self._culture_colors()
        colors = [palette.get(c, "gray") for c in cultures]
        unassigned = str(self.cfg.get("unassigned_label", "Unassigned"))

        fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.6), 5))
        ax.bar(range(len(order)), counts.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of Cells")
        ax.set_title(f"Cells per Sample Condition (n={int(counts.sum()):,} total)")
        legend = [Patch(facecolor=col, label=name)
                  for name, col in palette.items() if name != unassigned]
        ax.legend(handles=legend, fontsize=8)
        fig.tight_layout()
        return self._save(fig, "cells_per_sample")

    def qc_by_sample(self, adata):
        order, _ = self._ordered_conditions(adata)
        order = [c for c in order if (adata.obs["condition"] == c).sum() > 0]

        fig, axes = plt.subplots(2, 1, figsize=(max(8, len(order) * 0.6), 9))

        umi_by_cond = [np.log1p(adata.obs.loc[adata.obs["condition"] == c, "total_counts"].to_numpy())
                       for c in order]
        axes[0].boxplot(umi_by_cond, showfliers=False)
        axes[0].set_xticks(range(1, len(order) + 1))
        axes[0].set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("log1p(Total UMI Counts)")
        axes[0].set_title("UMI Counts per Cell by Sample")

        adt_by_cond = [adata.obs.loc[adata.obs["condition"] == c, "n_adts_detected"].to_numpy()
                       for c in order]
        axes[1].boxplot(adt_by_cond, showfliers=False)
        axes[1].set_xticks(range(1, len(order) + 1))
        axes[1].set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("ADTs Detected")
        axes[1].set_title("ADTs Detected per Cell by Sample")

        fig.tight_layout()
        return self._save(fig, "qc_by_sample")

    def adt_heatmap_by_sample(self, adata):
        """Mean expression per condition x ADT. Intended for CLR-normalized data."""
        order, _ = self._ordered_conditions(adata)
        order = [c for c in order if (adata.obs["condition"] == c).sum() > 0]

        X = self._dense(adata)
        means = np.zeros((len(order), adata.n_vars))
        for i, c in enumerate(order):
            mask = (adata.obs["condition"] == c).to_numpy()
            means[i] = X[mask].mean(axis=0)

        fig, ax = plt.subplots(figsize=(max(8, adata.n_vars * 0.6), max(5, len(order) * 0.4)))
        im = ax.imshow(means, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(adata.n_vars))
        ax.set_xticklabels(adata.var_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_title("Mean CLR ADT Expression by Sample")
        plt.colorbar(im, ax=ax, label="Mean CLR expression", shrink=0.8)
        fig.tight_layout()
        return self._save(fig, "adt_heatmap_by_sample")

    def saturation_curve(self, df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(df["mean_reads_per_cell"], df["median_umis_per_cell"], "o-",
                     color="steelblue", label="UMIs/cell")
        axes[0].plot(df["mean_reads_per_cell"], df["median_adts_detected_per_cell"], "o-",
                     color="coral", label="ADTs detected/cell")
        axes[0].set_xlabel("Mean Reads per Cell"); axes[0].set_ylabel("Median per Cell")
        axes[0].set_title("Complexity vs. Sequencing Depth"); axes[0].legend()

        axes[1].plot(df["mean_reads_per_cell"], df["sequencing_saturation"] * 100, "o-",
                     color="mediumseagreen")
        axes[1].set_xlabel("Mean Reads per Cell"); axes[1].set_ylabel("Sequencing Saturation (%)")
        axes[1].set_title("Sequencing Saturation")
        axes[1].set_ylim(0, 100)
        fig.tight_layout()
        return self._save(fig, "saturation_curve")

    def umi_by_sublibrary(self, adatas, names):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, sub in zip(names, adatas):
            vals = np.log1p(sub.obs["total_counts"])
            ax.hist(vals, bins=40, histtype="step", linewidth=1.5, label=name, density=True)
        ax.set_xlabel("log1p(Total UMI Counts)"); ax.set_ylabel("Density")
        ax.set_title("Total UMI Counts per Cell -- by Sublibrary")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save(fig, "umi_by_sublibrary")
