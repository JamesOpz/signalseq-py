"""Downsampling / sequencing-saturation curve (CellRanger/10x style): fixed
called-cell set, subsample raw (pre-collapse) reads at increasing fractions,
recompute per-cell complexity at each depth. Reuses the per-read arrays already
held in memory right after the hot loop -- no extra FASTQ pass needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .umi_dedup import collapse_umis


def compute_saturation_curve(cell_id_arr: np.ndarray, adt_idx_arr: np.ndarray, umi_packed_arr: np.ndarray,
                              called_cell_ids: np.ndarray,
                              fractions=config.SATURATION_FRACTIONS,
                              seed: int = config.SATURATION_SEED) -> pd.DataFrame:
    """called_cell_ids: packed cell ids of the knee-called cells (fixed cell set
    across all subsampling depths, from the full-depth call).

    Returns a DataFrame with one row per fraction: fraction, n_reads,
    mean_reads_per_cell, median_umis_per_cell, median_adts_detected_per_cell,
    sequencing_saturation.
    """
    called_set = np.isin(cell_id_arr, called_cell_ids)
    cell_id_c = cell_id_arr[called_set]
    adt_idx_c = adt_idx_arr[called_set]
    umi_c = umi_packed_arr[called_set]
    n_reads_total = len(cell_id_c)
    n_called_cells = len(called_cell_ids)
    sorted_called_ids = np.sort(called_cell_ids)

    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        n_sub = int(round(frac * n_reads_total))
        if n_sub == 0:
            continue
        if frac >= 1.0:
            sub_idx = np.arange(n_reads_total)
        else:
            sub_idx = rng.choice(n_reads_total, size=n_sub, replace=False)

        cell_sub, adt_sub, count_sub = collapse_umis(
            cell_id_c[sub_idx], adt_idx_c[sub_idx], umi_c[sub_idx]
        )

        pos = np.searchsorted(sorted_called_ids, cell_sub)
        per_cell_umi = np.zeros(n_called_cells, dtype=np.int64)
        per_cell_adt = np.zeros(n_called_cells, dtype=np.int64)
        np.add.at(per_cell_umi, pos, count_sub)
        np.add.at(per_cell_adt, pos, 1)  # one distinct ADT detected per row

        n_unique_molecules = int(count_sub.sum())
        saturation = 1.0 - (n_unique_molecules / n_sub) if n_sub > 0 else np.nan

        rows.append({
            "fraction": frac,
            "n_reads": n_sub,
            "mean_reads_per_cell": n_sub / max(n_called_cells, 1),
            "median_umis_per_cell": float(np.median(per_cell_umi)),
            "median_adts_detected_per_cell": float(np.median(per_cell_adt)),
            "sequencing_saturation": saturation,
        })

    return pd.DataFrame(rows)
