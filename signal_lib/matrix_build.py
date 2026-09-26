"""Sparse cell x ADT matrix + AnnData assembly.

Cell-barcode strings and well decoding are only ever computed for the small
set of *distinct* surviving cells (never per read) -- this is what keeps the
pilot notebook's per-read string-building cost out of the hot path entirely.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import encoding


def build_raw_anndata(cell_id: np.ndarray, adt_idx: np.ndarray, count: np.ndarray,
                       adt_whitelist, rt_whitelist, ligation_whitelist,
                       stats: dict, reads_per_rt: np.ndarray, sublibrary_name: str) -> ad.AnnData:
    """cell_id/adt_idx/count are the deduplicated (cell,adt)->molecule-count
    triples returned by umi_dedup.collapse_umis."""
    uniq_cell_ids, row_idx = np.unique(cell_id, return_inverse=True)
    row_idx = np.asarray(row_idx).ravel()
    n_cells = len(uniq_cell_ids)
    n_adts = len(adt_whitelist.seqs)

    X = sp.coo_matrix((count.astype(np.float32), (row_idx, adt_idx.astype(np.int64))),
                       shape=(n_cells, n_adts)).tocsr()

    rt_idx, r2_idx, r3_idx = encoding.unpack_cell_id(uniq_cell_ids)
    rt_wells = rt_whitelist.wells[rt_idx]
    r2_wells = ligation_whitelist.wells[r2_idx]
    r3_wells = ligation_whitelist.wells[r3_idx]
    rt_seqs = rt_whitelist.seqs_str[rt_idx]
    r2_seqs = ligation_whitelist.seqs_str[r2_idx]
    r3_seqs = ligation_whitelist.seqs_str[r3_idx]

    obs_names = [f"{sublibrary_name}_{rt}_{r2}_{r3}" for rt, r2, r3 in zip(rt_seqs, r2_seqs, r3_seqs)]
    obs = pd.DataFrame({
        "sublibrary": sublibrary_name,
        "RT_well": rt_wells,
        "R2_well": r2_wells,
        "R3_well": r3_wells,
        "RT_sequence": rt_seqs,
        "R2_sequence": r2_seqs,
        "R3_sequence": r3_seqs,
        "cell_id_packed": uniq_cell_ids,
    }, index=obs_names)

    var = adt_whitelist.meta_df.copy()
    var = var.set_index("Antigen")
    var.index.name = None

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    adata.obs["n_adts_detected"] = np.asarray((X > 0).sum(axis=1)).ravel()
    adata.var["total_counts"] = np.asarray(X.sum(axis=0)).ravel()
    adata.var["n_cells_detected"] = np.asarray((X > 0).sum(axis=0)).ravel()
    adata.var["detection_rate"] = adata.var["n_cells_detected"] / max(n_cells, 1)

    adata.uns["processing_stats"] = stats
    adata.uns["reads_per_rt"] = reads_per_rt
    adata.uns["sublibrary"] = sublibrary_name
    adata.uns["barcode_structure"] = {
        "R1_ADT": "0-15", "R2_UMI": "0-10", "R2_R3_ligation": "10-18",
        "R2_R2_ligation": "30-38", "R2_RT": "50-58",
    }
    return adata
