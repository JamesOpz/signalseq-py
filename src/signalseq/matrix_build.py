"""Sparse cell x ADT matrix and AnnData assembly.

Barcode strings and well decoding are computed only for the small set of
*distinct* surviving cells, never per read -- that is what keeps string building
out of the hot path entirely.

Column names in `.obs` come from each element's `obs_label`, so the Parse v3
config's `bc1_well` / `bc2_well` / `bc3_well` naming (split-pipe's) is not
hardcoded here.
"""

from __future__ import annotations

from typing import Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .encoding import Encoder
from .readstructure import ReadStructure
from .well_merge import WellMerge
from .whitelists import AdtPanel, BarcodeWhitelist


def build_raw_anndata(cell_id: np.ndarray, adt_idx: np.ndarray, count: np.ndarray,
                      reads: np.ndarray,
                      read_structure: ReadStructure,
                      panel: AdtPanel,
                      whitelists: Mapping[str, BarcodeWhitelist],
                      encoder: Encoder,
                      stats: dict,
                      reads_per_round: Mapping[str, np.ndarray],
                      sublibrary_name: str,
                      parse_index: int | None = None,
                      merge: WellMerge | None = None) -> ad.AnnData:
    """cell_id / adt_idx / count / reads are the deduplicated (cell, ADT) rows
    returned by umi_dedup.collapse_umis(..., return_reads=True).

    With `merge`, rows from barcodes sharing a well are summed into one cell,
    and each primer type's molecule and read totals are kept in obs.
    """
    type_code = None
    if merge is not None:
        if merge.type_codes is not None:
            type_code = merge.type_of(cell_id)
        cell_id = merge.merge(cell_id)

    uniq_cell_ids, row_idx = np.unique(cell_id, return_inverse=True)
    row_idx = np.asarray(row_idx).ravel()
    n_cells = len(uniq_cell_ids)
    n_adts = panel.n

    # Duplicate (cell, ADT) entries -- one per merged primer type -- are summed
    # by the CSR conversion.
    X = sp.coo_matrix(
        (count.astype(np.float32), (row_idx, adt_idx.astype(np.int64))),
        shape=(n_cells, n_adts),
    ).tocsr()

    # Decode each barcode round for the distinct cells only.
    round_idxs = encoder.unpack_cell_id(uniq_cell_ids)
    rounds = read_structure.rounds

    # Scalar, broadcast by pandas against the index -- as in the original.
    obs_data: dict = {"sublibrary": sublibrary_name}
    seq_by_label: dict[str, np.ndarray] = {}
    template_fields: dict[str, np.ndarray] = {}
    for element, idxs in zip(rounds, round_idxs):
        wl = whitelists[element.name]
        label = element.label
        obs_data[f"{label}_well"] = wl.wells[idxs]
        obs_data[f"{label}_wind"] = wl.winds[idxs]
        seq_by_label[label] = wl.seqs_str[idxs]
        for key in {element.name, label}:
            template_fields[key] = wl.seqs_str[idxs]
            template_fields[f"{key}_well"] = wl.wells[idxs]
            template_fields[f"{key}_wind"] = wl.winds[idxs]
    for element in rounds:
        label = element.label
        obs_data[f"{label}_sequence"] = seq_by_label[label]
    obs_data["cell_id_packed"] = uniq_cell_ids

    obs_names = _build_obs_names(read_structure.name_template, template_fields, n_cells,
                                 sublibrary=sublibrary_name, parse_index=parse_index)
    obs = pd.DataFrame(obs_data, index=obs_names)

    var = panel.meta_df.copy()
    var = var.set_index(panel.name_column)
    var.index.name = None

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    if type_code is not None:
        for k, type_label in enumerate(merge.type_labels):
            adata.obs[f"total_counts_{type_label}"] = np.bincount(
                row_idx, weights=count * (type_code == k), minlength=n_cells).astype(np.float32)
    adata.obs["total_reads"] = np.bincount(row_idx, weights=reads, minlength=n_cells).astype(np.int64)
    if type_code is not None:
        for k, type_label in enumerate(merge.type_labels):
            adata.obs[f"total_reads_{type_label}"] = np.bincount(
                row_idx, weights=reads * (type_code == k), minlength=n_cells).astype(np.int64)
    adata.obs["n_adts_detected"] = np.asarray((X > 0).sum(axis=1)).ravel()
    adata.var["total_counts"] = np.asarray(X.sum(axis=0)).ravel()
    adata.var["n_cells_detected"] = np.asarray((X > 0).sum(axis=0)).ravel()
    adata.var["detection_rate"] = adata.var["n_cells_detected"] / max(n_cells, 1)

    adata.uns["processing_stats"] = stats
    # Keyed by ELEMENT NAME, not obs_label: the original stored one array under
    # "reads_per_rt", and the Parse v3 config names that round `rt`, so the key
    # matches. Rounds beyond the first are additional -- the original kept only
    # the RT round.
    for name, arr in reads_per_round.items():
        adata.uns[f"reads_per_{name}"] = arr
    adata.uns["sublibrary"] = sublibrary_name
    adata.uns["barcode_structure"] = {
        f"{el.read}_{el.label}": f"{el.start0}-{el.end0}"
        for el in read_structure.elements.values()
    }
    return adata


def _build_obs_names(template: str, columns: Mapping[str, np.ndarray], n_cells: int,
                     **scalars) -> list[str]:
    """Render `cell_barcode.name_template` once per distinct cell.

    The template may reference `{sublibrary}`, `{parse_index}` and any barcode
    round, by element name or by obs_label: `{rt}` is its sequence, `{rt_well}`
    its well and `{rt_wind}` split-pipe's zero-padded well index.
    """
    names = []
    for i in range(n_cells):
        fields = {key: arr[i] for key, arr in columns.items()}
        fields.update(scalars)
        names.append(template.format(**fields))
    return names
