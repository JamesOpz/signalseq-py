"""Sparse cell x ADT matrix and AnnData assembly.

Barcode strings and well decoding are computed only for the small set of
*distinct* surviving cells, never per read -- that is what keeps string building
out of the hot path entirely.

Column names in `.obs` come from each element's `obs_label`, so the ported Parse
v3 config reproduces the original `RT_well` / `R2_well` / `R3_well` naming
without any of it being hardcoded here.
"""

from __future__ import annotations

from typing import Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .encoding import Encoder
from .readstructure import ReadStructure
from .whitelists import AdtPanel, BarcodeWhitelist


def build_raw_anndata(cell_id: np.ndarray, adt_idx: np.ndarray, count: np.ndarray,
                      read_structure: ReadStructure,
                      panel: AdtPanel,
                      whitelists: Mapping[str, BarcodeWhitelist],
                      encoder: Encoder,
                      stats: dict,
                      reads_per_round: Mapping[str, np.ndarray],
                      sublibrary_name: str) -> ad.AnnData:
    """cell_id / adt_idx / count are the deduplicated (cell, ADT) -> molecule
    triples returned by umi_dedup.collapse_umis."""
    uniq_cell_ids, row_idx = np.unique(cell_id, return_inverse=True)
    row_idx = np.asarray(row_idx).ravel()
    n_cells = len(uniq_cell_ids)
    n_adts = panel.n

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
    for element, idxs in zip(rounds, round_idxs):
        wl = whitelists[element.name]
        label = element.label
        obs_data[f"{label}_well"] = wl.wells[idxs]
        seq_by_label[label] = wl.seqs_str[idxs]
    for element in rounds:
        label = element.label
        obs_data[f"{label}_sequence"] = seq_by_label[label]
    obs_data["cell_id_packed"] = uniq_cell_ids

    obs_names = _build_obs_names(read_structure, sublibrary_name, seq_by_label, n_cells)
    obs = pd.DataFrame(obs_data, index=obs_names)

    var = panel.meta_df.copy()
    var = var.set_index(panel.name_column)
    var.index.name = None

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
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


def _build_obs_names(read_structure: ReadStructure, sublibrary_name: str,
                     seq_by_label: Mapping[str, np.ndarray], n_cells: int) -> list[str]:
    """Render `cell_barcode.name_template` once per distinct cell.

    The template may reference `{sublibrary}` and any barcode round, by element
    name or by obs_label.
    """
    template = read_structure.name_template
    rounds = read_structure.rounds
    columns = {}
    for element in rounds:
        seqs = seq_by_label[element.label]
        columns[element.name] = seqs
        columns[element.label] = seqs

    names = []
    for i in range(n_cells):
        fields = {key: arr[i] for key, arr in columns.items()}
        fields["sublibrary"] = sublibrary_name
        names.append(template.format(**fields))
    return names
