"""CLR normalization + PCA + PHATE, shared by both the per-sublibrary and the
aggregated-combined analysis stage (same function, called twice)."""

from __future__ import annotations

import numpy as np
import scanpy as sc

from . import config


def run_clr_pca_phate(adata, n_pcs: int = config.N_PCS):
    """Mutates and returns adata: CLR-normalizes .X (raw counts preserved in
    .layers['counts']), runs PCA, then PHATE on the CLR values."""
    import phate
    from muon import prot as pt

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].astype(np.float32)

    pt.pp.clr(adata, inplace=True)
    adata.layers["clr"] = adata.X.copy()

    n_comps = min(adata.n_vars - 1, n_pcs)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")

    phate_op = phate.PHATE(**config.PHATE_KWARGS, verbose=0)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    adata.obsm["X_phate"] = phate_op.fit_transform(X)

    return adata
