"""CLR normalization, PCA and PHATE.

Shared by the per-sublibrary and the aggregated stages -- the same function runs
in both, so a combined object is normalized exactly the way its parts were.

muon and phate are imported lazily: a counts-only run (`qc.analysis.enabled:
false`) should not require them to be installed.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc


class AnalysisDependencyError(ImportError):
    """Raised when the optional analysis extras are missing."""


def _import_optional():
    try:
        import phate
        from muon import prot as pt
    except ImportError as exc:
        raise AnalysisDependencyError(
            f"qc.analysis.enabled is true, which needs the optional analysis extras "
            f"(muon for CLR, phate for the embedding), but importing them failed: {exc}. "
            f"Install them with `pip install -e '.[analysis]'`, or set "
            f"qc.analysis.enabled: false to write counts and core QC only."
        ) from exc
    return phate, pt


def run_clr_pca_phate(adata, analysis_cfg: dict, threads: int = -1):
    """CLR-normalize .X (raw counts preserved in .layers['counts']), then PCA, then PHATE."""
    phate, pt = _import_optional()

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].astype(np.float32)

    pt.pp.clr(adata, inplace=True)
    adata.layers["clr"] = adata.X.copy()

    n_comps = min(adata.n_vars - 1, int(analysis_cfg["n_pcs"]))
    # random_state is passed explicitly rather than left to scanpy's default.
    # The ARPACK solver's result depends on its starting vector, so an implicit
    # default makes the embedding hostage to the installed scanpy/scipy version:
    # runs of identical code months apart were seen to diverge by 3e-5 in
    # X_pca, which amplified to ~0.01 in the PHATE embedding.
    sc.tl.pca(adata, n_comps=n_comps, svd_solver=str(analysis_cfg["pca_svd_solver"]),
              random_state=int(analysis_cfg["pca_random_state"]))

    phate_kwargs = dict(analysis_cfg.get("phate") or {})
    phate_kwargs["n_jobs"] = threads
    phate_op = phate.PHATE(**phate_kwargs, verbose=0)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    adata.obsm["X_phate"] = phate_op.fit_transform(X)

    return adata
