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


class InsufficientCellsError(ValueError):
    """Raised when too few cells were called for the embedding stage to mean anything.

    Carries a message naming the likely cause, because the underlying failure is
    an opaque solver error about component counts.
    """


# PHATE builds a kNN graph, so it needs meaningfully more cells than neighbours.
# Below this the embedding is noise even when the linear algebra happens to work.
MIN_CELLS_FOR_ANALYSIS = 10


def check_analysis_feasible(adata, analysis_cfg: dict) -> None:
    """Raise InsufficientCellsError if the embedding stage cannot run sensibly.

    Called before the expensive work so a doomed run fails with an explanation
    rather than a solver traceback.
    """
    n_obs, n_vars = adata.n_obs, adata.n_vars
    if n_obs >= MIN_CELLS_FOR_ANALYSIS and min(n_obs, n_vars) > 2:
        return
    raise InsufficientCellsError(
        f"Only {n_obs} cell(s) x {n_vars} ADT(s) passed filtering, too few for "
        f"CLR/PCA/PHATE (needs at least {MIN_CELLS_FOR_ANALYSIS} cells).\n"
        f"  This usually means the cell-calling thresholds are too strict for the "
        f"number of reads processed -- typical with --debug-reads on a small slice.\n"
        f"  Either lower filtering.min_umi_per_cell (or set it to null to use the "
        f"detected knee), or set qc.analysis.enabled: false to write counts and "
        f"core QC only."
    )


def effective_n_comps(adata, analysis_cfg: dict) -> int:
    """PCA components, capped by both matrix dimensions.

    scanpy/sklearn require n_comps < min(n_obs, n_vars); asking for more raises an
    error naming component counts rather than the real problem.
    """
    return max(1, min(int(analysis_cfg["n_pcs"]), adata.n_vars - 1, adata.n_obs - 1))


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
    check_analysis_feasible(adata, analysis_cfg)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].astype(np.float32)

    pt.pp.clr(adata, inplace=True)
    adata.layers["clr"] = adata.X.copy()

    # random_state is passed explicitly rather than left to scanpy's default.
    # The ARPACK solver's result depends on its starting vector, so an implicit
    # default makes the embedding hostage to the installed scanpy/scipy version:
    # runs of identical code months apart were seen to diverge by 3e-5 in
    # X_pca, which amplified to ~0.01 in the PHATE embedding.
    sc.tl.pca(adata, n_comps=effective_n_comps(adata, analysis_cfg),
              svd_solver=str(analysis_cfg["pca_svd_solver"]),
              random_state=int(analysis_cfg["pca_random_state"]))

    phate_kwargs = dict(analysis_cfg.get("phate") or {})
    phate_kwargs["n_jobs"] = threads
    # PHATE's kNN graph cannot ask for more neighbours than there are other cells.
    if "knn" in phate_kwargs:
        phate_kwargs["knn"] = max(2, min(int(phate_kwargs["knn"]), adata.n_obs - 1))
    phate_op = phate.PHATE(**phate_kwargs, verbose=0)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    adata.obsm["X_phate"] = phate_op.fit_transform(X)

    return adata
