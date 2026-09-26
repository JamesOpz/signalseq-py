#!/usr/bin/env python
"""Compare two pipeline runs for exact equivalence.

Built for the port: it proves that `signalseq adt run` reproduces the original
parsev3_pipeline on the same inputs. Counts must be EXACTLY identical after
aligning row and column order; floats in obs/var are compared within a tight
tolerance; the plot data is compared numerically rather than by image bytes,
because PNG encoding is not reproducible.

Usage:
    python scripts/compare_outputs.py --a RUN_A_DIR --b RUN_B_DIR [--name Sublibrary_3]
    python scripts/compare_outputs.py --a A --b B --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

FLOAT_TOL = 1e-6

# uns keys whose value is documentation rather than data: a difference is
# reported but does not fail the comparison.
INFORMATIONAL_UNS = {"barcode_structure"}

# uns keys that the port adds on purpose: the original kept a reads-per-barcode
# array only for the RT round, the port keeps one per declared round. Extra
# rounds are reported, not failed; "reads_per_rt" itself is compared as data.
def _is_additive(key: str, in_a: bool, in_b: bool) -> bool:
    return key.startswith("reads_per_") and not in_a and in_b


class Comparison:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def record(self, name: str, ok: bool, detail: str = "", informational: bool = False) -> bool:
        self.checks.append({
            "check": name, "ok": bool(ok), "detail": detail, "informational": informational,
        })
        return ok

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"] and not c["informational"]]

    @property
    def infos(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"] and c["informational"]]

    def report(self) -> str:
        lines = []
        for c in self.checks:
            if c["ok"]:
                mark = "PASS"
            elif c["informational"]:
                mark = "NOTE"
            else:
                mark = "FAIL"
            lines.append(f"  [{mark}] {c['check']}" + (f" -- {c['detail']}" if c["detail"] else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _find(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def _dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _align(a: ad.AnnData, b: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    """Reorder b to a's obs/var order, so ordering differences don't mask equality."""
    return a, b[a.obs_names, :][:, a.var_names]


def compare_series(sa: pd.Series, sb: pd.Series, label: str, cmp: Comparison) -> None:
    if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
        va = sa.to_numpy(dtype=float)
        vb = sb.to_numpy(dtype=float)
        both_nan = np.isnan(va) & np.isnan(vb)
        close = np.isclose(va, vb, rtol=0, atol=FLOAT_TOL, equal_nan=True)
        ok = bool((close | both_nan).all())
        if ok:
            cmp.record(label, True)
        else:
            bad = int((~(close | both_nan)).sum())
            worst = float(np.nanmax(np.abs(va - vb))) if len(va) else 0.0
            cmp.record(label, False, f"{bad} value(s) differ, max abs diff {worst:.3g}")
    else:
        va = sa.astype(str).to_numpy()
        vb = sb.astype(str).to_numpy()
        ok = bool((va == vb).all())
        bad = int((va != vb).sum())
        examples = ""
        if not ok:
            idx = np.flatnonzero(va != vb)[:3]
            examples = "; e.g. " + ", ".join(f"{va[i]!r} vs {vb[i]!r}" for i in idx)
        cmp.record(label, ok, "" if ok else f"{bad} value(s) differ{examples}")


def compare_frames(fa: pd.DataFrame, fb: pd.DataFrame, what: str, cmp: Comparison) -> None:
    cols_a, cols_b = list(fa.columns), list(fb.columns)
    if set(cols_a) != set(cols_b):
        only_a = sorted(set(cols_a) - set(cols_b))
        only_b = sorted(set(cols_b) - set(cols_a))
        cmp.record(f"{what} columns match", False,
                   f"only in A: {only_a}; only in B: {only_b}")
    else:
        cmp.record(f"{what} columns match", True, f"{len(cols_a)} columns")
        if cols_a != cols_b:
            cmp.record(f"{what} column order matches", False,
                       f"A: {cols_a[:4]}...; B: {cols_b[:4]}...")
    for col in cols_a:
        if col in fb.columns:
            compare_series(fa[col], fb[col], f"{what}['{col}']", cmp)


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def compare_anndata(path_a: Path, path_b: Path, tag: str, cmp: Comparison) -> None:
    a = ad.read_h5ad(path_a)
    b = ad.read_h5ad(path_b)

    # -- barcodes and features ------------------------------------------
    same_obs = set(a.obs_names) == set(b.obs_names)
    cmp.record(f"{tag}: obs_names are the same set", same_obs,
               f"A {a.n_obs:,} / B {b.n_obs:,}" if not same_obs else f"{a.n_obs:,} cells")
    cmp.record(f"{tag}: obs_names count matches", a.n_obs == b.n_obs,
               "" if a.n_obs == b.n_obs else f"{a.n_obs:,} vs {b.n_obs:,}")
    cmp.record(f"{tag}: no duplicate obs_names",
               not a.obs_names.duplicated().any() and not b.obs_names.duplicated().any())

    same_var = set(a.var_names) == set(b.var_names)
    cmp.record(f"{tag}: var_names are the same set", same_var,
               f"{a.n_vars} ADTs" if same_var else
               f"only in A: {sorted(set(a.var_names) - set(b.var_names))}")
    cmp.record(f"{tag}: var_names count matches", a.n_vars == b.n_vars)

    if not (same_obs and same_var):
        cmp.record(f"{tag}: matrices comparable", False,
                   "skipped: obs/var name sets differ")
        return

    a, b = _align(a, b)

    # -- counts: must be EXACT -------------------------------------------
    for layer in (None, "counts"):
        label = "X" if layer is None else f"layers['{layer}']"
        Xa = a.X if layer is None else a.layers.get(layer)
        Xb = b.X if layer is None else b.layers.get(layer)
        if Xa is None or Xb is None:
            cmp.record(f"{tag}: {label} present in both", Xa is None and Xb is None,
                       f"A has it: {Xa is not None}, B has it: {Xb is not None}")
            continue
        da, db = _dense(Xa), _dense(Xb)
        if da.shape != db.shape:
            cmp.record(f"{tag}: {label} shape matches", False, f"{da.shape} vs {db.shape}")
            continue
        exact = bool(np.array_equal(da, db))
        if exact:
            cmp.record(f"{tag}: {label} exactly identical", True,
                       f"{da.shape[0]:,}x{da.shape[1]}, sum {da.sum():,.0f}")
        else:
            diff = np.abs(da - db)
            n_diff = int((diff > 0).sum())
            cmp.record(f"{tag}: {label} exactly identical", False,
                       f"{n_diff:,} cell(s) differ, max abs diff {diff.max():.6g}, "
                       f"sums {da.sum():,.0f} vs {db.sum():,.0f}")

    # CLR layer is a float transform: compare within tolerance.
    if "clr" in a.layers and "clr" in b.layers:
        da, db = _dense(a.layers["clr"]), _dense(b.layers["clr"])
        ok = bool(np.allclose(da, db, rtol=0, atol=FLOAT_TOL, equal_nan=True))
        cmp.record(f"{tag}: layers['clr'] within {FLOAT_TOL:g}", ok,
                   "" if ok else f"max abs diff {np.nanmax(np.abs(da - db)):.3g}")

    # -- obs / var --------------------------------------------------------
    compare_frames(a.obs, b.obs, f"{tag}: obs", cmp)
    compare_frames(a.var, b.var, f"{tag}: var", cmp)

    # -- embeddings: sign-insensitive -------------------------------------
    for key in sorted(set(a.obsm) | set(b.obsm)):
        if key not in a.obsm or key not in b.obsm:
            cmp.record(f"{tag}: obsm['{key}'] in both", False,
                       f"A: {key in a.obsm}, B: {key in b.obsm}")
            continue
        ea, eb = np.asarray(a.obsm[key]), np.asarray(b.obsm[key])
        if ea.shape != eb.shape:
            cmp.record(f"{tag}: obsm['{key}'] shape", False, f"{ea.shape} vs {eb.shape}")
            continue
        # PCA/PHATE components are only defined up to a sign flip per component.
        same = np.allclose(ea, eb, rtol=0, atol=1e-4)
        flipped = all(
            np.allclose(ea[:, i], eb[:, i], rtol=0, atol=1e-4)
            or np.allclose(ea[:, i], -eb[:, i], rtol=0, atol=1e-4)
            for i in range(ea.shape[1])
        )
        if same:
            cmp.record(f"{tag}: obsm['{key}'] identical", True)
        elif flipped:
            cmp.record(f"{tag}: obsm['{key}'] identical up to component sign", True,
                       "sign flips are not meaningful")
        else:
            cmp.record(f"{tag}: obsm['{key}'] identical", False,
                       f"max abs diff {np.nanmax(np.abs(ea - eb)):.3g}")

    # -- uns: read-assignment summaries -----------------------------------
    compare_uns(a.uns, b.uns, tag, cmp)


def compare_uns(ua, ub, tag: str, cmp: Comparison) -> None:
    keys = sorted(set(ua) | set(ub))
    for key in keys:
        if key not in ua or key not in ub:
            additive = _is_additive(key, key in ua, key in ub)
            cmp.record(f"{tag}: uns['{key}'] in both", False,
                       f"A: {key in ua}, B: {key in ub}"
                       + (" (added by the port: an extra barcode round)" if additive else ""),
                       informational=(key in INFORMATIONAL_UNS) or additive)
            continue
        va, vb = ua[key], ub[key]
        informational = key in INFORMATIONAL_UNS
        if isinstance(va, dict) and isinstance(vb, dict):
            if set(va) != set(vb):
                cmp.record(f"{tag}: uns['{key}'] keys match", False,
                           f"only in A: {sorted(set(va) - set(vb))}; "
                           f"only in B: {sorted(set(vb) - set(va))}",
                           informational=informational)
                continue
            mismatched = {k: (va[k], vb[k]) for k in va if not _scalar_eq(va[k], vb[k])}
            cmp.record(f"{tag}: uns['{key}'] values match", not mismatched,
                       "" if not mismatched else f"{mismatched}", informational=informational)
        elif isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
            ok = va.shape == vb.shape and bool(np.array_equal(va, vb))
            cmp.record(f"{tag}: uns['{key}'] identical", ok,
                       "" if ok else f"shapes {va.shape} vs {vb.shape}",
                       informational=informational)
        else:
            cmp.record(f"{tag}: uns['{key}'] identical", _scalar_eq(va, vb),
                       "" if _scalar_eq(va, vb) else f"{va!r} vs {vb!r}",
                       informational=informational)


def _scalar_eq(x, y) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)) and \
       isinstance(y, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(x), float(y), rtol=0, atol=FLOAT_TOL, equal_nan=True))
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return bool(np.array_equal(x, y))
    return x == y


def compare_qc_stats(path_a: Path, path_b: Path, cmp: Comparison) -> None:
    """The read-assignment summary: total, per-stage and unassigned counts."""
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())

    for section in ("processing_stats", "matrix_stats", "knee", "well_stats",
                    "cells_per_condition"):
        sa, sb = a.get(section), b.get(section)
        if sa is None or sb is None:
            cmp.record(f"qc_stats.{section} present in both", False,
                       f"A: {sa is not None}, B: {sb is not None}")
            continue
        if set(sa) != set(sb):
            cmp.record(f"qc_stats.{section} keys match", False,
                       f"only in A: {sorted(set(sa) - set(sb))}; "
                       f"only in B: {sorted(set(sb) - set(sa))}")
            continue
        cmp.record(f"qc_stats.{section} keys match", True, f"{len(sa)} keys")
        for key in sa:
            cmp.record(f"qc_stats.{section}['{key}']", _scalar_eq(sa[key], sb[key]),
                       "" if _scalar_eq(sa[key], sb[key]) else f"{sa[key]!r} vs {sb[key]!r}")

    # Derived read-assignment consistency, stated explicitly.
    ps_a, ps_b = a.get("processing_stats", {}), b.get("processing_stats", {})
    for label, key in (("total reads", "total_reads"),
                       ("reads with a valid ADT tag", "valid_adt"),
                       ("reads fully assigned", "valid_all"),
                       ("reads with an exact barcode match", "bc_exact_triple")):
        if key in ps_a and key in ps_b:
            cmp.record(f"read assignment: {label}", ps_a[key] == ps_b[key],
                       f"{ps_a[key]:,}" if ps_a[key] == ps_b[key]
                       else f"{ps_a[key]:,} vs {ps_b[key]:,}")
    if "total_reads" in ps_a and "valid_all" in ps_a:
        ua = ps_a["total_reads"] - ps_a["valid_all"]
        ub = ps_b["total_reads"] - ps_b["valid_all"]
        cmp.record("read assignment: unassigned/failed reads", ua == ub,
                   f"{ua:,}" if ua == ub else f"{ua:,} vs {ub:,}")
    for key in ("overall_valid_rate",):
        if key in a and key in b:
            cmp.record(f"qc_stats.{key}", _scalar_eq(a[key], b[key]),
                       "" if _scalar_eq(a[key], b[key]) else f"{a[key]} vs {b[key]}")


def compare_saturation(path_a: Path, path_b: Path, cmp: Comparison) -> None:
    """The saturation curve is the numeric content of the saturation plot."""
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    if list(a.columns) != list(b.columns):
        cmp.record("saturation columns match", False, f"{list(a.columns)} vs {list(b.columns)}")
        return
    cmp.record("saturation columns match", True, f"{len(a.columns)} columns")
    if len(a) != len(b):
        cmp.record("saturation row count matches", False, f"{len(a)} vs {len(b)}")
        return
    cmp.record("saturation row count matches", True, f"{len(a)} depths")
    for col in a.columns:
        compare_series(a[col], b[col], f"saturation['{col}']", cmp)


def compare_plot_data(dir_a: Path, dir_b: Path, name: str, cmp: Comparison) -> None:
    """Judge the figures by the numbers behind them, not the image bytes.

    Every plot is a view of data already compared above: the funnel comes from
    processing_stats, the knee from the knee block, the well heatmaps from
    obs value_counts, the per-ADT panels from X and var, the saturation panel
    from the saturation table. What this adds is (a) the same PNGs exist in both
    runs and (b) the plate/well counts behind the heatmaps agree.
    """
    figs_a = {p.name for p in (dir_a / "figures").glob("*.png")}
    figs_b = {p.name for p in (dir_b / "figures").glob("*.png")}
    if figs_a == figs_b:
        cmp.record("figure set matches", True, f"{len(figs_a)} figures")
    else:
        cmp.record("figure set matches", False,
                   f"only in A: {sorted(figs_a - figs_b)}; only in B: {sorted(figs_b - figs_a)}")

    raw_a = _find(dir_a, f"{name}_raw.h5ad")
    raw_b = _find(dir_b, f"{name}_raw.h5ad")
    if not (raw_a and raw_b):
        return
    a, b = ad.read_h5ad(raw_a), ad.read_h5ad(raw_b)
    well_cols = [c for c in a.obs.columns if c.endswith("_well")]
    for col in well_cols:
        if col not in b.obs.columns:
            cmp.record(f"plate data: {col}", False, "column missing in B")
            continue
        ca = a.obs[col].value_counts().sort_index()
        cb = b.obs[col].value_counts().sort_index()
        ok = ca.index.equals(cb.index) and bool((ca.values == cb.values).all())
        cmp.record(f"plate data behind heatmap: {col}", ok,
                   f"{len(ca)} wells" if ok else "well counts differ")

    for key in sorted(k for k in a.uns if k.startswith("reads_per_")):
        if key in b.uns:
            ok = bool(np.array_equal(np.asarray(a.uns[key]), np.asarray(b.uns[key])))
            cmp.record(f"plot data behind reads-per-barcode: uns['{key}']", ok)


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, type=Path, help="first run directory (e.g. the original)")
    p.add_argument("--b", required=True, type=Path, help="second run directory (e.g. the port)")
    p.add_argument("--name", default=None,
                   help="sublibrary name; inferred from *_raw.h5ad if omitted")
    p.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    args = p.parse_args()

    dir_a, dir_b = args.a.resolve(), args.b.resolve()
    for d in (dir_a, dir_b):
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            return 2

    name = args.name
    if name is None:
        guess = _find(dir_a, "*_raw.h5ad") or _find(dir_a, "*_filtered.h5ad")
        if guess is None:
            print(f"No *_raw.h5ad or *_filtered.h5ad in {dir_a}", file=sys.stderr)
            return 2
        name = guess.name.rsplit("_", 1)[0]

    print(f"Comparing runs for {name!r}\n  A: {dir_a}\n  B: {dir_b}\n")
    cmp = Comparison()

    for tag, pattern in (("raw", f"{name}_raw.h5ad"), ("filtered", f"{name}_filtered.h5ad")):
        pa, pb = _find(dir_a, pattern), _find(dir_b, pattern)
        if pa and pb:
            compare_anndata(pa, pb, tag, cmp)
        else:
            cmp.record(f"{tag}: h5ad present in both", False,
                       f"A: {pa is not None}, B: {pb is not None}")

    qa, qb = _find(dir_a, f"{name}_qc_stats.json"), _find(dir_b, f"{name}_qc_stats.json")
    if qa and qb:
        compare_qc_stats(qa, qb, cmp)
    else:
        cmp.record("qc_stats present in both", False, f"A: {qa is not None}, B: {qb is not None}")

    sa, sb = _find(dir_a, f"{name}_saturation.csv"), _find(dir_b, f"{name}_saturation.csv")
    if sa and sb:
        compare_saturation(sa, sb, cmp)

    compare_plot_data(dir_a, dir_b, name, cmp)

    print(cmp.report())
    n_fail = len(cmp.failures)
    n_info = len(cmp.infos)
    total = len(cmp.checks)
    print(f"\n{total - n_fail - n_info}/{total} checks passed"
          + (f", {n_info} informational difference(s)" if n_info else ""))

    if args.json:
        args.json.write_text(json.dumps({
            "name": name, "a": str(dir_a), "b": str(dir_b),
            "n_checks": total, "n_failures": n_fail, "checks": cmp.checks,
        }, indent=2))
        print(f"Full report -> {args.json}")

    if n_fail:
        print(f"\nDIFFERENT -- {n_fail} check(s) failed:")
        for c in cmp.failures:
            print(f"  {c['check']}: {c['detail']}")
        return 1
    print("\nIDENTICAL -- the two runs agree on every compared quantity.")
    if n_info:
        print("Informational differences (documentation metadata, not data):")
        for c in cmp.infos:
            print(f"  {c['check']}: {c['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
