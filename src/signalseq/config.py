"""Load, resolve and validate the run configuration.

Everything the pipeline needs comes from here: read offsets and lengths,
whitelist and panel paths, mismatch tolerances, filter thresholds, plot
geometry. No chemistry-specific value is hardcoded in the package -- the
defaults in this module exist only so a config can omit a section and still
reproduce the Parse v3 behaviour it was ported from.

Validation is deliberately eager and specific: `signalseq adt validate` should
surface a missing file, an off-by-one offset or a mismatched barcode length
before a multi-hour run starts, and say which config key is at fault.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .readstructure import ConfigError, ReadStructure

SCHEMA_VERSION = 1

# Defaults reproducing the original parsev3_pipeline. A config may override any
# of these; none of them encode a read position or length, which must always be
# stated explicitly under `read_structure`.
DEFAULTS: dict[str, Any] = {
    "run": {"threads": -1, "overwrite": False},
    "matching": {
        "adt": {"max_mismatches": 1, "drop_ambiguous": True},
        "cell_barcode": {
            "max_edit_distance": 2,
            "census": {
                "offset_reads": 3_000_000,
                "n_reads": 1_000_000,
                "perfect_cell_threshold": 0.92,
                "min_count_threshold": 5,
                "accept": "strictly_greater",
            },
        },
        "umi": {"method": "directional", "max_mismatches": 1, "directional_count_ratio": 2},
    },
    "filtering": {
        "min_umi_per_cell": None,
        "max_umi_per_cell": None,
        "min_adts_detected": 2,
        "knee": {"exclude_top_n": 3},
    },
    "qc": {
        "saturation": {
            "enabled": True,
            "fractions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "seed": 0,
        },
        "analysis": {
            "enabled": True,
            "n_pcs": 10,
            "pca_svd_solver": "arpack",
            "phate": {"n_components": 2, "knn": 15, "decay": 40, "t": "auto", "random_state": 42},
        },
        "plots": {
            "enabled": True,
            "dpi": 150,
            "style": "seaborn-v0_8-whitegrid",
            "plate_rows": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "plate_cols": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "grid_columns": 4,
            "color_percentiles": [5, 95],
            # One colormap per cell-barcode round, cycled if there are more rounds.
            "well_heatmap_cmaps": ["Blues", "Greens", "Oranges"],
            # Colour the reads-per-round bars by the whitelist's sample-type column.
            "stype_colors": {},
            # Category order and palette for the per-condition composition plots.
            "treatment_order": [],
            "culture_colors": {},
            "unassigned_label": "Unassigned",
        },
    },
    "sample_annotation": {
        "culture_map": {},
        "treatment": {"control_token": "DMSO", "treated_label": "Treated"},
        "unassigned_label": "Unassigned",
        "condition_template": "{culture_type}_{timepoint_hrs}hr_{treatment}",
    },
    "outputs": {
        "raw_anndata": True,
        "filtered_anndata": True,
        "saturation_table": True,
        "qc_stats": True,
        "figures": True,
        "aggregate": {
            "combined_anndata": True,
            "combined_qc_stats": True,
            "copy_best_saturation": True,
        },
    },
    "performance": {"chunk_rows": 25_000_000, "gzip_binaries": ["libdeflate-gzip", "gzip"]},
    # Set true only by the post-verification bugfix pass; see compatibility.py
    # for exactly what each flag changes.
    "compatibility": {
        "fix_valid_cell_barcode_counter": False,
        "fix_missing_timepoint_label": False,
    },
}

_VALID_UMI_METHODS = ("directional", "exact")
_VALID_CENSUS_ACCEPT = ("strictly_greater", "greater_or_equal")


# --------------------------------------------------------------------------
# Small typed views over the config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Sublibrary:
    name: str
    r1: Path
    r2: Path

    def read_path(self, read: str) -> Path:
        return self.r1 if read.upper() == "R1" else self.r2


@dataclass(frozen=True)
class WhitelistSpec:
    """One named barcode whitelist.

    Parse v3 reuses a single 96-well ligation plate for both ligation rounds,
    but each round still declares its own entry here, so a chemistry with
    distinct plates per round needs no code change. Loading is cached on
    (csv, edit_distance_json), so repeating the files costs nothing.
    """

    name: str
    csv: Path
    sequence_column: str
    well_column: str | None
    stype_column: str | None
    edit_distance_json: Path | None

    @property
    def cache_key(self) -> tuple:
        return (str(self.csv), str(self.edit_distance_json))


@dataclass(frozen=True)
class PanelSpec:
    csv: Path
    barcode_column: str
    name_column: str


@dataclass(frozen=True)
class LayoutSpec:
    csv: Path
    well_column: str
    sample_column: str


@dataclass
class Config:
    """A validated configuration. `raw` keeps the merged mapping for provenance."""

    path: Path
    raw: dict
    read_structure: ReadStructure
    sublibraries: tuple[Sublibrary, ...]
    whitelists: Mapping[str, WhitelistSpec]
    panel: PanelSpec
    layout: LayoutSpec
    output_dir: Path
    warnings: list[str] = field(default_factory=list)

    # -- convenience accessors --------------------------------------------
    @property
    def name(self) -> str:
        return str(self.raw["run"]["name"])

    @property
    def threads(self) -> int:
        return int(self.raw["run"]["threads"])

    @property
    def overwrite(self) -> bool:
        return bool(self.raw["run"]["overwrite"])

    @property
    def matching(self) -> dict:
        return self.raw["matching"]

    @property
    def filtering(self) -> dict:
        return self.raw["filtering"]

    @property
    def qc(self) -> dict:
        return self.raw["qc"]

    @property
    def outputs(self) -> dict:
        return self.raw["outputs"]

    @property
    def performance(self) -> dict:
        return self.raw["performance"]

    @property
    def sample_annotation(self) -> dict:
        return self.raw["sample_annotation"]

    @property
    def compatibility(self) -> dict:
        return self.raw["compatibility"]

    def sublibrary(self, name: str) -> Sublibrary:
        for s in self.sublibraries:
            if s.name == name:
                return s
        raise ConfigError(
            f"No sublibrary named {name!r} in the config; available: "
            f"{[s.name for s in self.sublibraries]}"
        )

    def sublibrary_output_dir(self, name: str) -> Path:
        return self.output_dir / name

    @property
    def combined_output_dir(self) -> Path:
        return self.output_dir / "combined"


# --------------------------------------------------------------------------
# Merge + parse
# --------------------------------------------------------------------------

def _deep_merge(base: Mapping, override: Mapping) -> dict:
    out = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path_like: Any, base: Path, what: str) -> Path:
    """Resolve a config path relative to the config file's own directory.

    Relative paths in a config should mean "next to the config", not "wherever
    the shell happens to be", so a config plus its reference files can be moved
    around as a unit.
    """
    if path_like is None:
        raise ConfigError(f"{what}: required, but missing or null")
    p = Path(os.path.expanduser(str(path_like)))
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _require(m: Mapping, key: str, what: str) -> Any:
    if key not in m or m[key] is None:
        raise ConfigError(f"{what}.{key}: required, but missing or null")
    return m[key]


def load_config(path: str | Path) -> Config:
    """Read a YAML config, merge it onto the defaults and structurally validate it.

    Raises ConfigError on anything malformed. This does NOT touch input files
    beyond resolving their paths -- see validate.py for the checks that read
    from disk.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    with open(path) as fh:
        try:
            user = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: not valid YAML -- {exc}") from exc
    if not isinstance(user, Mapping):
        raise ConfigError(
            f"{path}: top level must be a mapping of sections, got {type(user).__name__}"
        )

    declared = user.get("schema_version", SCHEMA_VERSION)
    if declared != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version: this build understands version {SCHEMA_VERSION}, "
            f"config declares {declared!r}"
        )

    raw = _deep_merge(DEFAULTS, user)
    base = path.parent
    warnings: list[str] = []

    # -- run ---------------------------------------------------------------
    run = raw.get("run") or {}
    if not run.get("name"):
        raise ConfigError("run.name: required (used to label the run and its provenance)")
    output_dir = _resolve(_require(run, "output_dir", "run"), base, "run.output_dir")
    threads = run.get("threads", -1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads == 0 or threads < -1:
        raise ConfigError(
            f"run.threads: expected -1 (all cores) or a positive integer, got {threads!r}"
        )

    # -- read structure ----------------------------------------------------
    rs_raw = raw.get("read_structure")
    if not rs_raw:
        raise ConfigError("read_structure: required section (see configs/parse_v3.example.yaml)")
    read_structure = ReadStructure.from_mapping(rs_raw)

    # -- inputs ------------------------------------------------------------
    inputs = raw.get("inputs") or {}
    if not inputs:
        raise ConfigError("inputs: required section")

    subs_raw = inputs.get("sublibraries") or []
    if not subs_raw:
        raise ConfigError("inputs.sublibraries: at least one sublibrary is required")
    sublibraries = []
    seen: set[str] = set()
    for i, s in enumerate(subs_raw):
        what = f"inputs.sublibraries[{i}]"
        if not isinstance(s, Mapping):
            raise ConfigError(f"{what}: expected a mapping with name/r1/r2")
        sname = str(_require(s, "name", what))
        if sname in seen:
            raise ConfigError(f"{what}.name: duplicate sublibrary name {sname!r}")
        seen.add(sname)
        sublibraries.append(Sublibrary(
            name=sname,
            r1=_resolve(_require(s, "r1", what), base, f"{what}.r1"),
            r2=_resolve(_require(s, "r2", what), base, f"{what}.r2"),
        ))

    wl_raw = inputs.get("whitelists") or {}
    if not wl_raw:
        raise ConfigError("inputs.whitelists: at least one named whitelist is required")
    whitelists: dict[str, WhitelistSpec] = {}
    for wname, w in wl_raw.items():
        what = f"inputs.whitelists.{wname}"
        if not isinstance(w, Mapping):
            raise ConfigError(f"{what}: expected a mapping")
        ed = w.get("edit_distance_json")
        whitelists[str(wname)] = WhitelistSpec(
            name=str(wname),
            csv=_resolve(_require(w, "csv", what), base, f"{what}.csv"),
            sequence_column=str(w.get("sequence_column", "sequence")),
            well_column=None if w.get("well_column") is None else str(w["well_column"]),
            stype_column=None if w.get("stype_column") is None else str(w["stype_column"]),
            edit_distance_json=(
                None if ed is None else _resolve(ed, base, f"{what}.edit_distance_json")
            ),
        )

    panel_raw = inputs.get("adt_panel") or {}
    if not panel_raw:
        raise ConfigError("inputs.adt_panel: required")
    panel = PanelSpec(
        csv=_resolve(_require(panel_raw, "csv", "inputs.adt_panel"), base, "inputs.adt_panel.csv"),
        barcode_column=str(panel_raw.get("barcode_column", "Barcode")),
        name_column=str(panel_raw.get("name_column", "Antigen")),
    )

    layout_raw = inputs.get("rt_layout") or {}
    if not layout_raw:
        raise ConfigError("inputs.rt_layout: required")
    layout = LayoutSpec(
        csv=_resolve(_require(layout_raw, "csv", "inputs.rt_layout"), base, "inputs.rt_layout.csv"),
        well_column=str(layout_raw.get("well_column", "Well_ID")),
        sample_column=str(layout_raw.get("sample_column", "Sample_name")),
    )

    # -- cross-references between sections ---------------------------------
    known = set(whitelists) | {"adt_panel"}
    for el in read_structure.elements.values():
        if el.whitelist is not None and el.whitelist not in known:
            raise ConfigError(
                f"read_structure.elements.{el.name}.whitelist: {el.whitelist!r} is not defined. "
                f"Known whitelists: {sorted(whitelists)} (plus 'adt_panel' for the ADT element)."
            )
    adt_wl = read_structure.adt.whitelist
    if adt_wl != "adt_panel" and adt_wl not in whitelists:
        raise ConfigError(
            f"read_structure.elements.{read_structure.adt.name}.whitelist: the ADT element "
            f"should reference 'adt_panel'"
        )

    _validate_matching(raw["matching"])
    _validate_filtering(raw["filtering"])
    _validate_qc(raw["qc"])
    _validate_performance(raw["performance"])

    # -- structural read-layout checks -------------------------------------
    overlaps = read_structure.overlaps()
    if overlaps:
        detail = "; ".join(f"{a.describe()} overlaps {b.describe()}" for a, b in overlaps)
        raise ConfigError(
            f"read_structure: elements overlap on the same read: {detail}. "
            f"If this is intentional, set 'allow_overlap: true' on one of them."
        )

    cfg = Config(
        path=path, raw=raw, read_structure=read_structure,
        sublibraries=tuple(sublibraries), whitelists=whitelists,
        panel=panel, layout=layout, output_dir=output_dir, warnings=warnings,
    )
    _validate_output_dir(cfg)
    return cfg


def _validate_matching(m: Mapping) -> None:
    adt = m.get("adt") or {}
    mm = adt.get("max_mismatches")
    if not isinstance(mm, int) or isinstance(mm, bool) or mm < 0:
        raise ConfigError(f"matching.adt.max_mismatches: expected an integer >= 0, got {mm!r}")
    if mm > 2:
        raise ConfigError(
            f"matching.adt.max_mismatches: {mm} would build a very large mutant dictionary; "
            f"0 or 1 is normal (2 is the supported maximum)"
        )

    cb = m.get("cell_barcode") or {}
    ed = cb.get("max_edit_distance")
    if not isinstance(ed, int) or isinstance(ed, bool) or ed < 0:
        raise ConfigError(
            f"matching.cell_barcode.max_edit_distance: expected an integer >= 0, got {ed!r}"
        )

    census = cb.get("census") or {}
    for key in ("offset_reads", "n_reads", "min_count_threshold"):
        v = census.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ConfigError(
                f"matching.cell_barcode.census.{key}: expected an integer >= 0, got {v!r}"
            )
    thresh = census.get("perfect_cell_threshold")
    if not isinstance(thresh, (int, float)) or isinstance(thresh, bool) or not 0 < thresh <= 1:
        raise ConfigError(
            f"matching.cell_barcode.census.perfect_cell_threshold: expected a fraction "
            f"in (0, 1], got {thresh!r}"
        )
    accept = census.get("accept")
    if accept not in _VALID_CENSUS_ACCEPT:
        raise ConfigError(
            f"matching.cell_barcode.census.accept: expected one of {_VALID_CENSUS_ACCEPT}, "
            f"got {accept!r}"
        )

    umi = m.get("umi") or {}
    if umi.get("method") not in _VALID_UMI_METHODS:
        raise ConfigError(
            f"matching.umi.method: expected one of {_VALID_UMI_METHODS}, got {umi.get('method')!r}"
        )
    umm = umi.get("max_mismatches")
    if not isinstance(umm, int) or isinstance(umm, bool) or umm < 0:
        raise ConfigError(f"matching.umi.max_mismatches: expected an integer >= 0, got {umm!r}")
    ratio = umi.get("directional_count_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio <= 0:
        raise ConfigError(
            f"matching.umi.directional_count_ratio: expected a positive number, got {ratio!r}"
        )


def _validate_filtering(f: Mapping) -> None:
    for key in ("min_umi_per_cell", "max_umi_per_cell"):
        v = f.get(key)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ConfigError(f"filtering.{key}: expected a non-negative number or null, got {v!r}")
    lo, hi = f.get("min_umi_per_cell"), f.get("max_umi_per_cell")
    if lo is not None and hi is not None and lo > hi:
        raise ConfigError(
            f"filtering: min_umi_per_cell ({lo}) is greater than max_umi_per_cell ({hi}), "
            f"which would select no cells"
        )
    n = f.get("min_adts_detected")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ConfigError(f"filtering.min_adts_detected: expected an integer >= 0, got {n!r}")
    top = (f.get("knee") or {}).get("exclude_top_n")
    if not isinstance(top, int) or isinstance(top, bool) or top < 0:
        raise ConfigError(f"filtering.knee.exclude_top_n: expected an integer >= 0, got {top!r}")


def _validate_qc(q: Mapping) -> None:
    sat = q.get("saturation") or {}
    fracs = sat.get("fractions") or []
    if sat.get("enabled"):
        if not fracs:
            raise ConfigError("qc.saturation.fractions: required when qc.saturation.enabled is true")
        for fr in fracs:
            if not isinstance(fr, (int, float)) or isinstance(fr, bool) or not 0 < fr <= 1:
                raise ConfigError(
                    f"qc.saturation.fractions: every entry must be in (0, 1], got {fr!r}"
                )
        if list(fracs) != sorted(fracs):
            raise ConfigError("qc.saturation.fractions: must be in ascending order")

    an = q.get("analysis") or {}
    n_pcs = an.get("n_pcs")
    if not isinstance(n_pcs, int) or isinstance(n_pcs, bool) or n_pcs < 1:
        raise ConfigError(f"qc.analysis.n_pcs: expected an integer >= 1, got {n_pcs!r}")

    pl = q.get("plots") or {}
    pct = pl.get("color_percentiles") or []
    if len(pct) != 2 or not all(isinstance(p, (int, float)) for p in pct) or not 0 <= pct[0] < pct[1] <= 100:
        raise ConfigError(
            f"qc.plots.color_percentiles: expected [low, high] within 0..100 with low < high, "
            f"got {pct!r}"
        )
    gc = pl.get("grid_columns")
    if not isinstance(gc, int) or isinstance(gc, bool) or gc < 1:
        raise ConfigError(f"qc.plots.grid_columns: expected an integer >= 1, got {gc!r}")
    dpi = pl.get("dpi")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi < 1:
        raise ConfigError(f"qc.plots.dpi: expected a positive integer, got {dpi!r}")


def _validate_performance(p: Mapping) -> None:
    rows = p.get("chunk_rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ConfigError(f"performance.chunk_rows: expected a positive integer, got {rows!r}")
    bins = p.get("gzip_binaries")
    if not isinstance(bins, (list, tuple)):
        raise ConfigError(f"performance.gzip_binaries: expected a list of binary names, got {bins!r}")


# --------------------------------------------------------------------------
# Output-directory containment (hard rule: no outputs inside the repo)
# --------------------------------------------------------------------------

def repo_root() -> Path | None:
    """Root of the git repository containing this package, or None if installed
    from a wheel outside a working tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_dir(cfg: Config) -> None:
    """Refuse an output_dir inside the signalseq repository.

    Outputs are large and partly un-shareable (FASTQ-derived matrices, plots,
    barcode-derived tables); keeping them out of the tree means they can never
    be committed by accident.
    """
    out = cfg.output_dir
    root = repo_root()
    if root is not None and _is_within(out, root):
        raise ConfigError(
            f"run.output_dir resolves to {out}, which is inside the signalseq repository "
            f"({root}). Outputs, intermediates, logs and plots must live outside the repo so "
            f"they can never be committed. Point run.output_dir somewhere else, e.g. "
            f"~/signalseq-runs/{cfg.raw['run']['name']}."
        )
    if out.exists() and not out.is_dir():
        raise ConfigError(f"run.output_dir exists but is not a directory: {out}")
