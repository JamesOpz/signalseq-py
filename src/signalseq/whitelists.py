"""Loading barcode whitelists and building their correction dictionaries.

Two correction schemes are used, deliberately:

- **ADT tag** (the antibody barcode on R1): an exact + N-mismatch substitution
  dictionary built here, with ambiguous mutants dropped. Custom to the
  CITE-seq-style ADT protocol; there is no RNA-side equivalent to match against.
- **Cell-identity rounds** (RT and the ligation rounds): the precomputed
  edit-distance dictionaries shipped with the Parse Biosciences RNA pipeline are
  loaded and reused verbatim, so an ADT read and an RNA read from the same
  physical cell resolve to the same cell.

Column names and file paths are config-driven. Loads are cached on the resolved
file paths, so a chemistry that points two barcode rounds at the same plate file
(as Parse v3 does for its two ligation rounds) parses it once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PanelSpec, WhitelistSpec
from .readstructure import ConfigError

BASES = (b"A", b"C", b"G", b"T")

_WELL_RE = re.compile(r"^([A-Za-z])0*(\d+)$")
_PLATE_COLS = 12


class WhitelistError(ConfigError):
    """Raised when a whitelist or panel file is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------

@dataclass
class AdtPanel:
    """The antibody panel: tag sequences, display names and a correction dict."""

    seqs: list                 # list[bytes]
    names: np.ndarray          # display names, parallel to seqs
    meta_df: pd.DataFrame      # the panel CSV as read, for adata.var
    correction_dict: dict      # bytes -> int index
    seq_length: int
    max_mismatches: int
    name_column: str           # panel column holding `names`, i.e. the future var_names

    @property
    def n(self) -> int:
        return len(self.seqs)

    def match(self, tag: bytes) -> int | None:
        """O(1) lookup. Returns the panel index, or None if unmatched."""
        return self.correction_dict.get(tag)


@dataclass
class BarcodeWhitelist:
    """One cell-barcode round's whitelist plus its edit-distance dictionaries."""

    name: str
    seqs: list                  # list[bytes]
    seqs_str: np.ndarray        # str array, parallel
    wells: np.ndarray           # well IDs, parallel (falls back to sequences)
    winds: np.ndarray           # zero-padded 1-based well indexes, as split-pipe names them
    stype: np.ndarray | None    # optional sample-type column
    seq_length: int
    edit_dicts: dict            # {edit_distance: {observed_bytes: [true_idx, ...]}}
    exact: dict                 # bytes -> int, the 0-distance fast path

    @property
    def n(self) -> int:
        return len(self.seqs)


# --------------------------------------------------------------------------
# ADT panel
# --------------------------------------------------------------------------

def _mutants_at_distance(bc: bytes, distance: int):
    """Every substitution mutant of `bc` at exactly Hamming `distance`.

    Positions ascend and bases follow A,C,G,T, which for distance 1 reproduces
    the original pipeline's iteration order exactly -- and that order matters,
    because a collision between two tags' mutants deletes the entry.
    """
    n = len(bc)
    for positions in combinations(range(n), distance):
        originals = [bc[p:p + 1] for p in positions]
        for replacements in product(BASES, repeat=distance):
            if any(r == o for r, o in zip(replacements, originals)):
                continue
            mut = bytearray(bc)
            for p, r in zip(positions, replacements):
                mut[p:p + 1] = r
            yield bytes(mut)


def build_correction_dict(seqs: list, max_mismatches: int, drop_ambiguous: bool = True) -> dict:
    """Exact + up-to-`max_mismatches` substitution dictionary.

    A mutant reachable from two different tags is ambiguous. With
    `drop_ambiguous` it is removed entirely (the original behaviour, and the
    safe choice); otherwise the first tag to claim it keeps it.
    """
    bc_dict: dict = {}
    for i, bc in enumerate(seqs):
        bc_dict[bc] = i

    for distance in range(1, max_mismatches + 1):
        for i, bc in enumerate(seqs):
            for mut in _mutants_at_distance(bc, distance):
                if mut in bc_dict:
                    if bc_dict[mut] != i and drop_ambiguous:
                        del bc_dict[mut]
                else:
                    bc_dict[mut] = i
    return bc_dict


_panel_cache: dict[tuple, AdtPanel] = {}


def load_adt_panel(spec: PanelSpec, max_mismatches: int, drop_ambiguous: bool = True) -> AdtPanel:
    key = (str(spec.csv), spec.barcode_column, spec.name_column, max_mismatches, drop_ambiguous)
    if key in _panel_cache:
        return _panel_cache[key]

    if not spec.csv.is_file():
        raise WhitelistError(f"inputs.adt_panel.csv: file not found: {spec.csv}")
    df = pd.read_csv(spec.csv)
    for col, cfg_key in ((spec.barcode_column, "barcode_column"), (spec.name_column, "name_column")):
        if col not in df.columns:
            raise WhitelistError(
                f"inputs.adt_panel.{cfg_key}: column {col!r} not in {spec.csv.name}. "
                f"Columns present: {list(df.columns)}"
            )
    if df.empty:
        raise WhitelistError(f"inputs.adt_panel.csv: {spec.csv} contains no rows")

    seqs = [str(s).strip().upper().encode("ascii") for s in df[spec.barcode_column].values]
    lengths = sorted({len(s) for s in seqs})
    if len(lengths) != 1:
        raise WhitelistError(
            f"inputs.adt_panel.csv: {spec.csv.name} mixes barcode lengths {lengths}; "
            f"every ADT tag must be the same length"
        )
    names = df[spec.name_column].values
    dupes = pd.Index(names).duplicated()
    if dupes.any():
        raise WhitelistError(
            f"inputs.adt_panel.{spec.name_column}: duplicate names "
            f"{sorted(set(pd.Index(names)[dupes]))}; these become var_names and must be unique"
        )

    panel = AdtPanel(
        seqs=seqs,
        names=names,
        meta_df=df.copy(),
        correction_dict=build_correction_dict(seqs, max_mismatches, drop_ambiguous),
        seq_length=lengths[0],
        max_mismatches=max_mismatches,
        name_column=spec.name_column,
    )
    _panel_cache[key] = panel
    return panel


# --------------------------------------------------------------------------
# Cell-barcode whitelists
# --------------------------------------------------------------------------

def _load_parse_edit_dict(json_path: Path, seq_to_idx: dict, max_edit: int) -> dict:
    """Load a Parse splitpipe precomputed edit-distance dict.

    Remaps string keys/values to bytes-keyed observed -> [true_idx, ...], and
    trims to distances <= max_edit (production splitpipe caps at 2).
    """
    if not json_path.is_file():
        raise WhitelistError(f"edit_distance_json: file not found: {json_path}")
    try:
        with open(json_path) as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise WhitelistError(f"edit_distance_json: {json_path} is not valid JSON -- {exc}") from exc

    out: dict = {}
    for level_str, level_dict in raw.items():
        try:
            level = int(level_str)
        except (TypeError, ValueError):
            raise WhitelistError(
                f"edit_distance_json: {json_path} has non-integer top-level key {level_str!r}; "
                f"expected edit distances like \"0\", \"1\", \"2\""
            )
        if level > max_edit:
            continue
        remapped = {}
        for observed_seq, true_seqs in level_dict.items():
            idxs = [seq_to_idx[s] for s in true_seqs if s in seq_to_idx]
            if idxs:
                remapped[observed_seq.upper().encode("ascii")] = idxs
        out[level] = remapped
    return out


def well_indexes(wells) -> np.ndarray:
    """split-pipe's well index for each well: row-major on a 12-column plate
    (A1=01, A12=12, B1=13 ... H12=96), zero-padded to 2 digits, or 3 past 99.

    Wells that do not look like plate wells are numbered by first appearance
    in the whitelist instead, so a non-plate chemistry still gets stable indexes.
    """
    matches = [_WELL_RE.match(str(w)) for w in wells]
    if all(m and 1 <= int(m.group(2)) <= _PLATE_COLS for m in matches):
        idx = [(ord(m.group(1).upper()) - ord("A")) * _PLATE_COLS + int(m.group(2))
               for m in matches]
    else:
        first_seen: dict = {}
        idx = [first_seen.setdefault(w, len(first_seen) + 1) for w in wells]
    width = 3 if max(idx, default=0) > 99 else 2
    return np.array([f"{i:0{width}d}" for i in idx])


_whitelist_cache: dict[tuple, BarcodeWhitelist] = {}


def load_barcode_whitelist(spec: WhitelistSpec, max_edit: int) -> BarcodeWhitelist:
    """Load one round's whitelist. Cached, so shared plate files parse once."""
    key = spec.cache_key + (spec.sequence_column, spec.well_column, spec.stype_column, max_edit)
    if key in _whitelist_cache:
        cached = _whitelist_cache[key]
        # Same file and settings; only the round's label differs.
        if cached.name != spec.name:
            return BarcodeWhitelist(
                name=spec.name, seqs=cached.seqs, seqs_str=cached.seqs_str, wells=cached.wells,
                winds=cached.winds, stype=cached.stype, seq_length=cached.seq_length,
                edit_dicts=cached.edit_dicts, exact=cached.exact,
            )
        return cached

    what = f"inputs.whitelists.{spec.name}"
    if not spec.csv.is_file():
        raise WhitelistError(f"{what}.csv: file not found: {spec.csv}")
    df = pd.read_csv(spec.csv)
    if spec.sequence_column not in df.columns:
        raise WhitelistError(
            f"{what}.sequence_column: column {spec.sequence_column!r} not in {spec.csv.name}. "
            f"Columns present: {list(df.columns)}"
        )
    if df.empty:
        raise WhitelistError(f"{what}.csv: {spec.csv} contains no rows")

    seqs_str = df[spec.sequence_column].astype(str).str.strip().str.upper().values
    seqs = [s.encode("ascii") for s in seqs_str]
    lengths = sorted({len(s) for s in seqs})
    if len(lengths) != 1:
        raise WhitelistError(
            f"{what}.csv: {spec.csv.name} mixes barcode lengths {lengths}; "
            f"every barcode in a whitelist must be the same length"
        )

    dup = pd.Index(seqs_str).duplicated()
    if dup.any():
        raise WhitelistError(
            f"{what}.csv: duplicate barcode sequences {sorted(set(pd.Index(seqs_str)[dup]))}"
        )

    if spec.well_column is not None:
        if spec.well_column not in df.columns:
            raise WhitelistError(
                f"{what}.well_column: column {spec.well_column!r} not in {spec.csv.name}. "
                f"Columns present: {list(df.columns)}"
            )
        wells = df[spec.well_column].astype(str).values
    else:
        wells = seqs_str.copy()

    stype = None
    if spec.stype_column is not None:
        if spec.stype_column not in df.columns:
            raise WhitelistError(
                f"{what}.stype_column: column {spec.stype_column!r} not in {spec.csv.name}. "
                f"Columns present: {list(df.columns)}"
            )
        stype = df[spec.stype_column].astype(str).values

    seq_to_idx = {s: i for i, s in enumerate(seqs_str)}
    if spec.edit_distance_json is not None:
        edit_dicts = _load_parse_edit_dict(spec.edit_distance_json, seq_to_idx, max_edit)
    else:
        # No precomputed dictionary: exact matching only.
        edit_dicts = {0: {s: [i] for i, s in enumerate(seqs)}}

    wl = BarcodeWhitelist(
        name=spec.name, seqs=seqs, seqs_str=seqs_str, wells=wells, winds=well_indexes(wells),
        stype=stype,
        seq_length=lengths[0], edit_dicts=edit_dicts,
        exact={s: i for i, s in enumerate(seqs)},
    )
    _whitelist_cache[key] = wl
    return wl


def load_all(cfg) -> tuple[AdtPanel, dict[str, BarcodeWhitelist]]:
    """Load the panel and one whitelist per cell-barcode round named in the config."""
    adt_cfg = cfg.matching["adt"]
    panel = load_adt_panel(cfg.panel, adt_cfg["max_mismatches"], adt_cfg["drop_ambiguous"])

    max_edit = cfg.matching["cell_barcode"]["max_edit_distance"]
    rounds: dict[str, BarcodeWhitelist] = {}
    for element in cfg.read_structure.rounds:
        spec = cfg.whitelists[element.whitelist]
        rounds[element.name] = load_barcode_whitelist(spec, max_edit)
    return panel, rounds
