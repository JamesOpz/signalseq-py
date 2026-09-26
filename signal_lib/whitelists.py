"""Whitelist loading + barcode-correction-dictionary construction.

Two distinct correction schemes are used, deliberately:

- ADT tag (R1, 15bp, 14-plex panel): simple exact + 1bp-Hamming-mutant dictionary,
  ambiguous mutants dropped. This is custom to the CITE-seq-style ADT protocol and
  has no equivalent on the Parse RNA side.
- Cell-identity barcodes (bc1/RT, bc2, bc3 -- both from the 96-well ligation
  whitelist): the precomputed edit-distance dictionaries shipped with the Parse
  Biosciences RNA pipeline (`bc_dict_n99_v5.json`, `bc_dict_R3_v3.json`) are loaded
  and reused verbatim, so ADT and RNA sublibraries resolve the same physical read
  to the same cell (see bc_correct.py for how they're applied).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

BASES = (b"A", b"C", b"G", b"T")


@dataclass
class AdtWhitelist:
    seqs: list          # list[bytes], 15bp ADT tag sequences
    names: np.ndarray    # Antigen names, parallel to seqs
    meta_df: pd.DataFrame
    correction_dict: dict  # bytes -> int index, exact + 1bp mutants (ambiguous dropped)


@dataclass
class WellWhitelist:
    seqs: list           # list[bytes]
    seqs_str: np.ndarray  # str array, parallel
    wells: np.ndarray     # well IDs, parallel
    stype: np.ndarray | None  # sample-type column (RT whitelist only), else None
    n: int
    edit_dicts: dict      # {edit_dist:int -> {bytes_observed: [true_idx, ...]}}


def _build_adt_dict(seqs: list, allow_muts: int = 1) -> dict:
    """Exact + 1bp-mutant dict for the ADT tag. Ambiguous mutants are deleted
    (kept identical in spirit to the pilot notebook's build_adt_dict)."""
    bc_dict: dict = {}
    for i, bc in enumerate(seqs):
        bc_dict[bc] = i
    if allow_muts >= 1:
        for i, bc in enumerate(seqs):
            for pos in range(len(bc)):
                for base in BASES:
                    if bc[pos:pos + 1] == base:
                        continue
                    mut = bc[:pos] + base + bc[pos + 1:]
                    if mut in bc_dict:
                        if bc_dict[mut] != i:
                            del bc_dict[mut]
                    else:
                        bc_dict[mut] = i
    return bc_dict


def load_adt_whitelist(csv_path=config.ADT_WHITELIST_CSV) -> AdtWhitelist:
    df = pd.read_csv(csv_path)
    seqs = [s.upper().encode("ascii") for s in df["Barcode"].values]
    names = df["Antigen"].values
    meta_df = df.copy()
    correction_dict = _build_adt_dict(seqs, allow_muts=config.ADT_BC_N_MUTS)
    return AdtWhitelist(seqs=seqs, names=names, meta_df=meta_df, correction_dict=correction_dict)


def _load_parse_bc_dict(json_path, seq_to_idx: dict, max_edit: int) -> dict:
    """Load a Parse splitpipe precomputed edit-distance dict and remap it from
    string keys/values to bytes-keyed observed-barcode -> [true_idx, ...], trimmed
    to edit distances <= max_edit (production splitpipe hard-caps at 2).
    """
    with open(json_path) as fh:
        raw = json.load(fh)

    out: dict = {}
    for level_str, level_dict in raw.items():
        level = int(level_str)
        if level > max_edit:
            continue
        remapped = {}
        for observed_seq, true_seqs in level_dict.items():
            idxs = [seq_to_idx[s] for s in true_seqs if s in seq_to_idx]
            if idxs:
                remapped[observed_seq.upper().encode("ascii")] = idxs
        out[level] = remapped
    return out


def load_rt_whitelist(csv_path=config.RT_WHITELIST_CSV,
                       bc_dict_json=config.RT_BC_DICT_JSON,
                       max_edit: int = config.BC_EDIT_DIST_MAX) -> WellWhitelist:
    df = pd.read_csv(csv_path)
    seqs_str = df["sequence"].str.upper().values
    seqs = [s.encode("ascii") for s in seqs_str]
    wells = df["well"].values
    stype = df["stype"].values
    seq_to_idx = {s: i for i, s in enumerate(seqs_str)}
    edit_dicts = _load_parse_bc_dict(bc_dict_json, seq_to_idx, max_edit)
    return WellWhitelist(seqs=seqs, seqs_str=seqs_str, wells=wells, stype=stype,
                          n=len(seqs), edit_dicts=edit_dicts)


def load_ligation_whitelist(csv_path=config.LIGATION_WHITELIST_CSV,
                             bc_dict_json=config.LIGATION_BC_DICT_JSON,
                             max_edit: int = config.BC_EDIT_DIST_MAX) -> WellWhitelist:
    df = pd.read_csv(csv_path)
    seqs_str = df["sequence"].str.upper().values
    seqs = [s.encode("ascii") for s in seqs_str]
    wells = df["well"].values
    seq_to_idx = {s: i for i, s in enumerate(seqs_str)}
    edit_dicts = _load_parse_bc_dict(bc_dict_json, seq_to_idx, max_edit)
    return WellWhitelist(seqs=seqs, seqs_str=seqs_str, wells=wells, stype=None,
                          n=len(seqs), edit_dicts=edit_dicts)
