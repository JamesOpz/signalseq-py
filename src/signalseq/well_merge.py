"""Merging barcodes that share a well into one cell.

Parse Evercode loads each RT well with two primers, oligo-dT (polyA, stype T)
and random hexamer (stype R), plus a few X primers in some wells. Each has its
own barcode sequence, so the read scan resolves them to different whitelist
entries; split-pipe treats them as one cell because it keys cells by well. This
module does the same.

The merge runs AFTER UMI collapse, on per-primer (cell, ADT) molecule counts:
UMIs are deduplicated within each primer type, then summed across types. So
the per-primer totals add up exactly to the merged total.

Cell ids stay in the encoder's packed space throughout; merging only swaps the
merged round's whitelist index for its well's canonical (first-listed) index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .encoding import Encoder
from .readstructure import ReadStructure


@dataclass(frozen=True)
class WellMerge:
    """How one barcode round collapses onto wells."""

    multiplier: int               # place value of the merged round in a packed cell id
    size: int                     # that round's whitelist size
    canonical: np.ndarray         # whitelist index -> canonical index of its well
    type_codes: np.ndarray | None # whitelist index -> index into type_labels
    type_labels: tuple[str, ...]  # per-primer obs column suffixes, e.g. ("polyA", "randhex")

    @classmethod
    def build(cls, read_structure: ReadStructure, whitelists: Mapping,
              specs: Mapping, encoder: Encoder) -> "WellMerge | None":
        """The merge for the round whose whitelist sets merge_by_well, or None."""
        for pos, element in enumerate(read_structure.rounds):
            spec = specs[element.whitelist]
            if not spec.merge_by_well:
                continue
            wl = whitelists[element.name]
            first_in_well: dict = {}
            canonical = np.array(
                [first_in_well.setdefault(w, i) for i, w in enumerate(wl.wells)], dtype=np.int64
            )
            type_codes, labels = None, ()
            if wl.stype is not None:
                # Label order follows first appearance in the whitelist, so
                # Parse's file gives polyA before randhex.
                stypes = list(dict.fromkeys(wl.stype))
                code_of = {st: k for k, st in enumerate(stypes)}
                type_codes = np.array([code_of[st] for st in wl.stype], dtype=np.int64)
                labels = tuple(spec.stype_labels.get(st, st) for st in stypes)
            return cls(
                multiplier=encoder.radix_multipliers[pos],
                size=wl.n,
                canonical=canonical,
                type_codes=type_codes,
                type_labels=labels,
            )
        return None

    def _round_index(self, cell_id: np.ndarray) -> np.ndarray:
        return (np.asarray(cell_id, dtype=np.int64) // self.multiplier) % self.size

    def merge(self, cell_id: np.ndarray) -> np.ndarray:
        """Packed cell ids with the merged round replaced by its well's canonical index."""
        cell_id = np.asarray(cell_id, dtype=np.int64)
        idx = self._round_index(cell_id)
        return cell_id + (self.canonical[idx] - idx) * self.multiplier

    def type_of(self, cell_id: np.ndarray) -> np.ndarray:
        """Primer-type code of each (unmerged) cell id."""
        return self.type_codes[self._round_index(cell_id)]

    def members(self, merged_ids: np.ndarray) -> np.ndarray:
        """Every unmerged cell id that merges onto one of `merged_ids`."""
        merged_ids = np.asarray(merged_ids, dtype=np.int64)
        idx = self._round_index(merged_ids)
        out = [merged_ids]
        for j in range(self.size):
            c = self.canonical[j]
            if c == j:
                continue
            hit = merged_ids[idx == c]
            out.append(hit + (j - c) * self.multiplier)
        return np.concatenate(out)


def sum_by_cell_adt(cell_id: np.ndarray, adt_idx: np.ndarray, n_adts: int,
                    *weights: np.ndarray):
    """Collapse repeated (cell, ADT) rows, summing each weight array.

    Returns (cell_id, adt_idx, *summed_weights) with one row per distinct pair.
    """
    key = np.asarray(cell_id, dtype=np.int64) * n_adts + np.asarray(adt_idx, dtype=np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    sums = [np.bincount(inverse, weights=w, minlength=len(uniq)).astype(w.dtype)
            for w in weights]
    return (uniq // n_adts, uniq % n_adts, *sums)
