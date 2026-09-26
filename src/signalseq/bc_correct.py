"""Cell-barcode correction matching the Parse Biosciences RNA pipeline.

This mirrors splitpipe's `bcutils.correct_barcodes` / `get_min_edit_dists` and
its two-pass structure, so an ADT read and an RNA read from the same physical
cell resolve to the same cell:

1. `run_census()` samples a block of reads, keeps only those where every
   barcode round is an EXACT whitelist match, counts how often each round
   combination occurs, and derives a count threshold (the combo count at the
   configured cumulative-read-fraction percentile, floored).
2. `BarcodeCorrector.correct()` takes one read: fast-path when every round is
   exact, otherwise gather each round's candidate indices from the precomputed
   edit-distance dictionaries and accept only if exactly one combination clears
   the census threshold.

Generalised over the number of rounds: the original hardcoded three (RT, bc2,
bc3), this reads whatever `read_structure.cell_barcode.order` declares.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np

from .fastq_io import iter_sequences
from .readstructure import Element, ReadStructure
from .whitelists import BarcodeWhitelist

ACCEPT_STRICTLY_GREATER = "strictly_greater"


@dataclass
class CensusResult:
    """Output of the first pass over the reads."""

    combo_counts: Counter
    count_threshold: int
    n_reads_sampled: int
    n_exact_combos: int

    @property
    def n_distinct_combos(self) -> int:
        return len(self.combo_counts)


def run_census(r2_path, read_structure: ReadStructure,
               whitelists: Mapping[str, BarcodeWhitelist],
               census_cfg: Mapping,
               gzip_binaries: Sequence[str]) -> CensusResult:
    """Count exact-match round combinations in a sample of reads.

    The threshold is the combo count whose cumulative share of reads is closest
    to `perfect_cell_threshold`, floored at `min_count_threshold` -- splitpipe's
    `max(5, calc)`.
    """
    rounds = read_structure.rounds
    barcode_read = rounds[0].read
    exacts = [whitelists[el.name].exact for el in rounds]
    slices = [el.slice for el in rounds]
    min_len = read_structure.min_length(barcode_read)

    combo_counts: Counter = Counter()
    n_seen = 0
    n_exact = 0

    for seq in iter_sequences(r2_path, gzip_binaries,
                              offset_reads=int(census_cfg["offset_reads"]),
                              max_reads=int(census_cfg["n_reads"])):
        n_seen += 1
        if len(seq) < min_len:
            continue
        combo = []
        for sl, exact in zip(slices, exacts):
            idx = exact.get(seq[sl])
            if idx is None:
                break
            combo.append(idx)
        else:
            combo_counts[tuple(combo)] += 1
            n_exact += 1

    floor = int(census_cfg["min_count_threshold"])
    if not combo_counts:
        return CensusResult(combo_counts, floor, n_seen, 0)

    counts_sorted = np.array(sorted(combo_counts.values(), reverse=True))
    cum_frac = np.cumsum(counts_sorted) / counts_sorted.sum()
    idx = np.argmin(np.abs(cum_frac - float(census_cfg["perfect_cell_threshold"])))
    calc = int(counts_sorted[idx])
    return CensusResult(combo_counts, max(floor, calc), n_seen, n_exact)


def get_min_edit_matches(bc: bytes, edit_dicts: Mapping, max_d: int) -> tuple[list, int | None]:
    """Candidate indices at the smallest edit distance with any match.

    Mirrors splitpipe's get_min_edit_dists: check distance 0, 1, 2, ... and stop
    at the first that hits. Returns ([], None) if nothing matches within max_d.
    """
    for d in range(0, max_d + 1):
        matches = edit_dicts.get(d, {}).get(bc)
        if matches:
            return matches, d
    return [], None


@dataclass
class BarcodeCorrector:
    """Resolves one read's barcode rounds to whitelist indices.

    Holds the per-round slices and dictionaries so the read scan does no config
    lookups or attribute chasing per read.
    """

    rounds: tuple[Element, ...]
    edit_dicts: tuple[Mapping, ...]
    slices: tuple[slice, ...]
    max_edit_distance: int
    combo_counts: Mapping
    count_threshold: int
    accept_strictly_greater: bool

    @classmethod
    def build(cls, read_structure: ReadStructure,
              whitelists: Mapping[str, BarcodeWhitelist],
              census: CensusResult, cb_cfg: Mapping) -> "BarcodeCorrector":
        rounds = read_structure.rounds
        accept = str((cb_cfg.get("census") or {}).get("accept", ACCEPT_STRICTLY_GREATER))
        return cls(
            rounds=rounds,
            edit_dicts=tuple(whitelists[el.name].edit_dicts for el in rounds),
            slices=tuple(el.slice for el in rounds),
            max_edit_distance=int(cb_cfg["max_edit_distance"]),
            combo_counts=census.combo_counts,
            count_threshold=census.count_threshold,
            accept_strictly_greater=(accept == ACCEPT_STRICTLY_GREATER),
        )

    def extract(self, seq: bytes) -> tuple[bytes, ...]:
        return tuple(seq[sl] for sl in self.slices)

    def correct(self, seq: bytes) -> tuple[tuple[int, ...] | None, bool]:
        """Resolve one read. Returns (round indices or None, all_rounds_exact).

        Fast path: every round at edit distance 0 -> accept directly. Otherwise
        take the cartesian product of the per-round candidates and accept only
        if exactly one combination clears the census count threshold; two or
        more is ambiguous and the read is dropped.
        """
        max_d = self.max_edit_distance
        candidates = []
        all_exact = True
        for sl, edit_dict in zip(self.slices, self.edit_dicts):
            matches, distance = get_min_edit_matches(seq[sl], edit_dict, max_d)
            if not matches:
                return None, False
            candidates.append(matches)
            if distance != 0:
                all_exact = False

        if all_exact:
            return tuple(c[0] for c in candidates), True

        threshold = self.count_threshold
        counts = self.combo_counts
        accepted = None
        n_accepted = 0
        for combo in product(*candidates):
            count = counts.get(combo, 0)
            clears = count > threshold if self.accept_strictly_greater else count >= threshold
            if clears:
                accepted = combo
                n_accepted += 1
                if n_accepted > 1:
                    return None, False
        return accepted, False
